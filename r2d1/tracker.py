"""
r2d1 — core tracker.

Ships checkpoint files to Cloudflare R2.
Logs metadata (epoch, loss, timing, R2 path) to Cloudflare D1.

r2d1 is agnostic to what's inside the files.
Your model decides how to package its checkpoint.
"""
import time
import threading
import requests
import boto3
from botocore.config import Config


def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


class Job:
    """
    Handle for a single training run.
    Returned by Tracker.start_job() or Tracker.resume_job().
    """

    def __init__(self, job_id, tracker):
        self.job_id = job_id
        self._t = tracker
        self._pending_upload = None   # background thread for async uploads

    # ------------------------------------------------------------------
    # Metrics → D1
    # ------------------------------------------------------------------

    def log(self, epoch, loss=None, accuracy=None, duration_sec=None, **extra):
        """
        Log per-epoch metrics to D1.
        D1 always gets: epoch, loss, accuracy, duration, timestamp.
        Extra kwargs are ignored (log them in your own run.log file).
        """
        self._t._d1(
            """INSERT INTO epochs
               (job_id, epoch, loss, accuracy, duration_sec, logged_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [self.job_id, epoch, loss, accuracy, duration_sec, _now()]
        )

    # ------------------------------------------------------------------
    # Checkpoints → R2
    # ------------------------------------------------------------------

    def save(self, epoch, files, loss=None):
        """
        Upload checkpoint files to R2 (synchronous).

        files: dict of filename → bytes
            e.g. {
                "model.safetensors": ...,
                "config.json":       ...,
                "run.log":           ...,
            }

        Stored at: jobs/job_{id}/epoch_{epoch}/filename
        D1 is updated with the R2 prefix, epoch, loss, and timestamp.
        """
        prefix = self._upload_files(epoch, files)
        self._update_d1_checkpoint(epoch, prefix, loss)
        return prefix

    def save_async(self, epoch, files, loss=None):
        """
        Async checkpoint — files dict is captured immediately (your model
        can move on), uploaded to R2 in a background thread.

        Only one upload runs at a time — if the previous epoch's upload
        is still in flight, this waits for it first.
        """
        self.wait()   # ensure previous upload is done before starting next

        # Capture everything needed for the upload now
        # (so your training code can mutate model/buffers freely after this)
        _epoch  = epoch
        _files  = {k: bytes(v) if not isinstance(v, bytes) else v
                   for k, v in files.items()}
        _loss   = loss

        def _run():
            prefix = self._upload_files(_epoch, _files)
            self._update_d1_checkpoint(_epoch, prefix, _loss)

        self._pending_upload = threading.Thread(target=_run, daemon=True)
        self._pending_upload.start()

    def wait(self):
        """Block until any in-flight async upload finishes."""
        if self._pending_upload is not None:
            self._pending_upload.join()
            self._pending_upload = None

    def load_latest(self):
        """
        Download all files from the latest checkpoint.
        Returns dict of filename → bytes — same shape as what you passed to save().
        """
        result = self._t._d1(
            "SELECT last_checkpoint_prefix FROM jobs WHERE id=?", [self.job_id]
        )
        prefix = result['result'][0]['results'][0]['last_checkpoint_prefix']
        if not prefix:
            raise ValueError(f"No checkpoint found for job {self.job_id}")

        # List all files under this prefix and download them
        response = self._t._s3.list_objects_v2(
            Bucket=self._t.bucket, Prefix=prefix
        )
        files = {}
        for obj in response.get('Contents', []):
            key      = obj['Key']
            filename = key[len(prefix):]   # strip prefix → just the filename
            files[filename] = self._t._s3.get_object(
                Bucket=self._t.bucket, Key=key
            )['Body'].read()

        print(f"[r2d1] ✓ loaded checkpoint  job={self.job_id}  "
              f"files={list(files.keys())}")
        return files

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _upload_files(self, epoch, files):
        prefix = f"jobs/job_{self.job_id}/epoch_{epoch}/"
        for filename, data in files.items():
            self._t._s3.put_object(
                Bucket=self._t.bucket,
                Key=prefix + filename,
                Body=data,
            )
        print(f"[r2d1] ✓ checkpoint  job={self.job_id}  epoch={epoch}  "
              f"files={list(files.keys())}")
        return prefix

    def _update_d1_checkpoint(self, epoch, prefix, loss):
        self._t._d1(
            "UPDATE jobs SET last_checkpoint_prefix=?, updated_at=? WHERE id=?",
            [prefix, _now(), self.job_id]
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def complete(self):
        self.wait()   # flush any pending upload before marking complete
        self._t._d1(
            "UPDATE jobs SET status='completed', updated_at=? WHERE id=?",
            [_now(), self.job_id]
        )
        print(f"[r2d1] ✓ job {self.job_id} completed")

    def interrupt(self):
        """
        Mark job interrupted in D1.
        Alice (or any watcher) queries D1 for interrupted jobs and restarts.
        """
        self._t._d1(
            "UPDATE jobs SET status='interrupted', updated_at=? WHERE id=?",
            [_now(), self.job_id]
        )
        print(f"[r2d1] ⚡ job {self.job_id} interrupted")

    def status(self):
        """Return job metadata + all logged epochs from D1."""
        job    = self._t._d1("SELECT * FROM jobs WHERE id=?", [self.job_id])
        epochs = self._t._d1(
            "SELECT * FROM epochs WHERE job_id=? ORDER BY epoch", [self.job_id]
        )
        return {
            'job':    job['result'][0]['results'][0],
            'epochs': epochs['result'][0]['results'],
        }


class Tracker:
    """
    r2d1 Tracker.

    from r2d1 import Tracker

    tracker = Tracker(
        account_id     = "...",
        api_token      = "...",   # Cloudflare API token value
        d1_database_id = "...",
        r2_bucket      = "...",
        r2_access_key  = "...",
        r2_secret_key  = "...",
    )
    """

    def __init__(self, account_id, api_token, d1_database_id,
                 r2_bucket, r2_access_key, r2_secret_key):
        self.bucket = r2_bucket

        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self._d1_url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/d1/database/{d1_database_id}/query"
        )
        self._s3 = boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=r2_access_key,
            aws_secret_access_key=r2_secret_key,
            config=Config(signature_version='s3v4'),
            region_name='auto',
        )
        self._init_tables()

    def _d1(self, sql, params=None):
        r = requests.post(
            self._d1_url,
            headers=self._headers,
            json={"sql": sql, "params": params or []},
        )
        r.raise_for_status()
        return r.json()

    def _init_tables(self):
        self._d1("""
            CREATE TABLE IF NOT EXISTS jobs (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                name                     TEXT,
                dataset_key              TEXT,
                status                   TEXT DEFAULT 'pending',
                last_checkpoint_prefix   TEXT,
                dataset_size_mb          REAL,
                submitted_at             TEXT,
                updated_at               TEXT
            )
        """)
        self._d1("""
            CREATE TABLE IF NOT EXISTS epochs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id       INTEGER,
                epoch        INTEGER,
                loss         REAL,
                accuracy     REAL,
                duration_sec REAL,
                logged_at    TEXT,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
        """)

    def start_job(self, name, dataset_key=None, dataset_size_mb=None):
        """Register a new job in D1. Returns a Job handle."""
        result = self._d1(
            """INSERT INTO jobs
               (name, dataset_key, status, dataset_size_mb, submitted_at, updated_at)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            [name, dataset_key, dataset_size_mb, _now(), _now()]
        )
        job_id = result['result'][0]['meta']['last_row_id']
        print(f"[r2d1] ✓ started job {job_id}: {name}")
        return Job(job_id, self)

    def resume_job(self, job_id):
        """
        Resume an interrupted job. Returns Job handle.
        You load your own checkpoint files via job.load_latest().
        """
        self._d1(
            "UPDATE jobs SET status='running', updated_at=? WHERE id=?",
            [_now(), job_id]
        )
        job = Job(job_id, self)
        print(f"[r2d1] ✓ resuming job {job_id}")
        return job

    def get_job(self, job_id):
        """Get a Job handle without changing its status."""
        return Job(job_id, self)

    def list_jobs(self):
        """Return all jobs and their current status from D1."""
        return self._d1(
            "SELECT id, name, status, last_checkpoint_prefix, "
            "submitted_at, updated_at FROM jobs ORDER BY id DESC"
        )['result'][0]['results']

    def job(self, name, dataset_key=None, dataset_size_mb=None, resume_job_id=None):
        """
        Decorator — wraps a training function with job lifecycle management.
        Injects a Job as first argument. Handles interrupt/complete automatically.

        @tracker.job(name="dit_run1", dataset_key="hf://datasets/imagenet-1k")
        def train(job):
            for epoch in EpochLoop(range(400), job=job):
                files = model.make_checkpoint()
                epoch.loss  = float(loss)
                epoch.files = files

        train()
        """
        from .loop import job_decorator
        return job_decorator(self, name=name, dataset_key=dataset_key,
                             dataset_size_mb=dataset_size_mb,
                             resume_job_id=resume_job_id)
