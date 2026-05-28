"""
r2d1 — core tracker.
Saves checkpoints to Cloudflare R2, logs metrics to Cloudflare D1.

Works anywhere: local machine, Colab, vast.ai, any GPU server.
No knowledge of how jobs are scheduled or restarted.
"""
import time
import requests
import boto3
from botocore.config import Config

from .checkpoint import serialize, deserialize


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

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def log(self, epoch, loss, accuracy=None, duration_sec=None):
        """Log per-epoch metrics to D1."""
        self._t._d1(
            """INSERT INTO epochs
               (job_id, epoch, loss, accuracy, duration_sec, logged_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [self.job_id, epoch, loss, accuracy, duration_sec, _now()]
        )

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch, model_or_params, optimizer_state=None, loss=None):
        """
        Serialize and upload checkpoint to R2.
        Auto-detects PyTorch nn.Module or JAX param pytree.
        Returns the R2 key.
        """
        key  = f"checkpoints/job_{self.job_id}/epoch_{epoch}.pkl"
        data = serialize(epoch, model_or_params, optimizer_state, loss)

        self._t._s3.put_object(Bucket=self._t.bucket, Key=key, Body=data)
        self._t._d1(
            "UPDATE jobs SET last_checkpoint=?, updated_at=? WHERE id=?",
            [key, _now(), self.job_id]
        )
        print(f"[r2d1] ✓ checkpoint  job={self.job_id}  epoch={epoch}")
        return key

    def load_latest_checkpoint(self, model_or_params=None, optimizer_state=None):
        """
        Download and deserialize the latest checkpoint from R2.
        Returns (epoch, loss, params_or_model, optimizer_state).
        """
        result = self._t._d1(
            "SELECT last_checkpoint FROM jobs WHERE id=?", [self.job_id]
        )
        key = result['result'][0]['results'][0]['last_checkpoint']
        if not key:
            raise ValueError(f"No checkpoint found for job {self.job_id}")

        data = self._t._s3.get_object(
            Bucket=self._t.bucket, Key=key
        )['Body'].read()

        epoch, loss, params, opt = deserialize(data, model_or_params, optimizer_state)
        print(f"[r2d1] ✓ resumed  job={self.job_id}  epoch={epoch}")
        return epoch, loss, params, opt

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def complete(self):
        """Mark job as completed in D1."""
        self._t._d1(
            "UPDATE jobs SET status='completed', updated_at=? WHERE id=?",
            [_now(), self.job_id]
        )
        print(f"[r2d1] ✓ job {self.job_id} completed")

    def interrupt(self):
        """
        Mark job as interrupted in D1.
        Call this in your except block — whatever is watching jobs
        (e.g. a scheduler) can then restart from last checkpoint.
        """
        self._t._d1(
            "UPDATE jobs SET status='interrupted', updated_at=? WHERE id=?",
            [_now(), self.job_id]
        )
        print(f"[r2d1] ⚡ job {self.job_id} interrupted")

    def status(self):
        """Return job metadata + all epoch metrics."""
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

    Connects to Cloudflare R2 (checkpoints) and D1 (metrics + job metadata).
    Run anywhere — local, Colab, vast.ai, any GPU server.

    from r2d1 import Tracker

    tracker = Tracker(
        account_id     = "...",
        api_token      = "...",
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
        """Create D1 tables if they don't exist. Safe to call every time."""
        self._d1("""
            CREATE TABLE IF NOT EXISTS jobs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                name             TEXT,
                dataset_key      TEXT,
                status           TEXT DEFAULT 'pending',
                last_checkpoint  TEXT,
                dataset_size_mb  REAL,
                submitted_at     TEXT,
                updated_at       TEXT
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

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def start_job(self, name, dataset_key=None, dataset_size_mb=None, **meta):
        """
        Register a new job in D1 and return a Job handle.
        Any extra kwargs are ignored — pass whatever context you want,
        e.g. vast_image, script_key — store them yourself if needed.
        """
        result = self._d1(
            """INSERT INTO jobs
               (name, dataset_key, status, dataset_size_mb, submitted_at, updated_at)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            [name, dataset_key, dataset_size_mb, _now(), _now()]
        )
        job_id = result['result'][0]['meta']['last_row_id']
        print(f"[r2d1] ✓ started job {job_id}: {name}")
        return Job(job_id, self)

    def resume_job(self, job_id, model_or_params=None, optimizer_state=None):
        """
        Resume a job from its latest R2 checkpoint.
        Returns (Job, start_epoch, last_loss, params, optimizer_state).
        """
        self._d1(
            "UPDATE jobs SET status='running', updated_at=? WHERE id=?",
            [_now(), job_id]
        )
        job = Job(job_id, self)
        epoch, loss, params, opt = job.load_latest_checkpoint(
            model_or_params, optimizer_state
        )
        return job, epoch, loss, params, opt

    def list_jobs(self):
        """Return all jobs and their current status."""
        return self._d1(
            "SELECT id, name, status, last_checkpoint, submitted_at, updated_at "
            "FROM jobs ORDER BY id DESC"
        )['result'][0]['results']

    def get_job(self, job_id):
        """Return a Job handle for an existing job (no checkpoint loaded)."""
        return Job(job_id, self)

    def job(self, name, dataset_key=None, dataset_size_mb=None,
            resume_job_id=None, model_or_params=None, optimizer_state=None):
        """
        Decorator — wraps a training function with full job lifecycle.
        Injects a Job as the first argument. Handles start/interrupt/complete.

        @tracker.job(name="dit_run1", dataset_key="datasets/imagenet.tar")
        def train(job):
            for epoch in EpochLoop(range(100), job=job, model=model, optimizer=opt):
                loss = train_step(...)
                epoch.loss = float(loss)

        train()
        """
        from .loop import job_decorator
        return job_decorator(
            self, name=name, dataset_key=dataset_key,
            dataset_size_mb=dataset_size_mb,
            resume_job_id=resume_job_id,
            model_or_params=model_or_params,
            optimizer_state=optimizer_state,
        )
