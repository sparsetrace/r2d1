# r2d1

Lightweight ML checkpoint courier. Ships checkpoint folders to Cloudflare **R2**
and records metadata to Cloudflare **D1**. Pulls existing checkpoints back down
before a run. Model-agnostic — works with any training code that writes files
to a local directory.

```bash
pip install r2d1
```

---

## How it fits into bob.py

```
bob_job.json (secrets + config)
       │
       ▼
   bob.py
     │
     ├─ Fetcher.from_config(secrets)
     │    └─ pull("r2://jobs/<id>/latest", dest)   ← checkpoint on disk (or fresh start)
     │
     ├─ Courier.from_config(secrets)
     │    └─ watch(checkpoint_dir, job_id)          ← ships new checkpoints in background
     │
     ├─ subprocess: torchrun DDIT                   ← trains, writes chk_N/ + chk_N.json
     │
     └─ courier.flush()                             ← wait for final upload, then exit
```

DDIT (and any training code) sees only local disk. It has no knowledge of R2,
D1, or any cloud infrastructure.

---

## Sidecar convention

Training code writes two things per checkpoint, **folder first, sidecar last**:

```
./checkpoints/
├── chk_0042/            ← checkpoint folder  → uploaded to R2
│   ├── checkpoint.pt
│   └── config.json
└── chk_0042.json        ← sidecar (written last) → triggers upload, upserted to D1
```

The sidecar is the atomic signal. Writing it last guarantees the folder is
complete before r2d1 touches it.

### Sidecar schema

```json
{
    "name":      "chk_0042",
    "epoch":     42,
    "timestamp": 1748123456.7,
    "files":     ["checkpoint.pt", "config.json"],
    "metadata":  {"loss": 0.043}
}
```

Any checkpoint format works — `checkpoint.pt`, `model.safetensors`,
HuggingFace `save_pretrained()` output, etc. r2d1 ships whatever is in the folder.

---

## Fetcher — pull checkpoints from R2

```python
from r2d1 import Fetcher

# Build from bob_job.json secrets block (also exports to os.environ)
fetcher = Fetcher.from_config(cfg["secrets"])

# Pull latest checkpoint for a job — returns found=False on fresh start, never raises
info = fetcher.pull("r2://jobs/my_run/latest", dest="/root/checkpoints")

if info.found:
    print(f"resuming from epoch {info.epoch} at {info.local_dir}")
else:
    print("no checkpoint — fresh start")

# Pull a specific checkpoint by name
fetcher.pull("r2://jobs/my_run/chk_0042", dest="/root/checkpoints")

# Pull any R2 prefix (e.g. a dataset stored in R2)
fetcher.pull("r2://datasets/mnist", dest="/root/data")
```

### URI schemes

| URI | What it does |
|-----|-------------|
| `r2://jobs/<id>/latest` | Find highest-epoch checkpoint (D1 → R2 fallback), download it |
| `r2://jobs/<id>/<name>` | Download a specific named checkpoint folder |
| `r2://<any/prefix>` | Download all objects under that R2 prefix |

`found=False` is returned (not raised) when no checkpoint exists — this is
the normal fresh-start case. Check `info.found` before loading weights.

---

## Courier — ship checkpoints to R2 + D1

```python
from r2d1 import Courier

courier = Courier.from_config(cfg["secrets"])

# Start background thread — returns immediately
courier.watch("/root/checkpoints", job_id="my_run")

# ... training runs here, writes chk_N/ + chk_N.json as it goes ...

# Wait for all uploads to finish before exiting
courier.flush(timeout=300)
```

Or as a fully decoupled subprocess:

```bash
python -m r2d1 watch /root/checkpoints --job-id my_run
```

---

## Credentials

Credentials come from the `secrets` block of `bob_job.json`. `from_config()`
exports them to `os.environ` so boto3 and requests pick them up automatically.

For local dev, put them in a `.env` file (requires `pip install r2d1[dotenv]`)
or set them directly in your shell.

### Required for R2

```
R2D1_ACCOUNT_ID
R2D1_R2_BUCKET
R2D1_R2_ACCESS_KEY
R2D1_R2_SECRET_KEY
R2D1_R2_ENDPOINT_URL   (optional — auto-constructed from ACCOUNT_ID if omitted)
```

### Optional for D1

```
R2D1_API_TOKEN
R2D1_D1_DATABASE_ID
```

If D1 credentials are absent, r2d1 runs in R2-only mode — checkpoints are
still shipped and pulled, no metadata rows are written.

---

## D1 schema

```sql
CREATE TABLE IF NOT EXISTS checkpoints (
    job_id    TEXT    NOT NULL,
    name      TEXT    NOT NULL,
    epoch     INTEGER NOT NULL,
    timestamp REAL    NOT NULL,   -- doubles as heartbeat
    r2_prefix TEXT    NOT NULL,
    metadata  TEXT    DEFAULT '{}',
    PRIMARY KEY (job_id, name)
);
```

Alice (or any orchestrator) can check job progress with:

```sql
SELECT epoch, timestamp FROM checkpoints
WHERE job_id = ? ORDER BY epoch DESC LIMIT 1;
```

---

## CLI

```bash
# Ship checkpoints as they appear
python -m r2d1 watch ./checkpoints --job-id my_run [--poll-every 30]

# Pull from R2
python -m r2d1 pull r2://jobs/my_run/latest --dest ./checkpoints
python -m r2d1 pull r2://datasets/mnist      --dest ./data

# Show discovered credentials
python -m r2d1 secrets
```

---

## Repo structure

```
r2d1/
├── src/
│   └── r2d1/
│       ├── __init__.py      # exports: Courier, Fetcher, FetchInfo
│       ├── __main__.py      # CLI
│       ├── courier.py       # Courier + _AsyncUploader
│       ├── fetcher.py       # Fetcher + FetchInfo
│       ├── d1.py            # D1Client (REST)
│       └── secrets.py       # credential resolution
├── tests/
│   └── test_r2d1.py
└── pyproject.toml
```
