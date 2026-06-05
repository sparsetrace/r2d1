# r2d1

Lightweight ML checkpoint courier for Cloudflare R2 and D1.

Ships checkpoint folders from local disk to R2 storage. Pulls them back
down before a run. Records metadata to D1 as a heartbeat. Model-agnostic —
works with any training code that writes files to a local directory.

```bash
pip install r2d1
```

---

## Core idea

Training code writes to local disk. r2d1 moves data in both directions
without the training code knowing anything about cloud infrastructure:

```
[Fetcher]  R2  →  local disk   pull before training
[Courier]  local disk  →  R2   push during training (background)
```

An external orchestrator sequences these steps explicitly. Training code
starts only after Fetcher completes, and sees only local filesystem paths.

---

## Sidecar convention

r2d1 watches a directory for pairs of a folder and a JSON sidecar.
The sidecar is written **last**, after all folder contents are flushed —
this makes it an atomic signal that the checkpoint is complete:

```
./checkpoints/
├── slot_0/              ← checkpoint folder  → uploaded to R2
│   ├── checkpoint.pt
│   └── config.json
└── slot_0.json          ← sidecar (written last) → triggers upload
```

r2d1 never inspects file contents. Any checkpoint format works —
`checkpoint.pt`, `model.safetensors`, HuggingFace `save_pretrained()`
output, or anything else. Whatever is in the folder gets shipped.

### Sidecar schema

```json
{
    "name":      "slot_0",
    "epoch":     42,
    "timestamp": 1748123456.7,
    "files":     ["checkpoint.pt", "config.json"],
    "metadata":  {"loss": 0.043}
}
```

---

## Fetcher — pull data and checkpoints from R2

```python
from r2d1 import Fetcher

fetcher = Fetcher.from_env()

# Pull a dataset or any R2 prefix to local disk
fetcher.pull("r2://datasets/mnist", dest="/root/data")

# Pull latest checkpoint for a job
# Returns found=False on fresh start — never raises
info = fetcher.pull("r2://jobs/my_run/latest", dest="/root/checkpoints")

if info.found:
    print(f"resuming from epoch {info.epoch} at {info.local_dir}")
else:
    print("no checkpoint — fresh start")

# Pull a specific prefix
fetcher.pull("r2://jobs/my_run/slot_0", dest="/root/checkpoints/slot_0")
```

### URI schemes

| URI | What it does |
|-----|-------------|
| `r2://jobs/<id>/latest` | Find highest-epoch checkpoint (D1 → R2 fallback), download it |
| `r2://<any/prefix>` | Download all objects under that R2 prefix to local disk |

`found=False` is returned when no checkpoint exists — the normal
fresh-start case. Check `info.found` before loading weights.

---

## Courier — ship checkpoints to R2 + D1

```python
from r2d1 import Courier

courier = Courier.from_env()

# Start background thread — returns immediately
courier.watch("./checkpoints", job_id="my_run")

# ... training runs here, writes slot_N/ + slot_N.json ...

# Wait for all uploads to finish before exiting
courier.flush(timeout=300)
```

Or as a fully decoupled subprocess:

```bash
python -m r2d1 watch ./checkpoints --job-id my_run
```

The courier keys on `(sidecar_name, epoch)` — so a slot that gets
overwritten with a new epoch is treated as a new event and re-shipped.
R2 always holds the latest version of each slot.

---

## Credentials

r2d1 reads credentials from `os.environ`. Set them before importing r2d1,
or use a `.env` file (requires `pip install r2d1[dotenv]`).

### Required for R2

```
R2D1_ACCOUNT_ID
R2D1_R2_BUCKET
R2D1_R2_ACCESS_KEY
R2D1_R2_SECRET_KEY
R2D1_R2_ENDPOINT_URL    optional — auto-constructed from ACCOUNT_ID if omitted
R2D1_SESSION_TOKEN      optional — set when using scoped temporary credentials
```

### Optional for D1

```
R2D1_API_TOKEN
R2D1_D1_DATABASE_ID
```

If D1 credentials are absent, r2d1 runs in R2-only mode — checkpoints are
still shipped and pulled, no metadata rows are written, a warning is
printed once.

### Temporary / scoped credentials

r2d1 supports AWS-compatible session tokens for short-lived scoped
credentials. Set `R2D1_SESSION_TOKEN` alongside the access and secret keys.
boto3 will include it automatically on every request.

---

## D1 schema

```sql
CREATE TABLE IF NOT EXISTS checkpoints (
    job_id    TEXT    NOT NULL,
    name      TEXT    NOT NULL,
    epoch     INTEGER NOT NULL,
    timestamp REAL    NOT NULL,
    r2_prefix TEXT    NOT NULL,
    metadata  TEXT    DEFAULT '{}',
    PRIMARY KEY (job_id, name)
);
```

The `timestamp` column doubles as a heartbeat — query it to determine
whether a job is still making progress.

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
│       ├── __init__.py      exports: Courier, Fetcher, FetchInfo
│       ├── __main__.py      CLI
│       ├── courier.py       Courier + _AsyncUploader
│       ├── fetcher.py       Fetcher + FetchInfo
│       ├── d1.py            D1Client (REST)
│       └── secrets.py       credential resolution + boto3 client factory
├── tests/
│   └── test_r2d1.py
└── pyproject.toml
```
