# r2d1

Lightweight ML checkpoint courier. Ships checkpoint folders to Cloudflare **R2**
and records metadata to Cloudflare **D1**. Model-agnostic — works with any
training code that follows the sidecar convention.

```bash
pip install r2d1
```

## Core idea

Any training code writes two things per checkpoint:

```
./checkpoints/
|-- chk_0042/          # checkpoint folder  → shipped to R2
`-- chk_0042.json      # JSON sidecar       → sent to D1, triggers ship
```

The sidecar is written **last** (after all folder contents are flushed),
providing an atomic multi-file readiness signal. `r2d1` polls for new
`.json` sidecars; when one appears the folder is guaranteed complete.

## Sidecar schema

```json
{
    "name":      "chk_0042",
    "epoch":     42,
    "timestamp": 1748123456.7,
    "files":     ["checkpoint.pt", "config.json"],
    "metadata":  {"loss": 0.043}
}
```

## Courier — ship checkpoints as they appear

```python
from r2d1 import Courier

courier = Courier.from_env()

# Option A: background thread (in-process)
courier.watch("./checkpoints", job_id="my_run")
# ... training writes chk_N/ + chk_N.json ...
courier.flush(timeout=300)   # wait for final upload before exit

# Option B: subprocess (fully decoupled from training process)
# python -m r2d1 watch ./checkpoints --job-id my_run --poll-every 30
```

## Restarter — resume from latest checkpoint

```python
from r2d1 import Restarter

info = Restarter.from_env().pull(
    job_id = "my_run",
    dest   = "/workspace/checkpoints",
)

if info.found:
    # info.local_dir  -- Path to downloaded checkpoint folder
    # info.epoch      -- epoch number of the checkpoint
    model.load_state_dict(torch.load(info.local_dir / "checkpoint.pt"))
    start_epoch = info.epoch + 1
else:
    start_epoch = 0
```

Restarter queries D1 first (fast). Falls back to scanning R2 if D1 is
unavailable.

## Orchestration via bob.py

In a typical deployment a separate `bob.py` sequences things explicitly:

1. `Restarter.pull()` — **blocks** until checkpoint is downloaded locally
2. Model program starts — sees only local files, zero cloud knowledge
3. `Courier.watch()` — runs in background, ships new checkpoints as they appear

This keeps the model completely decoupled from cloud infrastructure.

## Credentials

`r2d1` searches in order (does not override existing env vars):

1. `.env` in current directory or parents
2. `os.environ` — covers Modal, Vast.ai, RunPod, Docker, CI, SageMaker, etc.
3. Google Colab `userdata`
4. Kaggle `UserSecretsClient`

### Required for R2

```bash
export R2D1_ACCOUNT_ID="..."
export R2D1_R2_BUCKET="..."
export R2D1_R2_ACCESS_KEY="..."
export R2D1_R2_SECRET_KEY="..."
# optional:
export R2D1_R2_ENDPOINT_URL="https://<account_id>.r2.cloudflarestorage.com"
```

Aliases: `CLOUDFLARE_ACCOUNT_ID`, `R2_BUCKET`, `AWS_ACCESS_KEY_ID`, etc.
`R2D1_*` names take priority.

### Optional for D1

```bash
export R2D1_API_TOKEN="..."
export R2D1_D1_DATABASE_ID="..."
```

If D1 credentials are absent, `r2d1` runs in R2-only mode — checkpoints are
still shipped, no metadata rows are written, a warning is printed once.

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

The table doubles as a **heartbeat** — check `timestamp` of the latest row
to determine whether a job is still making progress.

## CLI

```bash
# Watch and ship
python -m r2d1 watch ./checkpoints --job-id my_run

# Pull latest checkpoint
python -m r2d1 pull --job-id my_run --dest ./checkpoints

# Show discovered credentials
python -m r2d1 secrets
```

## Secret utility

```python
from r2d1 import secret, export_secrets, discover_common_secrets

hf_token = secret("HF_TOKEN", required=False)
export_secrets(["HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY"], required=False)
discover_common_secrets()   # exports all common ML tokens opportunistically
```
