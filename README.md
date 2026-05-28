# r2d1

Lightweight ML experiment tracker using Cloudflare **R2** (checkpoints) + **D1** (metrics & metadata).

- **Agnostic** — ships whatever files your model produces. Zero opinion on format.
- Works with **PyTorch, JAX, or anything else**
- Runs anywhere — local, Colab, vast.ai, any GPU server
- Async checkpoint uploads — GPU keeps training while files ship to R2

```bash
pip install r2d1
```

---

## How it works

```
Your model                        r2d1
----------                        ----
model.safetensors  ──────────→   ships to R2
config.json        ──────────→   jobs/job_3/epoch_42/
run.log            ──────────→

loss, accuracy, duration  ──→   logs to D1  (live feed for you, Alice, team)
```

D1 always gets: `epoch`, `loss`, `accuracy`, `duration_sec`, `timestamp`.  
R2 gets: whatever files you put in `epoch.files`. r2d1 never opens them.

---

## Setup

```python
from r2d1 import Tracker, EpochLoop

tracker = Tracker(
    account_id     = "your_cloudflare_account_id",
    api_token      = "your_cloudflare_api_token",   # Token value from R2 API token
    d1_database_id = "your_d1_database_id",
    r2_bucket      = "your_bucket_name",
    r2_access_key  = "your_r2_access_key",
    r2_secret_key  = "your_r2_secret_key",
)
# D1 tables created automatically on first run
```

---

## Usage

### tqdm-style loop

```python
job = tracker.start_job("dit_run1", dataset_key="hf://datasets/imagenet-1k")

try:
    for epoch in EpochLoop(range(400), job=job):

        loss = train_step(...)      # your JIT'd / multi-GPU code — untouched

        epoch.loss  = float(loss)
        epoch.files = {             # your model packages its own checkpoint
            "model.safetensors": model.to_safetensors_bytes(),
            "config.json":       model.to_config_bytes(),
            "run.log":           logger.flush_bytes(),
        }
        # ↑ files upload async to R2, loss logs to D1 — both happen automatically

    job.complete()

except Exception:
    job.interrupt()   # D1 status → 'interrupted', Alice can restart
    raise
```

### Resume after interruption

```python
job = tracker.resume_job(job_id=3)

files = job.load_latest()   # downloads all files from last checkpoint
# files["model.safetensors"], files["config.json"], etc.

model  = MyModel.from_safetensors_bytes(files["model.safetensors"],
                                        files["config.json"])
logger = Logger.from_bytes(files["run.log"])

for epoch in EpochLoop(range(400), job=job, start_epoch=last_epoch + 1):
    ...
```

### Decorator style

```python
@tracker.job(name="dit_run1", dataset_key="hf://datasets/imagenet-1k")
def train(job):
    for epoch in EpochLoop(range(400), job=job):
        loss = train_step(...)
        epoch.loss  = float(loss)
        epoch.files = model.checkpoint()

train()   # start/interrupt/complete handled automatically
```

### EpochLoop options

```python
EpochLoop(
    range(400),
    job              = job,
    checkpoint_every = 10,    # ship files every 10 epochs (default: 1)
    log_every        = 1,     # log to D1 every epoch (default: 1)
    start_epoch      = 50,    # skip first 50 epochs — for resuming
    async_checkpoint = True,  # upload in background (default: True)
)
```

---

## R2 layout

```
r2://your-bucket/
└── jobs/
    └── job_3/
        ├── epoch_0/
        │   ├── model.safetensors
        │   ├── config.json
        │   └── run.log
        ├── epoch_10/
        │   └── ...
        └── epoch_42/
            └── ...
```

---

## D1 schema (auto-created)

**jobs** — `id`, `name`, `dataset_key`, `status`, `last_checkpoint_prefix`, `submitted_at`, `updated_at`  
**epochs** — `job_id`, `epoch`, `loss`, `accuracy`, `duration_sec`, `logged_at`

---

## Files

```
r2d1/
├── r2d1/
│   ├── __init__.py   # exports Tracker, Job, EpochLoop
│   ├── tracker.py    # Tracker + Job — R2 upload, D1 logging
│   └── loop.py       # EpochLoop + @tracker.job decorator
├── setup.py
└── README.md
```
