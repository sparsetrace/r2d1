# R2D1

Lightweight ML experiment tracker using Cloudflare **R2** (checkpoints) + **D1** (metrics & metadata).  
Works with **PyTorch and JAX**. Designed to survive interruptions.

```
pip install r2d1
```

---

## Architecture

```
You
 └── submit job (D1)
      │
      ▼
Alice (Modal — persistent, always on)
 └── polls D1 every 5 min
 └── finds interrupted/pending jobs
 └── spins up Bob on vast.ai
 └── passes job info (R2 checkpoint + D1 metadata)
      │
      ▼
Bob (vast.ai GPU — cheap, disposable)
 └── installs r2d1
 └── downloads training script from R2
 └── resumes from last checkpoint
 └── logs epochs to D1, checkpoints to R2
 └── if interrupted → Alice restarts him
```

---

## Bob — training script (torch or JAX)

```python
import os
from r2d1 import Tracker

tracker = Tracker(
    account_id     = os.environ["CLOUDFLARE_ACCOUNT_ID"],
    api_token      = os.environ["CLOUDFLARE_API_TOKEN"],
    d1_database_id = os.environ["CLOUDFLARE_D1_DATABASE_ID"],
    r2_bucket      = os.environ["CLOUDFLARE_R2_BUCKET"],
    r2_access_key  = os.environ["CLOUDFLARE_R2_ACCESS_KEY"],
    r2_secret_key  = os.environ["CLOUDFLARE_R2_SECRET_KEY"],
)

job_id = int(os.environ.get("R2D1_JOB_ID", 0))

# --- New job ---
if job_id == 0:
    job = tracker.start_job(
        name          = "dit_run1",
        dataset_key   = "datasets/imagenet256.tar",
        vast_image    = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
        vast_gpu_type = "RTX_4090",
        script_key    = "scripts/train_dit.py",   # R2 path to this script
    )
    start_epoch = 0

# --- Resume interrupted job ---
else:
    job, start_epoch, last_loss, model, optimizer = tracker.resume_job(
        job_id, model, optimizer
    )
    start_epoch += 1

# --- Training loop ---
try:
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        loss, acc = train_one_epoch(model, dataloader, optimizer)
        duration  = time.time() - t0

        job.save_checkpoint(epoch, model, optimizer, loss)
        job.log(epoch=epoch, loss=loss, accuracy=acc, duration_sec=duration)

    job.complete()

except Exception as e:
    job.interrupt()   # Alice will restart Bob
    raise e
```

### JAX — same API, different objects

```python
# Pass params pytree instead of nn.Module
job.save_checkpoint(epoch, params, opt_state, loss)

job, start_epoch, loss, params, opt_state = tracker.resume_job(job_id)
```

---

## Alice — deploy once, runs forever

```bash
pip install r2d1[alice]
modal deploy alice.py
```

Set these in Modal dashboard → Secrets → `r2d1-secrets`:
```
CLOUDFLARE_ACCOUNT_ID
CLOUDFLARE_API_TOKEN
CLOUDFLARE_D1_DATABASE_ID
CLOUDFLARE_R2_BUCKET
CLOUDFLARE_R2_ACCESS_KEY
CLOUDFLARE_R2_SECRET_KEY
VASTAI_API_KEY
```

---

## Check progress (from anywhere)

```python
tracker.list_jobs()        # all jobs + status
job.status()               # this job's metrics per epoch
```

---

## Files

```
r2d1/
├── r2d1/
│   ├── __init__.py       # exports Tracker, Job
│   ├── tracker.py        # Tracker + Job classes
│   └── checkpoint.py     # serialize/deserialize (torch + JAX, numpy-based)
├── alice.py              # Modal server — polls D1, spins up vast.ai
├── setup.py
└── README.md
```

## D1 schema (auto-created)

**jobs** — one row per training run, includes vast.ai image + gpu type + script location  
**epochs** — one row per epoch with loss, accuracy, duration
