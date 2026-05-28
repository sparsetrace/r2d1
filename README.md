# R2D1

Lightweight ML experiment tracker using Cloudflare R2 (checkpoints) + D1 (metadata & metrics).  
Your own W&B — no extra servers, no third-party accounts.

## Install

```bash
pip install -e .   # local install from this folder
# or once published:
pip install cf-trainer
```

## Setup

You need:
- `ACCOUNT_ID` — from your Cloudflare dashboard URL
- `API_TOKEN` — Token value from R2 API token creation
- `D1_DATABASE_ID` — from Cloudflare D1 dashboard
- `R2_ACCESS_KEY` / `R2_SECRET_KEY` — from R2 API token creation
- `R2_BUCKET_NAME` — your R2 bucket name

## Usage

```python
from cf_trainer import Tracker

tracker = Tracker(
    account_id    = ACCOUNT_ID,
    api_token     = API_TOKEN,
    d1_database_id= D1_DATABASE_ID,
    r2_bucket     = BUCKET_NAME,
    r2_access_key = ACCESS_KEY,
    r2_secret_key = SECRET_KEY,
)

# --- Start a new job ---
job = tracker.start_job("resnet_run1", dataset_key="datasets/imagenet.tar")

try:
    for epoch in range(num_epochs):
        t0 = time.time()

        # ... your training loop here ...

        duration = time.time() - t0
        ckpt_key = job.save_checkpoint(epoch, model, optimizer, loss)
        job.log(epoch=epoch, loss=loss, accuracy=acc, duration_sec=duration)

    job.complete()

except Exception as e:
    job.interrupt()   # marks job as interrupted in D1
    raise e


# --- Resume an interrupted job ---
job, start_epoch, last_loss = tracker.resume_job(
    job_id=1, model=model, optimizer=optimizer
)

for epoch in range(start_epoch + 1, num_epochs):
    # ... continue training ...
    job.save_checkpoint(epoch, model, optimizer, loss)
    job.log(epoch=epoch, loss=loss, accuracy=acc)

job.complete()


# --- Check progress ---
print(tracker.list_jobs())   # all jobs + status
print(job.status())          # this job's epochs + metrics
```

## Structure

```
cf_trainer/
├── cf_trainer/
│   ├── __init__.py   # exports Tracker, Job
│   └── tracker.py    # all logic (~150 lines)
├── setup.py
└── README.md
```

## D1 Schema

**jobs** — one row per training run  
**epochs** — one row per epoch, linked to job

Both tables are created automatically on first run.
