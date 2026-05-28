# r2d1

Lightweight ML experiment tracker using Cloudflare **R2** (checkpoints) + **D1** (metrics & metadata).

- Works with **PyTorch and JAX**
- Runs anywhere — local, Colab, vast.ai, any GPU server
- No opinions about how your jobs are scheduled or restarted

```bash
pip install r2d1
```

---

## Setup

```python
from r2d1 import Tracker

tracker = Tracker(
    account_id     = "your_cloudflare_account_id",
    api_token      = "your_cloudflare_api_token",
    d1_database_id = "your_d1_database_id",
    r2_bucket      = "your_bucket_name",
    r2_access_key  = "your_r2_access_key",
    r2_secret_key  = "your_r2_secret_key",
)
# D1 tables are created automatically on first run
```

---

## Usage

### New job

```python
job = tracker.start_job("dit_run1", dataset_key="datasets/imagenet256.tar")

try:
    for epoch in range(num_epochs):
        t0 = time.time()

        loss, acc = train_one_epoch(...)   # your training code

        job.save_checkpoint(epoch, model, optimizer, loss)   # → R2
        job.log(epoch=epoch, loss=loss, accuracy=acc,        # → D1
                duration_sec=time.time() - t0)

    job.complete()

except Exception as e:
    job.interrupt()   # marks status='interrupted' in D1
    raise
```

### Resume after interruption

```python
job, start_epoch, last_loss, model, optimizer = tracker.resume_job(
    job_id=3, model_or_params=model, optimizer_state=optimizer
)

for epoch in range(start_epoch + 1, num_epochs):
    ...
```

### JAX — identical API, pass pytrees instead of nn.Module

```python
job.save_checkpoint(epoch, params, opt_state, loss)

job, start_epoch, loss, params, opt_state = tracker.resume_job(job_id=3)
```

### Check progress

```python
tracker.list_jobs()   # all jobs + status
job.status()          # this job's epochs + metrics
```

---

## D1 schema (auto-created)

**jobs** — one row per run: name, dataset_key, status, last_checkpoint, timestamps  
**epochs** — one row per epoch: loss, accuracy, duration_sec

---

## Files

```
r2d1/
├── r2d1/
│   ├── __init__.py     # exports Tracker, Job
│   ├── tracker.py      # Tracker + Job
│   └── checkpoint.py   # serialize/deserialize (torch + JAX → numpy)
├── setup.py
└── README.md
```

---

## Notes

- Checkpoints are serialized to **numpy** — framework-agnostic, so a JAX
  checkpoint could in principle be loaded by a torch script and vice versa
- `job.interrupt()` just sets a status flag in D1 — whatever monitors your
  jobs (a scheduler, a cron, a person) can query D1 and restart from
  `last_checkpoint`
- `r2d1` has no knowledge of vast.ai, Modal, or any scheduler
