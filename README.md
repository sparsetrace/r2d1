# r2d1

Lightweight ML experiment tracker using Cloudflare **R2** (checkpoints) + **D1** (metrics & metadata).

- Works with **PyTorch and JAX**, single or multi-GPU
- Runs anywhere — local, Colab, vast.ai, any GPU server
- No opinions about how jobs are scheduled or restarted

```bash
pip install r2d1
```

---

## Setup

```python
from r2d1 import Tracker, EpochLoop

tracker = Tracker(
    account_id     = "your_cloudflare_account_id",
    api_token      = "your_cloudflare_api_token",
    d1_database_id = "your_d1_database_id",
    r2_bucket      = "your_bucket_name",
    r2_access_key  = "your_r2_access_key",
    r2_secret_key  = "your_r2_secret_key",
)
```

---

## Usage

### tqdm-style loop (recommended)

```python
job = tracker.start_job("dit_run1", dataset_key="datasets/imagenet256.tar")

try:
    for epoch in EpochLoop(range(num_epochs), job=job, model=model, optimizer=opt):
        # your JIT'd / multi-GPU training code — completely untouched
        loss, grads = train_step(params, batch)
        params = update(params, grads)

        # set metrics on the epoch context object
        epoch.loss     = float(loss)
        epoch.accuracy = float(acc)
        # ↑ D1 log + R2 checkpoint happen automatically at end of each iteration

    job.complete()

except Exception as e:
    job.interrupt()   # Alice (or you) can restart from last checkpoint
    raise
```

### Resume after interruption

```python
job, start_epoch, last_loss, params, opt = tracker.resume_job(
    job_id=3, model_or_params=model, optimizer_state=opt
)

for epoch in EpochLoop(range(num_epochs), job=job, model=params,
                       optimizer=opt, start_epoch=start_epoch + 1):
    ...
```

### Decorator-style (handles lifecycle automatically)

```python
@tracker.job(name="dit_run1", dataset_key="datasets/imagenet256.tar")
def train(job):
    for epoch in EpochLoop(range(num_epochs), job=job, model=model, optimizer=opt):
        loss, grads = train_step(params, batch)
        epoch.loss = float(loss)

train()   # start/interrupt/complete handled for you
```

### Resume with decorator

```python
@tracker.job(name="dit_run1", resume_job_id=3,
             model_or_params=model, optimizer_state=opt)
def train(job, start_epoch=0, params=None, opt_state=None):
    for epoch in EpochLoop(range(num_epochs), job=job,
                           model=params, optimizer=opt_state,
                           start_epoch=start_epoch):
        ...

train()
```

### JAX — identical, pass pytrees instead of nn.Module

```python
for epoch in EpochLoop(range(num_epochs), job=job, model=params, optimizer=opt_state):
    params, opt_state, loss = train_step(params, opt_state, batch)
    epoch.loss = float(loss)
```

### EpochLoop options

```python
EpochLoop(
    range(num_epochs),
    job              = job,
    model            = model,        # nn.Module or JAX pytree
    optimizer        = opt,          # optional
    checkpoint_every = 5,            # checkpoint every 5 epochs (default: 1)
    log_every        = 1,            # log to D1 every epoch (default: 1)
    start_epoch      = 10,           # skip first 10 epochs (used when resuming)
)
```

---

## Dataset location

`dataset_key` is just a string — use whatever convention makes sense:

```python
tracker.start_job("run1", dataset_key="s3://my-bucket/imagenet.tar")
tracker.start_job("run1", dataset_key="hf://datasets/imagenet-1k")
tracker.start_job("run1", dataset_key="r2://my-bucket/imagenet256.tar")
```

`r2d1` stores it as metadata in D1. How you actually load the dataset is up to your training script.

---

## Check progress

```python
tracker.list_jobs()   # all jobs + status
job.status()          # this job's per-epoch metrics
```

---

## Files

```
r2d1/
├── r2d1/
│   ├── __init__.py     # exports Tracker, Job, EpochLoop
│   ├── tracker.py      # Tracker + Job
│   ├── loop.py         # EpochLoop + job_decorator
│   └── checkpoint.py   # serialize/deserialize (torch + JAX → numpy)
├── setup.py
└── README.md
```
