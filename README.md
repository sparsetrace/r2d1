# r2d1

Lightweight ML checkpoint courier using Cloudflare **R2** for durable artifacts and optional **D1** for metrics/metadata.

`r2d1` does not care how your model checkpoints are formatted. It ships whatever files or JSON blobs you hand it.

```bash
pip install r2d1
```

## Core idea

```python
from pathlib import Path
from r2d1 import start_job, r2d1

job = start_job("mnist_dit")

for epoch in r2d1(range(10), job=job, checkpoint_every=1, keep_last=2):
    loss = train_one_epoch(...)

    epoch.d1(loss=float(loss))  # optional D1 metrics if D1 is configured

    if epoch.should_checkpoint:
        epoch.r2({              # durable artifacts/checkpoints to R2
            "checkpoint.pt": Path("ckpt/checkpoint.pt"),
            "config.json": {"epoch": epoch.i},
        })

job.complete()
```

## Credentials

`r2d1` searches for secrets in:

1. `.env` in the current directory or parent directories
2. `os.environ`
3. Google Colab secrets via `google.colab.userdata`
4. Kaggle notebook secrets via `kaggle_secrets.UserSecretsClient`

Modal, Vast.ai, RunPod, Docker, CI, SageMaker, Vertex, Lightning AI, Paperspace, etc. are covered when those platforms inject secrets into environment variables.

### Required for R2 checkpointing

```bash
export R2D1_ACCOUNT_ID="..."
export R2D1_R2_BUCKET="..."
export R2D1_R2_ACCESS_KEY="..."
export R2D1_R2_SECRET_KEY="..."
# optional:
export R2D1_R2_ENDPOINT_URL="https://<account_id>.r2.cloudflarestorage.com"
```

Aliases such as `CLOUDFLARE_ACCOUNT_ID`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_KEY`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` are also recognized. `R2D1_*` names take priority.

### Optional for D1 metrics/status

```bash
export R2D1_API_TOKEN="..."
export R2D1_D1_DATABASE_ID="..."
```

If D1 credentials are missing, `r2d1` prints a warning and continues in R2-only mode.

## Secret utility

You can use `r2d1` as a general notebook/cloud secret resolver:

```python
from r2d1 import secret, export_secrets

hf_token = secret("HF_TOKEN", aliases=["HF_HUB_TOKEN"], required=False)
github_token = secret("GITHUB_TOKEN", aliases=["GH_TOKEN"], required=False)

export_secrets(["HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY"], required=False)
```

If found, values are copied into `os.environ` so downstream libraries can use them.

## Top-level API

The common path does not require `Tracker.from_env()`:

```python
from r2d1 import start_job, r2d1

job = start_job("my_run")
for epoch in r2d1(range(100), job=job):
    ...
```

Advanced/manual path:

```python
from r2d1 import Tracker

tracker = Tracker.from_env()  # lazy; does not require R2/D1 immediately
job = tracker.start_job("my_run")  # validates R2 here, warns if D1 is missing
```

## Last-two checkpoints

By default, `keep_last=2` rotates checkpoint uploads through two R2 slots:

```text
jobs/<job_id>/checkpoints/slot_0/
jobs/<job_id>/checkpoints/slot_1/
```

With `checkpoint_every=10`, epochs `0, 10, 20, 30` map to `slot_0, slot_1, slot_0, slot_1`. Use stable artifact names like `checkpoint.pt` so R2 storage stays bounded.

## R2-only mode

D1 is useful but optional. In R2-only mode, `epoch.r2(...)` still uploads checkpoints and writes:

```text
jobs/<job_id>/job.json
jobs/<job_id>/latest.json
jobs/<job_id>/checkpoints/slot_*/manifest.json
```

`epoch.d1(...)` warns once and does not write SQL metrics.

## Build/publish

```bash
python -m pip install -U build twine
python -m build
twine check dist/*
twine upload dist/*
```
