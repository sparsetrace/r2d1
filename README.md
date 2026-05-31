# r2d1

Tiny ML experiment tracking on Cloudflare **R2** + **D1**.

- **R2** stores checkpoint/artifact files.
- **D1** stores job metadata, epoch metrics, and checkpoint pointers.
- Works from local scripts, notebooks, Colab, Kaggle, Vast.ai, Modal, RunPod, Docker, CI, and ordinary GPU servers.
- Keeps training code framework-agnostic: PyTorch, JAX, or anything that can write files.

```bash
pip install r2d1
```

## 0.1.5 credential discovery

`r2d1` now has a universal secret resolver:

```python
from r2d1 import secret

hf_token = secret("HF_TOKEN", aliases=["HF_HUB_TOKEN"], required=False)
github_token = secret("GITHUB_TOKEN", aliases=["GH_TOKEN"], required=False)
```

The resolver searches:

1. `.env` in the current directory or parents, with `override=False`
2. `os.environ`
3. Google Colab secrets via `google.colab.userdata`, when available
4. Kaggle secrets via `kaggle_secrets.UserSecretsClient`, when available

Modal, Vast.ai, RunPod, Docker, GitHub Actions, SageMaker, Vertex AI Workbench, Lightning AI, Paperspace, JupyterHub, Hugging Face Spaces, and similar platforms are covered when they inject secrets as environment variables.

By default, found secrets are copied into `os.environ[NAME]`, so downstream libraries can see them too.

## R2D1 credentials

Put these in a private `.env`, notebook secret store, Modal secret, Vast.ai env vars, or your shell environment:

```bash
export R2D1_ACCOUNT_ID="..."
export R2D1_API_TOKEN="..."
export R2D1_D1_DATABASE_ID="..."
export R2D1_R2_BUCKET="..."
export R2D1_R2_ACCESS_KEY="..."
export R2D1_R2_SECRET_KEY="..."
# optional:
export R2D1_R2_ENDPOINT_URL="https://<account_id>.r2.cloudflarestorage.com"
```

Then use:

```python
from r2d1 import Tracker

tracker = Tracker.from_env()
```

`Tracker.from_env()` is strict by default. If any required R2D1 key is missing, it raises `MissingSecretError` listing all names and environments it tried.

## Basic usage

```python
from pathlib import Path
from r2d1 import Tracker, r2d1

tracker = Tracker.from_env()
job = tracker.start_job(
    "mnist_dit",
    dataset_key="hf://datasets/ylecun/mnist",
    config={"model": "tiny-pixel-dit"},
    tags=["mnist", "flow-matching"],
)

for epoch in r2d1(range(10), job=job, checkpoint_every=1, keep_last=2):
    loss = train_one_epoch(...)

    # small JSON metrics/metadata -> D1
    epoch.d1(loss=float(loss), lr=float(lr))

    # files/artifacts/checkpoints -> R2
    if epoch.should_checkpoint:
        epoch.r2({
            "checkpoint.pt": Path("ckpt/checkpoint.pt"),
            "config.json": {"epoch": epoch.i},
        })

job.complete()
```

Aliases are provided:

```python
epoch.log(...)        # same as epoch.d1(...)
epoch.checkpoint(...) # same as epoch.r2(...)
```

## Decorator style

```python
from r2d1 import Tracker, r2d1

tracker = Tracker.from_env()

@tracker.job(name="dit_run", dataset_key="hf://datasets/ylecun/mnist")
def train(job):
    for epoch in r2d1(range(10), job=job, checkpoint_every=1, keep_last=2):
        loss = train_one_epoch(...)
        epoch.d1(loss=float(loss))
        if epoch.should_checkpoint:
            epoch.r2({"checkpoint.pt": "ckpt/checkpoint.pt"})

train()
```

Clean exit marks the job completed. Exceptions/interrupts mark it interrupted.

## Rotating checkpoints

`keep_last=2` uses rotating R2 slots:

```text
jobs/job_3/checkpoints/slot_0/
jobs/job_3/checkpoints/slot_1/
```

Each upload writes files first, then `manifest.json`, then updates D1. D1 only points at complete checkpoints.

## Resume

```python
tracker = Tracker.from_env()
job = tracker.resume_job(3)
files, manifest = job.load_latest(include_manifest=True)

# files["checkpoint.pt"] contains checkpoint bytes.
# manifest["epoch"] tells you where to resume.
```

## Universal secrets for notebooks/cloud jobs

```python
from r2d1 import secret, export_secrets

# Optional secrets, no error if missing.
hf_token = secret("HF_TOKEN", aliases=["HF_HUB_TOKEN"], required=False)
github_token = secret("GITHUB_TOKEN", aliases=["GH_TOKEN"], required=False)

# Load several and populate os.environ.
export_secrets(["HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY"], required=False)
```

Strict mode:

```python
github_token = secret("GITHUB_TOKEN")  # raises MissingSecretError if absent
```

No secret values are logged by r2d1.

## Install extras

```bash
pip install r2d1[torch]
pip install r2d1[jax]
pip install r2d1[dev]
```

`r2d1` does not own your model serialization. Prefer framework-native or safe formats such as `.safetensors`, PyTorch `.pt`, Orbax/Flax outputs, JSON configs, logs, images, etc. `epoch.r2(...)` ships the files you provide.
