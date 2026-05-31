# r2d1

Tiny ML experiment tracking on Cloudflare **R2** + **D1**.

```python
from pathlib import Path
from r2d1 import Tracker, r2d1

tracker = Tracker.from_env()

@tracker.job(
    name="dit_imagenet_sm103_test",
    dataset_key="hf://datasets/imagenet-1k",
    config={"model": "DiT", "hardware": "sm103", "dtype": "bf16"},
    tags=["dit", "imagenet", "sm103"],
)
def train(job):
    for epoch in r2d1(range(400), job=job, log_every=1, checkpoint_every=10, keep_last=2):
        loss = train_step(...)

        # Small structured data -> D1
        epoch.d1(
            loss=float(loss),
            lr=float(lr),
            samples_per_sec=float(samples_per_sec),
        )

        # Big files/artifacts/checkpoints -> R2
        if epoch.should_checkpoint:
            save_checkpoint_to_disk(model, optimizer, "ckpt/")
            epoch.r2({
                "model.safetensors": Path("ckpt/model.safetensors"),
                "optimizer.safetensors": Path("ckpt/optimizer.safetensors"),
                "config.json": {"epoch": epoch.i, "model": "DiT"},
            })

train()
```

## Mental model

```text
epoch.d1({...})  -> small JSON metrics/metadata -> D1
epoch.r2({...})  -> files/artifacts/checkpoints -> R2
```

Aliases are included:

```python
epoch.log(...)        # same as epoch.d1(...)
epoch.checkpoint(...) # same as epoch.r2(...)
```

The loop is intentionally lowercase like `tqdm`:

```python
from r2d1 import r2d1

for epoch in r2d1(range(100), job=job):
    ...
```

## Install locally

```bash
pip install -e .
```

## Cloudflare credentials

`Tracker.from_env()` reads:

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

Or construct directly:

```python
tracker = Tracker(
    account_id="...",
    api_token="...",
    d1_database_id="...",
    r2_bucket="...",
    r2_access_key="...",
    r2_secret_key="...",
)
```

## Checkpoint rotation: keep only last two

Use `keep_last=2`:

```python
for epoch in r2d1(range(400), job=job, checkpoint_every=10, keep_last=2):
    if epoch.should_checkpoint:
        epoch.r2({"model.safetensors": Path("model.safetensors")})
```

R2 layout:

```text
jobs/job_3/checkpoints/slot_0/
  manifest.json
  model.safetensors

jobs/job_3/checkpoints/slot_1/
  manifest.json
  model.safetensors
```

Epoch 0 writes `slot_0`, epoch 10 writes `slot_1`, epoch 20 overwrites `slot_0`, etc. D1 only points at a checkpoint after the files and `manifest.json` have been uploaded successfully.

## D1 schema

`r2d1` creates three tables automatically:

- `jobs`: run lifecycle + latest checkpoint pointer
- `epochs`: per-epoch metrics, with fixed columns plus `metrics_json`
- `checkpoints`: committed checkpoint manifests

`epoch.d1(...)` stores both convenience columns and a full JSON payload:

```python
epoch.d1({
    "loss": 0.123,
    "lr": 3e-4,
    "fid": 18.2,
    "gpu": "B300",
})
```

## R2 inputs

`epoch.r2(...)` accepts:

- `Path` objects — streamed to R2, preferred for large checkpoints
- `bytes` / `bytearray`
- `str` — existing file path if it exists, otherwise UTF-8 text
- `dict` / `list` / JSON-ish objects — encoded as JSON
- file-like objects — spooled to a temp file, then uploaded

Example:

```python
epoch.r2({
    "model.safetensors": Path("ckpt/model.safetensors"),
    "config.json": {"hidden_size": 1152, "depth": 28},
    "notes.txt": "loss looked stable",
})
```

## Resume

```python
job = tracker.resume_job(3)
files, manifest = job.load_latest(include_manifest=True)

print(manifest["epoch"])
model_bytes = files["model.safetensors"]
```

`load_latest()` validates file SHA256 hashes against the manifest.

## Optional serialization helpers

Core r2d1 does not own model serialization. It ships whatever you provide.

There is an optional `r2d1.checkpoint` module for small/simple PyTorch/JAX cases, but for real training runs prefer framework-native checkpoint files such as `safetensors`, Orbax, Flax serialization, `torch.save`, etc., then pass those file paths to `epoch.r2(...)`.
