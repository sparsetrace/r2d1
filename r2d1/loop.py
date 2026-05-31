"""
r2d1 loop utilities.

The exported `r2d1` class is intentionally lowercase, like tqdm:

    from r2d1 import r2d1

    for epoch in r2d1(range(400), job=job):
        epoch.d1(loss=...)
        if epoch.should_checkpoint:
            epoch.r2({"model.safetensors": Path("model.safetensors")})
"""
from __future__ import annotations

import functools
import time
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional


class Epoch:
    """
    Context object yielded by r2d1(...).

    Methods
    -------
    d1(metrics=None, **kwargs)
        Queue small JSON-serializable metrics/metadata for D1.

    r2(files)
        Queue files/artifacts/checkpoints for R2.

    Aliases
    -------
    log == d1
    checkpoint == r2
    """

    def __init__(self, i: int, *, job: Any, should_log: bool, should_checkpoint: bool):
        self.i = int(i)
        self.epoch = int(i)
        self.job = job
        self.should_log = bool(should_log)
        self.should_checkpoint = bool(should_checkpoint)
        self._metrics: Dict[str, Any] = {}
        self._files: Optional[Mapping[str, Any]] = None

    def d1(self, metrics: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Queue metrics/metadata to be written to D1 at the end of this iteration."""
        data: Dict[str, Any] = {}
        if metrics:
            data.update(dict(metrics))
        data.update(kwargs)
        self._metrics.update(data)
        return self._metrics

    log = d1

    def r2(self, files: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Queue artifacts/checkpoint files to be uploaded to R2.

        Supported values include pathlib.Path, bytes, bytearray, str, dict/list JSON,
        and file-like objects. Paths are streamed; bytes are uploaded in-memory.
        """
        self._files = files
        return files

    checkpoint = r2


class r2d1:
    """
    tqdm-style epoch loop wrapper.

    Parameters
    ----------
    iterable:
        Iterable of epoch numbers, usually range(num_epochs).
    job:
        r2d1 Job handle returned by tracker.start_job/resume_job or injected by
        @tracker.job.
    log_every:
        Write metrics to D1 every N epochs.
    checkpoint_every:
        Upload queued R2 files every N epochs.
    keep_last:
        Number of rotating R2 checkpoint slots to keep. Default 2.
    start_epoch:
        Skip epochs before this value, for resuming.
    async_checkpoint:
        Upload R2 files in a background thread. D1 checkpoint pointer is updated
        only after upload + manifest succeed.
    progress:
        Print compact progress lines.
    """

    def __init__(
        self,
        iterable: Iterable[int],
        *,
        job: Any,
        log_every: int = 1,
        checkpoint_every: int = 1,
        keep_last: int = 2,
        start_epoch: int = 0,
        async_checkpoint: bool = True,
        progress: bool = True,
    ):
        if log_every <= 0:
            raise ValueError("log_every must be >= 1")
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be >= 1")
        if keep_last <= 0:
            raise ValueError("keep_last must be >= 1")

        self.iterable = iterable
        self.job = job
        self.log_every = int(log_every)
        self.checkpoint_every = int(checkpoint_every)
        self.keep_last = int(keep_last)
        self.start_epoch = int(start_epoch)
        self.async_checkpoint = bool(async_checkpoint)
        self.progress = bool(progress)

    def __iter__(self) -> Iterator[Epoch]:
        try:
            total = len(self.iterable)  # type: ignore[arg-type]
        except Exception:
            total = None

        for raw_i in self.iterable:
            i = int(raw_i)
            if i < self.start_epoch:
                continue

            should_log = (i % self.log_every) == 0
            should_checkpoint = (i % self.checkpoint_every) == 0
            epoch = Epoch(i, job=self.job, should_log=should_log, should_checkpoint=should_checkpoint)
            t0 = time.time()

            yield epoch

            duration_sec = round(time.time() - t0, 3)

            if epoch.should_log:
                self.job.d1(epoch=i, duration_sec=duration_sec, metrics=epoch._metrics)

            if epoch._files is not None and epoch.should_checkpoint:
                if self.async_checkpoint:
                    self.job.r2_async(
                        epoch=i,
                        files=epoch._files,
                        metrics=epoch._metrics,
                        keep_last=self.keep_last,
                        checkpoint_every=self.checkpoint_every,
                    )
                else:
                    self.job.r2(
                        epoch=i,
                        files=epoch._files,
                        metrics=epoch._metrics,
                        keep_last=self.keep_last,
                        checkpoint_every=self.checkpoint_every,
                    )

            if self.progress:
                self._print_progress(i, total, duration_sec, epoch._metrics)

        self.job.wait()

    @staticmethod
    def _print_progress(i: int, total: Optional[int], duration_sec: float, metrics: Mapping[str, Any]) -> None:
        pieces = [f"[r2d1] epoch {i}"]
        if total is not None:
            pieces[0] += f"/{max(total - 1, 0)}"
        pieces.append(f"{duration_sec:.1f}s")

        loss = metrics.get("loss") if metrics else None
        acc = metrics.get("accuracy", metrics.get("acc")) if metrics else None
        if isinstance(loss, (int, float)):
            pieces.append(f"loss={loss:.4f}")
        if isinstance(acc, (int, float)):
            pieces.append(f"acc={acc:.4f}")
        print("  ".join(pieces), flush=True)


def job_decorator(
    tracker: Any,
    *,
    name: str,
    dataset_key: Optional[str] = None,
    dataset_size_mb: Optional[float] = None,
    config: Optional[Mapping[str, Any]] = None,
    tags: Optional[Iterable[str]] = None,
    resume_job_id: Optional[int] = None,
):
    """
    Return a decorator that injects a Job as the first argument.

    Clean exit -> job.complete().
    Exception/KeyboardInterrupt -> job.interrupt(), then re-raise.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if resume_job_id is not None:
                job = tracker.resume_job(resume_job_id)
            else:
                job = tracker.start_job(
                    name=name,
                    dataset_key=dataset_key,
                    dataset_size_mb=dataset_size_mb,
                    config=config,
                    tags=list(tags) if tags is not None else None,
                )
            try:
                result = fn(job, *args, **kwargs)
                job.complete()
                return result
            except BaseException:
                job.interrupt()
                raise

        return wrapper

    return decorator
