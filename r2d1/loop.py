"""tqdm-style r2d1 loop wrapper."""
from __future__ import annotations

import time
from typing import Any, Iterable, Mapping, Optional


class Epoch:
    """Object yielded by r2d1(...)."""

    def __init__(self, i: int, *, should_log: bool, should_checkpoint: bool):
        self.i = i
        self.epoch = i
        self.should_log = should_log
        self.should_checkpoint = should_checkpoint
        self._metrics: dict[str, Any] | None = None
        self._files: Mapping[str, Any] | None = None

    def d1(self, metrics: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if metrics:
            data.update(dict(metrics))
        data.update(kwargs)
        self._metrics = data
        return data

    log = d1

    def r2(self, files: Mapping[str, Any]) -> Mapping[str, Any]:
        self._files = files
        return files

    checkpoint = r2


class r2d1:
    """Wrap an iterable like tqdm and attach epoch.d1/epoch.r2 to a Job."""

    def __init__(
        self,
        iterable: Iterable[int],
        *,
        job,
        log_every: int = 1,
        checkpoint_every: int = 1,
        keep_last: int = 2,
        start_epoch: int = 0,
        async_checkpoint: bool = True,
        print_progress: bool = True,
    ):
        self.iterable = iterable
        self.job = job
        self.log_every = max(1, int(log_every))
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.keep_last = max(1, int(keep_last))
        self.start_epoch = int(start_epoch)
        self.async_checkpoint = async_checkpoint
        self.print_progress = print_progress

    def __iter__(self):
        for i in self.iterable:
            if i < self.start_epoch:
                continue
            t0 = time.time()
            epoch = Epoch(
                int(i),
                should_log=(int(i) % self.log_every == 0),
                should_checkpoint=(int(i) % self.checkpoint_every == 0),
            )
            yield epoch
            duration = time.time() - t0

            if epoch.should_log:
                self.job.d1(epoch=epoch.i, metrics=epoch._metrics or {}, duration_sec=round(duration, 3))

            if epoch._files and epoch.should_checkpoint:
                if self.async_checkpoint:
                    self.job.r2_async(epoch=epoch.i, files=epoch._files, metrics=epoch._metrics, keep_last=self.keep_last, checkpoint_index=epoch.i // self.checkpoint_every)
                else:
                    self.job.r2(epoch=epoch.i, files=epoch._files, metrics=epoch._metrics, keep_last=self.keep_last, checkpoint_index=epoch.i // self.checkpoint_every)

            if self.print_progress:
                loss = (epoch._metrics or {}).get("loss")
                loss_s = f" loss={loss:.4f}" if isinstance(loss, (float, int)) else ""
                print(f"[r2d1] epoch {epoch.i} {duration:.1f}s{loss_s}")

        self.job.wait()


# Backward-compatible alias for older examples.
EpochLoop = r2d1
