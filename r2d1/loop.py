"""
r2d1 loop utilities — tqdm-style EpochLoop and @tracker.job decorator.
"""
import time
import functools


class EpochContext:
    """
    Yielded each iteration by EpochLoop.
    Set .loss, .accuracy, .files during the epoch body.
    All are flushed to D1 / R2 automatically at end of iteration.
    """
    def __init__(self, epoch_num):
        self.epoch    = epoch_num
        self.loss     = None
        self.accuracy = None
        self.files    = None   # dict of filename → bytes, shipped to R2


class EpochLoop:
    """
    tqdm-style wrapper for your epoch loop. Auto-times, logs to D1, ships
    checkpoint files to R2.

    for epoch in EpochLoop(range(400), job=job):
        loss = train_step(...)         # jit'd / multi-GPU — completely untouched
        epoch.loss  = float(loss)
        epoch.files = model.checkpoint()   # your model packages its own files

    Parameters
    ----------
    iterable         : range or list of epoch numbers
    job              : r2d1 Job handle
    checkpoint_every : upload files every N epochs (default: 1)
    log_every        : log to D1 every N epochs (default: 1)
    start_epoch      : skip epochs before this number (for resuming)
    async_checkpoint : upload in background while GPU trains (default: True)
    """

    def __init__(self, iterable, job,
                 checkpoint_every=1, log_every=1,
                 start_epoch=0, async_checkpoint=True):
        self._iter             = iterable
        self._job              = job
        self._checkpoint_every = checkpoint_every
        self._log_every        = log_every
        self._start_epoch      = start_epoch
        self._async            = async_checkpoint

    def __iter__(self):
        epochs = list(self._iter)
        total  = max(epochs) if epochs else 0

        for i in epochs:
            if i < self._start_epoch:
                continue

            ctx = EpochContext(i)
            t0  = time.time()

            yield ctx   # ← your training code runs here

            duration = time.time() - t0

            # Log metrics to D1
            if i % self._log_every == 0:
                self._job.log(
                    epoch=i,
                    loss=ctx.loss,
                    accuracy=ctx.accuracy,
                    duration_sec=round(duration, 3),
                )

            # Ship checkpoint files to R2
            if ctx.files and i % self._checkpoint_every == 0:
                if self._async:
                    self._job.save_async(i, ctx.files, loss=ctx.loss)
                else:
                    self._job.save(i, ctx.files, loss=ctx.loss)

            # Progress line
            loss_str = f"  loss={ctx.loss:.4f}" if ctx.loss is not None else ""
            acc_str  = f"  acc={ctx.accuracy:.4f}" if ctx.accuracy is not None else ""
            print(f"[r2d1] epoch {i}/{total}  {duration:.1f}s{loss_str}{acc_str}")

        # Always flush last async upload before loop exits
        self._job.wait()


def job_decorator(tracker, name, dataset_key=None,
                  dataset_size_mb=None, resume_job_id=None):
    """
    Returns a decorator that wraps a training function with job lifecycle.
    The decorated function receives a Job as its first argument.
    Calls job.interrupt() on any exception, job.complete() on clean exit.
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
                )
            try:
                result = fn(job, *args, **kwargs)
                job.complete()
                return result
            except (KeyboardInterrupt, Exception):
                job.interrupt()
                raise

        return wrapper
    return decorator
