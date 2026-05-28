"""
r2d1 loop utilities.

Two ways to use r2d1 in your training loop:

1. tqdm-style iterator — wraps range(), auto-times and logs each epoch:

    for epoch in EpochLoop(range(num_epochs), job=job, model=model, optimizer=opt):
        loss = train_step(...)
        epoch.loss = loss          # set metrics, flushed to D1 at end of epoch
        epoch.accuracy = acc

2. Decorator — handles job start/interrupt/complete around the whole function:

    @tracker.job(name="dit_run1", dataset_key="r2://bucket/data.tar")
    def train(job):
        for epoch in EpochLoop(range(num_epochs), job=job, model=model, optimizer=opt):
            ...
"""

import time
import functools


class EpochContext:
    """
    Yielded each iteration by EpochLoop.
    Set .loss, .accuracy (and any extra attrs) during the epoch body —
    they are flushed to D1 and used for checkpointing at the end.
    """
    def __init__(self, epoch_num):
        self.epoch     = epoch_num
        self.loss      = None
        self.accuracy  = None
        self._extras   = {}

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('epoch', 'loss', 'accuracy'):
            super().__setattr__(name, value)
        else:
            self._extras[name] = value

    def __getattr__(self, name):
        try:
            return self._extras[name]
        except KeyError:
            raise AttributeError(name)


class EpochLoop:
    """
    tqdm-style wrapper for your epoch loop.

    for epoch in EpochLoop(range(num_epochs), job=job, model=model, optimizer=opt):
        loss, grads = train_step(params, batch)   # jit'd, multi-GPU — untouched
        params = update(params, grads)
        epoch.loss     = float(loss)
        epoch.accuracy = float(acc)
        # checkpoint + D1 log happens automatically at end of each iteration

    Parameters
    ----------
    iterable        : range or list of epoch numbers
    job             : r2d1 Job handle
    model           : nn.Module or JAX param pytree (for checkpointing)
    optimizer       : optimizer state (optional)
    checkpoint_every: save checkpoint every N epochs (default: 1)
    log_every       : log to D1 every N epochs (default: 1)
    start_epoch     : skip epochs before this (used when resuming)
    async_checkpoint: upload checkpoint in background while GPU keeps training (default: True)
    """

    def __init__(self, iterable, job,
                 model=None, optimizer=None,
                 checkpoint_every=1, log_every=1,
                 start_epoch=0, async_checkpoint=True):
        self._iter             = iterable
        self._job              = job
        self._model            = model
        self._optimizer        = optimizer
        self._checkpoint_every = checkpoint_every
        self._log_every        = log_every
        self._start_epoch      = start_epoch
        self._async            = async_checkpoint
        self._epoch_start_time = None

    def __iter__(self):
        for i in self._iter:
            if i < self._start_epoch:
                continue

            ctx = EpochContext(i)
            self._epoch_start_time = time.time()

            yield ctx   # ← user's training code runs here

            duration = time.time() - self._epoch_start_time

            # Log to D1
            if i % self._log_every == 0:
                self._job.log(
                    epoch=i,
                    loss=ctx.loss,
                    accuracy=ctx.accuracy,
                    duration_sec=round(duration, 3),
                )

            # Checkpoint to R2
            if self._model is not None and i % self._checkpoint_every == 0:
                if self._async:
                    # Serialize NOW (copies weights off GPU), upload in background
                    # GPU is free to start next epoch immediately after serialize
                    self._job.save_checkpoint_async(
                        i, self._model, self._optimizer, ctx.loss
                    )
                else:
                    self._job.save_checkpoint(
                        i, self._model, self._optimizer, ctx.loss
                    )

            # Print progress
            loss_str = f"  loss={ctx.loss:.4f}" if ctx.loss is not None else ""
            acc_str  = f"  acc={ctx.accuracy:.4f}" if ctx.accuracy is not None else ""
            print(f"[r2d1] epoch {i}/{max(self._iter)}"
                  f"  {duration:.1f}s{loss_str}{acc_str}")

        # Always flush the last checkpoint before loop exits
        self._job.wait_for_checkpoint()


def job_decorator(tracker, name, dataset_key=None, dataset_size_mb=None,
                  resume_job_id=None, model_or_params=None, optimizer_state=None):
    """
    Returns a decorator that wraps a training function with job lifecycle management.
    The decorated function receives a Job as its first argument.

    Usage:
        @tracker.job(name="dit_run1", dataset_key="datasets/imagenet.tar")
        def train(job):
            for epoch in EpochLoop(range(100), job=job, model=model, optimizer=opt):
                ...

        train()   # starts or resumes, handles interrupt/complete automatically
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Resume or start
            if resume_job_id is not None:
                job, start_epoch, last_loss, params, opt = tracker.resume_job(
                    resume_job_id, model_or_params, optimizer_state
                )
                kwargs.setdefault('start_epoch', start_epoch)
                kwargs.setdefault('params', params)
                kwargs.setdefault('opt_state', opt)
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
            except (KeyboardInterrupt, Exception) as e:
                job.interrupt()
                raise

        return wrapper
    return decorator
