"""Dynamic batching: requests that arrive while the model is busy share its next pass.

On a CPU, one forward pass over 16 texts costs far less than 16 passes over one
text each (the benchmark measured 2.3 times the throughput). Clients mostly send
one text at a time, so the server does the batching:

* A request on an idle server starts a forward pass at once, alone: no added
  latency when traffic is light.
* Requests that arrive while a pass is running wait in a queue. When the model is
  free, the next pass takes everything waiting, in arrival order, up to
  ``max_texts`` texts. So batches form only under load, exactly when they help.
* ``max_wait`` optionally holds a pass back for a few milliseconds to collect more
  texts; 0 (the default) never waits.

``workers`` passes may run at once (MAX_CONCURRENT_INFERENCES), each in its own
thread, so the event loop is never blocked by the model.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .predictors import Prediction

logger = logging.getLogger("review_classifier")


@dataclass(frozen=True)
class BatchResult:
    """One request's share of a forward pass."""

    predictions: list[Prediction]
    seconds: float  # duration of the whole pass this request was part of
    queued: float  # seconds the request waited before its pass started
    batch_texts: int  # texts in that pass, from all requests


@dataclass
class _Job:
    texts: list[str]
    future: asyncio.Future[BatchResult]
    enqueued: float


class DynamicBatcher:
    def __init__(
        self,
        predict: Callable[[list[str]], Sequence[Prediction]],
        *,
        max_texts: int = 16,
        max_wait: float = 0.0,
        workers: int = 1,
        on_pass: Callable[[int, float], None] | None = None,
    ) -> None:
        if max_texts < 1 or workers < 1 or max_wait < 0:
            raise ValueError("max_texts and workers must be positive, max_wait not negative")
        self._predict = predict
        self._max_texts = max_texts
        self._max_wait = max_wait
        self._workers = workers
        self._on_pass = on_pass
        self._pending: deque[_Job] = deque()
        self._in_pass: dict[int, _Job] = {}
        self._wake: asyncio.Event | None = None
        self._tasks: list[asyncio.Task] = []
        self._executor: ThreadPoolExecutor | None = None

    @property
    def queued_texts(self) -> int:
        return sum(len(job.texts) for job in self._pending)

    async def start(self) -> None:
        if self._tasks:
            return
        self._wake = asyncio.Event()
        self._executor = ThreadPoolExecutor(self._workers, thread_name_prefix="inference")
        self._tasks = [asyncio.create_task(self._work()) for _ in range(self._workers)]

    async def stop(self) -> None:
        # Requests in a pass that is cut short, and those still queued, get an error
        # rather than waiting forever. Noted before the workers are cancelled, since
        # a cancelled worker forgets its pass.
        stranded = list(self._in_pass.values()) + list(self._pending)
        self._pending.clear()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        self._in_pass.clear()
        stranded += list(self._pending)  # arrivals while the workers were stopping
        self._pending.clear()
        for job in stranded:
            if not job.future.done():
                job.future.set_exception(RuntimeError("The service is shutting down."))
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    async def submit(self, texts: Sequence[str]) -> BatchResult:
        """Classify ``texts`` as part of the next forward pass."""
        if not self._tasks or self._wake is None:
            raise RuntimeError("The batcher is not running.")
        future: asyncio.Future[BatchResult] = asyncio.get_running_loop().create_future()
        self._pending.append(_Job(list(texts), future, time.perf_counter()))
        self._wake.set()
        return await future

    # -- workers -------------------------------------------------------------------

    def _take(self) -> list[_Job]:
        """The next batch: waiting jobs in arrival order, up to max_texts texts.

        A single job larger than max_texts still runs, alone. Jobs whose caller
        has gone away (cancelled futures) are dropped.
        """
        jobs: list[_Job] = []
        count = 0
        while self._pending:
            job = self._pending[0]
            if job.future.done():
                self._pending.popleft()
                continue
            if jobs and count + len(job.texts) > self._max_texts:
                break
            jobs.append(self._pending.popleft())
            count += len(job.texts)
        return jobs

    async def _work(self) -> None:
        assert self._wake is not None
        while True:
            while not self._pending:
                self._wake.clear()
                await self._wake.wait()
            if self._max_wait and self.queued_texts < self._max_texts:
                # Wait at most max_wait from the oldest request's arrival, so a
                # request that already waited through a pass is not held back again.
                oldest = self._pending[0].enqueued
                delay = oldest + self._max_wait - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            jobs = self._take()
            if not jobs:
                continue
            for job in jobs:
                self._in_pass[id(job)] = job
            try:
                await self._run_pass(jobs)
            except Exception:  # never let a worker die: later requests would hang
                logger.exception("dynamic batching: a pass failed unexpectedly")
                self._fail(jobs, RuntimeError("Inference failed."))
            finally:
                for job in jobs:
                    self._in_pass.pop(id(job), None)

    async def _run_pass(self, jobs: list[_Job]) -> None:
        texts = [text for job in jobs for text in job.texts]
        started = time.perf_counter()
        try:
            predictions = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._predict, texts
            )
            if len(predictions) != len(texts):
                raise RuntimeError(
                    f"Predictor returned {len(predictions)} results for {len(texts)}"
                )
        except Exception as exc:
            self._fail(jobs, exc)
            return
        seconds = time.perf_counter() - started
        if self._on_pass is not None:
            try:
                self._on_pass(len(texts), seconds)
            except Exception:
                logger.exception("dynamic batching: the on_pass callback failed")
        position = 0
        for job in jobs:
            share = list(predictions[position : position + len(job.texts)])
            position += len(job.texts)
            if not job.future.done():
                job.future.set_result(
                    BatchResult(share, seconds, started - job.enqueued, len(texts))
                )

    @staticmethod
    def _fail(jobs: list[_Job], exc: BaseException) -> None:
        """Fail every job of a pass, each with its own exception (tracebacks stay separate)."""
        for job in jobs:
            if not job.future.done():
                error = RuntimeError("Inference failed.")
                error.__cause__ = exc
                job.future.set_exception(error)
