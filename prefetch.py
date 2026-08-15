"""Single-session background prefetch support.

The worker functions submitted here must be pure with respect to Streamlit: they
must not read or mutate ``st.session_state`` and must not call Streamlit APIs.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class PrefetchKey:
    """Identity of a generated item.

    Revisions deliberately contain no credentials.  Incrementing the model
    configuration revision is enough to make work created with an old key stale.
    """

    card_id: str
    config_revision: int
    batch_id: str
    prompt_version: str


class PrefetchManager:
    """Own at most one background task for one Streamlit session."""

    def __init__(self, executor: Optional[ThreadPoolExecutor] = None) -> None:
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vocab-prefetch"
        )
        self._owns_executor = executor is None
        self._key: Optional[PrefetchKey] = None
        self._future: Optional[Future[Any]] = None

    @property
    def key(self) -> Optional[PrefetchKey]:
        return self._key

    @property
    def future(self) -> Optional[Future[Any]]:
        return self._future

    def submit(
        self,
        key: PrefetchKey,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        """Submit work unless the same key is already pending or completed."""

        if self._key == key and self._future is not None:
            return self._future
        self.cancel()
        self._key = key
        self._future = self._executor.submit(function, *args, **kwargs)
        return self._future

    def matches(self, key: PrefetchKey) -> bool:
        return self._key == key and self._future is not None

    def is_ready(self, key: PrefetchKey) -> bool:
        return self.matches(key) and bool(self._future and self._future.done())

    def consume(self, key: PrefetchKey) -> Any:
        """Wait for and return a matching result, then forget the task.

        ``KeyError`` means that the caller is asking for stale or absent work.
        Exceptions raised by the worker are intentionally re-raised so the UI can
        offer an explicit retry without inventing fallback learning data.
        """

        if not self.matches(key) or self._future is None:
            raise KeyError("No matching prefetch result")
        future = self._future
        try:
            return future.result()
        finally:
            if self._future is future:
                self._key = None
                self._future = None

    def cancel(self) -> None:
        """Cancel queued work and make running work stale."""

        if self._future is not None:
            self._future.cancel()
        self._key = None
        self._future = None

    def shutdown(self) -> None:
        self.cancel()
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
