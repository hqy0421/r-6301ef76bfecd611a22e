"""Multi-site concurrent HTTP GET collection.

The module turns a manifest of ``(request_id, url)`` pairs into a single
:class:`pyarrow.RecordBatch` with a declared, fixed schema
(:data:`HTTP_FETCH_SCHEMA`). Every request id produces exactly one terminal
row: either the real response status code and raw response body, or the final
error after retries are exhausted.

Concurrency is bounded by both a global in-flight limit and a per-site
in-flight limit, where a site is the URL origin (``scheme://host[:port]``).
Waiting (site pauses caused by ``Retry-After``, minimum start intervals and
backoff) never occupies an in-flight slot, so requests to other ready sites
keep running.
"""

import email.utils
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import BoundedSemaphore, Condition, Lock
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

import pyarrow as pa

from triad.utils.assertion import assert_or_throw

#: The declared schema of every batch returned by :func:`http_get_batch`.
HTTP_FETCH_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("site", pa.string(), nullable=False),
        pa.field("status_code", pa.int32(), nullable=True),
        pa.field("response_body", pa.binary(), nullable=True),
        pa.field("error", pa.string(), nullable=True),
        pa.field("attempts", pa.int32(), nullable=False),
    ]
)

#: HTTP status codes that honor the ``Retry-After`` header.
_RETRYABLE_STATUSES = (429, 503)


@dataclass(frozen=True)
class HttpResponse:
    """A normalized HTTP response returned by a :class:`Requester`.

    :param status_code: the real HTTP status code
    :param content: the raw response body bytes
    :param headers: the response headers (case-insensitive access expected)
    """

    status_code: int
    content: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


class Requester:  # pragma: no cover
    """Protocol of a GET transport used by :func:`http_get_batch`.

    Implementations must return an :class:`HttpResponse` for every completed
    HTTP exchange (including 4xx/5xx statuses) and raise an exception only for
    transport level failures (connection errors, timeouts, ...).
    """

    def get(
        self, url: str, headers: Mapping[str, str], timeout: float
    ) -> HttpResponse:  # pragma: no cover
        raise NotImplementedError


class UrllibRequester:
    """The default requester, built on the standard library only.

    ``HTTPError`` responses (including 429/503) are returned as
    :class:`HttpResponse` instead of being raised, so their status code, body
    and ``Retry-After`` header are visible to the retry logic.
    """

    def get(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return HttpResponse(
                    response.status,
                    response.read(),
                    {k: v for k, v in response.headers.items()},
                )
        except urllib.error.HTTPError as e:
            return HttpResponse(e.code, e.read(), {k: v for k, v in e.headers.items()})


class RequestsRequester:
    """A requester backed by the third party ``requests`` package.

    :param session: an optional ``requests.Session``, a new one is created
        when not provided

    :raises ImportError: if ``requests`` is not installed
    """

    def __init__(self, session: Any = None):
        try:
            import requests  # pylint: disable=import-outside-toplevel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "RequestsRequester requires the 'requests' package, "
                "install it or use the default UrllibRequester"
            ) from e
        self._session = session if session is not None else requests.Session()

    def get(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        response = self._session.get(url, headers=dict(headers), timeout=timeout)
        return HttpResponse(
            response.status_code, response.content, dict(response.headers)
        )


@dataclass
class FetchConfig:
    """Configuration for :func:`http_get_batch`.

    :param max_workers: global maximum number of requests truly in flight
    :param per_site_workers: maximum number of requests truly in flight per
        site (URL origin)
    :param min_interval: minimum interval in seconds between two request
        starts on the same site
    :param max_retries: maximum number of retries after the first attempt,
        so each id is attempted at most ``max_retries + 1`` times
    :param backoff_base: base seconds of the explicit exponential backoff
    :param backoff_max: upper bound in seconds of the backoff
    :param timeout: per request timeout in seconds passed to the requester
    :param headers: extra request headers sent with every GET
    :param requester: the transport, defaults to :class:`UrllibRequester`
    """

    max_workers: int = 8
    per_site_workers: int = 2
    min_interval: float = 0.2
    max_retries: int = 3
    backoff_base: float = 0.5
    backoff_max: float = 30.0
    timeout: float = 30.0
    headers: Optional[Dict[str, str]] = None
    requester: Optional[Requester] = None

    def validate(self) -> None:
        assert_or_throw(self.max_workers >= 1, ValueError("max_workers must be >= 1"))
        assert_or_throw(
            1 <= self.per_site_workers <= self.max_workers,
            ValueError(
                "per_site_workers must be between 1 and max_workers "
                f"({self.max_workers})"
            ),
        )
        assert_or_throw(self.min_interval >= 0, ValueError("min_interval must be >= 0"))
        assert_or_throw(self.max_retries >= 0, ValueError("max_retries must be >= 0"))
        assert_or_throw(self.backoff_base > 0, ValueError("backoff_base must be > 0"))
        assert_or_throw(self.backoff_max > 0, ValueError("backoff_max must be > 0"))
        assert_or_throw(self.timeout > 0, ValueError("timeout must be > 0"))


def site_of_url(url: str) -> str:
    """Extract the site (URL origin) of ``url``.

    :param url: an absolute http/https URL
    :raises ValueError: if the URL has no scheme or host
    :return: the normalized origin, e.g. ``https://example.com:8443``
    """
    parsed = urllib.parse.urlsplit(url)
    assert_or_throw(
        parsed.scheme != "" and parsed.netloc != "",
        ValueError(f"{url} is not an absolute URL with a host"),
    )
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def parse_retry_after(value: Optional[str], now: float) -> Optional[float]:
    """Parse an HTTP ``Retry-After`` header into an absolute pause deadline.

    Both header forms defined by RFC 9110 are supported:

    * a non-negative delay in seconds (integer per RFC, float also accepted)
    * an HTTP date, e.g. ``Wed, 21 Oct 2026 07:28:00 GMT``

    :param value: the raw header value
    :param now: the current absolute time (epoch seconds, as returned by
        :func:`time.time`)
    :return: the absolute epoch second until which the site should pause, or
        ``None`` if the value is missing or invalid
    """
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    if text.isdigit():
        return now + int(text)
    try:
        delay = float(text)
    except ValueError:
        delay = None
    if delay is not None:
        return now + delay if delay >= 0 else None
    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    return parsed.timestamp()


@dataclass
class _SiteState:
    """Scheduling state of one site."""

    origin: str
    semaphore: BoundedSemaphore
    cond: Condition = field(default_factory=Condition)
    #: absolute epoch second until which no request may start on this site
    paused_until: float = 0.0
    #: start time of the most recent request on this site
    last_started: float = 0.0


class _FetchScheduler:
    def __init__(self, config: FetchConfig, time_func: Callable[[], float]):
        self._config = config
        self._time = time_func
        self._requester = config.requester or UrllibRequester()
        self._headers = config.headers or {}
        self._global_sem = BoundedSemaphore(config.max_workers)
        self._register_lock = Lock()
        self._states: Dict[str, _SiteState] = {}

    def register(self, origin: str) -> _SiteState:
        with self._register_lock:
            state = self._states.get(origin)
            if state is None:
                state = _SiteState(
                    origin=origin,
                    semaphore=BoundedSemaphore(self._config.per_site_workers),
                )
                self._states[origin] = state
            return state

    def run(self, manifest: Iterable[Tuple[str, str]]) -> Dict[str, Dict[str, Any]]:
        items = list(manifest)
        results: Dict[str, Dict[str, Any]] = {}
        if len(items) == 0:
            return results
        with ThreadPoolExecutor(max_workers=self._config.max_workers) as pool:
            futures = [
                pool.submit(self._run_one, request_id, url) for request_id, url in items
            ]
            for future in futures:
                request_id, row = future.result()
                results[request_id] = row
        return results

    def _run_one(self, request_id: str, url: str) -> Tuple[str, Dict[str, Any]]:
        origin = site_of_url(url)
        state = self.register(origin)
        attempts = 0
        while attempts <= self._config.max_retries:
            attempts += 1
            # Pauses and minimum-interval waits never hold in-flight slots.
            self._wait_ready(state)
            state.semaphore.acquire()
            try:
                self._global_sem.acquire()
                try:
                    # A pause set while acquiring slots is honored without
                    # consuming an attempt; release both slots and re-wait.
                    with state.cond:
                        if state.paused_until > self._time():
                            attempts -= 1
                            continue
                        state.last_started = self._time()
                    try:
                        response = self._requester.get(
                            url, self._headers, self._config.timeout
                        )
                    except Exception as e:  # transport level failure
                        error = f"{type(e).__name__}: {e}"
                        if attempts > self._config.max_retries:
                            return request_id, self._error_row(
                                request_id, url, origin, attempts, error
                            )
                        self._sleep(self._backoff(attempts))
                        continue
                finally:
                    self._global_sem.release()
            finally:
                state.semaphore.release()
            if response.status_code in _RETRYABLE_STATUSES:
                self._apply_retry_after(state, response, attempts)
                if attempts > self._config.max_retries:
                    return request_id, self._response_row(
                        request_id,
                        url,
                        origin,
                        attempts,
                        response,
                        error=(
                            f"retry exhausted after {attempts} attempts, "
                            f"last status {response.status_code}"
                        ),
                    )
                continue
            return request_id, self._response_row(
                request_id, url, origin, attempts, response
            )
        # unreachable, the loop above always returns
        raise AssertionError("unreachable")  # pragma: no cover

    def _wait_ready(self, state: _SiteState) -> None:
        with state.cond:
            while True:
                delay = state.paused_until - self._time()
                if delay <= 0:
                    break
                state.cond.wait(delay)
            if self._config.min_interval > 0:
                while True:
                    delay = (
                        state.last_started + self._config.min_interval - (self._time())
                    )
                    if delay <= 0:
                        break
                    state.cond.wait(delay)

    def _apply_retry_after(
        self, state: _SiteState, response: HttpResponse, attempts: int
    ) -> None:
        deadline = parse_retry_after(response.headers.get("Retry-After"), self._time())
        if deadline is None:
            # Missing or invalid header: explicit, deterministic backoff.
            deadline = self._time() + self._backoff(attempts)
        with state.cond:
            # Never shorten an already later pause.
            if deadline > state.paused_until:
                state.paused_until = deadline
            state.cond.notify_all()

    def _backoff(self, attempts: int) -> float:
        return min(
            self._config.backoff_base * (2 ** (attempts - 1)),
            self._config.backoff_max,
        )

    def _sleep(self, seconds: float) -> None:
        # No in-flight slot is held while sleeping.
        if seconds > 0:
            time.sleep(seconds)

    @staticmethod
    def _response_row(
        request_id: str,
        url: str,
        origin: str,
        attempts: int,
        response: HttpResponse,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "id": request_id,
            "url": url,
            "site": origin,
            "status_code": response.status_code,
            "response_body": response.content,
            "error": error,
            "attempts": attempts,
        }

    @staticmethod
    def _error_row(
        request_id: str, url: str, origin: str, attempts: int, error: str
    ) -> Dict[str, Any]:
        return {
            "id": request_id,
            "url": url,
            "site": origin,
            "status_code": None,
            "response_body": None,
            "error": error,
            "attempts": attempts,
        }


def _normalize_manifest(
    items: Union[Mapping[str, str], Iterable[Tuple[str, str]]],
) -> Iterable[Tuple[str, str]]:
    if isinstance(items, Mapping):
        pairs = list(items.items())
    else:
        pairs = [tuple(item) for item in items]
    seen: set = set()
    for pair in pairs:
        assert_or_throw(
            len(pair) == 2, ValueError("each manifest item must be (id, url)")
        )
        request_id, url = pair
        assert_or_throw(
            isinstance(request_id, str) and isinstance(url, str),
            TypeError("manifest id and url must be strings"),
        )
        assert_or_throw(
            request_id not in seen,
            ValueError(f"duplicated request id in manifest: {request_id}"),
        )
        seen.add(request_id)
        # validate the URL early and register its site origin
        site_of_url(url)
        yield request_id, url


def rows_to_batch(rows: Iterable[Dict[str, Any]]) -> pa.RecordBatch:
    """Build a :class:`pyarrow.RecordBatch` from result rows using
    :data:`HTTP_FETCH_SCHEMA`.

    An empty iterable produces a zero-row batch with the same declared schema.

    :param rows: result rows as produced by the fetch scheduler
    :return: a record batch matching :data:`HTTP_FETCH_SCHEMA`
    """
    data: Dict[str, Any] = {name: [] for name in HTTP_FETCH_SCHEMA.names}
    for row in rows:
        for name in HTTP_FETCH_SCHEMA.names:
            data[name].append(row[name])
    return pa.RecordBatch.from_pydict(data, schema=HTTP_FETCH_SCHEMA)


def http_get_batch(
    items: Union[Mapping[str, str], Iterable[Tuple[str, str]]],
    config: Optional[FetchConfig] = None,
    **kwargs: Any,
) -> pa.RecordBatch:
    """Concurrently GET a manifest of ``(id, url)`` pairs into an Arrow batch.

    Each request id is collected exactly once and yields exactly one terminal
    row. A completed HTTP exchange (any status code, including 429/503) keeps
    the real status code and raw body; a transport failure or a retry that is
    exhausted yields an error string instead. Different ids pointing at the
    same URL are collected independently and each keep their own row.

    In-flight requests are bounded both globally (``max_workers``) and per
    site (``per_site_workers``). Starts on the same site are additionally
    spaced by ``min_interval``. A 429/503 response pauses the whole site
    honoring its ``Retry-After`` header (seconds or HTTP date); an invalid or
    missing header falls back to an explicit exponential backoff, and an
    already later pause is never shortened. Pauses, interval waits and
    backoffs do not occupy in-flight slots, so other ready sites keep
    collecting.

    :param items: a mapping of ``id -> url`` or an iterable of
        ``(id, url)`` tuples; ids must be unique
    :param config: the :class:`FetchConfig`, a default one is created when
        omitted
    :param kwargs: overrides for :class:`FetchConfig` fields, used only when
        ``config`` is not provided
    :raises ValueError: for invalid configuration or duplicated ids
    :raises TypeError: if ids or urls are not strings
    :return: a :class:`pyarrow.RecordBatch` matching
        :data:`HTTP_FETCH_SCHEMA`, in manifest order; an empty manifest
        returns a zero-row batch with the same fixed schema
    """
    if config is None:
        config = FetchConfig(**kwargs)
    else:
        assert_or_throw(
            len(kwargs) == 0,
            ValueError("kwargs can't be used together with an explicit config"),
        )
    config.validate()
    manifest = list(_normalize_manifest(items))
    scheduler = _FetchScheduler(config, time_func=time.time)
    results = scheduler.run(manifest)
    ordered_rows = [results[request_id] for request_id, _ in manifest]
    return rows_to_batch(ordered_rows)
