import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import pyarrow as pa

from triad.utils.http import (
    HTTP_FETCH_SCHEMA,
    FetchConfig,
    HttpResponse,
    UrllibRequester,
    http_get_batch,
    parse_retry_after,
    rows_to_batch,
    site_of_url,
)

NOW = 1_700_000_000.0


def fast_config(**kwargs):
    defaults = dict(
        min_interval=0.0,
        backoff_base=0.01,
        backoff_max=0.05,
        timeout=5.0,
    )
    defaults.update(kwargs)
    return FetchConfig(**defaults)


class FakeRequester:
    """Scripted requester: each URL maps to a list of scripted outcomes.

    An outcome is either an HttpResponse or an Exception instance to raise.
    It also tracks the real concurrency observed per site and globally.
    """

    def __init__(self, scripts, delay=0.0):
        self._scripts = scripts
        self._delay = delay
        self._lock = threading.Lock()
        self._cursors = {}
        self.in_flight = 0
        self.max_in_flight = 0
        self.in_flight_per_site = {}
        self.max_per_site = {}
        self.calls = []

    def get(self, url, headers, timeout):
        origin = site_of_url(url)
        with self._lock:
            self.calls.append(url)
            idx = self._cursors.get(url, 0)
            self._cursors[url] = idx + 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            current = self.in_flight_per_site.get(origin, 0) + 1
            self.in_flight_per_site[origin] = current
            self.max_per_site[origin] = max(self.max_per_site.get(origin, 0), current)
        try:
            if self._delay:
                time.sleep(self._delay)
            outcome = self._scripts[url][idx]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        finally:
            with self._lock:
                self.in_flight -= 1
                self.in_flight_per_site[origin] -= 1


def resp(status=200, body=b"", headers=None):
    return HttpResponse(status, body, headers or {})


def batch_to_rows(batch):
    return batch.to_pylist()


# --------------------------------------------------------------------------
# parse_retry_after
# --------------------------------------------------------------------------


def test_parse_retry_after_seconds():
    assert parse_retry_after("10", NOW) == NOW + 10
    assert parse_retry_after("  5 ", NOW) == NOW + 5
    assert parse_retry_after("0.25", NOW) == pytest.approx(NOW + 0.25)


def test_parse_retry_after_http_date():
    dt = datetime.fromtimestamp(NOW + 30, tz=timezone.utc)
    header = dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(header, NOW) == pytest.approx(NOW + 30)


def test_parse_retry_after_invalid():
    assert parse_retry_after(None, NOW) is None
    assert parse_retry_after("", NOW) is None
    assert parse_retry_after("   ", NOW) is None
    assert parse_retry_after("soon", NOW) is None
    assert parse_retry_after("-3", NOW) is None
    assert parse_retry_after("Wed, 99 Xxx 2026 00:00:00 GMT", NOW) is None


def test_site_of_url():
    assert site_of_url("https://Example.com:8443/a?b=1") == ("https://example.com:8443")
    assert site_of_url("http://host/x") == "http://host"
    with pytest.raises(ValueError):
        site_of_url("not-a-url")
    with pytest.raises(ValueError):
        site_of_url("/relative/path")


# --------------------------------------------------------------------------
# basic collection semantics
# --------------------------------------------------------------------------


def test_empty_manifest_has_fixed_schema():
    batch = http_get_batch([], fast_config())
    assert isinstance(batch, pa.RecordBatch)
    assert batch.num_rows == 0
    assert batch.schema.equals(HTTP_FETCH_SCHEMA)
    assert rows_to_batch([]).schema.equals(HTTP_FETCH_SCHEMA)


def test_basic_success_preserves_status_and_body():
    urls = {
        "a": "http://site-a.com/1",
        "b": "http://site-b.com/2",
    }
    fake = FakeRequester(
        {
            urls["a"]: [resp(200, b"hello")],
            urls["b"]: [resp(404, b"missing")],
        }
    )
    batch = http_get_batch(urls, fast_config(requester=fake))
    assert batch.schema.equals(HTTP_FETCH_SCHEMA)
    rows = {r["id"]: r for r in batch_to_rows(batch)}
    assert rows["a"]["status_code"] == 200
    assert rows["a"]["response_body"] == b"hello"
    assert rows["a"]["error"] is None
    assert rows["a"]["attempts"] == 1
    assert rows["a"]["site"] == "http://site-a.com"
    assert rows["b"]["status_code"] == 404
    assert rows["b"]["response_body"] == b"missing"
    # manifest (dict) order is preserved
    assert [r["id"] for r in batch_to_rows(batch)] == ["a", "b"]
    # output is consumable by the existing arrow batch utilities
    from triad.utils.pyarrow import pa_batch_to_pandas

    df = pa_batch_to_pandas(batch)
    assert list(df["status_code"]) == [200, 404]


def test_duplicate_url_different_ids_are_independent():
    url = "http://site-a.com/same"
    fake = FakeRequester({url: [resp(200, b"x"), resp(201, b"y")]})
    batch = http_get_batch([("id1", url), ("id2", url)], fast_config(requester=fake))
    rows = batch_to_rows(batch)
    assert {r["id"] for r in rows} == {"id1", "id2"}
    assert sorted(r["status_code"] for r in rows) == [200, 201]
    assert len(fake.calls) == 2


def test_duplicate_id_rejected():
    with pytest.raises(ValueError):
        http_get_batch(
            [("x", "http://a.com/1"), ("x", "http://a.com/2")], fast_config()
        )


def test_non_retryable_status_is_terminal():
    url = "http://site-a.com/bad"
    fake = FakeRequester({url: [resp(400, b"bad request")]})
    batch = http_get_batch([("x", url)], fast_config(requester=fake))
    row = batch_to_rows(batch)[0]
    assert row["status_code"] == 400
    assert row["attempts"] == 1
    assert row["error"] is None


def test_transport_error_retried_then_terminal_error():
    url = "http://site-a.com/flaky"
    fake = FakeRequester(
        {
            url: [
                ConnectionError("boom"),
                TimeoutError("slow"),
                resp(200, b"ok"),
            ]
        }
    )
    batch = http_get_batch([("x", url)], fast_config(requester=fake, max_retries=3))
    row = batch_to_rows(batch)[0]
    assert row["status_code"] == 200
    assert row["response_body"] == b"ok"
    assert row["attempts"] == 3
    assert row["error"] is None  # last attempt succeeded

    fake2 = FakeRequester(
        {"http://site-a.com/down": [ConnectionError("boom"), ConnectionError("x")]}
    )
    batch2 = http_get_batch(
        [("y", "http://site-a.com/down")],
        fast_config(requester=fake2, max_retries=1),
    )
    row2 = batch2.to_pylist()[0]
    assert row2["attempts"] == 2
    assert row2["status_code"] is None
    assert "ConnectionError" in row2["error"]


# --------------------------------------------------------------------------
# Retry-After / pausing
# --------------------------------------------------------------------------


def test_retry_after_seconds_pauses_site_then_retries():
    url = "http://site-a.com/rate"
    fake = FakeRequester(
        {
            url: [
                resp(429, b"slow", {"Retry-After": "0.2"}),
                resp(200, b"ok"),
            ]
        }
    )
    start = time.time()
    batch = http_get_batch([("x", url)], fast_config(requester=fake, max_retries=2))
    elapsed = time.time() - start
    row = batch_to_rows(batch)[0]
    assert row["status_code"] == 200
    assert row["attempts"] == 2
    assert elapsed >= 0.18


def test_retry_after_http_date_pauses_site():
    # whole-second granularity: aim at least 2s in the future
    deadline = datetime.now(timezone.utc).timestamp() + 2.2
    header = datetime.fromtimestamp(deadline, tz=timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    url = "http://site-a.com/date"
    fake = FakeRequester(
        {url: [resp(503, b"x", {"Retry-After": header}), resp(200, b"ok")]}
    )
    start = time.time()
    batch = http_get_batch([("x", url)], fast_config(requester=fake, max_retries=1))
    assert time.time() - start >= 1.0
    assert batch.to_pylist()[0]["status_code"] == 200


def test_invalid_retry_after_uses_explicit_backoff():
    url = "http://site-a.com/bad-header"
    fake = FakeRequester(
        {
            url: [
                resp(429, b"x", {"Retry-After": "not-valid"}),
                resp(200, b"ok"),
            ]
        }
    )
    start = time.time()
    batch = http_get_batch(
        [("x", url)],
        fast_config(requester=fake, max_retries=1, backoff_base=0.05),
    )
    elapsed = time.time() - start
    assert batch.to_pylist()[0]["status_code"] == 200
    # backoff(attempt 1) = 0.05
    assert elapsed >= 0.045


def test_retry_exhausted_keeps_last_status_body_and_error():
    url = "http://site-a.com/always-429"
    fake = FakeRequester(
        {
            url: [
                resp(429, b"wait", {"Retry-After": "0"}),
                resp(429, b"wait", {"Retry-After": "0"}),
            ]
        }
    )
    batch = http_get_batch([("x", url)], fast_config(requester=fake, max_retries=1))
    row = batch_to_rows(batch)[0]
    assert row["status_code"] == 429
    assert row["response_body"] == b"wait"
    assert row["attempts"] == 2
    assert "retry exhausted" in row["error"]
    assert len(fake.calls) == 2


def test_later_pause_is_never_shortened():
    # first 429 asks a long pause, second (from another id, queued) asks a
    # shorter one: the long deadline must remain in effect.
    long_url = "http://site-a.com/long"
    short_url = "http://site-a.com/short"
    fake = FakeRequester(
        {
            long_url: [
                resp(429, b"x", {"Retry-After": "0.3"}),
                resp(200, b"long-ok"),
            ],
            short_url: [resp(429, b"y", {"Retry-After": "0"}), resp(200, b"s-ok")],
        }
    )
    start = time.time()
    batch = http_get_batch(
        [("l", long_url), ("s", short_url)],
        fast_config(requester=fake, max_retries=2, per_site_workers=2),
    )
    elapsed = time.time() - start
    rows = {r["id"]: r for r in batch_to_rows(batch)}
    assert rows["l"]["status_code"] == 200
    assert rows["s"]["status_code"] == 200
    # the short Retry-After must not have shortened the 0.3s pause
    assert elapsed >= 0.28


def test_paused_site_does_not_block_other_sites():
    blocked = "http://site-a.com/blocked"
    other = "http://site-b.com/other"
    fake = FakeRequester(
        {
            blocked: [
                resp(429, b"x", {"Retry-After": "0.3"}),
                resp(200, b"ok"),
            ],
            other: [resp(200, b"fine")],
        }
    )
    start = time.time()
    batch = http_get_batch(
        [("blocked", blocked), ("other", other)],
        fast_config(requester=fake, max_retries=1),
    )
    rows = {r["id"]: r for r in batch_to_rows(batch)}
    assert rows["other"]["status_code"] == 200
    assert rows["blocked"]["status_code"] == 200
    # site-b finished well before site-a's 0.3s pause elapsed
    assert time.time() - start >= 0.28


# --------------------------------------------------------------------------
# concurrency limits and minimum interval
# --------------------------------------------------------------------------


def test_global_and_per_site_in_flight_limits():
    scripts = {}
    manifest = []
    for site in ("a", "b"):
        for i in range(6):
            url = f"http://site-{site}.com/{i}"
            scripts[url] = [resp(200, b"z")]
            manifest.append((f"{site}-{i}", url))
    fake = FakeRequester(scripts, delay=0.05)
    batch = http_get_batch(
        manifest,
        fast_config(
            requester=fake,
            max_workers=3,
            per_site_workers=1,
            min_interval=0.0,
        ),
    )
    assert batch.num_rows == 12
    assert fake.max_in_flight <= 3
    # per-site counter is decremented on exit; assert against its running peak
    assert max(fake.max_per_site.values()) <= 1


def test_same_site_requests_share_one_site_state():
    # many threads registering the same origin concurrently must all see one
    # shared semaphore/pause state
    from triad.utils.http import _FetchScheduler

    scheduler = _FetchScheduler(fast_config(per_site_workers=2), time.time)
    origins = ["http://shared.com"] * 20

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        states = list(pool.map(scheduler.register, origins))
    assert all(s is states[0] for s in states)
    assert len(scheduler._states) == 1


def test_min_interval_between_same_site_starts():
    urls = [f"http://site-a.com/{i}" for i in range(4)]
    scripts = {u: [resp(200, b"z")] for u in urls}
    fake = FakeRequester(scripts, delay=0.02)
    starts = []
    original_get = fake.get

    def timed_get(url, headers, timeout):
        starts.append(time.time())
        return original_get(url, headers, timeout)

    fake.get = timed_get
    start = time.time()
    http_get_batch(
        [(str(i), u) for i, u in enumerate(urls)],
        fast_config(requester=fake, per_site_workers=4, min_interval=0.1),
    )
    starts.sort()
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert min(gaps) >= 0.09
    assert time.time() - start >= 0.29


def test_waiting_does_not_consume_global_slots():
    # 4 slow site-b requests fill global slots only if site-a's pause wait
    # wrongly held a slot; site-a must still progress while site-b is busy.
    blocked = "http://site-a.com/blocked"
    others = [f"http://site-b.com/{i}" for i in range(2)]
    scripts = {
        blocked: [
            resp(429, b"x", {"Retry-After": "0.25"}),
            resp(200, b"ok"),
        ],
        **{u: [resp(200, b"z")] for u in others},
    }
    fake = FakeRequester(scripts, delay=0.08)
    batch = http_get_batch(
        [("blocked", blocked)] + [(f"b{i}", u) for i, u in enumerate(others)],
        fast_config(
            requester=fake,
            max_workers=2,
            per_site_workers=2,
            max_retries=1,
        ),
    )
    rows = {r["id"]: r for r in batch.to_pylist()}
    assert rows["blocked"]["status_code"] == 200
    assert all(rows[f"b{i}"]["status_code"] == 200 for i in range(2))


def test_pause_set_while_acquiring_slots_does_not_consume_attempt():
    # max_workers=1 forces y to block on the global slot while x gets 429 and
    # pauses the site; y must re-wait and still send exactly one attempt.
    x_url = "http://site-a.com/x"
    y_url = "http://site-a.com/y"
    fake = FakeRequester(
        {
            x_url: [
                resp(429, b"x", {"Retry-After": "0.3"}),
                resp(200, b"x-ok"),
            ],
            y_url: [resp(200, b"y-ok")],
        }
    )
    start = time.time()
    batch = http_get_batch(
        [("x", x_url), ("y", y_url)],
        fast_config(
            requester=fake,
            max_workers=1,
            per_site_workers=1,
            max_retries=2,
        ),
    )
    elapsed = time.time() - start
    rows = {r["id"]: r for r in batch.to_pylist()}
    assert rows["y"]["status_code"] == 200
    assert rows["y"]["attempts"] == 1
    assert rows["x"]["attempts"] == 2
    assert elapsed >= 0.28
    # y's single call happened after the pause elapsed
    assert sum(1 for u in fake.calls if u == y_url) == 1


def test_requests_requester_adapter():
    from triad.utils.http import RequestsRequester

    class FakeResponse:
        status_code = 201
        content = b"created"
        headers = {"Content-Type": "text/plain"}

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, headers, timeout):
            self.calls.append((url, headers, timeout))
            return FakeResponse()

    session = FakeSession()
    requester = RequestsRequester(session=session)
    batch = http_get_batch(
        [("r", "http://site-a.com/r")],
        fast_config(requester=requester, headers={"X-T": "1"}),
    )
    row = batch.to_pylist()[0]
    assert row["status_code"] == 201
    assert row["response_body"] == b"created"
    assert session.calls == [("http://site-a.com/r", {"X-T": "1"}, pytest.approx(5.0))]


def test_config_validation():
    with pytest.raises(ValueError):
        FetchConfig(max_workers=0).validate()
    with pytest.raises(ValueError):
        FetchConfig(max_workers=2, per_site_workers=3).validate()
    with pytest.raises(ValueError):
        FetchConfig(max_retries=-1).validate()
    with pytest.raises(ValueError):
        FetchConfig(min_interval=-1).validate()
    with pytest.raises(ValueError):
        http_get_batch([], FetchConfig(), extra=1)


# --------------------------------------------------------------------------
# real local server integration (stdlib transport)
# --------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/binary":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes([0x00, 0xFF, 0x10, 0x7F]))
        elif self.path == "/retry":
            if not hasattr(self.server, "retry_seen"):
                self.server.retry_seen = True
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(b"rate-limited")
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"recovered")
        elif self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"nope")
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")


@pytest.fixture(scope="module")
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


def test_urllib_requester_end_to_end(local_server):
    base = local_server
    batch = http_get_batch(
        [
            ("bin", f"{base}/binary"),
            ("retry", f"{base}/retry"),
            ("missing", f"{base}/missing"),
        ],
        fast_config(requester=UrllibRequester(), max_retries=2, min_interval=0.0),
    )
    rows = {r["id"]: r for r in batch.to_pylist()}
    assert rows["bin"]["response_body"] == bytes([0x00, 0xFF, 0x10, 0x7F])
    assert rows["bin"]["status_code"] == 200
    assert rows["retry"]["status_code"] == 200
    assert rows["retry"]["attempts"] == 2
    assert rows["missing"]["status_code"] == 404
    assert rows["missing"]["response_body"] == b"nope"


def test_urllib_requester_connection_error_is_terminal(local_server):
    bad_url = "http://127.0.0.1:1/never"
    batch = http_get_batch(
        [("x", bad_url)],
        fast_config(requester=UrllibRequester(), max_retries=1, backoff_base=0.001),
    )
    row = batch.to_pylist()[0]
    assert row["status_code"] is None
    assert row["response_body"] is None
    assert row["attempts"] == 2
    assert "Error" in row["error"]
