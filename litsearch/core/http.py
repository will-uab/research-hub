"""Small requests wrapper: shared session, rate limiting, retry on 429/5xx."""
from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

USER_AGENT = "litsearch/1.0 (personal literature digest)"
_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        delta = time.monotonic() - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


def request(method: str, url: str, limiter: RateLimiter | None = None, retries: int = 4,
            timeout: int = 60, **kw) -> requests.Response:
    delay = 2.0
    for attempt in range(retries + 1):
        if limiter:
            limiter.wait()
        try:
            r = _session.request(method, url, timeout=timeout, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} from {url}", response=r)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            retryable = status is None or status in (429, 500, 502, 503, 504)
            if attempt == retries or not retryable:
                raise
            log.warning("%s (attempt %d/%d), retrying in %.0fs", e, attempt + 1, retries + 1, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")
