"""Small JSON HTTP helper shared by provider drivers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from .base import ProviderError


class JsonHttpClient:
    def __init__(
        self,
        base_url: str,
        headers: dict[str, str],
        *,
        timeout: float = 60.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers)
        self.timeout = timeout
        self.opener = opener

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        ok_statuses: set[int] | None = None,
    ) -> Any:
        body = None
        headers = {"Accept": "application/json", **self.headers}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        statuses = ok_statuses or set(range(200, 300))
        try:
            with self.opener(request, timeout=self.timeout) as response:
                data = response.read()
                if response.status not in statuses:
                    raise ProviderError(
                        f"provider API {method} {path} returned HTTP {response.status}"
                    )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            lowered = detail.casefold()
            hard_stop = any(
                marker in lowered
                for marker in ("insufficient credit", "insufficient_credit", "billing disabled")
            )
            raise ProviderError(
                f"provider API {method} {path} failed with HTTP {exc.code}: {detail or exc.reason}",
                retryable=retryable and not hard_stop,
                account_hard_stop=hard_stop,
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"provider API {method} {path} failed: {exc.reason}", retryable=True
            ) from exc
        if not data:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ProviderError(f"provider API {method} {path} returned invalid JSON") from exc

