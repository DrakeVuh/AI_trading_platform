"""HTTP utilities for resilient API calls."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import requests
from requests import Response
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)

import time

DEFAULT_TIMEOUT = (5, 15)  # (connect, read)
MAX_RETRIES = 5
MAX_RETRY_WINDOW_SECONDS = 60
DEFAULT_USER_AGENT = "ai-trading-pipeline/1.0"


class HttpRequestError(RuntimeError):
    """Raised when an HTTP request fails."""

    def __init__(self, response: Response, detail: Optional[str] = None, *, elapsed_ms: Optional[int] = None) -> None:
        status = f"{response.status_code} {response.reason or ''}".strip()
        snippet = (response.text or "")[:200]
        message_parts = [f"Request failed: {status}"]
        if detail:
            message_parts.append(detail)
        if snippet:
            message_parts.append(f"Body: {snippet}")
        super().__init__(" - ".join(message_parts))
        self.response = response
        self.elapsed_ms = elapsed_ms
        self.body_snippet = snippet


class RetryableHttpError(HttpRequestError):
    """HTTP error that should be retried."""


def _build_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    final_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        final_headers.update(headers)
    return final_headers


def _request_once(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[Tuple[float, float]] = None,
) -> Tuple[Any, int]:
    request_headers = _build_headers(headers)
    timeout_value = timeout or DEFAULT_TIMEOUT
    start = time.perf_counter()
    try:
        response = requests.get(url, params=params, headers=request_headers, timeout=timeout_value)
    except requests.RequestException as exc:
        raise exc

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    status_code = response.status_code
    if status_code >= 400:
        error_cls = HttpRequestError
        detail = None
        if status_code == 429 or status_code >= 500:
            error_cls = RetryableHttpError
        raise error_cls(response, detail=detail, elapsed_ms=elapsed_ms)

    try:
        data = response.json()
    except ValueError as exc:
        raise HttpRequestError(response, detail="Invalid JSON response", elapsed_ms=elapsed_ms) from exc

    return data, elapsed_ms


_retry = retry(
    retry=retry_if_exception_type((requests.RequestException, RetryableHttpError)),
    wait=wait_random_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(MAX_RETRIES) | stop_after_delay(MAX_RETRY_WINDOW_SECONDS),
    reraise=True,
)


@_retry
def _request_with_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[Tuple[float, float]] = None,
) -> Tuple[Any, int]:
    return _request_once(url, params=params, headers=headers, timeout=timeout)


def request_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[Tuple[float, float]] = None,
    retries: bool = True,
) -> Tuple[Any, int]:
    """Request JSON data from an HTTP endpoint.

    Args:
        url: Target URL.
        params: Query parameters for the request.
        headers: Optional HTTP headers.
        timeout: Optional (connect, read) tuple overriding the default.
        retries: Whether to retry on transient errors.

    Raises:
        HttpRequestError: For HTTP errors or invalid JSON payloads.
        requests.RequestException: For network-related errors when retries are disabled.
        RetryError: When retries are exhausted.

    Returns:
        Tuple of parsed JSON response data and elapsed milliseconds.
    """

    try:
        if retries:
            return _request_with_retry(url, params=params, headers=headers, timeout=timeout)
        return _request_once(url, params=params, headers=headers, timeout=timeout)
    except RetryError as exc:  # pragma: no cover - defensive branch
        raise exc.last_attempt.result()

