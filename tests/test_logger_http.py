import io
import json
from pathlib import Path
import sys

import pytest
import responses

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.logger import get_logger, log_event
from src.utils.http import request_json, HttpRequestError


def test_log_event_emits_json_with_utc_z_suffix():
    logger_name = "tests.logger.json"
    logger = get_logger(logger_name)

    handler = logger.handlers[0]
    stream = io.StringIO()
    handler.stream = stream

    log_event(
        logger,
        service="collector",
        event="download",
        dataset="ohlcv",
        exchange="binance",
        symbol="BTCUSDT",
        tf=None,
        rows=123,
        ms=456,
    )

    handler.flush()
    log_line = stream.getvalue().strip()
    payload = json.loads(log_line)

    assert payload["ts"].endswith("Z")
    assert payload["service"] == "collector"
    assert payload["event"] == "download"
    assert payload["ms"] == 456
    assert "tf" not in payload


@responses.activate
def test_request_json_retries_on_server_errors():
    url = "https://api.example.com/data"
    responses.add(responses.GET, url, status=500, json={"error": "server"})
    responses.add(responses.GET, url, status=500, json={"error": "server"})
    responses.add(responses.GET, url, status=200, json={"ok": True})

    data, elapsed_ms = request_json(url)

    assert data == {"ok": True}
    assert isinstance(elapsed_ms, int)
    assert elapsed_ms >= 0
    assert len(responses.calls) == 3


@responses.activate
def test_request_json_retries_on_rate_limit():
    url = "https://api.example.com/ratelimit"
    responses.add(responses.GET, url, status=429, json={"error": "too many"})
    responses.add(responses.GET, url, status=200, json={"ok": True})

    data, _ = request_json(url)

    assert data["ok"] is True
    assert len(responses.calls) == 2


@responses.activate
def test_request_json_does_not_retry_on_not_found():
    url = "https://api.example.com/missing"
    responses.add(responses.GET, url, status=404, json={"error": "missing"})

    with pytest.raises(HttpRequestError):
        request_json(url)

    assert len(responses.calls) == 1


@responses.activate
def test_request_json_raises_on_non_json_response():
    url = "https://api.example.com/not-json"
    responses.add(responses.GET, url, status=200, body="not-json", content_type="text/plain")

    with pytest.raises(HttpRequestError) as exc:
        request_json(url)

    assert "Invalid JSON" in str(exc.value)
    assert len(exc.value.body_snippet) <= 200
    assert "not-json" in exc.value.body_snippet

