import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable


class _JsonLineFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    _DEFAULTS: Dict[str, Any] = {
        "service": "unknown",
        "event": "",
        "dataset": "",
        "exchange": "",
        "symbol": "",
        "tf": None,
        "rows": 0,
        "ms": 0,
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - docs inherited
        payload = {
            "ts": _utc_isoformat_z(),
        }

        for key, default in self._DEFAULTS.items():
            payload[key] = getattr(record, key, default)

        if not payload["event"]:
            payload["event"] = record.getMessage()

        # ensure message property is included for debugging, but avoid duplication
        if "message" in payload:
            payload.pop("message")

        payload = {key: value for key, value in payload.items() if value is not None or key == "ts"}

        return json.dumps(payload, separators=(",", ":"))


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for JSON line output."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonLineFormatter())

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _utc_isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


_ALLOWED_FIELDS: Iterable[str] = (
    "service",
    "event",
    "dataset",
    "exchange",
    "symbol",
    "tf",
    "rows",
    "ms",
)


def log_event(logger: logging.Logger, **fields: Any) -> None:
    """Log a structured event ensuring required schema."""

    allowed = set(_ALLOWED_FIELDS)
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unknown log fields: {sorted(unknown)}")

    payload = {key: value for key, value in fields.items() if value is not None}

    if "ms" in payload and not isinstance(payload["ms"], int):
        raise TypeError("Field 'ms' must be an int")

    if "rows" in payload and payload["rows"] is not None and not isinstance(payload["rows"], int):
        try:
            payload["rows"] = int(payload["rows"])
        except (TypeError, ValueError) as exc:
            raise TypeError("Field 'rows' must be an int") from exc

    message = payload.get("event", "")
    logger.info(message, extra=payload)

