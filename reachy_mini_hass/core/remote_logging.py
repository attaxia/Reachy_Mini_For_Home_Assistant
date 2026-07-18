"""Optional remote log streaming via OpenTelemetry (OTLP/HTTP).

Reads the standard OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_HEADERS
env vars, falling back to a `.env` file at the repo root (never committed -
see `.env.example`). These are the exact variable names Grafana Cloud (and
most other OTLP-compatible backends) hand you on their "Connect using
OpenTelemetry" setup page, so their snippet can be pasted into `.env`
unchanged. If the endpoint is missing, remote logging is skipped entirely
and the app behaves exactly as it does without this module.

All network I/O happens on a dedicated background thread so a slow or
unreachable connection can never stall a caller - including log calls made
from the 100Hz motion control loop.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from urllib.parse import unquote

import requests
from dotenv import load_dotenv

from .util import get_mac

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

_QUEUE_MAXSIZE = 2000
_FLUSH_INTERVAL_S = 1.0
_FLUSH_BATCH_SIZE = 100
_REQUEST_TIMEOUT_S = 5.0

_SERVICE_NAME = "reachy_mini_hass"

# Maps Python logging levels to OTLP severity numbers/text (see the OTel
# logs data model spec: DEBUG=5, INFO=9, WARN=13, ERROR=17, FATAL=21).
_SEVERITY = (
    (logging.CRITICAL, 21, "FATAL"),
    (logging.ERROR, 17, "ERROR"),
    (logging.WARNING, 13, "WARN"),
    (logging.INFO, 9, "INFO"),
    (logging.DEBUG, 5, "DEBUG"),
)

# Diagnostics about the handler itself must never re-enter it (that would
# recurse), so this logger never propagates to the root logger.
_internal_logger = logging.getLogger(f"{__name__}._internal")
_internal_logger.propagate = False
if not _internal_logger.handlers:
    _internal_logger.addHandler(logging.StreamHandler())


def _severity_for(levelno: int) -> tuple[int, str]:
    for threshold, number, text in _SEVERITY:
        if levelno >= threshold:
            return number, text
    return 1, "TRACE"


def _parse_otlp_headers(raw: str) -> dict[str, str]:
    """Parse the OTEL_EXPORTER_OTLP_HEADERS format: "k1=v1,k2=v2", %-encoded."""
    headers: dict[str, str] = {}
    for raw_pair in raw.split(","):
        pair = raw_pair.strip()
        if not pair or "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        headers[unquote(key.strip())] = unquote(value.strip())
    return headers


class OTLPLogHandler(logging.Handler):
    """A `logging.Handler` that batches records and ships them via OTLP/HTTP."""

    def __init__(self, logs_url: str, headers: dict[str, str], resource_attrs: dict[str, str]) -> None:
        super().__init__()
        self._url = logs_url
        self._headers = {**headers, "Content-Type": "application/json"}
        self._resource_attrs = resource_attrs
        self._queue: queue.Queue[tuple[int, int, str, str]] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._dropped = 0
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, name="otlp-log-shipper", daemon=True)
        self._worker.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            ts_ns = int(record.created * 1_000_000_000)
            severity_number, severity_text = _severity_for(record.levelno)
            self._queue.put_nowait((ts_ns, severity_number, severity_text, line))
        except queue.Full:
            self._dropped += 1
        except Exception:
            self.handleError(record)

    def _run(self) -> None:
        batch: list[tuple[int, int, str, str]] = []
        last_flush = time.monotonic()

        while not self._stop_event.is_set():
            remaining = _FLUSH_INTERVAL_S - (time.monotonic() - last_flush)
            try:
                batch.append(self._queue.get(timeout=max(0.1, remaining)))
            except queue.Empty:
                pass

            due = batch and (len(batch) >= _FLUSH_BATCH_SIZE or time.monotonic() - last_flush >= _FLUSH_INTERVAL_S)
            if due:
                self._send(batch)
                batch = []
                last_flush = time.monotonic()

        # Drain whatever is left so a graceful shutdown doesn't lose the
        # final log lines (e.g. the ones explaining why the app stopped).
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._send(batch)

    def _send(self, batch: list[tuple[int, int, str, str]]) -> None:
        if self._dropped:
            batch.append(
                (time.time_ns(), 13, "WARN", f"[otlp-handler] dropped {self._dropped} log lines (queue was full)")
            )
            self._dropped = 0

        log_records = [
            {
                "timeUnixNano": str(ts_ns),
                "severityNumber": severity_number,
                "severityText": severity_text,
                "body": {"stringValue": line},
            }
            for ts_ns, severity_number, severity_text, line in batch
        ]
        payload = {
            "resourceLogs": [
                {
                    "resource": {
                        "attributes": [
                            {"key": key, "value": {"stringValue": value}}
                            for key, value in self._resource_attrs.items()
                        ]
                    },
                    "scopeLogs": [{"scope": {"name": _SERVICE_NAME}, "logRecords": log_records}],
                }
            ]
        }
        try:
            response = requests.post(self._url, json=payload, headers=self._headers, timeout=_REQUEST_TIMEOUT_S)
            if response.status_code >= 300:
                _internal_logger.debug("OTLP log push rejected: %s %s", response.status_code, response.text[:200])
        except requests.RequestException as e:
            _internal_logger.debug("OTLP log push failed: %s", e)

    def close(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=_REQUEST_TIMEOUT_S + 1.0)
        super().close()


def install_remote_logging(level: int = logging.INFO) -> logging.Handler | None:
    """Attach an `OTLPLogHandler` to the root logger, if configured.

    Returns the installed handler, or None if OTEL_EXPORTER_OTLP_ENDPOINT
    isn't set (via the environment or `.env`) - in which case logging is
    left completely untouched.
    """
    load_dotenv(_ENV_PATH)

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        _internal_logger.debug("Remote logging not configured (OTEL_EXPORTER_OTLP_ENDPOINT not set); skipping.")
        return None

    headers = _parse_otlp_headers(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", ""))
    logs_url = f"{endpoint.rstrip('/')}/v1/logs"

    host_label = os.environ.get("REACHY_LOG_HOST_LABEL", "").strip() or f"reachy-mini-{get_mac()[-6:]}"
    resource_attrs = {"service.name": _SERVICE_NAME, "service.instance.id": host_label}

    handler = OTLPLogHandler(logs_url=logs_url, headers=headers, resource_attrs=resource_attrs)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(handler)
    _internal_logger.info("Remote logging enabled -> %s (host=%s)", logs_url, host_label)
    return handler
