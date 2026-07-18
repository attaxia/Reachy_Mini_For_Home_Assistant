import logging
import time
import unittest
from pathlib import Path
from unittest import mock

from reachy_mini_hass.core import remote_logging


class InstallRemoteLoggingTests(unittest.TestCase):
    def setUp(self):
        # Point at a path that can never exist so a real developer `.env`
        # never leaks into these tests.
        self._env_path_patch = mock.patch.object(remote_logging, "_ENV_PATH", Path("/nonexistent/.env"))
        self._env_path_patch.start()
        self._installed_handlers: list[logging.Handler] = []

    def tearDown(self):
        self._env_path_patch.stop()
        root = logging.getLogger()
        for handler in self._installed_handlers:
            root.removeHandler(handler)
            handler.close()

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_missing_config_is_a_no_op(self):
        handler = remote_logging.install_remote_logging()
        self.assertIsNone(handler)
        self.assertNotIn(remote_logging.OTLPLogHandler, [type(h) for h in logging.getLogger().handlers])

    @mock.patch.dict(
        "os.environ",
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otlp-gateway.example.test/otlp",
            "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Basic%20dXNlcjpwYXNz",
            "REACHY_LOG_HOST_LABEL": "test-robot",
        },
        clear=True,
    )
    @mock.patch("reachy_mini_hass.core.remote_logging._FLUSH_INTERVAL_S", 0.0)
    @mock.patch("reachy_mini_hass.core.remote_logging.requests.post")
    def test_configured_installs_handler_and_ships_logs(self, mock_post):
        mock_post.return_value = mock.Mock(status_code=200, text="")

        handler = remote_logging.install_remote_logging()
        self.assertIsInstance(handler, remote_logging.OTLPLogHandler)
        self._installed_handlers.append(handler)

        logger = logging.getLogger("reachy_mini_hass.test_remote_logging")
        logger.setLevel(logging.INFO)  # root defaults to WARNING; this test needs INFO through
        logger.info("hello from the robot")

        # With _FLUSH_INTERVAL_S patched to 0, the worker thread flushes on
        # its next wakeup rather than waiting a full second; poll briefly
        # instead of racing a fixed sleep against the background thread.
        deadline = time.monotonic() + 2.0
        while not mock_post.called and time.monotonic() < deadline:
            time.sleep(0.01)

        mock_post.assert_called()
        _, kwargs = mock_post.call_args
        self.assertEqual(mock_post.call_args.args[0], "https://otlp-gateway.example.test/otlp/v1/logs")
        self.assertEqual(kwargs["headers"]["Authorization"], "Basic dXNlcjpwYXNz")

        resource_log = kwargs["json"]["resourceLogs"][0]
        attrs = {a["key"]: a["value"]["stringValue"] for a in resource_log["resource"]["attributes"]}
        self.assertEqual(attrs, {"service.name": "reachy_mini_hass", "service.instance.id": "test-robot"})

        log_records = resource_log["scopeLogs"][0]["logRecords"]
        self.assertTrue(any("hello from the robot" in r["body"]["stringValue"] for r in log_records))


class ParseOtlpHeadersTests(unittest.TestCase):
    def test_parses_percent_encoded_single_header(self):
        headers = remote_logging._parse_otlp_headers("Authorization=Basic%20dXNlcjpwYXNz")
        self.assertEqual(headers, {"Authorization": "Basic dXNlcjpwYXNz"})

    def test_parses_multiple_comma_separated_headers(self):
        headers = remote_logging._parse_otlp_headers("a=1,b=2")
        self.assertEqual(headers, {"a": "1", "b": "2"})

    def test_empty_string_yields_no_headers(self):
        self.assertEqual(remote_logging._parse_otlp_headers(""), {})


class OTLPLogHandlerQueueOverflowTests(unittest.TestCase):
    @mock.patch("reachy_mini_hass.core.remote_logging._QUEUE_MAXSIZE", 1)
    def test_emit_drops_instead_of_blocking_when_queue_is_full(self):
        handler = remote_logging.OTLPLogHandler(
            logs_url="https://otlp-gateway.example.test/otlp/v1/logs",
            headers={},
            resource_attrs={"service.name": "test"},
        )
        try:
            # Stop the worker immediately so records pile up in the queue
            # instead of being drained, letting us exercise the overflow path.
            handler._stop_event.set()
            handler._worker.join(timeout=2)

            for i in range(5):
                record = logging.LogRecord("test", logging.INFO, __file__, 1, f"line {i}", (), None)
                handler.emit(record)  # must never raise or block

            self.assertGreaterEqual(handler._dropped, 1)
        finally:
            handler.close()


if __name__ == "__main__":
    unittest.main()
