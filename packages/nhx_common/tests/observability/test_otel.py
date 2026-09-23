# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenTelemetry observability initialization, particularly logging configuration."""

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from io import StringIO
from unittest import mock

import pytest
from nhx.common.observability.otel import initialize_logging, settings


@pytest.fixture
def reset_logging():
    """Reset logging configuration after each test."""
    yield
    # Clear all handlers after test
    logging.getLogger().handlers.clear()


@pytest.fixture
def capture_log_output():
    """Capture log output to a string buffer."""
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    yield log_capture, handler
    handler.close()


def _create_test_logger(name: str, log_capture: StringIO) -> logging.Logger:
    """Create a test logger that captures output to the given StringIO buffer."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(logging.getLogger().handlers[0].formatter)
    logger.addHandler(handler)
    return logger


class TestJsonFormat:
    """Tests for JSON log format."""

    def test_produces_valid_json(self, reset_logging, capture_log_output):
        """Test that JSON format produces valid JSON output."""
        log_capture, _ = capture_log_output

        with mock.patch.object(settings, "log_format", "json"):
            initialize_logging()
            logger = _create_test_logger("test_json_format", log_capture)
            logger.info("Test JSON log message")

            output = log_capture.getvalue()
            assert output, "Expected log output"

            log_entry = json.loads(output.strip())
            assert log_entry["message"] == "Test JSON log message"
            assert log_entry["level"] == "info"
            assert "timestamp" in log_entry
            assert log_entry["logger"] == "test_json_format"

    def test_includes_extra_fields(self, reset_logging, capture_log_output):
        """Test that JSON format includes extra fields passed to logger."""
        log_capture, _ = capture_log_output

        with mock.patch.object(settings, "log_format", "json"):
            initialize_logging()
            logger = _create_test_logger("test_extra_fields", log_capture)

            logger.info(
                "Platform starting",
                extra={
                    "version": "1.0.0",
                    "services": ["service1", "service2"],
                    "port": 8000,
                },
            )

            output = log_capture.getvalue()
            log_entry = json.loads(output.strip())

            assert log_entry["message"] == "Platform starting"
            assert log_entry["version"] == "1.0.0"
            assert log_entry["services"] == ["service1", "service2"]
            assert log_entry["port"] == 8000

    def test_includes_source_location(self, reset_logging, capture_log_output):
        """Test that JSON format includes filename, function name, and line number."""
        log_capture, _ = capture_log_output

        with mock.patch.object(settings, "log_format", "json"):
            initialize_logging()
            logger = _create_test_logger("test_source_location", log_capture)
            logger.info("Test source location")

            output = log_capture.getvalue()
            log_entry = json.loads(output.strip())

            assert "filename" in log_entry
            assert "func_name" in log_entry
            assert "lineno" in log_entry
            assert log_entry["filename"] == "test_otel.py"

    def test_timestamp_has_microsecond_precision(self, reset_logging, capture_log_output):
        """Test that JSON format uses ISO timestamp with microsecond precision."""
        log_capture, _ = capture_log_output

        with mock.patch.object(settings, "log_format", "json"):
            initialize_logging()
            logger = _create_test_logger("test_json_timestamp", log_capture)
            logger.info("Test JSON log message")

            output = log_capture.getvalue()
            assert output, "Expected log output"
            assert "timestamp" in output

            # ISO format: 2025-12-05T10:30:45.123456Z (6 digits = microseconds)
            iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}"
            assert re.search(iso_pattern, output), f"Expected ISO timestamp with microseconds: {output}"

            # Should NOT be milliseconds only (exactly 3 digits)
            millisecond_only_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}(?!\d)"
            assert not re.search(millisecond_only_pattern, output), (
                f"JSON format should use microseconds, not milliseconds: {output}"
            )


class TestPlainFormat:
    """Tests for plain text log format."""

    def test_is_human_readable(self, reset_logging, capture_log_output):
        """Test that plain format produces human-readable output (not JSON)."""
        log_capture, _ = capture_log_output

        with mock.patch.object(settings, "log_format", "plain"):
            initialize_logging()
            logger = _create_test_logger("test_plain_format", log_capture)
            logger.info("Test plain log message")

            output = log_capture.getvalue()
            assert output, "Expected log output"

            # Plain format should NOT be valid JSON
            with pytest.raises(json.JSONDecodeError):
                json.loads(output.strip())

            # But should contain the message and logger name
            assert "Test plain log message" in output
            assert "test_plain_format" in output

    def test_timestamp_has_millisecond_precision(self, reset_logging, capture_log_output):
        """Test that plain format uses ISO timestamp with milliseconds precision."""
        log_capture, _ = capture_log_output

        with mock.patch.object(settings, "log_format", "plain"):
            initialize_logging()
            logger = _create_test_logger("test_plain_timestamp", log_capture)
            logger.info("Test plain log message")

            output = log_capture.getvalue()
            assert output, "Expected log output"
            assert "Test plain log message" in output

            # Plain format: 2025-12-05T10:30:45.123 (3 digits = milliseconds)
            millisecond_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}(?!\d)"
            assert re.search(millisecond_pattern, output), (
                f"Expected ISO timestamp with milliseconds (3 digits): {output}"
            )

            # Verify it's NOT microseconds (6 digits)
            microsecond_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}"
            assert not re.search(microsecond_pattern, output), (
                f"Expected milliseconds precision, but found microseconds: {output}"
            )

    def test_multiple_messages_have_millisecond_timestamps(self, reset_logging, capture_log_output):
        """Test that plain format uses custom _stamper function with milliseconds."""
        log_capture, _ = capture_log_output

        with mock.patch.object(settings, "log_format", "plain"):
            initialize_logging()
            logger = _create_test_logger("test_stamper", log_capture)

            logger.info("First message")
            logger.info("Second message")

            output = log_capture.getvalue()

            # Count millisecond-precision timestamps
            millisecond_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}(?!\d)"
            matches = re.findall(millisecond_pattern, output)
            assert len(matches) >= 2, f"Expected at least 2 millisecond timestamps: {output}"

            # Verify none are microsecond precision
            microsecond_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}"
            microsecond_matches = re.findall(microsecond_pattern, output)
            assert len(microsecond_matches) == 0, f"Should not have microsecond timestamps: {output}"


class TestLogLevelFiltering:
    """Tests for log level filtering."""

    def test_filters_below_configured_level(self, reset_logging, capture_log_output):
        """Test that log level filtering works correctly."""
        log_capture, _ = capture_log_output

        with mock.patch.object(settings, "log_format", "json"):
            with mock.patch.object(settings, "log_level", "WARNING"):
                initialize_logging()

                # Create logger without setting level (inherits from root)
                logger = logging.getLogger("test_level_filter")
                handler = logging.StreamHandler(log_capture)
                handler.setFormatter(logging.getLogger().handlers[0].formatter)
                logger.addHandler(handler)

                # This should not appear (below WARNING level)
                logger.info("Info message")
                # This should appear
                logger.warning("Warning message")

                output = log_capture.getvalue()
                assert "Warning message" in output
                assert "Info message" not in output


def _get_clean_otel_env(**overrides) -> dict:
    """Create an environment with OTEL disabled to prevent network connections.

    This ensures subprocess tests don't hang when the parent environment has
    OTEL exporters configured that try to connect to unreachable endpoints.
    """
    env = os.environ.copy()
    # Disable OTEL exporters to prevent network connections that could hang
    env["OTEL_TRACES_EXPORTER"] = "none"
    env["OTEL_METRICS_EXPORTER"] = "none"
    env["OTEL_LOGS_EXPORTER"] = "none"
    env.update(overrides)
    return env


@pytest.mark.integration
class TestLoggingIntegration:
    """Integration tests for logging via subprocess (spawn Python subprocess; skip in unit/xdist to avoid worker crashes)."""

    def test_initialize_obs_configures_plain_logging_by_default(self):
        """Test that initialize_obs() enables plain text logging by default."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import logging
from nhx.common.observability import initialize_obs
from nhx.common.observability.otel import settings

assert settings.log_format == "plain", f"Expected 'plain', got '{settings.log_format}'"

initialize_obs()

logger = logging.getLogger("integration_test")
logger.info("Integration test message")
""",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=_get_clean_otel_env(),
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        output = result.stderr  # Logs go to stderr (StreamHandler default)
        assert "Integration test message" in output

        # Verify it's NOT valid JSON (it's plain text)
        for line in output.strip().split("\n"):
            if line.strip() and "Integration test message" in line:
                with pytest.raises(json.JSONDecodeError):
                    json.loads(line)

    def test_initialize_obs_configures_json_logging_when_env_set(self):
        """Test that LOG_FORMAT=json produces JSON logs."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import logging
from nhx.common.observability import initialize_obs

initialize_obs()

logger = logging.getLogger("integration_test")
logger.info("JSON test message")
""",
            ],
            capture_output=True,
            text=True,
            timeout=30,  # TODO: if this times out often in CI, consider 60s or a more robust fix (e.g. avoid subprocess)
            env=_get_clean_otel_env(LOG_FORMAT="json"),
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        output = result.stderr
        assert "JSON test message" in output

        # Verify it's valid JSON
        for line in output.strip().split("\n"):
            if line.strip():
                log_entry = json.loads(line)
                assert "message" in log_entry
                assert "level" in log_entry


class TestTimestampFormatComparison:
    """Tests demonstrating timestamp precision differences."""

    def test_microsecond_vs_millisecond_precision(self):
        """Test to demonstrate the difference between microsecond and millisecond precision."""
        now = datetime.now()

        # Default ISO format (microseconds - 6 digits)
        iso_default = now.isoformat()

        # Milliseconds precision (3 digits)
        iso_milliseconds = now.isoformat(timespec="milliseconds")

        # Verify the formats are different
        assert iso_default != iso_milliseconds or "." not in iso_default, "ISO formats should differ in precision"

        # Verify milliseconds has 3 digits after decimal
        if "." in iso_milliseconds:
            fractional_part = iso_milliseconds.split(".")[-1]
            assert len(fractional_part) == 3, f"Expected 3 digits for milliseconds, got {len(fractional_part)}"

        # Verify default has 6 digits after decimal (microseconds)
        if "." in iso_default:
            fractional_part = iso_default.split(".")[-1]
            assert len(fractional_part) == 6, f"Expected 6 digits for microseconds, got {len(fractional_part)}"
