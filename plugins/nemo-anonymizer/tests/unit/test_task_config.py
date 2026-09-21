# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``app/task_config.py`` request models."""

from __future__ import annotations

from pathlib import Path

import pytest
from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import Redact
from nemo_anonymizer_plugin.app.input import AnonymizerInputSpec
from nemo_anonymizer_plugin.app.task_config import (
    AnonymizerConfigRequest,
    AnonymizerRequest,
    PreviewRequest,
    RedactRequest,
)
from nemo_anonymizer_plugin.config import DEFAULT_MAX_PREVIEW_NUM_RECORDS
from pydantic import ValidationError


def _config() -> AnonymizerConfigRequest:
    return AnonymizerConfigRequest(replace=RedactRequest())


def test_preview_request_accepts_http_source() -> None:
    req = PreviewRequest(
        config=_config(),
        data=AnonymizerInputSpec(source="https://example.com/x.csv", text_column="text"),
        num_records=5,
    )
    assert req.data.source == "https://example.com/x.csv"


def test_anonymizer_request_accepts_local_path(tmp_path: Path) -> None:
    csv = tmp_path / "x.csv"
    csv.write_text("text\nhello\n")
    req = AnonymizerRequest(
        config=_config(),
        data=AnonymizerInputSpec(source=str(csv), text_column="text"),
    )
    assert req.data.source == str(csv)


def test_anonymizer_request_accepts_fileset_source_without_local_path_validation() -> None:
    req = AnonymizerRequest(
        config=_config(),
        data=AnonymizerInputSpec(source="team-a/pii-inputs#data/input.parquet", text_column="text"),
    )

    assert req.data.source == "team-a/pii-inputs#data/input.parquet"


def test_preview_num_records_advertises_the_default_maximum() -> None:
    schema = PreviewRequest.model_json_schema()["properties"]["num_records"]

    assert schema["minimum"] == 1
    assert schema["maximum"] == DEFAULT_MAX_PREVIEW_NUM_RECORDS
    assert schema["default"] == DEFAULT_MAX_PREVIEW_NUM_RECORDS


def test_preview_num_records_leaves_the_ceiling_to_the_configured_max() -> None:
    req = PreviewRequest(
        config=_config(),
        data=AnonymizerInputSpec(source="https://example.com/x.csv", text_column="text"),
        num_records=DEFAULT_MAX_PREVIEW_NUM_RECORDS + 1,
    )

    assert req.num_records == DEFAULT_MAX_PREVIEW_NUM_RECORDS + 1


def test_request_dump_preserves_replace_discriminator() -> None:
    req = PreviewRequest(
        config=_config(),
        data=AnonymizerInputSpec(source="https://example.com/x.csv", text_column="text"),
    )

    dumped = req.model_dump(mode="json")

    assert dumped["config"]["replace"]["kind"] == "redact"


def test_replace_missing_kind_raises_validation_error_not_type_error() -> None:
    """ASTD-328: a missing discriminator must 422, not crash the upstream callable Discriminator with a raw TypeError."""
    with pytest.raises(ValidationError, match="kind"):
        AnonymizerConfigRequest.model_validate({"replace": {}})


def test_replace_invalid_kind_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="kind"):
        AnonymizerConfigRequest.model_validate({"replace": {"kind": "not-a-real-strategy"}})


def test_replace_schema_models_kind_as_a_literal_field() -> None:
    """ASTD-329: ``kind`` must be a real modelled field so OpenAPI/generated SDKs see it."""
    schema = AnonymizerConfigRequest.model_json_schema()
    replace_variants = schema["$defs"]

    for variant_name in ("SubstituteRequest", "RedactRequest", "AnnotateRequest", "HashRequest"):
        variant = replace_variants[variant_name]
        assert (
            variant["properties"]["kind"]["const"]
            == {
                "SubstituteRequest": "substitute",
                "RedactRequest": "redact",
                "AnnotateRequest": "annotate",
                "HashRequest": "hash",
            }[variant_name]
        )


def test_replace_accepts_a_valid_kind_and_round_trips_to_upstream_config() -> None:
    req = AnonymizerConfigRequest.model_validate({"replace": {"kind": "redact"}})

    upstream = req.to_anonymizer_config()

    assert isinstance(upstream, AnonymizerConfig)
    assert isinstance(upstream.replace, Redact)


def test_config_still_accepts_an_upstream_anonymizer_config_instance() -> None:
    """SDK callers that build an upstream ``AnonymizerConfig`` directly (as before ASTD-329) keep working."""
    req = AnonymizerRequest(
        config=AnonymizerConfig(replace=Redact()),  # ty: ignore[invalid-argument-type]
        data=AnonymizerInputSpec(source="https://example.com/x.csv", text_column="text"),
    )

    assert req.config.replace is not None
    assert req.config.replace.kind == "redact"
    assert isinstance(req.config.to_anonymizer_config().replace, Redact)
