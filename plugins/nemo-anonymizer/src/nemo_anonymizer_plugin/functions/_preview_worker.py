# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronous preview worker that runs the upstream Anonymizer library.

Executes in a worker thread and emits frames back to the async caller via
the ``send_frame`` callback.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import pandas as pd
from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.interface.anonymizer import Anonymizer
from data_designer.config.models import ModelProvider as DDModelProvider
from nemo_anonymizer_plugin.app.upstream_logging import preserve_root_logging
from nemo_anonymizer_plugin.functions.preview import (
    FailedRecordsFrame,
    PreviewDatasetFrame,
    PreviewSpec,
    TraceDatasetFrame,
)
from pydantic import BaseModel


def _make_preview(
    send_frame: Callable[[BaseModel], None],
    spec: PreviewSpec,
    data: AnonymizerInput,
    model_configs_yaml: str,
    dd_providers: list[DDModelProvider] | None,
    num_records: int,
) -> None:
    """Run ``Anonymizer.preview(...)`` and stream the result frames."""
    anonymizer = _make_anonymizer(model_configs_yaml=model_configs_yaml, dd_providers=dd_providers)
    config: AnonymizerConfig = spec.config.to_anonymizer_config()

    result = anonymizer.preview(config=config, data=data, num_records=num_records)

    send_frame(PreviewDatasetFrame(records=_to_jsonable_records(result.dataframe)))
    send_frame(
        TraceDatasetFrame(
            records=_to_jsonable_records(result.trace_dataframe),
            original_text_column=_get_original_text_column(result.trace_dataframe, fallback=spec.data.text_column),
        )
    )

    failures = [
        {
            "record_id": getattr(f, "record_id", None),
            "step": getattr(f, "step", None),
            "reason": getattr(f, "reason", None),
        }
        for f in (result.failed_records or [])
    ]
    send_frame(FailedRecordsFrame(records=failures))


def _make_anonymizer(
    *,
    model_configs_yaml: str,
    dd_providers: list[DDModelProvider] | None,
) -> Anonymizer:
    if not model_configs_yaml:
        raise RuntimeError("Anonymizer preview requires resolved model_configs.")
    with preserve_root_logging():
        return Anonymizer(
            model_configs=model_configs_yaml,
            model_providers=dd_providers,
        )


def _to_jsonable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of JSON-serializable dicts.

    ``df.to_dict(orient="records")`` may include ``numpy`` scalars,
    ``pd.Timestamp``, or ``NaN`` values that won't round-trip through Pydantic
    as JSON. We use pandas' JSON exporter to coerce values, then parse back.
    """
    if df is None or df.empty:
        return []
    safe = df.copy()
    # Replace nan/inf with None for JSON safety.
    for col in safe.columns:
        if pd.api.types.is_float_dtype(safe[col]):
            safe[col] = safe[col].apply(
                lambda v: None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
            )
    records_json = safe.to_json(orient="records", date_format="iso", default_handler=str)
    if records_json is None:
        return []
    return json.loads(records_json)


def _get_original_text_column(df: pd.DataFrame, *, fallback: str) -> str:
    value = df.attrs.get("original_text_column")
    if isinstance(value, str) and value:
        return value
    return fallback
