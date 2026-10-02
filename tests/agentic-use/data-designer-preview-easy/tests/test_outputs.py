# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Verify that the Data Designer preview generates correct synthetic data
with sampler columns, column relationships, and LLM-generated text.

Uses two verification approaches:
1. Trace-based: Confirm the agent actually ran a preview (via CLI or SDK script)
2. SDK-based: Run the same preview config independently and validate output

The preview endpoint returns dataset as a list of record dicts
(pandas DataFrame.to_dict(orient="records") format).

TODO(mstaats): Verify LLM-generated column quality using an LLM-as-a-judge
approach — e.g., check that product descriptions are coherent, relevant to
the category/subcategory, and grammatically correct. Currently we only check
that the text is non-empty.
"""

import json
import os

import pytest
from nemo_helix_plugin.data_designer.client import DataDesignerClient
from nemo_helix_plugin.data_designer.types import PreviewRequest
from trace_reader import get_session

EXPECTED_COLUMNS = {"product_category", "product_subcategory", "price", "status", "product_description"}

SUBCATEGORY_MAPPING = {
    "Electronics": {"Smartphones", "Laptops", "Headphones"},
    "Clothing": {"Shirts", "Pants", "Shoes"},
    "Books": {"Fiction", "Non-Fiction", "Technical"},
    "Home": {"Furniture", "Kitchen", "Decor"},
}

VALID_CATEGORIES = set(SUBCATEGORY_MAPPING.keys())
VALID_STATUSES = {"in_stock", "out_of_stock", "discontinued"}

PREVIEW_CONFIG = {
    "columns": [
        {
            "name": "product_category",
            "column_type": "sampler",
            "sampler_type": "category",
            "params": {
                "values": list(VALID_CATEGORIES),
            },
        },
        {
            "name": "product_subcategory",
            "column_type": "sampler",
            "sampler_type": "subcategory",
            "params": {
                "category": "product_category",
                "values": {k: list(v) for k, v in SUBCATEGORY_MAPPING.items()},
            },
        },
        {
            "name": "price",
            "column_type": "sampler",
            "sampler_type": "uniform",
            "params": {
                "low": 5.0,
                "high": 500.0,
            },
        },
        {
            "name": "status",
            "column_type": "sampler",
            "sampler_type": "category",
            "params": {
                "values": list(VALID_STATUSES),
            },
        },
        {
            "name": "product_description",
            "column_type": "llm-text",
            "model_alias": "text",
            "prompt": (
                "Write a one-sentence product description for a {{ product_subcategory }} "
                "product in the {{ product_category }} category. Respond with only the description."
            ),
        },
    ],
    "model_configs": [
        {
            "alias": "text",
            "model": "aws/anthropic/bedrock-claude-sonnet-4-5-v1",
            "provider": "default/nvidia-inference",
            "skip_health_check": True,
        },
    ],
}


@pytest.fixture
def client() -> DataDesignerClient:
    nhx_base_url = os.environ.get("NHX_BASE_URL", "http://localhost:8080")
    return DataDesignerClient(base_url=nhx_base_url, workspace="default", timeout=120)


def _run_preview(client: DataDesignerClient, num_records: int = 5) -> list[dict]:
    """Run preview through the typed Data Designer client and return the dataset records."""
    try:
        with client.preview(body=PreviewRequest(config=PREVIEW_CONFIG, num_records=num_records)).stream() as frames:
            result = list(frames)
    except Exception as e:
        raise AssertionError(
            f"Preview API call failed with {type(e).__name__}: {e}\n"
            f"Config: {json.dumps(PREVIEW_CONFIG, indent=2)[:500]}...\n"
            f"num_records: {num_records}"
        ) from e

    kinds_seen = []
    for frame in result:
        kinds_seen.append(frame.kind)
        if frame.kind == "dataset":
            return list(frame.model_dump(mode="json")["records"])

    raise AssertionError(
        f"No frame with kind=='dataset' was returned. Received {len(kinds_seen)} frames with kinds: {kinds_seen}"
    )


def test_agent_ran_preview() -> None:
    """Test that the agent actually executed a preview (not just a help command)."""
    session = get_session()
    commands = session.get_bash_commands()

    ran_cli_preview = any(
        ("data-designer" in cmd or "data_designer" in cmd)
        and "preview" in cmd
        and "--help" not in cmd
        and "-h" not in cmd
        for cmd in commands
    )
    ran_python_script = any(
        ("python" in cmd or "uv run" in cmd)
        and ".py" in cmd
        and ("preview" in cmd or "data_designer" in cmd or "data-designer" in cmd)
        for cmd in commands
    )

    assert ran_cli_preview or ran_python_script, (
        f"Agent did not run a data-designer preview via CLI or Python script. Commands executed: {commands}"
    )
    print("Test passed: Agent executed a preview command")


def test_preview_generates_expected_columns(client: DataDesignerClient) -> None:
    """Test that the preview produces data with all five expected columns."""
    records = _run_preview(client)

    assert len(records) > 0, (
        "Preview did not return any records. The Data Designer preview endpoint may not be working."
    )

    actual_columns = set(records[0].keys())

    assert EXPECTED_COLUMNS.issubset(actual_columns), (
        f"Preview dataset missing expected columns. Expected: {EXPECTED_COLUMNS}, Got: {actual_columns}"
    )

    print(f"Test passed: Preview generated {len(records)} records with columns {actual_columns}")


def test_preview_subcategory_relationships(client: DataDesignerClient) -> None:
    """Test that subcategory values are consistent with their parent category."""
    records = _run_preview(client)

    assert len(records) > 0, "No records found in preview output"

    for i, record in enumerate(records):
        category = record["product_category"]
        subcategory = record["product_subcategory"]

        assert category in SUBCATEGORY_MAPPING, (
            f"Record {i}: Invalid category '{category}'. Expected one of {VALID_CATEGORIES}"
        )

        valid_subcategories = SUBCATEGORY_MAPPING[category]
        assert subcategory in valid_subcategories, (
            f"Record {i}: Subcategory '{subcategory}' is not valid for "
            f"category '{category}'. Expected one of {valid_subcategories}"
        )

    print(f"Test passed: All {len(records)} records have valid category-subcategory relationships")


def test_preview_data_values_valid(client: DataDesignerClient) -> None:
    """Test that price and status columns contain valid values."""
    records = _run_preview(client)

    assert len(records) > 0, "No records found in preview output"

    for i, record in enumerate(records):
        price = float(record["price"])
        assert 5.0 <= price <= 500.0, f"Record {i}: Price {price} is outside expected range [5.0, 500.0]"

        assert record["status"] in VALID_STATUSES, (
            f"Record {i}: Invalid status '{record['status']}'. Expected one of {VALID_STATUSES}"
        )

    print(f"Test passed: All {len(records)} records have valid price and status values")


def test_preview_llm_descriptions_non_empty(client: DataDesignerClient) -> None:
    """Test that the LLM-generated product_description column has non-empty text."""
    records = _run_preview(client)

    assert len(records) > 0, "No records found in preview output"

    for i, record in enumerate(records):
        desc = record.get("product_description", "")
        assert desc is not None and len(str(desc).strip()) > 0, f"Record {i}: product_description is empty or missing"

    print(f"Test passed: All {len(records)} records have non-empty product descriptions")
