# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from nemo_platform_sdk_tools.sdk.core.stainless import StainlessConfig


def _test_stainless_config():
    return Path(__file__).parent / "inputs" / "simple_stainless_config.yaml"


def test_extract_models():
    st_config = StainlessConfig.from_file(_test_stainless_config())
    models = st_config.extract_models()

    assert len(models) == 13

    error_response_model = next(iter(m for m in models if m.model_name == "error_response"))
    assert error_response_model.schema_name == "ErrorResponse"
    assert error_response_model.resource_path == ["$shared"]

    result_model = next(iter(m for m in models if m.model_name == "data_designer_result"))
    assert result_model.schema_name == "DataDesignerResult"
    assert result_model.resource_path == ["beta", "data_designer", "jobs", "results"]


def test_extract_methods():
    st_config = StainlessConfig.from_file(_test_stainless_config())
    methods = st_config.extract_methods()

    assert len(methods) == 5

    ndd_endpoint = next(
        iter(m for m in methods if m.endpoint.path == "/v1beta1/data-designer/jobs/{job_id}/results/{result_id}")
    )
    assert ndd_endpoint.method_name == "retrieve"
    assert ndd_endpoint.endpoint.method == "get"
    assert ndd_endpoint.resource_path == ["beta", "data_designer", "jobs", "results"]


def test_top_level_resource_names_omits_shared():
    st_config = StainlessConfig(
        {
            "resources": {
                "jobs": {},
                "models": {},
                "$shared": {"models": {}},
            }
        }
    )

    assert st_config.top_level_resource_names() == {"jobs", "models"}
