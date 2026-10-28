# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SKILL_REFERENCES = (
    Path(__file__).resolve().parents[1] / "src" / "nemo_customizer" / "skills" / "nemo-customizer" / "references"
)


def _load_eval_helpers():
    module_name = "nemo_customizer_eval_helpers_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        SKILL_REFERENCES / "eval_helpers.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve cls.__module__ during decoration; register before exec.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


eval_helpers = _load_eval_helpers()


def test_served_model_names() -> None:
    assert eval_helpers.served_model_name(workspace="default", entity_or_adapter="qwen3-1.7b") == "default/qwen3-1.7b"
    assert (
        eval_helpers.served_model_name(workspace="default", entity_or_adapter="my-lora", finetuning="lora")
        == "default--my-lora"
    )


def test_adapter_composite_entity_name() -> None:
    assert (
        eval_helpers.adapter_composite_entity_name(
            model_entity="qwen3-1.7b",
            workspace="default",
            adapter_name="my-lora",
        )
        == "qwen3-1.7b&adapters/default/my-lora"
    )


@pytest.mark.parametrize(
    ("served_models", "expected_name"),
    [
        # SGLang serves the base model for a bare adapter name, so the reconciler maps it to <base>:<adapter>.
        (
            [
                {
                    "model_entity_id": "default/qwen3-1.7b&adapters/default/my-lora",
                    "served_model_name": "default/qwen3-1.7b:default--my-lora",
                }
            ],
            "default/qwen3-1.7b:default--my-lora",
        ),
        # A provider that publishes no served_models (e.g. a manual endpoint) gets the vLLM name.
        ([], "default--my-lora"),
    ],
)
def test_build_platform_model_target_routes_lora_via_provider(
    monkeypatch: pytest.MonkeyPatch,
    served_models: list[dict[str, str]],
    expected_name: str,
) -> None:
    def fake_get(url: str) -> dict:
        assert url == "http://localhost:8080/apis/models/v2/workspaces/default/providers/my-provider"
        return {"served_models": served_models}

    monkeypatch.setattr(eval_helpers, "_platform_get_json", fake_get)
    target = eval_helpers.build_platform_model_target(
        base_url="http://localhost:8080",
        workspace="default",
        model_entity="qwen3-1.7b",
        adapter_name="my-lora",
        provider_name="my-provider",
    )
    assert "/provider/my-provider/-/v1" in target.url
    assert "/model/qwen3-1.7b/-/v1" not in target.url
    assert target.name == expected_name


def test_build_platform_model_target_rejects_a_provider_that_does_not_serve_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Falling back to the bare name here would score the base model on SGLang.
    monkeypatch.setattr(
        eval_helpers,
        "_platform_get_json",
        lambda url: {
            "served_models": [{"model_entity_id": "default/qwen3-1.7b", "served_model_name": "default/qwen3-1.7b"}]
        },
    )
    with pytest.raises(ValueError, match="does not serve adapter default/qwen3-1.7b&adapters/default/my-lora"):
        eval_helpers.build_platform_model_target(
            base_url="http://localhost:8080",
            workspace="default",
            model_entity="qwen3-1.7b",
            adapter_name="my-lora",
            provider_name="my-provider",
        )


def _providers_page(*providers: tuple[str, list[str]]) -> dict:
    return {
        "data": [
            {
                "name": name,
                "status": "READY",
                "served_models": [{"model_entity_id": e, "served_model_name": e} for e in entity_ids],
            }
            for name, entity_ids in providers
        ]
    }


def test_find_ready_provider_for_an_adapter_needs_its_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _providers_page(
        ("a-base-only", ["default/qwen3-1.7b"]),
        ("b-lora", ["default/qwen3-1.7b", "default/qwen3-1.7b&adapters/default/my-lora"]),
    )
    monkeypatch.setattr(eval_helpers, "_platform_get_json", lambda url: page)
    find = eval_helpers.find_ready_provider_for_model_entity
    assert find(base_url="http://x", workspace="default", model_entity="qwen3-1.7b") == "a-base-only"
    assert find(base_url="http://x", workspace="default", model_entity="qwen3-1.7b", adapter_name="my-lora") == "b-lora"
    assert find(base_url="http://x", workspace="default", model_entity="qwen3-1.7b", adapter_name="other") is None


def test_build_platform_model_target_fails_when_no_provider_serves_the_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eval_helpers, "_platform_get_json", lambda url: _providers_page(("a-base-only", ["default/qwen3-1.7b"]))
    )
    with pytest.raises(ValueError, match="No READY inference provider serves adapter default/my-lora"):
        eval_helpers.build_platform_model_target(
            base_url="http://localhost:8080",
            workspace="default",
            model_entity="qwen3-1.7b",
            adapter_name="my-lora",
        )


def test_build_platform_model_target_routes_base_via_model_entity() -> None:
    target = eval_helpers.build_platform_model_target(
        base_url="http://localhost:8080",
        workspace="default",
        model_entity="qwen3-1.7b",
        provider_name="my-provider",
    )
    assert "/model/qwen3-1.7b/-/v1" in target.url
    assert "/provider/" not in target.url
    assert target.name == "default/qwen3-1.7b"


def test_build_platform_model_target_requires_ready_provider_for_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_helpers, "find_ready_provider_for_model_entity", lambda **kwargs: None)
    with pytest.raises(ValueError, match="No READY inference provider"):
        eval_helpers.build_platform_model_target(
            base_url="http://localhost:8080",
            workspace="default",
            model_entity="qwen3-1.7b",
        )


def test_gateway_path_from_url() -> None:
    assert eval_helpers.gateway_path_from_url("http://x/provider/p/-/v1") == "provider"
    assert eval_helpers.gateway_path_from_url("http://x/model/m/-/v1") == "model-entity"


def test_normalize_mcqa_answer() -> None:
    assert eval_helpers.normalize_mcqa_answer("bank") == "bank"
    assert eval_helpers.normalize_mcqa_answer("A. bank") == "bank"
    assert eval_helpers.normalize_mcqa_answer("The correct answer is: **A. bank**") == "bank"


def test_assert_chat_row_rejects_flattened() -> None:
    with pytest.raises(ValueError, match="messages"):
        eval_helpers.assert_chat_row({"prompt": "hi", "expected": "bye"})


def test_assert_chat_row_accepts_single_turn() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "yes"},
        ]
    }
    eval_helpers.assert_chat_row(row)


def test_assert_chat_row_accepts_multi_turn() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "final label"},
        ]
    }
    eval_helpers.assert_chat_row(row)
    assert eval_helpers.reference_content(row) == "final label"


def test_assert_chat_row_rejects_non_dict_message_turns() -> None:
    with pytest.raises(ValueError, match="messages\\[0\\] must be an object"):
        eval_helpers.assert_chat_row({"messages": ["x", "y"]})


def test_assert_chat_row_rejects_missing_final_assistant() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Turn 2"},
        ]
    }
    with pytest.raises(ValueError, match="assistant"):
        eval_helpers.assert_chat_row(row)


def test_load_chat_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "val.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = eval_helpers.load_chat_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == "A"


def test_chat_templates_use_messages_slice() -> None:
    assert "item.messages[:-1]" in eval_helpers.CHAT_USER_PROMPT_TEMPLATE["messages"]
    assert "item.messages[-1]" in eval_helpers.CHAT_REFERENCE_TEMPLATE


def test_lift_vs_base() -> None:
    summaries = [
        eval_helpers.EvalSummary(
            target="base",
            model_name="default/m",
            gateway_url="http://x/model/m/-/v1",
            gateway_path="model-entity",
            num_samples=10,
            raw_exact_match=0.0,
            normalized_accuracy=0.5,
            aggregate_metrics={},
        ),
        eval_helpers.EvalSummary(
            target="lora-a",
            model_name="default--a",
            gateway_url="http://x/provider/p/-/v1",
            gateway_path="provider",
            num_samples=10,
            raw_exact_match=0.7,
            normalized_accuracy=0.75,
            aggregate_metrics={},
        ),
    ]
    assert eval_helpers.lift_vs_base(summaries) == {"lora-a": 0.25}


def test_routing_sanity_warnings_detects_flat_scores() -> None:
    summaries = [
        eval_helpers.EvalSummary(
            target="base",
            model_name="default/m",
            gateway_url="http://x/model/m/-/v1",
            gateway_path="model-entity",
            num_samples=10,
            raw_exact_match=0.0,
            normalized_accuracy=0.59,
            aggregate_metrics={},
        ),
        eval_helpers.EvalSummary(
            target="lora-a",
            model_name="default--a",
            gateway_url="http://x/model/m/-/v1",
            gateway_path="model-entity",
            num_samples=10,
            raw_exact_match=0.0,
            normalized_accuracy=0.59,
            aggregate_metrics={},
        ),
    ]
    warnings = eval_helpers.routing_sanity_warnings(summaries)
    assert any("provider" in warning for warning in warnings)
    assert any("within" in warning for warning in warnings)


def test_adapter_from_completed_job_parses_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "name": "unsloth-abc",
        "status": "completed",
        "created_at": "2026-06-16T20:22:09",
        "spec": {
            "schedule": {"epochs": 3},
            "output": {"name": "my-adapter"},
            "model": {"name": "default/qwen3-1.7b"},
            "dataset": {"path": "default/commonsense_qa"},
        },
    }

    def fake_get(url: str) -> dict:
        assert url.endswith("/jobs/unsloth-abc")
        return payload

    monkeypatch.setattr(eval_helpers, "_platform_get_json", fake_get)
    info = eval_helpers.adapter_from_completed_job(
        base_url="http://localhost:8080",
        workspace="default",
        job_name="unsloth-abc",
    )
    assert info.adapter_name == "my-adapter"
    assert info.epochs == 3
    assert info.backend == "unsloth"
