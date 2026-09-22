# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-training evaluation helpers — keep eval dataset shape aligned with CHAT training JSONL.

**LoRA** (``output.save_method: lora``): adapters registered on the base model entity
are hot-reloaded on deployments with ``lora_enabled: true`` — no deployment update or
new inference deployment before eval.

**Full SFT** (``finetuning_type: all_weights``) or **merged LoRA checkpoints**
(``save_method: merged_16bit`` / ``merged_4bit``): the job registers a new **model**
entity at ``output.name``. Deploy that entity for inference before chat or eval — full
weights are not hot-reloaded onto the base model's deployment.

Run from the nemo-helix git root (reads ``$NHX_BASE_URL`` when ``--base-url`` is omitted)::

    export NHX_BASE_URL=http://127.0.0.1:8080
    uv run python plugins/nemo-customizer/src/nemo_customizer/skills/nemo-customizer/references/eval_helpers.py \\
        --model-entity <model-entity> --adapter <adapter-a> --adapter <adapter-b> \\
        --provider <provider> --dataset-fileset <dataset-fileset> --split validation.jsonl

Import in agent scripts (add references/ to sys.path or run via uv from repo root).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# --- Train/eval format contract (CHAT JSONL) --------------------------------

CHAT_ROW_KEYS = frozenset({"messages"})

# Inference: all turns except the final assistant label (single- or multi-turn).
CHAT_USER_PROMPT_TEMPLATE: dict[str, Any] = {
    "messages": "{{ item.messages[:-1] }}",
}

# Metric reference: final assistant turn (the label to predict).
CHAT_REFERENCE_TEMPLATE = "{{ item.messages[-1].content }}"

# Back-compat alias for single-turn MCQA docs/snippets.
CHAT_SINGLE_TURN_USER_PROMPT_TEMPLATE = {
    "messages": [{"role": "user", "content": "{{ item.messages[0].content }}"}],
}

PLATFORM_HTTP_TIMEOUT_SEC = 60


def _assert_message_turn(turn: Any, *, label: str, index: int | str) -> dict[str, Any]:
    """Validate one messages[] element is a dict before reading role/content."""
    if not isinstance(turn, dict):
        raise ValueError(f"{label}: messages[{index}] must be an object with role/content, got {type(turn).__name__}")
    return turn


def assert_chat_row(row: dict[str, Any], *, index: int | None = None) -> None:
    """Validate one dataset row matches automodel/unsloth CHAT training shape."""
    label = f"row {index}" if index is not None else "row"
    if "messages" not in row:
        raise ValueError(
            f"{label}: expected CHAT format with 'messages' array; got keys {sorted(row)}. "
            "Do not flatten to prompt/expected — use references/post-training-eval.md."
        )
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{label}: messages must be a list with at least one prompt turn + final assistant label")
    first = _assert_message_turn(messages[0], label=label, index=0)
    if first.get("role") != "user":
        raise ValueError(f"{label}: expected messages[0]=user")
    last = _assert_message_turn(messages[-1], label=label, index=-1)
    if last.get("role") != "assistant":
        raise ValueError(f"{label}: expected final messages[-1]=assistant (the label to score)")


def reference_content(row: dict[str, Any]) -> str:
    """Return the assistant label for a CHAT row (final turn)."""
    assert_chat_row(row)
    return row["messages"][-1]["content"]


def load_chat_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load JSONL rows; validate CHAT shape; return rows unchanged."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            assert_chat_row(row, index=index)
            rows.append(row)
    return rows


def load_chat_jsonl_from_platform(
    *,
    base_url: str,
    workspace: str,
    fileset: str,
    remote_path: str,
) -> list[dict[str, Any]]:
    """Download a JSONL split from a platform fileset and validate CHAT rows."""
    url = f"{base_url.rstrip('/')}/apis/files/v2/workspaces/{workspace}/filesets/{fileset}/-/{remote_path.lstrip('/')}"
    with urllib.request.urlopen(url, timeout=PLATFORM_HTTP_TIMEOUT_SEC) as response:
        content = response.read().decode("utf-8")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        assert_chat_row(row, index=index)
        rows.append(row)
    return rows


def chat_metrics():
    """Build default metrics for CHAT SFT eval (exact match + ROUGE + BLEU)."""
    from nemo_evaluator_sdk import BLEUMetric, ROUGEMetric
    from nemo_evaluator_sdk.metrics.exact_match import ExactMatchMetric

    ref = CHAT_REFERENCE_TEMPLATE
    return [
        ExactMatchMetric(reference=ref),
        ROUGEMetric(reference=ref),
        BLEUMetric(references=[ref]),
    ]


def normalize_mcqa_answer(text: str) -> str:
    """Normalize MCQA model output for comparison with bare choice-text references."""
    text = text.strip()
    bold = re.search(r"\*\*(?:[A-E]\.\s*)?([^*]+)\*\*", text)
    if bold:
        text = bold.group(1)
    text = re.sub(r"^[A-E]\.\s*", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return text.strip().lower()


def served_model_name(*, workspace: str, entity_or_adapter: str, finetuning: str = "base") -> str:
    """Return the ``model`` field for base entity or LoRA adapter requests."""
    if finetuning == "base":
        return f"{workspace}/{entity_or_adapter}"
    if finetuning == "lora":
        return f"{workspace}--{entity_or_adapter}"
    raise ValueError("finetuning must be 'base' or 'lora'")


def adapter_composite_entity_name(*, model_entity: str, workspace: str, adapter_name: str) -> str:
    """LoRA composite model-entity path segment (for reference / OpenAI-route body only).

    The model-entity proxy path ``model/{composite}/-/v1`` requires a dedicated
    VirtualModel per composite and typically 404s on stock deployments. Prefer
    :func:`provider_gateway_url` for adapter eval.
    """
    return f"{model_entity}&adapters/{workspace}/{adapter_name}"


def model_entity_gateway_url(*, base_url: str, workspace: str, model_entity: str) -> str:
    """OpenAI-compatible inference-gateway URL for a registered base model entity."""
    return f"{base_url.rstrip('/')}/apis/inference-gateway/v2/workspaces/{workspace}/model/{model_entity}/-/v1"


def provider_gateway_url(*, base_url: str, workspace: str, provider_name: str) -> str:
    """OpenAI-compatible inference-gateway URL for a model provider (LoRA eval route)."""
    return f"{base_url.rstrip('/')}/apis/inference-gateway/v2/workspaces/{workspace}/provider/{provider_name}/-/v1"


def gateway_path_from_url(url: str) -> str:
    """Return ``model-entity`` or ``provider`` from a gateway base URL."""
    if "/provider/" in url:
        return "provider"
    if "/model/" in url:
        return "model-entity"
    return "unknown"


def _platform_get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=PLATFORM_HTTP_TIMEOUT_SEC) as response:
        return json.loads(response.read().decode("utf-8"))


def find_ready_provider_for_model_entity(
    *,
    base_url: str,
    workspace: str,
    model_entity: str,
) -> str | None:
    """Return a READY provider name that serves ``workspace/model_entity`` (base or LoRA)."""
    url = f"{base_url.rstrip('/')}/apis/models/v2/workspaces/{workspace}/providers?page_size=100&filter.status=READY"
    payload = _platform_get_json(url)
    base_entity_id = f"{workspace}/{model_entity}"
    matches: list[str] = []
    for provider in payload.get("data", []):
        if provider.get("status") != "READY":
            continue
        for served in provider.get("served_models") or []:
            entity_id = served.get("model_entity_id") or ""
            if entity_id == base_entity_id or entity_id.startswith(f"{base_entity_id}&adapters/"):
                matches.append(provider["name"])
                break
    if not matches:
        return None
    # Prefer deployment-backed providers (stable) over arbitrary first hit.
    return sorted(set(matches))[0]


@dataclass
class JobAdapterInfo:
    job_name: str
    adapter_name: str
    epochs: int | None
    backend: str
    model_entity: str
    dataset_ref: str
    status: str
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_name": self.job_name,
            "adapter_name": self.adapter_name,
            "epochs": self.epochs,
            "backend": self.backend,
            "model_entity": self.model_entity,
            "dataset_ref": self.dataset_ref,
            "status": self.status,
            "created_at": self.created_at,
        }


def adapter_from_completed_job(
    *,
    base_url: str,
    workspace: str,
    job_name: str,
) -> JobAdapterInfo:
    """Resolve adapter output name and training epochs from a platform job."""
    url = f"{base_url.rstrip('/')}/apis/jobs/v2/workspaces/{workspace}/jobs/{job_name}"
    try:
        job = _platform_get_json(url)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Job not found: {workspace}/{job_name}") from exc
    spec = job.get("spec") or {}
    output_name = (spec.get("output") or {}).get("name") or spec.get("name")
    if not output_name:
        raise ValueError(f"Job {job_name} has no output adapter name in spec")
    model = spec.get("model")
    model_entity = model.get("name", "") if isinstance(model, dict) else (model or "")
    if model_entity.startswith(f"{workspace}/"):
        model_entity = model_entity.split("/", 1)[1]
    dataset = spec.get("dataset") or {}
    dataset_ref = dataset.get("path") or dataset.get("training") or ""
    backend = job_name.split("-", 1)[0] if "-" in job_name else "unknown"
    return JobAdapterInfo(
        job_name=job_name,
        adapter_name=output_name,
        epochs=(spec.get("schedule") or {}).get("epochs"),
        backend=backend,
        model_entity=model_entity,
        dataset_ref=dataset_ref,
        status=job.get("status", "unknown"),
        created_at=job.get("created_at"),
    )


def list_completed_job_adapters(
    *,
    base_url: str,
    workspace: str,
    model_entity: str,
    dataset_fileset: str | None = None,
    page_size: int = 500,
) -> list[JobAdapterInfo]:
    """List completed customization jobs and their output adapter names."""
    url = (
        f"{base_url.rstrip('/')}/apis/jobs/v2/workspaces/{workspace}/jobs?page_size={page_size}&filter.status=completed"
    )
    payload = _platform_get_json(url)
    dataset_ref = f"{workspace}/{dataset_fileset}" if dataset_fileset else None
    model_ref = f"{workspace}/{model_entity}"
    results: list[JobAdapterInfo] = []
    for job in payload.get("data", []):
        if job.get("status") != "completed":
            continue
        spec = job.get("spec") or {}
        out = (spec.get("output") or {}).get("name") or spec.get("name")
        if not out:
            continue
        model = spec.get("model")
        job_model = model.get("name", "") if isinstance(model, dict) else (model or "")
        ds = spec.get("dataset") or {}
        job_ds = ds.get("path") or ds.get("training") or ""
        if model_ref not in str(job_model):
            continue
        if dataset_ref and dataset_ref not in str(job_ds):
            continue
        backend = job["name"].split("-", 1)[0] if "-" in job["name"] else "unknown"
        results.append(
            JobAdapterInfo(
                job_name=job["name"],
                adapter_name=out,
                epochs=(spec.get("schedule") or {}).get("epochs"),
                backend=backend,
                model_entity=model_entity,
                dataset_ref=job_ds,
                status=job.get("status", "completed"),
                created_at=job.get("created_at"),
            )
        )
    results.sort(key=lambda item: item.created_at or "", reverse=True)
    return results


def build_online_eval_config(
    *,
    max_tokens: int = 64,
    temperature: float = 0,
    parallelism: int = 8,
    enable_thinking: bool = False,
    limit_samples: int | None = None,
):
    """RunConfigOnlineModel defaults aligned with Qwen3 CHAT SFT eval."""
    from nemo_evaluator_sdk.values import InferenceParams, RunConfigOnlineModel

    extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}} if not enable_thinking else None
    inference_kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
    if extra_body:
        inference_kwargs["extra_body"] = extra_body
    return RunConfigOnlineModel(
        parallelism=parallelism,
        limit_samples=limit_samples,
        inference=InferenceParams(**inference_kwargs),
    )


def build_platform_model_target(
    *,
    base_url: str,
    workspace: str,
    model_entity: str,
    adapter_name: str | None = None,
    provider_name: str | None = None,
):
    """SDK Model target for base entity or LoRA adapter on the platform gateway.

    Base weights use the **model-entity** proxy
    (``/model/{entity}/-/v1``). LoRA adapters must use the **provider** proxy
    (``/provider/{name}/-/v1``) with ``model: {workspace}--{adapter}`` — the
    model-entity path always routes to the base VirtualModel and ignores adapter
    names in the request body.
    """
    from nemo_evaluator_sdk.values.models import Model

    resolved_provider = provider_name or find_ready_provider_for_model_entity(
        base_url=base_url,
        workspace=workspace,
        model_entity=model_entity,
    )
    if not resolved_provider:
        raise ValueError(
            f"No READY inference provider serves {workspace}/{model_entity}. "
            "Deploy the base model (with lora_enabled: true for LoRA eval) or pass --provider <name>."
        )

    if adapter_name:
        return Model(
            url=provider_gateway_url(
                base_url=base_url,
                workspace=workspace,
                provider_name=resolved_provider,
            ),
            name=served_model_name(workspace=workspace, entity_or_adapter=adapter_name, finetuning="lora"),
        )

    return Model(
        url=model_entity_gateway_url(base_url=base_url, workspace=workspace, model_entity=model_entity),
        name=served_model_name(workspace=workspace, entity_or_adapter=model_entity, finetuning="base"),
    )


@dataclass
class EvalSummary:
    target: str
    model_name: str
    gateway_url: str
    gateway_path: str
    num_samples: int
    raw_exact_match: float
    normalized_accuracy: float
    aggregate_metrics: dict[str, dict[str, float | None]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "model_name": self.model_name,
            "gateway_url": self.gateway_url,
            "gateway_path": self.gateway_path,
            "num_samples": self.num_samples,
            "raw_exact_match": self.raw_exact_match,
            "normalized_accuracy": self.normalized_accuracy,
            "metrics": self.aggregate_metrics,
        }


def summarize_chat_eval_result(*, target: str, model_name: str, gateway_url: str, result) -> EvalSummary:
    """Summarize Evaluator benchmark result for CHAT rows."""
    em_rows = result.per_metric["exact-match"].row_scores
    num_samples = len(em_rows)
    raw_correct = sum(
        1 for rs in em_rows if rs.sample.get("output_text", "").strip() == reference_content(rs.item).strip()
    )
    norm_correct = sum(
        1
        for rs in em_rows
        if normalize_mcqa_answer(rs.sample.get("output_text", "")) == normalize_mcqa_answer(reference_content(rs.item))
    )
    aggregate_metrics: dict[str, dict[str, float | None]] = {}
    for metric_name, metric_result in result.per_metric.items():
        aggregate_metrics[metric_name] = {
            score.name.split(".")[-1]: round(score.mean, 4) if score.mean is not None else None
            for score in metric_result.aggregate_scores.scores
        }
    return EvalSummary(
        target=target,
        model_name=model_name,
        gateway_url=gateway_url,
        gateway_path=gateway_path_from_url(gateway_url),
        num_samples=num_samples,
        raw_exact_match=round(raw_correct / num_samples, 4) if num_samples else 0.0,
        normalized_accuracy=round(norm_correct / num_samples, 4) if num_samples else 0.0,
        aggregate_metrics=aggregate_metrics,
    )


def run_chat_online_eval(
    *,
    rows: Sequence[dict[str, Any]],
    target,
    config,
    metrics=None,
    prompt_template: dict[str, Any] | None = None,
):
    """Run online eval on CHAT rows using shared templates."""
    from nemo_evaluator_sdk import Evaluator

    for index, row in enumerate(rows):
        assert_chat_row(row, index=index)
    if metrics is None:
        metrics = chat_metrics()
    return Evaluator().run_sync(
        metrics=metrics,
        dataset=list(rows),
        target=target,
        prompt_template=prompt_template or CHAT_USER_PROMPT_TEMPLATE,
        config=config,
    )


def _eval_target(
    *,
    base_url: str,
    workspace: str,
    model_entity: str,
    adapter_name: str | None,
    provider_name: str | None,
    rows: Sequence[dict[str, Any]],
    config,
    target_label: str,
) -> EvalSummary:
    target = build_platform_model_target(
        base_url=base_url,
        workspace=workspace,
        model_entity=model_entity,
        adapter_name=adapter_name,
        provider_name=provider_name,
    )
    result = run_chat_online_eval(rows=rows, target=target, config=config)
    return summarize_chat_eval_result(
        target=target_label,
        model_name=target.name,
        gateway_url=target.url,
        result=result,
    )


def compare_adapters(
    *,
    base_url: str,
    workspace: str,
    model_entity: str,
    adapter_names: Sequence[str],
    rows: Sequence[dict[str, Any]],
    include_base: bool = True,
    provider_name: str | None = None,
    max_tokens: int = 64,
    enable_thinking: bool = False,
    parallelism: int = 8,
    limit_samples: int | None = None,
) -> list[EvalSummary]:
    """Compare base (optional) and one or more LoRA adapters on the same CHAT rows."""
    config = build_online_eval_config(
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        parallelism=parallelism,
        limit_samples=limit_samples,
    )
    summaries: list[EvalSummary] = []
    if include_base:
        summaries.append(
            _eval_target(
                base_url=base_url,
                workspace=workspace,
                model_entity=model_entity,
                adapter_name=None,
                provider_name=provider_name,
                rows=rows,
                config=config,
                target_label="base",
            )
        )
    for adapter_name in adapter_names:
        summaries.append(
            _eval_target(
                base_url=base_url,
                workspace=workspace,
                model_entity=model_entity,
                adapter_name=adapter_name,
                provider_name=provider_name,
                rows=rows,
                config=config,
                target_label=adapter_name,
            )
        )
    return summaries


def compare_base_vs_adapter(
    *,
    base_url: str,
    workspace: str,
    model_entity: str,
    adapter_name: str,
    rows: Sequence[dict[str, Any]],
    provider_name: str | None = None,
    max_tokens: int = 64,
    enable_thinking: bool = False,
    parallelism: int = 8,
    limit_samples: int | None = None,
) -> list[EvalSummary]:
    """Compare base model vs one LoRA adapter on the same CHAT validation rows."""
    summaries = compare_adapters(
        base_url=base_url,
        workspace=workspace,
        model_entity=model_entity,
        adapter_names=[adapter_name],
        rows=rows,
        include_base=True,
        provider_name=provider_name,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        parallelism=parallelism,
        limit_samples=limit_samples,
    )
    if len(summaries) == 2:
        summaries[1].target = "lora"
    return summaries


def lift_vs_base(summaries: Sequence[EvalSummary]) -> dict[str, float]:
    """Normalized accuracy delta vs the base summary (if present)."""
    base = next((summary for summary in summaries if summary.target == "base"), None)
    if base is None:
        return {}
    return {
        summary.target: round(summary.normalized_accuracy - base.normalized_accuracy, 4)
        for summary in summaries
        if summary.target != "base"
    }


def routing_sanity_warnings(
    summaries: Sequence[EvalSummary],
    *,
    routing_tolerance_pp: float = 0.015,
) -> list[str]:
    """Return human-readable warnings when LoRA routing or scores look suspicious."""
    warnings: list[str] = []
    base = next((summary for summary in summaries if summary.target == "base"), None)
    for summary in summaries:
        if summary.target == "base":
            if summary.gateway_path != "model-entity":
                warnings.append(
                    f"base eval used {summary.gateway_path} route; expected model-entity ({summary.gateway_url})"
                )
            continue
        if summary.gateway_path != "provider":
            warnings.append(
                f"{summary.target}: LoRA eval used {summary.gateway_path} route "
                f"({summary.gateway_url}); expected provider gateway — scores may match base"
            )
        if base and abs(summary.normalized_accuracy - base.normalized_accuracy) <= routing_tolerance_pp:
            warnings.append(
                f"{summary.target}: normalized accuracy {summary.normalized_accuracy:.1%} is within "
                f"{routing_tolerance_pp:.1%} of base ({base.normalized_accuracy:.1%}) — verify provider routing"
            )
    return warnings


def build_eval_payload(
    *,
    summaries: Sequence[EvalSummary],
    base_url: str,
    workspace: str,
    model_entity: str,
    adapter_names: Sequence[str],
    provider_name: str | None,
) -> dict[str, Any]:
    """Assemble CLI/programmatic JSON output with routing metadata and warnings."""
    routing: dict[str, Any] = {}
    if any(summary.target == "base" for summary in summaries):
        routing["base"] = {
            "gateway_path": "model-entity",
            "url": model_entity_gateway_url(base_url=base_url, workspace=workspace, model_entity=model_entity),
            "model_field": served_model_name(workspace=workspace, entity_or_adapter=model_entity, finetuning="base"),
        }
    for adapter_name in adapter_names:
        target = build_platform_model_target(
            base_url=base_url,
            workspace=workspace,
            model_entity=model_entity,
            adapter_name=adapter_name,
            provider_name=provider_name,
        )
        routing[adapter_name] = {
            "gateway_path": "provider",
            "url": target.url,
            "model_field": target.name,
        }
    warnings = routing_sanity_warnings(summaries)
    payload: dict[str, Any] = {
        "dataset_format": "chat (messages)",
        "prompt_template": CHAT_USER_PROMPT_TEMPLATE,
        "reference_template": CHAT_REFERENCE_TEMPLATE,
        "routing": routing,
        "results": [summary.to_dict() for summary in summaries],
        "lift_vs_base": lift_vs_base(summaries),
        "primary_metric": "normalized_accuracy",
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


def default_base_url() -> str:
    """Platform URL from env or localhost default."""
    return os.environ.get("NHX_BASE_URL") or "http://127.0.0.1:8080"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs LoRA on CHAT validation JSONL")
    parser.add_argument(
        "--base-url",
        default=default_base_url(),
        help="Platform URL (default: $NHX_BASE_URL or http://127.0.0.1:8080)",
    )
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--model-entity", required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="Adapter name(s) registered on the model entity (repeat for multi-adapter compare)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Inference provider name for LoRA requests (auto-discovered when omitted)",
    )
    parser.add_argument("--dataset-fileset", required=True)
    parser.add_argument("--split", default="validation.jsonl")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="Skip base-model eval (adapter-only comparison)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = load_chat_jsonl_from_platform(
        base_url=args.base_url,
        workspace=args.workspace,
        fileset=args.dataset_fileset,
        remote_path=args.split,
    )
    summaries = compare_adapters(
        base_url=args.base_url,
        workspace=args.workspace,
        model_entity=args.model_entity,
        adapter_names=args.adapter,
        rows=rows,
        include_base=not args.no_base,
        provider_name=args.provider,
        max_tokens=args.max_tokens,
        enable_thinking=args.enable_thinking,
        limit_samples=args.limit_samples,
    )
    payload = build_eval_payload(
        summaries=summaries,
        base_url=args.base_url,
        workspace=args.workspace,
        model_entity=args.model_entity,
        adapter_names=args.adapter,
        provider_name=args.provider,
    )
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
