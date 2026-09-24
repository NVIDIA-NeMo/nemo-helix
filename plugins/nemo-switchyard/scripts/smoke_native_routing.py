#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["cyclopts>=4.0", "pydantic>=2.0"]
# ///
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native Switchyard routing through a running NeMo Inference Gateway."""

import json
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from cyclopts import App
from pydantic import BaseModel, Field

app = App(help=__doc__)

DEFAULT_STRONG = "default/nvidia-nemotron-3-super-120b-a12b"
DEFAULT_WEAK = "default/nvidia-nemotron-3-5-lightning-30b-a3b"


class CommandResult(BaseModel):
    """Captured CLI command result."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def display(self) -> str:
        """Return a shell-readable command."""
        return shlex.join([Path(self.argv[0]).name, *self.argv[1:]])


class ModelRef(BaseModel):
    """Model entity and upstream served-model name."""

    entity_id: str
    served_name: str

    @property
    def response_names(self) -> set[str]:
        """Names an OpenAI response may use for this model."""
        return {self.entity_id, self.entity_id.split("/", 1)[-1], self.served_name}


class CaseResult(BaseModel):
    """One smoke-test result rendered in the stakeholder report."""

    name: str
    passed: bool
    status: str
    models: list[str] = Field(default_factory=list)
    note: str
    commands: list[str] = Field(default_factory=list)
    detail: str = ""


class SmokeHarness:
    """Create temporary routers, invoke them, and collect evidence."""

    def __init__(
        self,
        *,
        nemo: str,
        workspace: str,
        prefix: str,
        keep: bool,
    ) -> None:
        self.nemo = nemo
        self.workspace = workspace
        self.prefix = prefix
        self.keep = keep
        self.created: list[str] = []

    def command(self, *args: str, check: bool = False) -> CommandResult:
        """Run one NeMo CLI command with JSON output configured."""
        argv = [self.nemo, *args]
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        result = CommandResult(
            argv=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if check and result.returncode:
            raise RuntimeError(f"{result.display}\n{result.stderr or result.stdout}")
        return result

    def create(
        self, suffix: str, *, models: list[ModelRef], request: list[dict], response: list[dict] | None = None
    ) -> CommandResult:
        """Create a temporary VirtualModel."""
        name = f"{self.prefix}-{suffix}"
        result = self.command(
            "inference",
            "virtual-models",
            "create",
            name,
            "--workspace",
            self.workspace,
            "--models",
            json.dumps([{"model": model.entity_id, "backend_format": "OPENAI_CHAT"} for model in models]),
            "--request-middleware",
            json.dumps(request),
            "--response-middleware",
            json.dumps(response or []),
            "--output",
            "json",
        )
        if result.returncode == 0:
            self.created.append(name)
        return result

    def invoke(
        self, suffix: str, *, path: str = "v1/chat/completions", prompt: str = "Reply with only: OK"
    ) -> CommandResult:
        """Invoke a temporary VirtualModel."""
        name = f"{self.prefix}-{suffix}"
        body = {
            "model": f"{self.workspace}/{name}",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
        }
        return self.command(
            "inference",
            "gateway",
            "model",
            "post",
            path,
            name,
            "--workspace",
            self.workspace,
            "--body",
            json.dumps(body),
            "--output",
            "json",
        )

    def invoke_with_retry(
        self, suffix: str, *, attempts: int = 10, prompt: str = "Reply with only: OK"
    ) -> CommandResult:
        """Wait for the IGW model cache to observe a newly-created router."""
        result = self.invoke(suffix, prompt=prompt)
        for _ in range(attempts - 1):
            if result.returncode == 0:
                return result
            time.sleep(1)
            result = self.invoke(suffix, prompt=prompt)
        return result

    def cleanup(self) -> None:
        """Delete only VirtualModels created by this run."""
        if self.keep:
            return
        for name in reversed(self.created):
            self.command("inference", "virtual-models", "delete", name, "--workspace", self.workspace)


def parse_json(text: str) -> Any:
    """Parse CLI JSON, tolerating raw control characters from reasoning models."""
    stripped = text.strip()
    try:
        return json.loads(stripped, strict=False)
    except json.JSONDecodeError:
        start = min(index for index in (stripped.find("{"), stripped.find("[")) if index >= 0)
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        return json.loads(stripped[start : end + 1], strict=False)


def response_model(result: CommandResult) -> str:
    """Extract the upstream model name from a successful completion."""
    payload = parse_json(result.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
        raise ValueError("completion did not contain a string model field")
    return payload["model"]


def failure_status(result: CommandResult) -> str:
    """Extract an HTTP status from CLI error text when available."""
    match = re.search(r"\bHTTP\s+(\d{3})\b|\bstatus(?:_code)?[=: ]+(\d{3})\b", result.stderr, re.IGNORECASE)
    return f"HTTP {next(group for group in match.groups() if group)}" if match else f"exit {result.returncode}"


def discover_models(harness: SmokeHarness, provider_name: str) -> dict[str, ModelRef]:
    """Return model entities advertised by one provider."""
    result = harness.command(
        "inference",
        "providers",
        "list",
        "--workspace",
        harness.workspace,
        "--all-pages",
        "--output",
        "json",
        check=True,
    )
    payload = parse_json(result.stdout)
    provider = next((item for item in payload["data"] if item["name"] == provider_name), None)
    if provider is None:
        raise ValueError(f"provider {provider_name!r} was not found in workspace {harness.workspace!r}")
    return {
        item["model_entity_id"]: ModelRef(entity_id=item["model_entity_id"], served_name=item["served_model_name"])
        for item in provider.get("served_models", [])
    }


def choose_model(models: dict[str, ModelRef], requested: str | None, preferred: str, label: str) -> ModelRef:
    """Resolve an explicit or preferred model entity."""
    entity_id = requested or preferred
    try:
        return models[entity_id]
    except KeyError as exc:
        available = ", ".join(sorted(models)[:10])
        raise ValueError(f"{label} model {entity_id!r} is unavailable; first discovered models: {available}") from exc


def middleware(config_type: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a Switchyard request middleware list."""
    return [{"name": "nemo-switchyard", "config_type": config_type, "config": config}]


def random_config(strong: ModelRef, weak: ModelRef, probability: float, seed: int) -> dict[str, Any]:
    """Build deterministic random-routing configuration."""
    return {
        "strong": {"model": strong.entity_id},
        "weak": {"model": weak.entity_id},
        "strong_probability": probability,
        "rng_seed": seed,
    }


def run_random_case(
    harness: SmokeHarness,
    *,
    suffix: str,
    strong: ModelRef,
    weak: ModelRef,
    probability: float,
    samples: int,
) -> CaseResult:
    """Create and exercise one random router."""
    create = harness.create(
        suffix,
        models=[strong, weak],
        request=middleware("random_routing", random_config(strong, weak, probability, 42)),
    )
    commands = [create.display]
    if create.returncode:
        return CaseResult(
            name=suffix,
            passed=False,
            status=failure_status(create),
            note="VirtualModel creation failed",
            commands=commands,
            detail=create.stderr,
        )
    calls = [harness.invoke_with_retry(suffix) if index == 0 else harness.invoke(suffix) for index in range(samples)]
    commands.extend(call.display for call in calls)
    if any(call.returncode for call in calls):
        failed = next(call for call in calls if call.returncode)
        return CaseResult(
            name=suffix,
            passed=False,
            status=failure_status(failed),
            note="Completion failed",
            commands=commands,
            detail=failed.stderr,
        )
    selected = [response_model(call) for call in calls]
    expected = (
        strong.response_names
        if probability == 1.0
        else weak.response_names
        if probability == 0.0
        else strong.response_names | weak.response_names
    )
    passed = all(model in expected for model in selected)
    if probability == 0.5:
        passed = passed and bool(set(selected) & strong.response_names) and bool(set(selected) & weak.response_names)
    return CaseResult(
        name=suffix,
        passed=passed,
        status="HTTP 200",
        models=selected,
        note=f"Observed split: {dict(Counter(selected))}",
        commands=commands,
    )


def run_stage_case(harness: SmokeHarness, strong: ModelRef, weak: ModelRef) -> CaseResult:
    """Exercise native stage routing without a nested classifier."""
    suffix = "stage"
    config = {
        "picker": "efficient_first",
        "confidence_threshold": 0.5,
        "models": {"capable": [strong.entity_id], "efficient": [weak.entity_id]},
    }
    return run_single_success(
        harness,
        suffix=suffix,
        models=[strong, weak],
        request=middleware("stage_router", config),
        allowed=strong.response_names | weak.response_names,
        note="Proves native routing through IGW; it does not assert a complete escalation policy.",
    )


def run_classifier_case(harness: SmokeHarness, strong: ModelRef, weak: ModelRef, judge: ModelRef) -> CaseResult:
    """Exercise capability classification with a live provider-direct judge."""
    suffix = "classifier"
    config = {
        "mode": "capability",
        "base_threshold": 0.5,
        "max_output_tokens": 256,
        "models": {
            "judge": [judge.entity_id],
            "capable": [strong.entity_id],
            "efficient": [weak.entity_id],
        },
    }
    note = "Judge HTTP is provider-direct using cached provider credentials; caller authorization is not forwarded."
    return run_single_success(
        harness,
        suffix=suffix,
        models=[strong, weak],
        request=middleware("llm_classifier", config),
        allowed=strong.response_names | weak.response_names,
        note=note,
        prompt="Classify and answer: what is 17 times 19?",
    )


def run_single_success(
    harness: SmokeHarness,
    *,
    suffix: str,
    models: list[ModelRef],
    request: list[dict[str, Any]],
    allowed: set[str],
    note: str,
    prompt: str = "Reply with only: OK",
) -> CaseResult:
    """Create a router and require one successful routed completion."""
    create = harness.create(suffix, models=models, request=request)
    commands = [create.display]
    if create.returncode:
        return CaseResult(
            name=suffix,
            passed=False,
            status=failure_status(create),
            note="VirtualModel creation failed",
            commands=commands,
            detail=create.stderr,
        )
    call = harness.invoke_with_retry(suffix, prompt=prompt)
    commands.append(call.display)
    if call.returncode:
        return CaseResult(
            name=suffix, passed=False, status=failure_status(call), note=note, commands=commands, detail=call.stderr
        )
    selected = response_model(call)
    return CaseResult(
        name=suffix, passed=selected in allowed, status="HTTP 200", models=[selected], note=note, commands=commands
    )


def run_rejection_cases(harness: SmokeHarness, strong: ModelRef, weak: ModelRef) -> list[CaseResult]:
    """Verify documented unsupported configurations fail closed."""
    base = middleware("random_routing", random_config(strong, weak, 1.0, 7))
    response = harness.create("reject-response", models=[strong, weak], request=[], response=base)
    translate = harness.create("reject-translate", models=[strong, weak], request=middleware("translate", {}))
    create = harness.create("reject-path", models=[strong, weak], request=base)
    path = harness.invoke("reject-path", path="v1/completions") if create.returncode == 0 else create
    return [
        rejected("reject-response", response, "Response middleware registration is rejected."),
        rejected("reject-translate", translate, "Legacy translate configuration is rejected."),
        rejected("reject-path", path, "Non-chat-completions path is rejected.", prefix=create),
    ]


def rejected(name: str, result: CommandResult, note: str, *, prefix: CommandResult | None = None) -> CaseResult:
    """Convert an expected CLI failure into a smoke result."""
    commands = [prefix.display] if prefix is not None else []
    commands.append(result.display)
    return CaseResult(
        name=name,
        passed=result.returncode != 0,
        status=failure_status(result) if result.returncode else "HTTP 200",
        note=note,
        commands=commands,
        detail=(result.stderr or result.stdout) if result.returncode == 0 else "",
    )


def package_version(name: str) -> str:
    """Read a package version without failing the smoke."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed in script interpreter"


def git_sha() -> str:
    """Return the repository HEAD containing this script."""
    root = Path(__file__).resolve().parents[3]
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def git_dirty() -> bool:
    """Return whether the repository has uncommitted changes."""
    root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(status.stdout.strip())


def render_report(
    *,
    platform_url: str,
    workspace: str,
    provider: str,
    strong: ModelRef,
    weak: ModelRef,
    judge: ModelRef,
    results: list[CaseResult],
) -> str:
    """Render a portable Markdown report."""
    passed = sum(result.passed for result in results)
    lines = [
        "# Native Switchyard routing smoke report",
        "",
        f"- Result: **{passed}/{len(results)} passed**",
        f"- Generated: `{datetime.now(UTC).isoformat(timespec='seconds')}`",
        f"- Platform: `{platform_url}`",
        f"- Source SHA: `{git_sha()}` (working tree {'dirty' if git_dirty() else 'clean'})",
        f"- Workspace / provider: `{workspace}` / `{provider}`",
        f"- Models: strong `{strong.entity_id}`, weak `{weak.entity_id}`, judge `{judge.entity_id}`",
        f"- Packages: `nemo-switchyard-plugin={package_version('nemo-switchyard-plugin')}`, `nemo-switchyard={package_version('nemo-switchyard')}`",
        "",
        "| Case | Result | Status | Routed model(s) | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        models = ", ".join(result.models) or "—"
        lines.append(
            f"| `{result.name}` | {'PASS' if result.passed else 'FAIL'} | {result.status} | {models} | {result.note} |"
        )
    lines.extend(["", "## Commands and failure details", ""])
    for result in results:
        lines.extend([f"### {result.name}", "", "```console", *[f"$ {command}" for command in result.commands]])
        if result.detail:
            lines.append(result.detail.strip())
        lines.extend(["```", ""])
    return "\n".join(lines)


@app.default
def main(
    *,
    workspace: str = "default",
    provider: str = "nvidia-build",
    strong_model: str | None = None,
    weak_model: str | None = None,
    judge_model: str | None = None,
    samples: int = 10,
    nemo: str = "nemo",
    platform_url: str = "http://localhost:8080",
    keep: bool = False,
    output: Path = Path("switchyard-routing-smoke-report.md"),
) -> None:
    """Run the native routing smoke and write a Markdown report."""
    prefix = f"switchyard-smoke-{int(time.time())}"
    harness = SmokeHarness(nemo=nemo, workspace=workspace, prefix=prefix, keep=keep)
    results: list[CaseResult] = []
    try:
        discovered = discover_models(harness, provider)
        strong = choose_model(discovered, strong_model, DEFAULT_STRONG, "strong")
        weak = choose_model(discovered, weak_model, DEFAULT_WEAK, "weak")
        judge = choose_model(discovered, judge_model, weak.entity_id, "judge")
        results.extend(
            [
                run_random_case(harness, suffix="random-strong", strong=strong, weak=weak, probability=1.0, samples=3),
                run_random_case(harness, suffix="random-weak", strong=strong, weak=weak, probability=0.0, samples=3),
                run_random_case(
                    harness, suffix="random-split", strong=strong, weak=weak, probability=0.5, samples=samples
                ),
                run_stage_case(harness, strong, weak),
                run_classifier_case(harness, strong, weak, judge),
                *run_rejection_cases(harness, strong, weak),
            ]
        )
        report = render_report(
            platform_url=platform_url,
            workspace=workspace,
            provider=provider,
            strong=strong,
            weak=weak,
            judge=judge,
            results=results,
        )
        output.write_text(report, encoding="utf-8")
        sys.stdout.write(f"{report}\n")
    finally:
        harness.cleanup()
    if not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    app()
