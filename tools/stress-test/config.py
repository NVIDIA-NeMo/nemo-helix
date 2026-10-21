# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load ``sweep.yaml``. The file is the only list of stress measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from names import NameError, check_token, job_name, resource_name

Kind = Literal["exit", "idle", "hammer"]
Service = Literal["files", "secrets", "auth"]
_KINDS = ("exit", "idle", "hammer")
_SERVICES = ("files", "secrets", "auth")


class ConfigError(ValueError):
    """The sweep file or a ``--step`` value cannot be used."""


@dataclass(frozen=True)
class PodSize:
    cpu_request: str
    memory_request: str
    cpu_limit: str
    memory_limit: str


@dataclass(frozen=True)
class Step:
    id: str
    kind: Kind
    count: int
    duration_seconds: int | None
    service: Service | None
    rate_per_second: float | None

    def measurement(self) -> dict[str, object]:
        """Fields that decide whether a stored result is still current."""
        return {
            "kind": self.kind,
            "count": self.count,
            "duration_seconds": self.duration_seconds,
            "service": self.service,
            "rate_per_second": self.rate_per_second,
        }


@dataclass(frozen=True)
class Sweep:
    run_key: str
    payload_bytes: int
    pods: dict[str, PodSize]
    steps: tuple[Step, ...]

    def step(self, step_id: str) -> Step:
        found = [item for item in self.steps if item.id == step_id]
        if not found:
            known = ", ".join(item.id for item in self.steps)
            raise ConfigError(f"unknown step {step_id!r}; configured steps: {known}")
        return found[0]


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{label} must be a positive integer")
    return value


def _pod(value: object, kind: str) -> PodSize:
    raw = _mapping(value, f"pods.{kind}")
    return PodSize(
        cpu_request=_text(raw.get("cpu_request"), f"pods.{kind}.cpu_request"),
        memory_request=_text(raw.get("memory_request"), f"pods.{kind}.memory_request"),
        cpu_limit=_text(raw.get("cpu_limit"), f"pods.{kind}.cpu_limit"),
        memory_limit=_text(raw.get("memory_limit"), f"pods.{kind}.memory_limit"),
    )


def _kind(value: object) -> Kind:
    if value == "exit":
        return "exit"
    if value == "idle":
        return "idle"
    if value == "hammer":
        return "hammer"
    raise ConfigError(f"kind must be one of {', '.join(_KINDS)}")


def _service(value: object) -> Service:
    if value == "files":
        return "files"
    if value == "secrets":
        return "secrets"
    if value == "auth":
        return "auth"
    raise ConfigError(f"service must be one of {', '.join(_SERVICES)}")


def _step(value: object, seen: set[str]) -> Step:
    raw = _mapping(value, "step")
    step_id = _text(raw.get("id"), "step id")
    check_token(step_id, "step id")
    if step_id in seen:
        raise ConfigError(f"duplicate step id {step_id!r}")
    seen.add(step_id)
    kind = _kind(raw.get("kind"))
    count = _positive_int(raw.get("count"), f"{step_id} count")
    duration: int | None = None
    service: Service | None = None
    rate: float | None = None
    if kind in ("idle", "hammer"):
        duration = _positive_int(raw.get("duration_seconds"), f"{step_id} duration_seconds")
    if kind == "hammer":
        service = _service(raw.get("service"))
        rate_value = raw.get("rate_per_second")
        if isinstance(rate_value, bool) or not isinstance(rate_value, (int, float)) or rate_value <= 0:
            raise ConfigError(f"{step_id} rate_per_second must be a positive number")
        rate = float(rate_value)
    return Step(
        id=step_id,
        kind=kind,
        count=count,
        duration_seconds=duration,
        service=service,
        rate_per_second=rate,
    )


def load_sweep(path: Path) -> Sweep:
    """Parse a sweep file and reject names that cannot be submitted."""
    try:
        return _load_sweep(path)
    except NameError as exc:
        raise ConfigError(str(exc)) from exc


def _load_sweep(path: Path) -> Sweep:
    loaded = yaml.safe_load(path.read_text())
    raw = _mapping(loaded, "sweep")
    run_key = _text(raw.get("run_key"), "run_key")
    check_token(run_key, "run key")
    resource_name(run_key)
    payload_bytes = _positive_int(raw.get("payload_bytes"), "payload_bytes")
    pods_raw = _mapping(raw.get("pods"), "pods")
    pods = {kind: _pod(pods_raw.get(kind), kind) for kind in _KINDS}
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ConfigError("steps must be a non-empty list")
    seen: set[str] = set()
    steps = tuple(_step(item, seen) for item in steps_raw)
    for step in steps:
        for index in range(1, step.count + 1):
            job_name(run_key, step.id, index, step.count)
    return Sweep(run_key=run_key, payload_bytes=payload_bytes, pods=pods, steps=steps)


def override_run_key(sweep: Sweep, run_key: str | None) -> Sweep:
    """Use a CLI run key when one is provided."""
    if run_key is None or run_key == sweep.run_key:
        return sweep
    check_token(run_key, "run key")
    resource_name(run_key)
    for step in sweep.steps:
        for index in range(1, step.count + 1):
            job_name(run_key, step.id, index, step.count)
    return Sweep(run_key=run_key, payload_bytes=sweep.payload_bytes, pods=sweep.pods, steps=sweep.steps)


def select_step(sweep: Sweep, step_id: str | None) -> Step | None:
    """Return the named step, or None when ``--step`` is omitted."""
    if step_id is None:
        return None
    return sweep.step(step_id)
