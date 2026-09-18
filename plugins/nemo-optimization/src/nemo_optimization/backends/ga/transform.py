# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt mutation and recombination through Platform inference."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol

from nemo_platform import NeMoPlatform
from nemo_platform_plugin.entities.base import parse_qualified_name


class PromptTransformError(RuntimeError):
    """Raised when a prompt transform cannot be produced."""


class PromptTransformer(Protocol):
    def mutate(
        self,
        *,
        prompt_name: str,
        prompt: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str: ...

    def recombine(
        self,
        *,
        prompt_name: str,
        parent_a: str,
        parent_b: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str: ...


class ModelPromptTransformer:
    """Invoke the configured optimizer model through Platform's inference gateway."""

    def __init__(
        self,
        *,
        sdk: NeMoPlatform | None,
        workspace: str,
        payload: Mapping[str, Any],
        model_name: str,
    ) -> None:
        if sdk is None:
            raise PromptTransformError("Prompt GA optimization requires a NeMo Platform SDK client.")
        declaration = _model_declaration(payload, model_name)
        target = declaration.get("model") or declaration.get("model_name")
        if not isinstance(target, str) or not target.strip():
            raise PromptTransformError(f"Prompt optimizer model {model_name!r} must declare a model name.")

        self._sdk = sdk
        self._workspace, self._model = parse_qualified_name(target.strip(), default_workspace=workspace)
        self._model_ref = f"{self._workspace}/{self._model}"
        self._settings = _model_settings(declaration, model_name)

    def mutate(
        self,
        *,
        prompt_name: str,
        prompt: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        return self._complete(
            _mutation_user_prompt(
                prompt_name=prompt_name,
                prompt=prompt,
                purpose=purpose,
                prompt_format=prompt_format,
                feedback=feedback,
            )
        )

    def recombine(
        self,
        *,
        prompt_name: str,
        parent_a: str,
        parent_b: str,
        purpose: str,
        prompt_format: str | None,
        feedback: str | None,
    ) -> str:
        return self._complete(
            _recombination_user_prompt(
                prompt_name=prompt_name,
                parent_a=parent_a,
                parent_b=parent_b,
                purpose=purpose,
                prompt_format=prompt_format,
                feedback=feedback,
            )
        )

    def _complete(self, user_prompt: str) -> str:
        body: dict[str, object] = {
            "model": self._model_ref,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the revised prompt text, without commentary or markdown fences.",
                },
                {"role": "user", "content": user_prompt},
            ],
        }
        body.update({key: self._settings[key] for key in ("temperature", "max_tokens") if key in self._settings})
        try:
            response = self._sdk.inference.gateway.model.post(
                "v1/chat/completions",
                workspace=self._workspace,
                name=self._model,
                body=body,
                timeout=self._settings.get("timeout_s", 60.0),
            )
        except Exception as exc:
            raise PromptTransformError(f"Prompt optimizer model call failed: {exc}") from exc

        if not isinstance(response, Mapping):
            raise PromptTransformError("Prompt optimizer model returned a non-object response.")
        content = _extract_chat_content(response)
        if not content:
            raise PromptTransformError("Prompt optimizer model returned an empty prompt.")
        return content


def _model_declaration(payload: Mapping[str, Any], model_name: str) -> Mapping[str, Any]:
    models = payload.get("models")
    declaration = models.get(model_name) if isinstance(models, Mapping) else None
    if not isinstance(declaration, Mapping):
        raise PromptTransformError(f"Prompt optimizer model {model_name!r} not found under payload.models.")
    return declaration


def _model_settings(declaration: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    nested = declaration.get("settings", {})
    if not isinstance(nested, Mapping):
        raise PromptTransformError(f"Prompt optimizer model {model_name!r} settings must be a mapping.")
    settings = dict(nested)
    for key in ("temperature", "max_tokens", "timeout_s"):
        if key in declaration:
            settings.setdefault(key, declaration[key])
    return settings


def _extract_chat_content(payload: Mapping[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return ""
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, Mapping) else choices[0].get("text")
    if not isinstance(content, str):
        return ""
    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else stripped


def _mutation_user_prompt(
    *,
    prompt_name: str,
    prompt: str,
    purpose: str,
    prompt_format: str | None,
    feedback: str | None,
) -> str:
    sections = [
        f"Prompt dimension: {prompt_name}",
        f"Purpose: {purpose}",
        f"Required format: {prompt_format or 'Preserve the current prompt format.'}",
        "Current prompt:",
        prompt,
    ]
    if feedback:
        sections.extend(["Evaluation feedback to address:", feedback])
    sections.append("Rewrite the prompt while preserving its variables, role, and constraints.")
    return "\n\n".join(sections)


def _recombination_user_prompt(
    *,
    prompt_name: str,
    parent_a: str,
    parent_b: str,
    purpose: str,
    prompt_format: str | None,
    feedback: str | None,
) -> str:
    sections = [
        f"Prompt dimension: {prompt_name}",
        f"Purpose: {purpose}",
        f"Required format: {prompt_format or 'Preserve the current prompt format.'}",
        "Parent prompt A:",
        parent_a,
        "Parent prompt B:",
        parent_b,
    ]
    if feedback:
        sections.extend(["Evaluation feedback to address:", feedback])
    sections.append("Combine the strongest parts of both parents while preserving variables and constraints.")
    return "\n\n".join(sections)


__all__ = ["ModelPromptTransformer", "PromptTransformError", "PromptTransformer"]
