# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON schemas for native libsy algorithms (stage_router, llm_classifier, random)."""

from __future__ import annotations

from typing import Any

from nemo_platform_plugin.inference_middleware import InferenceMiddlewareError
from nemo_switchyard._native_availability import load_libsy, native_rust_available

NATIVE_CONFIG_TYPES = frozenset({"stage_router", "llm_classifier"})

_NATIVE_UNAVAILABLE = (
    "config_type {config_type!r} requires the native Switchyard bindings "
    "(switchyard_rust). They are not installed in this process. "
    "Keep random_routing and translate on the May vendor, or run an isolated "
    "venv that has switchyard_rust and does not install switchyard-vendored."
)


def require_native_rust(config_type: str) -> None:
    if native_rust_available():
        return
    raise InferenceMiddlewareError(
        _NATIVE_UNAVAILABLE.format(config_type=config_type),
        status_code=400,
    )


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InferenceMiddlewareError(f"{field} must be an object", status_code=400)
    return value


def _model_names(raw: Any, field: str) -> list[str]:
    if isinstance(raw, str) and raw:
        return [raw]
    if isinstance(raw, list) and raw and all(isinstance(item, str) and item for item in raw):
        return [str(item) for item in raw]
    raise InferenceMiddlewareError(
        f"{field} must be a non-empty string or list of strings",
        status_code=400,
    )


def category_models(config: dict[str, Any], category: str, *, required: bool) -> list[str]:
    models = _require_mapping(config.get("models") or {}, "models")
    if category not in models:
        if required:
            raise InferenceMiddlewareError(
                f"models.{category} is required",
                status_code=400,
            )
        return []
    return _model_names(models[category], f"models.{category}")


def models_map_from_config(config: dict[str, Any], *, required: tuple[str, ...]) -> dict[str, list[str]]:
    mapping = {key: category_models(config, key, required=True) for key in required}
    extra = _require_mapping(config.get("models") or {}, "models")
    for key, raw in extra.items():
        if key not in mapping:
            mapping[key] = _model_names(raw, f"models.{key}")
    if "any" not in mapping:
        serving = [names for key, names in mapping.items() if key != "judge"]
        mapping["any"] = list(dict.fromkeys(name for names in serving for name in names))
    return mapping


def _require_unit_interval(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise InferenceMiddlewareError(f"{field} must be in [0, 1]", status_code=400)
    return float(value)


def map_random_routing_config(config: dict[str, Any]) -> tuple[list[float], int | None, dict[str, list[str]]]:
    """Map May-shaped random_routing JSON onto libsy ``random`` + ``models.any``.

    A weight of 0 is never selected.
    """
    strong = _require_mapping(config.get("strong"), "strong")
    weak = _require_mapping(config.get("weak"), "weak")
    strong_id = strong.get("model")
    weak_id = weak.get("model")
    if not isinstance(strong_id, str) or not strong_id:
        raise InferenceMiddlewareError("strong.model is required", status_code=400)
    if not isinstance(weak_id, str) or not weak_id:
        raise InferenceMiddlewareError("weak.model is required", status_code=400)
    strong_p = _require_unit_interval(config.get("strong_probability"), "strong_probability")
    seed = config.get("rng_seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise InferenceMiddlewareError("rng_seed must be an integer when set", status_code=400)
    return [strong_p, 1.0 - strong_p], seed, {"any": [strong_id, weak_id]}


def validate_stage_router_config(config: dict[str, Any]) -> dict[str, Any]:
    picker = config.get("picker", "efficient_first")
    if picker not in {"efficient_first", "capable_first"}:
        raise InferenceMiddlewareError(
            "picker must be 'efficient_first' or 'capable_first'",
            status_code=400,
        )
    threshold = _require_unit_interval(config.get("confidence_threshold"), "confidence_threshold")
    notes = config.get("handoff_notes") or {}
    if notes.get("deescalation_note") and not notes.get("escalation_note"):
        raise InferenceMiddlewareError("deescalation_note requires escalation_note", status_code=400)
    models_map_from_config(config, required=("capable", "efficient"))
    return {**config, "picker": picker, "confidence_threshold": threshold}


def validate_llm_classifier_config(config: dict[str, Any]) -> dict[str, Any]:
    mode = config.get("mode", "capability")
    if mode != "capability":
        raise InferenceMiddlewareError(
            "llm_classifier only supports mode='capability' in this release",
            status_code=400,
        )
    threshold = _require_unit_interval(config.get("base_threshold"), "base_threshold")
    session_affinity = bool(config.get("session_affinity", False))
    message_hash_fallback = bool(config.get("message_hash_fallback", False))
    if message_hash_fallback and not session_affinity:
        raise InferenceMiddlewareError(
            "message_hash_fallback requires session_affinity=true (classify_trigger=new_session)",
            status_code=400,
        )
    models_map_from_config(config, required=("judge", "capable", "efficient"))
    return {
        **config,
        "mode": mode,
        "base_threshold": threshold,
        "session_affinity": session_affinity,
        "message_hash_fallback": message_hash_fallback,
    }


def native_model_categories(config_type: str) -> tuple[str, ...]:
    if config_type == "stage_router":
        return ("capable", "efficient")
    return ("judge", "capable", "efficient")


def validate_native_config(config_type: str, config: Any) -> dict[str, Any]:
    require_native_rust(config_type)
    payload = _require_mapping(config or {}, "config")
    if config_type == "stage_router":
        return validate_stage_router_config(payload)
    if config_type == "llm_classifier":
        return validate_llm_classifier_config(payload)
    raise InferenceMiddlewareError(f"Unknown native config_type {config_type!r}", status_code=400)


def build_native_algorithm(config_type: str, config: dict[str, Any]) -> Any:
    """Construct a libsy Algorithm. Imports switchyard_rust only here."""
    libsy = load_libsy()
    if config_type == "stage_router":
        return _build_stage_router(libsy, config)
    if config_type == "llm_classifier":
        return _build_llm_classifier(libsy, config)
    if config_type == "random_routing":
        weights, seed, _models = map_random_routing_config(config)
        return libsy.random(weights=weights, seed=seed)
    raise InferenceMiddlewareError(f"Cannot build native algorithm for {config_type!r}", status_code=400)


def _build_stage_router(libsy: Any, config: dict[str, Any]) -> Any:
    classifier = None
    raw = config.get("classifier")
    if raw is not None:
        classifier_cfg = _require_mapping(raw, "classifier")
        threshold = classifier_cfg.get("base_threshold", 0.5)
        classifier = libsy.LlmFallback(
            config=libsy.TaskClassifierConfig(
                float(threshold),
                threshold_step=float(classifier_cfg.get("threshold_step", 0.0)),
                recent_turn_window=classifier_cfg.get("recent_turn_window"),
                prompt=classifier_cfg.get("prompt"),
            )
        )
    return libsy.stage_router(
        picker=config.get("picker", "efficient_first"),
        confidence_threshold=float(config["confidence_threshold"]),
        recent_window=config.get("recent_window"),
        classifier=classifier,
        tool_semantics=config.get("tool_semantics"),
        escalation_note=(config.get("handoff_notes") or {}).get("escalation_note"),
        deescalation_note=(config.get("handoff_notes") or {}).get("deescalation_note"),
    )


def _build_llm_classifier(libsy: Any, config: dict[str, Any]) -> Any:
    task = libsy.TaskClassifierConfig(
        float(config["base_threshold"]),
        threshold_step=float(config.get("threshold_step", 0.0)),
        session_affinity=bool(config.get("session_affinity", False)),
        message_hash_fallback=bool(config.get("message_hash_fallback", False)),
        recent_turn_window=config.get("recent_turn_window"),
        prompt=config.get("prompt"),
        response_format_type=config.get("response_format_type", "json_schema"),
    )
    return libsy.llm_classifier(libsy.LlmClassifierConfig.capability(config=task))
