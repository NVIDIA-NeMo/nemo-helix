# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging
import time
from pathlib import Path
from typing import List, Optional

import fsspec
from fastapi import FastAPI
from nhx.common.observability.otel import settings as otel_settings
from nhx.guardrails.app.handlers.utils import update_models_in_config
from nhx.guardrails.app.services.configs.sources import get_config
from nhx.guardrails.app.services.rails.registry import RailsRegistry
from nhx.guardrails.app.utils.model_routing import resolve_model_entity_references
from nhx.guardrails.config import settings
from nhx.guardrails.entities import GuardrailConfig
from nhx.guardrails.entities.values._private import Model, RailsConfig, TracingConfig
from opentelemetry import trace

log = logging.getLogger(__name__)

app = FastAPI()

rails_registry = RailsRegistry()


def extract_guardrails_models(guardrail_config: GuardrailConfig) -> List[str]:
    """Extract all models from a GuardrailConfig.

    Args:
        guardrail_config: The GuardrailConfig object to extract models from

    Returns:
        List[str]: List of models. Returns empty list if configuration contains no models.
    """
    if not guardrail_config.data or not guardrail_config.data.models:
        return []  # No models to extract

    models = []
    for model_config in guardrail_config.data.models:
        if model_config.model:
            models.append(model_config.model)

    return models


def enrich_config_with_data(guardrail_config: GuardrailConfig) -> GuardrailConfig:
    """
    Enriches a GuardrailConfig by populating its `data` with the RailsConfig loaded from its `files_url`, if present.
    Otherwise, returns the original object.

    Args:
        config_obj: The GuardrailConfig object to enrich with data

    Returns:
        The GuardrailConfig with data field populated, or the original config if no enrichment needed
    """
    files_url = getattr(guardrail_config, "files_url", None)
    if files_url:
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("enrich_config_with_data") as span:
            span.set_attribute("config.id", str(guardrail_config.id))
            span.set_attribute("config.workspace", guardrail_config.workspace)
            span.set_attribute("config.name", guardrail_config.name)
            span.set_attribute("config.files_url", str(files_url))

            start_time = time.time()

            try:
                rails_config = get_config(files_url=str(files_url))

                enrichment_time = time.time() - start_time
                span.set_attribute("enrichment.duration_ms", enrichment_time * 1000)
                span.set_attribute("enrichment.success", True)

                # Use model_copy to preserve private attributes (_id, _created_at, _updated_at)
                return guardrail_config.model_copy(update={"data": rails_config})

            except Exception as e:
                enrichment_time = time.time() - start_time
                span.set_attribute("enrichment.duration_ms", enrichment_time * 1000)
                span.set_attribute("enrichment.success", False)
                span.set_attribute("enrichment.error", str(e))

                log.warning(f"Failed to load data for config {guardrail_config.id} from files_url {files_url}: {e}")
                # Return the original config object if loading fails
                return guardrail_config

    return guardrail_config


def _load_and_execute_py_config(filepath: str):
    """Load a Python configuration file and execute its init function if it exists."""

    filename = Path(filepath).stem
    with fsspec.open(filepath):
        spec = importlib.util.spec_from_file_location(filename, filepath)
        if spec is not None:
            if spec.loader is not None:
                config_module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(config_module)
                except Exception as e:
                    log.error(f"Error loading config file {filepath}: {e}")

                if config_module is not None and hasattr(config_module, "init"):
                    config_module.init(app)
            else:
                log.error(f"Loader is None for spec of file {filepath}")
        else:
            raise ValueError(f"Could not load config file {filepath}")


def get_path_to_py_configs(files_url: str) -> List[str]:
    py_config_files = []
    fs, _ = fsspec.core.url_to_fs(files_url, **settings.storage_options)

    if fs.isdir(files_url):
        # find all config.py files at files_url
        py_config_files = fs.glob(f"{files_url}/**/config.py")
    return py_config_files


def invalidate_and_reload_config_cache(rails_registry: RailsRegistry, config_id: str, files_url: Optional[str] = None):
    """
    Invalidate the cache for a specific configuration and reload it.
    This replicates the logic from the on_any_event handler.
    """
    log.info(f"Configuration change detected for config_id: {config_id}. Attempting to invalidate cache.")

    rails_registry.invalidate_config_cache([config_id])

    # in delete we don't have files_url
    if files_url:
        py_config_files = get_path_to_py_configs(files_url)

        for config_file in py_config_files:
            _load_and_execute_py_config(config_file)
            log.info(f"Configuration '{config_id}' reloaded successfully.")


def configure_rails_config(rails_config: RailsConfig, model: Model) -> RailsConfig:
    """
    Updates and returns the given RailsConfig object with its model and tracing config.

    Args:
        rails_config: The RailsConfig object to configure
        model: The Model object to set as the main model

    Returns:
        The RailsConfig object with model, base URL, and tracing configured
    """
    # Update the config's main model
    rails_config = update_models_in_config(rails_config, model)

    # Resolve Model Entity references to Inference Gateway URLs for all models
    rails_config = resolve_model_entity_references(rails_config)

    # Disable tracing if telemetry is disabled globally
    if otel_settings.otel_sdk_disabled:
        rails_config.tracing = TracingConfig(enabled=False)

    return rails_config
