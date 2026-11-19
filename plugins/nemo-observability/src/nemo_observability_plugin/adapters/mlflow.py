# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLflow (Apache-2.0) tracking server."""

from __future__ import annotations

import base64
from typing import Any

from nemo_observability_plugin.adapters.base import ProviderAdapter
from nemo_observability_plugin.config import ObservabilityConnection

_DEFAULT_EXPERIMENT_ID = "0"


class MLflowAdapter(ProviderAdapter):
    provider = "mlflow"
    display_name = "MLflow"
    # `mlflow server --static-prefix` registers every route, UI and API, under the prefix.
    strip_prefix = False
    health_path = "/health"

    def landing_path(self, connection: ObservabilityConnection) -> str:
        experiment = connection.default_project or _DEFAULT_EXPERIMENT_ID
        return f"/#/experiments/{experiment}/traces"

    def upstream_headers(self, credential: str | None) -> dict[str, str]:
        if not credential:
            return {}
        # MLflow basic-auth app: the secret is "user:password".
        return {"Authorization": "Basic " + base64.b64encode(credential.encode()).decode()}

    def otlp_endpoint(self, connection: ObservabilityConnection) -> dict[str, Any]:
        return {
            "type": "gen_ai",
            "transport": "http_binary",
            "endpoint": connection.otlp_endpoint or f"{connection.base_url}/v1/traces",
            "headers": {"x-mlflow-experiment-id": connection.default_project or _DEFAULT_EXPERIMENT_ID},
        }

    def upstream_setup(self, connection: ObservabilityConnection, prefix: str) -> list[str]:
        return [f"mlflow server --static-prefix {prefix}"]
