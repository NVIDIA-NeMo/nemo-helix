# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization API endpoints using embedded WASM policy evaluation."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Path
from nemo_helix_plugin.auth.decisions import POLICY_DECISION_ADAPTER, DecisionEvaluationInput
from nemo_helix_plugin.iam.types import (
    AuthzErrorResponse,
    AuthzRequest,
    AuthzResponse,
)
from nhx.common.config import get_service_config
from nhx.common.entities import EntityClient
from nhx.common.service.dependencies import get_entity_client
from nhx.core.auth.app.account_resolution import (
    AccountResolutionError,
    ServicePrincipalNotAllowedError,
    resolve_authz_account_context,
)
from nhx.core.auth.app.embedded_pdp import (
    PolicyEngineError,
    evaluate,
    get_valid_entrypoints,
    load_policy_data,
    validate_entrypoint,
)
from nhx.core.auth.config import AuthServiceConfig
from pydantic import ValidationError

router = APIRouter(tags=["Authorization"])
logger = logging.getLogger(__name__)


@router.post(
    "/v2/authz/{entrypoint}",
    response_model=AuthzResponse,
    responses={
        400: {"model": AuthzErrorResponse, "description": "Invalid entrypoint"},
        500: {"description": "Policy evaluation error"},
    },
    summary="Evaluate authorization policy",
    description="""
Evaluate an authorization policy entrypoint with the provided input.

**Supported entrypoints:**

- `allow` - Main request authorization (checks permissions, scopes, path)
- `has_permissions` - Check if principal has specific permissions in a workspace
- `has_role` - Check if principal has a specific role in a workspace
- `decision` - Typed registered operation, collection or protected-field decision

**Example request for `allow`:**
```json
{
  "input": {
    "principal_id": "user@example.com",
    "method": "GET",
    "path": "/v2/workspaces/my-workspace/models"
  }
}
```

**Example request for `has_permissions`:**
```json
{
  "input": {
    "principal_id": "user@example.com",
    "workspace": "my-workspace",
    "permissions": ["models.read"]
  }
}
```
""",
)
async def evaluate_policy(
    request: AuthzRequest,
    entrypoint: str = Path(
        ...,
        description="Policy entrypoint to evaluate",
        examples=["allow", "has_permissions", "has_role"],
    ),
    entity_client: EntityClient = Depends(get_entity_client),
) -> AuthzResponse:
    """Evaluate an authorization policy entrypoint."""
    if not validate_entrypoint(entrypoint):
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid entrypoint: {entrypoint}",
                "valid_entrypoints": get_valid_entrypoints(),
            },
        )

    try:
        # If bundle_cache_seconds is 0, refresh policy data before each evaluation.
        # This enables instant permission changes in tests without waiting for propagation.
        config = get_service_config(AuthServiceConfig)
        bootstrap_read = False
        if entrypoint == "decision":
            # Trusted service facts are typed; policy owns bootstrap eligibility too.
            envelope = DecisionEvaluationInput.model_validate(request.input)
            request = AuthzRequest(input=envelope.model_dump(mode="json"))
            bootstrap = evaluate("decision", {**request.input, "operation": "auth.policy-data.read"})
            bootstrap_read = bootstrap.get("allowed") is True

        if config.bundle_cache_seconds == 0 and not bootstrap_read:
            if not await load_policy_data(entity_client):
                raise HTTPException(status_code=503, detail="Policy data could not be refreshed")

        account_context = await resolve_authz_account_context(request.input, config)
        enriched_input = {**request.input, **account_context.to_policy_fields()}

        result = evaluate(entrypoint, enriched_input)
        if entrypoint == "decision":
            return AuthzResponse(result=POLICY_DECISION_ADAPTER.validate_python(result).model_dump(mode="json"))
        return AuthzResponse(result={**result, **account_context.to_policy_fields()})

    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="Invalid policy decision request") from exc

    except ServicePrincipalNotAllowedError as e:
        logger.warning("Service principal resolution failed closed: %s", e)
        raise HTTPException(status_code=403, detail=str(e)) from e
    except AccountResolutionError as e:
        logger.warning("Account resolution failed: %s", e)
        raise HTTPException(status_code=403, detail=str(e)) from e

    except PolicyEngineError as e:
        logger.error("Policy engine error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
