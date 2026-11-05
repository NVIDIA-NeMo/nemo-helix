# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed transport for trusted policy queries from controllers and standalone drivers."""

from nemo_helix_plugin.auth.decisions import DecisionEvaluationInput, PolicyDecision, validate_decision_result
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.iam.client import AsyncIAMClient
from nemo_helix_plugin.iam.types import AuthzRequest


async def query_decision(client: AsyncNemoClient, document: DecisionEvaluationInput) -> PolicyDecision:
    response = await AsyncIAMClient.from_client(client).evaluate_authorization(
        entrypoint="decision", body=AuthzRequest(input=document.model_dump(mode="json"))
    )
    return validate_decision_result(document, response.data().result)
