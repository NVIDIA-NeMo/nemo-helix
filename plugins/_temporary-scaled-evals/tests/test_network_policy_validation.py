# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.schemas.benchmark_runs import CreateBenchmarkRunRequest
from scaled_evals.api.schemas.evaluations import CreateEvaluationRequest


def _config(cidr: str = "1.1.1.1/32", *, excluded: list[str] | None = None) -> dict:
    ip_block: dict[str, object] = {"cidr": cidr}
    if excluded is not None:
        ip_block["except"] = excluded
    return {
        "egress": [
            {
                "to": [{"ipBlock": ip_block}],
                "ports": [{"protocol": "TCP", "port": 443}],
            }
        ]
    }


@pytest.mark.parametrize(
    "request_type,identity",
    [
        (CreateEvaluationRequest, {"task_id": "task_x", "task_revision": 1}),
        (CreateBenchmarkRunRequest, {"benchmark_id": "bm_x", "benchmark_revision": 1}),
    ],
)
def test_scoped_egress_accepts_explicit_public_cidrs(request_type, identity) -> None:  # noqa: ANN001
    request = request_type(
        name="scoped",
        network_policy="scoped_egress",
        network_policy_config=_config("2606:4700:4700::1111/128"),
        **identity,
    )

    assert request.network_policy_config["egress"]


@pytest.mark.parametrize(
    "config",
    [
        {"egress": [{}]},
        {"egress": [{"ports": [{"protocol": "TCP", "port": 443}]}]},
        {"egress": [{"to": []}]},
        {"egress": [{"to": [{"podSelector": {}}]}]},
        {"egress": [{"to": [{"namespaceSelector": {}}]}]},
        _config("0.0.0.0/0"),
        _config("10.0.0.0/8"),
        _config("169.254.169.254/32"),
        _config("2001:db8::/32"),
        _config("1.1.1.1/24"),
        _config("1.1.1.0/24", excluded=["1.0.0.0/24"]),
        _config("1.1.1.0/24", excluded=["1.1.1.128/25", "1.1.1.128/25"]),
        _config("1.1.1.0/24", excluded=["not-a-cidr"]),
    ],
)
def test_scoped_egress_rejects_broad_internal_or_ambiguous_rules(config: dict) -> None:
    with pytest.raises(ValueError):
        CreateEvaluationRequest(
            name="scoped",
            task_id="task_x",
            task_revision=1,
            network_policy="scoped_egress",
            network_policy_config=config,
        )


def test_scoped_egress_accepts_valid_exclusions() -> None:
    request = CreateEvaluationRequest(
        name="scoped",
        task_id="task_x",
        task_revision=1,
        network_policy="scoped_egress",
        network_policy_config=_config("1.1.1.0/24", excluded=["1.1.1.128/25"]),
    )

    assert request.network_policy_config["egress"][0]["to"][0]["ipBlock"]["except"] == ["1.1.1.128/25"]
