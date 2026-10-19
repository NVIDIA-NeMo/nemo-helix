# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.sandbox_egress_qualification import Qualification, _parse_endpoint


def test_qualification_rejects_identical_endpoints() -> None:
    with pytest.raises(ValueError, match="must differ"):
        Qualification(
            object(),  # type: ignore[arg-type]
            namespace="sandbox",
            image="registry.example/probe:tag",
            allowed=("1.1.1.1", 443),
            denied=("1.1.1.1", 443),
        )


def test_parse_endpoint_requires_ipv4_host_and_port() -> None:
    assert _parse_endpoint("1.1.1.1:443") == ("1.1.1.1", 443)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_endpoint("example.com:443")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_endpoint("1.1.1.1:65536")


def test_assert_result_requires_dns_and_exact_reachability() -> None:
    Qualification._assert_result(
        "scoped",
        {
            "dns": True,
            "targets": {
                "allowed": {"reachable": True},
                "denied": {"reachable": False},
            },
        },
        {"allowed": True, "denied": False},
    )

    with pytest.raises(RuntimeError, match="egress mismatch"):
        Qualification._assert_result(
            "default-deny",
            {"dns": True, "targets": {"public": {"reachable": True}}},
            {"public": False},
        )
