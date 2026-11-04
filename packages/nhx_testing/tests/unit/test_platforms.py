# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``supplied_platform`` against a stub server, so no real platform is needed to check how it decides.

The stub answers the three probes ``supplied_platform`` makes. Each failure case changes one answer of a suitable
platform, so it shows that one check working on its own.
"""

from collections.abc import Mapping

import pytest
from nhx.testing.platforms import supplied_platform
from pytest_httpserver import HTTPServer

URL_ENV = "NHX_TEST_SUPPLIED_PLATFORM_URL"
CONTROLLER = "evals-taskset-runs"

#: What a platform suitable for an auth-off suite that must run without ``CONTROLLER`` answers, by path.
SUITABLE_AUTH_OFF_PLATFORM: dict[str, tuple[int, object]] = {
    "/health/ready": (200, {"status": "ready"}),
    "/apis/auth/discovery": (200, {"auth_enabled": False}),
    "/status": (200, {"controllers": {"healthy": True, "status": {"entities": True, "jobs": True, "models": True}}}),
}


def _serve(httpserver: HTTPServer, responses: Mapping[str, tuple[int, object]]) -> str:
    """Have the stub answer each path in ``responses`` with its status and JSON body; return its base URL."""
    for path, (status, body) in responses.items():
        httpserver.expect_request(path).respond_with_json(body, status=status)
    return httpserver.url_for("")


def test_an_unset_url_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(URL_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match=f"{URL_ENV} is not set"):
        supplied_platform(URL_ENV, auth_enabled=False)


def test_a_suitable_platform_is_used(httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch) -> None:
    base_url = _serve(httpserver, SUITABLE_AUTH_OFF_PLATFORM)
    monkeypatch.setenv(URL_ENV, base_url)  # url_for("") ends in "/", which supplied_platform drops
    assert supplied_platform(URL_ENV, auth_enabled=False, forbidden_controllers=[CONTROLLER]) == base_url.rstrip("/")


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        pytest.param({"/health/ready": (503, {"status": "not ready"})}, "is not a ready platform", id="not-ready"),
        pytest.param({"/apis/auth/discovery": (200, {"auth_enabled": True})}, "has auth on", id="auth-on"),
        pytest.param(
            {"/status": (200, {"controllers": {"healthy": True, "status": {CONTROLLER: True}}})},
            f"runs the {CONTROLLER} controller",
            id="forbidden-controller-running",
        ),
        # JSON from something that is not a platform: a missing field, or one of another type.
        pytest.param({"/apis/auth/discovery": (200, {})}, "is not a ready platform", id="no-auth-enabled"),
        pytest.param({"/status": (200, ["jobs"])}, "is not a ready platform", id="status-not-an-object"),
        pytest.param(
            {"/status": (200, {"controllers": {"status": None}})}, "is not a ready platform", id="no-controller-list"
        ),
    ],
)
def test_an_unsuitable_platform_fails(
    httpserver: HTTPServer,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, tuple[int, object]],
    problem: str,
) -> None:
    monkeypatch.setenv(URL_ENV, _serve(httpserver, {**SUITABLE_AUTH_OFF_PLATFORM, **overrides}))
    with pytest.raises(pytest.fail.Exception, match=problem) as failure:
        supplied_platform(URL_ENV, auth_enabled=False, forbidden_controllers=[CONTROLLER], start_hint="Start one.")
    assert str(failure.value).endswith("Start one.")
