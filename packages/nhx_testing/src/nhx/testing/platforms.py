# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point a test suite at a platform someone else already runs, instead of starting its own.

The suite names an environment variable for the platform's base URL. Unset, its tests skip; set, the platform has
to be ready and match what the suite assumes, or setup fails once with the reason, rather than each test failing
its own way::

    @pytest.fixture(scope="session")
    def auth_platform() -> str:
        return supplied_platform("MY_SUITE_AUTH_BASE_URL", auth_enabled=True)
"""

import os
from collections.abc import Collection

import httpx
import pytest


def supplied_platform(
    env_var: str, *, auth_enabled: bool, forbidden_controllers: Collection[str] = (), start_hint: str = ""
) -> str:
    """The platform URL in ``env_var``: skip when unset, fail unless it can serve the calling tests.

    Serving them takes a platform that answers ``/health/ready``, has auth on exactly when ``auth_enabled`` is set,
    and runs none of ``forbidden_controllers``. ``start_hint`` ends a failure, to say how to start one that does.

    ``forbidden_controllers`` is for a suite that stands in for a controller by writing that controller's records
    itself. If the real controller ran on the same platform, it would act on those records mid-test (the taskset-run
    controller fails runs whose plans don't exist) and fail tests at random; naming it here fails setup instead.
    """
    mode = "auth-on" if auth_enabled else "auth-off"
    base_url = os.environ.get(env_var, "").strip().rstrip("/")
    if not base_url:
        pytest.skip(f"{env_var} is not set: point it at a running {mode} platform")
    try:
        with httpx.Client(base_url=base_url, timeout=10) as http:
            http.get("/health/ready").raise_for_status()
            platform_auth = http.get("/apis/auth/discovery").raise_for_status().json()["auth_enabled"]
            running = http.get("/status").raise_for_status().json()["controllers"]["status"]
            if not isinstance(running, dict):  # a list or a string would let a forbidden controller pass unseen
                raise TypeError(f"/status controllers.status is a {type(running).__name__}, not an object")
            forbidden_running = [name for name in forbidden_controllers if name in running]
    # ValueError: a body that is not JSON. KeyError, TypeError: JSON that is not shaped like a platform's.
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, KeyError, TypeError) as error:
        problem = f"is not a ready platform: {error}"
    else:
        if platform_auth is not auth_enabled:
            problem = f"has auth {'on' if platform_auth else 'off'}; it must be an {mode} platform"
        elif forbidden_running:
            problem = f"runs the {forbidden_running[0]} controller, which these tests must run without"
        else:
            return base_url
    pytest.fail(f"{env_var}={base_url} {problem}. {start_hint}".rstrip(), pytrace=False)
