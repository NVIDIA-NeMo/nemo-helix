# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastapi.routing import APIRoute
from nemo_eval_author_plugin.jobs.first_eval import FirstEvalJob
from nemo_eval_author_plugin.service import EvalAuthorService
from nemo_helix_plugin.authz_discovery import _derive_service_contribution
from nemo_helix_plugin.scheduler import submit_path_for


def _mounted_post_paths() -> set[str]:
    paths: set[str] = set()
    for spec in EvalAuthorService().get_routers():
        for route in spec.router.routes:
            if isinstance(route, APIRoute) and "POST" in (route.methods or set()):
                paths.add(f"/apis/{EvalAuthorService.name}{spec.prefix}{route.path}")
    return paths


def test_first_eval_submit_route_matches_the_generated_path() -> None:
    assert submit_path_for(FirstEvalJob, workspace="{workspace}") in _mounted_post_paths()


def test_authz_derivation_has_no_problems() -> None:
    contrib, problems, _warnings = _derive_service_contribution(EvalAuthorService())

    assert problems == []
    jobs = "/apis/eval-author/v2/workspaces/{workspace}/jobs/first-eval"
    assert contrib.endpoints[jobs]["post"].permissions == ["eval-author.first-eval.create"]
