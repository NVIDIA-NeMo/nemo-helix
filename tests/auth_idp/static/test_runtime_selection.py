# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.auth_idp.runtime_factory import iter_auth_idp_cases, runtime_class_for_case

pytestmark = [pytest.mark.auth_idp]


def test_iter_auth_idp_cases_includes_authentik_compose() -> None:
    case_ids = {case.id for case in iter_auth_idp_cases()}

    assert "authentik-compose" in case_ids


def test_iter_auth_idp_cases_can_filter_by_backend() -> None:
    case_ids = {case.id for case in iter_auth_idp_cases(backend="kubernetes")}

    assert "authentik-kubernetes" in case_ids
    assert "zitadel-kubernetes" in case_ids
    assert "authentik-compose" not in case_ids


def test_iter_auth_idp_cases_can_filter_by_provider_name() -> None:
    case_ids = {case.id for case in iter_auth_idp_cases(provider_name="authentik")}

    assert case_ids == {"authentik-compose", "authentik-kubernetes"}


def test_iter_auth_idp_cases_can_filter_by_zitadel_provider_name() -> None:
    case_ids = {case.id for case in iter_auth_idp_cases(provider_name="zitadel")}

    assert case_ids == {"zitadel-kubernetes"}


def test_iter_auth_idp_cases_can_filter_by_runtime_id() -> None:
    case_ids = {case.id for case in iter_auth_idp_cases(runtime_id="authentik-kubernetes")}

    assert case_ids == {"authentik-kubernetes"}


def test_runtime_case_has_pytest_id() -> None:
    [case] = [case for case in iter_auth_idp_cases() if case.id == "authentik-compose"]

    assert pytest.param(case, id=case.id).id == "authentik-compose"


def test_compose_runtime_factory_selects_compose_runtime() -> None:
    case = next(case for case in iter_auth_idp_cases() if case.id == "authentik-compose")

    runtime_class = runtime_class_for_case(case)

    assert runtime_class.__name__ == "ComposeAuthIdpRuntime"


def test_kubernetes_runtime_factory_selects_kubernetes_runtime() -> None:
    case = next(case for case in iter_auth_idp_cases() if case.id == "authentik-kubernetes")

    runtime_class = runtime_class_for_case(case)

    assert runtime_class.__name__ == "KubernetesAuthIdpRuntime"
