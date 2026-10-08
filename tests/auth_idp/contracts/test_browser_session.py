# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import urljoin, urlparse

import pytest
from nhx.testing import grant_workspace_role
from playwright.sync_api import expect

from tests.auth_idp.browser_fixtures import AuthBrowser
from tests.auth_idp.common import require_capability, runtime_tls_config

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
    pytest.mark.xdist_group("idp-live"),
]


def test_studio_confidential_browser_session_enforces_browser_security_contract(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
    auth_browser: AuthBrowser,
    cross_origin_url: str,
) -> None:
    require_capability(auth_idp_case, "confidential_oidc")
    require_capability(auth_idp_case, "workspace_rbac")

    gateway_base_url = auth_idp_runtime.gateway_base_url.rstrip("/")
    expected_email = auth_idp_case.provider.interactive_user_expected_email
    studio_path = f"/studio/workspaces/{auth_idp_workspace}"
    studio_url = f"{gateway_base_url}{studio_path}"
    login_path = "/apis/auth/v2/login"
    page = auth_browser.page

    grant_workspace_role(
        auth_idp_runtime.e2e_setup_client(),
        workspace=auth_idp_workspace,
        principal=expected_email,
        roles=["Viewer"],
    )

    with page.expect_response(lambda response: urlparse(response.url).path == login_path) as login_response_info:
        page.goto(studio_url)
    login_response = login_response_info.value
    assert login_response.status in {302, 303, 307, 308}
    authorization_url = urljoin(login_response.url, login_response.headers["location"])

    callback_url = auth_idp_runtime.complete_confidential_authorization(
        authorization_url=authorization_url,
        username=auth_idp_case.provider.interactive_user_username,
        password=auth_idp_case.provider.interactive_user_password,
        tls_config=runtime_tls_config(auth_idp_runtime),
    )
    page.goto(callback_url)
    page.wait_for_url(studio_url)

    user_navigation = page.locator('[data-tour="nav-user"]')
    expect(user_navigation).to_contain_text(expected_email)

    cookies = {cookie["name"]: cookie for cookie in auth_browser.context.cookies(gateway_base_url)}
    session_cookie = cookies["nhx_session"]
    assert session_cookie["httpOnly"] is True
    assert session_cookie["secure"] is True
    assert session_cookie["sameSite"] == "Lax"

    session_url = f"{gateway_base_url}/apis/auth/v2/session"
    session = auth_browser.context.request.get(session_url)
    assert session.status == 200, session.text()
    assert session.json()["email"] == expected_email

    public_ext_authz = auth_browser.context.request.get(
        f"{gateway_base_url}/apis/auth/ext-authz/apis/entities/v2/workspaces/{auth_idp_workspace}"
    )
    assert public_ext_authz.status == 404, public_ext_authz.text()

    attacker = auth_browser.context.new_page()
    attacker.goto(cross_origin_url)
    with attacker.expect_request(
        lambda request: request.method == "POST" and urlparse(request.url).path == "/apis/auth/v2/logout"
    ) as cross_origin_logout_info:
        cross_origin_status = attacker.evaluate(
            """
            async (logoutUrl) => {
              const response = await fetch(logoutUrl, {
                method: 'POST',
                credentials: 'include',
                headers: {'X-Source': 'NeMo Studio'},
              });
              return response.status;
            }
            """,
            f"{gateway_base_url}/apis/auth/v2/logout",
        )
    cross_origin_logout = cross_origin_logout_info.value
    assert cross_origin_status == 204
    assert "nhx_session=" not in (cross_origin_logout.header_value("cookie") or "")
    attacker.close()

    session_after_attack = auth_browser.context.request.get(session_url)
    assert session_after_attack.status == 200, session_after_attack.text()

    user_navigation.get_by_test_id("nv-dropdown-trigger").click()
    with page.expect_request(
        lambda request: request.method == "POST" and urlparse(request.url).path == "/apis/auth/v2/logout"
    ) as logout_request_info:
        page.get_by_text("Sign Out", exact=True).click()
    logout_request = logout_request_info.value
    assert logout_request.headers["x-source"] == "NeMo Studio"
    logout_response = logout_request.response()
    assert logout_response is not None
    assert logout_response.status == 204

    revoked = auth_browser.context.request.get(session_url)
    assert revoked.status == 401, revoked.text()
