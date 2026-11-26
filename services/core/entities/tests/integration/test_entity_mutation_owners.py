# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nhx.common.auth.models import Principal
from nhx.testing import TEST_ADMIN_EMAIL, create_test_client
from nhx.testing.client import ClientContext


def test_job_group_mutations_require_the_owning_service() -> None:
    path = "/apis/entities/v2/workspaces/default/entities/platform_job_group"
    jobs = Principal(id="service:jobs").get_headers()
    delegated_jobs = Principal(id="service:jobs", on_behalf_of=TEST_ADMIN_EMAIL).get_headers()
    other_service = Principal(id="service:evals").get_headers()
    admin = Principal(id=TEST_ADMIN_EMAIL, email=TEST_ADMIN_EMAIL).get_headers()
    create = {"name": "protected-group", "data": {"controller_state": {"status": "pending"}}}
    update = {"data": {"controller_state": {"status": "complete"}}}
    with create_test_client(auth_enabled=True, client_type=ClientContext) as clients:
        client = clients.test_client
        encoded_workspace = path.replace("/default/", "/w%2Fentities%2Fother/")
        for request in (
            {"path": path},
            {"path": path, "target": {"service": "jobs", "entity_type": "platform_job"}},
            {"path": encoded_workspace},
            {"attributes": {"request": {"http": {"path": encoded_workspace.replace("%", "%25")}}}},
        ):
            decision = client.post(
                "/apis/auth/v2/authz/allow",
                headers=jobs,
                json={"input": {"principal_id": "service:evals", "method": "POST", **request}},
            )
            assert decision.status_code == 200, decision.text
            assert decision.json()["result"]["allowed"] is False
        for caller in (other_service, admin):
            for target_path in (path, path.replace("platform_job_group", "platform%5Fjob_group")):
                denied = client.post(target_path, headers=caller, json=create)
                assert denied.status_code == 403, denied.text
        created = client.post(path, headers=delegated_jobs, json=create)
        assert created.status_code == 201, created.text
        group_id = created.json()["id"]

        name_path = f"{path}/protected-group"
        id_path = f"{path}/by-id/{group_id}"
        assert client.get(name_path, headers=other_service).status_code == 200
        for caller in (other_service, admin):
            for target in (name_path, id_path):
                assert client.put(target, headers=caller, json=update).status_code == 403
                assert client.delete(target, headers=caller).status_code == 403
        assert client.put(name_path, headers=jobs, json=update).status_code == 200
        assert client.delete(id_path, headers=jobs).status_code == 200
        assert client.get(name_path, headers=jobs).status_code == 404
