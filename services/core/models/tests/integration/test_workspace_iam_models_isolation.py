# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace IAM isolation for model and adapter create (in-process, auth on).

In-process tests mirror a live e2e flow: four workspaces, direct membership on A/B, a
shared group on C, a single owner on D, then model/adapter create checks across those
workspaces. The first class uses the NeMoHelix SDK; the second issues hand-built
HTTP to the same routes using a ``requests`` Session (see
:class:`_TestClientToRequestsAdapter`) that forwards to the Starlette ``TestClient``,
since CPython ``requests`` cannot open an in-process ASGI app directly.

Per-workspace model entities exist in C and D. User A (group on C) may add an
adapter in A whose fileset (LoRA / base data) is in C; a fileset in D while
adapting a model in A must be denied (403) because the caller cannot read D.
"""

from __future__ import annotations

from collections.abc import Generator
from urllib.parse import urlparse
from uuid import uuid4

import pytest
import requests
from fastapi.testclient import TestClient
from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import (
    CreateModelAdapterRequest,
    CreateModelEntityRequest,
    FinetuningType,
)
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceQueryParams, CreateWorkspaceRequest
from nhx.core.files.service import FilesService
from nhx.core.models.service import ModelsService
from nhx.core.secrets.service import SecretsService
from nhx.testing import (
    TEST_ADMIN_EMAIL,
    ClientContext,
    as_user,
    create_test_client,
    grant_workspace_role,
    short_unique_name,
    unique_email,
)
from requests import Response
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict

# -- helpers for requests <-> TestClient (in-process) ---------------------------------


class _TestClientToRequestsAdapter(BaseAdapter):
    """Sends a prepared ``requests`` request through ``TestClient`` (ASGI)."""

    def __init__(self, test_client: TestClient) -> None:
        self._test_client = test_client

    def _path_and_query(self, request) -> str:
        p = urlparse(request.url)
        out = p.path or "/"
        if p.query:
            out = f"{out}?{p.query}"
        return out

    @staticmethod
    def _filter_headers(prep) -> dict[str, str]:
        skip = {"host", "content-length", "connection"}
        return {k: v for k, v in prep.headers.items() if k.lower() not in skip}

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None, **kwargs) -> Response:
        path = self._path_and_query(request)
        r = self._test_client.request(
            request.method,
            path,
            content=request.body,
            headers=self._filter_headers(request),
        )
        resp = Response()
        resp.status_code = r.status_code
        resp._content = r.content
        resp.url = str(request.url)
        resp.request = request
        resp.headers = CaseInsensitiveDict(r.headers)
        return resp

    def close(self) -> None:
        pass


def _requests_on_test_client(
    test_client: TestClient,
) -> requests.Session:
    s = requests.Session()
    a = _TestClientToRequestsAdapter(test_client)
    s.mount("http://testserver", a)
    s.mount("http://testserver/", a)
    return s


# -- fixtures -------------------------------------------------------------------------

# models_auth_context is module-scoped (expensive to boot, auth_enabled=True): keep both
# test classes on one xdist worker so they share it instead of each worker re-provisioning
# it from scratch.
pytestmark = pytest.mark.xdist_group("workspace_iam_models_isolation")


@pytest.fixture(scope="module")
def models_auth_context() -> Generator[ClientContext, None, None]:
    """ClientContext with models, files, and secrets; auth on."""
    with create_test_client(
        ModelsService,
        FilesService,
        SecretsService,
        auth_enabled=True,
        client_type=ClientContext,
    ) as ctx:
        yield ctx


@pytest.fixture(scope="module")
def sdk(models_auth_context: ClientContext) -> NeMoHelix:
    return models_auth_context.sdk


@pytest.mark.integration
class TestWorkspaceIamIsolationSDK:
    """End-to-end style IAM check using the NeMoHelix ``models`` / ``adapters`` SDK."""

    @pytest.mark.usefixtures("models_auth_context")
    def test_model_and_adapter_iam(self, sdk: NeMoHelix) -> None:
        user_a = unique_email("user-a")
        user_b = unique_email("user-b")
        owner_d = unique_email("owner-d")
        ws_a = short_unique_name("wks-a")
        ws_b = short_unique_name("wks-b")
        ws_c = short_unique_name("wks-c")
        ws_d = short_unique_name("wks-d")
        shared_group = f"team-{uuid4().hex[:12]}"

        admin: NeMoHelix = as_user(sdk, TEST_ADMIN_EMAIL)
        workspaces = client_from_platform(admin, WorkspacesClient)

        workspaces.create_workspace(
            query_params=CreateWorkspaceQueryParams(wait_role_propagation=True),
            body=CreateWorkspaceRequest(name=ws_a, description="user-a only"),
        ).data()
        workspaces.create_workspace(
            query_params=CreateWorkspaceQueryParams(wait_role_propagation=True),
            body=CreateWorkspaceRequest(name=ws_b, description="user-b only"),
        ).data()
        workspaces.create_workspace(
            query_params=CreateWorkspaceQueryParams(wait_role_propagation=True),
            body=CreateWorkspaceRequest(name=ws_c, description="shared via group"),
        ).data()

        as_user(sdk, owner_d).workspaces.create(
            name=ws_d, description="isolated from A and B", wait_role_propagation=True
        )

        grant_workspace_role(admin, workspace=ws_a, principal=user_a, roles=["Editor"])
        grant_workspace_role(admin, workspace=ws_b, principal=user_b, roles=["Editor"])
        grant_workspace_role(admin, workspace=ws_c, principal=shared_group, roles=["Editor"])

        model_a = short_unique_name("mdl-a")
        model_b = short_unique_name("mdl-b")
        ua: NeMoHelix = as_user(sdk, user_a)
        ub: NeMoHelix = as_user(sdk, user_b)
        uac: NeMoHelix = as_user(sdk, user_a, groups=[shared_group])
        ubc: NeMoHelix = as_user(sdk, user_b, groups=[shared_group])

        client_from_platform(ua, ModelsClient).create_model(
            workspace=ws_a, body=CreateModelEntityRequest(name=model_a)
        ).data()
        client_from_platform(ub, ModelsClient).create_model(
            workspace=ws_b, body=CreateModelEntityRequest(name=model_b)
        ).data()
        client_from_platform(uac, ModelsClient).create_model(
            workspace=ws_c, body=CreateModelEntityRequest(name=short_unique_name("mdl-c-a"))
        ).data()
        model_c_b = short_unique_name("mdl-c-b")
        client_from_platform(ubc, ModelsClient).create_model(
            workspace=ws_c, body=CreateModelEntityRequest(name=model_c_b)
        ).data()

        od: NeMoHelix = as_user(sdk, owner_d)
        model_d = short_unique_name("mdl-d")
        client_from_platform(od, ModelsClient).create_model(
            workspace=ws_d, body=CreateModelEntityRequest(name=model_d)
        ).data()

        # Filesets: fileset in C (for allow with group); fileset in D (for deny in C)
        fs_c = short_unique_name("fs-c")
        fs_d = short_unique_name("fs-d")
        admin_files = client_from_platform(admin, FilesClient)
        admin_files.create_fileset(workspace=ws_c, body=CreateFilesetRequest(name=fs_c))
        admin_files.create_fileset(workspace=ws_d, body=CreateFilesetRequest(name=fs_d))
        admin_files.upload_file(workspace=ws_c, name=fs_c, path="a.txt", content=b"x")
        admin_files.upload_file(workspace=ws_d, name=fs_d, path="a.txt", content=b"x")

        # 13: adapter in C with a fileset in D is denied (no access to D fileset)
        with pytest.raises(PermissionDeniedError):
            client_from_platform(uac, ModelsClient).create_model_adapter(
                model_name=model_c_b,
                workspace=ws_c,
                body=CreateModelAdapterRequest(
                    name=short_unique_name("adp-c-fileset-d"),
                    fileset=f"{ws_d}/{fs_d}",
                    finetuning_type=FinetuningType.LORA,
                ),
            )

        # Adapter in ws_a on local model, LoRA data in ws_c: allowed (user A can read C).
        client_from_platform(uac, ModelsClient).create_model_adapter(
            model_name=model_a,
            workspace=ws_a,
            body=CreateModelAdapterRequest(
                name=short_unique_name("adp-allow-c"),
                fileset=f"{ws_c}/{fs_c}",
                finetuning_type=FinetuningType.LORA,
            ),
        )
        # Same local model, LoRA / base storage in ws_d: denied (no D access; targets D "base").
        with pytest.raises(PermissionDeniedError):
            client_from_platform(uac, ModelsClient).create_model_adapter(
                model_name=model_a,
                workspace=ws_a,
                body=CreateModelAdapterRequest(
                    name=short_unique_name("adp-deny-d"),
                    fileset=f"{ws_d}/{fs_d}",
                    finetuning_type=FinetuningType.LORA,
                ),
            )

        client_from_platform(uac, ModelsClient).create_model(
            workspace=ws_c, body=CreateModelEntityRequest(name=short_unique_name("mdl-into-c-a"))
        ).data()
        client_from_platform(ubc, ModelsClient).create_model(
            workspace=ws_c, body=CreateModelEntityRequest(name=short_unique_name("mdl-into-c-b"))
        ).data()

        with pytest.raises(PermissionDeniedError):
            client_from_platform(uac, ModelsClient).create_model(
                workspace=ws_d, body=CreateModelEntityRequest(name=short_unique_name("deny-a-into-d"))
            ).data()
        with pytest.raises(PermissionDeniedError):
            client_from_platform(ubc, ModelsClient).create_model(
                workspace=ws_d, body=CreateModelEntityRequest(name=short_unique_name("deny-b-into-d"))
            ).data()


@pytest.mark.integration
class TestWorkspaceIamIsolationHttpRequests:
    """Same flow: ``requests`` to models/entities; fileset setup uses the NeMo file APIs via SDK (matches other integration tests)."""

    def test_model_and_adapter_iam(
        self,
        models_auth_context: ClientContext,
    ) -> None:
        user_a = unique_email("user-a")
        user_b = unique_email("user-b")
        owner_d = unique_email("owner-d")
        ws_a = short_unique_name("wks-a")
        ws_b = short_unique_name("wks-b")
        ws_c = short_unique_name("wks-c")
        ws_d = short_unique_name("wks-d")
        shared_group = f"team-{uuid4().hex[:12]}"
        http = _requests_on_test_client(models_auth_context.test_client)

        def h(email: str, groups: list[str] | None = None) -> dict[str, str]:
            hdr: dict[str, str] = {
                "X-NHX-Principal-Id": email,
                "X-NHX-Principal-Email": email,
            }
            if groups:
                hdr["X-NHX-Principal-Groups"] = ",".join(groups)
            return hdr

        b = "http://testserver"
        post = http.post
        assert (
            post(
                f"{b}/apis/entities/v2/workspaces?wait_role_propagation=true",
                json={"name": ws_a, "description": "a"},
                headers=h(TEST_ADMIN_EMAIL),
            ).status_code
            == 201
        )
        assert (
            post(
                f"{b}/apis/entities/v2/workspaces?wait_role_propagation=true",
                json={"name": ws_b, "description": "b"},
                headers=h(TEST_ADMIN_EMAIL),
            ).status_code
            == 201
        )
        assert (
            post(
                f"{b}/apis/entities/v2/workspaces?wait_role_propagation=true",
                json={"name": ws_c, "description": "c"},
                headers=h(TEST_ADMIN_EMAIL),
            ).status_code
            == 201
        )
        assert (
            post(
                f"{b}/apis/entities/v2/workspaces?wait_role_propagation=true",
                json={"name": ws_d, "description": "d"},
                headers=h(owner_d),
            ).status_code
            == 201
        )
        assert (
            post(
                f"{b}/apis/entities/v2/workspaces/{ws_a}/members?wait_role_propagation=true",
                json={"principal": user_a, "roles": ["Editor"]},
                headers=h(TEST_ADMIN_EMAIL),
            ).status_code
            == 201
        )
        assert (
            post(
                f"{b}/apis/entities/v2/workspaces/{ws_b}/members?wait_role_propagation=true",
                json={"principal": user_b, "roles": ["Editor"]},
                headers=h(TEST_ADMIN_EMAIL),
            ).status_code
            == 201
        )
        assert (
            post(
                f"{b}/apis/entities/v2/workspaces/{ws_c}/members?wait_role_propagation=true",
                json={"principal": shared_group, "roles": ["Editor"]},
                headers=h(TEST_ADMIN_EMAIL),
            ).status_code
            == 201
        )

        model_a = short_unique_name("mdl-a")
        model_b = short_unique_name("mdl-b")
        assert (
            post(
                f"{b}/apis/models/v2/workspaces/{ws_a}/models",
                json={"name": model_a},
                headers=h(user_a),
            ).status_code
            == 201
        )
        assert (
            post(
                f"{b}/apis/models/v2/workspaces/{ws_b}/models",
                json={"name": model_b},
                headers=h(user_b),
            ).status_code
            == 201
        )
        assert (
            post(
                f"{b}/apis/models/v2/workspaces/{ws_c}/models",
                json={"name": short_unique_name("mdl-c-a")},
                headers=h(user_a, [shared_group]),
            ).status_code
            == 201
        )
        mcb = short_unique_name("mdl-c-b")
        assert (
            post(
                f"{b}/apis/models/v2/workspaces/{ws_c}/models",
                json={"name": mcb},
                headers=h(user_b, [shared_group]),
            ).status_code
            == 201
        )
        mdl_d = short_unique_name("mdl-d")
        assert (
            post(
                f"{b}/apis/models/v2/workspaces/{ws_d}/models",
                json={"name": mdl_d},
                headers=h(owner_d),
            ).status_code
            == 201
        )

        fs_c = short_unique_name("fs-c")
        fs_d = short_unique_name("fs-d")
        admin_sdk = as_user(models_auth_context.sdk, TEST_ADMIN_EMAIL)
        files = client_from_platform(admin_sdk, FilesClient)
        files.create_fileset(workspace=ws_c, body=CreateFilesetRequest(name=fs_c))
        files.create_fileset(workspace=ws_d, body=CreateFilesetRequest(name=fs_d))
        files.upload_file(workspace=ws_c, name=fs_c, path="a.txt", content=b"x")
        files.upload_file(workspace=ws_d, name=fs_d, path="a.txt", content=b"x")

        # 13
        r13 = post(
            f"{b}/apis/models/v2/workspaces/{ws_c}/models/{mcb}/adapters",
            json={
                "name": short_unique_name("adp-13"),
                "fileset": f"{ws_d}/{fs_d}",
                "finetuning_type": "lora",
            },
            headers=h(user_a, [shared_group]),
        )
        assert r13.status_code == 403, r13.text

        # Adapter in ws_a: fileset in C ok; fileset in D (base in D) must be 403.
        r14 = post(
            f"{b}/apis/models/v2/workspaces/{ws_a}/models/{model_a}/adapters",
            json={
                "name": short_unique_name("adp-14-c"),
                "fileset": f"{ws_c}/{fs_c}",
                "finetuning_type": "lora",
            },
            headers=h(user_a, [shared_group]),
        )
        assert r14.status_code == 201, r14.text
        r14d = post(
            f"{b}/apis/models/v2/workspaces/{ws_a}/models/{model_a}/adapters",
            json={
                "name": short_unique_name("adp-14-d"),
                "fileset": f"{ws_d}/{fs_d}",
                "finetuning_type": "lora",
            },
            headers=h(user_a, [shared_group]),
        )
        assert r14d.status_code == 403, r14d.text
        # 15-16
        assert (
            post(
                f"{b}/apis/models/v2/workspaces/{ws_c}/models",
                json={"name": short_unique_name("c15")},
                headers=h(user_a, [shared_group]),
            ).status_code
            == 201
        )
        assert (
            post(
                f"{b}/apis/models/v2/workspaces/{ws_c}/models",
                json={"name": short_unique_name("c16")},
                headers=h(user_b, [shared_group]),
            ).status_code
            == 201
        )
        # 17-18
        r17 = post(
            f"{b}/apis/models/v2/workspaces/{ws_d}/models",
            json={"name": short_unique_name("n17")},
            headers=h(user_a, [shared_group]),
        )
        r18 = post(
            f"{b}/apis/models/v2/workspaces/{ws_d}/models",
            json={"name": short_unique_name("n18")},
            headers=h(user_b, [shared_group]),
        )
        assert r17.status_code == 403, r17.text
        assert r18.status_code == 403, r18.text
