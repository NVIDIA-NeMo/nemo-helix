# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-service hammer. Each call is one sample; a service error does not stop the loop."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Protocol

from history import HistoryLog, classify
from nemo_helix_plugin.auth.access_keys.client import AccessKeysClient
from nemo_helix_plugin.auth.access_keys.types import AccessKeyCreateRequest
from nemo_helix_plugin.client.errors import NemoClientError, NemoHTTPError
from nemo_helix_plugin.client.oidc import discover_nhx_config
from nemo_helix_plugin.client_provider import get_task_nemo_client
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.iam.client import IAMClient
from nemo_helix_plugin.iam.types import AuthzRequest
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest, HelixSecretUpdateRequest
from pydantic import SecretStr

FILE_OPS = (
    "fileset_create",
    "fileset_get",
    "fileset_list",
    "fileset_delete",
    "file_upload",
    "file_download",
    "file_list",
    "file_delete",
)
SECRET_OPS = ("create", "get", "list", "update", "delete", "value_read")
AUTH_OPS = ("discovery", "authz", "access_key_create", "access_key_list", "access_key_revoke")
OPERATIONS = {"files": FILE_OPS, "secrets": SECRET_OPS, "auth": AUTH_OPS}


class Operations(Protocol):
    def fileset_create(self, name: str) -> int: ...
    def fileset_get(self, name: str) -> int: ...
    def fileset_list(self) -> int: ...
    def fileset_delete(self, name: str) -> int: ...
    def file_upload(self, name: str, payload: bytes) -> int: ...
    def file_download(self, name: str) -> int: ...
    def file_list(self) -> int: ...
    def file_delete(self, name: str) -> int: ...
    def secret_create(self, name: str) -> int: ...
    def secret_get(self, name: str) -> int: ...
    def secret_list(self) -> int: ...
    def secret_update(self, name: str) -> int: ...
    def secret_delete(self, name: str) -> int: ...
    def secret_value(self, name: str) -> int: ...
    def discovery(self) -> int: ...
    def authz(self) -> int: ...
    def access_key_create(self) -> tuple[int, str]: ...
    def access_key_list(self) -> int: ...
    def access_key_revoke(self, jti: str) -> int: ...
    def cleanup(self) -> None: ...


def object_name(job: str, kind: str, iteration: int) -> str:
    digest = hashlib.sha256(f"{job}:{kind}:{iteration}".encode()).hexdigest()[:12]
    return f"w{digest}"


def run_hammer(
    *,
    service: str,
    duration: float,
    rate: float,
    payload: bytes,
    job: str,
    step: str,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    log: HistoryLog,
    calls: Operations,
    publish: Callable[[], None] | None = None,
) -> float:
    """Round-robin the service's operations until the duration elapses."""
    operations = OPERATIONS[service]
    started = clock()
    last_publish = started
    index = 0
    interval = 1.0 / rate
    try:
        while clock() - started < duration:
            operation = operations[index % len(operations)]
            index += 1
            began = clock()
            status = _invoke(service, operation, calls, payload, job, index, log)
            latency = clock() - began
            log.write(
                {
                    "record": "operation",
                    "job": job,
                    "step": step,
                    "service": service,
                    "operation": operation,
                    "status": status,
                    "latency": latency,
                    "error_class": classify(service, operation, status),
                }
            )
            if publish is not None and clock() - last_publish >= 30:
                publish()
                last_publish = clock()
            remaining = interval - (clock() - began)
            if remaining > 0:
                sleep(remaining)
    finally:
        calls.cleanup()
    elapsed = clock() - started
    if elapsed <= 0:
        return 0
    return index / elapsed


def _invoke(
    service: str,
    operation: str,
    calls: Operations,
    payload: bytes,
    job: str,
    iteration: int,
    log: HistoryLog,
) -> int:
    name = object_name(job, operation, iteration)
    if service == "files":
        return _files(operation, calls, name, payload)
    if service == "secrets":
        return _secrets(operation, calls, name)
    return _auth(operation, calls, log)


def _files(operation: str, calls: Operations, name: str, payload: bytes) -> int:
    if operation == "fileset_create":
        return calls.fileset_create(name)
    if operation == "fileset_get":
        return calls.fileset_get(name)
    if operation == "fileset_list":
        return calls.fileset_list()
    if operation == "fileset_delete":
        return calls.fileset_delete(name)
    if operation == "file_upload":
        return calls.file_upload(name, payload)
    if operation == "file_download":
        return calls.file_download(name)
    if operation == "file_list":
        return calls.file_list()
    return calls.file_delete(name)


def _secrets(operation: str, calls: Operations, name: str) -> int:
    if operation == "create":
        return calls.secret_create(name)
    if operation == "get":
        return calls.secret_get(name)
    if operation == "list":
        return calls.secret_list()
    if operation == "update":
        return calls.secret_update(name)
    if operation == "delete":
        return calls.secret_delete(name)
    return calls.secret_value(name)


def _auth(operation: str, calls: Operations, log: HistoryLog) -> int:
    if operation == "discovery":
        return calls.discovery()
    if operation == "authz":
        return calls.authz()
    if operation == "access_key_create":
        status, token = calls.access_key_create()
        log.deny(token)
        return status
    if operation == "access_key_list":
        return calls.access_key_list()
    return calls.access_key_revoke(object_name("auth", "key", 1))


class LiveOperations:
    """SDK calls using the workload identity already mounted in the task."""

    def __init__(self, workspace: str) -> None:
        client = get_task_nemo_client("jobs", workspace=workspace)
        self._workspace = workspace
        self._client = client
        self._files = FilesClient.from_client(client)
        self._secrets = SecretsClient.from_client(client)
        self._iam = IAMClient.from_client(client)
        self._keys = AccessKeysClient.from_client(client)
        self._filesets: list[str] = []
        self._secrets_created: list[str] = []
        self._jtis: list[str] = []
        self._uploaded: list[tuple[str, str]] = []

    def fileset_create(self, name: str) -> int:
        status = self._status(
            lambda: self._files.create_fileset(
                workspace=self._workspace, body=CreateFilesetRequest(name=name), exist_ok=True
            ).data()
        )
        if status < 300:
            self._filesets.append(name)
        return status

    def fileset_get(self, name: str) -> int:
        return self._status(lambda: self._files.get_fileset(workspace=self._workspace, name=name).data())

    def fileset_list(self) -> int:
        return self._status(lambda: self._files.list_filesets(workspace=self._workspace))

    def fileset_delete(self, name: str) -> int:
        return self._status(lambda: self._files.delete_fileset(workspace=self._workspace, name=name).data())

    def file_upload(self, name: str, payload: bytes) -> int:
        fileset = self._ensure_fileset(name)
        status = self._status(
            lambda: self._files.upload_file(
                workspace=self._workspace, name=fileset, path="blob", content=payload
            ).data()
        )
        if status < 300:
            self._uploaded.append((fileset, "blob"))
        return status

    def file_download(self, name: str) -> int:
        fileset = self._ensure_fileset(name)
        return self._status(
            lambda: self._files.download_file(workspace=self._workspace, name=fileset, path="blob").read()
        )

    def file_list(self) -> int:
        fileset = self._ensure_fileset(object_name(self._workspace, "list", 1))
        return self._status(lambda: self._files.list_files(workspace=self._workspace, name=fileset).data())

    def file_delete(self, name: str) -> int:
        fileset = self._ensure_fileset(name)
        return self._status(
            lambda: self._files.delete_file(workspace=self._workspace, name=fileset, path="blob").data()
        )

    def secret_create(self, name: str) -> int:
        body = HelixSecretCreateRequest(name=name, value=SecretStr("payload"))
        status = self._status(lambda: self._secrets.create_secret(workspace=self._workspace, body=body).data())
        if status < 300:
            self._secrets_created.append(name)
        return status

    def secret_get(self, name: str) -> int:
        return self._status(lambda: self._secrets.get_secret(workspace=self._workspace, name=name).data())

    def secret_list(self) -> int:
        return self._status(lambda: self._secrets.list_secrets(workspace=self._workspace))

    def secret_update(self, name: str) -> int:
        body = HelixSecretUpdateRequest(value=SecretStr("payload-next"))
        return self._status(lambda: self._secrets.update_secret(workspace=self._workspace, name=name, body=body).data())

    def secret_delete(self, name: str) -> int:
        return self._status(lambda: self._secrets.delete_secret(workspace=self._workspace, name=name))

    def secret_value(self, name: str) -> int:
        return self._status(lambda: self._secrets.access_secret(workspace=self._workspace, name=name).data())

    def discovery(self) -> int:
        try:
            found = discover_nhx_config(str(self._client.base_url))
        except (OSError, ValueError, NemoClientError):
            return 0
        return 200 if found.auth_enabled else 503

    def authz(self) -> int:
        body = AuthzRequest(input={"method": "GET", "path": f"/v2/workspaces/{self._workspace}/filesets"})
        return self._status(lambda: self._iam.evaluate_authorization(entrypoint="allow", body=body).data())

    def access_key_create(self) -> tuple[int, str]:
        try:
            created = self._keys.create_access_key(body=AccessKeyCreateRequest()).data()
        except NemoHTTPError as exc:
            return exc.status_code, ""
        except NemoClientError:
            return 0, ""
        self._jtis.append(created.jti)
        return 200, created.token

    def access_key_list(self) -> int:
        return self._status(lambda: self._keys.list_access_keys().data())

    def access_key_revoke(self, jti: str) -> int:
        target = self._jtis[-1] if self._jtis else jti
        return self._status(lambda: self._keys.revoke_access_key(jti=target).data())

    def cleanup(self) -> None:
        for jti in self._jtis:
            self._status(lambda key=jti: self._keys.revoke_access_key(jti=key).data())
        for name in self._secrets_created:
            self._status(
                lambda secret_name=name: self._secrets.delete_secret(workspace=self._workspace, name=secret_name)
            )
        for fileset, path in self._uploaded:
            self._status(
                lambda fileset_name=fileset, object_path=path: self._files.delete_file(
                    workspace=self._workspace, name=fileset_name, path=object_path
                ).data()
            )
        for name in self._filesets:
            self._status(
                lambda fileset_name=name: self._files.delete_fileset(
                    workspace=self._workspace, name=fileset_name
                ).data()
            )

    def _ensure_fileset(self, name: str) -> str:
        if name not in self._filesets:
            self.fileset_create(name)
        return name

    def _status(self, call: Callable[[], object]) -> int:
        try:
            call()
        except NemoHTTPError as exc:
            return exc.status_code
        except NemoClientError:
            return 0
        return 200


def live_operations() -> LiveOperations:
    return LiveOperations(os.environ["NEMO_JOB_WORKSPACE"])
