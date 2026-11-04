# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in image/Compose smoke test; no running Helix or gateway is needed.

Uses synthetic credentials and creates/removes only a unique test project.
On Docker VMs, pytest's --basetemp must be in a host directory shared with Docker.
"""

import json
import os
import secrets
import subprocess
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CREDENTIAL_DRIVER_COMPOSE_TEST") != "1",
    reason="Set RUN_CREDENTIAL_DRIVER_COMPOSE_TEST=1 and NEMO_HELIX_IMAGE to test a locally built API image.",
)

EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "openshell"
MODULE = "nemo_deployments_plugin.backends.openshell.credential_driver"

# Executed in the actual API image. A synthetic signed scope is registered through
# the sidecar's real UDS gRPC server. No real token or service credential is used.
REGISTER = """
import asyncio
import json
from pathlib import Path
import grpc
from nemo_deployments_plugin.backends.openshell.credential_driver import wire
from nemo_deployments_plugin.backends.openshell.credential_driver.registry import BindingRegistry, ProviderOwner
from nemo_deployments_plugin.backends.openshell.credential_driver.tickets import BindingScope, RegistrationTickets

async def run():
    tickets = RegistrationTickets(Path('/run/nemo-driver-credentials/private/registration.key').read_bytes(), 'sidecar-smoke')
    owner = ProviderOwner(workspace='gateway-test', provider='smoke', provider_id='smoke-id', credential_key='API_TOKEN')
    ticket = tickets.sign(BindingScope(gateway_id='sidecar-smoke', gateway_workspace=owner.workspace,
        provider=owner.provider, credential_key=owner.credential_key, secret_workspace='test', secret_name='token',
        job_id='smoke-job', principal_id='smoke-user'))
    async with grpc.aio.insecure_channel('unix:/run/nemo-credential-driver/private/driver.sock') as channel:
        request_name, response_name = wire.METHODS['StoreCredential']
        call = channel.unary_unary('/'+wire.SERVICE+'/StoreCredential',
            request_serializer=wire.message_type(request_name).SerializeToString,
            response_deserializer=wire.message_type(response_name).FromString)
        response = wire.data(await call(wire.message(request_name, {**owner.model_dump(), 'value': ticket.get_secret_value()}), timeout=5))
    handle = response['handle']['handle']
    registry = BindingRegistry(Path('/var/lib/nemo-credential-driver/private/bindings.db'), tickets)
    assert registry.resolve(owner, handle).claims.secret_name == 'token'
    print(json.dumps({'handle': handle, 'owner': owner.model_dump()}))

asyncio.run(run())
"""

VERIFY = """
import json, sys
from pathlib import Path
from nemo_deployments_plugin.backends.openshell.credential_driver.registry import BindingRegistry, ProviderOwner
from nemo_deployments_plugin.backends.openshell.credential_driver.tickets import RegistrationTickets
from nemo_deployments_plugin.backends.openshell.credential_driver.server import read_private_file
binding = json.loads(sys.argv[1])
tickets = RegistrationTickets(read_private_file(Path('/run/nemo-driver-credentials/private/registration.key')), 'sidecar-smoke')
registry = BindingRegistry(Path('/var/lib/nemo-credential-driver/private/bindings.db'), tickets)
assert registry.resolve(ProviderOwner(**binding['owner']), binding['handle']).claims.secret_name == 'token'
assert read_private_file(Path('/run/nemo-driver-credentials/private/helix.token')) == b'synthetic-rotated-token'
"""


def test_compose_sidecar_registration_crash_recovery_and_token_rotation(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    image = os.environ.get("NEMO_HELIX_IMAGE")
    assert image, "Set NEMO_HELIX_IMAGE to the API image built from this checkout."
    credentials = tmp_path / "mounted-credentials"
    credentials.mkdir(mode=0o700)
    for name, value in (("registration.key", secrets.token_bytes(32)), ("helix.token", b"synthetic-smoke-token")):
        path = credentials / name
        path.touch(mode=0o600)
        path.write_bytes(value)
    env = {
        **os.environ,
        "NEMO_HELIX_IMAGE": image,
        "NEMO_DRIVER_CREDENTIALS_DIR": str(credentials),
        "NEMO_DRIVER_HELIX_URL": "https://helix-sidecar-smoke.invalid",
        "NEMO_DRIVER_GATEWAY_ID": "sidecar-smoke",
    }
    compose = [
        "docker",
        "compose",
        "-p",
        f"nemo-credential-smoke-{uuid.uuid4().hex[:8]}",
        "-f",
        str(EXAMPLE / "docker-compose.yml"),
        "-f",
        str(EXAMPLE / "docker-compose.credentials.yml"),
    ]

    def run(*args: str, script: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [*compose, *args], env=env, input=script, text=True, capture_output=True, check=False, timeout=120
        )
        if check and result.returncode:
            pytest.fail(f"Compose {args[0]} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
        return result

    # A teardown failure is reported separately from the original test failure.
    # Only this disposable project's synthetic registration history is removed.
    request.addfinalizer(lambda: run("down", "--volumes", "--remove-orphans"))
    run("up", "-d", "--wait", "--wait-timeout", "90", "credential-driver")
    stored = json.loads(run("exec", "-T", "credential-driver", "python", "-", script=REGISTER).stdout)
    before = run("ps", "-q", "credential-driver").stdout.strip()
    assert before and len(before.splitlines()) == 1, "Expected one credential-driver container."

    def restart_count() -> int:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.RestartCount}}", before],
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        return int(result.stdout)

    restarts = restart_count()
    # Compose's init process is PID 1. Kill its driver child to simulate a
    # crash without Docker treating it as an explicit operator stop.
    run(
        "exec",
        "-T",
        "credential-driver",
        "python",
        "-c",
        "import os, signal; from pathlib import Path; "
        f"needle = {MODULE + '.server'!r}.encode(); "
        "pids = [int(p.parent.name) for p in Path('/proc').glob('[0-9]*/cmdline') "
        "if p.parent.name != '1' and needle in p.read_bytes().split(bytes([0]))]; "
        "assert len(pids) == 1, pids; os.kill(pids[0], signal.SIGKILL)",
        check=False,  # The crash may terminate this exec session too; verify RestartCount below.
    )
    deadline = time.monotonic() + 60
    while True:
        ready = run(
            "exec",
            "-T",
            "credential-driver",
            "python",
            "-m",
            f"{MODULE}.sidecar",
            "healthcheck",
            "--socket-path",
            "/run/nemo-credential-driver/private/driver.sock",
            check=False,
        )
        if ready.returncode == 0 and restart_count() > restarts:
            break
        if time.monotonic() > deadline:
            pytest.fail(f"Sidecar did not recover after SIGKILL: {ready.stderr}")
        time.sleep(0.5)
    assert run("ps", "-q", "credential-driver").stdout.strip() == before
    (credentials / "helix.token").write_bytes(b"synthetic-rotated-token")
    run("run", "--rm", "--no-deps", "credential-init")
    run("exec", "-T", "credential-driver", "python", "-c", VERIFY, json.dumps(stored))
    run("up", "-d", "--no-deps", "--force-recreate", "--wait", "--wait-timeout", "90", "credential-driver")
    run("exec", "-T", "credential-driver", "python", "-c", VERIFY, json.dumps(stored))
