# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from nemo_helix_ext.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHENTIK_ROOT = REPO_ROOT / "contrib/auth/authentik"
AUTHENTIK_COMPOSE_PROJECT_PREFIX = "authentik-e2e"
AUTHENTIK_WORKLOAD_NETWORK_NAME = f"{AUTHENTIK_COMPOSE_PROJECT_PREFIX}-${{gateway_port}}-workload"
AUTHENTIK_GATEWAY_TLS_VOLUME_NAME = f"{AUTHENTIK_COMPOSE_PROJECT_PREFIX}-${{gateway_port}}-gateway-tls"
AUTHENTIK_GATEWAY_BASE_URL = "${gateway_url}"
AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD = os.environ.get(
    "AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD", "svc-nemo-token-secret-e2e"
)
_GATEWAY_TLS_OPENSSL_CONFIG = """[req]
prompt = no
distinguished_name = dn
x509_extensions = v3_req

[dn]
CN = nemo-gateway

[v3_req]
basicConstraints = critical, CA:TRUE
keyUsage = critical, digitalSignature, keyEncipherment, keyCertSign
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = nemo-gateway
IP.1 = 127.0.0.1
"""


def _authentik_relative_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / value.removeprefix("./")


def _blueprint_output_dir(*, root: Path) -> Path:
    return _authentik_relative_path(os.environ.get("AUTHENTIK_BLUEPRINT_DIR", "./.generated/blueprints"), root=root)


def _gateway_tls_dir(*, root: Path) -> Path:
    return _authentik_relative_path(os.environ.get("AUTHENTIK_GATEWAY_TLS_DIR", "./.generated/gateway-tls"), root=root)


def authentik_gateway_tls_ca_bundle(*, root: Path = AUTHENTIK_ROOT) -> Path:
    return _gateway_tls_dir(root=root) / "tls.crt"


AUTHENTIK_GATEWAY_TLS_CA_BUNDLE = str(authentik_gateway_tls_ca_bundle())


def _workload_token_private_key_file(*, root: Path) -> Path:
    return root / ".generated/workload-token-private-key.pem"


def _ensure_workload_token_private_key(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)


def _ensure_gateway_tls_certificate(tls_dir: Path) -> None:
    cert_path = tls_dir / "tls.crt"
    key_path = tls_dir / "tls.key"
    if cert_path.exists() and key_path.exists():
        return

    tls_dir.mkdir(parents=True, exist_ok=True)
    (tls_dir / "openssl.cnf").write_text(_GATEWAY_TLS_OPENSSL_CONFIG, encoding="utf-8")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "nemo-gateway")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_cert_sign=True,
                key_agreement=False,
                content_commitment=False,
                data_encipherment=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.DNSName("nemo-gateway"),
                    x509.IPAddress(ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.chmod(0o600)
    cert_path.chmod(0o644)


def prepare_authentik_compose_inputs(*, root: Path = AUTHENTIK_ROOT) -> None:
    """Prepare generated files required before Docker Compose can start."""
    blueprint_source = root / "helm/files/blueprints/nemo.yaml"
    blueprint_output = _blueprint_output_dir(root=root) / "nemo.yaml"
    blueprint_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(blueprint_source, blueprint_output)
    _ensure_workload_token_private_key(_workload_token_private_key_file(root=root))
    _ensure_gateway_tls_certificate(_gateway_tls_dir(root=root))


AUTHENTIK_DOCKER_E2E_CONFIG = pytest.mark.e2e_config(
    "contrib/auth/authentik/config/platform-compose-authentik.yaml",
    {
        "auth": {
            "access_keys": {
                "enabled": True,
            },
            "oidc": {
                "additional_issuers": [
                    "http://authentik-server:9000/application/o/nemo/",
                    "${gateway_url}/application/o/nemo-cli/",
                    "${gateway_url}/application/o/nemo/",
                ],
                "token_endpoint": "${gateway_url}/application/o/token/",
                "device_authorization_endpoint": "${gateway_url}/application/o/device/",
                "workload_subject_issuers": [
                    "http://authentik-server:9000/application/o/nemo-workload/",
                    "https://nemo-gateway:8080/application/o/nemo-workload/",
                    "${gateway_url}/application/o/nemo-workload/",
                ],
            },
        },
    },
    {
        "jobs": {
            "executors": [
                {
                    "provider": "cpu",
                    "profile": "workload",
                    "backend": "docker",
                    "config": {
                        "cleanup_completed_jobs_immediately": False,
                        "launcher_tool_path": "/tools/jobs-launcher",
                        "env": {
                            "SSL_CERT_FILE": "/etc/nhx/gateway-tls/tls.crt",
                            "REQUESTS_CA_BUNDLE": "/etc/nhx/gateway-tls/tls.crt",
                        },
                        "storage": {
                            "additional_volume_mounts": [
                                {
                                    "volume_name": AUTHENTIK_GATEWAY_TLS_VOLUME_NAME,
                                    "mount_path": "/etc/nhx/gateway-tls",
                                }
                            ]
                        },
                    },
                }
            ]
        }
    },
    {
        "deployments": {
            "default_executor": "auth-idp-docker",
            "executors": [
                {
                    "name": "auth-idp-docker",
                    "backend": "docker",
                    "config": {
                        "pull_images": False,
                        "network": AUTHENTIK_WORKLOAD_NETWORK_NAME,
                        "endpoint_mode": "network",
                        "docker_timeout": 600,
                        "oneshot_observe_timeout_seconds": 30,
                        "additional_volume_mounts": [
                            {
                                "volume_name": AUTHENTIK_GATEWAY_TLS_VOLUME_NAME,
                                "mount_path": "/etc/nhx/gateway-tls",
                                "read_only": True,
                            }
                        ],
                    },
                }
            ],
            "controller": {
                "interval_seconds": 2,
                "terminal_orphan_grace_seconds": 300,
            },
        }
    },
    harness={
        "backend": "docker_compose",
        "compose_file": "contrib/auth/authentik/compose/docker-compose.yml",
        "compose_project_prefix": AUTHENTIK_COMPOSE_PROJECT_PREFIX,
        "lifecycle": "fresh",
        "dynamic_ports": {
            "gateway": {
                "host": "127.0.0.1",
                "scheme": "https",
            }
        },
        "service_url": AUTHENTIK_GATEWAY_BASE_URL,
        "auth_ready_url": f"{AUTHENTIK_GATEWAY_BASE_URL}/health/gateway/ready",
        "env": {
            "AUTHENTIK_GATEWAY_PORT": "${gateway_port}",
            "AUTHENTIK_GATEWAY_TLS_VOLUME": AUTHENTIK_GATEWAY_TLS_VOLUME_NAME,
            "AUTHENTIK_WORKLOAD_NETWORK_NAME": AUTHENTIK_WORKLOAD_NETWORK_NAME,
            "AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD": AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD,
            NHX_CLIENT_SSL_CERT_FILE_ENVVAR: AUTHENTIK_GATEWAY_TLS_CA_BUNDLE,
        },
    },
)

AUTHENTIK_DOCKER_PYTESTMARK = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_docker,
    pytest.mark.auth_idp_runtime("authentik-compose"),
    pytest.mark.e2e,
    AUTHENTIK_DOCKER_E2E_CONFIG,
    pytest.mark.xdist_group("idp-live"),
]
