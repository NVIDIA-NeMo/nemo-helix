# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
from urllib.parse import urlsplit

import yaml

type YamlScalar = str | int | float | bool | None
type YamlValue = YamlScalar | YamlMapping | list[YamlValue]
type YamlMapping = dict[str, YamlValue]


def _normalize_yaml(value: object) -> YamlValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_normalize_yaml(item) for item in value]
    if isinstance(value, dict):
        normalized: YamlMapping = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"expected a string YAML key, got {key!r}")
            normalized[key] = _normalize_yaml(item)
        return normalized
    raise ValueError(f"unsupported YAML value: {type(value).__name__}")


def _mapping(value: YamlValue, description: str) -> YamlMapping:
    if not isinstance(value, dict):
        raise ValueError(f"expected {description} to be a mapping")
    return value


def _list(value: YamlValue, description: str) -> list[YamlValue]:
    if not isinstance(value, list):
        raise ValueError(f"expected {description} to be a list")
    return value


def _mapping_at(config: YamlMapping, *path: str) -> YamlMapping:
    current = config
    for key in path:
        current = _mapping(current.get(key), ".".join(path))
    return current


def _url_with_port(value: YamlValue, gateway_port: int) -> str:
    if not isinstance(value, str):
        raise ValueError("expected an external OIDC URL")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname != "127.0.0.1":
        raise ValueError(f"expected an HTTPS loopback URL, got {value!r}")
    return parsed._replace(netloc=f"127.0.0.1:{gateway_port}").geturl()


def _single_matching(items: list[YamlValue], **expected: str) -> YamlMapping:
    mappings = [_mapping(item, "config entry") for item in items]
    matches = [item for item in mappings if all(item.get(key) == value for key, value in expected.items())]
    if len(matches) != 1:
        description = ", ".join(f"{key}={value!r}" for key, value in expected.items())
        raise ValueError(f"expected exactly one config entry matching {description}; found {len(matches)}")
    return matches[0]


def render_authentik_compose_e2e_config(
    source: Path,
    output: Path,
    *,
    workload_network: str,
    gateway_tls_volume: str,
    gateway_port: int,
) -> None:
    config = _mapping(_normalize_yaml(yaml.safe_load(source.read_text(encoding="utf-8"))), "platform config")

    platform = _mapping_at(config, "platform")
    platform["advertised_base_url"] = _url_with_port(platform.get("advertised_base_url"), gateway_port)

    oidc = _mapping_at(config, "auth", "oidc")
    public_client = _mapping_at(oidc, "public_client")
    for field in ("authorization_endpoint", "token_endpoint", "device_authorization_endpoint"):
        public_client[field] = _url_with_port(public_client.get(field), gateway_port)
    confidential_client = _mapping_at(oidc, "confidential_client")
    for field in ("login_redirect_uri", "authorization_endpoint"):
        confidential_client[field] = _url_with_port(confidential_client.get(field), gateway_port)
    oidc["additional_issuers"] = [
        _url_with_port(issuer, gateway_port)
        if isinstance(issuer, str) and urlsplit(issuer).hostname == "127.0.0.1"
        else issuer
        for issuer in _list(oidc.get("additional_issuers"), "auth.oidc.additional_issuers")
    ]
    workload = _mapping_at(oidc, "workload")
    workload["subject_issuers"] = [
        _url_with_port(issuer, gateway_port)
        if isinstance(issuer, str) and urlsplit(issuer).hostname == "127.0.0.1"
        else issuer
        for issuer in _list(workload.get("subject_issuers"), "auth.oidc.workload.subject_issuers")
    ]

    deployment_executor = _single_matching(
        _list(_mapping_at(config, "deployments").get("executors"), "deployments.executors"),
        name="auth-idp-docker",
        backend="docker",
    )
    deployment_config = _mapping(deployment_executor.get("config"), "deployment executor config")
    deployment_config["network"] = workload_network
    deployment_mount = _single_matching(
        _list(deployment_config.get("additional_volume_mounts"), "deployment volume mounts"),
        mount_path="/etc/nhx/gateway-tls",
    )
    deployment_mount["volume_name"] = gateway_tls_volume

    job_executor = _single_matching(
        _list(_mapping_at(config, "jobs").get("executors"), "jobs.executors"),
        provider="cpu",
        profile="workload",
        backend="docker",
    )
    job_config = _mapping(job_executor.get("config"), "job executor config")
    job_storage = _mapping(job_config.get("storage"), "job executor storage")
    job_mount = _single_matching(
        _list(job_storage.get("additional_volume_mounts"), "job volume mounts"),
        mount_path="/etc/nhx/gateway-tls",
    )
    job_mount["volume_name"] = gateway_tls_volume

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workload-network", required=True)
    parser.add_argument("--gateway-tls-volume", required=True)
    parser.add_argument("--gateway-port", type=int, required=True)
    args = parser.parse_args()
    render_authentik_compose_e2e_config(
        args.source,
        args.output,
        workload_network=args.workload_network,
        gateway_tls_volume=args.gateway_tls_volume,
        gateway_port=args.gateway_port,
    )


if __name__ == "__main__":
    main()
