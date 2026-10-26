# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for redirects on the mounted Fern docs site."""

import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

DOCS_CONFIG = Path(__file__).resolve().parents[1] / "fern" / "docs.yml"


def test_redirect_destinations_include_instance_basepath() -> None:
    config = yaml.safe_load(DOCS_CONFIG.read_text(encoding="utf-8"))
    for instance in config["instances"]:
        for host in (instance["url"], instance["custom-domain"]):
            basepath = urlsplit(f"https://{host}").path.rstrip("/")
            for redirect in config["redirects"]:
                destination = redirect["destination"]
                if destination.startswith("/"):
                    assert destination.startswith(f"{basepath}/"), redirect
                    assert not destination.startswith(f"{basepath}{basepath}/"), redirect


def test_redirect_sources_remain_site_relative() -> None:
    config = yaml.safe_load(DOCS_CONFIG.read_text(encoding="utf-8"))
    for redirect in config["redirects"]:
        assert redirect["source"].startswith("/"), redirect
        assert not redirect["source"].startswith("/nemo-helix/"), redirect


def test_redirects_preserve_wildcard_parameters() -> None:
    config = yaml.safe_load(DOCS_CONFIG.read_text(encoding="utf-8"))
    for redirect in config["redirects"]:
        source_parameters = re.findall(r":[A-Za-z_][A-Za-z_0-9]*[+*]?", redirect["source"])
        destination_parameters = re.findall(r":[A-Za-z_][A-Za-z_0-9]*[+*]?", redirect["destination"])
        assert destination_parameters == source_parameters, redirect
