# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kind action pulls OpenSandbox's images before installing it, from a list written out by hand.

If that list fell behind the values files, the pre-pull would fetch the old images and the new ones
would be pulled inside a sandbox's create wait, which is the failure the pre-pull exists to prevent.
"""

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
ACTION = ROOT / ".github" / "actions" / "setup-kind-cluster" / "action.yaml"
EXAMPLES = ROOT / "k8s" / "helm" / "examples" / "opensandbox"
PREPULL_STEP = "Pre-pull OpenSandbox images"


def _prepulled_images() -> set[str]:
    steps = yaml.safe_load(ACTION.read_text())["runs"]["steps"]
    run = next(step["run"] for step in steps if step.get("name") == PREPULL_STEP)
    listed = re.search(r"images=\((.*?)^\s*\)", run, re.DOTALL | re.MULTILINE)
    assert listed, f"{PREPULL_STEP!r} no longer lists its images as images=( ... )"
    images = set()
    for line in listed.group(1).splitlines():
        entry = line.split("#", 1)[0].strip().strip("'\"")
        if entry:
            images.add(entry)
    return images


def _image(values: dict) -> str:
    return f"{values['image']['repository']}:{values['image']['tag']}"


def _opensandbox_images() -> set[str]:
    controller = yaml.safe_load((EXAMPLES / "opensandbox-controller.yaml").read_text())["controller"]
    server = yaml.safe_load((EXAMPLES / "opensandbox-server.yaml").read_text())
    config = tomllib.loads(server["configToml"])
    verify = (EXAMPLES / "verify" / "lib.sh").read_text()
    sandbox_image = re.search(r'^SANDBOX_IMAGE="\$\{SANDBOX_IMAGE:-([^}]+)\}"', verify, re.MULTILINE)
    assert sandbox_image, "verify/lib.sh no longer sets a literal SANDBOX_IMAGE default"
    return {
        _image(controller),
        _image(server["server"]),
        config["runtime"]["execd_image"],
        config["egress"]["image"],
        sandbox_image.group(1),
    }


def test_prepull_list_matches_the_opensandbox_values() -> None:
    assert _prepulled_images() == _opensandbox_images()
