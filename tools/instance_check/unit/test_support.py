# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers that do not call a platform."""

from support import chat_target, task_image_from_launcher


def test_chat_target_uses_the_model_workspace() -> None:
    workspace, model = chat_target("system/meta-llama-3-1-8b-instruct", "check-1")
    assert workspace == "system"
    assert model == "system/meta-llama-3-1-8b-instruct"


def test_chat_target_keeps_a_bare_name_in_the_check_workspace() -> None:
    workspace, model = chat_target("my-model", "check-1")
    assert workspace == "check-1"
    assert model == "my-model"


def test_task_image_replaces_the_launcher_name() -> None:
    image = task_image_from_launcher("nvcr.io/example/nemo-helix-dev/nhx-api:abc123")
    assert image == "nvcr.io/example/nemo-helix-dev/nhx-tasks:abc123"


def test_task_image_ignores_other_launcher_images() -> None:
    assert task_image_from_launcher("nvcr.io/example/jobs-launcher:latest") is None
