# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers that do not call a platform."""

from support import accept_report, chat_target, report_path, task_image_from_launcher


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


def test_report_path_is_unique_to_the_run() -> None:
    assert report_path("job", "abc") == "job-abc.json"


def test_accept_report_rejects_a_stale_fileset_object() -> None:
    stale = {"report_id": "old", "mode": "anonymous"}
    current = {"report_id": "new", "mode": "on_behalf_of"}
    assert accept_report(stale, "new") is None
    assert accept_report(current, "new") == current
    assert accept_report(None, "new") is None
