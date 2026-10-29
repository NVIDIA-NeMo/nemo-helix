# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image preparation state stays consistent, and both preparers satisfy one protocol."""

import pytest
from nemo_evals.harbor.remote.preparation import (
    ImagePreparation,
    ImagePreparer,
    PreparationErrorInfo,
    TaskPreparation,
    prepare_task_images,
)
from nemo_evals.jobs.harbor_task_spec import PreparedHarborEnvironment, PreparedImage
from pydantic import ValidationError

DIGEST = "a" * 64
IMAGE = PreparedImage(image_ref=f"registry.invalid/sort@sha256:{DIGEST}", preparation_key="k")
ENVIRONMENT = PreparedHarborEnvironment(task_revision_digest="r", archive_digest="a", agent=IMAGE)
AGENT = ImagePreparation(type="agent", preparation_key="k", source="prebuilt", prepared=IMAGE)


def test_both_preparers_satisfy_the_protocol(fake_preparer):
    _check: ImagePreparer = prepare_task_images
    _check_fake: ImagePreparer = fake_preparer
    assert callable(_check) and callable(_check_fake)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"status": "ready", "images": {"agent": AGENT}}, "environment is set exactly"),
        ({"status": "pending", "environment": ENVIRONMENT}, "environment is set exactly"),
        ({"status": "failed"}, "error is set exactly"),
        ({"status": "pending", "error": PreparationErrorInfo(code="build_failed", detail="x")}, "error is set exactly"),
        ({"status": "ready", "environment": ENVIRONMENT}, "a ready task has an agent image"),
        ({"status": "pending", "images": {"verifier": AGENT}}, "has type 'agent'"),
    ],
)
def test_inconsistent_state_is_rejected(fields, match):
    with pytest.raises(ValidationError, match=match):
        TaskPreparation(**fields)


def test_unknown_error_code_is_rejected():
    with pytest.raises(ValidationError):
        PreparationErrorInfo(code="no_such_code", detail="x")  # ty: ignore[invalid-argument-type]


def test_ready_state_round_trips():
    ready = TaskPreparation(status="ready", resource_class="cpu", images={"agent": AGENT}, environment=ENVIRONMENT)
    assert TaskPreparation.model_validate_json(ready.model_dump_json()) == ready


def test_an_error_may_name_a_compose_service_but_an_image_record_may_not():
    error = PreparationErrorInfo(code="image_unavailable", detail="no ready db image", image_type="service")
    assert PreparationErrorInfo.model_validate_json(error.model_dump_json()) == error
    with pytest.raises(ValidationError):
        ImagePreparation(type="service", preparation_key="k", source="prebuilt")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValidationError):
        TaskPreparation(status="pending", images={"service": AGENT})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "ref", ["registry/sort:latest", "registry/sort@sha256:abc", f"registry/sort@sha256:{'A' * 64}"]
)
def test_prepared_image_requires_a_digest(ref):
    with pytest.raises(ValidationError):
        PreparedImage(image_ref=ref, preparation_key="k")


def test_environment_defaults():
    assert ENVIRONMENT.verifier is None and ENVIRONMENT.resource_class == "cpu" and ENVIRONMENT.compose_images == {}
    assert PreparedHarborEnvironment.model_validate_json(ENVIRONMENT.model_dump_json()) == ENVIRONMENT
