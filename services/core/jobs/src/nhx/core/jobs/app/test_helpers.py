# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers and constants for jobs service tests.

This module contains test constants, fixtures, and helper functions
that are shared across test files.
"""

from nhx.core.jobs.app.providers import ComputeResources, ComputeResourceSpec, ContainerSpec, CPUExecutionProvider
from nhx.core.jobs.app.schemas import (
    HelixJobSpec,
    HelixJobStepSpec,
)


class TestConstants:
    """Test constants using dictionary structures for better reusability."""

    # Basic values
    SOURCE = "test-source"
    PROJECT = "test-project"
    WORKSPACE = "test-workspace"

    # Descriptions
    DESC_TEST = "A test job for round-trip testing"
    DESC_MODIFIED = "A modified job using helper function"
    DESC_HELPER_MODIFIED = "This job has been modified"

    # Standard spec dictionaries
    SPEC_BASIC = {"task": "fine-tuning", "model": "llama-3.1-8b"}
    SPEC_INFERENCE = {"task": "inference"}
    SPEC_TEST = {"task": "test"}
    SPEC_COMPLEX = {
        "complex": {"learning_rate": 0.001, "batch_size": 32, "epochs": 10},
    }

    # The Jobs API translates `cpu/<profile>` steps into `subprocess/<profile>` steps
    # before validation when a matching subprocess profile is configured (see
    # `translate_cpu_container_steps_to_subprocess` in
    # services/core/jobs/src/nhx/core/jobs/api/v2/jobs/endpoints.py). That translation
    # requires the source step to declare an entrypoint and/or command on its
    # container, since the subprocess backend has no notion of an image to fall back
    # on. Real plugin compilers (Data Designer create, Evaluator metrics, hello-world,
    # etc.) all set both fields; the fixture mirrors that so submissions through the
    # core /apis/jobs/v2/workspaces/{ws}/jobs endpoint validate successfully and we
    # exercise the same translation path the user-facing tutorials do.
    TEST_EXECUTOR = CPUExecutionProvider(
        provider="cpu",
        profile="default",
        container=ContainerSpec(
            image="test-image",
            entrypoint=["python", "-m"],
            command=["nhx.testing.fake_task"],
        ),
        resources=ComputeResources(
            limits=ComputeResourceSpec(
                cpu="5",
                memory="2Gi",
            )
        ),
    )

    # Standard platform_spec dictionaries
    PLATFORM_SPEC = HelixJobSpec(steps=[HelixJobStepSpec(name="basic", executor=TEST_EXECUTOR, config={})])

    # Standard ownership dictionaries
    OWNERSHIP_BASIC = {"user": "test-user", "team": "test-team"}

    # Standard custom_fields dictionaries
    CUSTOM_FIELDS_BASIC = {"priority": "high", "tags": ["ml", "training"]}
    CUSTOM_FIELDS_COMPLEX = {
        "metadata": {"experiment_id": "exp-456"},
    }

    # Standard status_details dictionaries
    STATUS_DETAILS_QUEUED = {"message": "Job queued for processing"}
    STATUS_DETAILS_EMPTY = {}

    # Standard warnings lists
    WARNINGS_MEMORY = [{"type": "memory", "message": "High memory usage detected"}]

    # Update dictionaries for variations
    SPEC_UPDATE = {"hyperparameters": {"learning_rate": 0.002}}

    CUSTOM_FIELDS_UPDATE_HIGH = {"priority": "high"}
    CUSTOM_FIELDS_UPDATE_MEDIUM = {"priority": "medium"}
    CUSTOM_FIELDS_UPDATE_LOW = {"priority": "low"}
