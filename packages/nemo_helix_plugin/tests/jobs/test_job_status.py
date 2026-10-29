# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.jobs.schemas import HelixJobStatus


def test_queued_is_non_terminal():
    assert HelixJobStatus.QUEUED.value == "queued"
    assert HelixJobStatus.QUEUED in HelixJobStatus.non_terminals()
    assert not HelixJobStatus.QUEUED.is_terminal()


def test_every_status_is_terminal_or_non_terminal():
    assert set(HelixJobStatus.terminals()) | set(HelixJobStatus.non_terminals()) == set(HelixJobStatus)


@pytest.mark.parametrize(
    "new_status",
    [
        HelixJobStatus.PENDING,
        HelixJobStatus.ACTIVE,
        HelixJobStatus.CANCELLING,
        HelixJobStatus.CANCELLED,
        HelixJobStatus.ERROR,
    ],
)
def test_queued_can_transition_out(new_status: HelixJobStatus):
    assert HelixJobStatus.QUEUED.can_transition_to(new_status)


def test_created_can_transition_to_queued():
    assert HelixJobStatus.CREATED.can_transition_to(HelixJobStatus.QUEUED)


def test_queued_cannot_complete_or_return_to_created():
    assert not HelixJobStatus.QUEUED.can_transition_to(HelixJobStatus.COMPLETED)
    assert not HelixJobStatus.QUEUED.can_transition_to(HelixJobStatus.CREATED)
