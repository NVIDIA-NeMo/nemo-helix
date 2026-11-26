# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Following a submitted image until it settles (submitting is covered by nemo-evals' image build tests)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pytest
from nemo_builder_plugin.builds import wait_for_image
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_builder_plugin.schema import BuildMount, FileSetSource, ImageSource, SubmitBuildResponse

Status = Literal["pending", "ready", "failed"]


def _image(status: Status) -> ContainerImage:
    return ContainerImage(
        name="victim-1",
        workspace="default",
        status=status,
        registry="registry.test",
        repository="default/victim",
        digest="sha256:" + "a" * 64 if status == "ready" else None,
        provenance=Provenance(
            build_set="victim", revision=1, job="default/builder-victim-1", request_digest="sha256:" + "b" * 64
        ),
    )


class _Builder:
    """Answers each read with the next of *reads*; its job runs until *job_ends_after* reads."""

    def __init__(self, reads: list[Status], *, job_ends_after: int | None = None) -> None:
        self.reads = reads
        self.job_ends_after = job_ends_after
        self.read_count = 0

    async def get_image(self, name: str) -> ContainerImage | None:
        self.read_count += 1
        return _image(self.reads[min(self.read_count, len(self.reads)) - 1])

    async def job_status(self, job: str) -> str | None:
        ended = self.job_ends_after is not None and self.read_count >= self.job_ends_after
        return "error" if ended else "active"

    async def submit(
        self,
        *,
        name: str,
        revision: int,
        source: FileSetSource | ImageSource,
        dockerfile: str | None,
        platform: str | None,
        mounts: Sequence[BuildMount] = (),
    ) -> SubmitBuildResponse:
        raise NotImplementedError

    async def image_platforms(self, image: str) -> list[str]:
        return []


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.parametrize("settled", ["ready", "failed"])
async def test_waits_until_the_image_settles(settled: Status) -> None:
    builder = _Builder(["pending", "pending", settled])

    image = await wait_for_image(builder, _image("pending"), poll_interval=0, sleep=_no_sleep)

    assert image.status == settled
    assert builder.read_count == 3


async def test_stops_waiting_once_the_job_ends_without_the_image() -> None:
    builder = _Builder(["pending"], job_ends_after=2)

    image = await wait_for_image(builder, _image("pending"), poll_interval=0, sleep=_no_sleep)

    assert image.status == "pending"
    assert builder.read_count == 2


async def test_gives_up_after_the_timeout() -> None:
    ticks = iter(range(100))

    with pytest.raises(TimeoutError, match="victim-1"):
        await wait_for_image(
            _Builder(["pending"]), _image("pending"), timeout=3, sleep=_no_sleep, clock=lambda: next(ticks)
        )
