# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stand-in for a remote build service's step program, for ``e2e/test_builder_remote.py``.

A build service's step image ships a small program around the public driver, ``step_main``, and its own
``RemoteBuildService``. This is that program with a service that builds nothing and calls nothing: each build ends
as a file in its build context says. The test mounts this file into the remote-build step's pod from a ConfigMap,
and the step runs it in the release's own ``nhx-builder-tasks`` image, so it is in no image and no package.

The file is :data:`OUTCOME_FILE`, at the root of the build context, and holds one JSON object:

- ``{"outcome": "ready", "digest": "sha256:<64 hex>", "image_id": "<the service's handle>"}``: the build is done,
  and the service published the image at that digest, in the repository the deployment names.
- ``{"outcome": "failed", "reason": "<why>"}``: the service reports the build failed.
- ``{"outcome": "hang"}``: the build never ends, until the step's job is cancelled or the build times out.

It prints nothing and sets up no logging: the driver's last lines are the ones Jobs keeps of a failed step's log.
"""

from __future__ import annotations

import json

from nemo_builder_plugin.run.remote import PermanentError, RemoteBuild, RemoteBuildStatus, RemoteImage, step_main
from nemo_builder_plugin.steps import RemoteBuildStepConfig

OUTCOME_FILE = "nhx-e2e-outcome.json"
OUTCOMES = ("ready", "failed", "hang")
#: What every build's ID starts with.
BUILD_ID_PREFIX = "e2e-build-"
#: What the stand-in reports about each image it "built", beside its handle.
ATTRIBUTES = {"stand-in": "true"}


class ScriptedBuildService:
    """A ``RemoteBuildService`` in memory: each build ends as its context's outcome file says."""

    def __init__(self, destination: str) -> None:
        #: ``<registry>/<repository>``: where the deployment's build service publishes.
        self._destination = destination
        self._builds: dict[str, dict[str, str]] = {}

    def start(self, build: RemoteBuild) -> str:
        try:
            outcome = json.loads((build.context / OUTCOME_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # Not what the file holds: a reason is read by everyone who can read the image.
            raise PermanentError(f"the build context's {OUTCOME_FILE} can't be read ({type(exc).__name__})") from exc
        if not isinstance(outcome, dict) or outcome.get("outcome") not in OUTCOMES:
            raise PermanentError(f"the build context's {OUTCOME_FILE} names none of: {', '.join(OUTCOMES)}")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in outcome.items()):
            raise PermanentError(f"the build context's {OUTCOME_FILE} must hold strings only")
        # The same ID for a start the driver repeats: the key is one image in one attempt of one job.
        build_id = f"{BUILD_ID_PREFIX}{build.key[:16]}"
        self._builds[build_id] = outcome
        return build_id

    def status(self, build_id: str) -> RemoteBuildStatus:
        outcome = self._builds.get(build_id)
        if outcome is None:
            raise PermanentError(f"no build {build_id} was started, or it was cancelled")
        if outcome["outcome"] == "ready":
            image = RemoteImage(
                reference=f"{self._destination}@{outcome.get('digest', '')}",
                image_id=outcome.get("image_id"),
                attributes=ATTRIBUTES,
            )
            return RemoteBuildStatus("done", image=image)
        if outcome["outcome"] == "failed":
            return RemoteBuildStatus("failed", detail=outcome.get("reason"))
        return RemoteBuildStatus("building", detail="held until the job is cancelled")

    def cancel(self, build_id: str) -> None:
        # Returns normally for a build it doesn't know, as the driver's contract asks.
        self._builds.pop(build_id, None)


def make_service(config: RemoteBuildStepConfig, credential: str | None) -> ScriptedBuildService:
    """What ``step_main`` asks for. No credential reaches it: the deployment names no secret, and none is needed."""
    return ScriptedBuildService(config.destination)


if __name__ == "__main__":
    raise SystemExit(step_main(make_service))
