# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permission vocabulary for the builder's hand-written routes.

Note what is *not* here: a permission to choose an execution profile. ``validate_job_spec`` checks
only that a ``(provider, profile)`` pair resolves, with no authorization -- and ``jobs.create`` is
in the default Editor role. So anyone able to submit a build can also submit a raw
``CreateHelixJobRequest`` naming one of the builder's profiles, with an image and command of their
own, and run as that profile's ServiceAccount.

**That is why no step's identity can reach anything worth having** (Requirement 7). No credential
and no key exists in the build namespace. ``nhx-build-push`` holds no RBAC at all: the push step
gets everything it publishes with from the credential broker, which grants registry access -- and
signs -- only for a job's own ``pending`` rows under its own workspace. A raw job has no rows, so
it gets nothing. ``build-control``'s grant to create pods is held by an admission policy to pods
that run as the namespace's default ServiceAccount, with no token.

What that leaves open:

- **Same workspace: rows whose job was never created.** The broker ties rows to their job by the
  job's name. A submit that finds the name taken fails its rows, but a submit that dies between
  writing its rows and creating its job leaves them ``pending`` under a name no job holds. A raw
  job that takes it is granted their destinations, and can have them signed and completed with an
  image of its own.
- **Across workspaces: ``build-control``'s reads and deletes.** The admission policy narrows which
  pods it creates, not which it reads or deletes. A raw job under that profile can read any build
  pod's log in the namespace, and delete any build pod; and a sandbox-shaped pod it creates can
  mount any job's slice of the work volume, including a layout another job's push step is about
  to publish.

Closing both is an authorization control in Jobs -- these profiles usable only by
``service:builder``, with no image, command or environment override -- or the builder creating
its pods itself.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import PermissionSet, perm


class BuildPerms(PermissionSet, namespace="builder.builds"):
    CREATE = perm("Submit a container image build")


class ContainerImagePerms(PermissionSet, namespace="builder.container-images"):
    LIST = perm("List container images")
    READ = perm("Read a container image")
