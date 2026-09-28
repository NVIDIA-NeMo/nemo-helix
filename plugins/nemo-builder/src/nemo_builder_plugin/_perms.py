# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permission vocabulary for the builder's hand-written routes.

Note what is *not* here: a permission to choose an execution profile. ``validate_job_spec`` checks
only that a ``(provider, profile)`` pair resolves, with no authorization -- and ``jobs.create`` is
in the default Editor role. So anyone able to submit a build can also submit a raw
``CreateHelixJobRequest`` naming ``build-push``, with an image and command of their own, and run
as ``nhx-build-push``: read the signing key and the push credential, and publish and sign under
any workspace's path. ``build-control`` is equivalent, since it can create a pod that runs as
``nhx-build-push`` or mounts either Secret.

**This defeats the trust split, and this plugin cannot close it.** Closing it is an
authorization control in Jobs -- these profiles usable only by ``service:builder``, with no
image, command or environment override -- or the builder creating its pods itself.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import PermissionSet, perm


class BuildPerms(PermissionSet, namespace="builder.builds"):
    CREATE = perm("Submit a container image build")


class ContainerImagePerms(PermissionSet, namespace="builder.container-images"):
    LIST = perm("List container images")
    READ = perm("Read a container image")
