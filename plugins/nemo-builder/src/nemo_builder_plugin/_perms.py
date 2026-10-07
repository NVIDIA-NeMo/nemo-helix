# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permission vocabulary for the builder's routes.

No permission covers the build steps' execution profiles: the README's Limitations say what that leaves open.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import PermissionSet, perm


class BuildPerms(PermissionSet, namespace="builder.builds"):
    CREATE = perm("Submit a container image build")


class ContainerImagePerms(PermissionSet, namespace="builder.container-images"):
    LIST = perm("List container images")
    READ = perm("Read a container image")
    #: Held by whoever may submit a build, since the push step completes as the submitter. Anything
    #: that says it acts for the submitter may do the same: nothing ties the call to the image's own job.
    COMPLETE = perm("Complete a pending container image")
