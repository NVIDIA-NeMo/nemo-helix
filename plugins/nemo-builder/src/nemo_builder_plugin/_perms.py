# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Permission vocabulary for the builder's hand-written routes.

Note what is *not* here: a permission to choose an execution profile. RFC 001 records that
``validate_job_spec`` checks only that a `(provider, profile)` pair resolves, with no per-workspace
authorization -- so anyone able to submit a raw ``CreateHelixJobRequest`` can name any profile,
including ``build-control``. That is a residual this plugin does not close, and closing it is an
authorization control in front of Jobs rather than a permission here.
"""

from __future__ import annotations

from nemo_helix_plugin.authz import PermissionSet, perm


class BuildPerms(PermissionSet, namespace="builder.builds"):
    CREATE = perm("Submit a container image build")
    READ = perm("Read a submitted build")


class ContainerImagePerms(PermissionSet, namespace="builder.container-images"):
    LIST = perm("List container images")
    READ = perm("Read a container image")
