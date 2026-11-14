# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceAccess:
    owner_id: str
    include_all: bool
    internal: bool = False

    def can_read(self, owner_id: str | None, visibility: str) -> bool:
        """Allow owner/admin reads and the existing authenticated sharing convention."""
        return self.can_write(owner_id) or visibility in {"public", "team", "org"}

    def can_access(self, owner_id: str | None, visibility: str, *, write: bool) -> bool:
        """Select the same owner/visibility policy for single and batched resource checks."""
        return self.can_write(owner_id) if write else self.can_read(owner_id, visibility)

    def can_write(self, owner_id: str | None) -> bool:
        """Allow mutations only for the effective owner or an authorized admin."""
        return self.include_all or owner_id == self.owner_id


INTERNAL_ACCESS = ResourceAccess(owner_id="", include_all=True, internal=True)
