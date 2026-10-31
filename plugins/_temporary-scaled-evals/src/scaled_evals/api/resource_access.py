# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceAccess:
    owner_id: str
    include_all: bool
    internal: bool = False

    def can_read(self, owner_id: str | None, visibility: str) -> bool:
        return self.can_write(owner_id) or visibility in {"public", "team", "org"}

    def can_write(self, owner_id: str | None) -> bool:
        return self.include_all or owner_id == self.owner_id


INTERNAL_ACCESS = ResourceAccess(owner_id="", include_all=True, internal=True)
