# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process native Switchyard bindings keyed by ``workspace/name``."""

from __future__ import annotations

from nemo_switchyard._native_host import NativeBinding

BindingKey = tuple[str, str]

# Algorithms may keep routing state, so bindings are scoped to one VirtualModel
# and config type rather than shared by config hash.
BINDINGS: dict[BindingKey, NativeBinding] = {}
VM_BINDING_KEYS: dict[str, list[BindingKey]] = {}


def clear_all() -> None:
    BINDINGS.clear()
    VM_BINDING_KEYS.clear()


def replace_vm_bindings(
    vm_key: str,
    bindings: dict[BindingKey, NativeBinding],
) -> None:
    for key in VM_BINDING_KEYS.get(vm_key, []):
        BINDINGS.pop(key, None)
    BINDINGS.update(bindings)
    VM_BINDING_KEYS[vm_key] = list(bindings)


def remove_vm_bindings(vm_key: str) -> None:
    for key in VM_BINDING_KEYS.pop(vm_key, []):
        BINDINGS.pop(key, None)
