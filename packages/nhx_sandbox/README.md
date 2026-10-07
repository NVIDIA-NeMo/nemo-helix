<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nhx-sandbox

> **Work in progress.** This package is the first slice of a shared sandbox library. It currently holds only the egress-policy code, and its API may change without notice as ownership, reconciliation and provider contracts are added. `sandboxed_gym` still carries its own copy of this code until it migrates.

Shared sandbox semantics for NeMo Helix components that create sandboxes.

- `nhx_sandbox.egress`: provider-neutral, deny-by-default egress policy built from an allow-only list (`EgressAllowlist`, `EgressPolicy`, `build_egress_policy`).
- `nhx_sandbox.opensandbox_policy`: renders that policy into OpenSandbox's create options and verifies the policy a live sandbox reports (`to_opensandbox_policy`, `create_options_with_policy`, `verify_applied_egress`).

The base install depends only on Pydantic, so the package can be installed into third-party runner environments such as Harbor's.
