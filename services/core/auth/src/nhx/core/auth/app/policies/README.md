<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared Rego Authorization Policies

Both the embedded PDP's managed native OPA child and external OPA evaluate
these same sources. No compiled policy artifact or custom OPA build is needed.
The runtime requires OPA >=1.22.0,<2; automatic binary provisioning pins v1.22.0.

## Entry Points

The `authz` package exposes three main rules:

### `allow` — Request Authorization

Used by the API gateway to authorize HTTP requests.

**Input:**
```json
{
  "principal_id": "user@example.com",
  "principal_email": "user@example.com",
  "principal_groups": ["group1", "group2"],
  "method": "GET",
  "path": "/v1/models",
  "scopes": ["models:read"]
}
```

**Output:**
```json
{
  "allowed": true,
  "headers": {
    "X-NHX-Authorized": "true"
  }
}
```

### `has_permissions` — Permission Check

Used by services to verify if a principal has specific permissions in a workspace.

**Input:**
```json
{
  "principal_id": "user@example.com",
  "workspace": "my-workspace",
  "permissions": ["models.create", "models.update"]
}
```

**Output:**
```json
{
  "allowed": true
}
```

### `has_role` — Role Check

Used by services to check if a principal has a specific role in a workspace. Primary use case is polling for role membership propagation after changes.

**Input:**
```json
{
  "principal_id": "user@example.com",
  "workspace": "my-workspace",
  "role": "Editor"
}
```

**Output:**
```json
{
  "has_role": true
}
```

## File Structure

| File | Description |
|------|-------------|
| `authz.rego` | Main `allow` rule for request authorization |
| `authz_has_permissions.rego` | `has_permissions` rule for permission checks |
| `authz_has_role.rego` | `has_role` rule for role checks |
| `common.rego` | Shared helpers (principals, permissions, roles) |
| `extract.rego` | Input extraction (path, method, workspace) |
| `scopes.rego` | OAuth scope validation |

## Derived data: `data.authz.scope_families`

`scopes.rego` checks whether a scope belongs to a configured family with one lookup in `data.authz.scope_families`
(e.g. `{"platform": true, "models": true}`). The auth service derives it from `data.authz.endpoints` when it builds
the data document (`app/scope_families.py`). If it is missing, scope validation is skipped, so Rego tests that mock
`data.authz.endpoints` must also supply it.

## Testing

```bash
make test-policy      # Native OPA corpus + strict runner self-tests (>=1.22.0,<2)
make test-policy-opa  # Direct OPA CLI corpus execution (requires opa on PATH)
```

The shared Rego tests use native `with` modifiers to install request and
authorization-data fixtures. `script/test_opa_policies.py` uses `opa parse` for
rule discovery and `opa test --format=json --fail-on-empty` to execute unchanged
sources. There is no binding, source rewriting, or custom lowering layer.
`services/core/auth/tests/test_opa_compatibility.py` checks the shared corpus
and self-tests the runner against failing, undefined, skipped, empty, and
parameterized synthetic assertions. Indexed tests require exhaustive expected
case keys; literal `cases` tables are cross-checked with that manifest. Missing
cases, skipped tests, empty reports and nonzero OPA exit status fail validation.

For a provisioned binary, run:

```bash
uv run script/test_opa_policies.py --opa-binary-path /opt/opa/opa
```

Without that argument, `NHX_AUTH_OPA_BINARY_PATH` can select a provisioned binary
for the CLI and pytest runner. Otherwise the standard resolver uses compatible
OPA on PATH, a validated cache, or official pinned release downloads. Do not silently skip
these tests when OPA is unavailable. A passing CLI corpus does not replace
managed-child lifecycle/API tests; run the auth unit suite as well.

