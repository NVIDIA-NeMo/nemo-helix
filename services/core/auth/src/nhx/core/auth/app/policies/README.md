<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared Rego Authorization Policies

The embedded PDP interprets these sources with regopy; the external PDP evaluates
these same sources with OPA. No compiled policy artifact or OPA build is needed
for the embedded runtime.

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
make test-policy      # regopy: shared corpus plus test-runner safety checks
make test-policy-opa  # OPA compatibility oracle (requires the opa executable)
```

The shared Rego tests use `with` to install request and authorization-data
fixtures. regopy 1.5.2 cannot compile transitive calls under native `with`, so
`script/test_regopy_policies.py` installs those input/base-data overrides in fresh
interpreters and evaluates the original Rego assertions. The test-only runner
supports a bounded grammar and rejects unsupported constructs rather than
skipping tests. Production policies are loaded unchanged and do not use `with`.

When adding or changing policy behavior, run both evaluators. A passing native
OPA test alone does not verify the embedded evaluator's behavior.

