<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OPA Authorization Policies

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

## Testing

```bash
make test-policy
```

