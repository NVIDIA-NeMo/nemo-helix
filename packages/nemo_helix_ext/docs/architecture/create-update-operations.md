<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Create/Update Operations: Input Design

**Status:** Implemented (Phase 1: stdin/@file support), In Progress (Phase 2: field overrides)
**Date:** 2025-12-09

## Summary

Create and update operations support three input methods with clear precedence rules:

1. **`--input-file PATH`** - Read JSON from file or stdin (use `-` for stdin)
2. **`--input-data DATA`** - Provide inline data (JSON or YAML string)
3. **Top-level field flags** - Override specific fields (e.g., `--id`, `--description`)

**Precedence:** CLI field flags > input-file/input-json

**Mutual exclusivity:** Only one of `--input-file` OR `--input-json` can be used at a time.

## Examples

```bash
# File input
nemo namespaces create --input-file config.json

# Stdin (using dash convention)
echo '{"id": "prod"}' | nemo namespaces create --input-file -
cat config.json | nemo namespaces create --input-file -

# Inline JSON
nemo namespaces create --input-data '{"id": "prod", "description": "Production"}'
# Inline YAML
nemo namespaces create --input-data '{id: prod, description: Production}'

# Field overrides (work with any input method)
nemo namespaces create --input-file base.json --description "Override description"
nemo namespaces create --input-data '{"id": "x"}' --id "production"
[]()
# Update with path params + input
nemo namespaces update --input-file updates.json
echo '{"description": "Updated"}' | nemo namespaces update --input-file -
```

---

## Context & Problem

### Initial Requirements

When implementing create/update operations, we needed to support:
1. Simple inline JSON for quick testing
2. File-based configuration for production use
3. Stdin for pipeline/scripting scenarios
4. Ability to override specific fields without editing files
5. Consistent user experience across all operations

### User Feedback

Integration testing revealed that requiring JSON for all inputs was cumbersome:
- Simple cases required verbose JSON: `nemo namespaces create '{"id": "test"}'`
- No way to override single fields without editing entire JSON file
- Pipeline scenarios (stdin) weren't explicit enough

### Key Questions

1. **How should multiple input sources interact?** (file + CLI flags)
2. **What takes precedence when values conflict?**
3. **Should we follow a standard convention or create our own?**
4. **How explicit should stdin/file indicators be?**

---

## Decision

### Input Methods

#### 1. `--input-file PATH`

Reads JSON from a file or stdin.

- **File:** `--input-file config.json`
- **Stdin:** `--input-file -` (Unix dash convention)
- Supports both relative and absolute paths
- Uses `@file.json` syntax in payload argument for backwards compatibility

**Rationale:**
- `--input-file` is clearer than generic `--input`
- Dash (`-`) convention is standard Unix practice (tar, kubectl, etc.)
- Explicit flag makes scripts self-documenting

#### 2. `--input-data DATA_STRING`

Provides data as inline JSON/YAML string.

```bash
nemo namespaces create --input-data '{id: test, description: Test namespace}'
```

**Rationale:**
- Explicit and consistent with `--input-file`
- Self-documenting in scripts
- Clearer than magic positional argument

**Why not a positional argument?**
- Inconsistent with `--input-file` flag approach
- Less obvious in help text
- Can conflict with other positional args
- Not self-documenting in scripts

#### 3. Top-Level Field Flags

Individual CLI options for common/required fields:

```bash
nemo namespaces create --input-file base.json --id "production" --description "Prod env"
```

**Current Support:** Top-level fields only (e.g., `--id`, `--description`)
**Future:** Nested field overrides (e.g., `--custom-fields.key "value"`)

**Rationale:**
- Ergonomic for common use cases
- Allows templating base configs with overrides
- Follows AWS CLI, Terraform, Docker precedence patterns
- Natural for CI/CD workflows

### Precedence Rules

When values conflict, **CLI flags take precedence** over file/JSON input:

```bash
# config.json: {"id": "dev", "description": "Development"}
nemo namespaces create --input-file config.json --id "production"
# Result: id="production", description="Development"
```

**Precedence order (highest to lowest):**
1. CLI field flags (`--id`, `--description`)
2. `--input-data` / `--input-file` content
3. Default values (if any)

### Mutual Exclusivity

Only ONE of `--input-file` or `--input-data` can be used:

```bash
# ❌ Error: can't use both
nemo create --input-file a.json --input-data '{"id": "x"}'

# ✅ Use one input method + field overrides
nemo create --input-file a.json --id "x"
```

**Rationale:**
- Clear mental model - one source of truth
- Avoids complex merge semantics
- Simpler to implement and document
- Easier to debug when things go wrong

---

## Alternatives Considered

### Alternative 1: Magic Positional Argument

```bash
# Implicit detection
nemo create '{"id":"x"}'        # Inline JSON
nemo create @file.json          # File
cat x | nemo create             # Stdin (auto-detected)
```

**Pros:**
- Concise for simple cases
- Less typing

**Cons:**
- ❌ Magic syntax not discoverable
- ❌ Stdin auto-detection can surprise users
- ❌ Not self-documenting in scripts
- ❌ Harder to add YAML support later
- ❌ Inconsistent - flags for some things, positional for others

**Rejected:** Too much magic, poor discoverability

### Alternative 2: Single `--input` Flag

```bash
nemo create --input '{"id":"x"}'    # Detects JSON
nemo create --input file.json      # Detects file
nemo create --input -               # Stdin
```

**Pros:**
- Single flag to remember
- Consistent interface

**Cons:**
- ❌ Magic detection (JSON string vs filename)
- ❌ Edge cases: file named `{"id":"x"}`
- ❌ Less explicit about what's happening
- ❌ Can't distinguish intent clearly

**Rejected:** Too much implicit behavior

### Alternative 3: Separate `--stdin` Flag

```bash
nemo create --stdin              # Read from stdin
nemo create --file config.json  # Read from file
```

**Pros:**
- Very explicit
- No dash convention needed

**Cons:**
- ❌ Violates Unix conventions
- ❌ More flags to learn
- ❌ Redundant with `--input-file -`
- ❌ Other tools don't do this

**Rejected:** Reinventing the wheel, violates conventions

### Alternative 4: kubectl-Style (YAML is Truth)

```bash
# File defines resource, flags must match
nemo create -f config.json --id "x"  # Error if IDs don't match
```

**Pros:**
- File is single source of truth
- Prevents accidental overwrites

**Cons:**
- ❌ Can't override fields easily
- ❌ Less flexible for templating
- ❌ Not suitable for imperative CLI style
- ❌ Different from AWS/Docker patterns

**Rejected:** Too restrictive for our use case

---

## Detailed Design Decisions

### Why Follow AWS/Terraform Precedence?

**Industry Standard Pattern:**
- AWS CLI: flags override `--cli-input-json`
- Terraform: `-var` overrides `.tfvars` files
- Docker: runtime flags override Dockerfile

**Developer Expectations:**
- "More specific wins" - CLI is more specific than file
- Common workflow: template file + override specific values
- Natural for CI/CD: `nemo create --input-file $TEMPLATE --env $ENV`

**Alternative (kubectl):**
- File is source of truth, flags must match
- ❌ Too restrictive for our imperative CLI

### Why Dash (`-`) for Stdin?

**Unix Convention:**
```bash
tar xf -           # Extract from stdin
cat file | tar xf -
kubectl apply -f - # Apply from stdin
diff file1.txt -   # Compare with stdin
```

**Benefits:**
- ✅ Users already know this pattern
- ✅ Works with `--input-file` naturally
- ✅ No special implementation needed (treat as file descriptor)
- ✅ Consistent with ecosystem

**Why not auto-detect stdin?**
- ❌ Surprises users (not obvious when reading script)
- ❌ Can cause hangs if stdin not available
- ❌ Hard to debug

### Why Top-Level Fields Only (Initially)?

**Phase 1 Scope:**
```bash
# ✅ Supported
nemo create --input-file base.json --id "x" --description "y"

# ❌ Not yet supported
nemo create --input-file base.json --custom-fields.key "value"
nemo create --input-file base.json --metadata.labels.env "prod"
```

**Rationale:**
1. **Simplicity:** Covers 80% of use cases
2. **Generation complexity:** Nested fields require complex introspection
3. **Syntax clarity:** Dot notation can be ambiguous
4. **Incremental delivery:** Ship core functionality first

**Future enhancement (Phase 2):**
- Support nested field syntax: `--custom-fields.key "value"`
- Auto-parse JSON for complex types: `--metadata '{"key":"val"}'`
- Reference: AWS CLI uses similar approach

### Type Handling for CLI Options

**Simple types:** Use as-is
```bash
--id "my-namespace"           # string
--max-retries 3               # integer
--enabled true                # boolean
```

**Complex types:** Parse as JSON string
```bash
--custom-fields '{"key": "value", "count": 42}'
--training-options '[{"name": "epochs", "value": "10"}]'
```

**Rationale:**
- Simple types are ergonomic
- Complex types need structure - JSON is unambiguous
- Matches AWS CLI pattern
- Easy to generate from code generator

---

## Implementation Notes

### Code Generation

**Context Collectors:**
- `CreateContextCollector` - generates create command context
- `UpdateContextCollector` - generates update command context

**Templates:**
- `create_command.py.j2` - create operation template
- `update_command.py.j2` - update operation template

**Key additions:**
1. Generate CLI options for top-level fields (simple types only)
2. Add `--input-file` and `--input-data` flags
3. Generate merge logic (CLI options override input)
4. Generate validation (mutually exclusive inputs)

### Validation Rules

```python
# 1. Exactly one input method
import yaml

if not (bool(input_file) ^ bool(input_data)):
    raise ValueError("Must provide exactly one: --input-file or --input-json")

# 2. Read input based on method
if input_file == "-":
    data = read_data_from_stdin()
elif input_file:
    data = read_data_from_file(input_file)
else:
    data = yaml.safe_load(input_data)

# 3. Apply field overrides (CLI options take precedence)
for field, value in cli_options.items():
    if value is not None:
        data[field] = value

# 4. Call SDK method
client.resource.create(**data)
```

### Error Messages

**Clear errors for common mistakes:**

```bash
# Missing input
$ nemo namespaces create
Error: Must provide input via --input-file or --input-data

# Both inputs
$ nemo create --input-file a.json --input-data '{}'
Error: Cannot use both --input-file and --input-data. Choose one.

# File not found
$ nemo create --input-file missing.json
Error: File not found: missing.json

# Invalid JSON
$ nemo create --input-data '{bad json}'
Error: Invalid JSON: Expecting property name enclosed in double quotes

# Stdin not available
$ nemo create --input-file -
Error: No data available on stdin
```

---

## Future Enhancements

### Phase 2: Nested Field Overrides

```bash
# Dot notation for nested fields
nemo create --input-file base.json \
  --custom-fields.environment "production" \
  --custom-fields.region "us-west-2"

# Array indices
nemo create --input-file base.json \
  --training-options.0.value "10"
```

**Considerations:**
- Syntax: dot notation vs bracket notation
- Complex types: when to parse as JSON
- Array handling: append vs override
- Validation: type checking nested paths

---

## Related Decisions

- **[CLI Generation Architecture](./cli-generation.md)** - How commands are generated
- **[Field Override Support](./field-overrides.md)** - Future: nested field support
- **[Input Validation](./input-validation.md)** - Future: schema validation

---

## References

### Industry Patterns

- [AWS CLI Input Files](https://docs.aws.amazon.com/cli/latest/userguide/cli-usage-skeleton.html) - Flags override JSON
- [Terraform Variable Precedence](https://developer.hashicorp.com/terraform/language/values/variables) - CLI vars override files
- [Docker Runtime Flags](https://docs.docker.com/reference/cli/docker/container/run/) - Runtime overrides Dockerfile
- [kubectl Create from Stdin](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_create/) - File is source of truth
- [GitHub CLI gh api](https://cli.github.com/manual/gh_api) - Stdin for body, flags for query params

### Unix Conventions

- [Unix Dash Convention](https://www.baeldung.com/linux/dash-in-command-line-parameters) - Using `-` for stdin/stdout
- [Standard Streams](https://en.wikipedia.org/wiki/Standard_streams) - stdin, stdout, stderr

---

## Decision Log

| Date | Decision                                    | Rationale |
|------|---------------------------------------------|-----------|
| 2025-12-09 | Use `--input-file` and `--input-data` flags | Explicit, self-documenting, consistent |
| 2025-12-09 | Support `-` for stdin with `--input-file`   | Unix convention, user expectation |
| 2025-12-09 | CLI flags override input                    | Matches AWS/Terraform/Docker patterns |
| 2025-12-09 | Mutual exclusivity for input methods        | Simpler mental model, clearer semantics |
| 2025-12-09 | Top-level fields only (Phase 1)             | Cover common cases, incremental delivery |
| 2025-12-09 | Reject positional argument                  | Not consistent, not self-documenting |
| 2025-12-09 | Reject separate `--stdin` flag              | Violates Unix conventions, redundant |
