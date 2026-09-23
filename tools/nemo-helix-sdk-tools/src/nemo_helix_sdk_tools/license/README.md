<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# License Scanner

Generate and manage license reports for the Platform monorepo.

## Requirements

- `osv-scanner` must be installed

## Quick Start

```bash
# Generate licenses in default table format
nemo-helix-sdk-tools license generate

# Generate in JSONL format (recommended for automation)
nemo-helix-sdk-tools license generate --format jsonl

# Generate in CSV format (good for spreadsheets)
nemo-helix-sdk-tools license generate --format csv

# Generate CSV at a custom path
nemo-helix-sdk-tools license generate --format csv --output third_party/licenses.csv
```

## Available Formats

### JSONL (Recommended) ✨

**Best for automation and machine parsing**

```bash
nemo-helix-sdk-tools license generate --format jsonl
```

Output: One JSON object per line

```jsonl
{"name": "requests", "version": "2.31.0", "license": "APACHE-2.0", "compatible": true}
{"name": "numpy", "version": "1.24.0", "license": "BSD-3-CLAUSE", "compatible": true}
```

**Why JSONL?**

- ✅ Readable on one line per package
- ✅ Easy to parse with standard tools
- ✅ Can be processed line-by-line
- ✅ Perfect for CI/CD validation

**Usage examples:**

```bash
# Find all incompatible licenses
grep '"compatible": false' third_party/licenses.jsonl

# Use jq for complex queries
jq 'select(.compatible == false)' third_party/licenses.jsonl

# Count by license type
jq -r '.license' third_party/licenses.jsonl | sort | uniq -c
```

### CSV Format

**Best for spreadsheets**

```bash
nemo-helix-sdk-tools license generate --format csv
```

Output:

```csv
Package,License,License URL
aiofiles,APACHE-2.0,https://github.com/Tinche/aiofiles/blob/main/LICENSE
requests,APACHE-2.0,https://github.com/psf/requests/blob/main/LICENSE
```

Opens directly in Excel or Google Sheets.

### Markdown Format

**Best for documentation**

```bash
nemo-helix-sdk-tools license generate --format markdown
```

Output:

```markdown
| Compatible | Package  | Version | License      |
|------------|----------|---------|--------------|
| ✔          | requests | 2.31.0  | APACHE-2.0   |
| ✔          | numpy    | 1.24.0  | BSD-3-CLAUSE |
```

Renders nicely in GitHub/GitLab.

### JSON Format

**Standard JSON array**

```bash
nemo-helix-sdk-tools license generate --format json
```

Output: Standard JSON array format, good for API responses.

### Text Format

**Simple tab-separated values**

```bash
nemo-helix-sdk-tools license generate --format text
```

Output:

```
requests 2.31.0 APACHE-2.0 ✔
numpy 1.24.0 BSD-3-CLAUSE ✔
```

### Table Format (Default)

**Rich Unicode table for terminal viewing**

```bash
nemo-helix-sdk-tools license generate  # or --format table
```

Output: The current beautiful terminal table format.

## Common Workflows

### 1. Generate and Check for Issues

```bash
# Generate licenses
nemo-helix-sdk-tools license generate --format jsonl

# Find packages needing overrides
nemo-helix-sdk-tools license find-missing

# Discover license information from PyPI
nemo-helix-sdk-tools license discover-overrides
```

### 2. CI/CD Validation

```bash
# Generate JSONL for easy parsing
nemo-helix-sdk-tools license generate --format jsonl

# Check for incompatible licenses
if grep -q '"compatible": false' third_party/licenses.jsonl; then
    echo "Found incompatible licenses!"
    grep '"compatible": false' third_party/licenses.jsonl
    exit 1
fi
```

### 3. Generate Multiple Formats

```bash
# For human viewing in terminal
nemo-helix-sdk-tools license generate --format table

# For automation/scripts
nemo-helix-sdk-tools license generate --format jsonl

# For documentation
nemo-helix-sdk-tools license generate --format markdown
```

## Commands

### `generate`

Generate license report for the main project.

```bash
nemo-helix-sdk-tools license generate [OPTIONS]
```

**Options:**

- `--format, -f TEXT`: Output format (table, jsonl, json, csv, markdown, text) [default: table]
- `--output, -o PATH`: Optional path for the formatted license report [default: third_party/licenses.jsonl]
- `--sequential`: Run scans sequentially instead of in parallel
- `--verbose, -v`: Enable verbose logging

### `find-missing`

Find packages with UNKNOWN or NON-STANDARD licenses.

```bash
nemo-helix-sdk-tools license find-missing
```

Scans the OSV JSON files (format-independent) and reports which packages need manual license overrides.

**Note:** This command reads the raw OSV JSON files, so it works regardless of what output format you used when generating licenses (table, jsonl, csv, etc.).

### `discover-overrides`

Fetch license information from PyPI for packages with missing licenses.

```bash
nemo-helix-sdk-tools license discover-overrides [--verbose]
```

Prints suggested YAML overrides that can be added to `third_party/license_overrides.yaml`.

## License Overrides

To override licenses for specific packages, edit:

```bash
tools/nemo-helix-sdk-tools/src/nemo_helix_sdk_tools/license/overrides.yaml
```

Format:

```yaml
overrides:
  package-name: "LICENSE-TYPE"
  another-package: "MIT"
```

## Output Files

Generated files are saved to `third_party/`:

- `licenses.jsonl` (or `.jsonl`, `.csv`, etc.) - Python and Go dependency licenses (format depends on `--format` flag)
- `osv-licenses.json` - Merged raw OSV scanner output for the Python dependency snapshot and Go modules - **always generated**
- `osv-licenses-jobs-launcher.json`, `osv-licenses-callouts.json` - Raw intermediate Go module scan outputs generated before merging

**Note:** The `osv-licenses*.json` files are always created regardless of output format. `osv-licenses.json` is the merged artifact used by the `find-missing` and `discover-overrides` commands; the per-module files are retained as raw scan intermediates.
