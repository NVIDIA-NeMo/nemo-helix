<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Linting Documentation Python Snippets

Lint Markdown and MDX Python fenced code blocks for syntax and type errors without executing them using `docs/_scripts/lint_python_snippets.py`.

## Basic Usage

```sh
uv run python docs/_scripts/lint_python_snippets.py <path>
```

Examples:

```sh
# Lint a single doc page
uv run python docs/_scripts/lint_python_snippets.py docs/run-inference/tutorials/deploy-llm-nims.md

# Lint all Markdown/MDX pages in a directory
uv run python docs/_scripts/lint_python_snippets.py docs/run-inference/
```

### Running in Cursor Agent Sandbox Mode

When running through the Cursor agent in sandbox mode, the agent will set `UV_CACHE_DIR` in the shell environment to avoid read-only filesystem errors. The agent will also use `required_permissions: ["network", "git_write"]` since uv needs network access to download packages and git_write permissions to create cache directories.

## Type Checking

Add `--type-check` to run `ty` type checker on the combined snippets:

```sh
uv run python docs/_scripts/lint_python_snippets.py docs/run-inference/ --type-check
```

This catches:
- Wrong SDK method names (e.g., `gateway.create()` instead of `gateway.post_model()`)
- Missing attributes (e.g., `client.customizer` when customizer isn't available)
- Type mismatches

Note: `ty` is alpha software and reports some false positives for SDK attributes it can't fully resolve.

## What It Checks

1. **Syntax errors** - Python AST parsing (always enabled)
2. **Type errors** - `ty` type checker (with `--type-check` flag)

## How It Works

The script:
1. Finds Markdown and MDX files under the requested paths
2. Extracts all `python` and `py` fenced code blocks
3. Combines snippets from each page into a single file so earlier snippets can define later context
4. Runs `ty check` and maps diagnostics back to the original doc lines

## Fixing Linter Errors

### ⚠️ NEVER convert Python cells to text blocks

**Changing `\`\`\`python` to `\`\`\`text` is NOT acceptable.** All code must remain executable.

### Acceptable fixes

1. **Fix the actual bug** - Use correct method names, add missing arguments, etc.

2. **Add type ignore comments** for false positives from ty:
   ```python
   response = InferenceGatewayClient.from_client(client).provider_post(...)
   message = response["choices"][0]["message"]["content"]  # type: ignore[index]
   ```

3. **Add conditional checks** for optional dependencies:
   ```python
   if "API_KEY" in os.environ:
       # code that requires the API key
   else:
       print("Skipping - API_KEY not set")
   ```

4. **Use try/except** for optional imports:
   ```python
   try:
       from optional_package import Client  # type: ignore[import-not-found]
       # use Client
   except ImportError:
       print("Optional package not installed - skipping")
   ```

### Common false positives from ty

The SDK's gateway methods return `object` type, causing false positives when accessing dict keys:
- `error[non-subscriptable]` - Add `# type: ignore[index]`
- `error[not-iterable]` - Add `# type: ignore[union-attr]`
- `error[unsupported-operator]` - Add `# type: ignore[operator]`

## Markers

All Markdown and MDX pages under the requested paths are scanned. Use `<!-- @nemo-docs: skip-python-snippet-check -->` before a block to skip it.
