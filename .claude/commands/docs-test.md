<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Testing Documentation Notebooks

You can test documentation from the repository root using `docs/fern/scripts/run_notebooks.py`, which executes notebooks marked with `@nemo-nb: process`.

## Basic Usage

The script takes a file path or directory path:

```sh
uv run python docs/fern/scripts/run_notebooks.py <input_path>
```

Environment Variables
The script automatically loads environment variables from a .env file in the repository root (if present).
Example .env file:

```sh
NHX_BASE_URL=http://localhost:8080
NVIDIA_API_KEY=nvapi-your-key-here
HF_TOKEN=hf_your-token-here
NGC_API_KEY=your-ngc-key-here
```

You can also override environment variables inline:

```sh
NHX_BASE_URL=http://custom-url:8080 uv run python docs/fern/scripts/run_notebooks.py docs/run-inference/
```

## Language filters

```sh
# Run only Python cells (default)
uv run python docs/fern/scripts/run_notebooks.py docs/run-inference/ --language python

# Run only shell cells
uv run python docs/fern/scripts/run_notebooks.py docs/run-inference/ --language shell
```


## Markers

Only notebooks with the `@nemo-nb: process` marker will be executed.
