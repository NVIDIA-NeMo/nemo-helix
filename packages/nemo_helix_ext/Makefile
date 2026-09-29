# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

NHX_ROOT_PATH := $(shell git rev-parse --show-toplevel)

.PHONY: benchmark
benchmark:
	cd $(NHX_ROOT_PATH) && uv run python packages/nemo_helix_ext/scripts/benchmark.py run

.PHONY: benchmark-list
benchmark-list:
	cd $(NHX_ROOT_PATH) && uv run python packages/nemo_helix_ext/scripts/benchmark.py list

.PHONY: benchmark-compare
benchmark-compare:
	cd $(NHX_ROOT_PATH) && uv run python packages/nemo_helix_ext/scripts/benchmark.py compare-baseline

.PHONY: test-unit
test-unit:
	uv run --frozen pytest tests/ -m unit

.PHONY: test-integration
test-integration:
	uv run --frozen pytest tests/ -m integration
