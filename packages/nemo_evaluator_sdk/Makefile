# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: test

NHX_ROOT_PATH := $(shell dirname $(shell dirname $(shell pwd)))

# TODO: Uncomment this when we have a way to vendor the package into the sdk
# vendor: # Vendor the package into the sdk
# rm -r sdk/python/nemo-helix/src/nemo_helix/beta/evaluator/ || true
# cd ${NHX_ROOT_PATH} && uv run --frozen nemo-helix-sdk-tools vendor from-config nemo_evaluator_sdk
# cd ${NHX_ROOT_PATH} && uv run --frozen nemo-helix-sdk-tools post-generation update-license-headers

test: # Run nemo_evaluator_sdk tests
	uv run --frozen pytest tests

.PHONY: vendor

NHX_ROOT_PATH := $(shell dirname $(shell dirname $(shell pwd)))

vendor: # Vendor the package into the sdk
	rm -r sdk/python/nemo-helix/src/nemo_helix/beta/evaluator/ || true
	cd ${NHX_ROOT_PATH} && uv run --frozen nemo-helix-sdk-tools vendor from-config nemo_evaluator_sdk
	cd ${NHX_ROOT_PATH} && uv run --frozen nemo-helix-sdk-tools post-generation update-license-headers

