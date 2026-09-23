# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

NHX_ROOT_PATH := $(shell dirname $(shell dirname $(shell pwd)))

.PHONY: vendor-only
vendor-only:
	@echo "📦 Vendoring filesets to nemo-helix SDK..."
	cd ${NHX_ROOT_PATH} && uv run --no-sync nemo-helix-sdk-tools vendor from-config filesets

.PHONY: vendor
vendor: vendor-only
	cd ${NHX_ROOT_PATH} && uv run --no-sync nemo-helix-sdk-tools post-generation update-license-headers
