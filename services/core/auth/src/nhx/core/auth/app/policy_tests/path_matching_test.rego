# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package common_test

import data.common
import future.keywords.if

# Tests for path_matches_pattern function
# These tests verify that URL patterns with placeholders are correctly matched

# TEST: Simple pattern matching without placeholders
test_path_matches_exact if {
	common.path_matches_pattern("/apis/models/v2/workspaces/-/models", "/apis/models/v2/workspaces/-/models")
}

test_path_matches_exact_negative if {
	not common.path_matches_pattern("/apis/models/v2/workspaces/-/models", "/apis/entities/v2/workspaces")
}

# TEST: Pattern with single placeholder
test_path_matches_single_placeholder if {
	common.path_matches_pattern("/apis/entities/v2/workspaces/my-ws", "/apis/entities/v2/workspaces/{name}")
}

test_path_matches_multiple_placeholders if {
	common.path_matches_pattern(
		"/apis/models/v2/workspaces/my-workspace/models/my-model",
		"/apis/models/v2/workspaces/{workspace}/models/{name}",
	)
}

# TEST: Trailing URI patterns (the problematic case) — real inference-gateway layout
test_path_matches_trailing_uri_simple if {
	common.path_matches_pattern(
		"/apis/inference-gateway/v2/workspaces/my-ws/openai/-/v1/models",
		"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
	)
}

test_path_matches_trailing_uri_nested if {
	common.path_matches_pattern(
		"/apis/inference-gateway/v2/workspaces/my-ws/openai/-/v1/chat/completions",
		"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
	)
}

test_path_matches_trailing_uri_deep_nested if {
	common.path_matches_pattern(
		"/apis/inference-gateway/v2/workspaces/my-ws/openai/-/v1/engines/davinci/completions/stream",
		"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
	)
}

test_path_matches_model_gateway_trailing_uri if {
	common.path_matches_pattern(
		"/apis/inference-gateway/v2/workspaces/my-ns/model/my-model/-/v1/chat/completions",
		"/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/{trailing_uri}",
	)
}

test_path_matches_provider_gateway_trailing_uri if {
	common.path_matches_pattern(
		"/apis/inference-gateway/v2/workspaces/my-ns/provider/my-provider/-/api/v1/generate",
		"/apis/inference-gateway/v2/workspaces/{workspace}/provider/{name}/-/{trailing_uri}",
	)
}

# TEST: Negative cases for trailing URI
test_path_matches_trailing_uri_wrong_prefix if {
	not common.path_matches_pattern(
		"/apis/inference-gateway/v2/workspaces/my-ws/wrong/-/v1/models",
		"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
	)
}

test_path_matches_trailing_uri_missing_separator if {
	not common.path_matches_pattern(
		"/apis/inference-gateway/v2/workspaces/my-ws/openai/v1/models",
		"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
	)
}

# TEST: Edge cases
test_path_matches_empty_trailing if {
	not common.path_matches_pattern(
		"/apis/inference-gateway/v2/workspaces/my-ws/openai/-",
		"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
	)
}

test_path_matches_trailing_single_segment if {
	common.path_matches_pattern(
		"/apis/inference-gateway/v2/workspaces/my-ws/openai/-/models",
		"/apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}",
	)
}
