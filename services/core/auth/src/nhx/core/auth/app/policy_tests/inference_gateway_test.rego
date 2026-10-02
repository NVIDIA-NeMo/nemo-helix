# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import data.common
import future.keywords.if

# Tests for inference gateway URL pattern matching and permission checking
# These tests verify that gateway URLs with trailing paths are correctly
# matched to their endpoint patterns and the appropriate permissions are checked.
#
# API group path: /apis/inference-gateway/v2/workspaces/{workspace}/...
# - model: /apis/inference-gateway/v2/workspaces/{workspace}/model/{model_name}/-/{trailing_uri}
# - openai: /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/{trailing_uri}
# - provider: /apis/inference-gateway/v2/workspaces/{workspace}/provider/{provider_name}/-/{trailing_uri}

# Mock data for inference gateway tests - only principals and workspaces
# Roles and endpoints come from live static-authz.yaml data
mock_inference_gateway_workspaces := {
	"my-workspace": {},
	"other-workspace": {},
	"public-ns": {},
}

mock_inference_gateway_principals := {
	"user@example.com": {
		"workspaces": {
			"my-workspace": ["Editor"],
		},
		"scopes": ["inference:read", "inference:write"],
	},
	"viewer@example.com": {
		"workspaces": {
			"my-workspace": ["Viewer"],
			"public-ns": ["Viewer"],
		},
		"scopes": ["inference:read"],
	},
	"unauthorized@example.com": {
		"workspaces": {},
		"scopes": ["inference:read", "inference:write"],
	},
	# Wildcard principal grants Viewer access to public-ns for all authenticated users
	"*": {
		"workspaces": {
			"public-ns": ["Viewer"],
		},
	},
}

# TEST: OpenAI gateway endpoint with trailing URI - chat completions
test_openai_gateway_chat_completions if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1/chat/completions",
					"headers": {"x-nhx-principal-id": "user@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}

# TEST: OpenAI gateway endpoint with different trailing URI - completions
test_openai_gateway_completions if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1/completions",
					"headers": {"x-nhx-principal-id": "user@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}

# TEST: OpenAI gateway GET request with trailing URI
test_openai_gateway_get_models if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "GET",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1/models",
					"headers": {"x-nhx-principal-id": "user@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}

# TEST: Model gateway endpoint with workspace and model in path
test_model_gateway_chat_completions if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/model/my-model/-/v1/chat/completions",
					"headers": {"x-nhx-principal-id": "user@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}

# TEST: Model gateway endpoint - unauthorized workspace
test_model_gateway_unauthorized_workspace if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/other-workspace/model/other-model/-/v1/chat/completions",
					"headers": {"x-nhx-principal-id": "user@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as {
			"my-workspace": {},
			"other-workspace": {},
		}

	# User only has access to my-workspace, not other-workspace
	result.allowed == false
	result.headers["X-NHX-Authorized"] == "false"
}

# TEST: Provider gateway endpoint with workspace and provider in path
test_provider_gateway_models_list if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "GET",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/provider/my-provider/-/v1/models",
					"headers": {"x-nhx-principal-id": "user@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}

# TEST: Provider gateway POST with nested trailing URI
test_provider_gateway_post_nested_path if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/provider/my-provider/-/api/v1/generate",
					"headers": {"x-nhx-principal-id": "user@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}

# TEST: Viewer can run inference (inference.gateway.*.exec is a Viewer permission)
test_viewer_can_run_openai_inference if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1/chat/completions",
					"headers": {"x-nhx-principal-id": "viewer@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}

# TEST: Viewer can run inference via model gateway
test_viewer_can_run_model_inference if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/model/my-model/-/v1/chat/completions",
					"headers": {"x-nhx-principal-id": "viewer@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}

# TEST: Viewer can run inference via provider gateway
test_viewer_can_run_provider_inference if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/provider/my-provider/-/api/v1/generate",
					"headers": {"x-nhx-principal-id": "viewer@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}

# No-role principal used for denial tests
mock_no_role_principals := {
	"no-role@example.com": {
		"workspaces": {},
	},
}

# TEST: No-role user denied on model gateway
test_model_gateway_no_role_denied if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/model/my-model/-/v1/chat/completions",
					"headers": {"x-nhx-principal-id": "no-role@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_no_role_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == false
	result.headers["X-NHX-Authorized"] == "false"
}

# TEST: No-role user denied on openai gateway
test_openai_gateway_no_role_denied if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1/chat/completions",
					"headers": {"x-nhx-principal-id": "no-role@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_no_role_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == false
	result.headers["X-NHX-Authorized"] == "false"
}

# TEST: No-role user denied on provider gateway
test_provider_gateway_no_role_denied if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/provider/my-provider/-/api/v1/generate",
					"headers": {"x-nhx-principal-id": "no-role@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_no_role_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == false
	result.headers["X-NHX-Authorized"] == "false"
}

# TEST: No principal denied on model gateway
test_model_gateway_no_principal_denied if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/model/my-model/-/v1/chat/completions",
					"headers": {},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == false
	result.headers["X-NHX-Authorized"] == "false"
}

# TEST: No principal denied on openai gateway
test_openai_gateway_no_principal_denied if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1/chat/completions",
					"headers": {},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == false
	result.headers["X-NHX-Authorized"] == "false"
}

# TEST: No principal denied on provider gateway
test_provider_gateway_no_principal_denied if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/provider/my-provider/-/api/v1/generate",
					"headers": {},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == false
	result.headers["X-NHX-Authorized"] == "false"
}

# TEST: Unauthorized workspace denied on openai gateway
test_openai_gateway_unauthorized_workspace if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/other-workspace/openai/-/v1/chat/completions",
					"headers": {"x-nhx-principal-id": "user@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == false
	result.headers["X-NHX-Authorized"] == "false"
}

# TEST: Unauthorized workspace denied on provider gateway
test_provider_gateway_unauthorized_workspace if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/other-workspace/provider/my-provider/-/api/v1/generate",
					"headers": {"x-nhx-principal-id": "user@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == false
	result.headers["X-NHX-Authorized"] == "false"
}

# TEST: Verify pattern matching works with very long trailing URIs
test_openai_gateway_long_trailing_uri if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "POST",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1/engines/davinci/completions/stream",
					"headers": {"x-nhx-principal-id": "user@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}

# TEST: Verify pattern matching extracts workspace correctly from model gateway
test_model_gateway_workspace_extraction if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "GET",
					"path": "/apis/inference-gateway/v2/workspaces/my-workspace/model/llama-model/-/v1/models",
					"headers": {"x-nhx-principal-id": "user@example.com"},
				},
			},
		},
	}
		with data.authz.principals as mock_inference_gateway_principals
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}

# TEST: Verify pattern matching extracts workspace correctly from provider gateway
test_provider_gateway_workspace_extraction if {
	result := authz.allow with input as {
		"attributes": {
			"request": {
				"http": {
					"method": "GET",
					"path": "/apis/inference-gateway/v2/workspaces/public-ns/provider/ollama/-/v1/models",
					"headers": {"x-nhx-principal-id": "viewer@example.com"},
				},
			},
		},
	}
		with data.authz.principals as {
			"viewer@example.com": {
				"workspaces": {
					"public-ns": ["Viewer"],
				},
			},
		}
		with data.authz.workspaces as mock_inference_gateway_workspaces

	result.allowed == true
}
