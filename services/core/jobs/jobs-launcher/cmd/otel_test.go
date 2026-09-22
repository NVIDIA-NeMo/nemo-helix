// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import "testing"

func TestLauncherOTLPHTTPClientUsesUDSEndpoint(t *testing.T) {
	t.Setenv("NHX_BASE_URL", "unix:///tmp/nemo-helix.sock")

	client := launcherOTLPHTTPClient()
	if client == nil {
		t.Fatal("expected UDS HTTP client")
	}
	if client.Timeout != 0 {
		t.Fatalf("expected UDS HTTP client timeout to remain unset, got %s", client.Timeout)
	}
}

func TestLauncherOTLPHTTPClientSkipsTCP(t *testing.T) {
	t.Setenv("NHX_BASE_URL", "http://127.0.0.1:8080")

	if client := launcherOTLPHTTPClient(); client != nil {
		t.Fatal("expected no custom HTTP client for TCP endpoint")
	}
}
