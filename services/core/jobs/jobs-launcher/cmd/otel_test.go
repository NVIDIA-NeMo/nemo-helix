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

func TestBatchProcessorOptionsDefaultsAndOverrides(t *testing.T) {
	t.Setenv(logExportIntervalEnv, "")
	t.Setenv(logExportMaxBatchSizeEnv, "")
	t.Setenv(otelBLRPScheduleDelayEnv, "")
	t.Setenv(otelBLRPMaxExportBatchSizeEnv, "")
	t.Setenv(otelBLRPMaxQueueSizeEnv, "")
	if got := len(batchProcessorOptions()); got != 3 {
		t.Fatalf("expected 3 batch processor options, got %d", got)
	}

	if got := positiveIntEnv(logExportIntervalEnv); got != 0 {
		t.Fatalf("expected unset env to yield 0, got %d", got)
	}
	t.Setenv(logExportIntervalEnv, "2500")
	if got := positiveIntEnv(logExportIntervalEnv); got != 2500 {
		t.Fatalf("expected 2500, got %d", got)
	}
	for _, bad := range []string{"abc", "0", "-5"} {
		t.Setenv(logExportIntervalEnv, bad)
		if got := positiveIntEnv(logExportIntervalEnv); got != 0 {
			t.Fatalf("expected invalid value %q to yield 0, got %d", bad, got)
		}
	}
}

func TestBatchProcessorOptionsDefersToOTELEnv(t *testing.T) {
	t.Setenv(logExportIntervalEnv, "")
	t.Setenv(logExportMaxBatchSizeEnv, "")
	t.Setenv(otelBLRPScheduleDelayEnv, "7000")
	t.Setenv(otelBLRPMaxExportBatchSizeEnv, "256")
	t.Setenv(otelBLRPMaxQueueSizeEnv, "1024")
	if got := len(batchProcessorOptions()); got != 0 {
		t.Fatalf("expected OTEL_BLRP_* to be left to the SDK, got %d options", got)
	}

	// An explicit NHX override still applies; a configured OTEL queue size is not reduced.
	t.Setenv(logExportMaxBatchSizeEnv, "100")
	if got := len(batchProcessorOptions()); got != 1 {
		t.Fatalf("expected only the batch size option, got %d", got)
	}
}

func TestBatchProcessorOptionsKeepsQueueFloorForSmallBatches(t *testing.T) {
	t.Setenv(logExportIntervalEnv, "")
	t.Setenv(logExportMaxBatchSizeEnv, "1")
	t.Setenv(otelBLRPScheduleDelayEnv, "")
	t.Setenv(otelBLRPMaxExportBatchSizeEnv, "")
	t.Setenv(otelBLRPMaxQueueSizeEnv, "")
	// interval + batch size + queue size; the queue option must still be present.
	if got := len(batchProcessorOptions()); got != 3 {
		t.Fatalf("expected 3 batch processor options, got %d", got)
	}
}
