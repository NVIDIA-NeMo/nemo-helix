// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

const stepConfigPath = "/apis/jobs/v2/workspaces/ws/jobs/job-1/steps/step-a"

func stepConfigServer(t *testing.T, status int, body string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if r.URL.Path != stepConfigPath {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	}))
	t.Cleanup(server.Close)
	return server, &hits
}

// setStepConfigEnv points the launcher at server and returns the config file path,
// nested under a directory that does not exist yet.
func setStepConfigEnv(t *testing.T, serverURL string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config", "job_step_config.json")
	t.Setenv("NHX_JOBS_URL", serverURL)
	t.Setenv("NHX_PRINCIPAL", `{"id":"user-1"}`)
	t.Setenv("NEMO_JOB_WORKSPACE", "ws")
	t.Setenv("NEMO_JOB_ID", "job-1")
	t.Setenv("NEMO_JOB_STEP", "step-a")
	t.Setenv("NEMO_JOB_STEP_CONFIG_FILE_PATH", path)
	t.Setenv("NEMO_JOB_SECRETS", "")
	return path
}

func enableFetchStepConfig(t *testing.T) {
	t.Helper()
	previous := fetchStepConfig
	fetchStepConfig = true
	t.Cleanup(func() { fetchStepConfig = previous })
}

func TestRunExecFetchStepConfigWritesFileBeforeWorkload(t *testing.T) {
	server, hits := stepConfigServer(t, http.StatusOK, `{"config":{"_step_spec_name":"train","epochs":3}}`)
	path := setStepConfigEnv(t, server.URL)
	enableFetchStepConfig(t)

	// The workload exits 0 only if the config file exists when it starts.
	exitCode, err := runExec(context.Background(), []string{"sh", "-c", `test -f "$NEMO_JOB_STEP_CONFIG_FILE_PATH"`}, nil)
	if err != nil || exitCode != 0 {
		t.Fatalf("expected workload to see the config file, got exit %d err %v", exitCode, err)
	}
	if hits.Load() != 1 {
		t.Fatalf("expected one jobs API call, got %d", hits.Load())
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("config file not written: %v", err)
	}
	if string(data) != `{"epochs":3}` {
		t.Fatalf("unexpected config content %s", data)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat config file: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("expected mode 0600, got %o", info.Mode().Perm())
	}
}

func TestRunExecFetchStepConfigFailsBeforeWorkload(t *testing.T) {
	testCases := []struct {
		name          string
		status        int
		body          string
		unsetEnv      string
		errorContains string
		expectHits    int32
	}{
		{
			name:          "api_error",
			status:        http.StatusNotFound,
			body:          `{"detail":"not found"}`,
			errorContains: "failed to fetch step config for ws/job-1/step-a",
			expectHits:    1,
		},
		{
			name:          "missing_step_env",
			status:        http.StatusOK,
			body:          `{"config":{}}`,
			unsetEnv:      "NEMO_JOB_STEP",
			errorContains: "NEMO_JOB_STEP is required",
		},
		{
			name:          "missing_config_path_env",
			status:        http.StatusOK,
			body:          `{"config":{}}`,
			unsetEnv:      "NEMO_JOB_STEP_CONFIG_FILE_PATH",
			errorContains: "NEMO_JOB_STEP_CONFIG_FILE_PATH is required",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			server, hits := stepConfigServer(t, tc.status, tc.body)
			setStepConfigEnv(t, server.URL)
			if tc.unsetEnv != "" {
				t.Setenv(tc.unsetEnv, "")
			}
			enableFetchStepConfig(t)

			marker := filepath.Join(t.TempDir(), "ran")
			exitCode, err := runExec(context.Background(), []string{"touch", marker}, nil)
			if exitCode != 1 {
				t.Fatalf("expected exit 1, got %d", exitCode)
			}
			if err == nil || !strings.Contains(err.Error(), tc.errorContains) {
				t.Fatalf("expected error containing %q, got %v", tc.errorContains, err)
			}
			if _, statErr := os.Stat(marker); statErr == nil {
				t.Fatal("workload ran despite the step config fetch failing")
			}
			if hits.Load() != tc.expectHits {
				t.Fatalf("expected %d jobs API calls, got %d", tc.expectHits, hits.Load())
			}
		})
	}
}

func TestRunExecFetchStepConfigReplacesPermissiveExistingFile(t *testing.T) {
	server, _ := stepConfigServer(t, http.StatusOK, `{"config":{"epochs":3}}`)
	path := setStepConfigEnv(t, server.URL)
	enableFetchStepConfig(t)

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, 0o644); err != nil {
		t.Fatal(err)
	}

	exitCode, err := runExec(context.Background(), []string{"true"}, nil)
	if err != nil || exitCode != 0 {
		t.Fatalf("expected success, got exit %d err %v", exitCode, err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if mode := info.Mode().Perm(); mode != 0o600 {
		t.Fatalf("expected mode 0600, got %o", mode)
	}
	if data, _ := os.ReadFile(path); string(data) != `{"epochs":3}` {
		t.Fatalf("unexpected config content %s", data)
	}
}

func TestRunExecCancelledBeforeStartSkipsWorkload(t *testing.T) {
	server, _ := stepConfigServer(t, http.StatusOK, `{"config":{}}`)
	setStepConfigEnv(t, server.URL)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	marker := filepath.Join(t.TempDir(), "ran")
	exitCode, err := runExec(ctx, []string{"touch", marker}, nil)
	if exitCode != 1 || !errors.Is(err, context.Canceled) {
		t.Fatalf("expected exit 1 with context.Canceled, got exit %d err %v", exitCode, err)
	}
	if _, statErr := os.Stat(marker); statErr == nil {
		t.Fatal("workload started after termination")
	}
}

func TestRunExecWithoutFetchStepConfigSkipsJobsAPI(t *testing.T) {
	server, hits := stepConfigServer(t, http.StatusOK, `{"config":{}}`)
	path := setStepConfigEnv(t, server.URL)

	exitCode, err := runExec(context.Background(), []string{"true"}, nil)
	if err != nil || exitCode != 0 {
		t.Fatalf("expected success, got exit %d err %v", exitCode, err)
	}
	if hits.Load() != 0 {
		t.Fatalf("expected no jobs API calls, got %d", hits.Load())
	}
	if _, statErr := os.Stat(path); statErr == nil {
		t.Fatal("config file written without --fetch-step-config")
	}
}

func TestRunCmdRegistersFetchStepConfigFlag(t *testing.T) {
	flag := runCmd.Flags().Lookup("fetch-step-config")
	if flag == nil {
		t.Fatal("--fetch-step-config is not registered on the run command")
	}
	if flag.DefValue != "false" {
		t.Fatalf("expected default false, got %s", flag.DefValue)
	}
}
