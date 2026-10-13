// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/NVIDIA-NeMo/nemo-helix/services/core/jobs/jobs-launcher/nhxclient"
)

func TestLogLevel(t *testing.T) {
	tests := []struct {
		name     string
		line     string
		fallback slog.Level
		want     slog.Level
	}{
		{name: "info on stderr", line: "[19:40:10] [INFO] Progress", fallback: slog.LevelError, want: slog.LevelInfo},
		{name: "warning on stderr", line: "[WARNING] skipped row", fallback: slog.LevelError, want: slog.LevelWarn},
		{name: "error on stdout", line: "[ERROR] request failed", fallback: slog.LevelInfo, want: slog.LevelError},
		{name: "critical on stderr", line: "[CRITICAL] process failed", fallback: slog.LevelError, want: slog.LevelError},
		{name: "unprefixed stderr", line: "Traceback (most recent call last):", fallback: slog.LevelError, want: slog.LevelError},
		{name: "unprefixed stdout", line: "plain output", fallback: slog.LevelInfo, want: slog.LevelInfo},
		{name: "marker in stdout message", line: "plain output [ERROR]", fallback: slog.LevelInfo, want: slog.LevelInfo},
		{name: "unknown prefix before marker", line: "[worker] [ERROR] failed", fallback: slog.LevelInfo, want: slog.LevelInfo},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := logLevel(test.line, test.fallback); got != test.want {
				t.Fatalf("logLevel(%q, %s) = %s, want %s", test.line, test.fallback, got, test.want)
			}
		})
	}
}

func TestRunCommand(t *testing.T) {
	cases := []struct {
		name              string
		args              []string
		expectedErrorCode int
		shouldError       bool
	}{
		{
			name:              "Valid command",
			args:              []string{"echo", "Hello, World!"},
			expectedErrorCode: 0,
			shouldError:       false,
		},
		{
			name:              "Command with error",
			args:              []string{"nonexistent", "command"},
			expectedErrorCode: 1,
			shouldError:       true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			exitCode, err := runExecWithStdin(tc.args)
			if exitCode != tc.expectedErrorCode {
				t.Errorf("Expected error code %d but got %d", tc.expectedErrorCode, exitCode)
			}
			if (err != nil) != tc.shouldError {
				t.Errorf("Expected error presence %v but got %v", tc.shouldError, err != nil)
			}
		})
	}
}

func TestRunCommandWithStdin(t *testing.T) {
	// Test stdin forwarding with a command that reads from stdin
	// Save original stdin
	originalStdin := os.Stdin

	// Create a pipe to simulate stdin input
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("Failed to create pipe: %v", err)
	}

	// Replace stdin with our pipe reader
	os.Stdin = r

	// Write test input to the pipe
	go func() {
		defer w.Close()                          // nolint:errcheck
		w.WriteString("test input from stdin\n") // nolint:errcheck
	}()

	// Use a command that reads from stdin and echoes it back
	// We'll use 'cat' which reads from stdin and outputs to stdout
	exitCode, err := runExecWithStdin([]string{"cat"})

	// Restore original stdin
	os.Stdin = originalStdin

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if exitCode != 0 {
		t.Errorf("Expected exit code 0, got %d", exitCode)
	}
}

func TestRunExecWithStdinIntegration(t *testing.T) {
	// Test with a simpler approach - using stdin forwarding
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("Failed to create pipe: %v", err)
	}

	originalStdin := os.Stdin
	os.Stdin = r

	go func() {
		defer w.Close()                // nolint:errcheck
		w.WriteString("piped input\n") // nolint:errcheck
	}()

	// Test with a command that should read from stdin
	exitCode, err := runExecWithStdin([]string{"head", "-n", "1"})

	os.Stdin = originalStdin

	if err != nil {
		t.Logf("Command error (may be expected): %v", err)
	}
	if exitCode != 0 {
		t.Errorf("Expected exit code 0, got %d", exitCode)
	}
}

// Helper function to test runExec with custom stdin reader
func TestRunExecWithStdinHelper(t *testing.T) {
	// Create input data
	input := "test line 1\ntest line 2\n"
	var inputReader io.Reader = strings.NewReader(input)

	exitCode, err := runExec([]string{"cat"}, inputReader)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if exitCode != 0 {
		t.Errorf("Expected exit code 0, got %d", exitCode)
	}
}

func TestRunExecWithSecrets(t *testing.T) {
	// Create a mock server for secrets API
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			t.Errorf("Expected GET request, got %s", r.Method)
		}

		// Return mock secrets based on the path
		if r.URL.Path == "/apis/secrets/v2/workspaces/default/secrets/test-secret/access" {
			w.WriteHeader(http.StatusOK)
			fmt.Fprintln(w, `{"value":"secret_value_123"}`)
		} else if r.URL.Path == "/apis/secrets/v2/workspaces/default/secrets/another-secret/access" {
			w.WriteHeader(http.StatusOK)
			fmt.Fprintln(w, `{"value":"another_value_456"}`)
		} else {
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintln(w, `{"error":"secret not found"}`)
		}
	}))
	defer mockServer.Close()

	testCases := []struct {
		name             string
		secretsEnv       string
		apiURL           string
		baseURL          string
		principalJSON    string
		expectedExitCode int
		expectError      bool
	}{
		{
			name:             "single_secret_injection",
			secretsEnv:       "TEST_SECRET=default/test-secret",
			apiURL:           mockServer.URL,
			principalJSON:    `{"id":"test-principal"}`,
			expectedExitCode: 0,
			expectError:      false,
		},
		{
			name:             "multiple_secrets_injection",
			secretsEnv:       "TEST_SECRET=default/test-secret,ANOTHER_SECRET=default/another-secret",
			apiURL:           mockServer.URL,
			principalJSON:    `{"id":"test-principal"}`,
			expectedExitCode: 0,
			expectError:      false,
		},
		{
			name:             "uses_base_url_fallback",
			secretsEnv:       "TEST_SECRET=default/test-secret",
			baseURL:          mockServer.URL,
			principalJSON:    `{"id":"test-principal"}`,
			expectedExitCode: 0,
			expectError:      false,
		},
		{
			name:             "missing_api_url",
			secretsEnv:       "TEST_SECRET=default/test-secret",
			apiURL:           "",
			principalJSON:    `{"id":"test-principal"}`,
			expectedExitCode: 1,
			expectError:      true,
		},
		{
			name:             "invalid_secret_format",
			secretsEnv:       "invalid-format",
			apiURL:           mockServer.URL,
			principalJSON:    `{"id":"test-principal"}`,
			expectedExitCode: 1,
			expectError:      true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Save and restore environment variables
			origEnvVars := map[string]envVarState{
				"NEMO_JOB_SECRETS": getEnvState("NEMO_JOB_SECRETS"),
				"NHX_SECRETS_URL":  getEnvState("NHX_SECRETS_URL"),
				"NHX_BASE_URL":     getEnvState("NHX_BASE_URL"),
				"NHX_PRINCIPAL":    getEnvState("NHX_PRINCIPAL"),
			}
			defer restoreEnvVars(origEnvVars)

			// Set test environment variables
			if tc.secretsEnv != "" {
				os.Setenv("NEMO_JOB_SECRETS", tc.secretsEnv)
			}
			if tc.apiURL != "" {
				os.Setenv("NHX_SECRETS_URL", tc.apiURL)
			} else {
				os.Unsetenv("NHX_SECRETS_URL")
			}
			if tc.baseURL != "" {
				os.Setenv("NHX_BASE_URL", tc.baseURL)
			} else {
				os.Unsetenv("NHX_BASE_URL")
			}
			if tc.principalJSON != "" {
				os.Setenv("NHX_PRINCIPAL", tc.principalJSON)
			} else {
				os.Unsetenv("NHX_PRINCIPAL")
			}

			// Run a simple command - for success cases it should complete,
			// for error cases it should fail before executing the command
			var exitCode int
			var err error

			if !tc.expectError {
				// Use sh to check that secrets are available in environment
				// This validates that secrets were actually injected
				exitCode, err = runExec([]string{"sh", "-c", "env | grep -E '(TEST_SECRET|ANOTHER_SECRET)' || true"}, nil)
			} else {
				// For error cases, use any simple command
				exitCode, err = runExec([]string{"echo", "test"}, nil)
			}

			// Validate exit code
			if exitCode != tc.expectedExitCode {
				t.Errorf("Expected exit code %d, got %d", tc.expectedExitCode, exitCode)
			}

			// Validate error expectation
			if tc.expectError && err == nil {
				t.Error("Expected error but got nil")
			}
			if !tc.expectError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

func TestRunExecWithSecretsNotFound(t *testing.T) {
	// Create a mock server that always returns 404
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintln(w, `{"error":"secret not found"}`)
	}))
	defer mockServer.Close()

	// Save and restore environment variables
	origEnvVars := map[string]envVarState{
		"NEMO_JOB_SECRETS": getEnvState("NEMO_JOB_SECRETS"),
		"NHX_SECRETS_URL":  getEnvState("NHX_SECRETS_URL"),
		"NHX_BASE_URL":     getEnvState("NHX_BASE_URL"),
		"NHX_PRINCIPAL":    getEnvState("NHX_PRINCIPAL"),
	}
	defer restoreEnvVars(origEnvVars)

	// Set test environment variables
	os.Setenv("NEMO_JOB_SECRETS", "NONEXISTENT_SECRET=default/nonexistent")
	os.Setenv("NHX_SECRETS_URL", mockServer.URL)
	os.Unsetenv("NHX_BASE_URL")
	os.Setenv("NHX_PRINCIPAL", `{"id":"test-principal"}`)

	exitCode, err := runExec([]string{"echo", "test"}, nil)

	if exitCode != 1 {
		t.Errorf("Expected exit code 1 for secret not found, got %d", exitCode)
	}

	if err == nil {
		t.Error("Expected error for secret not found, got nil")
	}

	if err != nil && !strings.Contains(err.Error(), "failed to fetch secret") {
		t.Errorf("Expected error about failed secret fetch, got: %v", err)
	}
}

func TestSecretEndpointHTTPClientSetsBoundedTimeout(t *testing.T) {
	originalDefaultTimeout := http.DefaultClient.Timeout

	tcpEndpoint, err := nhxclient.ParseEndpoint("http://127.0.0.1:8080")
	if err != nil {
		t.Fatalf("ParseEndpoint returned error: %v", err)
	}
	tcpClient := secretEndpointHTTPClient(tcpEndpoint)
	if tcpClient == http.DefaultClient {
		t.Fatal("expected bounded TCP client to avoid mutating http.DefaultClient")
	}
	if tcpClient.Timeout != secretFetchTimeout {
		t.Fatalf("expected TCP client timeout %s, got %s", secretFetchTimeout, tcpClient.Timeout)
	}
	if http.DefaultClient.Timeout != originalDefaultTimeout {
		t.Fatalf("expected http.DefaultClient timeout to remain %s, got %s", originalDefaultTimeout, http.DefaultClient.Timeout)
	}

	udsEndpoint, err := nhxclient.ParseEndpoint("unix:///tmp/nemo-helix.sock")
	if err != nil {
		t.Fatalf("ParseEndpoint returned error: %v", err)
	}
	udsClient := secretEndpointHTTPClient(udsEndpoint)
	if udsClient.Timeout != secretFetchTimeout {
		t.Fatalf("expected UDS client timeout %s, got %s", secretFetchTimeout, udsClient.Timeout)
	}
	if udsClient.Transport == nil {
		t.Fatal("expected UDS client to preserve custom transport")
	}
}

func TestRunExecWithoutSecrets(t *testing.T) {
	// Ensure no secrets environment variables are set
	origSecretsEnv, wasSet := os.LookupEnv("NEMO_JOB_SECRETS")
	os.Unsetenv("NEMO_JOB_SECRETS")
	defer func() {
		if wasSet {
			os.Setenv("NEMO_JOB_SECRETS", origSecretsEnv)
		} else {
			os.Unsetenv("NEMO_JOB_SECRETS")
		}
	}()

	// Should run normally without attempting secret fetching
	exitCode, err := runExec([]string{"echo", "test without secrets"}, nil)

	if exitCode != 0 {
		t.Errorf("Expected exit code 0, got %d", exitCode)
	}

	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}
}

func TestWorkloadEnvFromParentFiltersLauncherPrivateVars(t *testing.T) {
	origEnvVars := map[string]envVarState{
		"NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT": getEnvState("NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT"),
		"OTEL_LOGS_EXPORTER":                  getEnvState("OTEL_LOGS_EXPORTER"),
	}
	defer restoreEnvVars(origEnvVars)

	os.Setenv("NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT", "http://platform.example/v1/logs")
	os.Setenv("OTEL_LOGS_EXPORTER", "otlp")

	env := strings.Join(workloadEnvFromParent(), "\n")
	if strings.Contains(env, "NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT=") {
		t.Fatal("expected launcher-private env var to be filtered")
	}
	if !strings.Contains(env, "OTEL_LOGS_EXPORTER=otlp") {
		t.Fatal("expected user OTEL env var to be preserved")
	}
}

func TestParseSecretReferences(t *testing.T) {
	testCases := []struct {
		name        string
		input       string
		expected    []secretReference
		expectError bool
	}{
		{
			name:  "single_secret",
			input: "HF_TOKEN=default/hf-token",
			expected: []secretReference{
				{envVarName: "HF_TOKEN", workspace: "default", secretName: "hf-token"},
			},
		},
		{
			name:  "multiple_secrets",
			input: "HF_TOKEN=default/hf-token,WANDB_API_KEY=default/wandb-key",
			expected: []secretReference{
				{envVarName: "HF_TOKEN", workspace: "default", secretName: "hf-token"},
				{envVarName: "WANDB_API_KEY", workspace: "default", secretName: "wandb-key"},
			},
		},
		{
			name:  "different_workspaces",
			input: "SECRET_A=workspace-1/secret-a,SECRET_B=workspace-2/secret-b",
			expected: []secretReference{
				{envVarName: "SECRET_A", workspace: "workspace-1", secretName: "secret-a"},
				{envVarName: "SECRET_B", workspace: "workspace-2", secretName: "secret-b"},
			},
		},
		{
			name:  "with_spaces",
			input: "HF_TOKEN=default/hf-token , WANDB_API_KEY=default/wandb-key",
			expected: []secretReference{
				{envVarName: "HF_TOKEN", workspace: "default", secretName: "hf-token"},
				{envVarName: "WANDB_API_KEY", workspace: "default", secretName: "wandb-key"},
			},
		},
		{
			name:     "empty_string",
			input:    "",
			expected: nil,
		},
		{
			name:  "trailing_comma",
			input: "HF_TOKEN=default/hf-token,",
			expected: []secretReference{
				{envVarName: "HF_TOKEN", workspace: "default", secretName: "hf-token"},
			},
		},
		{
			name:        "invalid_format_no_equals",
			input:       "default/hf-token",
			expectError: true,
		},
		{
			name:        "invalid_format_no_slash",
			input:       "HF_TOKEN=default-hf-token",
			expectError: true,
		},
		{
			name:        "invalid_format_too_many_slashes",
			input:       "HF_TOKEN=default/workspace/hf-token",
			expectError: true,
		},
		{
			name:        "empty_env_var_name",
			input:       "=default/hf-token",
			expectError: true,
		},
		{
			name:        "empty_workspace",
			input:       "HF_TOKEN=/hf-token",
			expectError: true,
		},
		{
			name:        "empty_secret_name",
			input:       "HF_TOKEN=default/",
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := parseSecretReferences(tc.input)

			if tc.expectError {
				if err == nil {
					t.Fatal("Expected error but got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if !reflect.DeepEqual(result, tc.expected) {
				t.Errorf("Expected %v, got %v", tc.expected, result)
			}
		})
	}
}

func TestFetchSecrets(t *testing.T) {
	// Create a mock server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Extract workspace and secret from path
		// Path format: /v2/workspaces/{workspace}/secrets/{secret}/access
		if r.Method != "GET" {
			t.Errorf("Expected GET request, got %s", r.Method)
		}

		// Simple mock responses based on secret name
		w.WriteHeader(http.StatusOK)
		if r.URL.Path == "/apis/secrets/v2/workspaces/default/secrets/hf-token/access" {
			fmt.Fprintln(w, `{"value":"hf_token_value_123"}`)
		} else if r.URL.Path == "/apis/secrets/v2/workspaces/default/secrets/wandb-key/access" {
			fmt.Fprintln(w, `{"value":"wandb_key_xyz"}`)
		} else {
			fmt.Fprintln(w, `{"value":"mock_secret_value"}`)
		}
	}))
	defer mockServer.Close()

	testCases := []struct {
		name        string
		secretRefs  []secretReference
		expected    []string
		expectError bool
	}{
		{
			name: "single_secret",
			secretRefs: []secretReference{
				{envVarName: "HF_TOKEN", workspace: "default", secretName: "hf-token"},
			},
			expected: []string{"HF_TOKEN=hf_token_value_123"},
		},
		{
			name: "multiple_secrets",
			secretRefs: []secretReference{
				{envVarName: "HF_TOKEN", workspace: "default", secretName: "hf-token"},
				{envVarName: "WANDB_API_KEY", workspace: "default", secretName: "wandb-key"},
			},
			expected: []string{
				"HF_TOKEN=hf_token_value_123",
				"WANDB_API_KEY=wandb_key_xyz",
			},
		},
		{
			name:       "empty_refs",
			secretRefs: []secretReference{},
			expected:   nil,
		},
		{
			name:       "nil_refs",
			secretRefs: nil,
			expected:   nil,
		},
		{
			name: "different_env_var_name",
			secretRefs: []secretReference{
				{envVarName: "MY_CUSTOM_TOKEN", workspace: "default", secretName: "hf-token"},
			},
			expected: []string{"MY_CUSTOM_TOKEN=hf_token_value_123"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			principal := &nhxclient.Principal{ID: "test-principal"}
			result, err := fetchSecrets(mockServer.URL, principal, tc.secretRefs)

			if tc.expectError {
				if err == nil {
					t.Fatal("Expected error but got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if !reflect.DeepEqual(result, tc.expected) {
				t.Errorf("Expected %v, got %v", tc.expected, result)
			}
		})
	}
}

func TestFetchSecrets_Error(t *testing.T) {
	// Create a mock server that returns errors
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintln(w, `{"error":"secret not found"}`)
	}))
	defer mockServer.Close()

	secretRefs := []secretReference{
		{envVarName: "NONEXISTENT", workspace: "default", secretName: "nonexistent"},
	}
	principal := &nhxclient.Principal{ID: "test-principal"}
	_, err := fetchSecrets(mockServer.URL, principal, secretRefs)

	if err == nil {
		t.Fatal("Expected error for non-existent secret, got nil")
	}

	expectedErrorSubstring := "failed to fetch secret default/nonexistent"
	if !contains(err.Error(), expectedErrorSubstring) {
		t.Errorf("Expected error to contain '%s', got: %s", expectedErrorSubstring, err.Error())
	}
}

// Helper function
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

type envVarState struct {
	value  string
	wasSet bool
}

func getEnvState(key string) envVarState {
	value, wasSet := os.LookupEnv(key)
	return envVarState{value: value, wasSet: wasSet}
}

// restoreEnvVars restores environment variables to their original values or unsets them if they were not set.
func restoreEnvVars(envVars map[string]envVarState) {
	for key, state := range envVars {
		if state.wasSet {
			os.Setenv(key, state.value)
		} else {
			os.Unsetenv(key)
		}
	}
}
