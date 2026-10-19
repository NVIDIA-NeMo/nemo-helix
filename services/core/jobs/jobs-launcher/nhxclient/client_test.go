// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestSecretClient_GetSecret(t *testing.T) {
	testCases := []struct {
		name          string
		workspace     string
		secretName    string
		principal     *Principal
		statusCode    int
		responseBody  string
		expectError   bool
		errorContains string
		expectedData  string
	}{
		{
			name:         "success",
			workspace:    "workspace-123",
			secretName:   "my-secret",
			principal:    &Principal{ID: "test-principal"},
			statusCode:   http.StatusOK,
			responseBody: `{"value":"mock-secret-data"}`,
			expectError:  false,
			expectedData: "mock-secret-data",
		},
		{
			name:         "different_workspace_and_secret",
			workspace:    "ws-001",
			secretName:   "api-key",
			principal:    &Principal{ID: "test-principal"},
			statusCode:   http.StatusOK,
			responseBody: `{"value":"secret-value"}`,
			expectError:  false,
			expectedData: "secret-value",
		},
		{
			name:         "production_workspace",
			workspace:    "production-workspace",
			secretName:   "database-password",
			principal:    &Principal{ID: "test-principal"},
			statusCode:   http.StatusOK,
			responseBody: `{"value":"db-pass-123"}`,
			expectError:  false,
			expectedData: "db-pass-123",
		},
		{
			name:          "not_found",
			workspace:     "workspace-123",
			secretName:    "nonexistent",
			principal:     &Principal{ID: "test-principal"},
			statusCode:    http.StatusNotFound,
			responseBody:  `{"error":"secret not found"}`,
			expectError:   true,
			errorContains: "failed to get secret: status code 404",
		},
		{
			name:          "server_error",
			workspace:     "workspace-123",
			secretName:    "my-secret",
			principal:     &Principal{ID: "test-principal"},
			statusCode:    http.StatusInternalServerError,
			responseBody:  `{"error":"internal server error"}`,
			expectError:   true,
			errorContains: "failed to get secret: status code 500",
		},
		{
			name:          "invalid_json",
			workspace:     "workspace-123",
			secretName:    "my-secret",
			principal:     &Principal{ID: "test-principal"},
			statusCode:    http.StatusOK,
			responseBody:  `{invalid json}`,
			expectError:   true,
			errorContains: "failed to decode secret response",
		},
		{
			name:         "nil_auth_context",
			workspace:    "workspace-123",
			secretName:   "my-secret",
			principal:    nil,
			statusCode:   http.StatusOK,
			responseBody: `{"value":"no-auth-data"}`,
			expectError:  false,
			expectedData: "no-auth-data",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			expectedPath := fmt.Sprintf("/apis/secrets/v2/workspaces/%s/secrets/%s/access", tc.workspace, tc.secretName)

			mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Verify the request method
				if r.Method != "GET" {
					t.Errorf("Expected GET request, got %s", r.Method)
				}

				// Verify the URL path format
				if r.URL.Path != expectedPath {
					t.Errorf("Expected path %s, got %s", expectedPath, r.URL.Path)
				}

				// Verify the auth headers
				if tc.principal != nil && tc.principal.ID != "" {
					// Service-to-service auth: jobs service acts on behalf of the user
					if r.Header.Get("X-NHX-Principal-Id") != "service:jobs" {
						t.Errorf("Expected principal ID 'service:jobs', got '%s'", r.Header.Get("X-NHX-Principal-Id"))
					}
					if r.Header.Get("X-NHX-Principal-On-Behalf-Of") != tc.principal.ID {
						t.Errorf("Expected on-behalf-of '%s', got '%s'", tc.principal.ID, r.Header.Get("X-NHX-Principal-On-Behalf-Of"))
					}
				}

				// Return the mock response
				w.WriteHeader(tc.statusCode)
				fmt.Fprintln(w, tc.responseBody)
			}))
			defer mockServer.Close()

			client := NewSecretClient(mockServer.URL, tc.principal)
			secret, err := client.GetSecret(tc.workspace, tc.secretName)

			// Check error expectations
			if tc.expectError {
				if err == nil {
					t.Fatal("Expected error, got nil")
				}
				if !strings.Contains(err.Error(), tc.errorContains) {
					t.Errorf("Expected error containing '%s', got '%s'", tc.errorContains, err.Error())
				}
				return
			}

			// Check success expectations
			if err != nil {
				t.Fatalf("Expected no error, got %v", err)
			}

			if secret.Value != tc.expectedData {
				t.Errorf("Expected secret data '%s', got %s", tc.expectedData, secret.Value)
			}

			if secret.Name != tc.secretName {
				t.Errorf("Expected secret name '%s', got %s", tc.secretName, secret.Name)
			}

			if secret.WorkspaceID != tc.workspace {
				t.Errorf("Expected workspace ID '%s', got %s", tc.workspace, secret.WorkspaceID)
			}
		})
	}
}

func TestSecretClient_GetSecretOverUDS(t *testing.T) {
	socketFile, err := os.CreateTemp("", "nhx-*.sock")
	if err != nil {
		t.Fatalf("failed to create temp socket path: %v", err)
	}
	socketPath := socketFile.Name()
	socketFile.Close()
	os.Remove(socketPath)
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("failed to listen on unix socket: %v", err)
	}

	server := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/apis/secrets/v2/workspaces/default/secrets/api-key/access" {
				t.Errorf("unexpected path: %s", r.URL.Path)
				http.Error(w, "unexpected path", http.StatusNotFound)
				return
			}
			w.WriteHeader(http.StatusOK)
			fmt.Fprintln(w, `{"value":"secret-over-uds"}`)
		}),
	}
	defer server.Close()
	t.Cleanup(func() {
		if err := os.Remove(socketPath); err != nil && !os.IsNotExist(err) {
			t.Errorf("failed to remove unix socket path: %v", err)
		}
	})
	go func() {
		_ = server.Serve(listener)
	}()

	endpoint, err := ParseEndpoint("unix://" + socketPath)
	if err != nil {
		t.Fatalf("ParseEndpoint returned error: %v", err)
	}
	client := NewSecretClientForEndpoint(endpoint, &Principal{ID: "test-principal"})

	secret, err := client.GetSecret("default", "api-key")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if secret.Value != "secret-over-uds" {
		t.Fatalf("unexpected secret value: %s", secret.Value)
	}
}

func TestSecretClient_AuthHeaders(t *testing.T) {
	testCases := []struct {
		name                    string
		principal               *Principal
		expectedPrincipalID     string
		expectedOnBehalfOf      string
		expectedPrincipalEmail  string
		expectedPrincipalGroups string
	}{
		{
			name: "all_fields",
			principal: &Principal{
				ID:     "user-123",
				Email:  "test@example.com",
				Groups: []string{"group1", "group2"},
			},
			expectedPrincipalID:     "service:jobs",
			expectedOnBehalfOf:      "user-123",
			expectedPrincipalEmail:  "",
			expectedPrincipalGroups: "",
		},
		{
			name: "id_only",
			principal: &Principal{
				ID: "user-456",
			},
			expectedPrincipalID:     "service:jobs",
			expectedOnBehalfOf:      "user-456",
			expectedPrincipalEmail:  "",
			expectedPrincipalGroups: "",
		},
		{
			name:                    "nil_context",
			principal:               nil,
			expectedPrincipalID:     "",
			expectedOnBehalfOf:      "",
			expectedPrincipalEmail:  "",
			expectedPrincipalGroups: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Verify headers
				if r.Header.Get("X-NHX-Principal-Id") != tc.expectedPrincipalID {
					t.Errorf("Expected X-NHX-Principal-Id '%s', got '%s'", tc.expectedPrincipalID, r.Header.Get("X-NHX-Principal-Id"))
				}
				if r.Header.Get("X-NHX-Principal-On-Behalf-Of") != tc.expectedOnBehalfOf {
					t.Errorf("Expected X-NHX-Principal-On-Behalf-Of '%s', got '%s'", tc.expectedOnBehalfOf, r.Header.Get("X-NHX-Principal-On-Behalf-Of"))
				}
				if r.Header.Get("X-NHX-Principal-Email") != tc.expectedPrincipalEmail {
					t.Errorf("Expected X-NHX-Principal-Email '%s', got '%s'", tc.expectedPrincipalEmail, r.Header.Get("X-NHX-Principal-Email"))
				}
				if r.Header.Get("X-NHX-Principal-Groups") != tc.expectedPrincipalGroups {
					t.Errorf("Expected X-NHX-Principal-Groups '%s', got '%s'", tc.expectedPrincipalGroups, r.Header.Get("X-NHX-Principal-Groups"))
				}

				w.WriteHeader(http.StatusOK)
				fmt.Fprintln(w, `{"value":"test"}`)
			}))
			defer mockServer.Close()

			client := NewSecretClient(mockServer.URL, tc.principal)
			_, _ = client.GetSecret("ws", "secret")
		})
	}
}

func TestPrincipalFromEnv(t *testing.T) {
	testCases := []struct {
		name           string
		envValue       string
		expectNil      bool
		expectedID     string
		expectedEmail  string
		expectedGroups []string
	}{
		{
			name:           "all_fields",
			envValue:       `{"id":"user-123","email":"test@example.com","groups":["group1","group2","group3"]}`,
			expectNil:      false,
			expectedID:     "user-123",
			expectedEmail:  "test@example.com",
			expectedGroups: []string{"group1", "group2", "group3"},
		},
		{
			name:           "id_only",
			envValue:       `{"id":"user-456"}`,
			expectNil:      false,
			expectedID:     "user-456",
			expectedEmail:  "",
			expectedGroups: nil,
		},
		{
			name:      "empty_env_returns_nil",
			envValue:  "",
			expectNil: true,
		},
		{
			name:      "invalid_json_returns_nil",
			envValue:  `{invalid json}`,
			expectNil: true,
		},
		{
			name:      "missing_id_returns_nil",
			envValue:  `{"email":"test@example.com"}`,
			expectNil: true,
		},
		{
			name:      "empty_id_returns_nil",
			envValue:  `{"id":"","email":"test@example.com"}`,
			expectNil: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Save and restore environment variable
			origValue, wasSet := os.LookupEnv("NHX_PRINCIPAL")
			defer func() {
				if wasSet {
					os.Setenv("NHX_PRINCIPAL", origValue)
				} else {
					os.Unsetenv("NHX_PRINCIPAL")
				}
			}()

			// Set environment variable for test
			if tc.envValue != "" {
				os.Setenv("NHX_PRINCIPAL", tc.envValue)
			} else {
				os.Unsetenv("NHX_PRINCIPAL")
			}

			ctx := PrincipalFromEnv()

			if tc.expectNil {
				if ctx != nil {
					t.Errorf("Expected nil, got %+v", ctx)
				}
				return
			}

			if ctx == nil {
				t.Fatal("Expected non-nil Principal")
			}

			if ctx.ID != tc.expectedID {
				t.Errorf("Expected ID '%s', got '%s'", tc.expectedID, ctx.ID)
			}

			if ctx.Email != tc.expectedEmail {
				t.Errorf("Expected Email '%s', got '%s'", tc.expectedEmail, ctx.Email)
			}

			if len(ctx.Groups) != len(tc.expectedGroups) {
				t.Errorf("Expected %d groups, got %d", len(tc.expectedGroups), len(ctx.Groups))
			} else {
				for i, expected := range tc.expectedGroups {
					if ctx.Groups[i] != expected {
						t.Errorf("Expected group[%d] '%s', got '%s'", i, expected, ctx.Groups[i])
					}
				}
			}
		})
	}
}
