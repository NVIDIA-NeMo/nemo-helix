// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/NVIDIA-NeMo/nemo-helix/services/core/jobs/jobs-launcher/nhxclient"
)

func TestRunExecAnonymousSecrets(t *testing.T) {
	for _, host := range []string{"nemo-helix-api:8080", "nemo-helix-api.nemo-helix.svc:8080"} {
		for _, principal := range []string{"", `{"id":"","groups":[]}`} {
			for _, status := range []int{http.StatusOK, http.StatusUnauthorized, http.StatusForbidden} {
				t.Run(fmt.Sprintf("%s/principal=%s/status=%d", host, principal, status), func(t *testing.T) {
					var requests atomic.Int32
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						requests.Add(1)
						if r.Method != http.MethodGet || r.URL.Path != "/apis/secrets/v2/workspaces/default/secrets/test-secret/access" || r.Host != host {
							t.Errorf("unexpected request: %s %s host=%s", r.Method, r.URL, r.Host)
						}
						for key := range r.Header {
							lower := strings.ToLower(key)
							if lower == "authorization" || strings.HasPrefix(lower, "x-nhx-principal-") || strings.HasPrefix(lower, "x-nhx-actor-") || strings.HasPrefix(lower, "x-nhx-subject-") {
								t.Errorf("anonymous request included identity header %s", key)
							}
						}
						w.WriteHeader(status)
						fmt.Fprint(w, `{"value":"injected-value"}`)
					}))
					defer server.Close()
					// Route a real HTTP request locally while preserving the runtime service URL.
					transport := &http.Transport{DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
						return (&net.Dialer{}).DialContext(ctx, network, server.Listener.Addr().String())
					}}
					defer transport.CloseIdleConnections()
					t.Setenv("NHX_BASE_URL", "")
					t.Setenv("NHX_AUTH_URL", "not-a-configured-auth-endpoint")
					t.Setenv("NHX_SECRETS_URL", "http://"+host)
					t.Setenv("NHX_PRINCIPAL", principal)
					t.Setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", "")
					t.Setenv("NHX_ACCESS_TOKEN_FILE", "")
					t.Setenv("NEMO_JOB_SECRETS", "TEST_SECRET=default/test-secret")
					t.Setenv("TEST_SECRET", "")
					marker := filepath.Join(t.TempDir(), "child-started")
					session := nhxclient.NewSessionWithHTTPClient(&http.Client{Transport: transport})
					ctx := context.WithValue(context.Background(), launcherSessionKey{}, session)
					exitCode, err := runExec(ctx, []string{"sh", "-c", `test "$TEST_SECRET" = injected-value && touch "$1"`, "sh", marker}, nil)
					if requests.Load() != 1 {
						t.Errorf("expected exactly one Secret request and no discovery, got %d requests", requests.Load())
					}
					_, markerErr := os.Stat(marker)
					if status == http.StatusOK {
						if err != nil || exitCode != 0 || markerErr != nil {
							t.Fatalf("expected injected Secret and child execution: exit=%d err=%v marker=%v", exitCode, err, markerErr)
						}
					} else {
						if err == nil || exitCode != 1 || !os.IsNotExist(markerErr) {
							t.Fatalf("expected server denial to stop child execution: exit=%d err=%v marker=%v", exitCode, err, markerErr)
						}
						if !strings.Contains(err.Error(), fmt.Sprintf("status code %d", status)) {
							t.Errorf("expected server denial status %d, got %v", status, err)
						}
					}
				})
			}
		}
	}
}

func TestRunExecAnonymousSecretsRejectURLCredentials(t *testing.T) {
	for _, scheme := range []string{"http", "https"} {
		for _, principal := range []string{"", `{"id":""}`} {
			t.Run(scheme+"/principal="+principal, func(t *testing.T) {
				var requests atomic.Int32
				server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					requests.Add(1)
					fmt.Fprint(w, `{"value":"injected-value"}`)
				}))
				if scheme == "https" {
					server.StartTLS()
				} else {
					server.Start()
				}
				defer server.Close()
				t.Setenv("NHX_BASE_URL", "")
				t.Setenv("NHX_SECRETS_URL", strings.Replace(server.URL, "://", "://embedded-user:embedded-password@", 1))
				t.Setenv("NHX_PRINCIPAL", principal)
				t.Setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", "")
				t.Setenv("NHX_ACCESS_TOKEN_FILE", "")
				t.Setenv("NEMO_JOB_SECRETS", "TEST_SECRET=default/test-secret")
				marker := filepath.Join(t.TempDir(), "child-started")
				session := nhxclient.NewSessionWithHTTPClient(server.Client())
				ctx := context.WithValue(context.Background(), launcherSessionKey{}, session)
				exitCode, err := runExec(ctx, []string{"touch", marker}, nil)
				if err == nil || exitCode != 1 || requests.Load() != 0 {
					t.Fatalf("expected URL credentials to be rejected before any request: exit=%d err=%v requests=%d", exitCode, err, requests.Load())
				}
				if strings.Contains(err.Error(), "embedded-user") || strings.Contains(err.Error(), "embedded-password") {
					t.Error("error exposed URL credentials")
				}
				if _, err := os.Stat(marker); !os.IsNotExist(err) {
					t.Fatalf("child ran despite URL credential rejection: %v", err)
				}
			})
		}
	}
}

func TestRunExecSecretCredentialFailureDoesNotFallback(t *testing.T) {
	for _, mode := range []string{"missing-access-token", "empty-access-token", "workload-discovery-failure"} {
		t.Run(mode, func(t *testing.T) {
			var secretRequests atomic.Int32
			var discoveryRequests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/apis/auth/discovery" {
					discoveryRequests.Add(1)
				} else {
					secretRequests.Add(1)
					t.Errorf("failed credentials must not reach Secrets: %s", r.URL.Path)
				}
				w.WriteHeader(http.StatusBadGateway)
			}))
			defer server.Close()
			t.Setenv("NHX_BASE_URL", server.URL)
			t.Setenv("NHX_SECRETS_URL", server.URL)
			t.Setenv("NHX_PRINCIPAL", `{"id":"creator"}`)
			t.Setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", "")
			tokenFile := filepath.Join(t.TempDir(), "token")
			t.Setenv("NHX_ACCESS_TOKEN_FILE", tokenFile)
			if mode == "empty-access-token" {
				if err := os.WriteFile(tokenFile, nil, 0o600); err != nil {
					t.Fatal(err)
				}
			} else if mode == "workload-discovery-failure" {
				t.Setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", tokenFile)
				accessFile := filepath.Join(t.TempDir(), "access-token")
				if err := os.WriteFile(accessFile, []byte("synthetic-lower-priority-token"), 0o600); err != nil {
					t.Fatal(err)
				}
				t.Setenv("NHX_ACCESS_TOKEN_FILE", accessFile)
			}
			t.Setenv("NEMO_JOB_SECRETS", "TEST_SECRET=default/test-secret")
			marker := filepath.Join(t.TempDir(), "child-started")
			exitCode, err := runExec(context.Background(), []string{"touch", marker}, nil)
			if err == nil || exitCode != 1 || secretRequests.Load() != 0 {
				t.Fatalf("expected selected credential failure: exit=%d err=%v requests=%d", exitCode, err, secretRequests.Load())
			}
			if mode == "workload-discovery-failure" {
				if discoveryRequests.Load() != 1 || !strings.Contains(err.Error(), "auth discovery failed: status 502") {
					t.Fatalf("expected the workload discovery error: err=%v discovery requests=%d", err, discoveryRequests.Load())
				}
			} else if discoveryRequests.Load() != 0 {
				t.Fatalf("access-file credentials unexpectedly attempted discovery: %d", discoveryRequests.Load())
			}
			if _, err := os.Stat(marker); !os.IsNotExist(err) {
				t.Fatalf("child ran after credential failure: %v", err)
			}
		})
	}
}
