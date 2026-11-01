// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
)

// Token format, exchange configuration, refresh, and legacy injection already
// have focused tests. These cases protect the Secret fetch's auth wiring.
func TestRunExecSecretAuthentication(t *testing.T) {
	for _, tc := range []struct {
		name         string
		uds          bool
		exchangeFail bool
		redirect     bool
		wantError    string
	}{
		{name: "bearer_without_legacy_headers"},
		{name: "bearer_over_unix_socket_without_principal", uds: true},
		{name: "exchange_failure_does_not_fall_back", exchangeFail: true, wantError: "workload token exchange failed"},
		{name: "credentials_are_not_redirected", redirect: true, wantError: "status code 302"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var exchanges, accesses, redirects atomic.Int32
			var baseURL string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.URL.Path == "/apis/auth/discovery":
					fmt.Fprintf(w, `{"oidc":{"workload_token_exchange_enabled":true,"workload_client_id":"jobs-client","workload_token_endpoint":%q}}`, baseURL+"/token")
				case r.URL.Path == "/token":
					exchanges.Add(1)
					if tc.exchangeFail {
						w.WriteHeader(http.StatusBadGateway)
						return
					}
					fmt.Fprint(w, `{"access_token":"delegated-access-token","expires_in":300}`)
				case strings.HasPrefix(r.URL.Path, "/apis/secrets/v2/workspaces/default/secrets/"):
					accesses.Add(1)
					if got := r.Header.Get("Authorization"); got != "Bearer delegated-access-token" {
						t.Errorf("Authorization = %q", got)
						w.WriteHeader(http.StatusUnauthorized)
						return
					}
					for key := range r.Header {
						if strings.HasPrefix(strings.ToLower(key), "x-nhx-principal") {
							t.Errorf("legacy identity header sent in workload mode: %s", key)
						}
					}
					if tc.redirect {
						http.Redirect(w, r, baseURL+"/redirected", http.StatusFound)
						return
					}
					fmt.Fprint(w, `{"value":"injected-value"}`)
				case r.URL.Path == "/redirected":
					redirects.Add(1)
					fmt.Fprint(w, `{"value":"injected-value"}`)
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			baseURL = server.URL
			t.Setenv(nhxBaseURLEnv, server.URL)
			t.Setenv("NHX_SECRETS_URL", server.URL)
			t.Setenv("NHX_PRINCIPAL", `{"id":"user@example.com"}`)
			if tc.uds {
				// Keep the socket path short enough for Unix-domain socket limits.
				dir, err := os.MkdirTemp("", "nhx-auth-")
				if err != nil {
					t.Fatal(err)
				}
				defer os.RemoveAll(dir)
				path := filepath.Join(dir, "api.sock")
				listener, err := net.Listen("unix", path)
				if err != nil {
					t.Fatal(err)
				}
				udsServer := &http.Server{Handler: server.Config.Handler}
				defer udsServer.Close()
				go func() { _ = udsServer.Serve(listener) }()
				t.Setenv("NHX_SECRETS_URL", "unix://"+path)
				t.Setenv("NHX_PRINCIPAL", "")
			}
			t.Setenv("NEMO_JOB_SECRETS", "FIRST=default/first,SECOND=default/second")
			proof := filepath.Join(t.TempDir(), "proof")
			if err := os.WriteFile(proof, []byte("nhx_obo_v1.delegation.proof"), 0o600); err != nil {
				t.Fatal(err)
			}
			t.Setenv(workloadIdentityTokenFileEnv, proof)
			marker := filepath.Join(t.TempDir(), "started")
			code, err := runExec(context.Background(), []string{"sh", "-c", `test "$FIRST" = injected-value && test "$SECOND" = injected-value && touch "$1"`, "sh", marker}, nil)
			if tc.wantError != "" {
				if code != 1 || err == nil || !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("runExec = %d, %v; want %q", code, err, tc.wantError)
				}
				if _, err := os.Stat(marker); !os.IsNotExist(err) {
					t.Fatalf("workload started despite Secret fetch failure: %v", err)
				}
				if tc.exchangeFail && accesses.Load() != 0 {
					t.Fatal("secret request sent despite authentication failure")
				}
			} else {
				if code != 0 || err != nil {
					t.Fatalf("runExec = %d, %v", code, err)
				}
				if _, err := os.Stat(marker); err != nil {
					t.Fatalf("workload did not receive injected secrets: %v", err)
				}
				if accesses.Load() != 2 {
					t.Fatalf("secret requests = %d, want 2", accesses.Load())
				}
			}
			if exchanges.Load() != 1 {
				t.Fatalf("exchanges = %d, want 1 (cached across secrets)", exchanges.Load())
			}
			if redirects.Load() != 0 {
				t.Fatal("followed redirect with workload credentials")
			}
		})
	}
}
