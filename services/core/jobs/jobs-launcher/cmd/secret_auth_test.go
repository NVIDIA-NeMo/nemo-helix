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

func TestRunExecSecretAuthentication(t *testing.T) {
	for _, tc := range []struct {
		name         string
		proof        string
		disabled     bool
		exchangeFail bool
		missingFile  bool
		redirect     bool
		uds          bool
		noPrincipal  bool
		wantError    string
	}{
		{name: "legacy"},
		{name: "workload_jwt", proof: "subject.jwt.token"},
		{name: "workload_opaque", proof: "nhx_obo_v1.delegation.proof", noPrincipal: true},
		{name: "workload_unix_socket", proof: "subject.jwt.token", uds: true},
		{name: "exchange_disabled", proof: "subject.jwt.token", disabled: true, wantError: "workload token exchange is not enabled"},
		{name: "exchange_failure", proof: "subject.jwt.token", exchangeFail: true, wantError: "workload token exchange failed"},
		{name: "missing_proof", proof: "subject.jwt.token", missingFile: true, wantError: "read NHX_WORKLOAD_IDENTITY_TOKEN_FILE"},
		{name: "redirect", proof: "subject.jwt.token", redirect: true, wantError: "status code 302"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var exchanges, accesses, redirects atomic.Int32
			var baseURL string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.URL.Path == "/apis/auth/discovery":
					fmt.Fprintf(w, `{"oidc":{"workload_token_exchange_enabled":%t,"workload_client_id":"jobs-client","workload_token_endpoint":%q}}`, !tc.disabled, baseURL+"/token")
				case r.URL.Path == "/token":
					exchanges.Add(1)
					if err := r.ParseForm(); err != nil {
						t.Errorf("parse exchange: %v", err)
					}
					if got := r.Form.Get("subject_token"); got != tc.proof {
						t.Errorf("unexpected subject proof %q", got)
					}
					wantType := jwtTokenType
					if strings.HasPrefix(tc.proof, dockerOpaqueWorkloadProofPrefix) {
						wantType = dockerOpaqueWorkloadProofTokenType
					}
					if got := r.Form.Get("subject_token_type"); got != wantType {
						t.Errorf("subject token type = %q, want %q", got, wantType)
					}
					if tc.exchangeFail {
						w.WriteHeader(http.StatusBadGateway)
						return
					}
					// Production exchange binds actor and delegated subject in this token.
					fmt.Fprint(w, `{"access_token":"delegated-access-token","expires_in":300}`)
				case strings.HasPrefix(r.URL.Path, "/apis/secrets/v2/workspaces/default/secrets/"):
					accesses.Add(1)
					if tc.proof != "" {
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
					} else {
						if r.Header.Get("Authorization") != "" || r.Header.Get("X-NHX-Principal-Id") != "service:jobs" || r.Header.Get("X-NHX-Principal-On-Behalf-Of") != "user@example.com" {
							t.Errorf("legacy identity headers not preserved: %v", r.Header)
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
			}
			t.Setenv("NHX_PRINCIPAL", `{"id":"user@example.com"}`)
			if tc.noPrincipal {
				t.Setenv("NHX_PRINCIPAL", "")
			}
			t.Setenv("NEMO_JOB_SECRETS", "FIRST=default/first,SECOND=default/second")
			t.Setenv(workloadIdentityTokenFileEnv, "")
			if tc.proof != "" {
				path := filepath.Join(t.TempDir(), "proof")
				if !tc.missingFile {
					if err := os.WriteFile(path, []byte(tc.proof+"\n"), 0o600); err != nil {
						t.Fatal(err)
					}
				}
				t.Setenv(workloadIdentityTokenFileEnv, path)
			}
			marker := filepath.Join(t.TempDir(), "started")
			code, err := runExec(context.Background(), []string{"sh", "-c", `test "$FIRST" = injected-value && test "$SECOND" = injected-value && touch "$1"`, "sh", marker}, nil)
			if tc.wantError != "" {
				if code != 1 || err == nil || !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("runExec = %d, %v; want %q", code, err, tc.wantError)
				}
				if _, err := os.Stat(marker); !os.IsNotExist(err) {
					t.Fatalf("workload started despite secret authentication failure: %v", err)
				}
				if !tc.redirect && accesses.Load() != 0 {
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
			if tc.proof != "" && !tc.disabled && !tc.missingFile && exchanges.Load() != 1 {
				t.Fatalf("exchanges = %d, want 1 (cached across secrets)", exchanges.Load())
			}
			if redirects.Load() != 0 {
				t.Fatal("followed redirect with workload credentials")
			}
		})
	}
}
