// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestLauncherAPIsUseWorkloadTokens(t *testing.T) {
	for _, operation := range []string{"secrets", "step-config"} {
		for _, principal := range []string{"", `{"id":"stale-user"}`} {
			for _, failure := range []string{"", "discovery", "exchange", "api", "redirect"} {
				t.Run(operation+"/principal="+principal+"/failure="+failure, func(t *testing.T) {
					var exchanges, apiCalls atomic.Int32
					var serverURL string
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						switch r.URL.Path {
						case "/apis/auth/discovery":
							if failure == "discovery" {
								http.Error(w, "unavailable", http.StatusServiceUnavailable)
								return
							}
							fmt.Fprintf(w, `{"oidc":{"workload_token_exchange_enabled":true,"workload_client_id":"helix","workload_token_endpoint":%q}}`, serverURL+"/apis/auth/token")
						case "/apis/auth/token":
							exchanges.Add(1)
							if failure == "exchange" {
								http.Error(w, "invalid subject", http.StatusBadRequest)
								return
							}
							if err := r.ParseForm(); err != nil || r.Form.Get("subject_token") != "projected-token" {
								t.Error("expected the projected token to be exchanged")
							}
							fmt.Fprint(w, `{"access_token":"delegated-token","expires_in":300}`)
						case "/service/apis/secrets/v2/workspaces/system/secrets/builder-registry-username/access", "/service" + stepConfigPath:
							apiCalls.Add(1)
							if r.Header.Get("Authorization") != "Bearer delegated-token" {
								t.Error("launcher API request missing workload bearer token")
							}
							for name := range r.Header {
								if strings.HasPrefix(strings.ToLower(name), "x-nhx-principal-") {
									t.Error("launcher forwarded a stale principal header")
								}
							}
							if failure == "api" {
								http.Error(w, "forbidden", http.StatusForbidden)
								return
							}
							if failure == "redirect" {
								http.Redirect(w, r, serverURL+"/redirected", http.StatusFound)
								return
							}
							if operation == "secrets" {
								fmt.Fprint(w, `{"value":"registry-user"}`)
							} else {
								fmt.Fprint(w, `{"config":{"epochs":3}}`)
							}
						default:
							t.Errorf("unexpected launcher request: %s", r.URL.Path)
							http.NotFound(w, r)
						}
					}))
					defer server.Close()
					serverURL = server.URL

					configPath := setStepConfigEnv(t, server.URL+"/service")
					t.Setenv(nhxBaseURLEnv, server.URL)
					t.Setenv("NHX_PRINCIPAL", principal)
					t.Setenv("NHX_SECRETS_URL", server.URL+"/service")
					tokenPath := filepath.Join(t.TempDir(), "token")
					if err := os.WriteFile(tokenPath, []byte("projected-token"), 0o600); err != nil {
						t.Fatal(err)
					}
					t.Setenv(workloadIdentityTokenFileEnv, tokenPath)
					command := `test "$REGISTRY_USER" = registry-user`
					if operation == "secrets" {
						t.Setenv("NEMO_JOB_SECRETS", "REGISTRY_USER=system/builder-registry-username")
					} else {
						enableFetchStepConfig(t)
						command = `test -f "$NEMO_JOB_STEP_CONFIG_FILE_PATH"`
					}
					marker := filepath.Join(t.TempDir(), "started")
					code, err := runExec(context.Background(), []string{"sh", "-c", command + ` && touch "$1"`, "sh", marker}, nil)
					if failure == "" {
						if err != nil || code != 0 {
							t.Fatalf("workload failed: code=%d err=%v", code, err)
						}
						if _, err := os.Stat(marker); err != nil {
							t.Fatal("workload did not start")
						}
					} else {
						if err == nil || code != 1 {
							t.Fatalf("expected launch failure, got code=%d err=%v", code, err)
						}
						if _, err := os.Stat(marker); !os.IsNotExist(err) {
							t.Fatal("workload started despite authentication/API failure")
						}
						if _, err := os.Stat(configPath); !os.IsNotExist(err) {
							t.Fatal("config written despite authentication/API failure")
						}
					}
					if failure == "discovery" || failure == "exchange" {
						if apiCalls.Load() != 0 {
							t.Fatal("authentication failed but launcher still called the API")
						}
					} else if apiCalls.Load() != 1 || exchanges.Load() != 1 {
						t.Fatalf("expected one exchange and API call, got %d and %d", exchanges.Load(), apiCalls.Load())
					}
				})
			}
		}
	}
}
