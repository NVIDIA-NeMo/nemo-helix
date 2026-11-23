// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package nhxclient

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"go.opentelemetry.io/otel/exporters/otlp/otlplog/otlploghttp"
	sdklog "go.opentelemetry.io/otel/sdk/log"
)

func TestValidateSecretPrincipalWithoutDiscoveryEndpoint(t *testing.T) {
	for _, tc := range []struct {
		name         string
		principal    *Principal
		workloadFile string
		accessFile   string
		allowed      bool
	}{
		{name: "missing_creator"},
		{name: "empty_creator", principal: &Principal{}},
		{name: "trusted_creator", principal: &Principal{ID: "creator"}, allowed: true},
		{name: "workload_bearer", workloadFile: "/configured/workload-token", allowed: true},
		{name: "access_bearer", accessFile: "/configured/access-token", allowed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("NHX_AUTH_URL", "")
			t.Setenv(nhxBaseURLEnv, "")
			t.Setenv(workloadIdentityTokenFileEnv, tc.workloadFile)
			t.Setenv("NHX_ACCESS_TOKEN_FILE", tc.accessFile)
			err := NewSessionFromEnv().ValidateSecretPrincipal(context.Background(), tc.principal)
			if (err == nil) != tc.allowed {
				t.Fatalf("allowed=%v, got error %v", tc.allowed, err)
			}
		})
	}
}

func TestSessionCredentialModes(t *testing.T) {
	for _, mode := range []string{"trusted", "access-file", "workload"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv(workloadIdentityTokenFileEnv, "")
			t.Setenv("NHX_ACCESS_TOKEN_FILE", "")
			t.Setenv("NHX_ACCESS_TOKEN", "ignored")
			file := filepath.Join(t.TempDir(), "token")
			os.WriteFile(file, []byte("token-1"), 0600)
			var exchanges atomic.Int32
			var baseURL string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/apis/auth/discovery":
					fmt.Fprintf(w, `{"oidc":{"workload_token_exchange_enabled":true,"workload_client_id":"runtime","workload_token_endpoint":%q}}`, baseURL+"/token")
				case "/token":
					exchanges.Add(1)
					fmt.Fprint(w, `{"access_token":"exchanged","expires_in":100}`)
				default:
					if mode == "trusted" {
						if r.URL.Path != "/logs" && r.Header.Get("X-NHX-Principal-On-Behalf-Of-Groups") != "readers" {
							t.Error("creator groups lost")
						}
						if r.Header.Get("Authorization") != "" {
							t.Error("ambient token used")
						}
					} else {
						expected := "Bearer exchanged"
						if mode == "access-file" {
							expected = "Bearer token-1"
						}
						if r.Header.Get("Authorization") != expected {
							t.Errorf("unexpected bearer %q", r.Header.Get("Authorization"))
						}
						if r.Header.Get("X-NHX-Principal-Id") != "" {
							t.Error("bearer mixed with trusted identity")
						}
					}
					fmt.Fprint(w, `{"value":"synthetic","config":{}}`)
				}
			}))
			defer server.Close()
			baseURL = server.URL
			t.Setenv(nhxBaseURLEnv, baseURL)
			if mode == "access-file" {
				t.Setenv("NHX_ACCESS_TOKEN_FILE", file)
			}
			if mode == "workload" {
				t.Setenv(workloadIdentityTokenFileEnv, file)
				t.Setenv("NHX_ACCESS_TOKEN_FILE", "/missing/ignored")
			}
			session := NewSessionFromEnv()
			endpoint, _ := ParseEndpoint(baseURL)
			principal := &Principal{ID: "creator", Groups: []string{"readers"}}
			var wg sync.WaitGroup
			for range 8 {
				wg.Go(func() {
					if _, err := session.SecretClient(endpoint, principal).GetSecret(context.Background(), "ws", "key"); err != nil {
						t.Error(err)
					}
				})
			}
			wg.Wait()
			if _, err := session.JobStepClient(endpoint, principal).GetJobStepConfig(context.Background(), "ws", "job", "step"); err != nil {
				t.Fatal(err)
			}
			response, err := session.OTLPClient(server.Client(), baseURL+"/logs").Get(baseURL + "/logs")
			if err != nil {
				t.Fatal(err)
			}
			response.Body.Close()
			if mode == "workload" && exchanges.Load() != 1 {
				t.Fatalf("shared session exchanged %d times", exchanges.Load())
			}
		})
	}
}

func TestSessionRotationExplicitAuthAndNeutralTransport(t *testing.T) {
	file := filepath.Join(t.TempDir(), "token")
	os.WriteFile(file, []byte("first"), 0600)
	t.Setenv(workloadIdentityTokenFileEnv, "")
	t.Setenv("NHX_ACCESS_TOKEN_FILE", file)
	session := NewSessionFromEnv()
	seen := []string{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Header.Get("Authorization"))
		fmt.Fprint(w, `{}`)
	}))
	defer server.Close()
	base := server.Client()
	for _, token := range []string{"first", "second"} {
		os.WriteFile(file, []byte(token), 0600)
		req, _ := http.NewRequestWithContext(context.Background(), "GET", server.URL, nil)
		req.Header.Set("X-NHX-Principal-Id", "must-not-cross")
		resp, err := session.do(req, base, nil)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if req.Header.Get("Authorization") != "" {
			t.Error("caller request mutated")
		}
	}
	os.Remove(file)
	req, _ := http.NewRequestWithContext(context.Background(), "GET", server.URL, nil)
	req.Header.Set("Authorization", "Bearer explicit")
	resp, err := session.do(req, base, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	resp, err = base.Get(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	want := []string{"Bearer first", "Bearer second", "Bearer explicit", ""}
	for i, v := range want {
		if seen[i] != v {
			t.Errorf("request %d: %q != %q", i, seen[i], v)
		}
	}
	req.Header.Del("Authorization")
	if _, err := session.do(req, base, &Principal{ID: "creator"}); err == nil {
		t.Fatal("missing selected file fell back")
	}
}

func TestSessionRefusesCredentialRedirectsAndUnsafeDestinations(t *testing.T) {
	var leaked atomic.Bool
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { leaked.Store(true) }))
	defer target.Close()
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer source.Close()
	req, _ := http.NewRequestWithContext(context.Background(), "GET", source.URL, nil)
	req.Header.Set("Authorization", "Bearer explicit")
	session := &Session{}
	resp, err := session.do(req, http.DefaultClient, nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 307 || leaked.Load() {
		t.Fatal("credential redirect followed")
	}
	req.URL, _ = url.Parse("http://remote.invalid/api")
	if _, err = session.do(req, http.DefaultClient, nil); err == nil {
		t.Fatal("insecure remote accepted")
	}
}

func TestSecretClientHonorsCancellation(t *testing.T) {
	reached := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { close(reached); <-r.Context().Done() }))
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := NewSecretClient(server.URL, &Principal{ID: "creator"}).GetSecret(ctx, "ws", "key")
		done <- err
	}()
	<-reached
	cancel()
	if err := <-done; err == nil {
		t.Fatal("cancelled request succeeded")
	}
}

func TestSessionBearerOverUDS(t *testing.T) {
	for _, mode := range []string{"access-file", "workload"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv(workloadIdentityTokenFileEnv, "")
			t.Setenv("NHX_ACCESS_TOKEN_FILE", "")
			file := filepath.Join(t.TempDir(), "token")
			if err := os.WriteFile(file, []byte("synthetic-proof"), 0600); err != nil {
				t.Fatal(err)
			}
			socketDir, err := os.MkdirTemp("/tmp", "nhx-session-")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { os.RemoveAll(socketDir) })
			socketPath := filepath.Join(socketDir, "platform.sock")
			listener, err := net.Listen("unix", socketPath)
			if err != nil {
				t.Fatal(err)
			}
			var exchanges atomic.Int32
			var requests atomic.Int32
			server := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/apis/auth/discovery":
					fmt.Fprintf(w, `{"oidc":{"workload_token_exchange_enabled":true,"workload_client_id":"runtime","workload_token_endpoint":%q}}`, UDSBaseURL+"/token")
				case "/token":
					if err := r.ParseForm(); err != nil {
						t.Error(err)
					}
					if r.Form.Get("subject_token") != "synthetic-proof" {
						t.Error("workload proof lost")
					}
					exchanges.Add(1)
					fmt.Fprint(w, `{"access_token":"exchanged","expires_in":100}`)
				default:
					requests.Add(1)
					want := "Bearer synthetic-proof"
					if mode == "workload" {
						want = "Bearer exchanged"
					}
					if r.Header.Get("Authorization") != want {
						t.Errorf("unexpected bearer %q", r.Header.Get("Authorization"))
					}
					if r.Header.Get("X-NHX-Principal-Id") != "" {
						t.Error("bearer mixed with trusted identity")
					}
					if r.URL.Path != "/logs" {
						fmt.Fprint(w, `{"value":"synthetic-secret","config":{"setting":true}}`)
					}
				}
			})}
			t.Cleanup(func() { server.Close() })
			go func() { _ = server.Serve(listener) }()
			t.Setenv(nhxBaseURLEnv, "unix://"+socketPath)
			if mode == "workload" {
				t.Setenv(workloadIdentityTokenFileEnv, file)
				t.Setenv("NHX_ACCESS_TOKEN_FILE", "/missing/ignored")
			} else {
				t.Setenv("NHX_ACCESS_TOKEN_FILE", file)
			}
			session := NewSessionFromEnv()
			endpoint, err := ResolveHelixEndpointFromEnv()
			if err != nil {
				t.Fatal(err)
			}
			principal := &Principal{ID: "creator", Groups: []string{"readers"}}
			ctx := context.Background()
			secret, err := session.SecretClient(endpoint, principal).GetSecret(ctx, "ws", "key")
			if err != nil {
				t.Fatal(err)
			}
			if secret.Value != "synthetic-secret" {
				t.Fatal("Secret value lost")
			}
			config, err := session.JobStepClient(endpoint, principal).GetJobStepConfig(ctx, "ws", "job", "step")
			if err != nil {
				t.Fatal(err)
			}
			if string(config) != `{"setting":true}` {
				t.Fatalf("unexpected step config %s", config)
			}
			logURL := endpoint.ConnectBaseURL + "/logs"
			exporter, err := otlploghttp.New(ctx,
				otlploghttp.WithEndpointURL(logURL),
				otlploghttp.WithHTTPClient(session.OTLPClient(endpoint.HTTPClient(), logURL)),
				otlploghttp.WithHeaders(map[string]string{}),
				otlploghttp.WithRetry(otlploghttp.RetryConfig{Enabled: false}),
			)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { exporter.Shutdown(ctx) })
			if err := exporter.Export(ctx, []sdklog.Record{{}}); err != nil {
				t.Fatal(err)
			}
			if requests.Load() != 3 {
				t.Fatalf("expected Secret, step config and OTLP requests, got %d", requests.Load())
			}
			wantExchanges := int32(0)
			if mode == "workload" {
				wantExchanges = 1
			}
			if exchanges.Load() != wantExchanges {
				t.Fatalf("expected %d exchanges, got %d", wantExchanges, exchanges.Load())
			}
			// A service override must use TCP even when platform discovery uses a socket.
			tcpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") == "" {
					t.Error("service override lost bearer authentication")
				}
				fmt.Fprint(w, `{"value":"tcp-secret"}`)
			}))
			defer tcpServer.Close()
			tcpEndpoint, err := ParseEndpoint(tcpServer.URL)
			if err != nil {
				t.Fatal(err)
			}
			tcpSecret, err := session.SecretClient(tcpEndpoint, principal).GetSecret(ctx, "ws", "key")
			if err != nil {
				t.Fatal(err)
			}
			if tcpSecret.Value != "tcp-secret" {
				t.Fatal("TCP service override used the platform socket")
			}
		})
	}
}

func TestSessionRejectsSyntheticHostOverTCP(t *testing.T) {
	t.Setenv(workloadIdentityTokenFileEnv, "")
	t.Setenv("NHX_ACCESS_TOKEN_FILE", "/unused-must-reject-before-reading")
	endpoint, err := ParseEndpoint(UDSBaseURL)
	if err != nil {
		t.Fatal(err)
	}
	session := NewSessionFromEnv()
	_, err = session.SecretClient(endpoint, nil).GetSecret(context.Background(), "ws", "key")
	if err == nil || !strings.Contains(err.Error(), "credentials require HTTPS") {
		t.Fatalf("expected unsafe TCP destination rejection, got %v", err)
	}
}

func TestOTLPAdapterMatchesExporterURL(t *testing.T) {
	for _, mode := range []string{"trusted", "access-file"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv(workloadIdentityTokenFileEnv, "")
			t.Setenv("NHX_ACCESS_TOKEN_FILE", "")
			if mode == "access-file" {
				file := filepath.Join(t.TempDir(), "token")
				if err := os.WriteFile(file, []byte("synthetic-token"), 0600); err != nil {
					t.Fatal(err)
				}
				t.Setenv("NHX_ACCESS_TOKEN_FILE", file)
			}
			for _, path := range []string{"", "/", "/logs", "/logs/", "/logs//nested", "/logs/../records", "/logs%2Fpart", "/logs?ignored=1#fragment"} {
				t.Run(path, func(t *testing.T) {
					target, err := url.Parse(path)
					if err != nil {
						t.Fatal(err)
					}
					wantPath := target.Path
					if wantPath == "" {
						wantPath = "/"
					}
					var requests atomic.Int32
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						requests.Add(1)
						if r.URL.Path != wantPath || r.URL.RawQuery != "" {
							t.Errorf("unexpected export URL %s", r.URL)
						}
						if mode == "access-file" {
							if r.Header.Get("Authorization") != "Bearer synthetic-token" || r.Header.Get("X-NHX-Principal-Id") != "" {
								t.Error("unexpected bearer identity")
							}
						} else if r.Header.Get("X-NHX-Principal-Id") != "service:jobs" {
							t.Error("service identity lost")
						}
					}))
					defer server.Close()
					endpoint := server.URL + path
					client := NewSessionFromEnv().OTLPClient(server.Client(), endpoint)
					ctx := context.Background()
					exporter, err := otlploghttp.New(ctx,
						otlploghttp.WithEndpointURL(endpoint),
						otlploghttp.WithHTTPClient(client),
						otlploghttp.WithHeaders(map[string]string{}),
						otlploghttp.WithRetry(otlploghttp.RetryConfig{Enabled: false}),
					)
					if err != nil {
						t.Fatal(err)
					}
					defer exporter.Shutdown(ctx)
					if err := exporter.Export(ctx, []sdklog.Record{{}}); err != nil {
						t.Fatal(err)
					}
					if _, err := client.Get(server.URL + "/unrelated"); err == nil {
						t.Fatal("OTLP adapter authenticated unrelated request")
					}
					if requests.Load() != 1 {
						t.Fatalf("expected one export, got %d requests", requests.Load())
					}
				})
			}
		})
	}
}

func TestSessionUDSDiscoveryWithExternalExchange(t *testing.T) {
	for _, mode := range []string{"success", "redirect", "unsafe"} {
		t.Run(mode, func(t *testing.T) {
			proofFile := filepath.Join(t.TempDir(), "proof")
			if err := os.WriteFile(proofFile, []byte("synthetic-proof"), 0600); err != nil {
				t.Fatal(err)
			}
			var exchanges atomic.Int32
			var redirects atomic.Int32
			exchange := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/redirect" {
					redirects.Add(1)
					http.Redirect(w, r, "/token", http.StatusTemporaryRedirect)
					return
				}
				exchanges.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/token" {
					t.Errorf("unexpected exchange request: %s %s", r.Method, r.URL)
				}
				if err := r.ParseForm(); err != nil {
					t.Error(err)
				}
				if r.Form.Get("subject_token") != "synthetic-proof" {
					t.Error("workload proof lost")
				}
				fmt.Fprint(w, `{"access_token":"exchanged","expires_in":100}`)
			}))
			defer exchange.Close()
			tokenURL := exchange.URL + "/token"
			if mode == "redirect" {
				tokenURL = exchange.URL + "/redirect"
			} else if mode == "unsafe" {
				tokenURL = "http://remote.invalid/token"
			}
			socketDir, err := os.MkdirTemp("/tmp", "nhx-exchange-")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { os.RemoveAll(socketDir) })
			socketPath := filepath.Join(socketDir, "platform.sock")
			listener, err := net.Listen("unix", socketPath)
			if err != nil {
				t.Fatal(err)
			}
			var secretRequests atomic.Int32
			platform := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/apis/auth/discovery":
					fmt.Fprintf(w, `{"oidc":{"workload_token_exchange_enabled":true,"workload_client_id":"runtime","workload_token_endpoint":%q}}`, tokenURL)
				case "/apis/secrets/v2/workspaces/ws/secrets/key/access":
					secretRequests.Add(1)
					if r.Header.Get("Authorization") != "Bearer exchanged" {
						t.Error("exchanged bearer lost")
					}
					fmt.Fprint(w, `{"value":"synthetic-secret"}`)
				default:
					t.Errorf("exchange was routed through platform socket: %s", r.URL)
					http.NotFound(w, r)
				}
			})}
			t.Cleanup(func() { platform.Close() })
			go func() { _ = platform.Serve(listener) }()
			t.Setenv(nhxBaseURLEnv, "unix://"+socketPath)
			t.Setenv(workloadIdentityTokenFileEnv, proofFile)
			t.Setenv("NHX_ACCESS_TOKEN_FILE", "/missing/ignored")
			session := NewSessionWithHTTPClient(exchange.Client())
			endpoint, err := ResolveHelixEndpointFromEnv()
			if err != nil {
				t.Fatal(err)
			}
			secret, err := session.SecretClient(endpoint, nil).GetSecret(context.Background(), "ws", "key")
			if mode == "success" {
				if err != nil {
					t.Fatal(err)
				}
				if secret.Value != "synthetic-secret" || exchanges.Load() != 1 || secretRequests.Load() != 1 {
					t.Fatalf("unexpected result: secret=%q exchanges=%d requests=%d", secret.Value, exchanges.Load(), secretRequests.Load())
				}
				return
			}
			wantError := "credentials require HTTPS"
			if mode == "redirect" {
				wantError = "workload token exchange failed: status 307"
				if redirects.Load() != 1 {
					t.Fatalf("expected one external exchange request, got %d", redirects.Load())
				}
			}
			if err == nil || !strings.Contains(err.Error(), wantError) {
				t.Fatalf("expected %q, got %v", wantError, err)
			}
			if exchanges.Load() != 0 || secretRequests.Load() != 0 {
				t.Fatal("failed exchange continued to a credentialed destination")
			}
		})
	}
}
