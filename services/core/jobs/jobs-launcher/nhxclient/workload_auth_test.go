// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func getWorkloadHeaders(ctx context.Context) (map[string]string, error) {
	source, err := NewWorkloadTokenSource(ctx)
	if err != nil || source == nil {
		return nil, err
	}
	return source.Headers(ctx)
}

func TestSubjectTokenTypeForExchange(t *testing.T) {
	if got := subjectTokenTypeForExchange("subject-token"); got != jwtTokenType {
		t.Fatalf("expected JWT subject token type, got %q", got)
	}
	if got := subjectTokenTypeForExchange("nhx_obo_v1.delegation.secret"); got != dockerOpaqueWorkloadProofTokenType {
		t.Fatalf("expected Docker opaque subject token type, got %q", got)
	}
}

func TestGetOTLPLogWorkloadAuthHeadersReturnsAuthorizationWithoutMutatingEnv(t *testing.T) {
	subjectTokenPath := filepath.Join(t.TempDir(), "subject.jwt")
	if err := os.WriteFile(subjectTokenPath, []byte("subject-token\n"), 0o600); err != nil {
		t.Fatalf("failed to write subject token: %v", err)
	}

	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/apis/auth/discovery":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(
				w,
				`{"auth_enabled":true,"oidc":{"workload_token_exchange_enabled":true,"workload_client_id":"nemo-helix-workload","workload_token_endpoint":%q,"workload_audience":"nemo-helix","workload_scope":"openid email groups"}}`,
				serverURL+"/apis/auth/token",
			)
		case "/apis/auth/token":
			if r.Method != http.MethodPost {
				t.Errorf("expected POST token exchange, got %s", r.Method)
			}
			if err := r.ParseForm(); err != nil {
				t.Errorf("failed to parse token exchange form: %v", err)
			}
			expectedForm := url.Values{
				"grant_type":           {tokenExchangeGrantType},
				"client_id":            {"nemo-helix-workload"},
				"subject_token":        {"subject-token"},
				"subject_token_type":   {jwtTokenType},
				"requested_token_type": {accessTokenType},
				"audience":             {"nemo-helix"},
				"scope":                {"openid email groups"},
			}
			expectedEncodedForm := expectedForm.Encode()
			actualEncodedForm := r.PostForm.Encode()
			if actualEncodedForm != expectedEncodedForm {
				t.Errorf(
					"form: expected encoded %q (%v), got %q (%v)",
					expectedEncodedForm,
					expectedForm,
					actualEncodedForm,
					r.PostForm,
				)
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"access_token": "access.token.value", "expires_in": 120}) // nolint:errcheck
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	t.Setenv(nhxBaseURLEnv, server.URL)
	t.Setenv(workloadIdentityTokenFileEnv, subjectTokenPath)
	t.Setenv("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "X-NHX-Principal-Id=nemo-user")

	headers, err := getWorkloadHeaders(context.Background())
	if err != nil {
		t.Fatalf("getWorkloadHeaders returned error: %v", err)
	}

	if got := headers["Authorization"]; got != "Bearer access.token.value" {
		t.Fatalf("expected returned bearer header, got %q", got)
	}
	if got := os.Getenv("OTEL_EXPORTER_OTLP_LOGS_HEADERS"); got != "X-NHX-Principal-Id=nemo-user" {
		t.Fatalf("expected OTEL headers env to remain untouched, got %s", got)
	}
}

func TestGetOTLPLogWorkloadAuthHeadersNoopsWithoutTokenFile(t *testing.T) {
	t.Setenv("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "X-NHX-Principal-Id=nemo-user")

	headers, err := getWorkloadHeaders(context.Background())
	if err != nil {
		t.Fatalf("getWorkloadHeaders returned error: %v", err)
	}
	if headers != nil {
		t.Fatalf("expected no auth headers without workload token file, got %v", headers)
	}

	if headers := os.Getenv("OTEL_EXPORTER_OTLP_LOGS_HEADERS"); headers != "X-NHX-Principal-Id=nemo-user" {
		t.Fatalf("expected headers to be unchanged, got %s", headers)
	}
}

func TestWorkloadAuthTokenSourceRefreshesNearExpiry(t *testing.T) {
	subjectTokenPath := filepath.Join(t.TempDir(), "subject.jwt")
	if err := os.WriteFile(subjectTokenPath, []byte("subject-token\n"), 0o600); err != nil {
		t.Fatalf("failed to write subject token: %v", err)
	}

	var serverURL string
	var tokenExchangeMu sync.Mutex
	var tokenExchangeCount int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/apis/auth/discovery":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(
				w,
				`{"auth_enabled":true,"oidc":{"workload_token_exchange_enabled":true,"workload_client_id":"nemo-helix-workload","workload_token_endpoint":%q}}`,
				serverURL+"/apis/auth/token",
			)
		case "/apis/auth/token":
			tokenExchangeMu.Lock()
			tokenExchangeCount++
			token := fmt.Sprintf("access-token-%d", tokenExchangeCount)
			tokenExchangeMu.Unlock()

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{ // nolint:errcheck
				"access_token": token,
				"expires_in":   100,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	t.Setenv(nhxBaseURLEnv, server.URL)
	t.Setenv(workloadIdentityTokenFileEnv, subjectTokenPath)

	source, err := NewWorkloadTokenSource(context.Background())
	if err != nil {
		t.Fatalf("NewWorkloadTokenSource returned error: %v", err)
	}
	if source == nil {
		t.Fatal("expected workload auth token source")
	}

	now := time.Unix(1_700_000_000, 0)
	source.now = func() time.Time { return now }

	header, err := source.AuthorizationHeader(context.Background())
	if err != nil {
		t.Fatalf("first AuthorizationHeader returned error: %v", err)
	}
	if header != "Bearer access-token-1" {
		t.Fatalf("expected first token, got %q", header)
	}

	now = now.Add(79 * time.Second)
	header, err = source.AuthorizationHeader(context.Background())
	if err != nil {
		t.Fatalf("cached AuthorizationHeader returned error: %v", err)
	}
	if header != "Bearer access-token-1" {
		t.Fatalf("expected cached token, got %q", header)
	}

	now = now.Add(2 * time.Second)
	header, err = source.AuthorizationHeader(context.Background())
	if err != nil {
		t.Fatalf("refreshed AuthorizationHeader returned error: %v", err)
	}
	if header != "Bearer access-token-2" {
		t.Fatalf("expected refreshed token, got %q", header)
	}
	tokenExchangeMu.Lock()
	actualTokenExchangeCount := tokenExchangeCount
	tokenExchangeMu.Unlock()
	if actualTokenExchangeCount != 2 {
		t.Fatalf("expected token exchange only when missing and near expiry, got %d", actualTokenExchangeCount)
	}
}

func TestWorkloadAuthTokenSourceSynchronizesConcurrentRefresh(t *testing.T) {
	subjectTokenPath := filepath.Join(t.TempDir(), "subject.jwt")
	if err := os.WriteFile(subjectTokenPath, []byte("subject-token\n"), 0o600); err != nil {
		t.Fatalf("failed to write subject token: %v", err)
	}

	var serverURL string
	var mu sync.Mutex
	var tokenExchangeCount int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/apis/auth/discovery":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(
				w,
				`{"auth_enabled":true,"oidc":{"workload_token_exchange_enabled":true,"workload_client_id":"nemo-helix-workload","workload_token_endpoint":%q}}`,
				serverURL+"/apis/auth/token",
			)
		case "/apis/auth/token":
			mu.Lock()
			tokenExchangeCount++
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"access_token": "access-token", "expires_in": 100}) // nolint:errcheck
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	t.Setenv(nhxBaseURLEnv, server.URL)
	t.Setenv(workloadIdentityTokenFileEnv, subjectTokenPath)

	source, err := NewWorkloadTokenSource(context.Background())
	if err != nil {
		t.Fatalf("NewWorkloadTokenSource returned error: %v", err)
	}
	if source == nil {
		t.Fatal("expected workload auth token source")
	}
	source.now = func() time.Time { return time.Unix(1_700_000_000, 0) }

	const callers = 8
	start := make(chan struct{})
	errs := make(chan error, callers)
	var wg sync.WaitGroup
	for range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			header, err := source.AuthorizationHeader(context.Background())
			if err != nil {
				errs <- err
				return
			}
			if header != "Bearer access-token" {
				errs <- fmt.Errorf("expected cached header, got %q", header)
			}
		}()
	}
	close(start)
	wg.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if tokenExchangeCount != 1 {
		t.Fatalf("expected concurrent refreshes to share one token exchange, got %d", tokenExchangeCount)
	}
}

func TestDiscoverAuthConfigLimitsNonOKResponseBody(t *testing.T) {
	largeBody := strings.Repeat("x", maxAuthResponseBodyBytes+1024) + "tail-sentinel"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, largeBody, http.StatusBadGateway)
	}))
	defer server.Close()

	_, err := discoverAuthConfig(context.Background(), server.URL)
	if err == nil {
		t.Fatal("expected auth discovery error")
	}
	if !strings.Contains(err.Error(), "auth discovery failed: status 502") {
		t.Fatalf("expected status error, got %v", err)
	}
	if strings.Contains(err.Error(), "tail-sentinel") {
		t.Fatalf("expected auth discovery error body to be limited, got %v", err)
	}
}

func TestDiscoverAuthConfigLimitsSuccessfulJSONResponseBody(t *testing.T) {
	padding := strings.Repeat("x", maxAuthResponseBodyBytes)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(
			w,
			`{"oidc":{"workload_token_exchange_enabled":true,"workload_client_id":"client","workload_token_endpoint":"https://idp.example.com/token","padding":%q}}`,
			padding,
		)
	}))
	defer server.Close()

	_, err := discoverAuthConfig(context.Background(), server.URL)
	if err == nil {
		t.Fatal("expected oversized auth discovery response to fail decoding")
	}
	if !strings.Contains(err.Error(), "decode auth discovery response") {
		t.Fatalf("expected decode error, got %v", err)
	}
}
