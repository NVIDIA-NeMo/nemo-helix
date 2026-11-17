// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	sdklog "go.opentelemetry.io/otel/sdk/log"
)

const otelExporterOTLPLogsHeadersEnv = "OTEL_EXPORTER_OTLP_LOGS_HEADERS"

func captureSlogOutput(t *testing.T, output *bytes.Buffer) {
	t.Helper()

	previousLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(output, &slog.HandlerOptions{Level: slog.LevelInfo})))
	t.Cleanup(func() {
		slog.SetDefault(previousLogger)
	})
}

func assertLogContains(t *testing.T, logOutput string, expected string) {
	t.Helper()

	if !strings.Contains(logOutput, expected) {
		t.Fatalf("expected log output to contain %q, got %q", expected, logOutput)
	}
}

func assertLogNotContains(t *testing.T, logOutput string, unexpected string) {
	t.Helper()

	if strings.Contains(logOutput, unexpected) {
		t.Fatalf("expected log output not to contain %q, got %q", unexpected, logOutput)
	}
}

func TestNewLogExporterCachesWorkloadAuthAcrossExports(t *testing.T) {
	subjectTokenPath := filepath.Join(t.TempDir(), "subject.jwt")
	if err := os.WriteFile(subjectTokenPath, []byte("subject-token\n"), 0o600); err != nil {
		t.Fatalf("failed to write subject token: %v", err)
	}
	var logOutput bytes.Buffer
	captureSlogOutput(t, &logOutput)

	var serverURL string
	var mu sync.Mutex
	var tokenExchangeCount int
	var exportAuthHeaders []string
	var exportThirdPartyHeaders []string
	var exportPrincipalHeaders []string
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
			token := fmt.Sprintf("access-token-%d", tokenExchangeCount)
			mu.Unlock()

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"access_token": token, "expires_in": 120}) // nolint:errcheck
		case "/v1/logs":
			mu.Lock()
			exportAuthHeaders = append(exportAuthHeaders, r.Header.Get("Authorization"))
			exportThirdPartyHeaders = append(exportThirdPartyHeaders, r.Header.Get("X-Third-Party"))
			exportPrincipalHeaders = append(exportPrincipalHeaders, r.Header.Get("X-NHX-Principal-Id"))
			mu.Unlock()
			w.WriteHeader(http.StatusOK)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	t.Setenv("NHX_BASE_URL", server.URL)
	t.Setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", subjectTokenPath)
	t.Setenv(launcherOTLPLogsEndpointEnv, server.URL+"/v1/logs")
	t.Setenv("NHX_JOB_LAUNCHER_OTLP_LOGS_HEADERS", "X-NHX-Principal-Id=nemo-user")
	t.Setenv(otelExporterOTLPLogsHeadersEnv, "Authorization=Bearer%20third-party,X-Third-Party=external")

	exporter, err := newLogExporter(context.Background())
	if err != nil {
		t.Fatalf("newLogExporter returned error: %v", err)
	}

	record := sdklog.Record{}
	record.SetBody(attribute.StringValue("first"))
	if err := exporter.Export(context.Background(), []sdklog.Record{record}); err != nil {
		t.Fatalf("first export returned error: %v", err)
	}

	record.SetBody(attribute.StringValue("second"))
	if err := exporter.Export(context.Background(), []sdklog.Record{record}); err != nil {
		t.Fatalf("second export returned error: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if tokenExchangeCount != 1 {
		t.Fatalf("expected cached token to avoid per-export token exchanges, got %d", tokenExchangeCount)
	}
	expectedAuthHeaders := []string{"Bearer access-token-1", "Bearer access-token-1"}
	if strings.Join(exportAuthHeaders, ",") != strings.Join(expectedAuthHeaders, ",") {
		t.Fatalf("expected export Authorization headers %v, got %v", expectedAuthHeaders, exportAuthHeaders)
	}
	expectedThirdPartyHeaders := []string{"", ""}
	if strings.Join(exportThirdPartyHeaders, ",") != strings.Join(expectedThirdPartyHeaders, ",") {
		t.Fatalf("expected user OTEL headers to stay off platform log exports, got %v", exportThirdPartyHeaders)
	}
	expectedPrincipalHeaders := []string{"", ""}
	if strings.Join(exportPrincipalHeaders, ",") != strings.Join(expectedPrincipalHeaders, ",") {
		t.Fatalf("expected launcher OTLP headers to stay off platform log exports, got %v", exportPrincipalHeaders)
	}
	if got := os.Getenv(otelExporterOTLPLogsHeadersEnv); got != "Authorization=Bearer%20third-party,X-Third-Party=external" {
		t.Fatalf("expected OTEL headers env to remain untouched, got %q", got)
	}
	assertLogContains(t, logOutput.String(), "auth_mechanism=workload_identity_token_exchange")
	assertLogContains(t, logOutput.String(), "reason=workload_token_file_configured")
}

func TestNewLogExporterFailsWhenWorkloadExchangeDisabledWithTokenFile(t *testing.T) {
	subjectTokenPath := filepath.Join(t.TempDir(), "subject.jwt")
	if err := os.WriteFile(subjectTokenPath, []byte("subject-token\n"), 0o600); err != nil {
		t.Fatalf("failed to write subject token: %v", err)
	}

	var mu sync.Mutex
	var tokenExchangeCount int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/apis/auth/discovery":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"auth_enabled":true,"oidc":{"workload_token_exchange_enabled":false}}`)
		case "/apis/auth/token":
			mu.Lock()
			tokenExchangeCount++
			mu.Unlock()
			http.Error(w, "unexpected token exchange", http.StatusInternalServerError)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	t.Setenv("NHX_BASE_URL", server.URL)
	t.Setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", subjectTokenPath)
	t.Setenv(launcherOTLPLogsEndpointEnv, server.URL+"/v1/logs")

	_, err := newLogExporter(context.Background())
	if err == nil {
		t.Fatal("expected newLogExporter to fail when token file is configured but workload exchange is disabled")
	}
	if !strings.Contains(err.Error(), "workload token exchange is not enabled by auth discovery") {
		t.Fatalf("expected disabled exchange error, got %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if tokenExchangeCount != 0 {
		t.Fatalf("expected disabled workload exchange to avoid token exchange, got %d calls", tokenExchangeCount)
	}
}

func TestNewLogExporterDoesNotLogAuthMechanismWhenWorkloadTokenExchangeFails(t *testing.T) {
	subjectTokenPath := filepath.Join(t.TempDir(), "subject.jwt")
	if err := os.WriteFile(subjectTokenPath, []byte("subject-token\n"), 0o600); err != nil {
		t.Fatalf("failed to write subject token: %v", err)
	}
	var logOutput bytes.Buffer
	captureSlogOutput(t, &logOutput)

	var serverURL string
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
			http.Error(w, "token exchange unavailable", http.StatusBadGateway)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	serverURL = server.URL

	t.Setenv("NHX_BASE_URL", server.URL)
	t.Setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", subjectTokenPath)
	t.Setenv(launcherOTLPLogsEndpointEnv, server.URL+"/v1/logs")

	_, err := newLogExporter(context.Background())
	if err == nil {
		t.Fatal("expected newLogExporter to fail when workload token exchange fails")
	}
	if !strings.Contains(err.Error(), "workload token exchange failed: status 502") {
		t.Fatalf("expected token exchange failure, got %v", err)
	}

	assertLogNotContains(t, logOutput.String(), "auth_mechanism=workload_identity_token_exchange")
	assertLogNotContains(t, logOutput.String(), "auth_mechanism=service_identity_principal_headers")
}

func TestNewLogExporterUsesServiceIdentityPrincipalHeadersWithoutWorkloadTokenFile(t *testing.T) {
	var logOutput bytes.Buffer
	captureSlogOutput(t, &logOutput)
	var mu sync.Mutex
	var exportHeaders http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/logs" {
			http.NotFound(w, r)
			return
		}

		mu.Lock()
		exportHeaders = r.Header.Clone()
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	t.Setenv(launcherOTLPLogsEndpointEnv, server.URL+"/v1/logs")
	t.Setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", "")
	t.Setenv("NHX_JOB_LAUNCHER_OTLP_LOGS_HEADERS", "X-NHX-Principal-Id=32ac8159-42b0-43b0-b0d3-bb891c859c92,X-NHX-Principal-Email=rsadler%40nvidia.com")
	t.Setenv(otelExporterOTLPLogsHeadersEnv, "Authorization=Bearer%20third-party,X-Third-Party=external")

	exporter, err := newLogExporter(context.Background())
	if err != nil {
		t.Fatalf("newLogExporter returned error: %v", err)
	}
	defer func() {
		if err := exporter.Shutdown(context.Background()); err != nil {
			t.Fatalf("shutdown returned error: %v", err)
		}
	}()

	record := sdklog.Record{}
	record.SetBody(attribute.StringValue("service identity auth"))
	if err := exporter.Export(context.Background(), []sdklog.Record{record}); err != nil {
		t.Fatalf("export returned error: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if exportHeaders == nil {
		t.Fatal("expected log export request")
	}
	if got := exportHeaders.Get("Authorization"); got != "" {
		t.Fatalf("expected no Authorization header without workload token file, got %q", got)
	}
	if got := exportHeaders.Get("X-Third-Party"); got != "" {
		t.Fatalf("expected user OTEL headers to stay off platform log exports, got X-Third-Party=%q", got)
	}
	if got := exportHeaders.Get("X-NHX-Principal-Id"); got != serviceJobsPrincipal {
		t.Fatalf("expected service identity principal header without workload token file, got X-NHX-Principal-Id=%q", got)
	}
	if got := os.Getenv(otelExporterOTLPLogsHeadersEnv); got != "Authorization=Bearer%20third-party,X-Third-Party=external" {
		t.Fatalf("expected OTEL headers env to remain untouched, got %q", got)
	}
	assertLogContains(t, logOutput.String(), "auth_mechanism=service_identity_principal_headers")
	assertLogContains(t, logOutput.String(), "reason=workload_token_file_not_configured")
}

func TestNewLogExporterPropagatesInitialWorkloadAuthFailure(t *testing.T) {
	subjectTokenPath := filepath.Join(t.TempDir(), "subject.jwt")
	if err := os.WriteFile(subjectTokenPath, []byte("subject-token\n"), 0o600); err != nil {
		t.Fatalf("failed to write subject token: %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "discovery unavailable", http.StatusBadGateway)
	}))
	defer server.Close()

	t.Setenv("NHX_BASE_URL", server.URL)
	t.Setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", subjectTokenPath)
	t.Setenv(launcherOTLPLogsEndpointEnv, server.URL+"/v1/logs")

	_, err := newLogExporter(context.Background())
	if err == nil {
		t.Fatal("expected newLogExporter to return workload auth error")
	}
	if !strings.Contains(err.Error(), "configure workload identity auth for OTLP logs") {
		t.Fatalf("expected setup auth context in error, got %v", err)
	}
	if !strings.Contains(err.Error(), "auth discovery failed: status 502") {
		t.Fatalf("expected discovery status in error, got %v", err)
	}
}
