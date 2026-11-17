// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/NVIDIA-NeMo/nemo-helix/services/core/jobs/jobs-launcher/nhxclient"
	"go.opentelemetry.io/otel/attribute"
	sdklog "go.opentelemetry.io/otel/sdk/log"
)

func TestLauncherFacilitiesShareOneAuthenticationSession(t *testing.T) {
	var mu sync.Mutex
	exchanges, requests := 0, 0
	var serverURL string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/apis/auth/discovery":
			_ = json.NewEncoder(w).Encode(map[string]any{"oidc": map[string]any{
				"workload_token_exchange_enabled": true,
				"workload_client_id":              "launcher",
				"workload_token_endpoint":         serverURL + "/token",
			}})
		case "/token":
			mu.Lock()
			exchanges++
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{"access_token": "shared-token", "expires_in": 120})
		default:
			mu.Lock()
			requests++
			mu.Unlock()
			if r.Header.Get("Authorization") != "Bearer shared-token" {
				t.Errorf("missing shared bearer on %s", r.URL.Path)
			}
			if r.Header.Get("X-NHX-Principal-Id") != "" || r.Header.Get("X-NHX-Principal-On-Behalf-Of") != "" {
				t.Errorf("legacy identity sent with bearer on %s", r.URL.Path)
			}
			switch r.URL.Path {
			case "/apis/secrets/v2/workspaces/team/secrets/registry/access":
				_, _ = w.Write([]byte(`{"value":"secret-value"}`))
			case "/apis/jobs/v2/workspaces/team/jobs/job/steps/step":
				_, _ = w.Write([]byte(`{"config":{"value":1}}`))
			case "/v1/logs":
				w.WriteHeader(http.StatusOK)
			default:
				http.NotFound(w, r)
			}
		}
	}))
	defer server.Close()
	serverURL = server.URL
	proof := filepath.Join(t.TempDir(), "proof")
	if err := os.WriteFile(proof, []byte("proof"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("NHX_BASE_URL", server.URL)
	t.Setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", proof)

	session, err := nhxclient.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ctx := nhxclient.WithSession(context.Background(), session)
	exporter, err := newLauncherOTLPLogExporter(ctx, server.URL+"/v1/logs")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = exporter.Shutdown(context.Background()) }()
	record := sdklog.Record{}
	record.SetBody(attribute.StringValue("log"))
	if err := exporter.Export(ctx, []sdklog.Record{record}); err != nil {
		t.Fatal(err)
	}

	principal := &nhxclient.Principal{ID: "alice"}
	secrets := nhxclient.NewSecretClientWithHTTPClient(server.URL, principal, session.HTTPClient(server.Client()))
	if _, err := secrets.GetSecret(ctx, "team", "registry"); err != nil {
		t.Fatal(err)
	}
	jobs := nhxclient.NewJobStepClientWithHTTPClient(server.URL, principal, session.HTTPClient(server.Client()))
	if _, err := jobs.GetJobStepConfig(ctx, "team", "job", "step"); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if exchanges != 1 || requests != 3 {
		t.Fatalf("got exchanges=%d requests=%d; want 1,3", exchanges, requests)
	}
}
