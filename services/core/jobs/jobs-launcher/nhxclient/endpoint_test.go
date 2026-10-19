// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"testing"
)

func TestParseEndpointTCP(t *testing.T) {
	endpoint, err := ParseEndpoint("http://127.0.0.1:8080/")
	if err != nil {
		t.Fatalf("ParseEndpoint returned error: %v", err)
	}
	if endpoint.ConnectBaseURL != "http://127.0.0.1:8080" {
		t.Fatalf("unexpected ConnectBaseURL: %s", endpoint.ConnectBaseURL)
	}
	if endpoint.Transport != TransportTCP {
		t.Fatalf("unexpected transport: %s", endpoint.Transport)
	}
}

func TestParseEndpointUDS(t *testing.T) {
	endpoint, err := ParseEndpoint("unix:///tmp/nemo-helix.sock")
	if err != nil {
		t.Fatalf("ParseEndpoint returned error: %v", err)
	}
	if endpoint.ConnectBaseURL != UDSBaseURL {
		t.Fatalf("unexpected ConnectBaseURL: %s", endpoint.ConnectBaseURL)
	}
	if endpoint.SocketPath != "/tmp/nemo-helix.sock" {
		t.Fatalf("unexpected socket path: %s", endpoint.SocketPath)
	}
	if endpoint.Transport != TransportUDS {
		t.Fatalf("unexpected transport: %s", endpoint.Transport)
	}
}

func TestParseEndpointRejectsRawSocketPath(t *testing.T) {
	if _, err := ParseEndpoint("/tmp/nemo-helix.sock"); err == nil {
		t.Fatal("expected raw socket path to be rejected")
	}
}

func TestResolveServiceEndpointPrefersServiceURL(t *testing.T) {
	t.Setenv("NHX_BASE_URL", "http://platform:8080")
	t.Setenv("NHX_SECRETS_URL", "unix:///tmp/secrets.sock")

	endpoint, err := ResolveServiceEndpointFromEnv("secrets")
	if err != nil {
		t.Fatalf("ResolveServiceEndpointFromEnv returned error: %v", err)
	}
	if endpoint.Transport != TransportUDS {
		t.Fatalf("expected UDS endpoint, got %s", endpoint.Transport)
	}
	if endpoint.SocketPath != "/tmp/secrets.sock" {
		t.Fatalf("unexpected socket path: %s", endpoint.SocketPath)
	}
}

func TestEndpointContractDoesNotReadEndpointEnvFamily(t *testing.T) {
	t.Setenv("NHX_PLATFORM_ENDPOINT", "unix:///tmp/platform.sock")
	t.Setenv("NHX_SECRETS_ENDPOINT", "unix:///tmp/secrets.sock")

	if _, err := ResolveHelixEndpointFromEnv(); err == nil {
		t.Fatal("expected missing NHX_BASE_URL to fail")
	}
	if _, err := ResolveServiceEndpointFromEnv("secrets"); err == nil {
		t.Fatal("expected missing NHX_SECRETS_URL and NHX_BASE_URL to fail")
	}
}

func TestUDSHTTPClient(t *testing.T) {
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

	receivedPath := make(chan string, 1)
	server := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedPath <- r.URL.Path
			if r.URL.Path != "/status" {
				http.Error(w, "unexpected path", http.StatusNotFound)
				return
			}
			fmt.Fprintln(w, `{"ok":true}`)
		}),
	}
	defer server.Close()
	defer os.Remove(socketPath)
	go func() {
		_ = server.Serve(listener)
	}()

	endpoint, err := ParseEndpoint("unix://" + socketPath)
	if err != nil {
		t.Fatalf("ParseEndpoint returned error: %v", err)
	}
	resp, err := endpoint.HTTPClient().Get(endpoint.ConnectBaseURL + "/status")
	if err != nil {
		t.Fatalf("UDS request failed: %v", err)
	}
	defer resp.Body.Close()
	if path := <-receivedPath; path != "/status" {
		t.Fatalf("unexpected path: %s", path)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("unexpected status: %d", resp.StatusCode)
	}
}
