// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modifications: Minor edits by NVIDIA.

package server

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	extproc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/mock"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health/grpc_health_v1"

	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/config"
)

// mockExternalProcessor is a mock implementation of the ExternalProcessorServer.
type mockExternalProcessor struct {
	extproc.UnimplementedExternalProcessorServer
	mock.Mock
}

// GenerateSelfSignedCert writes a self-signed ECDSA certificate and key to certPath/keyPath.
// SANs include 127.0.0.1 and localhost. CN is ignored by modern clients; SANs matter.
// Valid for 24h by default.
func GenerateSelfSignedCert(certPath, keyPath string) error {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return err
	}

	serial, err := rand.Int(rand.Reader, big.NewInt(1<<62))
	if err != nil {
		return err
	}

	notBefore := time.Now().Add(-time.Minute)
	notAfter := notBefore.Add(24 * time.Hour)

	tpl := &x509.Certificate{
		SerialNumber: serial,
		Subject: pkix.Name{
			CommonName: "127.0.0.1",
		},
		NotBefore: notBefore,
		NotAfter:  notAfter,

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,

		IPAddresses: []net.IP{net.ParseIP("127.0.0.1")},
		DNSNames:    []string{"localhost"},
	}

	der, err := x509.CreateCertificate(rand.Reader, tpl, tpl, &priv.PublicKey, priv)
	if err != nil {
		return err
	}

	certOut, err := os.Create(certPath)
	if err != nil {
		return err
	}
	defer certOut.Close()
	if err := pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: der}); err != nil {
		return err
	}

	keyBytes, err := x509.MarshalECPrivateKey(priv)
	if err != nil {
		return err
	}
	keyOut, err := os.Create(keyPath)
	if err != nil {
		return err
	}
	defer keyOut.Close()
	if err := pem.Encode(keyOut, &pem.Block{Type: "EC PRIVATE KEY", Bytes: keyBytes}); err != nil {
		return err
	}

	return nil
}

// TestStartSecureGRPCServer tests the start of the gRPC server with TLS.
func TestStartSecureGRPCServer(t *testing.T) {
	t.Parallel()

	cfg, err := config.Load("")
	if err != nil {
		t.Fatalf("Loading config: %v", err)
	}
	cfg.GRPC.Address = "127.0.0.1:0"
	// Build client TLS creds trusting the server cert.
	dir := t.TempDir()
	certPath := filepath.Join(dir, "server.cert")
	keyPath := filepath.Join(dir, "server.key")
	if err := GenerateSelfSignedCert(certPath, keyPath); err != nil {
		t.Fatalf("failed to generate TLS certs: %v", err)
	}
	cfg.GRPC.TLS.Enabled = true
	cfg.GRPC.TLS.CertFile = certPath
	cfg.GRPC.TLS.KeyFile = keyPath

	calloutServer, err := NewCalloutServer(cfg.GRPC)
	if err != nil {
		t.Fatalf("Failed to create CalloutServer: %v", err)
	}
	if calloutServer == nil {
		t.Fatalf("NewCalloutServer() = nil, want non-nil")
	}
	addr, err := calloutServer.StartGRPC(&mockExternalProcessor{})
	if err != nil {
		t.Fatalf("starting grpc server: %v", err)
	}
	t.Cleanup(func() { calloutServer.Stop(context.Background()) })

	// Connect with TLS aware gRPC client.
	caPEM, err := os.ReadFile(certPath)
	if err != nil {
		t.Fatalf("failed to read cert: %v", err)
	}
	pool := x509.NewCertPool()
	pool.AppendCertsFromPEM(caPEM)
	creds := &tls.Config{
		RootCAs:    pool,
		ServerName: "127.0.0.1", // must match SA
		MinVersion: tls.VersionTLS12,
	}
	// Use passthrough to avoid name resolution.
	conn, err := grpc.NewClient("passthrough:///"+addr, grpc.WithTransportCredentials(credentials.NewTLS(creds)))
	if err != nil {
		t.Fatalf("Failed to dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	// Perform a gRPC health check
	hc := grpc_health_v1.NewHealthClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	resp, err := hc.Check(ctx, &grpc_health_v1.HealthCheckRequest{Service: ""})
	if err != nil {
		t.Fatalf("health check failed: %v", err)
	}
	if resp.GetStatus() != grpc_health_v1.HealthCheckResponse_SERVING {
		t.Fatalf("server not server: %v", err)
	}

	// Make HTTP/2 health check
	// Build an HTTP client that uses the same RootCAs and prefers HTTP/2.
	httpClient := &http.Client{
		Timeout: 3 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig: creds,
			// ForceAttemptHTTP2 is default true in modern Go, but we set it explicitly for clarity.
			ForceAttemptHTTP2: true,
		},
	}

	httpResp, err := httpClient.Get("https://" + addr + "/")
	if err != nil {
		t.Fatalf("http2 GET /: %v", err)
	}
	defer httpResp.Body.Close()

	// Confirm HTTP/2 (optional but nice to assert)
	if httpResp.ProtoMajor != 2 {
		t.Fatalf("expected HTTP/2, got %s", httpResp.Proto)
	}
	if httpResp.StatusCode != http.StatusOK {
		t.Fatalf("unexpected status: %v", httpResp.Status)
	}
	body, _ := io.ReadAll(httpResp.Body)
	if !strings.Contains(string(body), "OK") {
		t.Fatalf("unexpected body: %q", string(body))
	}
}

// TestStartInsecureGRPCServer tests the start of the insecure gRPC server.
func TestStartInsecureGRPCServer(t *testing.T) {
	t.Parallel()

	cfg, err := config.Load("")
	if err != nil {
		t.Fatalf("Loading config: %v", err)
	}
	// Configure address to use OS-assigned port.
	cfg.GRPC.Address = "127.0.0.1:0"
	// NOTE that we don't explicitly set cfg.TLS.Enabled = false because this should be the default.

	calloutServer, err := NewCalloutServer(cfg.GRPC)
	if err != nil {
		t.Fatalf("Failed to create CalloutServer: %v", err)
	}
	if calloutServer == nil {
		t.Fatalf("NewCalloutServer() = nil, want non-nil")
	}
	addr, err := calloutServer.StartGRPC(&mockExternalProcessor{})
	if err != nil {
		t.Fatalf("starting grpc server: %v", err)
	}
	t.Cleanup(func() { calloutServer.Stop(context.Background()) })

	// Connect with gRPC client. Use passthrough to avoid name resolution.
	conn, err := grpc.NewClient("passthrough:///"+addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to dial: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	// Perform a gRPC health check
	hc := grpc_health_v1.NewHealthClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	resp, err := hc.Check(ctx, &grpc_health_v1.HealthCheckRequest{Service: ""})
	if err != nil {
		t.Fatalf("health check failed: %v", err)
	}
	if resp.GetStatus() != grpc_health_v1.HealthCheckResponse_SERVING {
		t.Fatalf("server not server: %v", err)
	}
}
