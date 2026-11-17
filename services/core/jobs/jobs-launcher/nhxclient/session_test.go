// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSessionDoesNotFollowCredentialRedirects(t *testing.T) {
	redirected := false
	destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		redirected = true
		t.Error("redirect destination must never be contacted")
	}))
	defer destination.Close()
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer token" {
			t.Error("origin not authenticated")
		}
		http.Redirect(w, r, destination.URL, http.StatusTemporaryRedirect)
	}))
	defer origin.Close()
	source := &WorkloadTokenSource{token: newWorkloadAccessToken("token", 3600, time.Now()), now: time.Now}
	session := &Session{source: source}
	response, err := session.HTTPClient(origin.Client()).Get(origin.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusTemporaryRedirect || redirected {
		t.Fatal("redirect was followed")
	}
}

func TestExchangeDoesNotForwardProofThroughRedirect(t *testing.T) {
	destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("subject proof forwarded to redirect destination")
	}))
	defer destination.Close()
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, destination.URL, http.StatusTemporaryRedirect)
	}))
	defer origin.Close()
	_, err := exchangeWorkloadToken(context.Background(), origin.URL, "client", "proof", authDiscoveryOIDC{}, time.Now())
	if err == nil {
		t.Fatal("redirecting exchange must fail")
	}
}

func TestAuthenticationDestinations(t *testing.T) {
	for _, test := range []struct {
		url     string
		allowed bool
	}{
		{"https://platform.example.test", true},
		{"http://localhost:8080", true},
		{"http://127.0.0.1:8080", true},
		{"http://[::1]:8080", true},
		{"http://platform.example.test", false},
		{"http://localhost.example.test", false},
		{(&url.URL{Scheme: "https", Host: "platform.example.test", User: url.UserPassword("test-user", "test-password")}).String(), false},
		{"file:///tmp/credential", false},
	} {
		t.Run(test.url, func(t *testing.T) {
			if allowed := validateAuthenticationEndpoint(test.url) == nil; allowed != test.allowed {
				t.Fatalf("destination allowed=%v, want %v", allowed, test.allowed)
			}
		})
	}
}

func TestSessionRefusesRemoteCleartextBeforeExchange(t *testing.T) {
	session := &Session{source: &WorkloadTokenSource{}}
	if _, err := session.HTTPClient(nil).Get("http://platform.example.test/secret"); err == nil {
		t.Fatal("cleartext request must fail before token exchange or network I/O")
	}
}

func TestSessionPreservesUnixSocketAuthentication(t *testing.T) {
	directory, err := os.MkdirTemp("/tmp", "nhx-session-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(directory) })
	socket := filepath.Join(directory, "platform.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer token" {
			t.Error("missing UDS bearer")
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	server.Listener = listener
	server.Start()
	defer server.Close()
	endpoint, err := ParseEndpoint("unix://" + socket)
	if err != nil {
		t.Fatal(err)
	}
	session := &Session{source: &WorkloadTokenSource{
		token: newWorkloadAccessToken("token", 3600, time.Now()), now: time.Now,
	}}
	response, err := session.HTTPClient(endpoint.HTTPClient()).Get(endpoint.ConnectBaseURL + "/secret")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusNoContent {
		t.Fatalf("unexpected UDS status %d", response.StatusCode)
	}
}
