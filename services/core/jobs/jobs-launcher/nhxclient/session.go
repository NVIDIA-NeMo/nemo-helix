// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
)

// Session owns one workload token cache for all launcher facilities.
// A nil source means explicitly configured legacy identity-header mode.
type Session struct{ source *WorkloadTokenSource }

func NewSession(ctx context.Context) (*Session, error) {
	source, err := NewWorkloadTokenSource(ctx)
	if err != nil {
		return nil, err
	}
	return &Session{source: source}, nil
}

type sessionContextKey struct{}

func WithSession(ctx context.Context, session *Session) context.Context {
	return context.WithValue(ctx, sessionContextKey{}, session)
}

func SessionFromContext(ctx context.Context) (*Session, error) {
	if session, ok := ctx.Value(sessionContextKey{}).(*Session); ok {
		return session, nil
	}
	return NewSession(ctx)
}

func (s *Session) WorkloadIdentity() bool { return s.source != nil }

// Headers supplies platform log authentication without reading user OTEL headers.
func (s *Session) Headers(ctx context.Context) (map[string]string, error) {
	if s.source != nil {
		return s.source.Headers(ctx)
	}
	return map[string]string{"X-NHX-Principal-Id": "service:jobs"}, nil
}

// HTTPClient preserves service routing/timeouts and owns bearer attachment.
// Redirects are never followed: an authenticated transport would otherwise
// reattach credentials even after net/http stripped sensitive redirect headers.
func (s *Session) HTTPClient(base *http.Client) *http.Client {
	if base == nil {
		base = http.DefaultClient
	}
	client := *base
	client.CheckRedirect = func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}
	if s.source != nil {
		client.Transport = &sessionTransport{source: s.source, base: base.Transport}
	}
	return &client
}

type sessionTransport struct {
	source *WorkloadTokenSource
	base   http.RoundTripper
}

func (t *sessionTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	_, unixSocket := t.base.(*unixSocketTransport)
	if !(unixSocket && request.URL.Scheme == "http" && request.URL.Host == "nemo-helix.local") {
		if err := requireSecureAuthenticationURL(request.URL); err != nil {
			return nil, err
		}
	}
	header, err := t.source.AuthorizationHeader(request.Context())
	if err != nil {
		return nil, fmt.Errorf("authenticate workload request: %w", err)
	}
	cloned := request.Clone(request.Context())
	cloned.Header = request.Header.Clone()
	for name := range cloned.Header {
		if strings.HasPrefix(strings.ToLower(name), "x-nhx-principal-") ||
			strings.HasPrefix(strings.ToLower(name), "x-nhx-actor-") ||
			strings.HasPrefix(strings.ToLower(name), "x-nhx-subject-") {
			cloned.Header.Del(name)
		}
	}
	cloned.Header.Set("Authorization", header)
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(cloned)
}

// Protect both the proof exchange and API bearer token in transit.
func requireSecureAuthenticationURL(destination *url.URL) error {
	if destination.User != nil || destination.Host == "" {
		return fmt.Errorf("authentication endpoint must have a host and no URL credentials")
	}
	if destination.Scheme == "https" {
		return nil
	}
	host := destination.Hostname()
	address := net.ParseIP(host)
	if destination.Scheme == "http" && (host == "localhost" || (address != nil && address.IsLoopback())) {
		return nil
	}
	return fmt.Errorf("authentication requires HTTPS or loopback HTTP")
}

func validateAuthenticationEndpoint(raw string) error {
	destination, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("invalid authentication endpoint")
	}
	return requireSecureAuthenticationURL(destination)
}
