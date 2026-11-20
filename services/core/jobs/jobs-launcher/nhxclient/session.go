// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
)

// Session owns credential selection and refresh. HTTP clients and transports stay
// neutral; API clients attach credentials to each request through this session.
type Session struct {
	workloadFile string
	accessFile   string
	mu           sync.Mutex
	workload     *workloadAuthTokenSource
	httpClient   *http.Client
	baseURL      string
}

func NewSessionFromEnv() *Session {
	return NewSessionWithHTTPClient(http.DefaultClient)
}

func NewSessionWithHTTPClient(client *http.Client) *Session {
	return &Session{workloadFile: os.Getenv(workloadIdentityTokenFileEnv), accessFile: os.Getenv("NHX_ACCESS_TOKEN_FILE"), baseURL: strings.TrimRight(os.Getenv(nhxBaseURLEnv), "/"), httpClient: safeHTTPClient(client)}
}

func (s *Session) BearerMode() bool { return s.workloadFile != "" || s.accessFile != "" }

// ValidateSecretPrincipal permits an anonymous creator only when the platform
// explicitly advertises auth-disabled operation. Missing credentials must not
// silently downgrade an authenticated deployment to anonymous Secret access.
func (s *Session) ValidateSecretPrincipal(ctx context.Context, principal *Principal) error {
	if s.BearerMode() || (principal != nil && principal.ID != "") {
		return nil
	}
	endpoint, err := ResolveServiceEndpointFromEnv("auth")
	if err != nil {
		return fmt.Errorf("resolve auth endpoint for anonymous Secret injection: %w", err)
	}
	discovery, err := discoverAuthConfigWithClient(ctx, endpoint.ConnectBaseURL, s.clientForEndpoint(endpoint))
	if err != nil {
		return fmt.Errorf("verify auth-disabled Secret injection: %w", err)
	}
	if discovery.AuthEnabled == nil || *discovery.AuthEnabled {
		return fmt.Errorf("NHX_PRINCIPAL with a creator is required for trusted Secret injection unless auth discovery explicitly disables authentication")
	}
	return nil
}

func safeHTTPClient(client *http.Client) *http.Client {
	if client == nil {
		client = http.DefaultClient
	}
	clone := *client
	clone.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	return &clone
}

func requireSafeBearerURL(raw string, transport http.RoundTripper) error {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" || u.User != nil {
		return fmt.Errorf("invalid credential destination")
	}
	// The synthetic HTTP origin is safe only on a transport bound to a Unix socket.
	if _, ok := transport.(*unixSocketTransport); ok && u.Scheme+"://"+u.Host == UDSBaseURL {
		return nil
	}
	if u.Scheme == "https" {
		return nil
	}
	host := u.Hostname()
	ip := net.ParseIP(host)
	if u.Scheme == "http" && (host == "localhost" || (ip != nil && ip.IsLoopback())) {
		return nil
	}
	return fmt.Errorf("credentials require HTTPS or a loopback endpoint")
}

// Headers is the OTLP adapter's credential source. Legacy log reporting retains
// direct service identity; API calls project the effective creator separately.
func (s *Session) Headers(ctx context.Context) (map[string]string, error) {
	token, err := s.authorization(ctx)
	if err != nil {
		return nil, err
	}
	if token != "" {
		return map[string]string{"Authorization": token}, nil
	}
	return map[string]string{"X-NHX-Principal-Id": "service:jobs"}, nil
}

func (s *Session) authorization(ctx context.Context) (string, error) {
	if s.workloadFile != "" {
		s.mu.Lock()
		if s.workload == nil {
			endpoint, err := ParseEndpoint(s.baseURL)
			if err != nil {
				s.mu.Unlock()
				return "", err
			}
			source, err := newWorkloadAuthTokenSourceForConfig(ctx, s.clientForEndpoint(endpoint), s.httpClient, s.workloadFile, endpoint.ConnectBaseURL)
			if err != nil {
				s.mu.Unlock()
				return "", err
			}
			if source == nil {
				s.mu.Unlock()
				return "", fmt.Errorf("configured workload credential is unavailable")
			}
			source.subjectTokenFile = s.workloadFile
			s.workload = source
		}
		source := s.workload
		s.mu.Unlock()
		return source.AuthorizationHeader(ctx)
	}
	if s.accessFile != "" {
		data, err := os.ReadFile(s.accessFile)
		if err != nil {
			return "", fmt.Errorf("read NHX_ACCESS_TOKEN_FILE: %w", err)
		}
		token := strings.TrimSpace(string(data))
		if token == "" {
			return "", fmt.Errorf("NHX_ACCESS_TOKEN_FILE is empty")
		}
		return "Bearer " + token, nil
	}
	return "", nil
}

func clearIdentityHeaders(h http.Header) {
	for key := range h {
		if strings.HasPrefix(strings.ToLower(key), "x-nhx-principal-") || strings.HasPrefix(strings.ToLower(key), "x-nhx-actor-") || strings.HasPrefix(strings.ToLower(key), "x-nhx-subject-") {
			h.Del(key)
		}
	}
}

func creatorHeaders(p *Principal) http.Header {
	h := http.Header{}
	if p == nil || p.ID == "" {
		return h
	}
	id, email, groups, account, aliases := p.ID, p.Email, p.Groups, p.AccountID, p.AuthzAliases
	if p.OnBehalfOf != "" {
		id, email, groups, account, aliases = p.OnBehalfOf, p.OnBehalfOfEmail, p.OnBehalfOfGroups, p.OnBehalfOfAccountID, p.OnBehalfOfAuthzAliases
	}
	h.Set("X-NHX-Principal-Id", "service:jobs")
	h.Set("X-NHX-Principal-On-Behalf-Of", id)
	if email != "" {
		h.Set("X-NHX-Principal-On-Behalf-Of-Email", email)
	}
	if len(groups) > 0 {
		h.Set("X-NHX-Principal-On-Behalf-Of-Groups", strings.Join(groups, ","))
	}
	if account != "" {
		h.Set("X-NHX-Subject-Account-Id", account)
	}
	if len(aliases) > 0 {
		h.Set("X-NHX-Subject-Aliases", strings.Join(aliases, ","))
	}
	return h
}

func (s *Session) authenticate(req *http.Request, p *Principal, transport http.RoundTripper) (*http.Request, error) {
	clone := req.Clone(req.Context())
	clone.Header = req.Header.Clone()
	if clone.Header == nil {
		clone.Header = make(http.Header)
	}
	if clone.Header.Get("Authorization") != "" || s.BearerMode() {
		if err := requireSafeBearerURL(clone.URL.String(), transport); err != nil {
			return nil, err
		}
		clearIdentityHeaders(clone.Header)
		if clone.Header.Get("Authorization") == "" {
			token, err := s.authorization(req.Context())
			if err != nil {
				return nil, err
			}
			clone.Header.Set("Authorization", token)
		}
	} else {
		for key, values := range creatorHeaders(p) {
			clone.Header[key] = values
		}
	}
	return clone, nil
}

func (s *Session) do(req *http.Request, client *http.Client, p *Principal) (*http.Response, error) {
	client = safeHTTPClient(client)
	authenticated, err := s.authenticate(req, p, client.Transport)
	if err != nil {
		return nil, err
	}
	return client.Do(authenticated)
}

// OTLPClient authenticates only requests to the configured export URL. It borrows
// the base transport and shares this session's provider with API requests.
func (s *Session) OTLPClient(client *http.Client, endpoint string) *http.Client {
	result := safeHTTPClient(client)
	base := result.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	// WithEndpointURL builds requests from the scheme, host and decoded path.
	// Preserve empty paths and trailing slashes exactly as the exporter does.
	target, err := url.Parse(endpoint)
	if err == nil && target.Host != "" && target.User == nil {
		endpoint = (&url.URL{Scheme: target.Scheme, Host: target.Host, Path: target.Path}).String()
	}
	result.Transport = &otlpAuthTransport{session: s, base: base, endpoint: endpoint}
	return result
}

type otlpAuthTransport struct {
	session  *Session
	base     http.RoundTripper
	endpoint string
}

func (t *otlpAuthTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.String() != t.endpoint {
		return nil, fmt.Errorf("unexpected OTLP credential destination")
	}
	cloned, err := t.session.authenticate(req, nil, t.base)
	if err != nil {
		return nil, err
	}
	if !t.session.BearerMode() && cloned.Header.Get("Authorization") == "" {
		cloned.Header.Set("X-NHX-Principal-Id", "service:jobs")
	}
	return t.base.RoundTrip(cloned)
}

func (s *Session) clientForEndpoint(endpoint Endpoint) *http.Client {
	if endpoint.Transport == TransportUDS {
		return safeHTTPClient(endpoint.HTTPClient())
	}
	return safeHTTPClient(s.httpClient)
}
