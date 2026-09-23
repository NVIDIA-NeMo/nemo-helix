// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	nhxBaseURLEnv                      = "NHX_BASE_URL"
	workloadIdentityTokenFileEnv       = "NHX_WORKLOAD_IDENTITY_TOKEN_FILE"
	tokenExchangeGrantType             = "urn:ietf:params:oauth:grant-type:token-exchange"
	jwtTokenType                       = "urn:ietf:params:oauth:token-type:jwt"
	accessTokenType                    = "urn:ietf:params:oauth:token-type:access_token"
	dockerOpaqueWorkloadProofTokenType = "urn:nvidia:nemo:params:oauth:token-type:docker-opaque-workload-proof"
	dockerOpaqueWorkloadProofPrefix    = "nhx_obo_v1."
	workloadAuthRequestTimeoutSeconds  = 30
	maxAuthResponseBodyBytes           = 64 * 1024
	workloadAuthRefreshMarginFraction  = 5
	workloadAuthMaxRefreshMargin       = time.Minute
)

type authDiscoveryResponse struct {
	OIDC authDiscoveryOIDC `json:"oidc"`
}

type authDiscoveryOIDC struct {
	ClientID                     string `json:"client_id"`
	TokenEndpoint                string `json:"token_endpoint"`
	WorkloadTokenExchangeEnabled bool   `json:"workload_token_exchange_enabled"`
	WorkloadClientID             string `json:"workload_client_id"`
	WorkloadTokenEndpoint        string `json:"workload_token_endpoint"`
	WorkloadAudience             string `json:"workload_audience"`
	WorkloadScope                string `json:"workload_scope"`
}

type tokenExchangeResponse struct {
	AccessToken string `json:"access_token"`
	ExpiresIn   int64  `json:"expires_in"`
}

type workloadAccessToken struct {
	value     string
	refreshAt time.Time
}

func newWorkloadAccessToken(value string, expiresIn int64, issuedAt time.Time) workloadAccessToken {
	lifetime := time.Duration(expiresIn) * time.Second
	refreshMargin := lifetime / workloadAuthRefreshMarginFraction
	if refreshMargin > workloadAuthMaxRefreshMargin {
		refreshMargin = workloadAuthMaxRefreshMargin
	}
	return workloadAccessToken{
		value:     value,
		refreshAt: issuedAt.Add(lifetime - refreshMargin),
	}
}

func (t workloadAccessToken) needsRefresh(now time.Time) bool {
	return t.value == "" || !now.Before(t.refreshAt)
}

type workloadAuthTokenSource struct {
	subjectTokenFile string
	tokenEndpoint    string
	clientID         string
	discovery        authDiscoveryOIDC

	mu    sync.Mutex
	token workloadAccessToken
	now   func() time.Time
}

func getOTLPLogWorkloadAuthHeaders(ctx context.Context) (map[string]string, error) {
	tokenSource, err := newOTLPLogWorkloadAuthTokenSource(ctx)
	if err != nil || tokenSource == nil {
		return nil, err
	}
	return tokenSource.authHeaders(ctx)
}

func newOTLPLogWorkloadAuthTokenSource(ctx context.Context) (*workloadAuthTokenSource, error) {
	subjectTokenFile := os.Getenv(workloadIdentityTokenFileEnv)
	if subjectTokenFile == "" {
		return nil, nil
	}

	baseURL := strings.TrimRight(os.Getenv(nhxBaseURLEnv), "/")
	if baseURL == "" {
		return nil, fmt.Errorf("%s is required when %s is set", nhxBaseURLEnv, workloadIdentityTokenFileEnv)
	}

	ctx, cancel := context.WithTimeout(ctx, workloadAuthRequestTimeoutSeconds*time.Second)
	defer cancel()

	discovery, err := discoverAuthConfig(ctx, baseURL)
	if err != nil {
		return nil, err
	}
	if !discovery.OIDC.WorkloadTokenExchangeEnabled {
		return nil, fmt.Errorf("workload token exchange is not enabled by auth discovery")
	}

	tokenEndpoint := discovery.OIDC.WorkloadTokenEndpoint
	if tokenEndpoint == "" {
		tokenEndpoint = discovery.OIDC.TokenEndpoint
	}
	if tokenEndpoint == "" {
		return nil, fmt.Errorf("auth discovery did not return workload_token_endpoint or token_endpoint")
	}

	clientID := discovery.OIDC.WorkloadClientID
	if clientID == "" {
		clientID = discovery.OIDC.ClientID
	}
	if clientID == "" {
		return nil, fmt.Errorf("auth discovery did not return workload_client_id or client_id")
	}

	return &workloadAuthTokenSource{
		subjectTokenFile: subjectTokenFile,
		tokenEndpoint:    tokenEndpoint,
		clientID:         clientID,
		discovery:        discovery.OIDC,
		now:              time.Now,
	}, nil
}

func (s *workloadAuthTokenSource) authHeaders(ctx context.Context) (map[string]string, error) {
	authHeader, err := s.AuthorizationHeader(ctx)
	if err != nil {
		return nil, err
	}
	return map[string]string{"Authorization": authHeader}, nil
}

func (s *workloadAuthTokenSource) Headers(ctx context.Context) (map[string]string, error) {
	return s.authHeaders(ctx)
}

func (s *workloadAuthTokenSource) AuthorizationHeader(ctx context.Context) (string, error) {
	accessToken, err := s.accessToken(ctx)
	if err != nil {
		return "", err
	}
	return "Bearer " + accessToken, nil
}

func (s *workloadAuthTokenSource) accessToken(ctx context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.token.needsRefresh(s.now()) {
		return s.token.value, nil
	}

	ctx, cancel := context.WithTimeout(ctx, workloadAuthRequestTimeoutSeconds*time.Second)
	defer cancel()

	subjectToken, err := readSubjectToken(s.subjectTokenFile)
	if err != nil {
		return "", err
	}

	accessToken, err := exchangeWorkloadToken(ctx, s.tokenEndpoint, s.clientID, subjectToken, s.discovery, s.now())
	if err != nil {
		return "", err
	}

	s.token = accessToken
	return s.token.value, nil
}

func discoverAuthConfig(ctx context.Context, baseURL string) (*authDiscoveryResponse, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/apis/auth/discovery", nil)
	if err != nil {
		return nil, err
	}

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, maxAuthResponseBodyBytes))
		return nil, fmt.Errorf("auth discovery failed: status %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}

	var discovery authDiscoveryResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, maxAuthResponseBodyBytes)).Decode(&discovery); err != nil {
		return nil, fmt.Errorf("decode auth discovery response: %w", err)
	}
	return &discovery, nil
}

func readSubjectToken(path string) (string, error) {
	tokenBytes, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %s at %s: %w", workloadIdentityTokenFileEnv, path, err)
	}
	token := strings.TrimSpace(string(tokenBytes))
	if token == "" {
		return "", fmt.Errorf("%s at %s is empty", workloadIdentityTokenFileEnv, path)
	}
	return token, nil
}

func subjectTokenTypeForExchange(subjectToken string) string {
	if strings.HasPrefix(subjectToken, dockerOpaqueWorkloadProofPrefix) {
		return dockerOpaqueWorkloadProofTokenType
	}
	return jwtTokenType
}

func exchangeWorkloadToken(
	ctx context.Context,
	tokenEndpoint string,
	clientID string,
	subjectToken string,
	discovery authDiscoveryOIDC,
	issuedAt time.Time,
) (workloadAccessToken, error) {
	form := url.Values{}
	form.Set("grant_type", tokenExchangeGrantType)
	form.Set("client_id", clientID)
	form.Set("subject_token", subjectToken)
	form.Set("subject_token_type", subjectTokenTypeForExchange(subjectToken))
	form.Set("requested_token_type", accessTokenType)
	if discovery.WorkloadAudience != "" {
		form.Set("audience", discovery.WorkloadAudience)
	}
	if discovery.WorkloadScope != "" {
		form.Set("scope", discovery.WorkloadScope)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return workloadAccessToken{}, err
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return workloadAccessToken{}, err
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(response.Body)
		return workloadAccessToken{}, fmt.Errorf("workload token exchange failed: status %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}

	var tokenResponse tokenExchangeResponse
	if err := json.NewDecoder(response.Body).Decode(&tokenResponse); err != nil {
		return workloadAccessToken{}, fmt.Errorf("decode workload token exchange response: %w", err)
	}
	if strings.TrimSpace(tokenResponse.AccessToken) == "" {
		return workloadAccessToken{}, fmt.Errorf("workload token exchange response did not include a non-empty access_token")
	}
	if tokenResponse.ExpiresIn <= 0 {
		return workloadAccessToken{}, fmt.Errorf("workload token exchange response did not include a positive expires_in")
	}
	return newWorkloadAccessToken(tokenResponse.AccessToken, tokenResponse.ExpiresIn, issuedAt), nil
}
