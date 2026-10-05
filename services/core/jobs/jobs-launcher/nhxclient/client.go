// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

/*
nhxclient is a thin Go client library for interacting with the NeMo Helix API for limited use cases.

It is not intended to be a full-featured SDK, but rather to provide just enough functionality
to support the jobs-launcher service.
*/

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

type HelixSecretAccessResponse struct {
	Name        string `json:"name"`
	WorkspaceID string `json:"workspace_id"`
	Value       string `json:"value"`
}

// Principal holds the authentication context captured from the job creator.
// This is propagated to task containers via the NHX_PRINCIPAL environment variable (JSON).
// The JSON field names match the Principal model (id, email, groups) used by the Python auth layer.
type Principal struct {
	ID     string   `json:"id"`               // Required - the principal's unique identifier
	Email  string   `json:"email,omitempty"`  // Optional - the principal's email
	Groups []string `json:"groups,omitempty"` // Optional - list of groups
}

// PrincipalFromEnv creates a Principal from the NHX_PRINCIPAL environment variable.
// Returns nil if the env var is not set, is invalid JSON, or has no id field.
func PrincipalFromEnv() *Principal {
	jsonStr := os.Getenv("NHX_PRINCIPAL")
	if jsonStr == "" {
		return nil
	}

	var ctx Principal
	if err := json.Unmarshal([]byte(jsonStr), &ctx); err != nil {
		return nil
	}

	if ctx.ID == "" {
		return nil
	}

	return &ctx
}

func getSecretURL(baseURL, workspace, secretName string) string {
	return fmt.Sprintf("%s/apis/secrets/v2/workspaces/%s/secrets/%s/access", baseURL, workspace, secretName)
}

type SecretClient interface {
	GetSecret(workspaceID, secretName string) (*HelixSecretAccessResponse, error)
}

type secretClient struct {
	httpClient *http.Client
	principal  *Principal
	apiBaseURL string
}

func NewSecretClient(apiBaseURL string, principal *Principal) SecretClient {
	return NewSecretClientWithHTTPClient(apiBaseURL, principal, http.DefaultClient)
}

func NewSecretClientWithHTTPClient(apiBaseURL string, principal *Principal, httpClient *http.Client) SecretClient {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &secretClient{
		httpClient: httpClient,
		principal:  principal,
		apiBaseURL: apiBaseURL,
	}
}

func NewSecretClientForEndpoint(endpoint Endpoint, principal *Principal) SecretClient {
	return NewSecretClientWithHTTPClient(endpoint.ConnectBaseURL, principal, endpoint.HTTPClient())
}

func (c *secretClient) GetSecret(workspaceID, secretName string) (*HelixSecretAccessResponse, error) {
	secretURL := getSecretURL(c.apiBaseURL, workspaceID, secretName)

	req, err := http.NewRequest("GET", secretURL, nil)
	if err != nil {
		return nil, err
	}

	// Set auth headers from the job's captured auth context
	if c.principal != nil {
		if c.principal.ID != "" {
			// Fetching the value of the secret requires a service principal, but we want to only access secrets for the principal the job was created by
			req.Header.Set("X-NHX-Principal-Id", "service:jobs")
			req.Header.Set("X-NHX-Principal-On-Behalf-Of", c.principal.ID)
		}
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get secret: status code %d", resp.StatusCode)
	}

	var secretData HelixSecretAccessResponse

	if err := json.NewDecoder(resp.Body).Decode(&secretData); err != nil {
		return nil, fmt.Errorf("failed to decode secret response: %w", err)
	}

	return &HelixSecretAccessResponse{
		Name:        secretName,
		WorkspaceID: workspaceID,
		Value:       secretData.Value,
	}, nil
}
