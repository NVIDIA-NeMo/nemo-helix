// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

// stepSpecNameConfigKey is the key the jobs controller adds to a step entity's
// config to record the platform step spec name. It is not part of the step
// spec config the workload reads, so it is stripped before delivery.
const stepSpecNameConfigKey = "_step_spec_name"

type JobStepClient interface {
	// GetJobStepConfig returns the step spec config for a job step as JSON,
	// in the shape the workload reads from NEMO_JOB_STEP_CONFIG_FILE_PATH.
	GetJobStepConfig(ctx context.Context, workspace, job, step string) ([]byte, error)
	GetAttemptStepConfig(ctx context.Context, workspace, job, attempt, step string) ([]byte, error)
}

type jobStepClient struct {
	httpClient *http.Client
	principal  *Principal
	apiBaseURL string
	session    *Session
}

func NewJobStepClientWithHTTPClient(apiBaseURL string, principal *Principal, httpClient *http.Client) JobStepClient {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &jobStepClient{
		httpClient: safeHTTPClient(httpClient),
		session:    &Session{},
		principal:  principal,
		apiBaseURL: apiBaseURL,
	}
}

func getJobStepURL(baseURL, workspace, job, step string) string {
	return fmt.Sprintf(
		"%s/apis/jobs/v2/workspaces/%s/jobs/%s/steps/%s",
		baseURL, url.PathEscape(workspace), url.PathEscape(job), url.PathEscape(step),
	)
}

func (c *jobStepClient) GetJobStepConfig(ctx context.Context, workspace, job, step string) ([]byte, error) {
	return c.getConfig(ctx, getJobStepURL(c.apiBaseURL, workspace, job, step))
}

func (c *jobStepClient) GetAttemptStepConfig(ctx context.Context, workspace, job, attempt, step string) ([]byte, error) {
	resource := fmt.Sprintf("%s/apis/jobs/v2/workspaces/%s/jobs/%s/attempts/%s/steps/%s", c.apiBaseURL, url.PathEscape(workspace), url.PathEscape(job), url.PathEscape(attempt), url.PathEscape(step))
	return c.getConfig(ctx, resource)
}

func (c *jobStepClient) getConfig(ctx context.Context, resource string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", resource, nil)
	if err != nil {
		return nil, err
	}

	resp, err := c.session.do(req, c.httpClient, c.principal)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get job step: status code %d", resp.StatusCode)
	}

	var body struct {
		Config map[string]json.RawMessage `json:"config"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("failed to decode job step response: %w", err)
	}

	config := body.Config
	if config == nil {
		config = map[string]json.RawMessage{}
	}
	delete(config, stepSpecNameConfigKey)
	return json.Marshal(config)
}

func (s *Session) JobStepClient(endpoint Endpoint, principal *Principal) JobStepClient {
	client := s.clientForEndpoint(endpoint)
	if client.Timeout == 0 {
		client.Timeout = 30 * time.Second
	}
	c := NewJobStepClientWithHTTPClient(endpoint.ConnectBaseURL, principal, client).(*jobStepClient)
	c.session = s
	return c
}
