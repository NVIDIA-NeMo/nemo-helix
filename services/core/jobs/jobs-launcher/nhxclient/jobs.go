// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
)

// stepSpecNameConfigKey is the key the jobs controller adds to a step entity's
// config to record the platform step spec name. It is not part of the step
// spec config the workload reads, so it is stripped before delivery.
const stepSpecNameConfigKey = "_step_spec_name"

type JobStepClient interface {
	// GetJobStepConfig returns the step spec config for a job step as JSON,
	// in the shape the workload reads from NEMO_JOB_STEP_CONFIG_FILE_PATH.
	GetJobStepConfig(workspace, job, step string) ([]byte, error)
}

type jobStepClient struct {
	httpClient *http.Client
	principal  *Principal
	apiBaseURL string
}

func NewJobStepClientWithHTTPClient(apiBaseURL string, principal *Principal, httpClient *http.Client) JobStepClient {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &jobStepClient{
		httpClient: httpClient,
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

func (c *jobStepClient) GetJobStepConfig(workspace, job, step string) ([]byte, error) {
	req, err := http.NewRequest("GET", getJobStepURL(c.apiBaseURL, workspace, job, step), nil)
	if err != nil {
		return nil, err
	}

	// Same principal scheme as the secrets fetch: the jobs service acting on
	// behalf of the job creator.
	if c.principal != nil && c.principal.ID != "" {
		req.Header.Set("X-NHX-Principal-Id", "service:jobs")
		req.Header.Set("X-NHX-Principal-On-Behalf-Of", c.principal.ID)
	}

	resp, err := c.httpClient.Do(req)
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
