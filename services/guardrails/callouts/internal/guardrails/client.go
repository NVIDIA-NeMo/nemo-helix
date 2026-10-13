// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package guardrails

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"maps"
	"net/http"
	"time"

	"github.com/openai/openai-go/v2"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"

	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/config"
)

type Status int

const (
	StatusUnknown Status = iota // 0
	StatusSuccess               // 1
	StatusBlocked               // 2
)

func (s Status) String() string {
	switch s {
	case StatusSuccess:
		return "success"
	case StatusBlocked:
		return "blocked"
	default:
		return "unknown"
	}
}

func (s Status) IsValid() bool {
	switch s {
	case StatusSuccess, StatusBlocked:
		return true
	default:
		return false
	}
}

func (s Status) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.String())
}
func (s *Status) UnmarshalJSON(data []byte) error {
	var str string
	if err := json.Unmarshal(data, &str); err != nil {
		return err
	}
	switch str {
	case "success":
		*s = StatusSuccess
	case "blocked":
		*s = StatusBlocked
	default:
		*s = StatusUnknown
	}
	return nil
}

// Client defines the Guardrails API
type Client interface {
	Check(ctx context.Context, req CheckRequest) (*CheckResponse, error)
}

// Data models

type GuardrailsConfig struct {
	// ConfigID  string   `json:"config_id"`
	ConfigIDs []string `json:"config_ids,omitzero,omitempty"`
}

type CheckRequest struct {
	// embed OpenAI ChatCompletion struct to get the relevant fields automatically
	openai.ChatCompletionNewParams
	Guardrails GuardrailsConfig `json:"guardrails,omitzero,omitempty"`
	// Expose Stream for serialization because ChatCompletionNewParams doesn't have it.
	// Instead, OpenAI's Go client does some clever dynamic JSON mutation when making a stream request.
	Stream bool `json:"stream,omitzero,omitempty"`
}

type CheckResponse struct {
	Status      Status `json:"status"`
	RailsStatus map[string]struct {
		Status Status `json:"status"`
	} `json:"rails_status"`
	GuardrailsData any `json:"guardrails_data"`

	// External Processor specific fields.
	// RefusalText is based on the CheckRequest's model's refusal text.
	RefusalText string `json:"-"`
}

// MarshalJSON solves the issue with the embedded openai.ChatCompletionNewParams's MarshalJSON taking over the json marshaler,
// and dropping our guardrails field.
// First, marshal the embedded struct using its own custom  MarshalJSON, then merge it with our guardrails JSON.
func (c CheckRequest) MarshalJSON() ([]byte, error) {
	// Avoid infinite recursion
	type Alias CheckRequest

	// Marshal the embedded struct separately with its custom MarshalJSON
	embeddedJSON, err := json.Marshal(c.ChatCompletionNewParams)
	if err != nil {
		return nil, err
	}

	// Marshal guardrails
	guardrailsJSON, err := json.Marshal(map[string]any{"guardrails": c.Guardrails})
	if err != nil {
		return nil, err
	}

	// Merge the JSON objects
	var combined map[string]any
	var guardrails map[string]any

	if err := json.Unmarshal(embeddedJSON, &combined); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(guardrailsJSON, &guardrails); err != nil {
		return nil, err
	}

	// Merge the two maps
	if guardrails["guardrails"] != nil {
		maps.Copy(combined, guardrails)
	}
	// Add stream
	if c.Stream {
		maps.Copy(combined, map[string]any{"stream": true})
	}

	return json.Marshal(combined)
}

// UnmarshalJSON complements MarshalJSON by first unmarshaling the entire
// JSON blob into a generic map, extracting the "guardrails" field into c.Guardrails first,
// then removing the guardrails portion, and feeding just the portion mapping to the embedded
// ChatCompletionNewParams (so its custom UnmarshalJSON runs).
//
// This function gracefully handles absent or null "guardrails".
func (c *CheckRequest) UnmarshalJSON(data []byte) error {
	// Avoid infinite recursion
	type Alias *CheckRequest

	// Decode to a generic map for splitting fields
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	// Extract and remove "guardrails" from the map (if present),
	// to avoid passing it into the embedded type if it doesn't expect it.
	if v, ok := raw["guardrails"]; ok {
		var gr GuardrailsConfig
		if err := json.Unmarshal(v, &gr); err != nil {
			return err
		}
		c.Guardrails = gr
		delete(raw, "guardrails")
	}
	// Extract and remove "stream" from the map (if present)
	if v, ok := raw["stream"]; ok {
		var stream bool
		if err := json.Unmarshal(v, &stream); err != nil {
			return err
		}
		c.Stream = stream
		delete(raw, "stream")
	}

	// Re-encode the remainder for the embedded type to unmarshal with its
	// own custom logic. This ensures we honor its custom UnmarshalJSON.
	rest, err := json.Marshal(raw)
	if err != nil {
		return err
	}

	// Unmarshal into the embedded field
	var embedded openai.ChatCompletionNewParams
	if err := json.Unmarshal(rest, &embedded); err != nil {
		return err
	}
	c.ChatCompletionNewParams = embedded
	return nil
}

type client struct {
	baseURL    string
	httpClient *http.Client
	config     *config.Guardrails
}

func NewClient(cfg *config.Guardrails) (Client, error) {
	to := cfg.Timeout
	if to <= 0 {
		to = 5 * time.Second
	}
	return &client{
		baseURL: cfg.BaseURL,
		httpClient: &http.Client{
			Timeout:   to,
			Transport: otelhttp.NewTransport(http.DefaultTransport),
		},
		config: cfg,
	}, nil
}

// Check sends a POST v1/guardrails/checks to Guardrails MS.
// Additionally, it does some useful things for extproc:
// - inject the guardrails configs into the request based on the input's model.
// - adds a refusal text to the CheckResponse, which is not available in the official Guardrails MS check response.
func (c *client) Check(ctx context.Context, input CheckRequest) (*CheckResponse, error) {
	streamID := GetStreamID(ctx)
	checkResp := &CheckResponse{Status: StatusBlocked, RefusalText: c.getRefusalTextForModel(input.Model)}
	if c == nil {
		return checkResp, fmt.Errorf("missing Guardrails Client")
	}

	url := fmt.Sprintf("%s/v1/guardrail/checks", c.baseURL)

	// We expose a mapping from models to guardrails configIDs to the user, which is configured at deploy-time.
	// Guardrails client is reponsible for mapping the user's model to the guardrails configIDs.
	input.Guardrails.ConfigIDs = c.config.Models[input.Model].GetGuardrailsConfigIDs()

	reqBody, err := json.Marshal(input)
	if err != nil {
		return checkResp, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return checkResp, err
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	// Note that Guardrails also accepts X-Model-Authorization for setting LLM API Key

	// TODO implement retries
	resp, err := c.httpClient.Do(req)
	if err != nil {
		// TODO handle network-level errors like timeouts.
		log.Printf("[%s] Failed to send Check request to Guardrails: %v", streamID, err)
		return checkResp, err
	}
	defer resp.Body.Close()
	// TODO handle error responses by actually parsing the resp body.
	// Attempt to decode an RFC 7807 Problem details object (application/problem+json in header).
	// Otherwise, return an HTTPError with status code and a truncated body.
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return checkResp, fmt.Errorf("HTTP error: %v", resp)
	}

	if err := json.NewDecoder(resp.Body).Decode(checkResp); err != nil {
		checkResp.Status = StatusBlocked
		return checkResp, err
	}

	return checkResp, nil
}

func (c *client) getRefusalTextForModel(model string) string {
	if c == nil {
		return config.DEFAULT_REFUSAL_TEXT
	}
	if c.config.Models[model].RefusalText == "" {
		return c.config.DefaultRefusalText
	}
	return c.config.Models[model].RefusalText
}

// Compile time type check
var _ Client = (*client)(nil)

// Helpers

// streamIDKey is a unique and unexported type for setting and getting streamID in contexts.
type streamIDKey struct{}

func SetStreamID(ctx context.Context, streamID string) context.Context {
	return context.WithValue(ctx, streamIDKey{}, streamID)
}

func GetStreamID(ctx context.Context) string {
	if v, ok := ctx.Value(streamIDKey{}).(string); ok {
		return v
	}
	return ""
}
