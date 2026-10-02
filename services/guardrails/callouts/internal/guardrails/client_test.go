// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package guardrails

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"

	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/config"
)

func TestGuardrailsClient(t *testing.T) {
	// Define some shared variables
	modelName := "fake-model"
	inputRailKey := "content safety check input $model=content_safety"
	outputRailKey := "content safety check output $model=content_safety"
	cfg, err := config.Load("")
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}
	cfg.Guardrails.Models = map[string]config.GuardrailsModelConfig{
		modelName: {
			RefusalText: "custom refusal",
			ConfigIDs:   []string{"config1", "config2"},
		},
	}

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify that the request method and path are correct
		if r.Method != "POST" {
			t.Errorf("Expected POST request, got %s", r.Method)
		}
		if r.URL.Path != "/v1/guardrail/checks" {
			t.Errorf("Expected path /v1/guardrail/checks, got %s", r.URL.Path)
		}

		// Verify the request body
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("Failed to read request body: %v", err)
		}
		// Verify guardrails config IDs got propagated
		var req CheckRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("Failed to unmarshal request body: %v", err)
		}
		for i, configID := range req.Guardrails.ConfigIDs {
			if configID != cfg.Guardrails.Models[req.Model].ConfigIDs[i] {
				t.Errorf("got guardrails configID %s, want %s", configID, cfg.Guardrails.Models[req.Model].ConfigIDs[i])
			}
		}

		// Send the response
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		resp := CheckResponse{
			Status:         StatusBlocked,
			GuardrailsData: nil,
			RailsStatus: map[string]struct {
				Status Status "json:\"status\""
			}{
				inputRailKey:  {Status: StatusSuccess},
				outputRailKey: {Status: StatusBlocked},
			},
		}
		data, err := json.Marshal(resp)
		if err != nil {
			t.Fatalf("Failed to marshal response body: %v", err)
		}
		w.Write(data)
	}))
	defer mockServer.Close()
	cfg.Guardrails.BaseURL = mockServer.URL

	// Make request against the mock server
	client, err := NewClient(&cfg.Guardrails)
	if err != nil {
		t.Fatalf("Failed to initialize client: %v", err)
	}
	req := CheckRequest{
		ChatCompletionNewParams: openai.ChatCompletionNewParams{
			Model: modelName,
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hi there."),
				openai.AssistantMessage("You are stupid."),
			},
			TopP: param.NewOpt(0.1),
		},
		Guardrails: GuardrailsConfig{ConfigIDs: []string{"This should get overriden"}},
	}
	resp, err := client.Check(context.Background(), req)
	if err != nil {
		t.Fatalf("Guardrails check failed: %v", err)
	}
	if resp.Status != StatusBlocked {
		t.Errorf("Overall Status %s != %s", resp.Status, StatusBlocked)
	}
	inputRailStatus := resp.RailsStatus[inputRailKey].Status
	if inputRailStatus != StatusSuccess {
		t.Errorf("Input rail status %s != %s", inputRailStatus, StatusSuccess)
	}
	outputRailStatus := resp.RailsStatus[outputRailKey].Status
	if outputRailStatus != StatusBlocked {
		t.Errorf("Output rail status %s != %s", outputRailStatus, StatusBlocked)
	}
	if resp.RefusalText != cfg.Guardrails.Models[modelName].RefusalText {
		t.Errorf("Got refusal text %s, want %s", resp.RefusalText, cfg.Guardrails.Models[modelName].RefusalText)
	}
}

func ignoreUnexportedType(pkg, typeName string) cmp.Option {
	return cmp.FilterPath(func(p cmp.Path) bool {
		sf, ok := p.Last().(cmp.StructField)
		if !ok {
			return false
		}
		t := sf.Type()
		return t.PkgPath() == pkg && t.Name() == typeName
	}, cmp.Ignore())
}

func TestCheckRequest__MarshalRoundTrip(t *testing.T) {
	in := CheckRequest{
		ChatCompletionNewParams: openai.ChatCompletionNewParams{
			Model: "meta/llama-3.3-70b-instruct",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hi there."),
				openai.AssistantMessage("You are stupid."),
			},
			TopP: param.NewOpt(0.1),
		},
		Guardrails: GuardrailsConfig{
			ConfigIDs: []string{"default/nemoguard"},
		},
		Stream: true,
	}
	// For some reason the constructors don't actually set the value of the roles. It delays it until serialization time.
	// This breaks comparison below when we re-construct the CheckRequest via Unmarshal, which already has the Role filled.
	// For now, lets just manually set the value. This could be a bug in the OpenAI library.
	ofUser := in.ChatCompletionNewParams.Messages[0].OfUser
	ofUser.Role = ofUser.Role.Default()
	ofAssistant := in.ChatCompletionNewParams.Messages[1].OfAssistant
	ofAssistant.Role = ofAssistant.Role.Default()
	b, err := json.MarshalIndent(in, "", "  ")
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}
	var out CheckRequest
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	// OpenAI structs fave a bunch of fields that are private, which causes cmp to panic
	// need to explicitly ignore those.
	opts := []cmp.Option{
		cmpopts.EquateEmpty(),
		ignoreUnexportedType("github.com/openai/openai-go/v2/packages/param", "APIUnion"),
		ignoreUnexportedType("github.com/openai/openai-go/v2/packages/param", "APIObject"),
		cmpopts.IgnoreUnexported(
			param.Opt[string]{},
			param.Opt[float64]{},
			param.Opt[int64]{},
			param.Opt[bool]{},
		),
	}
	want := in
	got := out
	if diff := cmp.Diff(want, got, opts...); diff != "" {
		t.Errorf("CheckRequest marshal-unmarshal mismatch (-want +got):\n%s", diff)
	}
}

func TestCheckRequest__OmitGuardrailsIfZero(t *testing.T) {
	checkReq := CheckRequest{
		ChatCompletionNewParams: openai.ChatCompletionNewParams{
			Model: "fake-model",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("Hi there."),
				openai.AssistantMessage("You are stupid."),
			},
			Temperature: param.NewOpt(0.0),
			TopP:        param.NewOpt(0.1),
		},
		Guardrails: GuardrailsConfig{
			ConfigIDs: nil,
		},
	}
	gotBytes, err := json.MarshalIndent(checkReq, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal CheckRequest: %v", err)
	}
	got := string(gotBytes)
	want := `{
  "guardrails": {},
  "messages": [
    {
      "content": "Hi there.",
      "role": "user"
    },
    {
      "content": "You are stupid.",
      "role": "assistant"
    }
  ],
  "model": "fake-model",
  "temperature": 0,
  "top_p": 0.1
}`
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("JSON is not equal: (-want +got)\n%s", diff)
	}
}

func TestCheckRequest__Unmarshal(t *testing.T) {
	// Test to see if we can unmarsal a regular chat completion request without guardrails field.
	raw := `
	{
		"messages": [
			{
				"content": "Hi there.",
				"role": "user"
			},
			{
				"content": "You are stupid.",
				"role": "assistant"
			}
		],
		"model": "meta/llama-3.3-70b-instruct",
		"top_p": 0.1
	}`
	var cr CheckRequest
	if err := json.Unmarshal([]byte(raw), &cr); err != nil {
		t.Fatalf("failed to unmarshal into CheckRequest: %v", err)
	}
}
