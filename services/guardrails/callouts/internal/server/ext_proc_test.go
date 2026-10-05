// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"testing"
	"time"

	envoycore "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extproc_filters "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	extproc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoytype "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/testing/protocmp"

	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/config"
	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/guardrails"
	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/streams"
)

// Global guardrails models map for establishing source of truth across tests.
const modelName = "fake-model"

var modelConfigs = map[string]config.GuardrailsModelConfig{
	modelName: {
		RefusalText: "custom refusal text",
		ConfigIDs:   []string{"content-safety", "jailbreak"},
	},
}

type checkFn func(context.Context, guardrails.CheckRequest) (*guardrails.CheckResponse, error)

type stubGuardrailsClient struct {
	guardrails.Client
	check checkFn
}

func (f stubGuardrailsClient) Check(ctx context.Context, req guardrails.CheckRequest) (*guardrails.CheckResponse, error) {
	if f.check != nil {
		return f.check(ctx, req)
	}
	// Default response if the test didn't inject a custom Check handler
	return &guardrails.CheckResponse{
		Status: guardrails.StatusSuccess,
		RailsStatus: map[string]struct {
			Status guardrails.Status `json:"status"`
		}{
			"input": {Status: guardrails.StatusSuccess},
		},
	}, nil
}

// bufSize defines the internal buffer size for the in-memory listener.
// 1 MiB is a common default; adjust for large payload tests.
const bufSize = 1024 * 1024

func newTestClient(t *testing.T, gr guardrails.Client) extproc.ExternalProcessorClient {
	t.Helper()

	// Create an in-memory listener
	lis := bufconn.Listen(bufSize)

	// Buil a gRPC server and register our extproc service.
	cfg, err := config.Load("")
	if err != nil {
		t.Fatalf("failed to load config: %v", err)
	}
	// Assume we split phrases into words, and we check every word
	cfg.ExtProc.EventsPerCheck = 1
	cfg.Guardrails.Models[modelName] = modelConfigs[modelName]
	grpcServer := grpc.NewServer()
	svc := NewExternalProcessor(cfg, gr)
	extproc.RegisterExternalProcessorServer(grpcServer, svc)

	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			t.Logf("grpc server exited: %v", err)
		}
	}()

	// Create a client connection to the server.
	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return lis.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("new client (bufnet): %v", err)
	}

	t.Cleanup(func() {
		_ = conn.Close()
		grpcServer.GracefulStop()
	})

	return extproc.NewExternalProcessorClient(conn)
}

func TestExternalProcessor_Simple(t *testing.T) {
	tests := map[string]struct {
		requests []*extproc.ProcessingRequest
		expected []*extproc.ProcessingResponse
	}{
		"EmptyData": {
			requests: []*extproc.ProcessingRequest{{}},
			expected: []*extproc.ProcessingResponse{{}},
		},
		"RequestHeaders": {
			requests: []*extproc.ProcessingRequest{
				{
					Request: &extproc.ProcessingRequest_RequestHeaders{
						RequestHeaders: &extproc.HttpHeaders{
							Headers: &envoycore.HeaderMap{
								Headers: []*envoycore.HeaderValue{
									{Key: "x-foo", RawValue: []byte("bar")},
								},
							},
						},
					},
				},
			},
			expected: []*extproc.ProcessingResponse{
				{
					Response: &extproc.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extproc.HeadersResponse{
							Response: &extproc.CommonResponse{},
						},
					},
				},
			},
		},
		"RequestTrailers": {
			requests: []*extproc.ProcessingRequest{
				{
					Request: &extproc.ProcessingRequest_RequestTrailers{
						RequestTrailers: &extproc.HttpTrailers{},
					},
				},
			},
			expected: []*extproc.ProcessingResponse{
				{
					Response: &extproc.ProcessingResponse_RequestTrailers{
						RequestTrailers: &extproc.TrailersResponse{},
					},
				},
			},
		},
		"ResponseHeaders": {
			requests: []*extproc.ProcessingRequest{
				{
					Request: &extproc.ProcessingRequest_ResponseHeaders{
						ResponseHeaders: &extproc.HttpHeaders{
							Headers: &envoycore.HeaderMap{
								Headers: []*envoycore.HeaderValue{
									{Key: "x-foo", RawValue: []byte("bar")},
								},
							},
						},
					},
				},
			},
			expected: []*extproc.ProcessingResponse{
				{
					Response: &extproc.ProcessingResponse_ResponseHeaders{
						ResponseHeaders: &extproc.HeadersResponse{
							Response: &extproc.CommonResponse{
								HeaderMutation: &extproc.HeaderMutation{
									SetHeaders: []*envoycore.HeaderValueOption{
										{
											Header: &envoycore.HeaderValue{
												Key: STREAM_ID_HEADER,
											},
										},
									},
								},
							},
						},
					},
					ModeOverride: &extproc_filters.ProcessingMode{
						ResponseBodyMode: extproc_filters.ProcessingMode_FULL_DUPLEX_STREAMED,
					},
				},
			},
		},
		"ResponseTrailers": {
			requests: []*extproc.ProcessingRequest{
				{
					Request: &extproc.ProcessingRequest_ResponseTrailers{
						ResponseTrailers: &extproc.HttpTrailers{},
					},
				},
			},
			expected: []*extproc.ProcessingResponse{
				{
					Response: &extproc.ProcessingResponse_ResponseTrailers{
						ResponseTrailers: &extproc.TrailersResponse{},
					},
				},
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			client := newTestClient(t, stubGuardrailsClient{})
			responses := makeRequests(t, client, tc.requests)
			if len(responses) != len(tc.expected) {
				t.Fatalf("Expected %d responses, got %d", len(tc.expected), len(responses))
			}

			opts := cmp.Options{
				protocmp.Transform(),
				protocmp.IgnoreFields(&envoycore.HeaderValue{}, "raw_value"),
			}

			for i, resp := range responses {
				if diff := cmp.Diff(tc.expected[i], resp, opts...); diff != "" {
					t.Errorf("Response mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestExternalProcessor_ResquestBody(t *testing.T) {
	t.Parallel()

	// Assume that Envoy's RequestBody mode is "streamed", which means 1 ProcessingRequest: 1 ProcessingResponse
	tests := map[string]struct {
		chunkSize    int
		stream       bool
		mockCheck    checkFn
		customExpect func(t *testing.T, got []*extproc.ProcessingResponse, input []*extproc.ProcessingRequest)
	}{
		"safe_16B_chunks_ack_each": {
			chunkSize: 16,
			mockCheck: func(ctx context.Context, req guardrails.CheckRequest) (*guardrails.CheckResponse, error) {
				return &guardrails.CheckResponse{Status: guardrails.StatusSuccess}, nil
			},
			customExpect: func(t *testing.T, got []*extproc.ProcessingResponse, _ []*extproc.ProcessingRequest) {
				for i, r := range got {
					if _, ok := r.Response.(*extproc.ProcessingResponse_RequestBody); !ok {
						t.Errorf("resp[%d]: want RequestBody ack, got %T", i, r.Response)
					}
				}
			},
		},
		"safe_2B_chunks_ack_each": {
			chunkSize: 2,
			mockCheck: func(ctx context.Context, req guardrails.CheckRequest) (*guardrails.CheckResponse, error) {
				return &guardrails.CheckResponse{Status: guardrails.StatusSuccess}, nil
			},
			customExpect: func(t *testing.T, got []*extproc.ProcessingResponse, _ []*extproc.ProcessingRequest) {
				for i, r := range got {
					if _, ok := r.Response.(*extproc.ProcessingResponse_RequestBody); !ok {
						t.Errorf("resp[%d]: want RequestBody ack, got %T", i, r.Response)
					}
				}
			},
		},
		"unsafe_input_with_immediate_response": {
			chunkSize: 24,
			mockCheck: func(ctx context.Context, req guardrails.CheckRequest) (*guardrails.CheckResponse, error) {
				return &guardrails.CheckResponse{Status: guardrails.StatusBlocked, RefusalText: modelConfigs[req.Model].RefusalText}, nil
			},
			customExpect: func(t *testing.T, got []*extproc.ProcessingResponse, _ []*extproc.ProcessingRequest) {
				for i, r := range got {
					if i == len(got)-1 {
						if _, ok := r.Response.(*extproc.ProcessingResponse_ImmediateResponse); !ok {
							t.Errorf("want ImmediateResponse, got %T", r.Response)
						}
					} else {
						if _, ok := r.Response.(*extproc.ProcessingResponse_RequestBody); !ok {
							t.Errorf("resp[%d]: want RequestBody ack, got %T", i, r.Response)
						}
					}
				}
				gotImmediateResponse := got[len(got)-1]
				wantImmediateResponse := &extproc.ProcessingResponse{
					Response: &extproc.ProcessingResponse_ImmediateResponse{
						ImmediateResponse: &extproc.ImmediateResponse{
							Status: &envoytype.HttpStatus{Code: 200},
							Headers: &extproc.HeaderMutation{
								SetHeaders: []*envoycore.HeaderValueOption{
									{
										Header: &envoycore.HeaderValue{
											Key:      "content-type",
											RawValue: []byte("application/json"),
										},
									},
									{
										Header: &envoycore.HeaderValue{
											Key: STREAM_ID_HEADER,
										},
									},
								},
							},
						},
					},
				}
				opts := cmp.Options{
					protocmp.Transform(),
					protocmp.IgnoreFields(&envoycore.HeaderValue{}, "raw_value"),
					protocmp.IgnoreFields(&extproc.ImmediateResponse{}, "body"),
				}

				if diff := cmp.Diff(wantImmediateResponse, gotImmediateResponse, opts...); diff != "" {
					t.Errorf("Response mismatch (-want +got):\n%s", diff)
				}
				// explicitly validate content-type because we ignored the value above
				if diff := cmp.Diff(
					wantImmediateResponse.GetImmediateResponse().Headers.GetSetHeaders()[0],
					gotImmediateResponse.GetImmediateResponse().Headers.GetSetHeaders()[0],
					protocmp.Transform(),
				); diff != "" {
					t.Errorf("Response headers mismatch (-want +got):\n%s", diff)
				}
				// ensure body can be serialized into an openai chat completion object
				var gotChatCmpl openai.ChatCompletion
				body := gotImmediateResponse.GetImmediateResponse().Body
				if err := json.Unmarshal(body, &gotChatCmpl); err != nil {
					t.Errorf("could not unmarshal immediate response from extproc: %v", err)
				}
				wantChatCmpl := openai.ChatCompletion{
					Model: modelName,
					Choices: []openai.ChatCompletionChoice{
						{
							FinishReason: "stop",
							Message: openai.ChatCompletionMessage{
								Role:    "assistant",
								Content: modelConfigs[modelName].RefusalText,
							},
						},
					},
					Object: "chat.completion",
				}

				if diff := cmp.Diff(wantChatCmpl, gotChatCmpl, ignoreOpts()); diff != "" {
					t.Errorf("Chat Completion Response mismatch (-want +got):\n%s", diff)
				}
			},
		},
		"streaming_unsafe_input_with_immediate_response": {
			chunkSize: 24,
			stream:    true,
			mockCheck: func(ctx context.Context, req guardrails.CheckRequest) (*guardrails.CheckResponse, error) {
				return &guardrails.CheckResponse{Status: guardrails.StatusBlocked, RefusalText: modelConfigs[req.Model].RefusalText}, nil
			},
			customExpect: func(t *testing.T, got []*extproc.ProcessingResponse, _ []*extproc.ProcessingRequest) {
				for i, r := range got {
					if i == len(got)-1 {
						if _, ok := r.Response.(*extproc.ProcessingResponse_ImmediateResponse); !ok {
							t.Errorf("want ImmediateResponse, got %T", r.Response)
						}
					} else {
						if _, ok := r.Response.(*extproc.ProcessingResponse_RequestBody); !ok {
							t.Errorf("resp[%d]: want RequestBody ack, got %T", i, r.Response)
						}
					}
				}
				gotImmediateResponse := got[len(got)-1]
				wantImmediateResponse := &extproc.ProcessingResponse{
					Response: &extproc.ProcessingResponse_ImmediateResponse{
						ImmediateResponse: &extproc.ImmediateResponse{
							Status: &envoytype.HttpStatus{Code: 200},
							Headers: &extproc.HeaderMutation{
								SetHeaders: []*envoycore.HeaderValueOption{
									{
										Header: &envoycore.HeaderValue{
											Key:      "content-type",
											RawValue: []byte("text/event-stream; charset=utf-8"),
										},
									},
									{
										Header: &envoycore.HeaderValue{
											Key: STREAM_ID_HEADER,
										},
									},
								},
							},
						},
					},
				}
				opts := cmp.Options{
					protocmp.Transform(),
					protocmp.IgnoreFields(&envoycore.HeaderValue{}, "raw_value"),
					protocmp.IgnoreFields(&extproc.ImmediateResponse{}, "body"),
				}
				if diff := cmp.Diff(wantImmediateResponse, gotImmediateResponse, opts...); diff != "" {
					t.Errorf("Response mismatch (-want +got):\n%s", diff)
				}
				// explicitly validate content-type because we ignored the value above
				if diff := cmp.Diff(
					wantImmediateResponse.GetImmediateResponse().Headers.GetSetHeaders()[0],
					gotImmediateResponse.GetImmediateResponse().Headers.GetSetHeaders()[0],
					protocmp.Transform(),
				); diff != "" {
					t.Errorf("Response headers mismatch (-want +got):\n%s", diff)
				}

				// ensure body can be serialized into an openai chat completion chunk object
				// need to parse the SSE into the chat completion chunk + DONE
				var gotChatCmplChunks []openai.ChatCompletionChunk
				body := gotImmediateResponse.GetImmediateResponse().Body
				parser := streams.Parser{}
				acc := streams.ContentAggregator{}
				for _, chunk := range parser.Push(body) {
					acc.AppendChunk(chunk)
					if bytes.Equal(chunk, []byte("[DONE]")) {
						continue
					}
					var chatCmplChunk openai.ChatCompletionChunk
					if err := json.Unmarshal(chunk, &chatCmplChunk); err != nil {
						t.Errorf("failed to unmarshal chat completion chunk: %v", err)
					} else {
						gotChatCmplChunks = append(gotChatCmplChunks, chatCmplChunk)
					}
				}
				if acc.GetText() != modelConfigs[modelName].RefusalText {
					t.Errorf("got %q, want %q", acc.GetText(), modelConfigs[modelName].RefusalText)
				}
				wantChunks := []openai.ChatCompletionChunk{
					{
						Model: modelName,
						Choices: []openai.ChatCompletionChunkChoice{
							{
								Delta: openai.ChatCompletionChunkChoiceDelta{
									Content: modelConfigs[modelName].RefusalText,
									Role:    "assistant",
								},
								FinishReason: "stop",
							},
						},
						Object: "chat.completion.chunk",
					},
				}
				if diff := cmp.Diff(wantChunks, gotChatCmplChunks, ignoreOpts()); diff != "" {
					t.Errorf("Chat Completion Chunks mismatch (-want +got):\n%s", diff)
				}
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Initialize the client to talk to the in-memory extproc server
			client := newTestClient(t, stubGuardrailsClient{check: tc.mockCheck})

			// Generate the ProcessingRequests
			payload := simpleChatCompletionRequestJSON(tc.stream)
			chunks := chunkify(payload, tc.chunkSize)
			var reqs []*extproc.ProcessingRequest
			for i, p := range chunks {
				reqs = append(reqs, makeProcessingRequestWithBodyChunk(p, i == len(chunks)-1))
			}
			// Send the requests
			got := makeRequests(t, client, reqs)

			// Make asserts on the response from our extproc
			if len(got) != len(reqs) {
				t.Errorf("len(responses from extproc) = %v, want %v", len(got), len(reqs))
			}
			if tc.customExpect != nil {
				tc.customExpect(t, got, reqs)
			}
		})
	}
}

func TestExternalProcessor_ResponseBody(t *testing.T) {
	t.Parallel()

	// Assume FULL_DUPLEX_STREAMED mode for ResponseBody processing, i.e. M:N ProcessingRequest to ProcessingResponse.
	tests := map[string]struct {
		stream       bool
		phrase       string
		mockCheck    checkFn
		customExpect func(t *testing.T, got []*extproc.ProcessingResponse, reqs []*extproc.ProcessingRequest, wantPhrase string)
	}{
		"safe_chat_completion_passthrough": {
			stream: false,
			phrase: "Hello there from fake-model",
			mockCheck: func(ctx context.Context, req guardrails.CheckRequest) (*guardrails.CheckResponse, error) {
				return &guardrails.CheckResponse{Status: guardrails.StatusSuccess}, nil
			},
			customExpect: func(t *testing.T, got []*extproc.ProcessingResponse, reqs []*extproc.ProcessingRequest, wantPhrase string) {
				var wantBytes []byte
				var gotBytes []byte
				for _, req := range reqs {
					wantBytes = append(wantBytes, req.GetResponseBody().Body...)
				}
				for _, resp := range got {
					gotBytes = append(gotBytes, resp.GetResponseBody().GetResponse().BodyMutation.GetStreamedResponse().Body...)
				}

				if !bytes.Equal(gotBytes, wantBytes) {
					t.Errorf("got bytes=%q, want %q", gotBytes, wantBytes)
				}

				var chatCompl openai.ChatCompletion
				if err := json.Unmarshal(gotBytes, &chatCompl); err != nil {
					t.Fatal(err)
				}
				if chatCompl.Choices[0].Message.Content != wantPhrase {
					t.Fatal(chatCompl.Choices[0].Message.Content)
				}
			},
		},
		"safe_streaming_chat_completion_passthrough": {
			stream: true,
			phrase: "Hello there from fake-model",
			mockCheck: func(ctx context.Context, req guardrails.CheckRequest) (*guardrails.CheckResponse, error) {
				return &guardrails.CheckResponse{Status: guardrails.StatusSuccess}, nil
			},
			customExpect: func(t *testing.T, got []*extproc.ProcessingResponse, reqs []*extproc.ProcessingRequest, wantPhrase string) {
				var wantBytes []byte
				var gotBytes []byte
				for _, req := range reqs {
					wantBytes = append(wantBytes, req.GetResponseBody().Body...)
				}
				for _, resp := range got {
					gotBytes = append(gotBytes, resp.GetResponseBody().GetResponse().BodyMutation.GetStreamedResponse().Body...)
				}

				if !bytes.Equal(gotBytes, wantBytes) {
					t.Errorf("got bytes=%q, want %q", gotBytes, wantBytes)
				}
				parser := streams.Parser{}
				acc := streams.ContentAggregator{}
				for _, chunk := range parser.Push(gotBytes) {
					acc.AppendChunk(chunk)
				}
				if acc.GetText() != wantPhrase {
					t.Errorf("got %q, want %q", acc.GetText(), wantPhrase)
				}
			},
		},
		"unsafe_chat_completion": {
			stream: false,
			phrase: "You are stupid",
			mockCheck: func(ctx context.Context, req guardrails.CheckRequest) (*guardrails.CheckResponse, error) {
				return &guardrails.CheckResponse{Status: guardrails.StatusBlocked, RefusalText: modelConfigs[req.Model].RefusalText}, nil
			},
			customExpect: func(t *testing.T, got []*extproc.ProcessingResponse, reqs []*extproc.ProcessingRequest, unsafePhrase string) {
				var unsafeBytes []byte
				var gotBytes []byte
				for _, req := range reqs {
					unsafeBytes = append(unsafeBytes, req.GetResponseBody().Body...)
				}
				for _, resp := range got {
					gotBytes = append(gotBytes, resp.GetResponseBody().GetResponse().BodyMutation.GetStreamedResponse().Body...)
				}

				if bytes.Equal(gotBytes, unsafeBytes) {
					t.Errorf("got unsafe bytes=%q, should block", gotBytes)
				}

				var chatCompl openai.ChatCompletion
				if err := json.Unmarshal(gotBytes, &chatCompl); err != nil {
					t.Fatal(err)
				}
				if chatCompl.Choices[0].Message.Content != modelConfigs[modelName].RefusalText {
					t.Fatalf("got chat completion content %q, want %q", chatCompl.Choices[0].Message.Content, modelConfigs[modelName].RefusalText)
				}
			},
		},
		"unsafe_streaming_chat_completion": {
			stream: true,
			phrase: "You are stupid",
			mockCheck: func(ctx context.Context, req guardrails.CheckRequest) (*guardrails.CheckResponse, error) {
				// We split the phrase into words and generate an SSE data even per word.
				content := req.ChatCompletionNewParams.Messages[0].OfAssistant.Content.OfString.String()
				if strings.Contains(content, "stupid") {
					return &guardrails.CheckResponse{Status: guardrails.StatusBlocked, RefusalText: modelConfigs[req.Model].RefusalText}, nil
				}
				return &guardrails.CheckResponse{Status: guardrails.StatusSuccess}, nil
			},
			customExpect: func(t *testing.T, got []*extproc.ProcessingResponse, reqs []*extproc.ProcessingRequest, unsafePhrase string) {
				var unsafeBytes []byte
				var gotBytes []byte
				for _, req := range reqs {
					unsafeBytes = append(unsafeBytes, req.GetResponseBody().Body...)
				}
				for _, resp := range got {
					gotBytes = append(gotBytes, resp.GetResponseBody().GetResponse().BodyMutation.GetStreamedResponse().Body...)
				}

				if bytes.Equal(gotBytes, unsafeBytes) {
					t.Errorf("got unsafe bytes=%q, should block", gotBytes)
				}

				// Ensure that each chunk is serializable
				var gotChatCmplChunks []openai.ChatCompletionChunk
				parser := streams.Parser{}
				acc := streams.ContentAggregator{}
				for _, chunk := range parser.Push(gotBytes) {
					acc.AppendChunk(chunk)
					if bytes.Equal(chunk, []byte("[DONE]")) {
						continue
					}
					var chatCmplChunk openai.ChatCompletionChunk
					if err := json.Unmarshal(chunk, &chatCmplChunk); err != nil {
						t.Errorf("failed to unmarshal chat completion chunk: %v", err)
					} else {
						gotChatCmplChunks = append(gotChatCmplChunks, chatCmplChunk)
					}
				}
				wantAcc := "You are " + modelConfigs[modelName].RefusalText
				if acc.GetText() != wantAcc {
					t.Errorf("got %q, want %q", acc.GetText(), wantAcc)
				}

				wantChunks := []openai.ChatCompletionChunk{
					{
						Model: modelName,
						Choices: []openai.ChatCompletionChunkChoice{
							{
								Delta: openai.ChatCompletionChunkChoiceDelta{
									Content: "",
									Role:    "assistant",
								},
							},
						},
						Object: "chat.completion.chunk",
					},
					{
						Model: modelName,
						Choices: []openai.ChatCompletionChunkChoice{
							{
								Delta: openai.ChatCompletionChunkChoiceDelta{
									Content: "You ",
									Role:    "assistant",
								},
							},
						},
						Object: "chat.completion.chunk",
					},
					{
						Model: modelName,
						Choices: []openai.ChatCompletionChunkChoice{
							{
								Delta: openai.ChatCompletionChunkChoiceDelta{
									Content: "are ",
									Role:    "assistant",
								},
							},
						},
						Object: "chat.completion.chunk",
					},
					{
						Model: modelName,
						Choices: []openai.ChatCompletionChunkChoice{
							{
								Delta: openai.ChatCompletionChunkChoiceDelta{
									Content: modelConfigs[modelName].RefusalText,
									Role:    "assistant",
								},
								FinishReason: "stop",
							},
						},
						Object: "chat.completion.chunk",
					},
				}
				if diff := cmp.Diff(wantChunks, gotChatCmplChunks, ignoreOpts()); diff != "" {
					t.Errorf("Chat Completion Chunks mismatch (-want +got):\n%s", diff)
				}
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			client := newTestClient(t, stubGuardrailsClient{check: tc.mockCheck})

			id := makeChatCompletionID()
			var reqs []*extproc.ProcessingRequest
			var chunks [][]byte
			if tc.stream {
				// Streaming case
				reqs = append(reqs, &extproc.ProcessingRequest{
					Request: &extproc.ProcessingRequest_ResponseHeaders{
						ResponseHeaders: &extproc.HttpHeaders{
							Headers: &envoycore.HeaderMap{
								Headers: []*envoycore.HeaderValue{
									{Key: "content-type", RawValue: []byte("text/event-stream")},
								},
							},
						},
					},
				})
				// Break the SSE stream into Envoy ResponseBody chunks.
				sseBytes := makeSSEFromChatCompletionChunks(t, makeChatCompletionChunks(id, modelName, tc.phrase), true)
				chunks = chunkify(sseBytes, 1)
			} else {
				// Non-streaming case
				reqs = append(reqs, &extproc.ProcessingRequest{
					Request: &extproc.ProcessingRequest_ResponseHeaders{
						ResponseHeaders: &extproc.HttpHeaders{
							Headers: &envoycore.HeaderMap{
								Headers: []*envoycore.HeaderValue{
									{Key: "content-type", RawValue: []byte("application/json")},
								},
							},
						},
					},
				})
				chatCompl := makeChatCompletion(id, time.Now().Unix(), modelName, tc.phrase, "")
				chatComplBytes, err := json.Marshal(chatCompl)
				if err != nil {
					t.Fatal(err)
				}
				chunks = chunkify(chatComplBytes, 1)
			}
			for i, rc := range chunks {
				reqs = append(reqs, &extproc.ProcessingRequest{
					Request: &extproc.ProcessingRequest_ResponseBody{
						ResponseBody: &extproc.HttpBody{Body: rc, EndOfStream: i == len(chunks)-1},
					},
				})
			}
			got := makeRequests(t, client, reqs)
			if len(got) == 0 {
				t.Fatalf("no response")
			}
			// First response should always be a header response
			if _, ok := got[0].Response.(*extproc.ProcessingResponse_ResponseHeaders); !ok {
				t.Fatalf("got=%T, want ResponseHeaders", got[0].Response)
			}

			if len(got) < 2 {
				t.Fatalf("got only 1 ProcessingResponse, want at least 2")
			}

			if tc.customExpect != nil {
				tc.customExpect(t, got[1:], reqs[1:], tc.phrase)
			}
		})
	}
}

func TestMakeRefusalChatCompletionBytes(t *testing.T) {
	t.Parallel()

	// simply make sure json marshals without errors
	tests := map[string]struct {
		id     string
		reason any
	}{
		"no id and no reason": {},
		"no id with reason": {
			reason: errors.New("foo"),
		},
		"id with no reason": {
			id: "chatcmpl-123",
		},
		"id with reason": {
			id:     "chatcmpl-123",
			reason: errors.New("foo"),
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			payload, err := makeRefusalChatCompletionBytes(tc.id, "", "", tc.reason)
			if err != nil {
				t.Errorf("makeRefusalChatCompletionBytes(%v, %v) errored: %v", tc.id, tc.reason, err)
			}
			if len(payload) == 0 {
				t.Errorf("makeRefusalChatCompletionBytes(%v, %v) returned empty", tc.id, tc.reason)
			}
		})
	}
}
func TestMakeRefusalChatCompletionChunkSSEBytes(t *testing.T) {
	t.Parallel()

	// simply make sure json marshals without errors
	tests := map[string]struct {
		id     string
		reason any
	}{
		"no id and no reason": {},
		"no id with reason": {
			reason: errors.New("foo"),
		},
		"id with no reason": {
			id: "chatcmpl-123",
		},
		"id with reason": {
			id:     "chatcmpl-123",
			reason: errors.New("foo"),
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			payload, err := makeRefusalChatCompletionChunkSSEBytes(tc.id, "", "", tc.reason)
			if err != nil {
				t.Errorf("makeRefusalChatCompletionChunkSSEBytes(%v, %v) errored: %v", tc.id, tc.reason, err)
			}
			if len(payload) == 0 {
				t.Errorf("makeRefusalChatCompletionChunkSSEBytes(%v, %v) returned empty payload", tc.id, tc.reason)
			}
			if !bytes.HasPrefix(payload, []byte("data:")) {
				t.Errorf("expected 'data:' prefix in %q", payload)
			}
		})
	}
}

// --- Test helper functions

func makeRequests(t *testing.T, client extproc.ExternalProcessorClient, requests []*extproc.ProcessingRequest) []*extproc.ProcessingResponse {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	stream, err := client.Process(ctx)
	if err != nil {
		t.Fatalf("open Process stream: %v", err)
	}

	for i, req := range requests {
		if err := stream.Send(req); err != nil {
			t.Fatalf("Send request[%d]: %v", i, err)
		}
	}

	// IMPORTANT: close the send direction of the stream to signal the server that all messages have been sent.
	// In response, the server terminates the processing loop, which signals the EOF in the recv loop below.
	if err := stream.CloseSend(); err != nil {
		t.Fatalf("CloseSend: %v", err)
	}

	var responses []*extproc.ProcessingResponse
	for {
		resp, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			// Do not fail completely here, because we want to see what the server responded with.
			t.Errorf("Recv response: %v", err)
			return responses
		}
		responses = append(responses, resp)
	}
	return responses
}

// chunkify splits b into consecutive slices of at most n bytes each.
func chunkify(b []byte, n int) [][]byte {
	if n <= 0 {
		panic("chunk size must be > 0")
	}
	var parts [][]byte
	for i := 0; i < len(b); i += n {
		j := i + n
		if j > len(b) {
			j = len(b)
		}
		parts = append(parts, b[i:j])
	}
	return parts
}

// This is a representative minimal payload. Adjust fields to match your OpenAI client’s schema.
func simpleChatCompletionRequestJSON(stream bool) []byte {
	// We can't use openai.ChatCompletionNewParams because it doesn't have the "stream" field
	return fmt.Appendf(nil, `{
  "model": %q,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Hello, world!"
        }
      ]
    }
  ],
  "stream": %t,
  "temperature": 0.2,
  "top_p": 1.0
}`, modelName, stream)
}

func makeProcessingRequestWithBodyChunk(b []byte, eos bool) *extproc.ProcessingRequest {
	return &extproc.ProcessingRequest{
		Request: &extproc.ProcessingRequest_RequestBody{
			RequestBody: &extproc.HttpBody{Body: b, EndOfStream: eos},
		},
	}
}

// makeSSEFromChatCompletionChunks serializes a sequence of openai.ChatCompletionChunk into an SSE byte stream.
// Each chunk produces "data: <json>\n\n". If includeDone is true, appends "data: [DONE]\n\n".
func makeSSEFromChatCompletionChunks(t *testing.T, chunks []openai.ChatCompletionChunk, includeDone bool) []byte {
	t.Helper()
	var buf bytes.Buffer
	for _, ch := range chunks {
		b, err := json.Marshal(ch)
		if err != nil {
			t.Fatalf("json.Marshal(ChatCompletionChunk): %v", err)
		}
		// json.Encoder adds a trailing newline; trim it so we can control line breaks precisely.
		buf.WriteString("data: ")
		buf.Write(b)
		buf.WriteString("\n\n")
	}
	if includeDone {
		buf.WriteString("data: [DONE]\n\n")
	}
	return buf.Bytes()
}

// makeChatCompletionChunks builds a minimal but realistic chunk sequence:
// - first chunk: delta.role = "assistant"
// - middle chunks: delta.content tokens forming the phrase provided
func makeChatCompletionChunks(id, model, phrase string) []openai.ChatCompletionChunk {
	now := time.Now().Unix()
	var out []openai.ChatCompletionChunk

	// Initial empty chunk to follow OpenAI best practices
	out = append(out, makeChatCompletionChunk(id, now, model, "", ""))

	// tokenized content as simple words and spaces
	tokens := strings.Split(phrase, " ")
	for i, token := range tokens {
		if i < len(tokens)-1 {
			token += " "
		}
		out = append(out, makeChatCompletionChunk(id, now, model, token, ""))
	}

	return out
}

func ignoreOpts() cmp.Options {
	return cmp.Options{
		cmpopts.EquateEmpty(),
		// Broadly ignore unexported on openai and param option wrappers used.
		cmpopts.IgnoreUnexported(
			param.Opt[string]{},
			param.Opt[float64]{},
			param.Opt[int64]{},
			param.Opt[bool]{},
			openai.ChatCompletion{},
			openai.ChatCompletionAudio{},
			openai.ChatCompletionChoice{},
			openai.ChatCompletionChoiceLogprobs{},
			openai.ChatCompletionChunk{},
			openai.ChatCompletionChunkChoice{},
			openai.ChatCompletionChunkChoiceDelta{},
			openai.ChatCompletionChunkChoiceDeltaFunctionCall{},
			openai.ChatCompletionChunkChoiceLogprobs{},
			openai.ChatCompletionMessage{},
			openai.ChatCompletionMessageFunctionCall{},
			openai.CompletionUsage{},
			openai.CompletionUsageCompletionTokensDetails{},
			openai.CompletionUsagePromptTokensDetails{},
		),
		// Ignore internal JSON metadata structs that hold unexported "raw" strings.
		cmpopts.IgnoreFields(openai.ChatCompletion{}, "Created"),
		cmpopts.IgnoreFields(openai.ChatCompletion{}, "ID"),
		cmpopts.IgnoreFields(openai.ChatCompletion{}, "JSON"),
		cmpopts.IgnoreFields(openai.ChatCompletionAudio{}, "JSON"),
		cmpopts.IgnoreFields(openai.ChatCompletionChoice{}, "JSON"),
		cmpopts.IgnoreFields(openai.ChatCompletionChoiceLogprobs{}, "JSON"),
		cmpopts.IgnoreFields(openai.ChatCompletionChunk{}, "Created"),
		cmpopts.IgnoreFields(openai.ChatCompletionChunk{}, "ID"),
		cmpopts.IgnoreFields(openai.ChatCompletionChunk{}, "JSON"),
		cmpopts.IgnoreFields(openai.ChatCompletionChunkChoice{}, "JSON"),
		cmpopts.IgnoreFields(openai.ChatCompletionChunkChoiceDelta{}, "JSON"),
		cmpopts.IgnoreFields(openai.ChatCompletionChunkChoiceDeltaFunctionCall{}, "JSON"),
		cmpopts.IgnoreFields(openai.ChatCompletionChunkChoiceLogprobs{}, "JSON"),
		cmpopts.IgnoreFields(openai.ChatCompletionMessage{}, "JSON"),
		cmpopts.IgnoreFields(openai.ChatCompletionMessageFunctionCall{}, "JSON"),
		cmpopts.IgnoreFields(openai.CompletionUsage{}, "JSON"),
		cmpopts.IgnoreFields(openai.CompletionUsageCompletionTokensDetails{}, "JSON"),
		cmpopts.IgnoreFields(openai.CompletionUsagePromptTokensDetails{}, "JSON"),
	}
}
