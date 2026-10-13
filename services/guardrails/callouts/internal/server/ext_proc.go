// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"strings"
	"time"

	envoycore "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extproc_filters "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	extproc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoytype "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/config"
	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/guardrails"
	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/streams"
)

const (
	PROCESSING_RESPONSE_BODY_SIZE_LIMIT = 128 * 1000 // max 128K bytes

	STREAM_ID_HEADER = "x-guardrails-callout-stream-id" // sent back to the client for tracing
)

var (
	ErrDecodeChatCompletionRequest  = errors.New("decoding chat completion request failed")
	ErrDecodeChatCompletionResponse = errors.New("decoding chat completion response failed")
	ErrGuardrailsCheckOnRequest     = errors.New("guardrails check on request failed")
	ErrGuardrailsCheckOnResponse    = errors.New("guardrails check on response failed")
)

// NewExternalProcessor creates a new instance of the Guardrails implementation of Envoy external processor gRPC API.
func NewExternalProcessor(config config.Config, c guardrails.Client) *externalProcessor {
	return &externalProcessor{config: config, guardrailsClient: c}
}

// emptyRequestHeadersResponse returns an empty ProcessingResponse message that tells Envoy
// that we acknowledged the matching RequestHeaders message, and to move on to the next phase, which is RequestBody processing.
func emptyRequestHeadersResponse() *extproc.ProcessingResponse {
	return &extproc.ProcessingResponse{
		Response: &extproc.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extproc.HeadersResponse{
				Response: &extproc.CommonResponse{},
			},
		},
		// OverrideMessageTimeout: durationpb.New(30 * time.Second),
	}
}

// emptyRequestBodyResponse returns an empty ProcessingResponse message that tells Envoy
// that we acknowledged the matching RequestBody message, and to move on to the next message, which could be more RequestBody messages,
// or switch to the ResponseHeader processing phase.
func emptyRequestBodyResponse() *extproc.ProcessingResponse {
	return &extproc.ProcessingResponse{
		Response: &extproc.ProcessingResponse_RequestBody{
			RequestBody: &extproc.BodyResponse{
				Response: &extproc.CommonResponse{},
			},
		},
	}
}

// newImmediateResponse returns a ProcessingResponse containing an ImmediateResponse which would terminate Envoy's request processing.
// This only works during the RequestHeaders and RequestBody phase of the processing lifecycle.
func newImmediateResponse(streamID string, statusCode envoytype.StatusCode, body, contentType string) *extproc.ProcessingResponse {
	if contentType == "" {
		contentType = "application/json"
	}
	return &extproc.ProcessingResponse{
		Response: &extproc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &extproc.ImmediateResponse{
				Status: &envoytype.HttpStatus{Code: statusCode},
				Headers: &extproc.HeaderMutation{
					SetHeaders: []*envoycore.HeaderValueOption{
						{
							Header: &envoycore.HeaderValue{
								Key:      "content-type",
								RawValue: []byte(contentType),
							},
						},
						{
							Header: &envoycore.HeaderValue{
								Key:      STREAM_ID_HEADER,
								RawValue: []byte(streamID),
							},
						},
					},
				},
				Body: []byte(body),
			},
		},
	}
}

// streamedMutationResponseBody returns a ProcessingResponse containing a StreamedBodyResponse
// https://pkg.go.dev/github.com/envoyproxy/go-control-plane/envoy@v1.32.4/service/ext_proc/v3#BodyMutation_StreamedResponse
func streamedMutationResponseBody(body []byte, end bool) *extproc.ProcessingResponse {
	return &extproc.ProcessingResponse{
		Response: &extproc.ProcessingResponse_ResponseBody{
			ResponseBody: &extproc.BodyResponse{
				Response: &extproc.CommonResponse{
					Status: extproc.CommonResponse_CONTINUE_AND_REPLACE,
					BodyMutation: &extproc.BodyMutation{
						Mutation: &extproc.BodyMutation_StreamedResponse{
							StreamedResponse: &extproc.StreamedBodyResponse{
								Body:        body,
								EndOfStream: end,
							},
						},
					},
				},
			},
		},
	}
}

// isResponseStreaming determines whether the response from an LLM is a stream based on the header.
func isResponseStreaming(headers []*envoycore.HeaderValue) bool {
	// Check if streaming is enabled in the headers
	// content-type: text/event-stream; charset=utf-8
	for _, header := range headers {
		if header.Key == "content-type" && strings.HasPrefix(string(header.RawValue), "text/event-stream") {
			return true
		}
	}
	return false
}

type respStreamState struct {
	// SSE parser and content aggregator
	parser streams.Parser
	agg    streams.ContentAggregator

	// Window of complete events
	events          [][]byte
	bytesSinceCheck int

	// Tunables
	eventsPerCheck int
}

func (ss *respStreamState) isTimeToCheck() bool {
	return len(ss.events) >= ss.eventsPerCheck || ss.agg.Done()
}
func (ss *respStreamState) Reset() {
	ss.events = ss.events[:0]
	ss.bytesSinceCheck = 0
	ss.agg.Reset()
}

// formatSSE takes a parsed data message from an SSE (the part after "data: ")
// and return the original to be sent back to the client.
func formatSSE(chunk []byte, end bool) []byte {
	prefix := []byte(streams.DATA_FIELD_PREFIX + " ")
	sep := []byte("\n\n")
	sentinel := []byte(streams.DONE_SENTINEL)
	size := len(prefix) + len(chunk) + len(sep)
	if end {
		size += len(prefix) + len(sentinel) + len(sep)
	}
	out := make([]byte, 0, size)
	out = append(out, prefix...)
	out = append(out, chunk...)
	out = append(out, sep...)
	if end {
		out = append(out, prefix...)
		out = append(out, sentinel...)
		out = append(out, sep...)
	}
	return out
}

// externalProcessor implements the ext_proc API
type externalProcessor struct {
	// embedding for forward compatibility
	extproc.UnimplementedExternalProcessorServer
	config           config.Config
	guardrailsClient guardrails.Client
}

func (s *externalProcessor) Process(stream extproc.ExternalProcessor_ProcessServer) error {
	ctx := stream.Context()
	sc := trace.SpanFromContext(ctx).SpanContext()
	var streamID string
	if sc.IsValid() {
		streamID = sc.TraceID().String()
	} else {
		// If can't get traceID, generate a new UUID per stream to identify in the logs.
		streamID = uuid.New().String()
	}
	log.Printf("[%s] New stream started...", streamID)

	var (
		// extract the model name from the request
		modelName string
		// Buffer Envoy's request body chunks
		requestBodyBuffer []byte
		// Buffer Envoy's resposne body chunks (for non-streaming chat completions)
		responseBodyBuff []byte
		// Keep a reference to the check request because it can be re-used for request and response checking
		inputCheckRequest guardrails.CheckRequest
		// Keep a reference to the user's input message to be re-used later
		originalRequestMessages []openai.ChatCompletionMessageParamUnion
		// Determine whether the LLM response is streaming or not
		isStreaming bool
		// Manage stream state for parsing and aggregating streaming chat completion chunks
		streamState *respStreamState
	)
	for {
		select {
		case <-ctx.Done():
			log.Printf("[%s] Stream context cancelled: %v", streamID, ctx.Err())
			return nil
		default:
			req, err := stream.Recv()
			if err != nil {
				if err == io.EOF || status.Code(err) == codes.Canceled {
					// It's important to return nil here. It is not our responsibility to return EOF to the client.
					// By ending the stream normally, the gRPC server will return EOF to the client.
					return nil
				}
				log.Printf("[%s] Failed to receive ProcessingRequest from stream: %v", streamID, err)
				return err
			}

			switch {
			case req.GetRequestHeaders() != nil:
				log.Printf("[%s] Received RequestHeaders", streamID)
				if err := stream.Send(emptyRequestHeadersResponse()); err != nil {
					log.Printf("[%s] Failed to send ack on RequestHeaders: %v", streamID, err)
					return err
				}
			case req.GetRequestBody() != nil:
				if len(requestBodyBuffer) == 0 {
					log.Printf("[%s] Received RequestBody", streamID)
				}
				body := req.GetRequestBody()
				requestBodyBuffer = append(requestBodyBuffer, body.Body...)
				if !body.EndOfStream {
					// Acknowledge the chunk and continue
					if err := stream.Send(emptyRequestBodyResponse()); err != nil {
						log.Printf("[%s] Failed to send ack on RequestBody: %v", streamID, err)
						return err
					}
					continue
				}
				if err := json.Unmarshal(requestBodyBuffer, &inputCheckRequest); err != nil {
					// TODO return a HTTP error json with application/problem+json instead
					log.Printf("[%s] Could not Unmarshal ChatCompletion request: %v", streamID, err)
					// We have no choice but to use the default refusal text here because we don't know the model.
					payload, _ := makeRefusalChatCompletionBytes("" /* ID */, "" /* model */, s.config.Guardrails.DefaultRefusalText, ErrDecodeChatCompletionRequest)
					return stream.Send(newImmediateResponse(streamID, envoytype.StatusCode_BadRequest, string(payload), ""))
				}
				originalRequestMessages = inputCheckRequest.Messages
				// reset the buffer because we no longer need it
				requestBodyBuffer = requestBodyBuffer[:0]
				checkResp, err := s.check(ctx, streamID, inputCheckRequest)
				if err != nil || checkResp.Status != guardrails.StatusSuccess {
					// UNSAFE: send a denial message, and end request immediately
					statusCode := envoytype.StatusCode_OK
					contentType := "application/json"
					var payload []byte
					// return external error in case there's sensitive info in the internal err
					var extErr error
					if err != nil {
						log.Printf("[%s] Guardrails check on input failed: %v", streamID, err)
						statusCode = envoytype.StatusCode_InternalServerError
						extErr = ErrGuardrailsCheckOnRequest
					}
					if inputCheckRequest.Stream {
						contentType = "text/event-stream; charset=utf-8"
						payload, _ = makeRefusalChatCompletionChunkSSEBytes("", inputCheckRequest.Model, checkResp.RefusalText, extErr)
					} else {
						payload, _ = makeRefusalChatCompletionBytes("", inputCheckRequest.Model, checkResp.RefusalText, extErr)
					}
					// Terminate processing stream
					return stream.Send(newImmediateResponse(streamID, statusCode, string(payload), contentType))
				}
				// SAFE: send back an ack and move on to the next phase
				log.Printf("[%s] Input is checked and determined to be safe", streamID)
				if err := stream.Send(emptyRequestBodyResponse()); err != nil {
					log.Printf("[%s] Failed to send ack on RequestBody: %v", streamID, err)
					return err
				}
			case req.GetRequestTrailers() != nil:
				log.Printf("[%s] Received RequestTrailers", streamID)
				if err := stream.Send(&extproc.ProcessingResponse{
					Response: &extproc.ProcessingResponse_RequestTrailers{
						RequestTrailers: &extproc.TrailersResponse{},
					},
				}); err != nil {
					return err
				}
			case req.GetResponseHeaders() != nil:
				headers := req.GetResponseHeaders().Headers.Headers
				isStreaming = isResponseStreaming(headers)
				log.Printf("[%s] Received ResponseHeaders, isStreaming=%v", streamID, isStreaming)
				// Ensure FULL_DUPLEX_STREAMED mode for response body sends.
				// Note: this only works if Envoy is configured to allows mode overrides.
				response := &extproc.ProcessingResponse{
					Response: &extproc.ProcessingResponse_ResponseHeaders{
						ResponseHeaders: &extproc.HeadersResponse{
							Response: &extproc.CommonResponse{
								HeaderMutation: &extproc.HeaderMutation{
									SetHeaders: []*envoycore.HeaderValueOption{
										{
											Header: &envoycore.HeaderValue{
												Key:      STREAM_ID_HEADER,
												RawValue: []byte(streamID),
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
				}
				if err := stream.Send(response); err != nil {
					log.Printf("[%s] Failed to send ack on ResponseHeaders: %v", streamID, err)
					return err
				}
			case req.GetResponseBody() != nil:
				body := req.GetResponseBody()
				chunk := body.Body
				eos := body.EndOfStream
				if isStreaming {
					// Initialize stream state to help implement the sliding window algorithm below.
					// Lazy init, since we might not even get to this stage if the request body is denied.
					if streamState == nil {
						log.Printf("[%s] Received ResponseBody, isStreaming=%v", streamID, isStreaming)
						streamState = &respStreamState{
							eventsPerCheck: s.config.ExtProc.EventsPerCheck,
						}
					}
					// Parse stream into chat completion chunks, and aggregate content.
					for _, event := range streamState.parser.Push(chunk) {
						if err := streamState.agg.AppendChunk(event); err != nil {
							// Tolerate errors due to JSON processing during stream processing.
							log.Printf("[%s] Failed to parse server sent event: %v", streamID, err)
						}
						streamState.events = append(streamState.events, event)
						streamState.bytesSinceCheck += len(event)
					}
					// Treat the response model name as the source of truth
					modelName = streamState.agg.GetModelName()
					inputCheckRequest.Model = modelName
					// Run Guardrails Check either when the EOS is reached or stream buffer is ready to be checked.
					if eos || streamState.isTimeToCheck() {
						// Envoy sometimes sends us an empty body at the EOS for some reason, so only call guardrails when there are relevant events.
						if len(streamState.events) > 0 {
							newMessages := make([]openai.ChatCompletionMessageParamUnion, 0, len(originalRequestMessages)+1)
							newMessages = append(newMessages, originalRequestMessages...)
							newMessages = append(newMessages, streamState.agg.BuildChatCompletionMessage())
							inputCheckRequest.Messages = newMessages
							checkResp, err := s.check(ctx, streamID, inputCheckRequest)
							if err != nil || checkResp.Status != guardrails.StatusSuccess {
								// UNSAFE: send replacement and end processing stream
								extErr := err
								if err != nil {
									log.Printf("[%s] Guardrails check on streaming response failed: %v", streamID, err)
									extErr = ErrGuardrailsCheckOnResponse
								}
								payload, _ := makeRefusalChatCompletionChunkSSEBytes(streamState.agg.GetChatCompletionID(), modelName, checkResp.RefusalText, extErr)
								return stream.Send(streamedMutationResponseBody(payload, true /* end */))
							}
							// SAFE: flush complete events back out
							for _, event := range streamState.events {
								outputMsg := formatSSE(event, false)
								if len(outputMsg) >= PROCESSING_RESPONSE_BODY_SIZE_LIMIT {
									// TODO make sure not to exceed the 128KB limit per ProcessingResponse message.
									log.Printf("Streaming response body size %d exceeded limit of %d", len(outputMsg), PROCESSING_RESPONSE_BODY_SIZE_LIMIT)
								}
								if err := stream.Send(streamedMutationResponseBody(outputMsg, false)); err != nil {
									log.Printf("[%s] Failed to send response body: %v", streamID, err)
									return err
								}
							}
							// Reset stream window, and clear aggregator's internal text buffer.
							streamState.Reset()
						}
						if eos {
							log.Printf("[%s] ResponseBody processing reached end of stream", streamID)
							// Make sure that we terminate the stream otherwise the downstream client hangs.
							return stream.Send(streamedMutationResponseBody(nil, true /* end */))
						}
					}
				} else {
					// Non-streaming JSON path: accumulate until EOS
					responseBodyBuff = append(responseBodyBuff, chunk...)
					if !eos {
						continue
					}
					// EOS reached: parse JSON once
					var chatCmpl openai.ChatCompletion
					if err := json.Unmarshal(responseBodyBuff, &chatCmpl); err != nil {
						// Decoding the ChatCompletion response failed, block to avoid leaking unsafe information.
						log.Printf("[%s] Failed to decode chat completion response: %v", streamID, err)
						// HACK: we need to retrieve the refusal text for the given model.
						// This is the only reason for embedding the top-level config into the externalProcessor.
						// Consider moving this logic to the guardrails package so that externalProcessor only needs to embed ExtProc config.
						refusalText := s.config.Guardrails.Models[modelName].RefusalText
						if refusalText == "" {
							refusalText = s.config.Guardrails.DefaultRefusalText
						}
						payload, _ := makeRefusalChatCompletionBytes(
							"", /* chatcmpl-id */
							modelName,
							refusalText,
							ErrDecodeChatCompletionResponse,
						)
						return stream.Send(streamedMutationResponseBody(payload, true))
					}
					// Treat the response model name as the source of truth.
					modelName = chatCmpl.Model
					inputCheckRequest.Model = modelName
					// Check the response
					if len(chatCmpl.Choices) > 0 {
						newMsgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(originalRequestMessages)+1)
						newMsgs = append(newMsgs, inputCheckRequest.Messages...)
						newMsgs = append(newMsgs, openai.AssistantMessage(chatCmpl.Choices[0].Message.Content))
						inputCheckRequest.Messages = newMsgs
						checkResp, err := s.check(ctx, streamID, inputCheckRequest)
						var extErr error
						if err != nil || checkResp.Status != guardrails.StatusSuccess {
							if err != nil {
								log.Printf("[%s] Guardrails check on response failed: %v", streamID, err)
								extErr = ErrGuardrailsCheckOnResponse
							}
							payload, _ := makeRefusalChatCompletionBytes(chatCmpl.ID, modelName, checkResp.RefusalText, extErr)
							return stream.Send(streamedMutationResponseBody(payload, true))
						}
					}
					// SAFE: pass through original body unchanged
					// TODO make sure this doesn't go over the 128KB limit.
					return stream.Send(streamedMutationResponseBody(responseBodyBuff, true))
				}
			case req.GetResponseTrailers() != nil:
				log.Printf("[%s] Received ResponseTrailers", streamID)
				if err := stream.Send(&extproc.ProcessingResponse{
					Response: &extproc.ProcessingResponse_ResponseTrailers{
						ResponseTrailers: &extproc.TrailersResponse{},
					},
				}); err != nil {
					return err
				}
			default:
				log.Printf("[%s] Received empty request: this should never happen", streamID)
				if err := stream.Send(&extproc.ProcessingResponse{}); err != nil {
					return err
				}
			}
		}
	}
}

func (s *externalProcessor) check(ctx context.Context, streamID string, checkReq guardrails.CheckRequest) (*guardrails.CheckResponse, error) {
	ctx = guardrails.SetStreamID(ctx, streamID)
	return s.guardrailsClient.Check(ctx, checkReq)
}

// Compile-time check
var _ extproc.ExternalProcessorServer = (*externalProcessor)(nil)

// --- Helpers ---

func makeChatCompletionID() string {
	return "chatcmpl-" + uuid.New().String()
}

func makeChatCompletionChoice(content string, finishReason openai.CompletionChoiceFinishReason) openai.ChatCompletionChoice {
	choice := openai.ChatCompletionChoice{
		Message: openai.ChatCompletionMessage{
			Content: content,
		},
	}
	// ChatCompletionMessage struct's role is hard coded to a constant Assistant type
	// however the string value "assistant" is not automatically populated upon initialization.
	choice.Message.Role = choice.Message.Role.Default()
	if finishReason != "" {
		choice.FinishReason = string(finishReason)
	}
	return choice
}

func makeAssistantChunkChoice(content string, finishReason openai.CompletionChoiceFinishReason) openai.ChatCompletionChunkChoice {
	choice := openai.ChatCompletionChunkChoice{
		Delta: openai.ChatCompletionChunkChoiceDelta{
			Content: content,
			Role:    "assistant",
		},
	}
	if finishReason != "" {
		choice.FinishReason = string(finishReason)
	}
	return choice

}

func makeChatCompletion(id string, createdAt int64, model string, content string, finishReason openai.CompletionChoiceFinishReason) openai.ChatCompletion {
	if id == "" {
		id = makeChatCompletionID()
	}
	out := openai.ChatCompletion{
		ID:      id,
		Created: createdAt,
		Model:   model,
		Choices: []openai.ChatCompletionChoice{
			makeChatCompletionChoice(content, finishReason),
		},
	}
	out.Object = out.Object.Default()
	return out
}

func makeChatCompletionChunk(id string, createdAt int64, model string, content string, finishReason openai.CompletionChoiceFinishReason) openai.ChatCompletionChunk {
	if id == "" {
		id = makeChatCompletionID()
	}
	out := openai.ChatCompletionChunk{
		ID:      id,
		Created: createdAt,
		Model:   model,
		Choices: []openai.ChatCompletionChunkChoice{makeAssistantChunkChoice(content, finishReason)},
	}
	out.Object = out.Object.Default()
	return out
}

func makeRefusalChatCompletionBytes(id string, model string, refusalText string, reason any) ([]byte, error) {
	if reason != nil {
		refusalText = fmt.Sprintf("%s because %v", refusalText, reason)
	}
	chatCompletion := makeChatCompletion(id, time.Now().Unix(), model, refusalText, openai.CompletionChoiceFinishReasonStop)
	return json.Marshal(chatCompletion)
}

func makeRefusalChatCompletionChunkSSEBytes(id string, model string, refusalText string, reason any) ([]byte, error) {
	if reason != nil {
		refusalText = fmt.Sprintf("%s because %v", refusalText, reason)
	}
	chunk := makeChatCompletionChunk(id, time.Now().Unix(), model, refusalText, openai.CompletionChoiceFinishReasonStop)
	payload, err := json.Marshal(chunk)
	if err != nil {
		return nil, err
	}
	return formatSSE(payload, true), nil
}
