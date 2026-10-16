// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestJobStepClient_GetJobStepConfig(t *testing.T) {
	testCases := []struct {
		name          string
		principal     *Principal
		statusCode    int
		responseBody  string
		expectError   string
		expected      map[string]any
		expectHeaders bool
	}{
		{
			name:          "strips_step_spec_name",
			principal:     &Principal{ID: "user-1"},
			statusCode:    http.StatusOK,
			responseBody:  `{"name":"step-a","config":{"_step_spec_name":"train","epochs":3,"nested":{"k":"v"}}}`,
			expected:      map[string]any{"epochs": float64(3), "nested": map[string]any{"k": "v"}},
			expectHeaders: true,
		},
		{
			name:         "missing_config_is_empty_object",
			statusCode:   http.StatusOK,
			responseBody: `{"name":"step-a"}`,
			expected:     map[string]any{},
		},
		{
			name:         "not_found",
			statusCode:   http.StatusNotFound,
			responseBody: `{"detail":"not found"}`,
			expectError:  "status code 404",
		},
		{
			name:         "invalid_json",
			statusCode:   http.StatusOK,
			responseBody: `not json`,
			expectError:  "failed to decode job step response",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/apis/jobs/v2/workspaces/ws/jobs/job-1/steps/step-a" {
					t.Errorf("unexpected path %s", r.URL.Path)
				}
				gotID := r.Header.Get("X-NHX-Principal-Id")
				gotOnBehalf := r.Header.Get("X-NHX-Principal-On-Behalf-Of")
				if tc.expectHeaders && (gotID != "service:jobs" || gotOnBehalf != tc.principal.ID) {
					t.Errorf("unexpected principal headers id=%q on-behalf-of=%q", gotID, gotOnBehalf)
				}
				if !tc.expectHeaders && (gotID != "" || gotOnBehalf != "") {
					t.Errorf("expected no principal headers, got id=%q on-behalf-of=%q", gotID, gotOnBehalf)
				}
				w.WriteHeader(tc.statusCode)
				fmt.Fprint(w, tc.responseBody)
			}))
			defer server.Close()

			client := NewJobStepClientWithHTTPClient(server.URL, tc.principal, server.Client())
			data, err := client.GetJobStepConfig(context.Background(), "ws", "job-1", "step-a")

			if tc.expectError != "" {
				if err == nil || !strings.Contains(err.Error(), tc.expectError) {
					t.Fatalf("expected error containing %q, got %v", tc.expectError, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatalf("config is not JSON: %v", err)
			}
			if !reflect.DeepEqual(got, tc.expected) {
				t.Fatalf("got %v, want %v", got, tc.expected)
			}
		})
	}
}

func TestJobStepClient_GetJobStepConfigCancelled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	client := NewJobStepClientWithHTTPClient(server.URL, nil, server.Client())
	if _, err := client.GetJobStepConfig(ctx, "ws", "job-1", "step-a"); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestGetJobStepURLEscapesSegments(t *testing.T) {
	got := getJobStepURL("http://h", "ws", "job/x", "step a")
	want := "http://h/apis/jobs/v2/workspaces/ws/jobs/job%2Fx/steps/step%20a"
	if got != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}
