// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/NVIDIA-NeMo/nemo-helix/services/core/jobs/jobs-launcher/nhxclient"
)

func TestRunTransfersApplyStepDecisionAndPreserveWorkloadResult(t *testing.T) {
	for _, exitCode := range []int{0, 7, 75} {
		t.Run(strconv.Itoa(exitCode), func(t *testing.T) {
			directory := t.TempDir()
			var mu sync.Mutex
			uploads := []string{}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if strings.HasPrefix(r.URL.Path, "/apis/jobs/") {
					var report struct {
						Transfer nhxclient.TransferProgress `json:"transfer"`
					}
					if err := json.NewDecoder(r.Body).Decode(&report); err != nil {
						t.Error(err)
						w.WriteHeader(400)
						return
					}
					action := "wait"
					switch report.Transfer.Phase {
					case "ready":
						action = "run"
					case "finished":
						if exitCode == 75 {
							t.Error("paused workload reported completion")
						}
						action = "upload_success"
						if exitCode != 0 {
							action = "upload_failure"
						}
					case "uploaded":
						action = "done"
					}
					fmt.Fprintf(w, `{"transfer_action":%q}`, action)
					return
				}
				if strings.Contains(r.URL.Path, "/by-id/") {
					id := filepath.Base(r.URL.Path)
					fmt.Fprintf(w, `{"id":%q,"workspace":"ws"}`, id)
					return
				}
				if r.Method == "PUT" {
					mu.Lock()
					uploads = append(uploads, r.URL.Path)
					mu.Unlock()
				}
				fmt.Fprint(w, `{}`)
			}))
			defer server.Close()
			for name, value := range map[string]string{"NHX_BASE_URL": server.URL, "NHX_JOBS_URL": server.URL, "NHX_FILES_URL": server.URL, "NEMO_JOB_WORKSPACE": "ws", "NEMO_JOB_ID": "job", "NEMO_JOB_ATTEMPT_ID": "attempt", "NEMO_JOB_STEP": "step", "NEMO_JOB_TASK": "task", "NHX_ACCESS_TOKEN_FILE": "", "NEMO_WORKLOAD_TOKEN_FILE": "", "OUTPUT": directory} {
				t.Setenv(name, value)
			}
			config := nhxclient.StepFilesets{Push: []nhxclient.FilesetPush{
				{FilesetPull: nhxclient.FilesetPull{FilesetID: "always", Directory: directory}},
				{FilesetPull: nhxclient.FilesetPull{FilesetID: "success", Directory: directory}, When: "success"},
				{FilesetPull: nhxclient.FilesetPull{FilesetID: "failure", Directory: directory}, When: "failure"},
			}}
			raw, _ := json.Marshal(config)
			t.Setenv(nhxclient.TransfersEnv, string(raw))
			code, err := runExec(context.Background(), []string{"/bin/sh", "-c", `printf payload > "$OUTPUT/model.bin"; exit ` + strconv.Itoa(exitCode)}, nil)
			if code != exitCode || (err != nil) != (exitCode != 0) {
				t.Fatalf("code=%d err=%v", code, err)
			}
			mu.Lock()
			defer mu.Unlock()
			if exitCode == 75 {
				if len(uploads) != 0 {
					t.Fatalf("paused workload uploaded: %v", uploads)
				}
				return
			}
			if len(uploads) != 2 {
				t.Fatalf("wrong uploads: %v", uploads)
			}
			expected := "success"
			skipped := "failure"
			if exitCode != 0 {
				expected, skipped = skipped, expected
			}
			joined := strings.Join(uploads, ",")
			if !strings.Contains(joined, "id:"+expected) || strings.Contains(joined, "id:"+skipped) || !strings.Contains(joined, "id:always/-/model.bin") {
				t.Fatalf("wrong destinations: %v", uploads)
			}
		})
	}
}
func TestRunPullFailureDoesNotStartCommand(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "ran")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/apis/jobs/") {
			fmt.Fprint(w, `{"transfer_action":"wait"}`)
		} else {
			w.WriteHeader(404)
		}
	}))
	defer server.Close()
	for name, value := range map[string]string{"NHX_BASE_URL": server.URL, "NHX_JOBS_URL": server.URL, "NHX_FILES_URL": server.URL, "NEMO_JOB_WORKSPACE": "ws", "NEMO_JOB_ID": "job", "NEMO_JOB_ATTEMPT_ID": "attempt", "NEMO_JOB_STEP": "step", "NEMO_JOB_TASK": "task", "NHX_ACCESS_TOKEN_FILE": "", "NEMO_WORKLOAD_TOKEN_FILE": ""} {
		t.Setenv(name, value)
	}
	raw, _ := json.Marshal(nhxclient.StepFilesets{Pull: []nhxclient.FilesetPull{{FilesetID: "missing", Directory: t.TempDir()}}})
	t.Setenv(nhxclient.TransfersEnv, string(raw))
	code, err := runExec(context.Background(), []string{"/bin/sh", "-c", fmt.Sprintf("touch %q", marker)}, nil)
	if code == 0 || err == nil {
		t.Fatal("pull failure succeeded")
	}
	if _, err = os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("workload ran")
	}
}
