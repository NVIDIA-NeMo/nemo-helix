// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func transferTestClient(t *testing.T, handler http.HandlerFunc) *TransferClient {
	t.Helper()
	t.Setenv("NEMO_WORKLOAD_TOKEN_FILE", "")
	t.Setenv("NHX_ACCESS_TOKEN_FILE", "")
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	endpoint, err := ParseEndpoint(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	return NewSessionWithHTTPClient(server.Client()).TransferClient(endpoint, endpoint, nil)
}
func TestTransferUploadsPreservePathsAndAllowOverwrites(t *testing.T) {
	writes := []string{}
	client := transferTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/by-id/") {
			fmt.Fprint(w, `{"id":"dst","workspace":"ws"}`)
			return
		}
		if r.Method != "PUT" || r.URL.Path != "/apis/files/v2/workspaces/ws/filesets/id:dst/-/results/model.bin" {
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			w.WriteHeader(400)
			return
		}
		body, _ := io.ReadAll(r.Body)
		writes = append(writes, string(body))
		fmt.Fprint(w, `{}`)
	})
	for _, content := range []string{"first", "second"} {
		directory := t.TempDir()
		if err := os.WriteFile(filepath.Join(directory, "model.bin"), []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
		result := client.Upload(context.Background(), "ws", "success", []FilesetPush{{FilesetPull: FilesetPull{FilesetID: "dst", Directory: directory, Path: "results"}}})
		if len(result) != 1 || !result[0].Success {
			t.Fatalf("upload failed: %+v", result)
		}
	}
	if strings.Join(writes, ",") != "first,second" {
		t.Fatalf("unexpected writes %v", writes)
	}
}
func TestTransferPullStreamsNestedBytes(t *testing.T) {
	content := []byte{0, 1, 255, 2}
	client := transferTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/by-id/"):
			fmt.Fprint(w, `{"id":"src","workspace":"ws"}`)
		case strings.HasSuffix(r.URL.Path, "/files"):
			fmt.Fprint(w, `{"data":[{"path":"data/sub/file.bin"}]}`)
		default:
			w.Write(content)
		}
	})
	directory := t.TempDir()
	if err := client.Pull(context.Background(), "ws", []FilesetPull{{FilesetID: "src", Directory: directory, Path: "data"}}); err != nil {
		t.Fatal(err)
	}
	actual, err := os.ReadFile(filepath.Join(directory, "sub", "file.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(actual, content) {
		t.Fatal("download bytes changed")
	}
}
func TestTransferPullRejectsTraversal(t *testing.T) {
	client := transferTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/by-id/") {
			fmt.Fprint(w, `{"id":"src","workspace":"ws"}`)
			return
		}
		fmt.Fprint(w, `{"data":[{"path":"../escape"}]}`)
	})
	if err := client.Pull(context.Background(), "ws", []FilesetPull{{FilesetID: "src", Directory: t.TempDir()}}); err == nil {
		t.Fatal("accepted path traversal")
	}
}
func TestTransferUploadAttemptsIndependentDestinations(t *testing.T) {
	goodWrites := 0
	client := transferTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "bad") {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		if strings.Contains(r.URL.Path, "/by-id/") {
			fmt.Fprint(w, `{"id":"good","workspace":"ws"}`)
			return
		}
		goodWrites++
		fmt.Fprint(w, `{}`)
	})
	directory := t.TempDir()
	os.WriteFile(filepath.Join(directory, "x"), []byte("payload"), 0600)
	results := client.Upload(context.Background(), "ws", "failure", []FilesetPush{
		{FilesetPull: FilesetPull{FilesetID: "bad", Directory: directory}},
		{FilesetPull: FilesetPull{FilesetID: "good", Directory: directory}, When: "failure"},
		{FilesetPull: FilesetPull{FilesetID: "skipped", Directory: directory}, When: "success"},
	})
	if len(results) != 2 || results[0].Success || !results[1].Success || goodWrites != 1 {
		t.Fatalf("wrong results: %+v writes=%d", results, goodWrites)
	}
}
func TestTransferUploadRejectsSymlink(t *testing.T) {
	client := transferTestClient(t, func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, `{"id":"dst","workspace":"ws"}`) })
	directory := t.TempDir()
	if err := os.Symlink("/etc/passwd", filepath.Join(directory, "link")); err != nil {
		t.Fatal(err)
	}
	result := client.Upload(context.Background(), "ws", "success", []FilesetPush{{FilesetPull: FilesetPull{FilesetID: "dst", Directory: directory}}})
	if len(result) != 1 || result[0].Success {
		t.Fatal("uploaded symlink")
	}
}
