// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"
)

const TransfersEnv = "NEMO_JOB_FILESET_TRANSFERS"

type FilesetPull struct {
	FilesetID string `json:"fileset_id"`
	Directory string `json:"directory"`
	Path      string `json:"path"`
}
type FilesetPush struct {
	FilesetPull
	When string `json:"when"`
}
type StepFilesets struct {
	Pull []FilesetPull `json:"pull"`
	Push []FilesetPush `json:"push"`
}
type UploadResult struct {
	FilesetID string `json:"fileset_id"`
	Success   bool   `json:"success"`
	Error     string `json:"error,omitempty"`
}
type TransferProgress struct {
	Attempts  map[string]int `json:"attempts,omitempty"`
	RuntimeID string         `json:"runtime_id"`
	Phase     string         `json:"phase"`
	ExitCode  *int           `json:"exit_code,omitempty"`
	Stopped   bool           `json:"stopped"`
	Leader    bool           `json:"leader"`
	Error     string         `json:"error,omitempty"`
	Uploads   []UploadResult `json:"uploads,omitempty"`
}
type ExecutionRef struct{ Workspace, Job, Attempt, Step, Task string }
type TransferAttemptObserver func(context.Context, string, int) error

type TransferClient struct {
	observer    TransferAttemptObserver
	session     *Session
	jobs, files Endpoint
	principal   *Principal
}

func (s *Session) TransferClient(jobs, files Endpoint, principal *Principal) *TransferClient {
	return &TransferClient{session: s, jobs: jobs, files: files, principal: principal}
}

type TransferHTTPError struct {
	Status        int
	RetryConflict bool
}

func (e *TransferHTTPError) Error() string {
	return fmt.Sprintf("Fileset transfer HTTP status %d", e.Status)
}
func (e *TransferHTTPError) retryable() bool {
	return e.Status == 408 || e.Status == 429 || e.Status >= 500 || (e.Status == 409 && e.RetryConflict)
}

// RetryTransfer retries transient I/O only. Each operation owns its request body
// so a retry reopens files instead of replaying an already consumed stream.
func RetryTransfer(ctx context.Context, operation func() error) error {
	return retryTransfer(ctx, operation, nil)
}

func (c *TransferClient) ObserveAttempts(observer TransferAttemptObserver) { c.observer = observer }

func (c *TransferClient) retry(ctx context.Context, key string, operation func() error) error {
	return retryTransfer(ctx, operation, func(attempt int) error {
		if c.observer != nil {
			return c.observer(ctx, key, attempt)
		}
		return nil
	})
}

func retryTransfer(ctx context.Context, operation func() error, before func(int) error) error {
	for attempt := 0; ; attempt++ {
		if before != nil {
			if err := before(attempt + 1); err != nil {
				return err
			}
		}
		err := operation()
		if err == nil {
			return nil
		}
		var status *TransferHTTPError
		var network net.Error
		retryable := errors.Is(err, io.ErrUnexpectedEOF) || errors.As(err, &network)
		if errors.As(err, &status) {
			retryable = status.retryable()
		}
		if attempt == 2 || !retryable {
			return err
		}
		delay := time.Second
		if attempt == 1 {
			delay = 5 * time.Second
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}
	}
}
func (c *TransferClient) request(ctx context.Context, endpoint Endpoint, method, resource string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, endpoint.ConnectBaseURL+resource, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	if method == http.MethodPost {
		req.Header.Set("Content-Type", "application/json")
	}
	client := c.session.clientForEndpoint(endpoint)
	client.Timeout = 5 * time.Minute
	resp, err := c.session.do(req, client, c.principal)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		resp.Body.Close()
		return nil, &TransferHTTPError{Status: resp.StatusCode}
	}
	return resp, nil
}
func (c *TransferClient) Report(ctx context.Context, ref ExecutionRef, progress TransferProgress) (string, error) {
	resource := fmt.Sprintf("/apis/jobs/v2/workspaces/%s/jobs/%s/attempts/%s/steps/%s/tasks/%s", url.PathEscape(ref.Workspace), url.PathEscape(ref.Job), url.PathEscape(ref.Attempt), url.PathEscape(ref.Step), url.PathEscape(ref.Task))
	body, err := json.Marshal(struct {
		Status   string           `json:"status"`
		Transfer TransferProgress `json:"transfer"`
	}{"active", progress})
	if err != nil {
		return "", err
	}
	var action string
	err = RetryTransfer(ctx, func() error {
		req, err := http.NewRequestWithContext(ctx, http.MethodPut, c.jobs.ConnectBaseURL+resource, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		client := c.session.clientForEndpoint(c.jobs)
		client.Timeout = 30 * time.Second
		resp, err := c.session.do(req, client, c.principal)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return &TransferHTTPError{Status: resp.StatusCode, RetryConflict: true}
		}
		var result struct {
			Action string `json:"transfer_action"`
		}
		if err = json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return err
		}
		action = result.Action
		switch action {
		case "wait", "run", "stop", "upload_success", "upload_failure", "done", "abort":
			return nil
		default:
			return fmt.Errorf("unsupported Task transfer action %q", action)
		}
	})
	return action, err
}
func filesetRoot(workspace, id string) string {
	return "/apis/files/v2/workspaces/" + url.PathEscape(workspace) + "/filesets/" + url.PathEscape("id:"+id)
}
func (c *TransferClient) checkIdentity(ctx context.Context, workspace, id string) error {
	resp, err := c.request(ctx, c.files, http.MethodGet, "/apis/files/v2/workspaces/"+url.PathEscape(workspace)+"/filesets/by-id/"+url.PathEscape(id), nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	var result struct {
		ID        string `json:"id"`
		Workspace string `json:"workspace"`
	}
	if err = json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return err
	}
	if result.ID != id || result.Workspace != workspace {
		return fmt.Errorf("Fileset identity/workspace mismatch")
	}
	return nil
}
func validateLocation(entry FilesetPull) error {
	if entry.FilesetID == "" || !filepath.IsAbs(entry.Directory) || strings.ContainsRune(entry.Directory, 0) {
		return fmt.Errorf("Fileset ID and absolute directory are required")
	}
	if strings.HasPrefix(entry.Path, "/") || strings.ContainsAny(entry.Path, "\\\x00") {
		return fmt.Errorf("Fileset path must be relative")
	}
	for _, component := range strings.Split(entry.Path, "/") {
		if component == ".." {
			return fmt.Errorf("Fileset path escapes root")
		}
	}
	return nil
}
func safeTarget(root, relative string) (string, error) {
	if relative == "" || path.IsAbs(relative) || strings.ContainsAny(relative, "\\\x00") {
		return "", fmt.Errorf("invalid relative file path")
	}
	for _, part := range strings.Split(relative, "/") {
		if part == ".." {
			return "", fmt.Errorf("file path escapes selected directory")
		}
	}
	resolved, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", err
	}
	target := filepath.Join(resolved, filepath.FromSlash(relative))
	current := resolved
	for _, part := range strings.Split(filepath.FromSlash(relative), string(filepath.Separator)) {
		current = filepath.Join(current, part)
		info, err := os.Lstat(current)
		if err == nil && info.Mode()&os.ModeSymlink != 0 {
			return "", fmt.Errorf("symlink in transfer destination")
		}
		if err != nil && !os.IsNotExist(err) {
			return "", err
		}
	}
	return target, nil
}
func (c *TransferClient) Pull(ctx context.Context, workspace string, entries []FilesetPull) error {
	for index, entry := range entries {
		if err := validateLocation(entry); err != nil {
			return err
		}
		err := c.retry(ctx, fmt.Sprintf("pull/%d", index), func() error {
			if err := c.checkIdentity(ctx, workspace, entry.FilesetID); err != nil {
				return err
			}
			if err := os.MkdirAll(entry.Directory, 0755); err != nil {
				return err
			}
			resource := filesetRoot(workspace, entry.FilesetID) + "/files"
			prefix := strings.Trim(entry.Path, "/")
			if prefix != "" {
				resource += "?path=" + url.QueryEscape(prefix)
			}
			resp, err := c.request(ctx, c.files, http.MethodGet, resource, nil)
			if err != nil {
				return err
			}
			var list struct {
				Data []struct {
					Path string `json:"path"`
				} `json:"data"`
			}
			err = json.NewDecoder(resp.Body).Decode(&list)
			resp.Body.Close()
			if err != nil {
				return err
			}
			for _, item := range list.Data {
				relative := item.Path
				if prefix != "" {
					if !strings.HasPrefix(relative, prefix+"/") {
						continue
					}
					relative = strings.TrimPrefix(relative, prefix+"/")
				}
				target, err := safeTarget(entry.Directory, relative)
				if err != nil {
					return err
				}
				if err = os.MkdirAll(filepath.Dir(target), 0755); err != nil {
					return err
				}
				fileResp, err := c.request(ctx, c.files, http.MethodGet, filesetRoot(workspace, entry.FilesetID)+"/-/"+escapeFilePath(item.Path), nil)
				if err != nil {
					return err
				}
				temp, err := os.CreateTemp(filepath.Dir(target), ".fileset-*")
				if err != nil {
					fileResp.Body.Close()
					return err
				}
				_, copyErr := io.Copy(temp, fileResp.Body)
				fileResp.Body.Close()
				closeErr := temp.Close()
				if copyErr == nil {
					copyErr = closeErr
				}
				if copyErr == nil {
					copyErr = os.Rename(temp.Name(), target)
				}
				if copyErr != nil {
					os.Remove(temp.Name())
					return copyErr
				}
			}
			return nil
		})
		if err != nil {
			return fmt.Errorf("pull Fileset %s: %w", entry.FilesetID, err)
		}
	}
	return nil
}
func escapeFilePath(value string) string {
	parts := strings.Split(value, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}
func (c *TransferClient) uploadDirectory(ctx context.Context, workspace string, entry FilesetPull) error {
	if err := validateLocation(entry); err != nil {
		return err
	}
	if err := c.checkIdentity(ctx, workspace, entry.FilesetID); err != nil {
		return err
	}
	rootInfo, err := os.Lstat(entry.Directory)
	if err != nil {
		return err
	}
	if !rootInfo.IsDir() {
		return fmt.Errorf("upload source must be a directory")
	}
	return filepath.WalkDir(entry.Directory, func(filename string, item fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if item.IsDir() {
			return nil
		}
		info, err := item.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("unsupported upload file type")
		}
		relative, err := filepath.Rel(entry.Directory, filename)
		if err != nil {
			return err
		}
		file, err := os.Open(filename)
		if err != nil {
			return err
		}
		defer file.Close()
		remote := path.Join(entry.Path, filepath.ToSlash(relative))
		resp, err := c.request(ctx, c.files, http.MethodPut, filesetRoot(workspace, entry.FilesetID)+"/-/"+escapeFilePath(remote), file)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		_, err = io.Copy(io.Discard, resp.Body)
		return err
	})
}
func (c *TransferClient) Upload(ctx context.Context, workspace, outcome string, entries []FilesetPush) []UploadResult {
	results := []UploadResult{}
	indexes := map[string]int{}
	for entryIndex, entry := range entries {
		if entry.When != "" && entry.When != "always" && entry.When != outcome {
			continue
		}
		index, ok := indexes[entry.FilesetID]
		if !ok {
			index = len(results)
			indexes[entry.FilesetID] = index
			results = append(results, UploadResult{FilesetID: entry.FilesetID, Success: true})
		}
		err := c.retry(ctx, fmt.Sprintf("push/%d", entryIndex), func() error { return c.uploadDirectory(ctx, workspace, entry.FilesetPull) })
		if err != nil {
			results[index].Success = false
			results[index].Error = err.Error()
		}
	}
	return results
}
