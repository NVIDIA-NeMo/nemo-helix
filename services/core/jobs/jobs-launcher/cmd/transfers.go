// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/NVIDIA-NeMo/nemo-helix/services/core/jobs/jobs-launcher/nhxclient"
)

type taskTransfers struct {
	attempts  map[string]int
	runtimeID string
	client    *nhxclient.TransferClient
	ref       nhxclient.ExecutionRef
	config    nhxclient.StepFilesets
	leader    bool
	stopped   atomic.Bool
}

func transfersFromEnv(ctx context.Context) (*taskTransfers, error) {
	raw := os.Getenv(nhxclient.TransfersEnv)
	if raw == "" {
		return nil, nil
	}
	var config nhxclient.StepFilesets
	decoder := json.NewDecoder(strings.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&config); err != nil {
		return nil, fmt.Errorf("decode Step Filesets: %w", err)
	}
	if len(config.Pull) == 0 && len(config.Push) == 0 {
		return nil, nil
	}
	ref := nhxclient.ExecutionRef{Workspace: os.Getenv("NEMO_JOB_WORKSPACE"), Job: os.Getenv("NEMO_JOB_UID"), Attempt: os.Getenv("NEMO_JOB_ATTEMPT_ID"), Step: os.Getenv("NEMO_JOB_STEP"), Task: os.Getenv("NEMO_JOB_TASK")}
	if ref.Job == "" {
		ref.Job = os.Getenv("NEMO_JOB_ID")
	}
	if ref.Workspace == "" || ref.Job == "" || ref.Attempt == "" || ref.Step == "" || ref.Task == "" {
		return nil, fmt.Errorf("Fileset transfers require concrete Job/Attempt/Step/Task context")
	}
	jobs, err := nhxclient.ResolveServiceEndpointFromEnv("jobs")
	if err != nil {
		return nil, err
	}
	files, err := nhxclient.ResolveServiceEndpointFromEnv("files")
	if err != nil {
		return nil, err
	}
	return &taskTransfers{attempts: map[string]int{}, runtimeID: rand.Text(), client: launcherSession(ctx).TransferClient(jobs, files, nhxclient.PrincipalFromEnv()), ref: ref, config: config, leader: os.Getenv("NEMO_JOB_TRANSFER_LEADER") == "true"}, nil
}
func (t *taskTransfers) progress(phase string, exitCode *int) nhxclient.TransferProgress {
	return nhxclient.TransferProgress{Attempts: t.attempts, RuntimeID: t.runtimeID, Phase: phase, ExitCode: exitCode, Leader: t.leader, Stopped: t.stopped.Load()}
}
func (t *taskTransfers) wait(ctx context.Context, progress nhxclient.TransferProgress) (string, error) {
	for {
		action, err := t.client.Report(ctx, t.ref, progress)
		if err != nil {
			return "", err
		}
		if action != "wait" {
			return action, nil
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
}
func (t *taskTransfers) observe(phase string, code *int) {
	t.client.ObserveAttempts(func(ctx context.Context, key string, attempt int) error {
		t.attempts[key] = attempt
		directive, err := t.client.Report(ctx, t.ref, t.progress(phase, code))
		if err != nil {
			return err
		}
		if directive == "abort" {
			return fmt.Errorf("Step transfers aborted")
		}
		return nil
	})
}

func (t *taskTransfers) prepare(ctx context.Context) error {
	t.observe("pulling", nil)
	if _, err := t.client.Report(ctx, t.ref, t.progress("pulling", nil)); err != nil {
		return err
	}
	if err := t.client.Pull(ctx, t.ref.Workspace, t.config.Pull); err != nil {
		progress := t.progress("pull_failed", nil)
		progress.Error = err.Error()
		_, reportErr := t.client.Report(ctx, t.ref, progress)
		return errors.Join(err, reportErr)
	}
	action, err := t.wait(ctx, t.progress("ready", nil))
	if err != nil {
		return err
	}
	if action != "run" {
		return fmt.Errorf("Step input staging aborted (%s)", action)
	}
	return nil
}
func (t *taskTransfers) monitor(ctx context.Context, command *exec.Cmd) {
	for {
		action, err := t.client.Report(ctx, t.ref, t.progress("running", nil))
		if err == nil && (action == "stop" || action == "abort") {
			t.stopped.Store(true)
			syscall.Kill(-command.Process.Pid, syscall.SIGTERM) // nolint:errcheck
			select {
			case <-ctx.Done():
				return
			case <-time.After(5 * time.Second):
				syscall.Kill(-command.Process.Pid, syscall.SIGKILL)
			} // nolint:errcheck
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(250 * time.Millisecond):
		}
	}
}
func (t *taskTransfers) finish(ctx context.Context, code int) (bool, error) {
	progress := t.progress("finished", &code)
	action, err := t.wait(ctx, progress)
	if err != nil {
		return false, err
	}
	if action != "upload_success" && action != "upload_failure" {
		return false, fmt.Errorf("Step publication aborted (%s)", action)
	}
	outcome := strings.TrimPrefix(action, "upload_")
	t.observe("finished", &code)
	progress.Uploads = t.client.Upload(ctx, t.ref.Workspace, outcome, t.config.Push)
	progress.Phase = "uploaded"
	for _, result := range progress.Uploads {
		if !result.Success {
			progress.Phase = "upload_failed"
			progress.Error = result.Error
		}
	}
	action, err = t.wait(ctx, progress)
	if err != nil {
		return false, err
	}
	if action != "done" {
		return false, fmt.Errorf("Step publication aborted (%s)", action)
	}
	return outcome == "success" && progress.Phase == "uploaded", nil
}

func runExec(ctx context.Context, args []string, stdinReader io.Reader) (int, error) {
	ctx = withLauncherSession(ctx)
	transfers, err := transfersFromEnv(ctx)
	if err != nil {
		return 1, err
	}
	if transfers == nil {
		if requireFilesetTransfers {
			return 1, fmt.Errorf("Fileset transfer configuration is required")
		}
		return runCommand(ctx, args, stdinReader, func(*exec.Cmd) {})
	}
	if err = transfers.prepare(ctx); err != nil {
		return 1, err
	}
	monitorCtx, cancel := context.WithCancel(ctx)
	var monitor sync.WaitGroup
	code, commandErr := runCommand(ctx, args, stdinReader, func(command *exec.Cmd) {
		monitor.Add(1)
		go func() { defer monitor.Done(); transfers.monitor(monitorCtx, command) }()
	})
	cancel()
	monitor.Wait()
	if ctx.Err() != nil {
		return code, errors.Join(commandErr, ctx.Err())
	}
	// Exit 75 is the existing workload pause handshake. Preserve it for the
	// backend without treating retained work as a completed workload.
	if code == 75 {
		return code, commandErr
	}
	success, err := transfers.finish(ctx, code)
	if err != nil {
		return 1, errors.Join(commandErr, err)
	}
	if success {
		return 0, nil
	}
	if code == 0 || transfers.stopped.Load() {
		code = 1
	}
	return code, errors.Join(commandErr, fmt.Errorf("Step workload or publication failed"))
}
