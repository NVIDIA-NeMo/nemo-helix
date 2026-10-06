// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/NVIDIA-NeMo/nemo-helix/services/core/jobs/jobs-launcher/nhxclient"
	"github.com/spf13/cobra"
)

const secretFetchTimeout = 30 * time.Second

const stepConfigFetchTimeout = 30 * time.Second

// fetchStepConfig makes the launcher download the step config from the jobs API
// before starting the workload. Runtimes with no way to place a file in the
// container before it starts (OpenShell) set it instead of pre-writing the file.
var fetchStepConfig bool

var runCmd = &cobra.Command{
	Use:   "run <command> [args...]",
	Short: "Run a subprocess and tail its logs",
	Args:  cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		exitCode, err := runExecWithStdin(args)
		if err != nil {
			logger.Printf("Error: %v\n", err)
		}
		// Stash exit code instead of calling os.Exit here. os.Exit skips
		// deferred functions, including the OTEL shutdown in runExecWithStdin
		// that flushes remaining log batches. Execute() calls os.Exit after
		// cobra returns and all defers have run.
		launcherExitCode = exitCode
	},
}

// launcherExitCode holds the subprocess exit code. Set by the run command,
// read by Execute() to exit after defers (including OTEL shutdown) complete.
var launcherExitCode int

func init() {
	runCmd.Flags().BoolVar(
		&fetchStepConfig,
		"fetch-step-config",
		false,
		"Fetch the step config from the jobs API and write it to NEMO_JOB_STEP_CONFIG_FILE_PATH before starting the command",
	)
	rootCmd.AddCommand(runCmd)
}

// writeStepConfigFromAPI fetches this job step's config and writes it where the
// workload expects it. The step is identified by the NEMO_JOB_* env vars the
// jobs controller sets on every job.
func writeStepConfigFromAPI() error {
	required := map[string]string{}
	for _, name := range []string{"NEMO_JOB_WORKSPACE", "NEMO_JOB_ID", "NEMO_JOB_STEP", "NEMO_JOB_STEP_CONFIG_FILE_PATH"} {
		value := os.Getenv(name)
		if value == "" {
			return fmt.Errorf("%s is required with --fetch-step-config", name)
		}
		required[name] = value
	}

	endpoint, err := nhxclient.ResolveServiceEndpointFromEnv("jobs")
	if err != nil {
		return fmt.Errorf("jobs endpoint is not configured (NHX_JOBS_URL or NHX_BASE_URL): %w", err)
	}
	httpClient := *endpoint.HTTPClient()
	httpClient.Timeout = stepConfigFetchTimeout
	client := nhxclient.NewJobStepClientWithHTTPClient(endpoint.ConnectBaseURL, nhxclient.PrincipalFromEnv(), &httpClient)

	workspace, job, step := required["NEMO_JOB_WORKSPACE"], required["NEMO_JOB_ID"], required["NEMO_JOB_STEP"]
	logger.Printf("Fetching step config for %s/%s/%s...\n", workspace, job, step)
	config, err := client.GetJobStepConfig(workspace, job, step)
	if err != nil {
		return fmt.Errorf("failed to fetch step config for %s/%s/%s: %w", workspace, job, step, err)
	}

	path := required["NEMO_JOB_STEP_CONFIG_FILE_PATH"]
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("failed to create step config directory: %w", err)
	}
	if err := os.WriteFile(path, config, 0o600); err != nil {
		return fmt.Errorf("failed to write step config: %w", err)
	}
	logger.Printf("Wrote step config to %s\n", path)
	return nil
}

// secretReference represents a mapping from an environment variable to a secret
type secretReference struct {
	envVarName string
	workspace  string
	secretName string
}

// parseSecretReferences parses the NEMO_JOB_SECRETS environment variable
// Format: ENV_VAR=workspace/secret_name,ENV_VAR2=workspace/secret_name2
// Returns a list of secret references
func parseSecretReferences(secretsEnv string) ([]secretReference, error) {
	if secretsEnv == "" {
		return nil, nil
	}

	pairs := strings.Split(secretsEnv, ",")
	result := make([]secretReference, 0, len(pairs))

	for _, pair := range pairs {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}

		// Split by '=' to get env var name and secret reference
		eqParts := strings.Split(pair, "=")
		if len(eqParts) != 2 {
			return nil, fmt.Errorf("invalid secret reference format: %s (expected ENV_VAR=workspace/secret_name)", pair)
		}

		envVarName := strings.TrimSpace(eqParts[0])
		secretRef := strings.TrimSpace(eqParts[1])

		if envVarName == "" {
			return nil, fmt.Errorf("invalid secret reference: %s (environment variable name cannot be empty)", pair)
		}

		// Split secret reference by '/' to get workspace and secret name
		parts := strings.Split(secretRef, "/")
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid secret reference format: %s (expected workspace/secret_name)", secretRef)
		}

		workspace := strings.TrimSpace(parts[0])
		secretName := strings.TrimSpace(parts[1])

		if workspace == "" || secretName == "" {
			return nil, fmt.Errorf("invalid secret reference: %s (workspace and secret_name cannot be empty)", secretRef)
		}

		result = append(result, secretReference{
			envVarName: envVarName,
			workspace:  workspace,
			secretName: secretName,
		})
	}

	return result, nil
}

// fetchSecrets retrieves secrets using the NeMo Helix API client and returns them as environment variables
func fetchSecrets(apiBaseURL string, principal *nhxclient.Principal, secretRefs []secretReference) ([]string, error) {
	return fetchSecretsWithClient(nhxclient.NewSecretClient(apiBaseURL, principal), secretRefs)
}

func fetchSecretsWithEndpoint(endpoint nhxclient.Endpoint, principal *nhxclient.Principal, secretRefs []secretReference) ([]string, error) {
	return fetchSecretsWithClient(
		nhxclient.NewSecretClientWithHTTPClient(endpoint.ConnectBaseURL, principal, secretEndpointHTTPClient(endpoint)),
		secretRefs,
	)
}

func secretEndpointHTTPClient(endpoint nhxclient.Endpoint) *http.Client {
	httpClient := *endpoint.HTTPClient()
	httpClient.Timeout = secretFetchTimeout
	return &httpClient
}

func fetchSecretsWithClient(client nhxclient.SecretClient, secretRefs []secretReference) ([]string, error) {
	if len(secretRefs) == 0 {
		return nil, nil
	}

	envVars := make([]string, 0, len(secretRefs))

	for _, ref := range secretRefs {
		logger.Printf("Fetching secret %s from workspace %s...\n", ref.secretName, ref.workspace)
		secret, err := client.GetSecret(ref.workspace, ref.secretName)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch secret %s/%s: %w", ref.workspace, ref.secretName, err)
		}

		// Use the specified environment variable name
		envVar := fmt.Sprintf("%s=%s", ref.envVarName, secret.Value)
		envVars = append(envVars, envVar)
		logger.Printf("Successfully fetched secret %s and mapped to %s\n", ref.secretName, ref.envVarName)
	}

	return envVars, nil
}

func workloadEnvFromParent() []string {
	env := os.Environ()
	filtered := make([]string, 0, len(env))
	for _, item := range env {
		key, _, _ := strings.Cut(item, "=")
		if strings.HasPrefix(key, "NHX_JOB_LAUNCHER_") {
			continue
		}
		filtered = append(filtered, item)
	}
	return filtered
}

// runExecWithStdin sets up OTEL and runs the specified command with stdin
func runExecWithStdin(args []string) (exitCode int, err error) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	otelShutdown, _, err := setupOTELSDK(ctx)
	if err != nil {
		return 1, err
	}
	// Handle shutdown properly so nothing leaks.
	defer func() {
		err = errors.Join(err, otelShutdown(context.Background()))
	}()

	return runExec(args, os.Stdin)
}

// runExec runs the specified command with arguments, injecting secrets as environment variables if specified
func runExec(args []string, stdinReader io.Reader) (int, error) {
	// Command and arguments
	cmdName := args[0]
	cmdArgs := []string{}
	if len(args) > 1 {
		cmdArgs = args[1:]
	}

	// Prepare the subprocess
	cmd := exec.Command(cmdName, cmdArgs...)

	// Inherit parent environment, excluding launcher-private control variables.
	cmd.Env = workloadEnvFromParent()

	if fetchStepConfig {
		if err := writeStepConfigFromAPI(); err != nil {
			logger.Printf("Error: %v\n", err)
			return 1, err
		}
	}

	// Parse and fetch secrets if NEMO_JOB_SECRETS is set
	secretsEnv := os.Getenv("NEMO_JOB_SECRETS")
	if secretsEnv != "" {
		secretRefs, err := parseSecretReferences(secretsEnv)
		if err != nil {
			logger.Printf("Error parsing NEMO_JOB_SECRETS: %v\n", err)
			return 1, err
		}

		if len(secretRefs) > 0 {
			secretEndpoint, err := nhxclient.ResolveServiceEndpointFromEnv("secrets")
			if err != nil {
				logger.Printf("Error: NHX_SECRETS_URL or NHX_BASE_URL is required when NEMO_JOB_SECRETS is set: %v\n", err)
				return 1, fmt.Errorf("secrets endpoint is not configured: %w", err)
			}

			// Build auth context from NHX_PRINCIPAL JSON env var set by the jobs controller
			principal := nhxclient.PrincipalFromEnv()

			secretEnvVars, err := fetchSecretsWithEndpoint(secretEndpoint, principal, secretRefs)
			if err != nil {
				logger.Printf("Error fetching secrets: %v\n", err)
				return 1, err
			}

			// Add secret environment variables to subprocess
			cmd.Env = append(cmd.Env, secretEnvVars...)
			logger.Printf("Injected %d secret(s) as environment variables\n", len(secretEnvVars))
		}
	}

	// Set up process group so we can forward signals to the subprocess and its children
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}

	// Connect stdin from parent process or os.Stdin to subprocess
	if stdinReader != nil {
		cmd.Stdin = stdinReader
	}

	// Get stdout pipe
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		logger.Printf("Error creating stdout pipe: %v\n", err)
		return 1, err
	}

	// Optionally capture stderr as well
	stderr, err := cmd.StderrPipe()
	if err != nil {
		logger.Printf("Error creating stderr pipe: %v\n", err)
		return 1, err
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		logger.Printf("Error starting command: %v\n", err)
		return 1, err
	}

	// WaitGroup to ensure all output is processed before returning
	var wg sync.WaitGroup

	// Function to tail output
	tailOutput := func(reader io.Reader, level slog.Level) {
		defer wg.Done()
		scanner := bufio.NewScanner(reader)
		for scanner.Scan() {
			line := scanner.Text()
			fmt.Println(line)                                           // Print to console without any extra formatting, so it can be captured by stdout logging collectors
			slog.Log(context.Background(), logLevel(line, level), line) // Submit structured log to OTEL pipeline within the platform
		}
	}

	// Stream stdout and stderr concurrently
	wg.Add(2)
	go tailOutput(stdout, slog.LevelInfo)
	go tailOutput(stderr, slog.LevelError)

	// Log that we are launching to application
	logger.Printf("Running main process: %s %v\n", cmdName, cmdArgs)

	// Forward signals
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

	go func() {
		for sig := range signals {
			logger.Printf("Received signal: %s, forwarding to subprocess...\n", sig)
			syscall.Kill(-cmd.Process.Pid, sig.(syscall.Signal)) // nolint:errcheck
		}
	}()

	// Wait for all output to be read before calling cmd.Wait().
	// cmd.Wait() closes stdout/stderr pipes, so readers must finish first.
	// Once readers finish, all log records have been submitted to the OTEL
	// batch processor. The deferred otelShutdown in runExecWithStdin flushes
	// remaining batches before the process exits.
	wg.Wait()

	// Now that all output has been read, wait for the process to finish.
	err = cmd.Wait()

	exitCode := cmd.ProcessState.ExitCode()
	if err != nil {
		logger.Printf("Process exited with error: %v\n", err)
		return exitCode, err
	}

	logger.Printf("Process completed successfully.")
	return exitCode, nil
}

// logLevel preserves an application's explicit textual level even when the
// application writes all logs to stderr. Many Python logging configurations do
// that, so treating every stderr line as an error mislabels INFO and WARNING
// records in OpenTelemetry. Unprefixed stderr remains an error.
func logLevel(line string, fallback slog.Level) slog.Level {
	prefix := strings.TrimSpace(line)
	marker, rest := nextLogPrefix(prefix)
	if _, err := time.Parse("15:04:05", marker); err == nil {
		marker, _ = nextLogPrefix(strings.TrimSpace(rest))
	}

	switch marker {
	case "DEBUG":
		return slog.LevelDebug
	case "INFO":
		return slog.LevelInfo
	case "WARNING", "WARN":
		return slog.LevelWarn
	case "ERROR", "CRITICAL":
		return slog.LevelError
	default:
		return fallback
	}
}

func nextLogPrefix(line string) (string, string) {
	if !strings.HasPrefix(line, "[") {
		return "", line
	}
	end := strings.IndexByte(line, ']')
	if end < 0 {
		return "", line
	}
	return line[1:end], line[end+1:]
}
