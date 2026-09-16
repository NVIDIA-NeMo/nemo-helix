// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/NVIDIA-NeMo/nemo-helix/services/core/jobs/jobs-launcher/nhxclient"
	"go.opentelemetry.io/contrib/bridges/otelslog"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlplog/otlploghttp"
	"go.opentelemetry.io/otel/exporters/stdout/stdoutlog"
	"go.opentelemetry.io/otel/log/global"
	"go.opentelemetry.io/otel/sdk/log"
	"go.opentelemetry.io/otel/sdk/resource"
)

const (
	name                    = "nhx.nvidia.com/nemo-helix/jobs-launcher"
	NEMO_JOB_WORKSPACE      = "NEMO_JOB_WORKSPACE"
	NEMO_JOB_ID_ENV         = "NEMO_JOB_ID"
	NEMO_JOB_ATTEMPT_ID_ENV = "NEMO_JOB_ATTEMPT_ID"
	NEMO_JOB_STEP_NAME_ENV  = "NEMO_JOB_STEP"
	NEMO_JOB_TASK_ID_ENV    = "NEMO_JOB_TASK"
	serviceJobsPrincipal    = "service:jobs"

	launcherOTLPLogsEndpointEnv = "NHX_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT"
	otlpHTTPLogExportTimeout    = 10 * time.Second
)

var (
	workspaceID = os.Getenv(NEMO_JOB_WORKSPACE)
	jobID       = os.Getenv(NEMO_JOB_ID_ENV)
	attemptID   = os.Getenv(NEMO_JOB_ATTEMPT_ID_ENV)
	step        = os.Getenv(NEMO_JOB_STEP_NAME_ENV)
	taskID      = os.Getenv(NEMO_JOB_TASK_ID_ENV)
)

// setupOTELSDK bootstraps the OpenTelemetry pipeline.
// If it does not return an error, make sure to call shutdown for proper cleanup.
// Returns the shutdown function, logger provider, and any error encountered.
func setupOTELSDK(ctx context.Context) (func(context.Context) error, *log.LoggerProvider, error) {
	var shutdownFuncs []func(context.Context) error

	// shutdown calls cleanup functions registered via shutdownFuncs.
	// The errors from the calls are joined.
	// Each registered cleanup will be invoked once.
	shutdown := func(ctx context.Context) error {
		var err error
		for _, fn := range shutdownFuncs {
			err = errors.Join(err, fn(ctx))
		}
		shutdownFuncs = nil
		return err
	}

	// Create a resource to include all OTEL attributes.
	res, err := resource.Merge(
		resource.Default(),
		// Ensure our job metadata is included in the resource.
		resource.NewSchemaless(
			attribute.String("workspace", workspaceID),
			attribute.String("job", jobID),
			attribute.String("job_attempt", attemptID),
			attribute.String("job_step", step),
			attribute.String("job_task", taskID),
		),
	)
	if err != nil {
		return shutdown, nil, err
	}

	// handleErr calls shutdown for cleanup and makes sure that all errors are returned.
	handleErr := func(inErr error) {
		err = errors.Join(inErr, shutdown(ctx))
	}

	// Set up logger provider.
	loggerProvider, err := newLoggerProvider(ctx, res)
	if err != nil {
		handleErr(err)
		return shutdown, nil, err
	}
	shutdownFuncs = append(shutdownFuncs, loggerProvider.Shutdown)
	global.SetLoggerProvider(loggerProvider)

	// Set default slog provider bridge.
	slog.SetDefault(otelslog.NewLogger(name, otelslog.WithLoggerProvider(loggerProvider)))

	return shutdown, loggerProvider, err
}

// newLoggerProvider creates a new OTEL logger provider with the given resource.
func newLoggerProvider(ctx context.Context, res *resource.Resource) (*log.LoggerProvider, error) {
	logExporter, err := newLogExporter(ctx)
	if err != nil {
		return nil, err
	}

	loggerProvider := log.NewLoggerProvider(
		log.WithProcessor(log.NewBatchProcessor(logExporter)),
		log.WithResource(res),
	)
	return loggerProvider, nil
}

func newLogExporter(ctx context.Context) (log.Exporter, error) {
	endpointURL := strings.TrimSpace(os.Getenv(launcherOTLPLogsEndpointEnv))
	if endpointURL == "" {
		return stdoutlog.New()
	}
	return newLauncherOTLPLogExporter(ctx, endpointURL)
}

func newLauncherOTLPLogExporter(ctx context.Context, endpointURL string) (log.Exporter, error) {
	authConfig, err := newLauncherOTLPLogAuthConfig(ctx)
	if err != nil {
		return nil, err
	}
	if _, err := authConfig.source.Headers(ctx); err != nil {
		return nil, fmt.Errorf("configure %s for OTLP logs: %w", authConfig.description, err)
	}
	logOTLPLogAuthMechanism(authConfig.mechanism, authConfig.reason)

	httpClient := launcherOTLPHTTPClient()
	if httpClient == nil {
		httpClient = &http.Client{Timeout: otlpHTTPLogExportTimeout}
	}
	httpClient.Transport = &authHeadersTransport{
		source:          authConfig.source,
		authDescription: authConfig.description,
		base:            httpClient.Transport,
	}

	options := []otlploghttp.Option{
		otlploghttp.WithEndpointURL(endpointURL),
		// Platform job log upload is intentionally independent from OTEL_* header
		// env vars. User workloads may use OTEL_EXPORTER_OTLP_LOGS_HEADERS for
		// third-party telemetry; application logs construct platform auth headers in Go.
		// Passing explicit empty headers prevents the upstream exporter from
		// applying OTEL_EXPORTER_OTLP_LOGS_HEADERS here.
		otlploghttp.WithHeaders(map[string]string{}),
		otlploghttp.WithHTTPClient(httpClient),
	}

	return otlpHTTPLogExporter(ctx, options...)
}

type launcherOTLPLogAuthConfig struct {
	source      requestHeaderSource
	description string
	mechanism   string
	reason      string
}

func newLauncherOTLPLogAuthConfig(ctx context.Context) (launcherOTLPLogAuthConfig, error) {
	if os.Getenv(workloadIdentityTokenFileEnv) != "" {
		// True auth: exchange the workload subject token and send the resulting bearer token.
		tokenSource, err := newOTLPLogWorkloadAuthTokenSource(ctx)
		if err != nil {
			return launcherOTLPLogAuthConfig{}, fmt.Errorf("configure workload identity auth for OTLP logs: %w", err)
		}
		return launcherOTLPLogAuthConfig{
			source:      tokenSource,
			description: "workload identity auth",
			mechanism:   "workload_identity_token_exchange",
			reason:      "workload_token_file_configured",
		}, nil
	}

	// Service identity default: when workload identity is not enabled, identify the
	// launcher as service:jobs without touching user OTEL_* headers.
	return launcherOTLPLogAuthConfig{
		source:      serviceIdentityPrincipalHeaderSource{},
		description: "service identity principal headers",
		mechanism:   "service_identity_principal_headers",
		reason:      "workload_token_file_not_configured",
	}, nil
}

type requestHeaderSource interface {
	Headers(context.Context) (map[string]string, error)
}

type authHeadersTransport struct {
	source          requestHeaderSource
	authDescription string
	base            http.RoundTripper
}

func (t *authHeadersTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	headers, err := t.source.Headers(req.Context())
	if err != nil {
		return nil, fmt.Errorf("refresh %s for OTLP logs: %w", t.authDescription, err)
	}

	clonedReq := req.Clone(req.Context())
	clonedReq.Header = req.Header.Clone()
	for key, value := range headers {
		clonedReq.Header.Set(key, value)
	}
	return t.baseTransport().RoundTrip(clonedReq)
}

func (t *authHeadersTransport) baseTransport() http.RoundTripper {
	if t.base != nil {
		return t.base
	}
	return http.DefaultTransport
}

func logOTLPLogAuthMechanism(mechanism string, reason string) {
	slog.Info(
		"using OTLP log authentication",
		"auth_mechanism", mechanism,
		"reason", reason,
	)
}

type serviceIdentityPrincipalHeaderSource struct{}

func (serviceIdentityPrincipalHeaderSource) Headers(context.Context) (map[string]string, error) {
	return map[string]string{"X-NHX-Principal-Id": serviceJobsPrincipal}, nil
}

func otlpHTTPLogExporter(ctx context.Context, opts ...otlploghttp.Option) (log.Exporter, error) {
	return otlploghttp.New(ctx, opts...)
}

func launcherOTLPHTTPClient() *http.Client {
	endpoint, err := nhxclient.ResolvePlatformEndpointFromEnv()
	if err != nil || endpoint.Transport != nhxclient.TransportUDS {
		return nil
	}
	return endpoint.HTTPClient()
}
