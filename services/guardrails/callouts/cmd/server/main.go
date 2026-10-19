// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/config"
	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/guardrails"
	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/server"
	telemetry "github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/telemetry"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	configPath := os.Getenv("CONFIG_PATH")
	if configPath == "" {
		configPath = "config.yaml"
	}
	cfg, err := config.Load(configPath)
	if err != nil {
		log.Fatalf("loading config: %v", err)
	}

	telemetryEnabled := cfg.Telemetry.Enabled
	var otelMgr *telemetry.OtelManager
	if telemetryEnabled {
		log.Println("Setting up OpenTelemetry for service: ", cfg.Telemetry.ServiceName, " using endpoint: ", cfg.Telemetry.OTLPEndpoint)

		otelMgr, err = telemetry.NewOtelManager(ctx, telemetry.Options{
			ServiceName:          cfg.Telemetry.ServiceName,
			TracingEnabled:       cfg.Telemetry.TracingEnabled,
			MetricsEnabled:       cfg.Telemetry.MetricsEnabled,
			TracingStdoutEnabled: cfg.Telemetry.TracingStdoutEnabled,
			MetricsStdoutEnabled: cfg.Telemetry.MetricsStdoutEnabled,
			TracingPeriod:        time.Minute,
			MetricsPeriod:        time.Minute,
			OTLPEndpoint:         cfg.Telemetry.OTLPEndpoint,
			OTLPInsecure:         cfg.Telemetry.OTLPInsecure,
		})
		if err != nil {
			log.Fatalf("failed to initialize OpenTelemetry: %v", err)
		}
		defer otelMgr.Shutdown(ctx)
	} else {
		log.Println("Telemetry disabled by environment variable (OTEL_SDK_DISABLED=true)")
	}

	guardrailsClient, err := guardrails.NewClient(&cfg.Guardrails)
	if err != nil {
		log.Fatalf("Failed to initialize Guardrails client: %v", err)
	}
	guardrailsExtProc := server.NewExternalProcessor(cfg, guardrailsClient)
	calloutServer, err := server.NewCalloutServer(cfg.GRPC)
	if err != nil {
		log.Fatalf("Failed to create CalloutServer: %v", err)
	}
	_, err = calloutServer.StartGRPC(guardrailsExtProc)
	if err != nil {
		log.Fatalf("Failed to start gRPC server: %v", err)
	}

	// Handle signals
	<-ctx.Done()
	if err := calloutServer.Stop(context.Background()); err != nil && !errors.Is(err, context.Canceled) {
		log.Printf("Server failed to gracefully shutdown: %v", err)
	} else {
		log.Println("Server gracefully shutdown")
	}
}
