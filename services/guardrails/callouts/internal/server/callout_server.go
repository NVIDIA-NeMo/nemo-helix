// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modifications: Added OpenTelemetry instrumentation.

package server

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"strings"
	"time"

	extproc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"

	"github.com/NVIDIA-NeMo/nemo-helix/services/guardrails/callouts/internal/config"
)

// CalloutServer represents a server that handles callouts.
type CalloutServer struct {
	Config config.GRPC
	Cert   tls.Certificate

	// internals
	listener net.Listener
	httpSrv  *http.Server
	grpcSrv  *grpc.Server
}

// NewCalloutServer creates a new CalloutServer with the given configuration.
func NewCalloutServer(config config.GRPC) (*CalloutServer, error) {
	var cert tls.Certificate
	var err error

	if config.TLS.Enabled && config.TLS.CertFile != "" && config.TLS.KeyFile != "" {
		cert, err = tls.LoadX509KeyPair(config.TLS.CertFile, config.TLS.KeyFile)
		if err != nil {
			log.Printf("Failed to load server certificate: %v", err)
			return nil, err
		}
	}

	return &CalloutServer{
		Config: config,
		Cert:   cert,
	}, nil
}

// newCombinedServer creates a single server to handle both gRPC and HTTP traffic.
func newCombinedServer(grpcServer *grpc.Server, httpHandler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Route gRPC requests to the gRPC server, and all other requests
		// to the httpHandler (which will handle the health checks).
		// gRPC-Go sets the Content-Type header to "application/grpc".
		if r.ProtoMajor == 2 && strings.HasPrefix(r.Header.Get("Content-Type"), "application/grpc") {
			grpcServer.ServeHTTP(w, r)
		} else {
			httpHandler.ServeHTTP(w, r)
		}
	})
}

// StartGRPC starts the gRPC server with the specified service.
func (s *CalloutServer) StartGRPC(service extproc.ExternalProcessorServer) (addr string, err error) {
	// Create gRPC server with OpenTelemetry interceptors
	if s.Config.TLS.Enabled {
		creds := credentials.NewServerTLSFromCert(&s.Cert)
		s.grpcSrv = grpc.NewServer(grpc.Creds(creds), grpc.StatsHandler(otelgrpc.NewServerHandler()))
	} else {
		s.grpcSrv = grpc.NewServer(grpc.StatsHandler(otelgrpc.NewServerHandler()))
	}
	extproc.RegisterExternalProcessorServer(s.grpcSrv, service)
	reflection.Register(s.grpcSrv)

	// It's best practice to register the gRPC Health checking protocol.
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(s.grpcSrv, healthServer)
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)

	// Create an explicit listener so we can bind :0 and return the port to clients like unit tests.
	s.listener, err = net.Listen("tcp", s.Config.Address)
	if err != nil {
		return "", err
	}
	addr = s.listener.Addr().String()

	go func() {
		// Serve in the background using the existing listener.
		if s.Config.TLS.Enabled {
			// Create a HTTP router for health checks and wrap with otelhttp
			// Note that this trick only works when we ServeTLS because it uses HTTP/2 protocol.
			mux := http.NewServeMux()
			mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				w.Write([]byte("OK"))
			})
			// Create the combined server that multiplexes the traffic
			combinedHandler := newCombinedServer(s.grpcSrv, mux)
			s.httpSrv = &http.Server{
				Handler: combinedHandler,
			}
			log.Printf("Serving secured gRPC & HTTP Server on %s", addr)
			if err := s.httpSrv.ServeTLS(s.listener, s.Config.TLS.CertFile, s.Config.TLS.KeyFile); err != nil && err != http.ErrServerClosed {
				log.Printf("Failed to serve : %v", err)
			}
		} else {
			log.Printf("Serving insecure gRPC on %s", addr)
			if err := s.grpcSrv.Serve(s.listener); err != nil && err != http.ErrServerClosed {
				log.Printf("Failed to serve : %v", err)
			}
		}
	}()
	return addr, nil
}

// Stop performs an ordered shutdown:
// 1) HTTP graceful shutdown -- skipped in our insecure gRPC setup.
// 2) gRPC graceful shutdown -- optional in our combined secured gRPC + HTTP/2 setup.
func (s *CalloutServer) Stop(ctx context.Context) error {
	var combinedErrs error
	if s.httpSrv != nil {
		_ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := s.httpSrv.Shutdown(_ctx); err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, http.ErrServerClosed) {
			combinedErrs = errors.Join(combinedErrs, fmt.Errorf("failed to gracefully shutdown http server: %w", err))
		}
	}
	// This is required for the insecure setup, where we don't have a http server.
	// IDEA: this blocks until all pending RPCs are finished -- should we wrap this in a goroutine?
	if s.grpcSrv != nil {
		s.grpcSrv.GracefulStop()
	}
	return combinedErrs
}
