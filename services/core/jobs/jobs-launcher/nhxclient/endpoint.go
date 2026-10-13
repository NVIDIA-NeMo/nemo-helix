// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package nhxclient

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
)

const UDSBaseURL = "http://nemo-helix.local"

type Transport string

const (
	TransportTCP Transport = "tcp"
	TransportUDS Transport = "uds"
)

type Endpoint struct {
	ConnectBaseURL string
	SocketPath     string
	Transport      Transport
}

func ParseEndpoint(raw string) (Endpoint, error) {
	if raw == "" {
		return Endpoint{}, fmt.Errorf("platform endpoint URL is not configured")
	}
	if strings.HasPrefix(raw, "http://") || strings.HasPrefix(raw, "https://") {
		parsed, err := url.Parse(raw)
		if err != nil || parsed.Host == "" {
			return Endpoint{}, fmt.Errorf("invalid platform endpoint URL %q", raw)
		}
		return Endpoint{
			ConnectBaseURL: strings.TrimRight(raw, "/"),
			Transport:      TransportTCP,
		}, nil
	}
	if strings.HasPrefix(raw, "unix://") {
		socketPath := strings.TrimPrefix(raw, "unix://")
		if !strings.HasPrefix(socketPath, "/") {
			return Endpoint{}, fmt.Errorf("UDS endpoint must use an absolute socket path, got %q", raw)
		}
		return Endpoint{
			ConnectBaseURL: UDSBaseURL,
			SocketPath:     socketPath,
			Transport:      TransportUDS,
		}, nil
	}
	if strings.HasPrefix(raw, "/") {
		return Endpoint{}, fmt.Errorf("raw socket paths are not valid endpoint URLs; use unix://%s", raw)
	}
	return Endpoint{}, fmt.Errorf("unsupported platform endpoint URL %q; expected http://, https://, or unix://", raw)
}

func ResolveHelixEndpointFromEnv() (Endpoint, error) {
	return ParseEndpoint(os.Getenv("NHX_BASE_URL"))
}

func ResolveServiceEndpointFromEnv(service string) (Endpoint, error) {
	if serviceEnv := os.Getenv(serviceURLEnvName(service)); serviceEnv != "" {
		return ParseEndpoint(serviceEnv)
	}
	return ResolveHelixEndpointFromEnv()
}

func serviceURLEnvName(service string) string {
	normalized := strings.ToUpper(strings.ReplaceAll(service, "-", "_"))
	return "NHX_" + normalized + "_URL"
}

func (e Endpoint) HTTPClient() *http.Client {
	if e.Transport != TransportUDS {
		return http.DefaultClient
	}
	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", e.SocketPath)
		},
	}
	return &http.Client{Transport: transport}
}
