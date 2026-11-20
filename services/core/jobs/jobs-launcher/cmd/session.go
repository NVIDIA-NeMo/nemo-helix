// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cmd

import (
	"context"

	"github.com/NVIDIA-NeMo/nemo-helix/services/core/jobs/jobs-launcher/nhxclient"
)

type launcherSessionKey struct{}

func withLauncherSession(ctx context.Context) context.Context {
	if _, ok := ctx.Value(launcherSessionKey{}).(*nhxclient.Session); ok {
		return ctx
	}
	return context.WithValue(ctx, launcherSessionKey{}, nhxclient.NewSessionFromEnv())
}
func launcherSession(ctx context.Context) *nhxclient.Session {
	return withLauncherSession(ctx).Value(launcherSessionKey{}).(*nhxclient.Session)
}
