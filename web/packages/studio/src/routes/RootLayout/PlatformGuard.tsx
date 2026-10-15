// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getPlatformHealthUrl, usePlatformHealth } from '@studio/api/platformHealth';
import { Loading } from '@studio/components/Layouts/Loading';
import { PlatformUnavailable } from '@studio/components/Layouts/PlatformUnavailable';
import { ROUTES } from '@studio/constants/routes';
import type { ReactNode } from 'react';
import { useLocation } from 'react-router';

interface PlatformGuardProps {
  children: ReactNode;
}

// The OIDC callback must finish even when the platform is down, otherwise the user is
// stuck in a redirect loop between the identity provider and a blocked Studio.
const UNGUARDED_PATHS: ReadonlySet<string> = new Set([ROUTES.auth.success]);

/**
 * Blocks the app only while platform liveness cannot be confirmed. Without it, a stopped
 * platform surfaces as dozens of unrelated per-query errors across the UI.
 */
export const PlatformGuard = ({ children }: PlatformGuardProps) => {
  const { pathname } = useLocation();
  const { data: status, isPending, isFetching, refetch } = usePlatformHealth();

  if (UNGUARDED_PATHS.has(pathname) || status === 'live') return <>{children}</>;
  if (isPending) return <Loading description="Connecting to NeMo Helix..." />;

  return (
    <PlatformUnavailable
      healthUrl={getPlatformHealthUrl()}
      onRetry={() => void refetch()}
      isRetrying={isFetching}
    />
  );
};
