// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getHelixHealthUrl, useHelixHealth } from '@studio/api/helixHealth';
import { HelixUnavailable } from '@studio/components/Layouts/HelixUnavailable';
import { Loading } from '@studio/components/Layouts/Loading';
import { ROUTES } from '@studio/constants/routes';
import type { ReactNode } from 'react';
import { useLocation } from 'react-router';

interface HelixGuardProps {
  children: ReactNode;
}

// The OIDC callback must finish even when the platform is down, otherwise the user is
// stuck in a redirect loop between the identity provider and a blocked Studio.
const UNGUARDED_PATHS: ReadonlySet<string> = new Set([ROUTES.auth.success]);

/**
 * Blocks the app behind a single platform readiness probe. Without it, a stopped
 * platform surfaces as dozens of unrelated per-query errors across the UI.
 */
export const HelixGuard = ({ children }: HelixGuardProps) => {
  const { pathname } = useLocation();
  const { data: status, isPending, isFetching, refetch } = useHelixHealth();

  if (UNGUARDED_PATHS.has(pathname) || status === 'ready') return <>{children}</>;
  if (isPending) return <Loading description="Connecting to NeMo Helix..." />;

  return (
    <HelixUnavailable
      status={status ?? 'unreachable'}
      healthUrl={getHelixHealthUrl()}
      onRetry={() => void refetch()}
      isRetrying={isFetching}
    />
  );
};
