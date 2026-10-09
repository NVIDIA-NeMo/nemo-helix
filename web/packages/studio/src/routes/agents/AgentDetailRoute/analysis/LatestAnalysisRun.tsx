// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { parseISOWithUTCFallback } from '@nemo/common/src/components/RelativeTime/util';
import { type BadgeStatus, StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { formatDurationMs } from '@nemo/common/src/utils/date';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { LatestAnalysisRun as LatestRun } from '@studio/api/useLatestAnalysisRun';
import { JOBS_ENABLED } from '@studio/constants/environment';
import { getWorkspaceJobDetailRoute } from '@studio/routes/utils';
import { type FC, useEffect, useState } from 'react';
import { useNavigate } from 'react-router';

const RUN_BADGES: Partial<Record<HelixJobStatus, { status: BadgeStatus; label: string }>> = {
  created: { status: 'pending', label: 'Queued' },
  pending: { status: 'pending', label: 'Queued' },
  active: { status: 'active', label: 'Running' },
  error: { status: 'error', label: 'Failed' },
};

const NOT_SUBMITTED = { status: 'unavailable', label: 'Not submitted' } as const;

const Elapsed: FC<{ since: string }> = ({ since }) => {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const seconds = Math.max(1, Math.floor((now - parseISOWithUTCFallback(since).getTime()) / 1000));
  return <>{formatDurationMs(seconds * 1000)}</>;
};

interface LatestAnalysisRunProps {
  workspace: string;
  run: LatestRun;
  isActive: boolean;
}

export const LatestAnalysisRun: FC<LatestAnalysisRunProps> = ({ workspace, run, isActive }) => {
  const navigate = useNavigate();
  const badge = run.submitted
    ? ((run.status && RUN_BADGES[run.status]) ?? { status: run.status, label: undefined })
    : NOT_SUBMITTED;

  return (
    <Stack gap="1" data-testid="latest-analysis-run">
      <Text kind="label/regular/sm" className="text-secondary">
        Latest run
      </Text>
      <Flex align="center" gap="2" wrap="wrap">
        <StatusBadge status={badge.status} label={badge.label} />
        <Text kind="body/regular/sm" className="text-secondary">
          {isActive && run.status === 'active' ? (
            <>
              for <Elapsed since={run.startedAt} />
            </>
          ) : (
            <>
              started <RelativeTime datetime={run.startedAt} />
            </>
          )}
        </Text>
        {JOBS_ENABLED && run.submitted ? (
          <Button
            kind="tertiary"
            size="small"
            onClick={() => navigate(getWorkspaceJobDetailRoute(workspace, run.name))}
          >
            View job
          </Button>
        ) : null}
      </Flex>
      {isActive ? (
        <Text kind="body/regular/xs" className="text-secondary">
          Analysis running. You can start another run when this one finishes.
        </Text>
      ) : null}
    </Stack>
  );
};
