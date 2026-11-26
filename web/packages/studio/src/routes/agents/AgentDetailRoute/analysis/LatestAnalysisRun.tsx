// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { type BadgeStatus, StatusBadge } from '@nemo/common/src/components/StatusBadge';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import { Button, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { LatestAnalysisRun as LatestRun } from '@studio/api/useLatestAnalysisRun';
import { getWorkspaceJobDetailRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { useNavigate } from 'react-router';

const RUN_BADGES: Partial<Record<HelixJobStatus, { status: BadgeStatus; label: string }>> = {
  created: { status: 'pending', label: 'Queued' },
  queued: { status: 'pending', label: 'Queued' },
  pending: { status: 'pending', label: 'Queued' },
  active: { status: 'active', label: 'Running' },
  error: { status: 'error', label: 'Failed' },
};

interface LatestAnalysisRunProps {
  workspace: string;
  run: LatestRun;
  isActive: boolean;
}

export const LatestAnalysisRun: FC<LatestAnalysisRunProps> = ({ workspace, run, isActive }) => {
  const navigate = useNavigate();
  const badge = (run.status && RUN_BADGES[run.status]) ?? { status: run.status, label: undefined };

  return (
    <Stack gap="1" data-testid="latest-analysis-run">
      <Text kind="label/regular/sm" className="text-secondary">
        Latest run
      </Text>
      <Flex align="center" gap="2" wrap="wrap">
        <StatusBadge status={badge.status} label={badge.label} />
        <Text kind="body/regular/sm" className="text-secondary">
          requested <RelativeTime datetime={run.requestedAt} />
        </Text>
        <Button
          kind="tertiary"
          size="small"
          onClick={() => navigate(getWorkspaceJobDetailRoute(workspace, run.name))}
        >
          View job
        </Button>
      </Flex>
      {isActive ? (
        <Text kind="body/regular/xs" className="text-secondary">
          This run is still in progress. You can start another run when this one finishes.
        </Text>
      ) : null}
    </Stack>
  );
};
