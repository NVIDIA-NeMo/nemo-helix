// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusBadge, type StatusConfigEntry } from '@nemo/common/src/components/StatusBadge';
import { ModelDeploymentStatus } from '@nemo/sdk/generated/platform/schema';
import { Skeleton } from '@nvidia/foundations-react-core';
import type { DeploymentIndicatorState } from '@studio/hooks/useModelDeploymentStatuses';
import type { FC } from 'react';

/**
 * Two vocabularies on purpose. "Deployed" describes the deployment itself, which is
 * what a top-level row has. An adapter never has its own deployment — it is loaded
 * into the one serving its base model — so it reads "Served" / "Not served".
 *
 * No icons, matching `CustomizationStatusBadge` in this same table.
 */
const STATUS_CONFIG: Record<string, StatusConfigEntry> = {
  deployed: { label: 'Deployed', color: 'green' },
  served: { label: 'Served', color: 'green' },
  available: { label: 'Available', color: 'teal' },
  deploying: { label: 'Deploying', color: 'blue' },
  deleting: { label: 'Deleting', color: 'yellow' },
  'not-served': { label: 'Not served', color: 'yellow' },
  failed: { label: 'Failed', color: 'red' },
  unavailable: { label: 'Unavailable', color: 'gray' },
  'not-deployed': { label: 'Not deployed', color: 'gray' },
  unknown: { label: 'Unknown', color: 'purple' },
};

function toStatusKey(state: DeploymentIndicatorState, isAdapter: boolean): string {
  switch (state.kind) {
    case 'adapter-not-loaded':
      return 'not-served';

    case 'not-deployed':
      return 'not-deployed';

    case 'unknown':
      return 'unknown';

    case 'served': {
      // Reachable through a provider that names no deployment (an external provider
      // such as `default/build`), so there is genuinely no deployment status to
      // report. Only this case is "Available".
      if (!state.hasDeployment) return 'available';

      // A deployment is expected but its status could not be read — a failed
      // request, or a response that omitted the optional field. Saying "Available"
      // here would report a green, confident status we never actually confirmed.
      if (!state.status) return 'unknown';

      switch (state.status) {
        case ModelDeploymentStatus.READY:
          return isAdapter ? 'served' : 'deployed';
        case ModelDeploymentStatus.CREATED:
        case ModelDeploymentStatus.PENDING:
          return 'deploying';
        case ModelDeploymentStatus.DELETING:
          return 'deleting';
        case ModelDeploymentStatus.ERROR:
          return 'failed';
        case ModelDeploymentStatus.DELETED:
        case ModelDeploymentStatus.LOST:
          return 'unavailable';
        case ModelDeploymentStatus.UNKNOWN:
        default:
          return 'unknown';
      }
    }

    default:
      return 'unknown';
  }
}

interface DeploymentStatusBadgeProps {
  /**
   * Resolved status for the row. Resolved once for the whole table rather than
   * per badge, so the row-actions menu reads the same answer. `undefined` while
   * the row has not resolved yet.
   */
  state: DeploymentIndicatorState | undefined;
  /** Whether the row is an adapter subrow, which changes the wording. */
  isAdapter?: boolean;
}

export const DeploymentStatusBadge: FC<DeploymentStatusBadgeProps> = ({ state, isAdapter }) => {
  if (!state || state.kind === 'loading') {
    return <Skeleton animated className="h-5 w-24 rounded" />;
  }

  return (
    <StatusBadge status={toStatusKey(state, Boolean(isAdapter))} statusConfig={STATUS_CONFIG} />
  );
};
