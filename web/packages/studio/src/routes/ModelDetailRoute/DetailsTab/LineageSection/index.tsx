// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import type { HelixJobStatus } from '@nemo/sdk/generated/platform/schema';
import type { FilesetRef } from '@studio/hooks/useModelLineage';
import { LineageCard } from '@studio/routes/ModelDetailRoute/DetailsTab/LineageSection/LineageCard';
import { getFilesetRoute } from '@studio/routes/utils';
import type { FC } from 'react';

export interface LineageSectionProps {
  /**
   * Base model card: the model an adapter belongs to, or a checkpoint's `base_model`. This and
   * `job` are omitted for base models, which have neither.
   */
  base?: { name: string; to?: string; note: string };
  job?: {
    jobName: string | undefined;
    isLoading: boolean;
    status?: HelixJobStatus;
    createdAt?: string;
    to: string;
  };
  fileset: FilesetRef | null;
}

export const LineageSection: FC<LineageSectionProps> = ({ base, job, fileset }) => (
  <>
    {base && <LineageCard label="Base Model" value={base.name} to={base.to} note={base.note} />}

    {job && (
      <LineageCard
        label="Training Job"
        loading={job.isLoading}
        value={job.jobName ?? 'Not found'}
        to={job.jobName ? job.to : undefined}
        badge={job.status && <StatusBadge status={job.status} />}
        note={
          job.jobName
            ? job.createdAt && formatAbsoluteTimestamp(job.createdAt)
            : 'No job lists this model as its output'
        }
      />
    )}

    <LineageCard
      label="Files"
      value={fileset?.name ?? 'None'}
      to={
        fileset
          ? getFilesetRoute(fileset.workspace, `${fileset.workspace}/${fileset.name}`)
          : undefined
      }
      note={fileset ? `Fileset in ${fileset.workspace}` : 'No fileset'}
    />
  </>
);
