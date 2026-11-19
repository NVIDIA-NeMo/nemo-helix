// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { PageHeader, Stack } from '@nvidia/foundations-react-core';
import { JobsDataView } from '@studio/components/dataViews/JobsDataView';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import type { FC } from 'react';

export const JobsRoute: FC = () => {
  useBreadcrumbs({ items: [{ slotLabel: 'Jobs' }] });

  return (
    <AccessibleTitle title="Jobs">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Jobs"
          slotDescription="Track and monitor all jobs across services."
        />
        <JobsDataView />
      </Stack>
    </AccessibleTitle>
  );
};
