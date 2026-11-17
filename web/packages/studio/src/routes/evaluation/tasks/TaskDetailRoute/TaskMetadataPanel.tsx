// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KVPair } from '@nemo/common/src/components/KVPair';
import type { Task } from '@nemo/sdk/generated/evals/schema';
import { Stack } from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/components/DetailPanel';
import { formatValue } from '@studio/routes/evaluation/tasksets/helpers';
import type { FC } from 'react';

/** The task's key/value annotations; hidden when it has none. */
export const TaskMetadataPanel: FC<{ task: Task }> = ({ task: { metadata = [] } }) =>
  metadata.length > 0 && (
    <DetailPanel title="Metadata">
      <Stack gap="3">
        {metadata.map(({ key, value }) => (
          <KVPair
            key={key}
            label={key}
            value={formatValue(value)}
            size="narrow"
            orientation="horizontal"
          />
        ))}
      </Stack>
    </DetailPanel>
  );
