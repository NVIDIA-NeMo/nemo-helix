// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { PageHeader, Panel, Stack } from '@nvidia/foundations-react-core';
import type { PluginLoadFailure } from '@studio/plugins/types';
import type { ReactElement } from 'react';

interface PluginLoadFailurePanelProps {
  failure: PluginLoadFailure;
}

// Shown in place of a plugin that is installed but could not be loaded.
export const PluginLoadFailurePanel = ({ failure }: PluginLoadFailurePanelProps): ReactElement => (
  <Stack className="h-full" padding="4" gap="3">
    <PageHeader slotHeading={`Plugin: ${failure.name}`} />
    {failure.issues.map((issue, i) => (
      <Panel key={i} elevation="high">
        <ErrorMessage
          header={issue.code}
          message={issue.message}
          slotFooter={
            <ol className="list-decimal pl-6 text-left">
              {issue.steps.map((step, s) => (
                <li key={s}>{step}</li>
              ))}
            </ol>
          }
        />
      </Panel>
    ))}
  </Stack>
);
