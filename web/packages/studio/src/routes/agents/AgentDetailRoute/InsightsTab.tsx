// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Agent } from '@nemo/sdk/generated/agents/schema/Agent';
import { Stack } from '@nvidia/foundations-react-core';
import { AnalysisConfigPanel } from '@studio/routes/agents/AgentDetailRoute/analysis/AnalysisConfigPanel';
import type { FC } from 'react';

interface InsightsTabProps {
  workspace: string;
  agentName?: string;
  agent?: Agent;
}

/** Per-agent insights customization: periodic analysis and the model pair it runs with. */
export const InsightsTab: FC<InsightsTabProps> = ({ workspace, agentName, agent }) => (
  <Stack gap="4" className="mx-auto w-full max-w-3xl pb-6">
    <AnalysisConfigPanel
      workspace={agent?.workspace ?? workspace}
      agent={agent?.name ?? agentName}
    />
  </Stack>
);
