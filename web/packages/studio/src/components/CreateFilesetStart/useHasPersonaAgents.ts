// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsListAgents } from '@nemo/sdk/generated/agents/agents';

/** Whether the workspace has any agent to build a persona test set for; undefined while loading. */
export const useHasPersonaAgents = (workspace: string, enabled: boolean): boolean | undefined => {
  const { data } = useAgentsListAgents(
    workspace,
    { page_size: 1 },
    { query: { enabled: enabled && !!workspace } }
  );
  return data && data.data.length > 0;
};
