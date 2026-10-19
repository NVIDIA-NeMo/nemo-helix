// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fetchAllPages } from '@nemo/common/src/api/fetchAllPages';
import { agentsListAgents } from '@nemo/sdk/generated/agents/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema';

export const fetchAgentsForSelect = (workspace: string, signal: AbortSignal): Promise<Agent[]> =>
  fetchAllPages((page, pageSize) =>
    agentsListAgents(workspace, { page, page_size: pageSize, sort: 'name' }, signal)
  );
