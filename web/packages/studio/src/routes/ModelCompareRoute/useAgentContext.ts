// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsGetAgent } from '@nemo/sdk/generated/agents/agents';
import type { AgentConfig as StudioAgentConfig } from '@studio/components/dataViews/AgentsDataView';
import { useMemo } from 'react';

export interface AgentContext {
  name: string;
  /** Workspace-qualified model URN, e.g. "default/nvcf-meta-llama-3-3-70b-instruct". */
  currentModelUrn: string;
  /** May be empty — NAT ReAct workflows don't carry a per-agent system prompt. */
  systemPrompt: string;
}

interface AgentContextResult {
  /** Resolved context, or null when no agent is selected / lookup failed / agent missing fields. */
  context: AgentContext | null;
  isLoading: boolean;
  /** Non-null when the lookup failed (404, etc.). */
  error: Error | null;
}

/**
 * `config.llms[<key>]` may carry a system prompt on some workflow shapes (e.g.
 * langgraph_agent stores it on the LLM dict). The studio's typed `AgentConfig`
 * doesn't enumerate this — read defensively.
 */
function extractSystemPrompt(llm: unknown): string {
  if (!llm || typeof llm !== 'object') return '';
  const candidate = (llm as { system_prompt?: unknown }).system_prompt;
  return typeof candidate === 'string' ? candidate : '';
}

/**
 * Projects the platform's `Agent` entity into the lean `AgentContext` shape
 * that drives the Chat overlay (locked baseline, banner, system-prompt seed).
 * Returns null context when no agent name is supplied, when the lookup is
 * still in flight, or when the agent's config doesn't expose the workflow's
 * primary LLM in a recognizable shape.
 */
export const useAgentContext = (
  workspace: string,
  agentName: string | null
): AgentContextResult => {
  const enabled = !!(workspace && agentName);
  const { data, isLoading, error } = useAgentsGetAgent(workspace, agentName ?? '', {
    query: { enabled },
  });

  const context = useMemo<AgentContext | null>(() => {
    if (!enabled || !data) return null;
    const config = (data.config ?? {}) as StudioAgentConfig;
    const llmKey = config.workflow?.llm_name;
    const llm = llmKey ? config.llms?.[llmKey] : undefined;
    const modelName = llm?.model_name;
    if (!modelName) return null;
    const agentWorkspace = data.workspace ?? workspace;
    return {
      name: data.name ?? agentName ?? '',
      currentModelUrn: `${agentWorkspace}/${modelName}`,
      systemPrompt: extractSystemPrompt(llm),
    };
  }, [enabled, data, agentName, workspace]);

  return {
    context,
    isLoading: enabled && isLoading,
    error: (error as Error | null) ?? null,
  };
};
