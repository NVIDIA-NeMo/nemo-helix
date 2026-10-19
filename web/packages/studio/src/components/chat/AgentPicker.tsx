// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useAgentsListAgents } from '@nemo/sdk/generated/agents/agents';
import { Select } from '@nvidia/foundations-react-core';
import { useMemo, type FC } from 'react';

interface AgentPickerProps {
  workspace: string;
  /** Currently selected agent name from `?agent=`, or null. */
  value: string | null;
  /** Pass null to clear the selection. */
  onChange: (next: string | null) => void;
  disabled?: boolean;
}

const NO_AGENT_VALUE = '__none__';

/**
 * Lists deployed agents in the workspace and writes the selection back via
 * `onChange` (which the route maps to `setSearchParams`). The "(no agent)"
 * option clears the overlay and falls back to plain Chat.
 */
export const AgentPicker: FC<AgentPickerProps> = ({ workspace, value, onChange, disabled }) => {
  const { data, isLoading } = useAgentsListAgents(workspace, undefined, {
    query: { enabled: !!workspace },
  });

  const items = useMemo(() => {
    const agents = (data?.data ?? []).filter((a) => !!a.name);
    return [
      { value: NO_AGENT_VALUE, children: '(no agent — plain Chat)' },
      ...agents.map((a) => ({ value: a.name as string, children: a.name as string })),
    ];
  }, [data]);

  return (
    <Select
      items={items}
      value={value ?? NO_AGENT_VALUE}
      onValueChange={(next) => onChange(next === NO_AGENT_VALUE ? null : next)}
      disabled={disabled || isLoading}
      placeholder={isLoading ? 'Loading agents…' : 'Test for agent…'}
      className="w-[260px]"
    />
  );
};
