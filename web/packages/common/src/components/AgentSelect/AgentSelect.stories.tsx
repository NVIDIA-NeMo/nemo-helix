// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AgentSelect } from '@nemo/common/src/components/AgentSelect/AgentSelect';
import {
  groupAgentsByFormat,
  matchesAgentSearch,
} from '@nemo/common/src/components/AgentSelect/utils';
import { FABRIC_CONFIG_FORMAT } from '@nemo/common/src/utils/agents';
import type { Agent } from '@nemo/sdk/generated/agents/schema';
import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';

const makeAgent = (name: string, overrides: Partial<Agent> = {}): Agent =>
  ({
    id: name,
    name,
    workspace: 'default',
    created_at: '2026-10-01T00:00:00Z',
    updated_at: '2026-10-20T00:00:00Z',
    created_by: 'steramae',
    ...overrides,
  }) as Agent;

const mockAgents: Agent[] = [
  makeAgent('customer-support', {
    config_format: FABRIC_CONFIG_FORMAT,
    description: 'Answers support tickets using the product knowledge base.',
    project: 'support',
    labels: { 'helix-agents/managed-by': 'studio', team: 'cx' },
    config: {
      models: { default: { provider: 'nvidia', model: 'nvidia-nemotron-3-nano-30b-a3b' } },
    },
  }),
  makeAgent('research-assistant', {
    description: 'Multi-step web research with citations.',
    config: { llms: { llm: { _type: 'nim', model_name: 'meta/llama-3.3-70b-instruct' } } },
  }),
  makeAgent('ticket-triage', {
    config_format: FABRIC_CONFIG_FORMAT,
    description: 'Routes incoming tickets to the right queue.',
  }),
  makeAgent('traces-only', { description: 'Registered from imported traces.', config: {} }),
];

const noConfig = (agent: Agent) =>
  agent.config && Object.keys(agent.config).length > 0
    ? undefined
    : 'This agent has no configuration, so it cannot be deployed.';

const meta: Meta<typeof AgentSelect> = {
  component: AgentSelect,
  title: 'Studio Common/AgentSelect',
};

export default meta;

type Story = StoryObj<typeof AgentSelect>;

const SingleRender = () => {
  const [value, setValue] = useState<string | null>(null);
  return <AgentSelect agents={mockAgents} value={value} onValueChange={setValue} />;
};

export const Single: Story = { render: SingleRender };

const MultipleRender = () => {
  const [value, setValue] = useState<string[]>(['customer-support']);
  return <AgentSelect multiple agents={mockAgents} value={value} onValueChange={setValue} />;
};

export const Multiple: Story = { render: MultipleRender };

const MultipleCappedRender = () => {
  const [value, setValue] = useState<string[]>([]);
  return (
    <AgentSelect
      multiple
      maxSelected={2}
      agents={mockAgents}
      value={value}
      onValueChange={setValue}
    />
  );
};

export const MultipleCapped: Story = { render: MultipleCappedRender };

const GroupedRender = () => {
  const [value, setValue] = useState<string | null>(null);
  return (
    <AgentSelect
      agents={mockAgents}
      value={value}
      onValueChange={setValue}
      groupBy={groupAgentsByFormat}
    />
  );
};

export const GroupedByFormat: Story = { render: GroupedRender };

const DisabledReasonRender = () => {
  const [value, setValue] = useState<string | null>(null);
  return (
    <AgentSelect
      agents={mockAgents}
      value={value}
      onValueChange={setValue}
      getDisabledReason={noConfig}
    />
  );
};

export const WithDisabledReason: Story = { render: DisabledReasonRender };

export const Loading: Story = {
  render: () => <AgentSelect agents={[]} loading value={null} onValueChange={() => {}} />,
};

export const Empty: Story = {
  render: () => <AgentSelect agents={[]} value={null} onValueChange={() => {}} />,
};

const PAGE_SIZE = 25;
const MANY_AGENTS = Array.from({ length: 300 }, (_, i) =>
  makeAgent(`agent-${String(i).padStart(3, '0')}`, {
    config_format: i % 3 ? FABRIC_CONFIG_FORMAT : undefined,
  })
);

/** Mimics `WorkspaceAgentSelect`: server-side search, 25 per page, 600ms latency. */
const ProgressiveRender = () => {
  const [value, setValue] = useState<string[]>([]);
  const [search, setSearch] = useState('');
  const [pageCount, setPageCount] = useState(1);
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  const matches = MANY_AGENTS.filter((agent) => matchesAgentSearch(agent, search));
  const loaded = matches.slice(0, pageCount * PAGE_SIZE);

  return (
    <AgentSelect
      multiple
      value={value}
      onValueChange={setValue}
      agents={loaded}
      onSearchChange={(next) => {
        setSearch(next);
        setPageCount(1);
      }}
      onLoadMore={async () => {
        setIsLoadingMore(true);
        await new Promise((resolve) => setTimeout(resolve, 600));
        setPageCount((count) => count + 1);
        setIsLoadingMore(false);
      }}
      hasMore={loaded.length < matches.length}
      isLoadingMore={isLoadingMore}
      doneLoadingMessage="No more agents"
      fullWidth
    />
  );
};

export const ProgressiveLoading: Story = { render: ProgressiveRender };
