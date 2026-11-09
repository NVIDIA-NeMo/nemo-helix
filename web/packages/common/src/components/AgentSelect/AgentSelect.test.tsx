// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AgentSelect } from '@nemo/common/src/components/AgentSelect/AgentSelect';
import type {
  MultiAgentSelectProps,
  SingleAgentSelectProps,
} from '@nemo/common/src/components/AgentSelect/types';
import { groupAgentsByFormat } from '@nemo/common/src/components/AgentSelect/utils';
import type { Agent } from '@nemo/sdk/generated/agents/schema';
import { act, fireEvent, render, screen, within } from '@testing-library/react';

const makeAgent = (name: string, overrides: Partial<Agent> = {}): Agent =>
  ({ id: name, name, workspace: 'default', ...overrides }) as Agent;

const agents = [
  makeAgent('support-bot', {
    config_format: 'nemo-agents-spec-v1',
    description: 'Answers tickets',
  }),
  makeAgent('research-bot'),
  makeAgent('triage-bot', { config_format: 'nemo-agents-spec-v1' }),
];

const openMenu = () => fireEvent.click(screen.getByTestId('agent-select-trigger'));
const rowFor = (name: string) =>
  screen.getAllByTestId('agent-dropdown-item').find((row) => row.textContent?.includes(name))!;

const setupSingle = (props: Partial<SingleAgentSelectProps> = {}) => {
  const onValueChange = vi.fn();
  render(<AgentSelect agents={agents} value={null} onValueChange={onValueChange} {...props} />);
  return onValueChange;
};

const setupMulti = (props: Partial<MultiAgentSelectProps> = {}) => {
  const onValueChange = vi.fn();
  render(
    <AgentSelect multiple agents={agents} value={[]} onValueChange={onValueChange} {...props} />
  );
  return onValueChange;
};

describe('AgentSelect', () => {
  afterEach(() => vi.useRealTimers());

  describe('single', () => {
    it('reports the picked agent and closes', () => {
      const onValueChange = setupSingle();
      openMenu();
      fireEvent.click(rowFor('research-bot'));

      expect(onValueChange).toHaveBeenCalledWith('research-bot', agents[1]);
      expect(screen.queryByTestId('agent-select-filter')).not.toBeInTheDocument();
    });

    it('shows the selected name, even before its page has loaded', () => {
      setupSingle({ agents: [], value: 'not-loaded-yet' });
      expect(screen.getByTestId('agent-select-trigger')).toHaveTextContent('not-loaded-yet');
    });

    it('ignores clicks on a disabled agent', () => {
      const onValueChange = setupSingle({
        getDisabledReason: (agent) => (agent.name === 'triage-bot' ? 'No config' : undefined),
      });
      openMenu();
      fireEvent.click(rowFor('triage-bot'));

      expect(onValueChange).not.toHaveBeenCalled();
      expect(rowFor('triage-bot')).toHaveAttribute('aria-disabled', 'true');
    });
  });

  describe('multiple', () => {
    it('adds to the selection, keeps the menu open, and hands back every entity', () => {
      const onValueChange = setupMulti({ value: ['support-bot'] });
      openMenu();
      fireEvent.click(rowFor('triage-bot'));

      expect(onValueChange).toHaveBeenCalledWith(
        ['support-bot', 'triage-bot'],
        [agents[0], agents[2]]
      );
      expect(screen.getByTestId('agent-select-filter')).toBeInTheDocument();
    });

    it('removes an already-selected agent', () => {
      const onValueChange = setupMulti({ value: ['support-bot', 'triage-bot'] });
      openMenu();
      fireEvent.click(rowFor('support-bot'));

      expect(onValueChange).toHaveBeenCalledWith(['triage-bot'], [agents[2]]);
    });

    it('summarizes the selection on the trigger and clears it', () => {
      const onValueChange = setupMulti({ value: ['support-bot', 'triage-bot', 'research-bot'] });
      expect(screen.getByTestId('agent-select-trigger')).toHaveTextContent('support-bot+2');

      openMenu();
      expect(screen.getByText('3 selected')).toBeInTheDocument();
      fireEvent.click(screen.getByTestId('agent-select-clear'));
      expect(onValueChange).toHaveBeenCalledWith([], []);
    });

    it('disables unselected agents once the cap is reached', () => {
      const onValueChange = setupMulti({ value: ['support-bot'], maxSelected: 1 });
      openMenu();
      fireEvent.click(rowFor('research-bot'));

      expect(onValueChange).not.toHaveBeenCalled();
      expect(screen.getByText('1 of 1 selected')).toBeInTheDocument();
      expect(rowFor('support-bot')).not.toHaveAttribute('aria-disabled');
    });
  });

  describe('search', () => {
    const type = (text: string) => {
      fireEvent.change(screen.getByTestId('agent-select-filter'), { target: { value: text } });
      act(() => vi.advanceTimersByTime(300));
    };

    it('filters by name and description itself when no onSearchChange is given', () => {
      vi.useFakeTimers();
      setupSingle();
      openMenu();
      type('tickets');

      expect(screen.getAllByTestId('agent-dropdown-item')).toHaveLength(1);
      expect(rowFor('support-bot')).toBeInTheDocument();
    });

    it('hands the term to the caller and leaves the list alone when onSearchChange is given', () => {
      vi.useFakeTimers();
      const onSearchChange = vi.fn();
      setupSingle({ onSearchChange });
      openMenu();
      type('triage');

      expect(onSearchChange).toHaveBeenLastCalledWith('triage');
      expect(screen.getAllByTestId('agent-dropdown-item')).toHaveLength(3);
    });

    it('shows the empty message when nothing matches', () => {
      setupSingle({ agents: [], emptyMessage: 'Nothing here' });
      openMenu();
      expect(screen.getByText('Nothing here')).toBeInTheDocument();
    });
  });

  it('groups agents under a heading per group', () => {
    setupSingle({ groupBy: groupAgentsByFormat });
    openMenu();
    const content = screen.getByTestId('agent-select-content');

    expect(within(content).getByText('Fabric')).toBeInTheDocument();
    expect(within(content).getByText('NAT workflow')).toBeInTheDocument();
    const order = within(content)
      .getAllByText(/^(Fabric|NAT workflow|.+-bot)$/)
      .map((el) => el.textContent);
    expect(order).toEqual(['Fabric', 'support-bot', 'triage-bot', 'NAT workflow', 'research-bot']);
  });

  it('mounts an agent’s details only once its row is hovered', async () => {
    setupSingle();
    openMenu();
    expect(screen.queryByTestId('agent-details-panel')).not.toBeInTheDocument();

    fireEvent.pointerEnter(rowFor('support-bot'));
    const panel = await screen.findByTestId('agent-details-panel');
    expect(within(panel).getByText('Answers tickets')).toBeInTheDocument();
    expect(within(panel).getByText('Fabric')).toBeInTheDocument();
  });
});
