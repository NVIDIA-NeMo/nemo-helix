// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EntityAgentBridge } from '@nemo/common/src/components/EntityAgentBridge';
import { ENTITY_EMPTY_STATES } from '@nemo/common/src/components/EntityEmptyState/registry';
import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';

const wrap = (ui: ReactNode) => render(<ToastProvider>{ui}</ToastProvider>);

describe('EntityAgentBridge', () => {
  it('defaults to the skill prompt, since the handoff is the point', () => {
    wrap(<EntityAgentBridge entity="guardrails" />);

    expect(screen.getByTestId('entity-empty-state-help')).toHaveTextContent(
      ENTITY_EMPTY_STATES.guardrails.skillPrompt as string
    );
  });

  it('switches to the CLI command', async () => {
    wrap(<EntityAgentBridge entity="guardrails" />);

    await userEvent.click(screen.getByText('CLI'));
    expect(screen.getByTestId('entity-empty-state-help')).toHaveTextContent(
      ENTITY_EMPTY_STATES.guardrails.cliCommand as string
    );
  });

  it('resolves <workspace> against the workspace it is given', () => {
    wrap(<EntityAgentBridge entity="telemetryTraces" workspace="my-workspace" />);

    const help = screen.getByTestId('entity-empty-state-help');
    expect(help).toHaveTextContent('my-workspace');
    expect(help).not.toHaveTextContent('<workspace>');
  });

  it('leaves the placeholder alone when no workspace is known', () => {
    wrap(<EntityAgentBridge entity="telemetryTraces" />);

    expect(screen.getByTestId('entity-empty-state-help')).toHaveTextContent('<workspace>');
  });

  it('offers no toggle when the entity has only one of the two', () => {
    // `members` ships a CLI command and no skill prompt.
    wrap(<EntityAgentBridge entity="members" />);

    expect(screen.getByTestId('entity-empty-state-help')).toBeInTheDocument();
    expect(screen.queryByText('Ask an agent')).not.toBeInTheDocument();
    expect(screen.queryByText('CLI')).not.toBeInTheDocument();
  });

  it('renders nothing for an entity with no bridge at all', () => {
    // `baseModels` are surfaced by registering a provider, not created directly.
    wrap(<EntityAgentBridge entity="baseModels" />);

    expect(screen.queryByTestId('entity-empty-state-help')).not.toBeInTheDocument();
  });
});
