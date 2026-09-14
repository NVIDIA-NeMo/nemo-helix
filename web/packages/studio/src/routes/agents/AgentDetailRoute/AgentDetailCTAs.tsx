// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Block, Button, Flex } from '@nvidia/foundations-react-core';
import { INTAKE_ENABLED } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import type { AgentDetailTab } from '@studio/routes/agents/AgentDetailRoute/tabs';
import { getIntakeTracesRoute } from '@studio/routes/utils';
import { type FC, type RefObject } from 'react';
import { useNavigate } from 'react-router';

/** The action each tab promotes to the brand-colored slot. Tabs left out promote `deploy`, which
 *  is the only action every tab can offer — the others depend on a tab-specific target. */
const PRIMARY_ACTION_BY_TAB: Partial<Record<AgentDetailTab, ActionId>> = {
  evaluations: 'evaluate',
  optimizations: 'optimize',
};

type ActionId = 'evaluate' | 'deploy' | 'optimize';

interface Action {
  id: ActionId;
  label: string;
  disabled: boolean;
  onClick?: () => void;
  /** Anchors the walkthrough coachmark, which must follow the button between kinds. */
  ref?: RefObject<HTMLDivElement | null>;
}

export interface AgentDetailCTAsProps {
  tab: AgentDetailTab;
  agentName?: string;
  canDeploy: boolean;
  canRunEvaluation: boolean;
  isDeploying: boolean;
  canOptimize: boolean;
  deployButtonRef: RefObject<HTMLDivElement | null>;
  onDeploy: () => void;
  onRunEvaluation: () => void;
  /** Omitted until the tab can render the optimization form; the button stays visible but inert. */
  onOptimize?: () => void;
}

/**
 * Header actions for the agent detail page.
 *
 * Overview is the hub, so it shows the full set of actions with the tab's own next step as the
 * brand-colored button and the rest secondary. Every other tab is already scoped to one job, so it
 * shows only its primary action and leaves the rest to Overview. Optimize is the exception — it
 * only appears on the tab that can render its form.
 */
export const AgentDetailCTAs: FC<AgentDetailCTAsProps> = ({
  tab,
  agentName,
  canDeploy,
  canRunEvaluation,
  isDeploying,
  canOptimize,
  deployButtonRef,
  onDeploy,
  onRunEvaluation,
  onOptimize,
}) => {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();

  const actions: Action[] = [
    {
      id: 'evaluate',
      label: 'Run evaluation',
      disabled: !canRunEvaluation,
      onClick: onRunEvaluation,
    },
    {
      id: 'deploy',
      label: isDeploying ? 'Deploying...' : 'Deploy',
      disabled: !agentName || !canDeploy || isDeploying,
      onClick: onDeploy,
      ref: deployButtonRef,
    },
  ];

  if (tab === 'optimizations' && canOptimize) {
    actions.push({
      id: 'optimize',
      label: 'Optimize',
      disabled: !agentName || !onOptimize,
      onClick: onOptimize,
    });
  }

  const primaryId = actions.some((action) => action.id === PRIMARY_ACTION_BY_TAB[tab])
    ? PRIMARY_ACTION_BY_TAB[tab]
    : 'deploy';
  const showSecondaryActions = tab === 'overview';
  const ordered = showSecondaryActions
    ? [
        ...actions.filter((action) => action.id !== primaryId),
        ...actions.filter((action) => action.id === primaryId),
      ]
    : actions.filter((action) => action.id === primaryId);

  return (
    <Flex gap="2" wrap="wrap" justify="end">
      {INTAKE_ENABLED && showSecondaryActions && (
        <Button kind="secondary" onClick={() => navigate(getIntakeTracesRoute(workspace))}>
          Open traces
        </Button>
      )}
      {ordered.map((action) => {
        const isPrimary = action.id === primaryId;
        const button = (
          <Button
            key={isPrimary ? 'primary' : 'secondary'}
            kind={isPrimary ? undefined : 'secondary'}
            color={isPrimary ? 'brand' : undefined}
            onClick={action.onClick}
            disabled={action.disabled}
          >
            {action.label}
          </Button>
        );
        return action.ref ? (
          <Block key={action.id} ref={action.ref}>
            {button}
          </Block>
        ) : (
          <Block key={action.id}>{button}</Block>
        );
      })}
    </Flex>
  );
};
