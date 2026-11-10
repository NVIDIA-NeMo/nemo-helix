// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AgentDetailsPanel } from '@nemo/common/src/components/AgentSelect/AgentDetailsPanel';
import { formatAgentConfigFormat } from '@nemo/common/src/components/AgentSelect/utils';
import { Check } from '@nemo/icons/icons';
import type { Agent } from '@nemo/sdk/generated/agents/schema';
import {
  DropdownSub,
  DropdownSubContent,
  DropdownSubTrigger,
  Flex,
  Text,
} from '@nvidia/foundations-react-core';
import { memo, useState, type FC } from 'react';

interface AgentDropdownItemProps {
  agent: Agent;
  isSelected: boolean;
  /** Set when the agent cannot be picked; the row stays hoverable so the reason can be read. */
  disabledReason?: string;
  /** Hides the format hint on the right when the list is already grouped by it. */
  showFormat: boolean;
  onToggle: (agent: Agent) => void;
}

const AgentDropdownItemImpl: FC<AgentDropdownItemProps> = ({
  agent,
  isSelected,
  disabledReason,
  showFormat,
  onToggle,
}) => {
  // KUI's DropdownSubContent stays in the DOM while closed; defer the panel until first hover.
  const [hasOpened, setHasOpened] = useState(false);

  return (
    <DropdownSub onOpenChange={(open) => open && setHasOpened(true)}>
      <DropdownSubTrigger
        slotEnd={false}
        data-testid="agent-dropdown-item"
        aria-disabled={disabledReason ? true : undefined}
        aria-checked={isSelected}
        onClick={() => !disabledReason && onToggle(agent)}
      >
        <Flex
          className={`w-full min-w-0 ${disabledReason ? 'opacity-50' : ''}`}
          align="center"
          gap="density-sm"
        >
          <Flex className="w-3.5 shrink-0">
            {isSelected && <Check size="14" data-testid="agent-dropdown-selected-check" />}
          </Flex>
          <Text className="truncate flex-1">{agent.name}</Text>
          {showFormat && (
            <Text className="text-secondary whitespace-nowrap" kind="body/regular/sm">
              {formatAgentConfigFormat(agent.config_format)}
            </Text>
          )}
        </Flex>
      </DropdownSubTrigger>
      {/* eslint-disable-next-line no-restricted-syntax -- KUI ignores Tailwind width classes */}
      <DropdownSubContent style={{ width: 360 }}>
        {hasOpened && <AgentDetailsPanel agent={agent} disabledReason={disabledReason} />}
      </DropdownSubContent>
    </DropdownSub>
  );
};

/** Memoized so typing in the filter box does not re-render every row in view. */
export const AgentDropdownItem = memo(AgentDropdownItemImpl);
