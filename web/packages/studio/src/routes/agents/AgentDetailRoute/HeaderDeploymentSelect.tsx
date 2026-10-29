// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { StatusDot } from '@nemo/common/src/components/StatusDot';
import { Check } from '@nemo/icons/icons';
import type { AgentDeployment } from '@nemo/sdk/generated/agents/schema/AgentDeployment';
import {
  DropdownContent,
  DropdownItem,
  DropdownRoot,
  DropdownTrigger,
  MenuSection,
} from '@nvidia/foundations-react-core';
import type { FC } from 'react';

interface HeaderDeploymentSelectProps {
  /** Every deployment of the agent, any status: unhealthy ones stay pickable for their logs. */
  deployments: AgentDeployment[];
  selected?: AgentDeployment;
  canCreate: boolean;
  onSelect: (name: string) => void;
  onCreate: () => void;
}

/** Picks the deployment the Chat and Logs tabs target. */
export const HeaderDeploymentSelect: FC<HeaderDeploymentSelectProps> = ({
  deployments,
  selected,
  canCreate,
  onSelect,
  onCreate,
}) => {
  const named = deployments.filter((d): d is AgentDeployment & { name: string } => !!d.name);

  return (
    <DropdownRoot>
      <DropdownTrigger kind="secondary" size="small" aria-label="Deployment" className="font-bold">
        {selected ? <StatusDot status={selected.status} /> : null}
        {selected?.name ?? 'No deployments'}
      </DropdownTrigger>
      <DropdownContent align="start" className="min-w-72">
        {named.length > 0 ? (
          <MenuSection slotHeading="Switch Deployments">
            {named.map((d) => (
              <DropdownItem
                key={d.name}
                onSelect={() => onSelect(d.name)}
                slotStart={<StatusDot status={d.status} />}
                slotEnd={
                  <Check
                    aria-hidden
                    className={d.name === selected?.name ? undefined : 'invisible'}
                  />
                }
              >
                {d.name}
              </DropdownItem>
            ))}
          </MenuSection>
        ) : null}
        <MenuSection>
          <DropdownItem disabled={!canCreate} onSelect={onCreate}>
            New Deployment
          </DropdownItem>
        </MenuSection>
      </DropdownContent>
    </DropdownRoot>
  );
};
