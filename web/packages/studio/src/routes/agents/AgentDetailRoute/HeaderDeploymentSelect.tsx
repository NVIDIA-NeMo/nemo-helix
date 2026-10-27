// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
  healthyDeployments: AgentDeployment[];
  selected?: AgentDeployment;
  /** Shown in place of a deployment name when there is none to pick, e.g. "No deployments". */
  statusLabel: string;
  canCreate: boolean;
  onSelect: (name: string) => void;
  onCreate: () => void;
}

/** Picks the healthy deployment the page talks to; falls back to the deployment status when none is healthy. */
export const HeaderDeploymentSelect: FC<HeaderDeploymentSelectProps> = ({
  healthyDeployments,
  selected,
  statusLabel,
  canCreate,
  onSelect,
  onCreate,
}) => {
  const named = healthyDeployments.filter((d): d is AgentDeployment & { name: string } => !!d.name);

  return (
    <DropdownRoot>
      <DropdownTrigger kind="secondary" size="small" aria-label="Deployment" className="font-bold">
        {selected?.name ?? statusLabel}
      </DropdownTrigger>
      <DropdownContent align="start" className="min-w-72">
        <MenuSection slotHeading="Switch Deployments">
          {named.length > 0 ? (
            named.map((d) => (
              <DropdownItem
                key={d.name}
                onSelect={() => onSelect(d.name)}
                slotEnd={d.name === selected?.name ? <Check aria-hidden /> : undefined}
              >
                {d.name}
              </DropdownItem>
            ))
          ) : (
            <DropdownItem disabled>No deployments available</DropdownItem>
          )}
        </MenuSection>
        <MenuSection>
          <DropdownItem disabled={!canCreate} onSelect={onCreate}>
            New Deployment
          </DropdownItem>
        </MenuSection>
      </DropdownContent>
    </DropdownRoot>
  );
};
