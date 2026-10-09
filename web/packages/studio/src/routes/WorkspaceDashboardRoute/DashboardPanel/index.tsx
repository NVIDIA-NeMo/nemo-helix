// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Panel, Stack, Text } from '@nvidia/foundations-react-core';
import { FC, ReactNode } from 'react';
import { Link } from 'react-router';

export interface DashboardPanelAction {
  icon: ReactNode;
  label: string;
  href: string;
}

interface DashboardPanelProps {
  icon: ReactNode;
  title: string;
  description: string;
  actions: DashboardPanelAction[];
}

const isExternalHref = (href: string): boolean => /^([a-z][a-z0-9+.-]*:|\/\/)/i.test(href);

// TEMPORARY: softens the secondary button border to 20% opacity. Remove once
// the global KUI button styles are updated to match.
const SECONDARY_ACTION_CLASS = 'w-full justify-start border-current/20';

export const DashboardPanel: FC<DashboardPanelProps> = ({ icon, title, description, actions }) => {
  return (
    <Panel slotIcon={icon} slotHeading={title} density="compact" className="h-full">
      <Stack gap="density-lg">
        <Text className="text-secondary" kind="body/regular/md">
          {description}
        </Text>
        {actions.length > 0 && (
          <Stack gap="density-sm" className="w-full">
            {actions.map((action, index) => (
              <Button
                key={`${action.label}::${action.href}`}
                kind={index === 0 ? 'primary' : 'secondary'}
                color={index === 0 ? 'brand' : 'neutral'}
                size="medium"
                className={index === 0 ? 'w-full' : SECONDARY_ACTION_CLASS}
                asChild
              >
                {isExternalHref(action.href) ? (
                  <a href={action.href} target="_blank" rel="noopener noreferrer">
                    {action.icon}
                    {action.label}
                  </a>
                ) : (
                  <Link to={action.href}>
                    {action.icon}
                    {action.label}
                  </Link>
                )}
              </Button>
            ))}
          </Stack>
        )}
      </Stack>
    </Panel>
  );
};
