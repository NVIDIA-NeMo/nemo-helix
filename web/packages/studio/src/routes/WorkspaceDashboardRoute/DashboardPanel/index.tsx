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

export const DashboardPanel: FC<DashboardPanelProps> = ({ icon, title, description, actions }) => {
  return (
    <Panel
      slotIcon={icon}
      slotHeading={title}
      className="h-full"
      slotFooter={
        actions.length > 0 && (
          <Stack gap="density-xxs" className="w-full">
            {actions.map((action) =>
              isExternalHref(action.href) ? (
                <Button
                  key={`${action.label}::${action.href}`}
                  kind="tertiary"
                  color="neutral"
                  size="small"
                  asChild
                >
                  <a href={action.href} target="_blank" rel="noopener noreferrer">
                    {action.icon}
                    {action.label}
                  </a>
                </Button>
              ) : (
                <Button
                  key={`${action.label}::${action.href}`}
                  kind="tertiary"
                  color="neutral"
                  size="small"
                  asChild
                >
                  <Link to={action.href}>
                    {action.icon}
                    {action.label}
                  </Link>
                </Button>
              )
            )}
          </Stack>
        )
      }
    >
      <Text className="text-secondary" kind="body/regular/md">
        {description}
      </Text>
    </Panel>
  );
};
