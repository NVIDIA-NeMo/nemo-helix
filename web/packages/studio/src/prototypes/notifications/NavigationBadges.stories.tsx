// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ENTITY_ICONS } from '@nemo/common/src/constants/entityIcons';
import { Button, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { NavNotificationBadge } from '@studio/prototypes/notifications/components/NavNotificationBadge';
import { NavNotificationDot } from '@studio/prototypes/notifications/components/NavNotificationDot';
import {
  NAV_HREFS,
  NAV_LABELS,
  SAMPLE_NOTIFICATIONS,
  buildStudioNav,
  type NavId,
  type NotificationMap,
} from '@studio/prototypes/notifications/fixtures';
import { ProtoNavigationDrawer } from '@studio/prototypes/notifications/NavigationDrawer';
import type { NotificationSeverity } from '@studio/prototypes/notifications/types';
import { rollupNotifications } from '@studio/prototypes/notifications/utils';
import { useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router';

interface NavPreviewArgs {
  collapsed: boolean;
  parentsOpen: boolean;
  notifications: NotificationMap;
}

const NavPreview = ({ collapsed, parentsOpen, notifications }: NavPreviewArgs) => {
  const items = useMemo(
    () => buildStudioNav(notifications, { parentsOpen }),
    [notifications, parentsOpen]
  );

  return (
    <div className="h-[780px] w-fit">
      <ProtoNavigationDrawer key={String(parentsOpen)} items={items} collapsed={collapsed} />
    </div>
  );
};

const meta = {
  component: NavPreview,
  title: 'Prototypes/Notifications/Navigation Badges',
  args: {
    collapsed: false,
    parentsOpen: true,
    notifications: SAMPLE_NOTIFICATIONS,
  },
  parameters: {
    router: { initialPath: NAV_HREFS['custom-models'] },
  },
} satisfies Meta<typeof NavPreview>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Expanded: Story = {};

export const ParentRollup: Story = {
  args: { parentsOpen: false },
};

export const Rail: Story = {
  args: { collapsed: true },
};

export const NoNotifications: Story = {
  args: { notifications: {} },
};

const SEVERITIES: NotificationSeverity[] = ['info', 'success', 'warning', 'error'];
const COUNTS = [1, 9, 42, 120];
const SampleIcon = ENTITY_ICONS.customModels;

export const BadgeMatrix: Story = {
  render: () => (
    <table className="border-separate border-spacing-x-6 border-spacing-y-3 text-left">
      <thead>
        <tr>
          <th>
            <Text kind="label/bold/sm">Severity</Text>
          </th>
          {COUNTS.map((count) => (
            <th key={count}>
              <Text kind="label/bold/sm">{count}</Text>
            </th>
          ))}
          <th>
            <Text kind="label/bold/sm">Rail dot</Text>
          </th>
        </tr>
      </thead>
      <tbody>
        {SEVERITIES.map((severity) => (
          <tr key={severity}>
            <td>
              <Text kind="body/regular/sm">{severity}</Text>
            </td>
            {COUNTS.map((count) => (
              <td key={count}>
                <NavNotificationBadge notification={{ count, severity }} className="ml-0" />
              </td>
            ))}
            <td>
              <NavNotificationDot notification={{ count: 1, severity }}>
                <SampleIcon />
              </NavNotificationDot>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  ),
};

interface SimulatedEvent {
  label: string;
  id: NavId;
  severity: NotificationSeverity;
}

const SIMULATED_EVENTS: SimulatedEvent[] = [
  { label: 'Fine-tuning job completed', id: 'custom-models', severity: 'success' },
  { label: 'Evaluation job failed', id: 'evaluation-results', severity: 'error' },
  { label: 'Agent suggestion awaiting review', id: 'agents-group', severity: 'info' },
  { label: 'Optimizer insight found', id: 'optimizer', severity: 'info' },
  { label: 'Guardrail check degraded', id: 'guardrails', severity: 'warning' },
  { label: 'Data Designer job completed', id: 'data-designer', severity: 'success' },
];

const NAV_IDS_WITH_HREF = Object.keys(NAV_HREFS) as (keyof typeof NAV_HREFS)[];

const LiveNavigationBadges = () => {
  const { pathname } = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [notifications, setNotifications] = useState<NotificationMap>(SAMPLE_NOTIFICATIONS);
  const items = useMemo(() => buildStudioNav(notifications), [notifications]);

  useEffect(() => {
    const visited = NAV_IDS_WITH_HREF.find((id) => NAV_HREFS[id] === pathname);
    if (visited === undefined) {
      return;
    }
    setNotifications((prev) => {
      if (prev[visited] === undefined) {
        return prev;
      }
      const next = { ...prev };
      delete next[visited];
      return next;
    });
  }, [pathname]);

  const push = ({ id, severity }: SimulatedEvent) => {
    setNotifications((prev) => ({
      ...prev,
      [id]: rollupNotifications([prev[id], { count: 1, severity }]),
    }));
  };

  return (
    <div className="flex h-[780px] gap-8">
      <ProtoNavigationDrawer items={items} collapsed={collapsed} />
      <div className="flex max-w-sm flex-col gap-3">
        <Text kind="label/bold/md">Simulate events</Text>
        {SIMULATED_EVENTS.map((event) => (
          <Button key={event.label} kind="secondary" onClick={() => push(event)}>
            {event.label} → {NAV_LABELS[event.id]}
          </Button>
        ))}
        <Button kind="tertiary" onClick={() => setNotifications({})}>
          Mark all as read
        </Button>
        <Button kind="tertiary" onClick={() => setCollapsed((prev) => !prev)}>
          {collapsed ? 'Expand drawer' : 'Collapse to rail'}
        </Button>
        <Text kind="body/regular/sm">
          Clicking a nav row navigates there and clears that row&apos;s badge. Current path:{' '}
          <code>{pathname}</code>
        </Text>
      </div>
    </div>
  );
};

export const Live: Story = {
  parameters: { router: { initialPath: NAV_HREFS.dashboard } },
  render: () => <LiveNavigationBadges />,
};
