// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import {
  NAV_HREFS,
  SAMPLE_NOTIFICATIONS,
  buildStudioNav,
  type NavId,
  type NotificationMap,
} from '@studio/prototypes/notifications/fixtures';
import { ProtoNavigationDrawer } from '@studio/prototypes/notifications/NavigationDrawer';
import { BriefToastCard } from '@studio/prototypes/notifications/toasts/BriefToastCard';
import {
  DATASET_UPLOADED,
  DEMO_EVENTS,
  EVALUATION_FAILED,
  GUARDRAIL_DEGRADED,
  JOB_COMPLETED,
  SETTINGS_SAVED,
  buildRoutingSuggestion,
} from '@studio/prototypes/notifications/toasts/fixtures';
import { NotificationCard } from '@studio/prototypes/notifications/toasts/NotificationCard';
import { ProtoNotificationProvider } from '@studio/prototypes/notifications/toasts/ProtoNotificationProvider';
import { useProtoNotifications } from '@studio/prototypes/notifications/toasts/useProtoNotifications';
import { rollupNotifications } from '@studio/prototypes/notifications/utils';
import { useEffect, useMemo, useState, type FC } from 'react';
import { useLocation } from 'react-router';

const noop = () => {};
const staticApi = { notify: () => '', toast: () => '' };

const Gallery: FC = () => (
  <div className="grid grid-cols-[auto_1fr] items-start gap-x-8 gap-y-6">
    <Text kind="label/bold/sm">Job completed + link</Text>
    <NotificationCard {...JOB_COMPLETED} durationMs={false} onDismiss={noop} />

    <Text kind="label/bold/sm">Job failed + link</Text>
    <NotificationCard {...EVALUATION_FAILED} durationMs={false} onDismiss={noop} />

    <Text kind="label/bold/sm">Suggestion + accept / decline</Text>
    <NotificationCard {...buildRoutingSuggestion(staticApi)} onDismiss={noop} />

    <Text kind="label/bold/sm">Warning + link</Text>
    <NotificationCard {...GUARDRAIL_DEGRADED} durationMs={false} onDismiss={noop} />

    <Text kind="label/bold/sm">Upload complete + link</Text>
    <NotificationCard {...DATASET_UPLOADED} durationMs={false} onDismiss={noop} />

    <Text kind="label/bold/sm">Brief toast (success)</Text>
    <BriefToastCard {...SETTINGS_SAVED} durationMs={false} onDismiss={noop} className="w-fit" />

    <Text kind="label/bold/sm">Brief toast (info)</Text>
    <BriefToastCard
      status="info"
      message="Suggestion dismissed"
      durationMs={false}
      onDismiss={noop}
      className="w-fit"
    />
  </div>
);

const meta = {
  title: 'Prototypes/Notifications/Toasts',
  component: Gallery,
  decorators: [
    (Story) => (
      <ProtoNotificationProvider>
        <Story />
      </ProtoNotificationProvider>
    ),
  ],
} satisfies Meta<typeof Gallery>;

export default meta;
type Story = StoryObj<typeof meta>;

export const AllVariants: Story = {};

const CurrentPath: FC = () => {
  const { pathname, search } = useLocation();
  return (
    <Text kind="body/regular/sm">
      Current path: <code>{`${pathname}${search}`}</code>
    </Text>
  );
};

const LiveToasts: FC = () => {
  const api = useProtoNotifications();

  return (
    <div className="flex max-w-sm flex-col gap-3">
      <Text kind="label/bold/md">Trigger notifications</Text>
      {DEMO_EVENTS.map((event) => (
        <Button key={event.label} kind="secondary" onClick={() => api.notify(event.build(api))}>
          {event.label}
        </Button>
      ))}
      <Button kind="secondary" onClick={() => api.notify(DATASET_UPLOADED)}>
        Dataset uploaded
      </Button>
      <Button kind="secondary" onClick={() => api.toast(SETTINGS_SAVED)}>
        Brief toast: Settings saved
      </Button>
      <Button
        kind="tertiary"
        onClick={() => DEMO_EVENTS.forEach((event) => api.notify(event.build(api)))}
      >
        Burst: all four at once
      </Button>
      <CurrentPath />
    </div>
  );
};

export const Live: Story = {
  parameters: { router: { initialPath: NAV_HREFS.dashboard } },
  render: () => <LiveToasts />,
};

const HREF_ENTRIES = Object.entries(NAV_HREFS) as [keyof typeof NAV_HREFS, string][];

const matchNavId = (pathname: string): NavId | undefined =>
  HREF_ENTRIES.filter(([, href]) => pathname === href || pathname.startsWith(`${href}/`)).reduce<
    [NavId, string] | undefined
  >(
    (best, entry) => (best === undefined || entry[1].length > best[1].length ? entry : best),
    undefined
  )?.[0];

const ToastsWithNavigation: FC = () => {
  const api = useProtoNotifications();
  const { pathname } = useLocation();
  const [badges, setBadges] = useState<NotificationMap>(SAMPLE_NOTIFICATIONS);
  const items = useMemo(() => buildStudioNav(badges), [badges]);

  useEffect(() => {
    const visited = matchNavId(pathname);
    if (visited === undefined) {
      return;
    }
    setBadges((prev) => {
      if (prev[visited] === undefined) {
        return prev;
      }
      const next = { ...prev };
      delete next[visited];
      return next;
    });
  }, [pathname]);

  const fire = (event: (typeof DEMO_EVENTS)[number]) => {
    const options = event.build(api);
    api.notify(options);
    setBadges((prev) => ({
      ...prev,
      [event.sourceId]: rollupNotifications([
        prev[event.sourceId],
        { count: 1, severity: options.status },
      ]),
    }));
  };

  return (
    <div className="flex h-[780px] gap-8">
      <ProtoNavigationDrawer items={items} />
      <div className="flex max-w-sm flex-col gap-3">
        <Text kind="label/bold/md">Simulate events</Text>
        {DEMO_EVENTS.map((event) => (
          <Button key={event.label} kind="secondary" onClick={() => fire(event)}>
            {event.label}
          </Button>
        ))}
        <Text kind="body/regular/sm">
          Each event raises a notification and bumps the matching nav badge. Following a
          notification&apos;s link, or clicking the nav row, lands on that section and clears its
          badge.
        </Text>
        <CurrentPath />
      </div>
    </div>
  );
};

export const WithNavigationBadges: Story = {
  parameters: { router: { initialPath: NAV_HREFS.dashboard } },
  render: () => <ToastsWithNavigation />,
};
