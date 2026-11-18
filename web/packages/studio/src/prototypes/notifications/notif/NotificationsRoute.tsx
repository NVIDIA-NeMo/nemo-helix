// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { BellOff } from '@nemo/icons/icons';
import {
  Button,
  PageHeader,
  SegmentedControl,
  Spinner,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { usePluginInstalled, usePluginsLoaded } from '@studio/plugins/PluginContext';
import { InboxTimeline } from '@studio/prototypes/notifications/inbox/InboxTimeline';
import {
  useMarkAllNotifEventsRead,
  useNotifEventList,
  useSetNotifEventRead,
} from '@studio/prototypes/notifications/notif/queries';
import {
  NOTIF_PLUGIN_NAME,
  notifEventToInboxItem,
} from '@studio/prototypes/notifications/notif/utils';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { useMemo, useState, type FC, type ReactNode } from 'react';

type Filter = 'all' | 'unread';

const PAGE_SIZE = 50;
const MAX_LIMIT = 1000;

const Message: FC<{ heading: string; children?: ReactNode }> = ({ heading, children }) => (
  <Stack align="center" gap="density-sm" className="px-6 py-16 text-center">
    <BellOff size="24" className="text-secondary" />
    <Text kind="label/bold/md">{heading}</Text>
    {children}
  </Stack>
);

export const NotificationsRoute: FC = () => {
  useBreadcrumbs({ items: [{ slotLabel: 'Notifications' }] });
  const workspace = useWorkspaceFromPath();
  const pluginsLoaded = usePluginsLoaded();
  const installed = usePluginInstalled(NOTIF_PLUGIN_NAME);
  const [filter, setFilter] = useState<Filter>('unread');
  const [limit, setLimit] = useState(PAGE_SIZE);

  const { data, isPending, isError, isFetching, refetch } = useNotifEventList(
    workspace,
    { limit, unreadOnly: filter === 'unread' },
    { enabled: installed }
  );
  const { mutate: setRead } = useSetNotifEventRead(workspace);
  const { mutate: markAllRead, isPending: markingAll } = useMarkAllNotifEventsRead(workspace);

  const items = useMemo(() => (data?.data ?? []).map(notifEventToInboxItem), [data]);
  const unreadCount = data?.unread_count ?? 0;
  const canLoadMore = (data?.data.length ?? 0) >= limit && limit < MAX_LIMIT;

  const changeFilter = (value: string) => {
    setFilter(value === 'unread' ? 'unread' : 'all');
    setLimit(PAGE_SIZE);
  };

  const renderBody = (): ReactNode => {
    if (pluginsLoaded && !installed) {
      return (
        <Message heading="Notifications aren't available">
          <Text kind="body/regular/sm" className="text-secondary">
            The notif plugin isn&apos;t installed on this platform.
          </Text>
        </Message>
      );
    }
    if (isPending) {
      return (
        <Stack align="center" className="py-16">
          <Spinner slotDescription="Loading notifications" />
        </Stack>
      );
    }
    if (isError) {
      return (
        <Message heading="Couldn't load notifications">
          <Button kind="secondary" size="small" onClick={() => void refetch()}>
            Retry
          </Button>
        </Message>
      );
    }
    if (items.length === 0) {
      return filter === 'unread' ? (
        <Message heading="You're all caught up">
          <Text kind="body/regular/sm" className="text-secondary">
            There are no unread notifications in this workspace.
          </Text>
        </Message>
      ) : (
        <Message heading="No notifications yet">
          <Text kind="body/regular/sm" className="text-secondary">
            Updates about jobs, evaluations, and agents will appear here.
          </Text>
        </Message>
      );
    }
    return (
      <Stack gap="density-lg">
        <InboxTimeline
          items={items}
          onOpenItem={(item) => {
            if (!item.read) {
              setRead({ eventId: item.id, read: true });
            }
          }}
          renderAction={(item) => (
            <Button
              kind="tertiary"
              size="small"
              onClick={() => setRead({ eventId: item.id, read: !item.read })}
              data-testid="proto-notification-toggle-read"
            >
              {item.read ? 'Mark as unread' : 'Mark as read'}
            </Button>
          )}
        />
        {canLoadMore && (
          <Button
            kind="secondary"
            className="self-center"
            disabled={isFetching}
            onClick={() => setLimit((current) => Math.min(current + PAGE_SIZE, MAX_LIMIT))}
          >
            {isFetching ? 'Loading…' : 'Load more'}
          </Button>
        )}
      </Stack>
    );
  };

  return (
    <AccessibleTitle title="Notifications">
      <Stack className="h-full" gap="density-2xl" padding="density-2xl">
        <PageHeader
          className="p-0"
          slotHeading="Notifications"
          slotDescription="Updates from jobs, evaluations, and agents in this workspace."
          slotActions={
            <Button
              kind="secondary"
              disabled={!installed || unreadCount === 0 || markingAll}
              onClick={() => markAllRead()}
            >
              Mark all as read
            </Button>
          }
        />
        <Stack gap="density-xl" className="w-full" data-testid="proto-notifications-page">
          <SegmentedControl
            value={filter}
            onValueChange={changeFilter}
            items={[
              { value: 'unread', children: unreadCount > 0 ? `Unread (${unreadCount})` : 'Unread' },
              { value: 'all', children: 'All' },
            ]}
            className="self-start"
          />
          {renderBody()}
        </Stack>
      </Stack>
    </AccessibleTitle>
  );
};
