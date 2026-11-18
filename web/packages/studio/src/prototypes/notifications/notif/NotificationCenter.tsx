// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ROUTES } from '@studio/constants/routes';
import { useWorkspaceFromPathIfExists } from '@studio/hooks/useWorkspaceFromPath';
import { usePluginInstalled } from '@studio/plugins/PluginContext';
import { NotificationDropdown } from '@studio/prototypes/notifications/inbox/NotificationDropdown';
import { summarizeUnread } from '@studio/prototypes/notifications/inbox/utils';
import {
  useInvalidateNotifEvents,
  useMarkAllNotifEventsRead,
  useNotifEventList,
  useSetNotifEventRead,
} from '@studio/prototypes/notifications/notif/queries';
import type { NotifEvent, NotifListParams } from '@studio/prototypes/notifications/notif/types';
import { useNotifEvents } from '@studio/prototypes/notifications/notif/useNotifEvents';
import {
  NOTIF_PLUGIN_NAME,
  inboxItemToNotifyOptions,
  notifEventToInboxItem,
} from '@studio/prototypes/notifications/notif/utils';
import { ProtoNotificationProvider } from '@studio/prototypes/notifications/toasts/ProtoNotificationProvider';
import { useProtoNotifications } from '@studio/prototypes/notifications/toasts/useProtoNotifications';
import { useCallback, useMemo, useRef, type FC } from 'react';
import { generatePath } from 'react-router';

const UNREAD_PARAMS: NotifListParams = { limit: 50, unreadOnly: true };

const WorkspaceNotificationCenter: FC<{ workspace: string }> = ({ workspace }) => {
  const { notify } = useProtoNotifications();
  const inboxOpen = useRef(false);
  const { data } = useNotifEventList(workspace, UNREAD_PARAMS);
  const invalidate = useInvalidateNotifEvents(workspace);
  const { mutate: setRead } = useSetNotifEventRead(workspace);
  const { mutate: markAllRead } = useMarkAllNotifEventsRead(workspace);

  const unread = useMemo(
    () => (data?.data ?? []).filter((event) => !event.read).map(notifEventToInboxItem),
    [data]
  );
  const summary = useMemo(
    () => summarizeUnread(unread, data?.unread_count ?? 0),
    [unread, data?.unread_count]
  );

  const handleEvent = useCallback(
    (event: NotifEvent) => {
      void invalidate();
      if (!inboxOpen.current) {
        notify(inboxItemToNotifyOptions(notifEventToInboxItem(event)));
      }
    },
    [invalidate, notify]
  );

  useNotifEvents(workspace, handleEvent);

  return (
    <NotificationDropdown
      unread={unread}
      summary={summary}
      onMarkRead={(item) => setRead({ eventId: item.id, read: true })}
      onMarkAllRead={() => markAllRead()}
      viewAllHref={generatePath(ROUTES.workspace.notifications, { workspace })}
      onOpenChange={(open) => {
        inboxOpen.current = open;
      }}
    />
  );
};

export const NotificationCenter: FC = () => {
  const workspace = useWorkspaceFromPathIfExists();
  const installed = usePluginInstalled(NOTIF_PLUGIN_NAME);

  if (!workspace || !installed) {
    return null;
  }

  return (
    <ProtoNotificationProvider key={workspace}>
      <WorkspaceNotificationCenter workspace={workspace} />
    </ProtoNotificationProvider>
  );
};
