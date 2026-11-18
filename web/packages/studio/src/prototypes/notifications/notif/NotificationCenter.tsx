// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useWorkspaceFromPathIfExists } from '@studio/hooks/useWorkspaceFromPath';
import { usePluginInstalled } from '@studio/plugins/PluginContext';
import { NotificationDropdown } from '@studio/prototypes/notifications/inbox/NotificationDropdown';
import { useInbox } from '@studio/prototypes/notifications/inbox/useInbox';
import type { NotifEvent } from '@studio/prototypes/notifications/notif/types';
import { useNotifEvents } from '@studio/prototypes/notifications/notif/useNotifEvents';
import {
  NOTIF_PLUGIN_NAME,
  inboxItemToNotifyOptions,
  notifEventToInboxItem,
} from '@studio/prototypes/notifications/notif/utils';
import { ProtoNotificationProvider } from '@studio/prototypes/notifications/toasts/ProtoNotificationProvider';
import { useProtoNotifications } from '@studio/prototypes/notifications/toasts/useProtoNotifications';
import { useCallback, useRef, type FC } from 'react';

const WorkspaceNotificationCenter: FC<{ workspace: string }> = ({ workspace }) => {
  const { notify } = useProtoNotifications();
  const inbox = useInbox([]);
  const { add } = inbox;
  const inboxOpen = useRef(false);

  const handleEvent = useCallback(
    (event: NotifEvent) => {
      const item = notifEventToInboxItem(event);
      add(item);
      if (!inboxOpen.current) {
        notify(inboxItemToNotifyOptions(item));
      }
    },
    [add, notify]
  );

  useNotifEvents(workspace, handleEvent);

  return (
    <NotificationDropdown
      inbox={inbox}
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
