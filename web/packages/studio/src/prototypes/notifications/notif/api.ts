// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { platformFetch } from '@nemo/sdk/src/utils/platformRequest';
import type {
  NotifEvent,
  NotifEventList,
  NotifListParams,
} from '@studio/prototypes/notifications/notif/types';
import { notifEventsPath, toNotifEvent } from '@studio/prototypes/notifications/notif/utils';

const requestJson = async (path: string, init?: RequestInit): Promise<unknown> => {
  const response = await platformFetch(path, init);
  if (!response.ok) {
    throw new Error(`Notification request failed: ${response.status}`);
  }
  return response.json();
};

export const listNotifEvents = async (
  workspace: string,
  { limit, unreadOnly }: NotifListParams
): Promise<NotifEventList> => {
  const query = new URLSearchParams({ limit: String(limit), unread_only: String(unreadOnly) });
  const body = (await requestJson(`${notifEventsPath(workspace)}?${query}`)) as {
    data?: unknown[];
    unread_count?: number;
  };
  return {
    data: (body.data ?? []).flatMap((value) => toNotifEvent(value) ?? []),
    unread_count: body.unread_count ?? 0,
  };
};

export const setNotifEventRead = async (
  workspace: string,
  eventId: string,
  read: boolean
): Promise<NotifEvent | undefined> =>
  toNotifEvent(
    await requestJson(
      `${notifEventsPath(workspace)}/${encodeURIComponent(eventId)}/${read ? 'read' : 'unread'}`,
      { method: 'POST' }
    )
  );

export const markAllNotifEventsRead = async (workspace: string): Promise<void> => {
  await requestJson(`${notifEventsPath(workspace)}/read-all`, { method: 'POST' });
};
