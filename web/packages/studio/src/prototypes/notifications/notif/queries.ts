// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  listNotifEvents,
  markAllNotifEventsRead,
  setNotifEventRead,
} from '@studio/prototypes/notifications/notif/api';
import type { NotifEventList, NotifListParams } from '@studio/prototypes/notifications/notif/types';
import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
  type QueryClient,
} from '@tanstack/react-query';
import { useCallback } from 'react';

export const notifQueryKey = (workspace: string) => ['notif', workspace] as const;

const notifListQueryKey = (workspace: string, params: NotifListParams) =>
  [...notifQueryKey(workspace), 'events', params] as const;

export const useNotifEventList = (
  workspace: string,
  params: NotifListParams,
  { enabled = true }: { enabled?: boolean } = {}
) =>
  useQuery({
    queryKey: notifListQueryKey(workspace, params),
    queryFn: () => listNotifEvents(workspace, params),
    placeholderData: keepPreviousData,
    enabled,
  });

const updateCachedLists = (
  queryClient: QueryClient,
  workspace: string,
  update: (list: NotifEventList) => NotifEventList
) => {
  queryClient.setQueriesData<NotifEventList>({ queryKey: notifQueryKey(workspace) }, (list) =>
    list ? update(list) : list
  );
};

const setRead =
  (eventId: string, read: boolean) =>
  (list: NotifEventList): NotifEventList => {
    const target = list.data.find(({ id }) => id === eventId);
    if (!target || target.read === read) {
      return list;
    }
    return {
      data: list.data.map((event) => (event.id === eventId ? { ...event, read } : event)),
      unread_count: Math.max(0, list.unread_count + (read ? -1 : 1)),
    };
  };

const readAll = (list: NotifEventList): NotifEventList => ({
  data: list.data.map((event) => (event.read ? event : { ...event, read: true })),
  unread_count: 0,
});

export const useInvalidateNotifEvents = (workspace: string) => {
  const queryClient = useQueryClient();
  return useCallback(
    () => queryClient.invalidateQueries({ queryKey: notifQueryKey(workspace) }),
    [queryClient, workspace]
  );
};

export const useSetNotifEventRead = (workspace: string) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ eventId, read }: { eventId: string; read: boolean }) =>
      setNotifEventRead(workspace, eventId, read),
    onMutate: async ({ eventId, read }) => {
      await queryClient.cancelQueries({ queryKey: notifQueryKey(workspace) });
      updateCachedLists(queryClient, workspace, setRead(eventId, read));
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: notifQueryKey(workspace) }),
  });
};

export const useMarkAllNotifEventsRead = (workspace: string) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => markAllNotifEventsRead(workspace),
    onMutate: async () => {
      await queryClient.cancelQueries({ queryKey: notifQueryKey(workspace) });
      updateCachedLists(queryClient, workspace, readAll);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: notifQueryKey(workspace) }),
  });
};
