// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { InboxItem, InboxState } from '@studio/prototypes/notifications/inbox/types';
import { summarizeUnread } from '@studio/prototypes/notifications/inbox/utils';
import { useCallback, useMemo, useState } from 'react';

export const useInbox = (initialItems: InboxItem[]): InboxState => {
  const [items, setItems] = useState(initialItems);

  const add = useCallback((item: InboxItem) => {
    setItems((prev) => (prev.some(({ id }) => id === item.id) ? prev : [item, ...prev]));
  }, []);

  const markRead = useCallback((id: string) => {
    setItems((prev) => prev.map((item) => (item.id === id ? { ...item, read: true } : item)));
  }, []);

  const markAllRead = useCallback(() => {
    setItems((prev) => prev.map((item) => (item.read ? item : { ...item, read: true })));
  }, []);

  const unread = useMemo(() => items.filter((item) => !item.read), [items]);
  const summary = useMemo(() => summarizeUnread(unread, unread.length), [unread]);

  return { items, unread, summary, add, markRead, markAllRead };
};
