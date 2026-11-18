// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { InboxItem, InboxState } from '@studio/prototypes/notifications/inbox/types';
import { summarizeInbox } from '@studio/prototypes/notifications/inbox/utils';
import { useCallback, useMemo, useState } from 'react';

export const useInbox = (initialItems: InboxItem[]): InboxState => {
  const [items, setItems] = useState(initialItems);

  const add = useCallback((item: InboxItem) => {
    setItems((prev) => (prev.some(({ id }) => id === item.id) ? prev : [item, ...prev]));
  }, []);

  const dismiss = useCallback((id: string) => {
    setItems((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const summary = useMemo(() => summarizeInbox(items), [items]);

  return { items, summary, add, dismiss };
};
