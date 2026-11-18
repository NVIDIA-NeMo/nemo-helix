// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import { InboxEntry } from '@studio/prototypes/notifications/inbox/InboxEntry';
import type { InboxItem } from '@studio/prototypes/notifications/inbox/types';
import { groupInboxItems } from '@studio/prototypes/notifications/inbox/utils';
import { useMemo, type FC, type ReactNode } from 'react';

interface InboxTimelineProps {
  items: InboxItem[];
  onOpenItem?: (item: InboxItem) => void;
  renderAction?: (item: InboxItem) => ReactNode;
}

export const InboxTimeline: FC<InboxTimelineProps> = ({ items, onOpenItem, renderAction }) => {
  const groups = useMemo(() => groupInboxItems(items), [items]);

  return (
    <div className="flex flex-col gap-4">
      {groups.map((group) => (
        <section key={group.label} aria-label={group.label} className="flex flex-col gap-2">
          <Text kind="label/bold/sm" className="text-secondary">
            {group.label}
          </Text>
          <ol className="m-0 flex list-none flex-col p-0">
            {group.items.map((item, index) => (
              <InboxEntry
                key={item.id}
                item={item}
                isLast={index === group.items.length - 1}
                onOpen={onOpenItem}
                slotAction={renderAction?.(item)}
              />
            ))}
          </ol>
        </section>
      ))}
    </div>
  );
};
