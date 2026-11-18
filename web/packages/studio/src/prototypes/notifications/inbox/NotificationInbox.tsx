// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BellOff } from '@nemo/icons/icons';
import { Button, Divider, Stack, Text } from '@nvidia/foundations-react-core';
import { InboxEntry } from '@studio/prototypes/notifications/inbox/InboxEntry';
import type { InboxItem } from '@studio/prototypes/notifications/inbox/types';
import { groupInboxItems } from '@studio/prototypes/notifications/inbox/utils';
import { useMemo, type FC } from 'react';

interface NotificationInboxProps {
  headingId: string;
  items: InboxItem[];
  pendingCount: number;
  onShowPending: () => void;
  onOpenItem: (item: InboxItem) => void;
  onDismissItem: (item: InboxItem) => void;
}

const EmptyInbox: FC = () => (
  <Stack align="center" gap="density-sm" className="px-6 py-10 text-center">
    <BellOff size="24" className="text-secondary" />
    <Text kind="label/bold/md">You&apos;re all caught up</Text>
    <Text kind="body/regular/sm" className="text-secondary">
      Updates about jobs, evaluations, and agents will appear here.
    </Text>
  </Stack>
);

export const NotificationInbox: FC<NotificationInboxProps> = ({
  headingId,
  items,
  pendingCount,
  onShowPending,
  onOpenItem,
  onDismissItem,
}) => {
  const groups = useMemo(() => groupInboxItems(items), [items]);

  return (
    <div
      className="flex max-h-[min(560px,calc(100vh-var(--nv-app-bar-height,0px)-48px))] w-[400px] max-w-[calc(100vw-32px)] flex-col"
      data-testid="proto-notification-inbox"
    >
      <div className="px-4 py-3">
        <Text id={headingId} kind="label/bold/md">
          Notifications
        </Text>
      </div>
      <Divider />
      {pendingCount > 0 && (
        <div className="flex justify-center px-4 pt-3">
          <Button kind="secondary" size="small" onClick={onShowPending}>
            {pendingCount} new notification{pendingCount === 1 ? '' : 's'}
          </Button>
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {groups.length === 0 ? (
          <EmptyInbox />
        ) : (
          <div className="flex flex-col gap-4 px-4 py-3">
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
                      onDismiss={onDismissItem}
                    />
                  ))}
                </ol>
              </section>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
