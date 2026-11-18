// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BellOff, Close } from '@nemo/icons/icons';
import { Anchor, Button, Divider, Stack, Text } from '@nvidia/foundations-react-core';
import { InboxTimeline } from '@studio/prototypes/notifications/inbox/InboxTimeline';
import type { InboxItem } from '@studio/prototypes/notifications/inbox/types';
import type { FC } from 'react';
import { Link } from 'react-router';

interface NotificationInboxProps {
  headingId: string;
  items: InboxItem[];
  pendingCount: number;
  hasUnread: boolean;
  viewAllHref?: string;
  onShowPending: () => void;
  onOpenItem: (item: InboxItem) => void;
  onMarkRead: (item: InboxItem) => void;
  onMarkAllRead: () => void;
  onViewAll: () => void;
}

const EmptyInbox: FC = () => (
  <Stack align="center" gap="density-sm" className="px-6 py-10 text-center">
    <BellOff size="24" className="text-secondary" />
    <Text kind="label/bold/md">You&apos;re all caught up</Text>
    <Text kind="body/regular/sm" className="text-secondary">
      New updates about jobs, evaluations, and agents will appear here.
    </Text>
  </Stack>
);

export const NotificationInbox: FC<NotificationInboxProps> = ({
  headingId,
  items,
  pendingCount,
  hasUnread,
  viewAllHref,
  onShowPending,
  onOpenItem,
  onMarkRead,
  onMarkAllRead,
  onViewAll,
}) => (
  <div
    className="flex max-h-[min(560px,calc(100vh-var(--nv-app-bar-height,0px)-48px))] w-[400px] max-w-[calc(100vw-32px)] flex-col"
    data-testid="proto-notification-inbox"
  >
    <div className="flex items-center justify-between gap-2 px-4 py-3">
      <Text id={headingId} kind="label/bold/md">
        Notifications
      </Text>
      <Button kind="tertiary" size="small" disabled={!hasUnread} onClick={onMarkAllRead}>
        Mark all as read
      </Button>
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
      {items.length === 0 ? (
        <EmptyInbox />
      ) : (
        <div className="px-4 py-3">
          <InboxTimeline
            items={items}
            onOpenItem={onOpenItem}
            renderAction={(item) => (
              <Button
                aria-label={`Mark ${item.title} as read`}
                color="neutral"
                kind="tertiary"
                size="small"
                onClick={() => onMarkRead(item)}
                data-testid="proto-inbox-entry-mark-read"
              >
                <Close size="16" />
              </Button>
            )}
          />
        </div>
      )}
    </div>
    {viewAllHref && (
      <>
        <Divider />
        <div className="flex justify-center px-4 py-3">
          <Anchor asChild textKind="body/regular/sm">
            <Link to={viewAllHref} onClick={onViewAll}>
              View all notifications
            </Link>
          </Anchor>
        </div>
      </>
    )}
  </div>
);
