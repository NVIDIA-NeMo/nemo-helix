// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Bell } from '@nemo/icons/icons';
import {
  Button,
  PopoverContent,
  PopoverRoot,
  PopoverTrigger,
  StatusIndicator,
} from '@nvidia/foundations-react-core';
import { MAX_INBOX_ITEMS } from '@studio/prototypes/notifications/inbox/fixtures';
import { NotificationInbox } from '@studio/prototypes/notifications/inbox/NotificationInbox';
import type { InboxItem, InboxState } from '@studio/prototypes/notifications/inbox/types';
import { SEVERITY_COLOR, hasNotification } from '@studio/prototypes/notifications/utils';
import { useId, useState, type FC } from 'react';

interface NotificationDropdownProps {
  inbox: InboxState;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
}

const idsOf = (items: InboxItem[]): ReadonlySet<string> => new Set(items.map(({ id }) => id));

export const NotificationDropdown: FC<NotificationDropdownProps> = ({
  inbox,
  defaultOpen = false,
  onOpenChange,
}) => {
  const { items, unread, markRead, markAllRead, dismiss } = inbox;
  const [open, setOpen] = useState(defaultOpen);
  const [shownIds, setShownIds] = useState<ReadonlySet<string> | null>(() =>
    defaultOpen ? idsOf(items) : null
  );
  const headingId = useId();

  const handleOpenChange = (next: boolean) => {
    setOpen(next);
    setShownIds(next ? idsOf(items) : null);
    onOpenChange?.(next);
  };

  const shown = shownIds === null ? items : items.filter(({ id }) => shownIds.has(id));
  const pendingCount = items.length - shown.length;

  const label = hasNotification(unread)
    ? `Notifications (${unread.count} unread)`
    : 'Notifications';

  return (
    <PopoverRoot open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <Button
          aria-label={label}
          color="neutral"
          kind="tertiary"
          size="medium"
          className="relative"
          data-testid="proto-notification-bell"
        >
          <Bell />
          {hasNotification(unread) && (
            <StatusIndicator
              aria-hidden
              size="small"
              color={SEVERITY_COLOR[unread.severity]}
              className="absolute right-1.5 top-1.5"
              data-testid="proto-notification-bell-dot"
            />
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        side="bottom"
        align="end"
        aria-labelledby={headingId}
        className="max-h-none! overflow-hidden! p-0!"
      >
        <NotificationInbox
          headingId={headingId}
          items={shown.slice(0, MAX_INBOX_ITEMS)}
          pendingCount={pendingCount}
          hasUnread={hasNotification(unread)}
          onShowPending={() => setShownIds(idsOf(items))}
          onOpenItem={(item) => {
            markRead(item.id);
            handleOpenChange(false);
          }}
          onDismissItem={(item) => dismiss(item.id)}
          onMarkAllRead={markAllRead}
          onViewAll={() => handleOpenChange(false)}
        />
      </PopoverContent>
    </PopoverRoot>
  );
};
