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
import { NotificationInbox } from '@studio/prototypes/notifications/inbox/NotificationInbox';
import type { InboxItem } from '@studio/prototypes/notifications/inbox/types';
import type { NavNotification } from '@studio/prototypes/notifications/types';
import { SEVERITY_COLOR, hasNotification } from '@studio/prototypes/notifications/utils';
import { useId, useState, type FC } from 'react';

interface NotificationDropdownProps {
  unread: InboxItem[];
  summary: NavNotification | undefined;
  onMarkRead: (item: InboxItem) => void;
  onMarkAllRead: () => void;
  viewAllHref?: string;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
}

const idsOf = (items: InboxItem[]): ReadonlySet<string> => new Set(items.map(({ id }) => id));

export const NotificationDropdown: FC<NotificationDropdownProps> = ({
  unread,
  summary,
  onMarkRead,
  onMarkAllRead,
  viewAllHref,
  defaultOpen = false,
  onOpenChange,
}) => {
  const [open, setOpen] = useState(defaultOpen);
  const [shownIds, setShownIds] = useState<ReadonlySet<string> | null>(() =>
    defaultOpen ? idsOf(unread) : null
  );
  const headingId = useId();

  const handleOpenChange = (next: boolean) => {
    setOpen(next);
    setShownIds(next ? idsOf(unread) : null);
    onOpenChange?.(next);
  };

  const shown = shownIds === null ? unread : unread.filter(({ id }) => shownIds.has(id));
  const pendingCount = unread.length - shown.length;

  const label = hasNotification(summary)
    ? `Notifications (${summary.count} unread)`
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
          {hasNotification(summary) && (
            <StatusIndicator
              aria-hidden
              size="small"
              color={SEVERITY_COLOR[summary.severity]}
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
          items={shown}
          pendingCount={pendingCount}
          hasUnread={hasNotification(summary)}
          viewAllHref={viewAllHref}
          onShowPending={() => setShownIds(idsOf(unread))}
          onOpenItem={(item) => {
            onMarkRead(item);
            handleOpenChange(false);
          }}
          onMarkRead={onMarkRead}
          onMarkAllRead={onMarkAllRead}
          onViewAll={() => handleOpenChange(false)}
        />
      </PopoverContent>
    </PopoverRoot>
  );
};
