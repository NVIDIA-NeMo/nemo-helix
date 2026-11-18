// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRelativeTimeSince } from '@nemo/common/src/components/RelativeTime';
import { formatAbsoluteTimestamp } from '@nemo/common/src/components/RelativeTime/util';
import { Close } from '@nemo/icons/icons';
import { Anchor, Button, StatusIndicator, Text } from '@nvidia/foundations-react-core';
import { NAV_LABELS } from '@studio/prototypes/notifications/fixtures';
import type { InboxItem } from '@studio/prototypes/notifications/inbox/types';
import { SEVERITY_LABEL } from '@studio/prototypes/notifications/inbox/utils';
import { SEVERITY_COLOR } from '@studio/prototypes/notifications/utils';
import cn from 'classnames';
import type { FC } from 'react';
import { Link } from 'react-router';

interface InboxEntryProps {
  item: InboxItem;
  isLast: boolean;
  onOpen: (item: InboxItem) => void;
  onDismiss: (item: InboxItem) => void;
}

export const InboxEntry: FC<InboxEntryProps> = ({ item, isLast, onOpen, onDismiss }) => {
  const timeSince = useRelativeTimeSince(item.createdAt);

  return (
    <li className="flex gap-3" data-testid="proto-inbox-entry">
      <div className="flex w-3 shrink-0 flex-col items-center">
        <span className="flex h-6 items-center">
          <StatusIndicator
            size="medium"
            color={SEVERITY_COLOR[item.severity]}
            role="img"
            aria-label={SEVERITY_LABEL[item.severity]}
          />
        </span>
        {!isLast && <span aria-hidden className="w-px flex-1 bg-(--border-color-base)" />}
      </div>
      <div className={cn('flex min-w-0 flex-1 flex-col gap-1', !isLast && 'pb-4')}>
        <div className="flex items-start gap-2">
          <Text
            kind={item.read ? 'body/regular/md' : 'body/bold/md'}
            className={cn('min-w-0 flex-1', item.read && 'text-secondary')}
          >
            {!item.read && <span className="sr-only">Unread: </span>}
            {item.title}
          </Text>
          <Button
            aria-label={`Dismiss ${item.title}`}
            color="neutral"
            kind="tertiary"
            size="small"
            className="-my-1 shrink-0"
            onClick={() => onDismiss(item)}
            data-testid="proto-inbox-entry-dismiss"
          >
            <Close size="16" />
          </Button>
        </div>
        <Text kind="body/regular/sm" className="text-secondary">
          <time dateTime={item.createdAt} title={formatAbsoluteTimestamp(item.createdAt)}>
            {timeSince}
          </time>
          {` · ${NAV_LABELS[item.sourceId]}`}
        </Text>
        {item.description && (
          <Text
            kind="body/regular/sm"
            className={cn('line-clamp-2', item.read && 'text-secondary')}
          >
            {item.description}
          </Text>
        )}
        {item.link && (
          <Anchor asChild textKind="body/regular/sm" className="w-fit">
            <Link to={item.link.href} onClick={() => onOpen(item)}>
              {item.link.label}
            </Link>
          </Anchor>
        )}
      </div>
    </li>
  );
};
