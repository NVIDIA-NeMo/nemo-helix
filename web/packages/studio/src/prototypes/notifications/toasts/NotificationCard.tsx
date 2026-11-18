// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LoadingButton } from '@nemo/common/src/components/LoadingButton';
import { Button, Notification } from '@nvidia/foundations-react-core';
import { NOTIFICATION_DISMISS_MS } from '@studio/prototypes/notifications/toasts/constants';
import type { NotifyOptions } from '@studio/prototypes/notifications/toasts/types';
import { useAutoDismiss } from '@studio/prototypes/notifications/toasts/useAutoDismiss';
import cn from 'classnames';
import { useState, type FC } from 'react';
import { Link } from 'react-router';

interface NotificationCardProps extends NotifyOptions {
  onDismiss: () => void;
  className?: string;
}

type Pending = 'accept' | 'decline' | null;

export const NotificationCard: FC<NotificationCardProps> = ({
  status,
  heading,
  description,
  link,
  decision,
  durationMs,
  onDismiss,
  className,
}) => {
  const [pending, setPending] = useState<Pending>(null);
  const effectiveDuration =
    decision !== undefined ? false : (durationMs ?? NOTIFICATION_DISMISS_MS[status]);
  const pauseHandlers = useAutoDismiss(pending ? false : effectiveDuration, onDismiss);

  const runDecision = async (choice: Exclude<Pending, null>) => {
    if (!decision) {
      return;
    }
    setPending(choice);
    try {
      await decision[choice].onAction();
      onDismiss();
    } catch {
      setPending(null);
    }
  };

  const linkButton = link && (
    <Button kind={decision ? 'tertiary' : 'secondary'} size="small" asChild>
      <Link to={link.href} onClick={onDismiss}>
        {link.label}
      </Link>
    </Button>
  );

  const footer =
    decision !== undefined ? (
      <div className="flex w-full items-center gap-2">
        {linkButton}
        <div className="ml-auto flex items-center gap-2">
          <LoadingButton
            kind="secondary"
            color="neutral"
            size="small"
            height={32}
            loading={pending === 'decline'}
            disabled={pending !== null}
            onClick={() => runDecision('decline')}
          >
            {decision.decline.label}
          </LoadingButton>
          <LoadingButton
            kind="primary"
            color="neutral"
            size="small"
            height={32}
            loading={pending === 'accept'}
            disabled={pending !== null}
            onClick={() => runDecision('accept')}
          >
            {decision.accept.label}
          </LoadingButton>
        </div>
      </div>
    ) : (
      linkButton
    );

  return (
    <Notification
      status={status}
      slotSubheading={description}
      slotFooter={footer || undefined}
      onClose={pending ? undefined : onDismiss}
      role={status === 'error' || decision ? 'alert' : 'status'}
      className={cn('w-[380px] shadow-lg', className)}
      data-testid="proto-notification"
      {...pauseHandlers}
    >
      {heading}
    </Notification>
  );
};
