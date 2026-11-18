// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@nvidia/foundations-react-core';
import { BriefToastCard } from '@studio/prototypes/notifications/toasts/BriefToastCard';
import { MAX_VISIBLE_NOTIFICATIONS } from '@studio/prototypes/notifications/toasts/constants';
import { NotificationCard } from '@studio/prototypes/notifications/toasts/NotificationCard';
import type {
  BriefToastOptions,
  NotificationEntry,
  NotifyOptions,
  ProtoNotificationsContextValue,
} from '@studio/prototypes/notifications/toasts/types';
import { ProtoNotificationsContext } from '@studio/prototypes/notifications/toasts/useProtoNotifications';
import {
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FC,
  type PropsWithChildren,
} from 'react';

export const ProtoNotificationProvider: FC<PropsWithChildren> = ({ children }) => {
  const [entries, setEntries] = useState<NotificationEntry[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el || entries.length === 0) {
      return;
    }
    if (el.matches(':popover-open')) {
      el.hidePopover();
    }
    el.showPopover();
  }, [entries.length]);

  const dismiss = useCallback((id: string) => {
    setEntries((prev) => prev.filter((entry) => entry.id !== id));
  }, []);

  const notify = useCallback((options: NotifyOptions) => {
    const id = `notification-${crypto.randomUUID()}`;
    setEntries((prev) => [...prev, { id, kind: 'notification', ...options }]);
    return id;
  }, []);

  const toast = useCallback((options: BriefToastOptions) => {
    const id = `toast-${crypto.randomUUID()}`;
    setEntries((prev) => [...prev, { id, kind: 'toast', ...options }]);
    return id;
  }, []);

  const value = useMemo<ProtoNotificationsContextValue>(
    () => ({ notify, toast, dismiss }),
    [notify, toast, dismiss]
  );

  const visible = entries.slice(0, MAX_VISIBLE_NOTIFICATIONS);
  const queued = entries.length - visible.length;

  return (
    <ProtoNotificationsContext.Provider value={value}>
      {children}
      <div
        ref={containerRef}
        popover="manual"
        aria-live="polite"
        className="fixed top-[calc(var(--nv-app-bar-height,0px)+1rem)] right-4 left-auto bottom-auto m-0 hidden flex-col items-end gap-3 bg-transparent p-0 z-1100 max-w-md overflow-visible [&:popover-open]:flex"
      >
        {visible.map((entry) =>
          entry.kind === 'notification' ? (
            <NotificationCard key={entry.id} {...entry} onDismiss={() => dismiss(entry.id)} />
          ) : (
            <BriefToastCard key={entry.id} {...entry} onDismiss={() => dismiss(entry.id)} />
          )
        )}
        {queued > 0 && (
          <Text kind="body/regular/xs" className="text-secondary">
            +{queued} more
          </Text>
        )}
      </div>
    </ProtoNotificationsContext.Provider>
  );
};
