// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AppBar, Avatar, Button, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { DocumentationLink } from '@studio/components/Layouts/GlobalNav/components/DocumentationLink';
import { NAV_HREFS } from '@studio/prototypes/notifications/fixtures';
import { SAMPLE_INBOX } from '@studio/prototypes/notifications/inbox/fixtures';
import { NotificationDropdown } from '@studio/prototypes/notifications/inbox/NotificationDropdown';
import type { InboxItem, InboxState } from '@studio/prototypes/notifications/inbox/types';
import { useInbox } from '@studio/prototypes/notifications/inbox/useInbox';
import { toInboxItem } from '@studio/prototypes/notifications/inbox/utils';
import { DEMO_EVENTS } from '@studio/prototypes/notifications/toasts/fixtures';
import { ProtoNotificationProvider } from '@studio/prototypes/notifications/toasts/ProtoNotificationProvider';
import { useProtoNotifications } from '@studio/prototypes/notifications/toasts/useProtoNotifications';
import { useEffect, useRef, useState, type FC, type ReactNode } from 'react';
import { useLocation } from 'react-router';

const STREAM_INTERVAL_MS = 4000;

interface ProtoAppBarProps {
  inbox: InboxState;
  defaultOpen?: boolean;
  onInboxOpenChange?: (open: boolean) => void;
}

const ProtoAppBar: FC<ProtoAppBarProps> = ({ inbox, defaultOpen, onInboxOpenChange }) => (
  <AppBar
    slotStart={<span>NeMo Studio</span>}
    slotEnd={
      <>
        <NotificationDropdown
          unread={inbox.unread}
          summary={inbox.summary}
          onMarkRead={(item) => inbox.markRead(item.id)}
          onMarkAllRead={inbox.markAllRead}
          viewAllHref={`${NAV_HREFS.dashboard}/notifications`}
          defaultOpen={defaultOpen}
          onOpenChange={onInboxOpenChange}
        />
        <DocumentationLink />
        <Avatar fallback="N" interactive />
      </>
    }
  />
);

const CurrentPath: FC = () => {
  const { pathname, search } = useLocation();
  return (
    <Text kind="body/regular/sm">
      Current path: <code>{`${pathname}${search}`}</code>
    </Text>
  );
};

const Shell: FC<{ appBar: ReactNode; children?: ReactNode }> = ({ appBar, children }) => (
  <div className="flex min-h-[720px] flex-col">
    {appBar}
    <div className="flex flex-col gap-3 p-6">{children}</div>
  </div>
);

interface PreviewArgs {
  initialItems: InboxItem[];
  defaultOpen: boolean;
}

const DropdownPreview: FC<PreviewArgs> = ({ initialItems, defaultOpen }) => {
  const inbox = useInbox(initialItems);
  return (
    <Shell appBar={<ProtoAppBar inbox={inbox} defaultOpen={defaultOpen} />}>
      <CurrentPath />
    </Shell>
  );
};

const meta = {
  title: 'Prototypes/Notifications/Notification Dropdown',
  component: DropdownPreview,
  args: {
    initialItems: SAMPLE_INBOX,
    defaultOpen: true,
  },
  argTypes: {
    initialItems: { control: false },
  },
  parameters: {
    layout: 'fullscreen',
    router: { initialPath: NAV_HREFS.dashboard },
  },
  render: (args) => <DropdownPreview key={String(args.defaultOpen)} {...args} />,
} satisfies Meta<typeof DropdownPreview>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Open: Story = {};

export const Closed: Story = {
  args: { defaultOpen: false },
};

export const AllCaughtUp: Story = {
  args: { initialItems: SAMPLE_INBOX.map((item) => ({ ...item, read: true })) },
};

const LivePreview: FC = () => {
  const api = useProtoNotifications();
  const inbox = useInbox(SAMPLE_INBOX);
  const [streaming, setStreaming] = useState(false);
  const nextEvent = useRef(0);
  const inboxOpen = useRef(false);

  const fire = (event: (typeof DEMO_EVENTS)[number]) => {
    const options = event.build(api);
    if (!inboxOpen.current) {
      api.notify(options);
    }
    inbox.add(toInboxItem(options, event.sourceId));
  };

  const fireRef = useRef(fire);
  useEffect(() => {
    fireRef.current = fire;
  });

  useEffect(() => {
    if (!streaming) {
      return;
    }
    const interval = setInterval(() => {
      fireRef.current(DEMO_EVENTS[nextEvent.current % DEMO_EVENTS.length]);
      nextEvent.current += 1;
    }, STREAM_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [streaming]);

  return (
    <Shell
      appBar={
        <ProtoAppBar
          inbox={inbox}
          onInboxOpenChange={(open) => {
            inboxOpen.current = open;
          }}
        />
      }
    >
      <div className="flex max-w-sm flex-col gap-3">
        <Text kind="label/bold/md">Simulate events</Text>
        {DEMO_EVENTS.map((event) => (
          <Button key={event.label} kind="secondary" onClick={() => fire(event)}>
            {event.label}
          </Button>
        ))}
        <Button kind={streaming ? 'primary' : 'tertiary'} onClick={() => setStreaming((s) => !s)}>
          {streaming ? 'Stop event stream' : `Stream an event every ${STREAM_INTERVAL_MS / 1000}s`}
        </Button>
        <Text kind="body/regular/sm">
          Each event raises a floating notification and lands in the bell&apos;s unread list.
          Marking an item read (×, or following its link) removes it from the dropdown. Start the
          stream, then open the bell: while it is open, floating notifications are suppressed and
          new events collect behind a &quot;new notifications&quot; button.
        </Text>
        <CurrentPath />
      </div>
    </Shell>
  );
};

export const Live: Story = {
  render: () => (
    <ProtoNotificationProvider>
      <LivePreview />
    </ProtoNotificationProvider>
  ),
};
