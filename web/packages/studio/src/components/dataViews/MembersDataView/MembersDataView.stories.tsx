// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { WorkspaceMemberListResponse } from '@nemo/sdk/generated/platform/schema';
import type { Meta, StoryObj } from '@storybook/react';
import { MembersDataView } from '@studio/components/dataViews/MembersDataView';
import { http, HttpResponse } from 'msw';

const MEMBERS_API = '/apis/entities/v2/workspaces/:workspace/members';

const membersPage: WorkspaceMemberListResponse = {
  data: [
    {
      principal: 'ada@example.com',
      roles: ['Admin'],
      granted_at: '2026-01-04T10:00:00Z',
      granted_by: 'grace@example.com',
    },
    {
      principal: 'linus@example.com',
      roles: ['Viewer'],
      granted_at: '2026-02-18T16:30:00Z',
      granted_by: 'ada@example.com',
    },
  ],
};

const meta = {
  component: MembersDataView,
  title: 'DataViews/MembersDataView',
  args: {
    workspace: 'default',
    onAddMember: () => {},
    onEditMember: () => {},
    onRemoveMember: () => {},
  },
} satisfies Meta<typeof MembersDataView>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithData: Story = {
  parameters: {
    msw: {
      handlers: [http.get(MEMBERS_API, () => HttpResponse.json(membersPage))],
    },
  },
};

/**
 * The 403 a non-admin gets. Needs a real RBAC-enabled backend to reproduce by hand, so this story
 * is the practical way to review the copy — note there is no "Refresh Page" call to action, since
 * retrying cannot grant the permission.
 */
export const Forbidden: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get(MEMBERS_API, () => HttpResponse.json({ detail: 'Forbidden' }, { status: 403 })),
      ],
    },
  },
};

export const ServerError: Story = {
  parameters: {
    msw: {
      handlers: [
        http.get(MEMBERS_API, () =>
          HttpResponse.json({ detail: 'Something went wrong' }, { status: 500 })
        ),
      ],
    },
  },
};
