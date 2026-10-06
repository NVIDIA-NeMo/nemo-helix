// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from '@storybook/react';
import { UnauthorizedWorkspace } from '@studio/components/Layouts/UnauthorizedWorkspace';

const meta: Meta<typeof UnauthorizedWorkspace> = {
  component: UnauthorizedWorkspace,
  title: 'Studio/Layouts/UnauthorizedWorkspace',
};

export default meta;

type Story = StoryObj<typeof UnauthorizedWorkspace>;

/**
 * Rendered by WorkspaceGuard when the workspace access check returns 403. The
 * footer actions come from ErrorMessage's defaults so a user whose access check
 * went stale can recover without hand-reloading the browser.
 */
export const Default: Story = {};
