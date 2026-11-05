// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { BreadcrumbsItemFieldProps } from '@nvidia/foundations-react-core';
import { WorkspaceDropdown } from '@studio/components/WorkspaceDropdown';

export const WORKSPACE_BREADCRUMB_ITEM: BreadcrumbsItemFieldProps = {
  children: <WorkspaceDropdown />,
};
