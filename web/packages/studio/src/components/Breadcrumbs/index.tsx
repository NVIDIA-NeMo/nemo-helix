// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Breadcrumbs as KuiBreadcrumbs } from '@nvidia/foundations-react-core';
import { WORKSPACE_BREADCRUMB_ITEM } from '@studio/components/Breadcrumbs/constants';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { FC, useMemo } from 'react';
import { Link, useParams } from 'react-router';

// Breadcrumb links navigate "up" the hierarchy, so query/hash from the current detail context is irrelevant at the parent level and only leaks state.
// Items that need theirs — a parent whose tab comes from `?tab=` — opt out with `preserveQuery`.
const pathnameOnly = (href: string) => href.split(/[?#]/)[0];

export const Breadcrumbs: FC = () => {
  const { breadcrumbs } = useBreadcrumbs();
  const { workspace } = useParams();

  const items = useMemo(() => {
    const allItems = [];
    if (workspace) {
      allItems.push(WORKSPACE_BREADCRUMB_ITEM);
    }
    return allItems.concat(
      breadcrumbs.map(({ href = '#', slotLabel, preserveQuery }) => ({
        children: <Link to={preserveQuery ? href : pathnameOnly(href)}>{slotLabel}</Link>,
      }))
    );
  }, [breadcrumbs, workspace]);

  return <KuiBreadcrumbs items={items} />;
};
