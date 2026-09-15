// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { BreadcrumbsContextValue } from '@studio/providers/breadcrumbs/types';
import { createContext, useContext, useEffect, type ReactNode } from 'react';

export const BreadcrumbsContext = createContext<BreadcrumbsContextValue | null>(null);

export type BreadcrumbsItemProps = {
  href?: string;
  slotLabel: ReactNode;
  preserveQuery?: boolean;
};

export type BreadCrumbItemsProps = {
  items?: BreadcrumbsItemProps[];
};

export const useBreadcrumbs = ({ items }: BreadCrumbItemsProps = {}) => {
  const context = useContext(BreadcrumbsContext);
  if (!context) {
    throw new Error('useBreadcrumbs must be used within a BreadcrumbsProvider');
  }
  useEffect(() => {
    if (items) {
      context.setBreadcrumbs(items);
      return () => {
        context.setBreadcrumbs([]);
      };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return context;
};
