// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { BreadcrumbsItemProps } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import type { Dispatch, SetStateAction } from 'react';

export type SetBreadcrumbsFn = Dispatch<SetStateAction<BreadcrumbsItemProps[]>>;

export interface BreadcrumbsContextValue {
  breadcrumbs: BreadcrumbsItemProps[];
  setBreadcrumbs: SetBreadcrumbsFn;
}
