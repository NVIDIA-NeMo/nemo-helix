// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react';

export interface EntityCardStat {
  readonly key: string;
  readonly icon: ReactNode;
  readonly label: string;
}

export interface EntityCardTag {
  readonly key: string;
  readonly label: string;
  readonly title?: string;
}
