// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { type LucideIcon } from 'lucide-react';

export interface SubtleToolAction {
  readonly detail: string;
  readonly details?: readonly string[];
  readonly Icon: LucideIcon;
  readonly invocation: string;
  readonly invocations?: readonly string[];
  readonly message: string;
  readonly title?: string;
  readonly toolCallId: string;
  readonly toolName: string;
}

export interface FileChangeSummary {
  readonly action: 'Edited' | 'Wrote';
  readonly additions: number;
  readonly deletions: number;
  readonly path: string;
  readonly reviewContent: string;
}
