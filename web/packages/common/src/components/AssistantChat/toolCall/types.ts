// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AssistantToolArgs } from '@nemo/common/src/components/AssistantChat/toolParts';

export interface SubtleToolAction {
  readonly args: AssistantToolArgs;
  readonly toolCallId: string;
  readonly toolName: string;
}

export interface SubtleToolActionGroup {
  readonly actions: readonly [SubtleToolAction, ...SubtleToolAction[]];
  readonly repeatedTitle?: string;
}

export interface FileChangeSummary {
  readonly action: 'Edited' | 'Wrote';
  readonly additions: number;
  readonly deletions: number;
  /** True when matched by argument shape rather than a known tool name, so it may be wrong. */
  readonly inferred?: boolean;
  readonly path: string;
  readonly reviewContent: string;
}
