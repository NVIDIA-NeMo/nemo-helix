// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SpanKind } from '@nemo/sdk/generated/platform/schema';
import { Badge } from '@nvidia/foundations-react-core';
import {
  getSpanKindColorClass,
  getSpanKindConfig,
} from '@studio/components/SpanKindBadge/spanKindConfig';
import type { FC } from 'react';

export interface SpanKindBadgeProps {
  kind: SpanKind | string | undefined;
}

/** Outline badge identifying a span's kind (Agent, LLM, Tool, …). */
export const SpanKindBadge: FC<SpanKindBadgeProps> = ({ kind }) => {
  const config = getSpanKindConfig(kind);
  const Icon = config.icon;

  return (
    <Badge color={config.color} kind="outline">
      <Icon className={`size-3 shrink-0 ${getSpanKindColorClass(kind)}`} role="img" aria-hidden />
      {config.label}
    </Badge>
  );
};
