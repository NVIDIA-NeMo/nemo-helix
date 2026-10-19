// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Text } from '@nvidia/foundations-react-core';
import type { TextSegment } from '@studio/components/AnonymizerRecordView/types';
import { entityTagColor } from '@studio/routes/AnonymizerBuilderRoute/constants';
import { memo, type FC } from 'react';

interface HighlightedTextProps {
  readonly segments: readonly TextSegment[];
  readonly emptyMessage: string;
}

export const HighlightedText: FC<HighlightedTextProps> = memo(({ segments, emptyMessage }) =>
  segments.length ? (
    <Text className="whitespace-pre-wrap break-words" kind="body/regular/md">
      {segments.map((segment, index) => {
        if (!segment.label) return <span key={index}>{segment.text}</span>;
        const color = entityTagColor(segment.label);
        return (
          <span className="inline-flex items-center" key={index}>
            <Badge color={color} kind="solid">
              {segment.text}
            </Badge>
            <Badge color={color} kind="outline">
              {segment.label}
            </Badge>
          </span>
        );
      })}
    </Text>
  ) : (
    <Text className="text-secondary" kind="body/regular/md">
      {emptyMessage}
    </Text>
  )
);
