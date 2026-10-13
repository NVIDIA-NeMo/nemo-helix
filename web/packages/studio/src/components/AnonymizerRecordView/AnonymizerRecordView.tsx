// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { HighlightedText } from '@studio/components/AnonymizerRecordView/HighlightedText';
import { RecordSection } from '@studio/components/AnonymizerRecordView/RecordSection';
import { ReplacementMapTable } from '@studio/components/AnonymizerRecordView/ReplacementMapTable';
import type { AnonymizerRecord } from '@studio/components/AnonymizerRecordView/types';
import { memo, type FC } from 'react';

interface AnonymizerRecordViewProps {
  readonly record: AnonymizerRecord;
  readonly outputHeading: string;
}

export const AnonymizerRecordView: FC<AnonymizerRecordViewProps> = memo(
  ({ record, outputHeading }) => (
    <Stack gap="density-2xl">
      <Flex align="start" gap="density-2xl">
        <RecordSection className="flex-1 min-w-0" heading="Original">
          <HighlightedText
            emptyMessage="This record has no text."
            segments={record.originalSegments}
          />
        </RecordSection>
        <RecordSection className="flex-1 min-w-0" heading={outputHeading}>
          <HighlightedText
            emptyMessage="No output was produced for this record."
            segments={record.replacedSegments}
          />
        </RecordSection>
      </Flex>

      <RecordSection heading="Replacement Map">
        {record.replacements.length ? (
          <ReplacementMapTable replacements={record.replacements} />
        ) : (
          <Text className="text-secondary" kind="body/regular/md">
            No entities were replaced in this record.
          </Text>
        )}
      </RecordSection>
    </Stack>
  )
);
