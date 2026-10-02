// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Block, Button, Flex, Modal, Text } from '@nvidia/foundations-react-core';
import { AnonymizerRecordView } from '@studio/components/AnonymizerRecordView/AnonymizerRecordView';
import {
  buildAnonymizerRecord,
  outputColumn,
  REWRITTEN_SUFFIX,
} from '@studio/components/AnonymizerRecordView/parse';
import type { DataFileRow } from '@studio/components/FileRowEditor/types';
import { RecordPager } from '@studio/routes/AnonymizerBuilderRoute/components/RecordPager';
import {
  OUTPUT_HEADING_REPLACED,
  OUTPUT_HEADING_REWRITTEN,
} from '@studio/routes/AnonymizerBuilderRoute/utils';
import { resolveTextColumn } from '@studio/routes/AnonymizerJobDetailRoute/util';
import { useMemo, type FC } from 'react';

interface RecordDetailModalProps {
  readonly rows: readonly DataFileRow[];
  readonly selectedIndex: number | null;
  readonly textColumn: string | undefined;
  readonly onIndexChange: (index: number) => void;
  readonly onClose: () => void;
}

export const RecordDetailModal: FC<RecordDetailModalProps> = ({
  rows,
  selectedIndex,
  textColumn,
  onIndexChange,
  onClose,
}) => {
  const row = selectedIndex !== null ? (rows[selectedIndex] ?? null) : null;
  const resolvedColumn = row ? resolveTextColumn(row, textColumn) : '';
  const record = useMemo(
    () => (row ? buildAnonymizerRecord(row, resolvedColumn) : undefined),
    [row, resolvedColumn]
  );
  const outputHeading = outputColumn(row ?? {}, resolvedColumn)?.endsWith(REWRITTEN_SUFFIX)
    ? OUTPUT_HEADING_REWRITTEN
    : OUTPUT_HEADING_REPLACED;

  return (
    <Modal
      open={row !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      slotHeading={
        <Flex align="center" className="w-full" gap="density-md" justify="between">
          <Text kind="label/bold/xl">Record Preview</Text>
          {selectedIndex !== null ? (
            <RecordPager index={selectedIndex} onChange={onIndexChange} total={rows.length} />
          ) : null}
        </Flex>
      }
      className="w-[90vw] max-w-[1200px]"
      slotFooter={
        <Flex justify="end" align="center" className="w-full">
          <Button kind="tertiary" onClick={onClose}>
            Close
          </Button>
        </Flex>
      }
    >
      <Block className="max-h-[70vh] overflow-auto">
        {record ? <AnonymizerRecordView outputHeading={outputHeading} record={record} /> : null}
      </Block>
    </Modal>
  );
};
