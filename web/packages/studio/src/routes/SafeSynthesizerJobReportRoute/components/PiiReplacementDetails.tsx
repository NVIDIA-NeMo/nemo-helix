// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { Badge, Stack, Text } from '@nvidia/foundations-react-core';
import type { PiiColumnGroup } from '@studio/routes/SafeSynthesizerJobReportRoute/chartData';
import { ReportTable } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ReportTable';
import { PII_COPY } from '@studio/routes/SafeSynthesizerJobReportRoute/reportCopy';
import type {
  PIIReplacement,
  PIIReplacementColumn,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';
import { type ComponentProps, type FC, useMemo } from 'react';

const PII_PAGE_SIZE = 50;
const MAPPINGS_PAGE_SIZE = 20;

interface PiiRow {
  id: string;
  columnName: string | null;
  replacement: PIIReplacementColumn | null | undefined;
  piiType: string;
  found: number;
  unique: number | null;
  synthetic: number | null;
  replayed: number | null;
  replayPercentage: number | null;
}

/** One row per PII type; a column's name and replacement show only on its first row, as the library's merged cells. */
const piiRows = (groups: readonly PiiColumnGroup[]): PiiRow[] =>
  groups.flatMap((group) =>
    group.entities.map((entity, index) => ({
      id: `${group.columnName}:${entity.piiType}`,
      columnName: index === 0 ? group.columnName : null,
      replacement: index === 0 ? group.replacement : undefined,
      ...entity,
    }))
  );

const formatCount = (value: number | null) => (value === null ? '—' : value.toLocaleString());

const ReplacedWith: FC<{ replacement: PIIReplacementColumn | null }> = ({ replacement }) => {
  if (replacement === null) {
    return (
      <Text kind="body/regular/sm" className="text-secondary">
        Not replaced
      </Text>
    );
  }
  return (
    <Stack gap="density-xs" className="whitespace-normal">
      <Text kind="body/regular/sm">{replacement.replaced_with}</Text>
      {replacement.pattern ? (
        <Stack gap="density-xs">
          <code>{replacement.pattern}</code>
          <ul className="list-disc pl-density-xl">
            {replacement.pattern_symbols.map((symbol) => (
              <li key={symbol.symbol}>
                <Text kind="body/regular/sm">
                  {symbol.symbol !== replacement.pattern ? <code>{symbol.symbol}</code> : null}{' '}
                  {symbol.meaning}
                </Text>
              </li>
            ))}
          </ul>
        </Stack>
      ) : null}
      {replacement.depends_on.length ? (
        <Text kind="body/regular/sm" className="text-secondary">
          Depends on {replacement.depends_on.map((dependency) => dependency.column_name).join(', ')}
        </Text>
      ) : null}
      {replacement.found_by.length ? (
        <Text kind="body/regular/sm" className="text-secondary">
          Found by {replacement.found_by.join(', ')}
        </Text>
      ) : null}
    </Stack>
  );
};

const makePiiColumns: ComponentProps<typeof StudioDataView<PiiRow>>['makeColumns'] = (col) => [
  col.accessor('columnName', {
    header: 'Column Name',
    cell: ({ getValue }) => <Text kind="body/semibold/sm">{getValue() ?? ''}</Text>,
  }),
  col.display({
    id: 'replacedWith',
    header: 'Replaced With',
    size: 280,
    cell: ({ row }) =>
      row.original.replacement === undefined ? null : (
        <ReplacedWith replacement={row.original.replacement} />
      ),
  }),
  col.accessor('piiType', {
    header: 'PII Type',
    cell: ({ getValue }) => (
      <Badge kind="outline" color="gray">
        {getValue()}
      </Badge>
    ),
  }),
  col.accessor('found', {
    header: 'Training Total',
    cell: ({ getValue }) => formatCount(getValue()),
  }),
  col.accessor('unique', {
    header: 'Training Unique',
    cell: ({ getValue }) => formatCount(getValue()),
  }),
  col.accessor('synthetic', {
    header: 'Synthetic Values',
    cell: ({ getValue }) => formatCount(getValue()),
  }),
  col.display({
    id: 'replay',
    header: 'PII Replay',
    cell: ({ row }) => {
      const { replayed, unique, replayPercentage } = row.original;
      if (replayed === null) {
        return '—';
      }
      return (
        <Stack gap="density-xs">
          <Text kind="body/regular/sm">{replayPercentage}%</Text>
          <Text kind="body/regular/sm" className="text-secondary">
            {formatCount(replayed)} of {formatCount(unique)}
          </Text>
        </Stack>
      );
    },
  }),
];

interface MappingRow {
  id: string;
  column: string;
  value: string;
  samplerValues: string;
}

const mappingRows = (mappings: PIIReplacement['value_mappings']): MappingRow[] =>
  Object.entries(mappings).flatMap(([column, values]) =>
    Object.entries(values).map(([value, samplerValues]) => ({
      id: `${column}:${value}`,
      column,
      value,
      samplerValues: samplerValues.join(', '),
    }))
  );

const makeMappingColumns: ComponentProps<typeof StudioDataView<MappingRow>>['makeColumns'] = (
  col
) => [
  col.accessor('column', { header: 'Column Name' }),
  col.accessor('value', { header: 'Data Value' }),
  col.accessor('samplerValues', { header: 'Sampler Values' }),
];

const getRowId = (row: { id: string }) => row.id;

const Paragraph: FC<{ children: string }> = ({ children }) => (
  <Text kind="body/regular/sm" className="text-secondary">
    {children}
  </Text>
);

interface PiiReplacementDetailsProps {
  pii: PIIReplacement;
  groups: readonly PiiColumnGroup[];
  replayNote: string | null;
}

export const PiiReplacementDetails: FC<PiiReplacementDetailsProps> = ({
  pii,
  groups,
  replayNote,
}) => {
  const rows = useMemo(() => piiRows(groups), [groups]);
  const mappings = useMemo(() => mappingRows(pii.value_mappings), [pii.value_mappings]);
  const seed = pii.seed === null ? 'random seed' : `seed ${pii.seed}`;
  return (
    <Stack gap="density-lg" data-testid="pii-replacement-details">
      {pii.replaced ? (
        <Paragraph>{`Sampler: ${pii.sampler}, locale ${pii.locale}, ${seed}. The test data is not replaced.`}</Paragraph>
      ) : (
        <Paragraph>{PII_COPY.contextOff}</Paragraph>
      )}
      {replayNote ? <Paragraph>{replayNote}</Paragraph> : null}
      {rows.length ? (
        <ReportTable
          rows={rows}
          makeColumns={makePiiColumns}
          getRowId={getRowId}
          pageSize={PII_PAGE_SIZE}
        />
      ) : null}
      {!rows.length && !replayNote ? <Paragraph>{PII_COPY.noPii}</Paragraph> : null}
      {mappings.length ? (
        <Stack gap="density-sm">
          <Text kind="label/bold/md">Value Mappings</Text>
          <ReportTable
            rows={mappings}
            makeColumns={makeMappingColumns}
            getRowId={getRowId}
            pageSize={MAPPINGS_PAGE_SIZE}
          />
        </Stack>
      ) : null}
      {pii.kept_columns.length ? (
        <Text kind="body/regular/sm">
          <strong>Kept as is ({pii.kept_columns.length}):</strong> {pii.kept_columns.join(', ')}
        </Text>
      ) : null}
      <Paragraph>{PII_COPY.footnote}</Paragraph>
    </Stack>
  );
};
