// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { dateTimeFilter } from '@nemo/common/src/components/DataView/dateTimeFilter';
import * as DataView from '@nemo/common/src/components/DataView/internal';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { Eye, Trash } from '@nemo/icons/icons';
import {
  Anchor,
  Badge,
  Button,
  type DropdownEntry,
  Flex,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { ComponentProps, FC, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Mock datasets
// ---------------------------------------------------------------------------

interface Job {
  id: string;
  name: string;
  status: 'completed' | 'running' | 'failed' | 'pending';
  model: string;
  created_at: string;
  duration: string;
}

interface Dataset {
  id: string;
  name: string;
  description: string;
  format: string;
  rows: number;
  size: string;
  created_at: string;
  updated_at: string;
}

interface Secret {
  id: string;
  key: string;
  created_at: string;
}

const JOBS: Job[] = [
  {
    id: 'j-001',
    name: 'llama-3-sft-run-1',
    status: 'completed',
    model: 'meta/llama-3.1-8b-instruct',
    created_at: '2026-02-20T10:30:00Z',
    duration: '2h 14m',
  },
  {
    id: 'j-002',
    name: 'mistral-lora-v2',
    status: 'running',
    model: 'mistralai/mistral-7b-instruct-v0.3',
    created_at: '2026-02-22T08:15:00Z',
    duration: '45m',
  },
  {
    id: 'j-003',
    name: 'gemma-dpo-alignment',
    status: 'failed',
    model: 'google/gemma-2-9b-it',
    created_at: '2026-02-19T14:00:00Z',
    duration: '1h 02m',
  },
  {
    id: 'j-004',
    name: 'phi-3-full-ft',
    status: 'completed',
    model: 'microsoft/phi-3-mini-128k-instruct',
    created_at: '2026-02-18T09:45:00Z',
    duration: '5h 30m',
  },
  {
    id: 'j-005',
    name: 'llama-3-eval-safety',
    status: 'pending',
    model: 'meta/llama-3.1-70b-instruct',
    created_at: '2026-02-24T16:20:00Z',
    duration: '-',
  },
  {
    id: 'j-006',
    name: 'nemotron-sft-customer-support',
    status: 'completed',
    model: 'nvidia/nemotron-4-340b-instruct',
    created_at: '2026-02-17T11:00:00Z',
    duration: '8h 05m',
  },
  {
    id: 'j-007',
    name: 'llama-guard-finetune',
    status: 'running',
    model: 'meta/llama-guard-3-8b',
    created_at: '2026-02-23T13:30:00Z',
    duration: '1h 20m',
  },
  {
    id: 'j-008',
    name: 'mixtral-lora-code',
    status: 'completed',
    model: 'mistralai/mixtral-8x7b-instruct-v0.1',
    created_at: '2026-02-15T07:00:00Z',
    duration: '3h 47m',
  },
];

const DATASETS: Dataset[] = [
  {
    id: 'd-001',
    name: 'customer-support-v3',
    description:
      'Customer support conversations collected from Q4 2025 ticketing system, cleaned and deduplicated with PII removal applied.',
    format: 'JSONL',
    rows: 45200,
    size: '128 MB',
    created_at: '2026-01-10T09:00:00Z',
    updated_at: '2026-02-20T14:30:00Z',
  },
  {
    id: 'd-002',
    name: 'code-review-pairs',
    description:
      'Paired code review comments and suggested fixes extracted from public repositories, covering Python, TypeScript, and Go.',
    format: 'JSONL',
    rows: 18700,
    size: '52 MB',
    created_at: '2026-01-15T11:00:00Z',
    updated_at: '2026-02-18T09:15:00Z',
  },
  {
    id: 'd-003',
    name: 'safety-eval-prompts',
    description:
      'Red-teaming evaluation prompts for testing model safety boundaries across multiple risk categories.',
    format: 'CSV',
    rows: 3200,
    size: '4.1 MB',
    created_at: '2025-12-01T08:30:00Z',
    updated_at: '2026-01-05T16:00:00Z',
  },
  {
    id: 'd-004',
    name: 'medical-qa-synthetic',
    description:
      'Synthetically generated medical question-answer pairs reviewed by domain experts. Not intended for clinical use. Covers general medicine, pharmacology, and diagnostic reasoning.',
    format: 'JSONL',
    rows: 92100,
    size: '310 MB',
    created_at: '2026-02-01T10:00:00Z',
    updated_at: '2026-02-22T11:45:00Z',
  },
  {
    id: 'd-005',
    name: 'financial-reports-summarization',
    description:
      'Quarterly earnings reports paired with human-written summaries from publicly traded companies.',
    format: 'Parquet',
    rows: 6800,
    size: '89 MB',
    created_at: '2026-01-20T14:00:00Z',
    updated_at: '2026-02-10T08:30:00Z',
  },
  {
    id: 'd-006',
    name: 'multilingual-intent-classification',
    description:
      'Intent classification dataset spanning English, Spanish, French, German, and Japanese with 42 intent categories.',
    format: 'JSONL',
    rows: 156000,
    size: '420 MB',
    created_at: '2025-11-15T09:00:00Z',
    updated_at: '2026-02-24T17:00:00Z',
  },
];

const SECRETS: Secret[] = [
  { id: 's-001', key: 'HF_TOKEN', created_at: '2026-01-05T10:00:00Z' },
  { id: 's-002', key: 'WANDB_API_KEY', created_at: '2026-01-10T14:30:00Z' },
  { id: 's-003', key: 'NGC_API_KEY', created_at: '2025-12-20T09:00:00Z' },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STATUS_COLORS: Record<Job['status'], 'green' | 'blue' | 'red' | 'gray'> = {
  completed: 'green',
  running: 'blue',
  failed: 'red',
  pending: 'gray',
};

function StatusBadge({ status }: { status: Job['status'] }) {
  return <Badge color={STATUS_COLORS[status]}>{status}</Badge>;
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

function formatNumber(n: number) {
  return n.toLocaleString();
}

// ---------------------------------------------------------------------------
// Meta
// ---------------------------------------------------------------------------

const fixedHeightDecorator = (Story: FC) => (
  <div className="h-[600px]">
    <Story />
  </div>
);

const meta: Meta<typeof StudioDataView> = {
  component: StudioDataView,
  title: 'Studio Common/StudioDataView',
  parameters: { layout: 'padded' },
};

export default meta;

// ---------------------------------------------------------------------------
// Stories
// ---------------------------------------------------------------------------

export const Default: StoryObj = {
  decorators: [fixedHeightDecorator],
  name: 'Default (Jobs)',
  render: function DefaultStory() {
    const dataViewState = useStudioDataViewState({
      defaultSort: [{ id: 'created_at', desc: true }],
      defaultPageSize: 10,
    });

    const makeColumns: ComponentProps<typeof StudioDataView<Job>>['makeColumns'] = useCallback(
      ({ accessor }) => [
        accessor('name', { header: 'Job Name', enableSorting: false }),
        accessor('model', { header: 'Model', enableSorting: false }),
        accessor('status', {
          header: 'Status',
          enableSorting: false,
          size: 130,
          cell: ({ row }) => <StatusBadge status={row.original.status} />,
        }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: true,
          size: 160,
          cell: ({ row }) => formatDate(row.original.created_at),
        }),
        accessor('duration', { header: 'Duration', enableSorting: false, size: 120 }),
      ],
      []
    );

    return (
      <StudioDataView<Job>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        attributes={{
          DataViewRoot: {
            data: JOBS,
            totalCount: JOBS.length,
          },
        }}
      />
    );
  },
};

export const WithRowClick: StoryObj = {
  decorators: [fixedHeightDecorator],
  name: 'Row Click (Jobs)',
  render: function RowClickStory() {
    const dataViewState = useStudioDataViewState({
      defaultSort: [{ id: 'created_at', desc: true }],
      defaultPageSize: 10,
    });

    const makeColumns: ComponentProps<typeof StudioDataView<Job>>['makeColumns'] = useCallback(
      ({ accessor }) => [
        accessor('name', { header: 'Job Name', enableSorting: false }),
        accessor('model', { header: 'Model', enableSorting: false }),
        accessor('status', {
          header: 'Status',
          enableSorting: false,
          size: 130,
          cell: ({ row }) => <StatusBadge status={row.original.status} />,
        }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: true,
          size: 160,
          cell: ({ row }) => formatDate(row.original.created_at),
        }),
        accessor('duration', { header: 'Duration', enableSorting: false, size: 120 }),
      ],
      []
    );

    return (
      <StudioDataView<Job>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        onRowClick={(row) => alert(`Clicked row: ${row.name} (${row.id})`)}
        attributes={{
          DataViewRoot: {
            data: JOBS,
            totalCount: JOBS.length,
          },
        }}
      />
    );
  },
};

export const WithSelectionAndActions: StoryObj = {
  decorators: [fixedHeightDecorator],
  name: 'Selection + Actions (Jobs)',
  render: function SelectionActionsStory() {
    const dataViewState = useStudioDataViewState({
      defaultSort: [{ id: 'created_at', desc: true }],
      defaultPageSize: 10,
      columnPinning: {
        left: ['row-selection'],
        right: ['row-actions'],
      },
    });

    const makeColumns: ComponentProps<typeof StudioDataView<Job>>['makeColumns'] = useCallback(
      ({ accessor }, { rowSelectionColumn, rowActionsColumn }) => [
        rowSelectionColumn({ size: 48 }),
        accessor('name', {
          header: 'Job Name',
          enableSorting: false,
          cell: ({ row }) => (
            <Anchor asChild>
              <a href={`#/jobs/${row.original.id}`}>{row.original.name}</a>
            </Anchor>
          ),
        }),
        accessor('model', { header: 'Model', enableSorting: false }),
        accessor('status', {
          header: 'Status',
          enableSorting: false,
          size: 130,
          cell: ({ row }) => <StatusBadge status={row.original.status} />,
        }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: true,
          size: 160,
          cell: ({ row }) => formatDate(row.original.created_at),
        }),
        accessor('duration', { header: 'Duration', enableSorting: false, size: 120 }),
        rowActionsColumn({
          size: 58,
          enableResizing: false,
          rowActions: (job): DropdownEntry[] => [
            {
              children: 'View details',
              onSelect: () => alert(`View: ${job.name}`),
            },
            {
              children: 'Evaluate',
              onSelect: () => alert(`Evaluate: ${job.name}`),
            },
            {
              children: 'Chat',
              disabled: job.status !== 'completed',
              onSelect: () => alert(`Chat: ${job.name}`),
            },
            {
              children: 'Delete',
              danger: true,
              onSelect: () => alert(`Delete: ${job.name}`),
            },
          ],
        }),
      ],
      []
    );

    return (
      <StudioDataView<Job>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        onRowClick={(row) => alert(`Row clicked: ${row.name}`)}
        renderBulkActions={({ selectedRows }) => (
          <Button
            color="danger"
            kind="tertiary"
            onClick={() => alert(`Delete ${selectedRows.length} jobs`)}
          >
            <Trash /> Delete
          </Button>
        )}
        attributes={{
          DataViewRoot: {
            data: JOBS,
            totalCount: JOBS.length,
          },
        }}
      />
    );
  },
};

export const LongTextWrapping: StoryObj = {
  decorators: [fixedHeightDecorator],
  name: 'Text Wrapping (Datasets)',
  render: function LongTextStory() {
    const dataViewState = useStudioDataViewState({
      defaultSort: [{ id: 'created_at', desc: true }],
      defaultPageSize: 10,
    });

    const makeColumns: ComponentProps<typeof StudioDataView<Dataset>>['makeColumns'] = useCallback(
      ({ accessor }, { rowActionsColumn }) => [
        accessor('name', {
          header: 'Dataset Name',
          enableSorting: false,
          size: 200,
        }),
        accessor('description', {
          header: 'Description',
          enableSorting: false,
        }),
        accessor('format', {
          header: 'Format',
          enableSorting: false,
          size: 100,
          cell: ({ row }) => <Badge color="gray">{row.original.format}</Badge>,
        }),
        accessor('rows', {
          header: 'Rows',
          enableSorting: false,
          size: 100,
          cell: ({ row }) => formatNumber(row.original.rows),
        }),
        accessor('size', { header: 'Size', enableSorting: false, size: 100 }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: true,
          size: 140,
          cell: ({ row }) => formatDate(row.original.created_at),
        }),
        rowActionsColumn({
          size: 58,
          enableResizing: false,
          rowActions: (dataset): DropdownEntry[] => [
            {
              children: 'View',
              onSelect: () => alert(`View: ${dataset.name}`),
            },
            {
              children: 'Edit',
              onSelect: () => alert(`Edit: ${dataset.name}`),
            },
            {
              children: 'Delete',
              danger: true,
              onSelect: () => alert(`Delete: ${dataset.name}`),
            },
          ],
        }),
      ],
      []
    );

    return (
      <StudioDataView<Dataset>
        dataViewState={dataViewState}
        maxTwoLines
        makeColumns={makeColumns}
        onRowClick={(row) => alert(`Dataset: ${row.name}`)}
        attributes={{
          DataViewRoot: {
            data: DATASETS,
            totalCount: DATASETS.length,
          },
        }}
      />
    );
  },
};

export const MinimalTable: StoryObj = {
  decorators: [fixedHeightDecorator],
  name: 'Minimal (Secrets)',
  render: function MinimalStory() {
    const dataViewState = useStudioDataViewState({ defaultPageSize: 10 });

    const makeColumns: ComponentProps<typeof StudioDataView<Secret>>['makeColumns'] = useCallback(
      ({ accessor }) => [
        accessor('key', { header: 'Secret Key', enableSorting: false }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: false,
          size: 200,
          cell: ({ row }) => formatDate(row.original.created_at),
        }),
      ],
      []
    );

    return (
      <StudioDataView<Secret>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        attributes={{
          DataViewRoot: {
            data: SECRETS,
            totalCount: SECRETS.length,
          },
        }}
      />
    );
  },
};

export const EmptyState: StoryObj = {
  decorators: [fixedHeightDecorator],
  name: 'Empty State',
  render: function EmptyStory() {
    const dataViewState = useStudioDataViewState({ defaultPageSize: 10 });

    const makeColumns: ComponentProps<typeof StudioDataView<Job>>['makeColumns'] = useCallback(
      ({ accessor }) => [
        accessor('name', { header: 'Job Name', enableSorting: false }),
        accessor('model', { header: 'Model', enableSorting: false }),
        accessor('status', { header: 'Status', enableSorting: false, size: 130 }),
        accessor('created_at', { header: 'Created', enableSorting: false, size: 160 }),
      ],
      []
    );

    return (
      <StudioDataView<Job>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        attributes={{
          DataViewRoot: {
            data: [],
            totalCount: 0,
          },
        }}
      />
    );
  },
};

export const CustomEmptyState: StoryObj = {
  decorators: [fixedHeightDecorator],
  name: 'Custom Empty State',
  render: function CustomEmptyStory() {
    const dataViewState = useStudioDataViewState({ defaultPageSize: 10 });

    const makeColumns: ComponentProps<typeof StudioDataView<Job>>['makeColumns'] = useCallback(
      ({ accessor }) => [
        accessor('name', { header: 'Job Name', enableSorting: false }),
        accessor('model', { header: 'Model', enableSorting: false }),
        accessor('status', { header: 'Status', enableSorting: false, size: 130 }),
      ],
      []
    );

    return (
      <StudioDataView<Job>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        attributes={{
          DataViewRoot: {
            data: [],
            totalCount: 0,
          },
          DataViewTableContent: {
            renderEmptyState: () => (
              <div className="flex flex-col items-center justify-center gap-4 py-16">
                <Text kind="title/sm">No Jobs Yet</Text>
                <Text kind="body/regular/md" className="text-secondary">
                  Create your first customization job to get started.
                </Text>
                <Button kind="primary" onClick={() => alert('Create job')}>
                  Create Job
                </Button>
              </div>
            ),
          },
        }}
      />
    );
  },
};

export const LoadingState: StoryObj = {
  decorators: [fixedHeightDecorator],
  name: 'Loading State',
  render: function LoadingStory() {
    const dataViewState = useStudioDataViewState({ defaultPageSize: 10 });

    const makeColumns: ComponentProps<typeof StudioDataView<Job>>['makeColumns'] = useCallback(
      ({ accessor }, { rowSelectionColumn, rowActionsColumn }) => [
        rowSelectionColumn({ size: 48 }),
        accessor('name', { header: 'Job Name', enableSorting: false }),
        accessor('model', { header: 'Model', enableSorting: false }),
        accessor('status', { header: 'Status', enableSorting: false, size: 130 }),
        accessor('created_at', { header: 'Created', enableSorting: true, size: 160 }),
        accessor('duration', { header: 'Duration', enableSorting: false, size: 120 }),
        rowActionsColumn({ size: 58, enableResizing: false, rowActions: () => [] }),
      ],
      []
    );

    return (
      <StudioDataView<Job>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        attributes={{
          DataViewRoot: {
            data: undefined,
            totalCount: 0,
            requestStatus: 'loading',
          },
        }}
      />
    );
  },
};

// Generate many rows to force the table body to scroll within a constrained container.
const MANY_JOBS: Job[] = Array.from({ length: 100 }, (_, i) => ({
  id: `j-${String(i + 1).padStart(3, '0')}`,
  name: `job-run-${i + 1}`,
  status: (['completed', 'running', 'failed', 'pending'] as const)[i % 4],
  model: [
    'meta/llama-3.1-8b-instruct',
    'mistralai/mistral-7b-instruct-v0.3',
    'google/gemma-2-9b-it',
    'nvidia/nemotron-4-340b-instruct',
  ][i % 4],
  created_at: new Date(2026, 1, 1, 0, 0, 0, 0).toISOString(),
  duration: `${Math.floor(Math.random() * 8) + 1}h ${Math.floor(Math.random() * 59)}m`,
}));

export const ManyRows: StoryObj = {
  decorators: [fixedHeightDecorator],
  name: 'Many Rows (Scrolling)',
  render: function ManyRowsStory() {
    const dataViewState = useStudioDataViewState({
      defaultSort: [{ id: 'created_at', desc: true }],
      defaultPageSize: 100,
    });

    const makeColumns: ComponentProps<typeof StudioDataView<Job>>['makeColumns'] = useCallback(
      ({ accessor }) => [
        accessor('name', { header: 'Job Name', enableSorting: false }),
        accessor('model', { header: 'Model', enableSorting: false }),
        accessor('status', {
          header: 'Status',
          enableSorting: false,
          size: 130,
          cell: ({ row }) => <StatusBadge status={row.original.status} />,
        }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: true,
          size: 160,
          cell: ({ row }) => formatDate(row.original.created_at),
        }),
        accessor('duration', { header: 'Duration', enableSorting: false, size: 120 }),
      ],
      []
    );

    return (
      <StudioDataView<Job>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        attributes={{
          DataViewRoot: {
            data: MANY_JOBS,
            totalCount: MANY_JOBS.length,
          },
        }}
      />
    );
  },
};

/**
 * Demonstrates the recommended consumer pattern for full-height tables.
 * The page layout uses a flex column with a header at natural height and the
 * data view filling remaining space via `flex-1 min-h-0`.
 */
export const FlexHeightLayout: StoryObj = {
  name: 'Flex Height Layout (Page Pattern)',
  parameters: { layout: 'fullscreen' },
  render: function FlexHeightStory() {
    const dataViewState = useStudioDataViewState({
      defaultSort: [{ id: 'created_at', desc: true }],
      defaultPageSize: 100,
    });

    const makeColumns: ComponentProps<typeof StudioDataView<Job>>['makeColumns'] = useCallback(
      ({ accessor }) => [
        accessor('name', { header: 'Job Name', enableSorting: false }),
        accessor('model', { header: 'Model', enableSorting: false }),
        accessor('status', {
          header: 'Status',
          enableSorting: false,
          size: 130,
          cell: ({ row }) => <StatusBadge status={row.original.status} />,
        }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: true,
          size: 160,
          cell: ({ row }) => formatDate(row.original.created_at),
        }),
        accessor('duration', { header: 'Duration', enableSorting: false, size: 120 }),
      ],
      []
    );

    return (
      <Stack className="h-dvh" gap="density-2xl" padding="density-2xl">
        <Flex justify="between" align="center" className="shrink-0">
          <Text kind="title/lg">Page Header</Text>
          <Button kind="primary">Action</Button>
        </Flex>
        <div className="flex-1 min-h-0">
          <StudioDataView<Job>
            dataViewState={dataViewState}
            makeColumns={makeColumns}
            attributes={{
              DataViewRoot: {
                data: MANY_JOBS,
                totalCount: MANY_JOBS.length,
              },
            }}
          />
        </div>
      </Stack>
    );
  },
};

export const InteractiveElementExclusion: StoryObj = {
  decorators: [fixedHeightDecorator],
  name: 'Interactive Element Exclusion',
  render: function InteractiveExclusionStory() {
    const dataViewState = useStudioDataViewState({
      defaultSort: [{ id: 'created_at', desc: true }],
      defaultPageSize: 10,
    });

    const makeColumns: ComponentProps<typeof StudioDataView<Job>>['makeColumns'] = useCallback(
      ({ accessor }, { rowSelectionColumn, rowActionsColumn }) => [
        rowSelectionColumn({ size: 48 }),
        accessor('name', {
          header: 'Job Name',
          enableSorting: false,
          cell: ({ row }) => (
            <Anchor asChild>
              <a href={`#/jobs/${row.original.id}`}>{row.original.name}</a>
            </Anchor>
          ),
        }),
        accessor('model', { header: 'Model', enableSorting: false }),
        accessor('status', {
          header: 'Status',
          enableSorting: false,
          size: 130,
          cell: ({ row }) => <StatusBadge status={row.original.status} />,
        }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: true,
          size: 160,
          cell: ({ row }) => formatDate(row.original.created_at),
        }),
        {
          id: 'inline_actions',
          header: 'Quick Actions',
          size: 180,
          cell: ({ row }: { row: DataView.TanstackTable.Row<Job> }) => (
            <div className="flex gap-2">
              <Tooltip slotContent="This button click won't trigger onRowClick">
                <Button
                  kind="tertiary"
                  size="small"
                  onClick={() => alert(`Inline action on: ${row.original.name}`)}
                >
                  <Eye /> View
                </Button>
              </Tooltip>
            </div>
          ),
        },
        rowActionsColumn({
          size: 58,
          enableResizing: false,
          rowActions: (job): DropdownEntry[] => [
            { children: 'View', onSelect: () => alert(`Menu: ${job.name}`) },
            {
              children: 'Delete',
              danger: true,
              onSelect: () => alert(`Delete: ${job.name}`),
            },
          ],
        }),
      ],
      []
    );

    return (
      <StudioDataView<Job>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        onRowClick={(row) =>
          alert(
            `Row clicked: ${row.name}\n\n(Checkbox, link, button, and menu clicks are excluded)`
          )
        }
        attributes={{
          DataViewRoot: {
            data: JOBS,
            totalCount: JOBS.length,
          },
        }}
      />
    );
  },
};

export const AllFilterTypes: StoryObj = {
  decorators: [fixedHeightDecorator],
  name: 'All Filter Types',
  render: function AllFilterTypesStory() {
    const dataViewState = useStudioDataViewState({
      defaultSort: [{ id: 'created_at', desc: true }],
      defaultPageSize: 10,
    });

    const makeColumns: ComponentProps<typeof StudioDataView<Job>>['makeColumns'] = useCallback(
      ({ accessor }) => [
        accessor('name', {
          header: 'Job Name',
          enableSorting: false,
          meta: {
            filter: {
              type: 'text',
              label: 'Job Name',
              placeholder: 'Filter by name',
            },
          },
        }),
        accessor('status', {
          header: 'Status',
          enableSorting: false,
          size: 130,
          cell: ({ row }) => <StatusBadge status={row.original.status} />,
          meta: {
            filter: {
              type: 'multi-select',
              label: 'Status',
              options: [
                { value: 'completed', label: 'Completed' },
                { value: 'running', label: 'Running' },
                { value: 'failed', label: 'Failed' },
                { value: 'pending', label: 'Pending' },
              ],
            },
          },
        }),
        accessor('model', {
          header: 'Model',
          enableSorting: false,
          meta: {
            filter: {
              type: 'single-select',
              label: 'Model',
              options: [
                { value: 'meta/llama-3.1-8b-instruct', label: 'Llama 3.1 8B' },
                { value: 'mistralai/mistral-7b-instruct-v0.3', label: 'Mistral 7B' },
                { value: 'google/gemma-2-9b-it', label: 'Gemma 2 9B' },
                { value: 'microsoft/phi-3-mini-128k-instruct', label: 'Phi-3 Mini' },
                { value: 'meta/llama-3.1-70b-instruct', label: 'Llama 3.1 70B' },
                { value: 'nvidia/nemotron-4-340b-instruct', label: 'Nemotron 340B' },
                { value: 'meta/llama-guard-3-8b', label: 'Llama Guard 3' },
                { value: 'mistralai/mixtral-8x7b-instruct-v0.1', label: 'Mixtral 8x7B' },
              ],
            },
          },
        }),
        accessor('created_at', {
          header: 'Created',
          enableSorting: true,
          size: 160,
          cell: ({ row }) => formatDate(row.original.created_at),
          meta: {
            filter: dateTimeFilter('Created'),
          },
        }),
        accessor('duration', {
          header: 'Duration',
          enableSorting: false,
          size: 120,
          meta: {
            filter: {
              type: 'boolean',
              label: 'Has Duration',
            },
          },
        }),
        accessor('id', {
          header: 'ID',
          enableSorting: false,
          size: 120,
          meta: {
            filter: {
              type: 'custom' as const,
              label: 'Job ID (Custom)',
              renderFilter: ({ value, setValue }) => (
                <TextInput
                  placeholder="Custom filter..."
                  value={(value as string) ?? ''}
                  onChange={(e) => setValue(e.target.value || undefined)}
                />
              ),
            },
          },
        }),
      ],
      []
    );

    return (
      <StudioDataView<Job>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        searchField="name"
        attributes={{
          DataViewRoot: {
            data: JOBS,
            totalCount: JOBS.length,
          },
        }}
      />
    );
  },
};
