// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { NumberRangeFilterControl } from '@nemo/common/src/components/DataView/FilterPanel/NumberRangeFilter';
import {
  formatNumberRange,
  numberRangeFilter,
  type NumberRangeFilterValue,
} from '@nemo/common/src/components/DataView/FilterPanel/NumberRangeFilter/util';
import type { DataViewColumn } from '@nemo/common/src/components/DataView/FilterPanel/types';
import * as DataView from '@nemo/common/src/components/DataView/internal';
import { StudioAppliedFilters } from '@nemo/common/src/components/DataView/StudioAppliedFilters';
import { StudioDataView } from '@nemo/common/src/components/DataView/StudioDataView';
import { useStudioDataViewState } from '@nemo/common/src/hooks/useStudioDataViewState';
import { Stack, Text } from '@nvidia/foundations-react-core';
import type { Meta, StoryObj } from '@storybook/react';
import { type ComponentProps, type FC, useCallback, useState } from 'react';

// ---------------------------------------------------------------------------
// Sandbox — the NumberRangeFilterControl in isolation, with a live value readout
// ---------------------------------------------------------------------------

interface SandboxArgs {
  label: string;
  /** Omit `min`/`max` to render the unbounded (slider-less) variant. */
  min?: number;
  max?: number;
  step: number;
}

/**
 * Drives the control with local state and a minimal column stub that exposes
 * only what the control reads (`columnDef.meta.filter`, `getFilterValue`,
 * `setFilterValue`). The emitted `{ $gte, $lte }` value is rendered below so the
 * filter's output is visible while iterating on the component.
 */
function SandboxHarness({ label, min, max, step }: SandboxArgs) {
  const [value, setValue] = useState<NumberRangeFilterValue | undefined>(undefined);

  const column = {
    id: 'sandbox',
    columnDef: {
      header: label,
      meta: { filter: numberRangeFilter(label, { min, max, step }) },
    },
    getFilterValue: () => value,
    setFilterValue: (next: unknown) => setValue(next as NumberRangeFilterValue | undefined),
  } as unknown as DataViewColumn;

  return (
    <Stack gap="density-2xl" className="max-w-[340px]">
      <div className="w-[300px] rounded-lg border border-base bg-surface-raised p-density-xl">
        <NumberRangeFilterControl column={column} />
      </div>
      <Stack gap="density-sm">
        <Text kind="body/semibold/sm">Emitted filter value</Text>
        <pre className="rounded-md bg-surface p-density-md font-mono text-xs whitespace-pre-wrap">
          {JSON.stringify(value ?? null, null, 2)}
        </pre>
        <Text kind="body/regular/sm" className="text-secondary">
          Applied: {formatNumberRange(value?.$gte, value?.$lte) || '(no filter)'}
        </Text>
      </Stack>
    </Stack>
  );
}

const meta: Meta<SandboxArgs> = {
  title: 'Studio Common/DataView/NumberRangeFilter',
  parameters: { layout: 'padded' },
  argTypes: {
    label: { control: { type: 'text' } },
    min: { control: { type: 'number' } },
    max: { control: { type: 'number' } },
    step: { control: { type: 'number' } },
  },
  args: { label: 'Score', min: 0, max: 100, step: 1 },
};

export default meta;

/**
 * Isolated sandbox for working on the filter. Adjust the track bounds with the
 * Storybook controls; drag the slider, or type into the min/max inputs and blur
 * (or press Enter), and watch the emitted `{ $gte, $lte }` value update below.
 */
export const Sandbox: StoryObj<SandboxArgs> = {
  render: (args) => <SandboxHarness {...args} />,
};

/**
 * Bounded variant — both `min` and `max` are supplied, so the RangeSlider
 * renders and its bounds seed the min/max inputs (they start at 0 and 100).
 */
export const Bounded: StoryObj<SandboxArgs> = {
  name: 'Bounded (with slider)',
  args: { label: 'Score', min: 0, max: 100, step: 1 },
  render: (args) => <SandboxHarness {...args} />,
};

/**
 * Unbounded variant — no `min`/`max`, so the slider is hidden and the inputs
 * start empty (placeholders shown); typing emits an open-ended `{ $gte }` / `{ $lte }`.
 */
export const Unbounded: StoryObj<SandboxArgs> = {
  name: 'Unbounded (no slider)',
  args: { label: 'Tokens', min: undefined, max: undefined, step: 1 },
  render: (args) => <SandboxHarness {...args} />,
};

/** A wider range with a non-default track and a coarse step. */
export const CustomBounds: StoryObj<SandboxArgs> = {
  args: { label: 'Price (USD)', min: 0, max: 1000, step: 50 },
  render: (args) => <SandboxHarness {...args} />,
};

// ---------------------------------------------------------------------------
// In DataView — the filter wired into a real StudioDataView filter panel
// ---------------------------------------------------------------------------

interface JobMetric {
  id: string;
  name: string;
  durationMinutes: number;
  costUsd: number;
}

const JOB_METRICS: JobMetric[] = [
  { id: 'j-001', name: 'llama-3-sft-run-1', durationMinutes: 134, costUsd: 18 },
  { id: 'j-002', name: 'mistral-lora-v2', durationMinutes: 45, costUsd: 4 },
  { id: 'j-003', name: 'gemma-dpo-alignment', durationMinutes: 62, costUsd: 9 },
  { id: 'j-004', name: 'phi-3-full-ft', durationMinutes: 330, costUsd: 41 },
  { id: 'j-005', name: 'nemotron-sft-support', durationMinutes: 485, costUsd: 48 },
  { id: 'j-006', name: 'llama-guard-finetune', durationMinutes: 80, costUsd: 11 },
  { id: 'j-007', name: 'mixtral-lora-code', durationMinutes: 227, costUsd: 32 },
];

const matchesRange = (n: number, range: NumberRangeFilterValue | undefined): boolean => {
  if (range?.$gte != null && n < range.$gte) return false;
  if (range?.$lte != null && n > range.$lte) return false;
  return true;
};

const fixedHeightDecorator = (Story: FC) => (
  <div className="h-[600px]">
    <Story />
  </div>
);

/**
 * The numeric range filter inside a real DataView filter panel. Open the filter
 * panel (toolbar funnel icon) to filter by duration or cost; rows, the applied
 * filter chips, and the result count all update live.
 */
export const InDataView: StoryObj = {
  name: 'In DataView (Job Metrics)',
  decorators: [fixedHeightDecorator],
  render: function InDataViewStory() {
    const dataViewState = useStudioDataViewState({ defaultPageSize: 10 });

    // The DataView runs in manual mode, so we apply the column filters ourselves
    // to demonstrate the emitted `{ $gte, $lte }` values end to end.
    const activeFilters = dataViewState.columnFiltering.state;
    const filteredRows = JOB_METRICS.filter((row) =>
      activeFilters.every(({ id, value }) => {
        const range = value as NumberRangeFilterValue | undefined;
        if (id === 'durationMinutes') return matchesRange(row.durationMinutes, range);
        if (id === 'costUsd') return matchesRange(row.costUsd, range);
        return true;
      })
    );

    const makeColumns: ComponentProps<typeof StudioDataView<JobMetric>>['makeColumns'] =
      useCallback(
        ({ accessor }) => [
          accessor('name', { header: 'Job Name', enableSorting: false }),
          accessor('durationMinutes', {
            header: 'Duration (min)',
            enableSorting: true,
            size: 180,
            cell: ({ row }) => `${row.original.durationMinutes} min`,
            meta: {
              filter: numberRangeFilter('Duration (minutes)', { min: 0, max: 600, step: 15 }),
            },
          }),
          accessor('costUsd', {
            header: 'Cost (USD)',
            enableSorting: true,
            size: 160,
            cell: ({ row }) => `$${row.original.costUsd}`,
            meta: { filter: numberRangeFilter('Cost (USD)', { min: 0, max: 50, step: 1 }) },
          }),
        ],
        []
      );

    return (
      <StudioDataView<JobMetric>
        dataViewState={dataViewState}
        makeColumns={makeColumns}
        attributes={{
          DataViewRoot: {
            data: filteredRows,
            totalCount: filteredRows.length,
          },
        }}
      />
    );
  },
};

// ---------------------------------------------------------------------------
// Applied Filters — how StudioAppliedFilters renders a range value as a chip
// ---------------------------------------------------------------------------

interface RangeRow {
  durationMinutes: number;
  costUsd: number;
  accuracyPct: number;
}

/**
 * Renders StudioAppliedFilters inside a headless DataView.Root seeded with
 * pre-applied numeric range filters, so you can see how each committed
 * `{ $gte, $lte }` value is formatted into a chip via `formatNumberRange`:
 * both bounds ("30 – 480"), lower-only ("≥ 10"), and upper-only ("≤ 90").
 * Click a chip to clear that filter, or "Clear Filters" to clear all.
 */
export const AppliedFilters: StoryObj = {
  name: 'Applied Filter Chips',
  render: function AppliedFiltersStory() {
    const dataViewState = DataView.useDataViewState({
      columnFilters: [
        { id: 'durationMinutes', value: { $gte: 30, $lte: 480 } }, // both bounds
        { id: 'costUsd', value: { $gte: 10 } }, // lower bound only
        { id: 'accuracyPct', value: { $lte: 90 } }, // upper bound only
      ],
    });

    const makeColumns: ComponentProps<typeof DataView.Root<RangeRow>>['makeColumns'] = useCallback(
      ({ accessor }) => [
        accessor('durationMinutes', {
          header: 'Duration',
          meta: { filter: numberRangeFilter('Duration (minutes)', { min: 0, max: 600, step: 15 }) },
        }),
        accessor('costUsd', {
          header: 'Cost',
          meta: { filter: numberRangeFilter('Cost (USD)', { min: 0, max: 50, step: 1 }) },
        }),
        accessor('accuracyPct', {
          header: 'Accuracy',
          meta: { filter: numberRangeFilter('Accuracy (%)', { min: 0, max: 100, step: 1 }) },
        }),
      ],
      []
    );

    return (
      <DataView.Root
        dataMode="manual"
        state={dataViewState}
        data={[]}
        totalCount={0}
        makeColumns={makeColumns}
      >
        <Stack gap="density-lg" className="max-w-[640px]">
          <Text kind="body/regular/sm" className="text-secondary">
            StudioAppliedFilters formats each committed range into a chip — both bounds ("30 –
            480"), lower-only ("≥ 10"), and upper-only ("≤ 90"). Click a chip to clear that filter.
          </Text>
          <StudioAppliedFilters />
        </Stack>
      </DataView.Root>
    );
  },
};
