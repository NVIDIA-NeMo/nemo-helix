// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { snakeCaseToTitleCase } from '@nemo/common/src/utils/formatters';
import { Add, Close } from '@nemo/icons/icons';
import {
  Button,
  FormField,
  SegmentedControl,
  SelectContent,
  SelectItem,
  SelectListbox,
  SelectRoot,
  SelectTrigger,
  Stack,
  Text,
} from '@nvidia/foundations-react-core';
import {
  DEFAULT_SORT,
  EVALUATOR_PREFIX,
  STATIC_FIELDS,
  evaluatorField,
  evaluatorNameOf,
  formatSortList,
  isEvaluatorField,
  parseSortList,
} from '@studio/components/DefaultSortControl/util';
import type { FC } from 'react';

/**
 * A group's **default sort**: a comma-separated, ordered list of `sort`-param fields (e.g.
 * `-cost_usd.mean,latency_ms.mean`) the client applies as the experiments list's `sort` param on load.
 * The first field is the primary sort; the rest break ties. Always set (defaults to `-created_at`);
 * each field is any the experiments list can sort by, matching the sort/filter API grammar.
 */

interface SortEntry {
  field: string;
  desc: boolean;
}

interface SortFieldRowProps {
  field: string;
  desc: boolean;
  /** Evaluator names to offer as first-class options (union of discovered + currently-selected). */
  evaluators: string[];
  disabled?: boolean;
  /** Removing the last row is disallowed — a default sort always has at least one field. */
  canRemove: boolean;
  onFieldChange: (field: string) => void;
  onDescChange: (desc: boolean) => void;
  onRemove: () => void;
}

const SortFieldRow: FC<SortFieldRowProps> = ({
  field,
  desc,
  evaluators,
  disabled,
  canRemove,
  onFieldChange,
  onDescChange,
  onRemove,
}) => {
  // Map a stored field to the Select's option value (static id or `evaluator:<name>`).
  const selectValueFor = (f: string): string =>
    STATIC_FIELDS.some((s) => s.value === f) ? f : `${EVALUATOR_PREFIX}${evaluatorNameOf(f)}`;

  const labelForOption = (optionValue: string): string => {
    const staticField = STATIC_FIELDS.find((f) => f.value === optionValue);
    if (staticField) return staticField.label;
    return `Avg ${snakeCaseToTitleCase(optionValue.slice(EVALUATOR_PREFIX.length))}`;
  };

  const onSelectField = (selected: string) => {
    if (selected.startsWith(EVALUATOR_PREFIX)) {
      onFieldChange(evaluatorField(selected.slice(EVALUATOR_PREFIX.length)));
    } else {
      onFieldChange(selected);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <SelectRoot value={selectValueFor(field)} onValueChange={onSelectField} disabled={disabled}>
        <SelectTrigger
          className="w-48"
          placeholder="Select field"
          aria-label="Sort field"
          renderValue={(v) => (typeof v === 'string' && v ? labelForOption(v) : undefined)}
        />
        <SelectContent className="w-(--radix-popper-anchor-width)">
          <SelectListbox>
            {STATIC_FIELDS.map((f) => (
              <SelectItem key={f.value} value={f.value}>
                {f.label}
              </SelectItem>
            ))}
            {evaluators.map((name) => (
              <SelectItem key={name} value={`${EVALUATOR_PREFIX}${name}`}>
                {`Avg ${snakeCaseToTitleCase(name)}`}
              </SelectItem>
            ))}
          </SelectListbox>
        </SelectContent>
      </SelectRoot>

      <SegmentedControl
        size="tiny"
        value={desc ? 'desc' : 'asc'}
        onValueChange={(d: string) => onDescChange(d === 'desc')}
        items={[
          { value: 'asc', children: 'Asc' },
          { value: 'desc', children: 'Desc' },
        ]}
      />

      <Button
        kind="tertiary"
        color="neutral"
        size="small"
        aria-label="Remove sort field"
        onClick={onRemove}
        disabled={disabled || !canRemove}
      >
        <Close className="size-4" />
      </Button>
    </div>
  );
};

export interface DefaultSortControlProps {
  value: string;
  onChange: (next: string) => void;
  /** Known evaluator names to offer as first-class options (edit modal). Empty at create time. */
  evaluatorOptions?: string[];
  disabled?: boolean;
}

export const DefaultSortControl: FC<DefaultSortControlProps> = ({
  value,
  onChange,
  evaluatorOptions = [],
  disabled,
}) => {
  // A default sort always has at least one field; fall back to a single default row if empty.
  const parsed = parseSortList(value);
  const rows: SortEntry[] = parsed.length ? parsed : parseSortList(DEFAULT_SORT);

  // Keep every currently-selected evaluator selectable even if it wasn't among the discovered options
  // (e.g. a saved sort whose evaluator isn't in the sampled experiments), so it stays visible.
  const selectedEvaluators = rows
    .filter((row) => isEvaluatorField(row.field))
    .map((row) => evaluatorNameOf(row.field));
  const evaluators = Array.from(new Set([...evaluatorOptions, ...selectedEvaluators]));

  const emit = (next: SortEntry[]) => onChange(next.length ? formatSortList(next) : DEFAULT_SORT);
  const updateRow = (index: number, patch: Partial<SortEntry>) =>
    emit(rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  const removeRow = (index: number) => emit(rows.filter((_, i) => i !== index));
  const addRow = () => {
    // Default the new row to a static field not already used, so it isn't a duplicate of an existing row.
    const used = new Set(rows.map((row) => row.field));
    const nextField =
      STATIC_FIELDS.find((f) => !used.has(f.value))?.value ?? STATIC_FIELDS[0].value;
    emit([...rows, { field: nextField, desc: true }]);
  };

  return (
    <FormField slotLabel="Default Sort">
      <Stack gap="density-md">
        <Text kind="body/regular/sm" className="text-secondary">
          Sets the default sort order for all users when they open this group. The first field is
          the primary sort; add more fields to break ties.
        </Text>
        <Stack gap="density-sm">
          {rows.map((row, index) => (
            <SortFieldRow
              // Rows are fully controlled by props (no internal state), so an index key is stable here.
              key={index}
              field={row.field}
              desc={row.desc}
              evaluators={evaluators}
              disabled={disabled}
              canRemove={rows.length > 1}
              onFieldChange={(field) => updateRow(index, { field })}
              onDescChange={(desc) => updateRow(index, { desc })}
              onRemove={() => removeRow(index)}
            />
          ))}
        </Stack>
        <div>
          <Button kind="tertiary" color="neutral" size="small" onClick={addRow} disabled={disabled}>
            <Add className="size-4" />
            Add sort field
          </Button>
        </div>
      </Stack>
    </FormField>
  );
};
