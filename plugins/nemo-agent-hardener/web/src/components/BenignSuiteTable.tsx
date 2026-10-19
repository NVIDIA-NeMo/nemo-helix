// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SuiteRow } from '@agent-hardener/components/hitlTypes';
import { Button, Flex, Stack, Text, TextInput } from '@nvidia/foundations-react-core';
import { Check, Pencil, Plus, Trash, X } from 'lucide-react';
import { FC, useState } from 'react';

interface BenignSuiteTableProps {
  value: SuiteRow[];
  onChange: (rows: SuiteRow[]) => void;
  disabled?: boolean;
}

const EMPTY_ROW: SuiteRow = { tool: '', payload: '', label: 'benign', persona: '', rationale: '' };

const COLUMNS: { key: keyof SuiteRow; label: string; width: string }[] = [
  { key: 'tool', label: 'Tool', width: 'w-[14%]' },
  { key: 'payload', label: 'Payload', width: 'w-[26%]' },
  { key: 'label', label: 'Label', width: 'w-[12%]' },
  { key: 'rationale', label: 'Rationale', width: 'w-[26%]' },
  { key: 'persona', label: 'Persona', width: 'w-[14%]' },
];

interface EditState {
  index: number; // an existing row index, or value.length when adding a new row
  draft: SuiteRow;
}

const cell = (row: SuiteRow, key: keyof SuiteRow): string => String(row[key] ?? '');

// The manifest's benign suite as an editable requests.csv: a scrollable table with per-row inline edit +
// delete and an add-row. Each commit (save / delete / add) calls onChange so the parent persists the suite.
export const BenignSuiteTable: FC<BenignSuiteTableProps> = ({ value, onChange, disabled }) => {
  const [edit, setEdit] = useState<EditState | null>(null);
  const busy = disabled || edit !== null;
  const isAdding = edit !== null && edit.index === value.length;

  const setField = (key: keyof SuiteRow, v: string) =>
    setEdit((e) => (e ? { ...e, draft: { ...e.draft, [key]: v } } : e));
  const cancel = () => setEdit(null);
  const save = () => {
    if (!edit) return;
    onChange(
      edit.index < value.length
        ? value.map((r, i) => (i === edit.index ? edit.draft : r))
        : [...value, edit.draft]
    );
    setEdit(null);
  };
  const remove = (index: number) => {
    onChange(value.filter((_, i) => i !== index));
    setEdit(null);
  };

  const editCells = (draft: SuiteRow) =>
    COLUMNS.map((c) => (
      <td key={c.key} className="px-3 py-2 align-top">
        <TextInput
          value={cell(draft, c.key)}
          disabled={disabled}
          onChange={(e) => setField(c.key, e.target.value)}
        />
      </td>
    ));
  const rowActions = (editing: boolean, index: number) => (
    <td className="px-3 py-2 align-top">
      <Flex gap="density-xs">
        {editing ? (
          <>
            <Button
              kind="tertiary"
              size="small"
              aria-label="Save row"
              disabled={disabled}
              onClick={save}
            >
              <Check className="h-4 w-4" />
            </Button>
            <Button kind="tertiary" size="small" aria-label="Cancel edit" onClick={cancel}>
              <X className="h-4 w-4" />
            </Button>
          </>
        ) : (
          <>
            <Button
              kind="tertiary"
              size="small"
              aria-label="Edit row"
              disabled={busy}
              onClick={() => setEdit({ index, draft: { ...value[index] } })}
            >
              <Pencil className="h-4 w-4" />
            </Button>
            <Button
              kind="tertiary"
              color="danger"
              size="small"
              aria-label="Delete row"
              disabled={busy}
              onClick={() => remove(index)}
            >
              <Trash className="h-4 w-4" />
            </Button>
          </>
        )}
      </Flex>
    </td>
  );

  return (
    <Stack gap="density-md">
      <div className="max-h-[360px] overflow-auto rounded-md border border-base">
        <table className="w-full table-fixed border-collapse text-sm">
          <thead>
            <tr className="border-b border-base">
              {COLUMNS.map((c) => (
                <th key={c.key} className={`${c.width} px-3 py-2 text-left`}>
                  <Text kind="body/semibold/sm" className="text-subtle">
                    {c.label}
                  </Text>
                </th>
              ))}
              <th className="px-3 py-2" style={{ width: 88 }} />
            </tr>
          </thead>
          <tbody>
            {value.length === 0 && !isAdding ? (
              <tr>
                <td colSpan={COLUMNS.length + 1} className="px-3 py-4">
                  <Text kind="body/regular/sm" className="text-subtle">
                    No benign requests yet. Add one, or generate the suite.
                  </Text>
                </td>
              </tr>
            ) : null}
            {value.map((row, index) => {
              const editing = edit?.index === index;
              return (
                <tr key={index} className="border-b border-base">
                  {editing
                    ? editCells(edit.draft)
                    : COLUMNS.map((c) => (
                        <td key={c.key} className="px-3 py-2 align-top">
                          <span className="whitespace-pre-wrap break-words">
                            {cell(row, c.key) || '—'}
                          </span>
                        </td>
                      ))}
                  {rowActions(Boolean(editing), index)}
                </tr>
              );
            })}
            {isAdding ? (
              <tr className="border-b border-base">
                {editCells(edit.draft)}
                {rowActions(true, edit.index)}
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      <Flex>
        <Button
          kind="secondary"
          disabled={busy}
          onClick={() => setEdit({ index: value.length, draft: { ...EMPTY_ROW } })}
        >
          <Plus className="h-4 w-4" /> Add request
        </Button>
      </Flex>
    </Stack>
  );
};
