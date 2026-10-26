// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ChevronLeft, ChevronRight, LockClosed, LayoutRows, Trash } from '@nemo/icons/icons';
import {
  Button,
  Flex,
  Select,
  SidePanel,
  Stack,
  Tag,
  Text,
  TextArea,
  TextInput,
} from '@nvidia/foundations-react-core';
import { COLUMN_TYPE_TAG_COLOR } from '@studio/components/FileRowEditor/constants';
import { formatCellValue } from '@studio/components/FileRowEditor/schema';
import {
  ROW_ID_KEY,
  type DataFileColumn,
  type DataFileColumnType,
  type DataFileRow,
} from '@studio/components/FileRowEditor/types';
import { type FC, type FormEvent, type ReactNode, useEffect, useRef, useState } from 'react';

const BOOLEAN_OPTIONS = ['true', 'false'];

/** Coerces any value to the string an editor text control expects. */
const toInputString = (value: unknown): string =>
  value === null || value === undefined ? '' : String(value);

interface EditorFieldProps {
  label: string;
  type: DataFileColumnType;
  children: ReactNode;
}

/** Field shell: bold label + a logical-type tag, then the control. */
const EditorField: FC<EditorFieldProps> = ({ label, type, children }) => (
  <Stack gap="density-xs" className="w-full">
    <Flex align="center" gap="density-sm">
      <Text kind="body/bold/sm">{label}</Text>
      <Tag kind="outline" color={COLUMN_TYPE_TAG_COLOR[type]} readOnly>
        {type}
      </Tag>
    </Flex>
    {children}
  </Stack>
);

interface JsonFieldProps {
  label: string;
  value: unknown;
  onChange: (value: unknown) => void;
}

/**
 * JSON editor with its own text + validity state, so partial/invalid edits stay in the
 * box without clobbering the committed value. Only valid JSON is propagated upward.
 */
const JsonField: FC<JsonFieldProps> = ({ label, value, onChange }) => {
  const [text, setText] = useState(() => JSON.stringify(value ?? null, null, 2));
  const [error, setError] = useState(false);
  // Track the value we last propagated so we can tell our own commits (skip — they would
  // reformat mid-edit) apart from an external change, e.g. the same row being reopened
  // after a cancel. The latter must reset any stale invalid draft back to committed JSON.
  const committedValue = useRef(value);

  useEffect(() => {
    if (value !== committedValue.current) {
      committedValue.current = value;
      setText(JSON.stringify(value ?? null, null, 2));
      setError(false);
    }
  }, [value]);

  const commit = (next: string) => {
    setText(next);
    try {
      const parsed: unknown = JSON.parse(next);
      setError(false);
      committedValue.current = parsed;
      onChange(parsed);
    } catch {
      setError(true);
    }
  };

  return (
    <>
      <TextArea
        value={text}
        onValueChange={commit}
        rows={5}
        status={error ? 'error' : undefined}
        className="font-mono text-[12px]"
        attributes={{ TextAreaElement: { 'aria-label': label } }}
      />
      {error ? (
        <Text kind="body/regular/xs" className="text-danger">
          Invalid JSON — changes are not saved until valid.
        </Text>
      ) : null}
    </>
  );
};

interface FieldControlProps {
  column: DataFileColumn;
  value: unknown;
  onChange: (value: unknown) => void;
}

/** Renders the editor control appropriate to a column's logical type. */
const FieldControl: FC<FieldControlProps> = ({ column, value, onChange }) => {
  if (column.editable === false) {
    return (
      <TextInput
        value={formatCellValue(value, column.type)}
        readOnly
        slotEnd={<LockClosed size="14" className="text-secondary" />}
        attributes={{ Input: { 'aria-label': column.key } }}
      />
    );
  }

  switch (column.type) {
    case 'boolean':
      return (
        <Select
          items={BOOLEAN_OPTIONS}
          value={String(Boolean(value))}
          onValueChange={(next) => onChange(next === 'true')}
          attributes={{ SelectTrigger: { 'aria-label': column.key } }}
        />
      );
    case 'int':
    case 'float':
      return (
        <TextInput
          type="number"
          step={column.type === 'int' ? 1 : 'any'}
          value={toInputString(value)}
          onValueChange={(next) => onChange(next === '' ? null : Number(next))}
          className="font-mono"
          attributes={{ Input: { 'aria-label': column.key } }}
        />
      );
    case 'json':
      return <JsonField label={column.key} value={value} onChange={onChange} />;
    default: {
      // Enum-like string column → single-select. Keep the current value selectable
      // even when it falls outside the inferred option set (e.g. edited data).
      if (column.options) {
        const current = toInputString(value);
        const items =
          current === '' || column.options.includes(current)
            ? column.options
            : [current, ...column.options];
        return (
          <Select
            items={items}
            value={current}
            onValueChange={onChange}
            attributes={{ SelectTrigger: { 'aria-label': column.key } }}
          />
        );
      }
      return column.multiline ? (
        <TextArea
          value={toInputString(value)}
          onValueChange={onChange}
          rows={4}
          attributes={{ TextAreaElement: { 'aria-label': column.key } }}
        />
      ) : (
        <TextInput
          value={toInputString(value)}
          onValueChange={onChange}
          attributes={{ Input: { 'aria-label': column.key } }}
        />
      );
    }
  }
};

interface EditorBodyProps {
  columns: DataFileColumn[];
  draft: DataFileRow;
  onFieldChange: (key: string, value: unknown) => void;
}

/**
 * The editable field stack, built from the schema. Mounted with a `key` of the row id
 * (by the panel) so field-local state — e.g. JSON text — resets per row. Autofocuses
 * the first editable control on open.
 */
const EditorBody: FC<EditorBodyProps> = ({ columns, draft, onFieldChange }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const focusable = containerRef.current?.querySelector<HTMLElement>(
      'input:not([readonly]), textarea:not([readonly]), [role="combobox"]'
    );
    focusable?.focus();
  }, []);

  return (
    <div ref={containerRef}>
      <Stack gap="density-lg" className="w-full">
        {columns.map((column) => (
          <EditorField key={column.key} label={column.label} type={column.type}>
            <FieldControl
              column={column}
              value={draft[column.key]}
              onChange={(value) => onFieldChange(column.key, value)}
            />
          </EditorField>
        ))}
      </Stack>
    </div>
  );
};

export interface RowEditorPanelProps {
  /** Whether the panel is open. */
  open: boolean;
  /** The schema describing which fields to render, in order. */
  columns: DataFileColumn[];
  /** The working copy being edited, or null when nothing is selected. */
  draft: DataFileRow | null;
  /** 1-based position of the row within the dataset. */
  rowNumber: number;
  /** Total number of rows in the dataset. */
  totalRows: number;
  /** Whether the draft differs from the committed row. */
  isDirty: boolean;
  onFieldChange: (key: string, value: unknown) => void;
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
  onDelete: () => void;
  onSave: () => void;
}

/**
 * Row editor rendered as a KUI SidePanel — providing the slide-in animation, focus
 * trap (tab through fields), and Escape-to-close behavior natively. The panel stays
 * mounted and toggles `open` so the open/close transitions animate.
 */
export const RowEditorPanel: FC<RowEditorPanelProps> = ({
  open,
  columns,
  draft,
  rowNumber,
  totalRows,
  isDirty,
  onFieldChange,
  onClose,
  onPrev,
  onNext,
  onDelete,
  onSave,
}) => {
  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    onSave();
  };

  return (
    <SidePanel
      open={open}
      onOpenChange={(next) => {
        if (!next) {
          onClose();
        }
      }}
      side="right"
      bordered
      modal
      className="w-[440px]"
      slotHeading="Edit Row"
      slotNavigation={
        draft ? (
          <Flex align="center" justify="between" className="w-full" paddingY="3">
            <Flex align="center" gap="density-sm">
              <Button
                kind="secondary"
                size="small"
                aria-label="Previous row"
                onClick={onPrev}
                disabled={rowNumber <= 1}
              >
                <ChevronLeft size="16" />
              </Button>
              <Flex align="center" gap="density-xs" className="text-secondary">
                <LayoutRows size="14" />
                <Text kind="body/regular/sm" className="text-secondary">
                  Row {rowNumber.toLocaleString()} of {totalRows.toLocaleString()}
                </Text>
              </Flex>
              <Button
                kind="secondary"
                size="small"
                aria-label="Next row"
                onClick={onNext}
                disabled={rowNumber >= totalRows}
              >
                <ChevronRight size="16" />
              </Button>
            </Flex>
            {isDirty ? (
              <Tag kind="solid" color="yellow" readOnly>
                Unsaved edits
              </Tag>
            ) : null}
          </Flex>
        ) : null
      }
      slotFooter={
        draft ? (
          <Flex align="center" justify="between" className="w-full">
            <Button type="button" kind="secondary" color="danger" onClick={onDelete}>
              <Trash size="15" />
              Delete
            </Button>
            <Flex align="center" gap="density-sm">
              <Button type="button" kind="secondary" color="neutral" onClick={onClose}>
                Cancel
              </Button>
              <Button type="submit" kind="primary" color="brand">
                Stage Changes
              </Button>
            </Flex>
          </Flex>
        ) : null
      }
      renderContent={({ children }) => <form onSubmit={handleSubmit}>{children}</form>}
    >
      {draft ? (
        <EditorBody
          key={String(draft[ROW_ID_KEY])}
          columns={columns}
          draft={draft}
          onFieldChange={onFieldChange}
        />
      ) : null}
    </SidePanel>
  );
};
