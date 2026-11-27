// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Combobox } from '@nvidia/foundations-react-core';
import { FINETUNING_TYPE_FILTER_OPTIONS } from '@studio/components/dataViews/CustomModelsDataView/constants';
import { type FC, useEffect, useRef, useState } from 'react';

const ITEMS = FINETUNING_TYPE_FILTER_OPTIONS.map((o) => ({
  value: o.value,
  children: o.label,
  filterValue: o.label,
}));

export interface FinetuningTypeFilterFieldProps {
  value: string[];
  onValueChange: (value: string[]) => void;
}

/**
 * Multi-select finetuning types. Picks are held locally and committed when the listbox closes
 * (click away, Escape, Tab) or the field unmounts, so choosing several types fires one query
 * instead of one per click. Changes made while closed (the clear button) commit immediately.
 */
export const FinetuningTypeFilterField: FC<FinetuningTypeFilterFieldProps> = ({
  value,
  onValueChange,
}) => {
  const [draft, setDraft] = useState(value);
  const draftRef = useRef(value);
  const openRef = useRef(false);
  const committedRef = useRef(value);
  const onValueChangeRef = useRef(onValueChange);
  onValueChangeRef.current = onValueChange;

  const valueKey = value.join(',');
  // Pick up external changes (applied-filter chip removed, filters reset).
  useEffect(() => {
    committedRef.current = value;
    draftRef.current = value;
    setDraft(value);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on contents, not identity
  }, [valueKey]);

  const commit = () => {
    if (draftRef.current.join(',') === committedRef.current.join(',')) return;
    committedRef.current = draftRef.current;
    onValueChangeRef.current(draftRef.current);
  };

  // Commit pending picks if the filter UI unmounts while the listbox is open (refs only).
  useEffect(() => () => commit(), []);

  return (
    <Combobox
      aria-label="Finetuning type"
      data-testid="custom-models-finetuning-type-filter"
      dismissible
      multiple
      items={ITEMS}
      value={draft}
      placeholder="Search finetuning types..."
      onOpenChange={(open: boolean) => {
        openRef.current = open;
        if (!open) commit();
      }}
      onValueChange={(next: string[]) => {
        draftRef.current = next;
        setDraft(next);
        if (!openRef.current) commit();
      }}
    />
  );
};
