// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useInnerDataViewContext } from '@nemo/common/src/components/DataView/internal';
import { Filter } from '@nemo/icons/icons';
import { Button } from '@nvidia/foundations-react-core';

interface FilterPanelToggleProps {
  showFilters: boolean;
  onToggle: () => void;
}

/** Renders the filter toggle only when at least one column defines `meta.filter`. */
export function FilterPanelToggle({ showFilters, onToggle }: FilterPanelToggleProps) {
  const { table } = useInnerDataViewContext();
  if (!table.getAllLeafColumns().some((col) => col.getCanFilter())) return null;

  return (
    <Button
      kind="secondary"
      onClick={onToggle}
      data-testid="open-filters-button"
      aria-pressed={showFilters}
      className="ml-auto"
    >
      <Filter width={12} height={12} />
      Filter
    </Button>
  );
}
