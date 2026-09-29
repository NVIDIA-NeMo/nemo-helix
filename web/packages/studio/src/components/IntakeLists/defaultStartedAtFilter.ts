// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  encodeColumnFiltersParam,
  FILTERS_SEARCH_PARAM,
} from '@nemo/common/src/hooks/useStudioDataViewState/columnFiltersParam';
import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router';

export const INTAKE_DEFAULT_LOOKBACK_DAYS = 30;

export interface StartedAtFilterEntry {
  id: 'started_at';
  value: { $gte: string };
}

/**
 * Default window for the intake browse views: telemetry started in the last
 * 30 days. Seeded as a visible, clearable column filter so unbounded history
 * is one click away — the intake API itself applies no implicit time bound.
 */
export const makeDefaultStartedAtFilter = (): StartedAtFilterEntry => {
  const from = new Date();
  from.setDate(from.getDate() - INTAKE_DEFAULT_LOOKBACK_DAYS);
  from.setHours(0, 0, 0, 0);
  return { id: 'started_at', value: { $gte: from.toISOString() } };
};

/** Whether a column filter entry is exactly the seeded default (untouched by the user). */
export const isDefaultStartedAtFilter = (
  filter: { id: string; value: unknown },
  defaultFilter: StartedAtFilterEntry
): boolean =>
  filter.id === defaultFilter.id &&
  JSON.stringify(filter.value) === JSON.stringify(defaultFilter.value);

/**
 * Seed the default filter into the URL `filters` param when a browse view
 * mounts without one. `useStudioDataViewState` then adopts it exactly like a
 * user-set filter (chip, URL sharing, clearing), so no shared-hook changes are
 * needed. Clearing lasts for the visit; the next mount without a `filters`
 * param re-seeds.
 *
 * Returns false until the URL reflects the seed — gate list queries on it so
 * the first request is never accidentally unbounded.
 */
export const useSeededStartedAtFilter = (defaultFilter: StartedAtFilterEntry | null): boolean => {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlHasFilters = searchParams.has(FILTERS_SEARCH_PARAM);
  const [seeded, setSeeded] = useState(() => defaultFilter === null || urlHasFilters);

  useEffect(() => {
    if (seeded) return;
    if (urlHasFilters) {
      setSeeded(true);
      return;
    }
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (!next.has(FILTERS_SEARCH_PARAM) && defaultFilter) {
          next.set(FILTERS_SEARCH_PARAM, encodeColumnFiltersParam([defaultFilter]));
        }
        return next;
      },
      { replace: true }
    );
  }, [seeded, urlHasFilters, defaultFilter, setSearchParams]);

  return seeded;
};
