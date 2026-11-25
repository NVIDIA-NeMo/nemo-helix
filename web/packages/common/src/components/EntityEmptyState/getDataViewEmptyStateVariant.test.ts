// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getDataViewEmptyStateVariant } from '@nemo/common/src/components/EntityEmptyState/getDataViewEmptyStateVariant';

describe('getDataViewEmptyStateVariant', () => {
  const clear = () => {};

  it('is first-use when nothing narrows the view', () => {
    expect(
      getDataViewEmptyStateVariant({ hasFiltersApplied: false, hasSearchApplied: false }, clear)
    ).toEqual({ variant: 'first-use' });
  });

  it.each([
    { hasFiltersApplied: true, hasSearchApplied: false },
    { hasFiltersApplied: false, hasSearchApplied: true },
    { hasFiltersApplied: true, hasSearchApplied: true },
  ])('is no-results with the clear handler when %o', (state) => {
    expect(getDataViewEmptyStateVariant(state, clear)).toEqual({
      variant: 'no-results',
      onClearFilters: clear,
    });
  });
});
