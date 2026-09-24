// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  decodeColumnFiltersParam,
  encodeColumnFiltersParam,
  FILTERS_SEARCH_PARAM,
} from '@nemo/common/src/hooks/useStudioDataViewState/columnFiltersParam';

describe('columnFiltersParam', () => {
  it('round-trips column filters through a URL search param', () => {
    const filters = [
      { id: 'agent_name', value: 'email-security-triage' },
      { id: 'started_at', value: { $gte: '2026-09-01T00:00:00Z' } },
    ];
    const params = new URLSearchParams({
      [FILTERS_SEARCH_PARAM]: encodeColumnFiltersParam(filters),
    });

    const reparsed = new URLSearchParams(params.toString());

    expect(decodeColumnFiltersParam(reparsed.get(FILTERS_SEARCH_PARAM))).toEqual(filters);
  });

  it.each([null, '', 'not-json', encodeURIComponent('{"id":"agent_name"}')])(
    'yields no filters for %j',
    (value) => {
      expect(decodeColumnFiltersParam(value)).toEqual([]);
    }
  );
});
