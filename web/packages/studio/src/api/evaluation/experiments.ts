// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { listExperiments } from '@nemo/sdk/generated/platform/experiments';
import type { ExperimentResponse } from '@nemo/sdk/generated/platform/schema';

/** Backend caps page_size at 1000. */
const PAGE_SIZE = 1000;

/** Stops a pathological workspace from paging forever. */
const MAX_PAGES = 20;

/** Resolves experiments by id. The list endpoint has no id filter and the get endpoint keys on
 *  name, so page through the list (newest first) until every id is accounted for. Ids that are
 *  never found — deleted or inaccessible experiments — are simply absent from the map. */
export const fetchExperimentsByIds = async (
  workspace: string,
  ids: string[],
  signal?: AbortSignal
): Promise<Map<string, ExperimentResponse>> => {
  const found = new Map<string, ExperimentResponse>();
  const wanted = new Set(ids);
  if (wanted.size === 0) return found;

  for (let page = 1; page <= MAX_PAGES; page++) {
    const res = await listExperiments(
      workspace,
      { page, page_size: PAGE_SIZE, sort: '-created_at' },
      signal
    );
    const batch = res?.data ?? [];
    for (const experiment of batch) {
      if (wanted.delete(experiment.id)) found.set(experiment.id, experiment);
    }
    if (wanted.size === 0 || batch.length < PAGE_SIZE) break;
  }
  return found;
};
