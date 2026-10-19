// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SwarmEvent } from '@agent-hardener/components/eventTypes';
import { useAgentHardenerGetEvents } from '@agent-hardener/generated/api';
import { useEffect, useState } from 'react';

const POLL_INTERVAL_MS = 1000;

/**
 * Accumulate a run's event stream by polling.
 *
 * The full prefix is kept deliberately: `deriveSwarmState` folds from the first event (all nodes
 * `pending`) and relies on seeing every `phase_started`/`round_started`, so dropping the oldest events
 * would silently corrupt phase and round on long runs — exactly the determinism the fold promises.
 *
 * Pass *isTerminal* once the run can emit nothing further, so polling stops; without it the hook polls
 * for as long as the tab stays open. The trailing fetch still lands: the last batch advances `afterId`,
 * which changes the query key and triggers one final request before the interval goes idle.
 */
export const useSwarmEvents = (
  workspace: string,
  runName: string,
  isTerminal = false
): SwarmEvent[] => {
  const [afterId, setAfterId] = useState(0);
  const [allEvents, setAllEvents] = useState<SwarmEvent[]>([]);

  const { data } = useAgentHardenerGetEvents(
    workspace,
    runName,
    { after: afterId },
    {
      query: {
        enabled: Boolean(runName),
        refetchInterval: isTerminal ? false : POLL_INTERVAL_MS,
      },
    }
  );

  useEffect(() => {
    if (!data?.events?.length) return;
    // An event with no numeric id cannot advance the cursor: any stand-in value would sit far above
    // real ids, so `after` would match nothing and the feed would stop for the rest of the run.
    const next: SwarmEvent[] = data.events
      .filter((e: Record<string, unknown>) => typeof e['id'] === 'number')
      .map((e: Record<string, unknown>) => ({
        id: e['id'] as number,
        event: typeof e['event'] === 'string' ? e['event'] : '',
        payload: (e['payload'] ?? {}) as Record<string, unknown>,
        ts: Date.now(),
      }));
    if (!next.length) return;
    setAllEvents((prev) => [...prev, ...next]);
    setAfterId((prev) => Math.max(prev, ...next.map((e) => e.id)));
  }, [data]);

  useEffect(() => {
    setAllEvents([]);
    setAfterId(0);
  }, [workspace, runName]);

  return allEvents;
};
