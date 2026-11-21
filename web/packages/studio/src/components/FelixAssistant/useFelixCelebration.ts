// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

/** How long Felix celebrates a finished run, hands up, before going back to normal. */
export const CELEBRATION_MS = 3_000;

interface UseFelixCelebrationOptions {
  /** Whether a run is in progress. */
  isRunning: boolean;
  /** Whether the latest reply finished properly (not an error, not cancelled). */
  lastReplySucceeded: boolean;
}

/**
 * Tells Felix when to celebrate: for a few seconds right after a run finishes successfully.
 *
 * - It starts on the run's trailing edge only (running, then not running with a successful reply),
 *   so merely loading an old conversation or resetting the chat never triggers it.
 * - Failed or cancelled runs don't celebrate.
 * - A new run starting ends it straight away, so he gets back to work without a leftover cheer.
 *
 * This relies on the runtime marking the reply `complete` and ending the run in the same tick
 * (useCustomAssistantChatRuntime does both in one continuation), which React batches into a single
 * render, so `isRunning` and `lastReplySucceeded` always change together.
 */
export const useFelixCelebration = ({
  isRunning,
  lastReplySucceeded,
}: UseFelixCelebrationOptions): boolean => {
  const [isCelebrating, setIsCelebrating] = useState(false);
  const wasRunningRef = useRef(isRunning);

  useEffect(() => {
    const justFinished = wasRunningRef.current && !isRunning;
    wasRunningRef.current = isRunning;

    if (isRunning) {
      setIsCelebrating(false);
      return;
    }
    if (!justFinished || !lastReplySucceeded) return;

    setIsCelebrating(true);
    const timer = setTimeout(() => setIsCelebrating(false), CELEBRATION_MS);
    return () => clearTimeout(timer);
  }, [isRunning, lastReplySucceeded]);

  return isCelebrating;
};
