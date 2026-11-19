// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FelixActivity } from '@studio/components/FelixAssistant/felixActivity';
import { usePrefersReducedMotion } from '@studio/components/FelixAssistant/usePrefersReducedMotion';
import { useEffect, useState } from 'react';

/** A pose stays up this long after its tool call ends, so quick back-to-back calls don't strobe. */
export const POSE_HOLD_MS = 2_000;
/** How long each hammer frame is shown while he hammers (one full swing is two of these). */
export const HAMMER_FRAME_MS = 450;
/** While reading he mostly looks at the page, glancing up every 3-6 seconds (randomised). */
export const GLANCE_MIN_DELAY_MS = 3_000;
export const GLANCE_MAX_DELAY_MS = 6_000;
/** How long each glance up lasts. */
export const GLANCE_HOLD_MS = 800;

export const getGlanceDelayMs = (random: () => number = Math.random): number =>
  GLANCE_MIN_DELAY_MS + random() * (GLANCE_MAX_DELAY_MS - GLANCE_MIN_DELAY_MS);

interface UseFelixPoseOptions {
  /** What the assistant is doing right now, if it matches a pose. */
  activity: FelixActivity | null;
  /** Whether a run is in progress; when it ends, any pose ends with it. */
  isRunning: boolean;
}

/**
 * Turns the assistant's current activity into Felix's pose, and drives its small animations.
 *
 * - A new activity shows immediately; when it stops, the pose is held for a moment (so a run of
 *   quick tool calls reads as one steady pose) unless the whole run has ended.
 * - While `building`, `hammerUp` alternates so the hammer goes up and down.
 * - While `docs`, he reads with his eyes on the page and `glancingUp` flips on briefly every few
 *   seconds, as if checking in.
 * - For users who prefer reduced motion both stay at rest: they still see what he is doing, without
 *   the repeated motion.
 */
export const useFelixPose = ({ activity, isRunning }: UseFelixPoseOptions) => {
  const [pose, setPose] = useState<FelixActivity | null>(null);
  const [hammerUp, setHammerUp] = useState(false);
  const [glancingUp, setGlancingUp] = useState(false);
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (!isRunning) {
      setPose(null);
      return;
    }
    if (activity) {
      setPose(activity);
      return;
    }

    const timer = setTimeout(() => setPose(null), POSE_HOLD_MS);
    return () => clearTimeout(timer);
  }, [activity, isRunning]);

  const isHammering = pose === 'building' && !prefersReducedMotion;
  useEffect(() => {
    if (!isHammering) {
      setHammerUp(false);
      return;
    }

    const timer = setInterval(() => setHammerUp((up) => !up), HAMMER_FRAME_MS);
    return () => clearInterval(timer);
  }, [isHammering]);

  const isReading = pose === 'docs' && !prefersReducedMotion;
  useEffect(() => {
    if (!isReading) {
      setGlancingUp(false);
      return;
    }

    // Each new reading spell starts eyes-down, then glances up after a random wait, and so on.
    let timer: ReturnType<typeof setTimeout>;
    const scheduleGlance = () => {
      timer = setTimeout(() => {
        setGlancingUp(true);
        timer = setTimeout(() => {
          setGlancingUp(false);
          scheduleGlance();
        }, GLANCE_HOLD_MS);
      }, getGlanceDelayMs());
    };
    scheduleGlance();

    return () => clearTimeout(timer);
  }, [isReading]);

  return { pose, hammerUp, glancingUp };
};
