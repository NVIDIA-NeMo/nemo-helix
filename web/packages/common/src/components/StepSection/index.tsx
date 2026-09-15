// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { Check } from 'lucide-react';
import { FC, ReactNode } from 'react';

/** Where a step sits in the sequence. One value rather than a pair of booleans,
 *  so "locked but complete" cannot be expressed. */
export type StepStatus = 'upcoming' | 'active' | 'complete';

/** Mirrors KUI's `.nv-stepper-node`: an unfilled ring that turns green when the
 *  step is live, and only fills once the step is done. */
const BADGE_CLASS_NAME: Record<StepStatus, string> = {
  upcoming: 'border-base text-secondary',
  active: 'border-interaction-selected text-strong',
  complete: 'border-transparent bg-interaction-primary-base text-inverse',
};

const TITLE_CLASS_NAME: Record<StepStatus, string> = {
  upcoming: 'text-disabled',
  active: '',
  complete: '',
};

const CONTENT_CLASS_NAME: Record<StepStatus, string> = {
  upcoming: 'opacity-60',
  active: '',
  complete: '',
};

interface StepSectionProps {
  step: number;
  title: string;
  status: StepStatus;
  /** The prerequisite, phrased as what to do rather than what is blocked.
   *  Shown only while the step is upcoming. */
  lockedHint?: string;
  children: ReactNode;
}

/**
 * One numbered step of a guided, single-page form: a step badge and title with
 * its content beneath, where later steps stay visible but inert until earlier
 * ones are satisfied.
 *
 * Distinct from KUI's ``Stepper``, which tracks movement BETWEEN screens and
 * renders the whole sequence as one horizontal track. This is for steps that are
 * all on screen at once, each heading its own region. The badge deliberately
 * mirrors ``.nv-stepper-node`` so the two read as the same language.
 *
 * Upcoming steps stay visible so the shape of the whole task is legible up
 * front; only their inputs go inert. Disabling (rather than hiding) is right
 * here because the prerequisite is proximate -- it is the step immediately
 * above, on the same screen -- and every upcoming step states that prerequisite
 * as an action rather than a refusal.
 *
 * The lock is a real ``fieldset disabled`` rather than opacity plus
 * pointer-events: that genuinely inerts the controls, so they are skipped by
 * keyboard and assistive tech instead of merely looking unavailable.
 */
export const StepSection: FC<StepSectionProps> = ({
  step,
  title,
  status,
  lockedHint,
  children,
}) => {
  const upcoming = status === 'upcoming';

  return (
    <Stack justify="start" gap="density-lg" className="min-w-0">
      <Flex align="center" gap="density-sm">
        <Flex
          align="center"
          justify="center"
          aria-hidden
          className={`size-6 shrink-0 rounded-full border-2 text-sm ${BADGE_CLASS_NAME[status]}`}
        >
          {status === 'complete' ? <Check width={14} height={14} /> : step}
        </Flex>
        <Text kind="label/bold/xl" className={TITLE_CLASS_NAME[status]}>
          {title}
        </Text>
      </Flex>

      {upcoming && lockedHint ? (
        <Text kind="body/regular/md" className="text-placeholder">
          {lockedHint}
        </Text>
      ) : null}

      <fieldset
        disabled={upcoming}
        className={`m-0 min-w-0 border-0 p-0 ${CONTENT_CLASS_NAME[status]}`}
      >
        <Stack justify="start" gap="density-lg" className="min-w-0">
          {children}
        </Stack>
      </fieldset>
    </Stack>
  );
};
