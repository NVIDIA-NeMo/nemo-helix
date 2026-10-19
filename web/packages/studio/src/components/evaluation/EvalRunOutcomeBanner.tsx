// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Badge, Banner } from '@nvidia/foundations-react-core';
import {
  type EvalRunOutcome,
  evalRunOutcomeLabel,
  evalRunOutcomeOf,
  evalRunOutcomeSeverity,
} from '@studio/api/evaluation/runOutcome';
import type { FC, ReactNode } from 'react';

interface EvalRunOutcomeBannerProps {
  /** The job's `status_details`, read for the evaluator's run rollup. */
  statusDetails: unknown;
  /** Rendered only on the error variant, where the reader will want the job logs. */
  slotActions?: ReactNode;
}

/** The evaluator's own account of a run, when it has something to say: an error banner for a run
 *  that scored nothing, a warning for one that scored only some of its units. Renders nothing for
 *  a clean run or a job that recorded no outcome. */
export const EvalRunOutcomeBanner: FC<EvalRunOutcomeBannerProps> = ({
  statusDetails,
  slotActions,
}) => {
  const outcome = evalRunOutcomeOf(statusDetails);
  const severity = outcome ? evalRunOutcomeSeverity(outcome) : null;
  if (!outcome || !severity) return null;
  return (
    <Banner
      kind="inline"
      status={severity}
      slotActions={severity === 'error' ? slotActions : undefined}
      data-testid="eval-run-outcome-banner"
    >
      {outcome.message}
    </Banner>
  );
};

interface EvalRunOutcomeBadgeProps {
  outcome: EvalRunOutcome | null;
}

/** List-cell companion to the banner: "0 of 28 scored" beside a status badge, coloured by the same
 *  severity. Renders nothing for a clean run so healthy rows stay uncluttered. */
export const EvalRunOutcomeBadge: FC<EvalRunOutcomeBadgeProps> = ({ outcome }) => {
  const severity = outcome ? evalRunOutcomeSeverity(outcome) : null;
  if (!outcome || !severity) return null;
  return (
    <Badge kind="outline" color={severity === 'error' ? 'red' : 'yellow'} title={outcome.message}>
      {evalRunOutcomeLabel(outcome)}
    </Badge>
  );
};
