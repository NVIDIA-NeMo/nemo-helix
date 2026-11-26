// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AttackOutcome,
  AttackStatus,
  BenignOutcome,
  BenignStatus,
  ValidationAttackRow,
  ValidationBenignRow,
  ValidationReport,
  ValidationSummary,
} from '@agent-hardener/components/useSanityCheck';
import { FEEDBACK } from '@agent-hardener/theme';
import { Badge, Card, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { FC } from 'react';

// Scorecard tallies: `errored` items could not be decided, and `notCounted` ones are left out of `total`.
export interface ScorecardCounts {
  ok: number;
  bad: number;
  errored: number;
  notCounted: number;
  total: number;
}

// How an attack counts; a result written before the traces labels has only a status.
export const attackOutcome = (row: ValidationAttackRow): AttackOutcome => {
  if (row.outcome) {
    return row.outcome;
  }
  if (row.status === 'blocked') {
    return 'blocked';
  }
  if (row.status === 'not_blocked') {
    return 'landed';
  }
  return 'error';
};

// How a benign request counts, with the same fallback for results without an outcome.
export const benignOutcome = (row: ValidationBenignRow): BenignOutcome => {
  if (row.outcome) {
    return row.outcome;
  }
  if (row.status === 'passed') {
    return 'served';
  }
  if (row.status === 'refused') {
    return 'refused';
  }
  return 'error';
};

const count = <T,>(values: T[], value: T): number => values.filter((v) => v === value).length;

export const countBenign = (
  rows: ValidationBenignRow[],
  summary: ValidationSummary
): ScorecardCounts => {
  // A truncated artifact still has a summary, so count from it.
  if (rows.length === 0) {
    const notCounted = summary.benign_not_counted ?? 0;
    const total = Math.max((summary.benign_total ?? 0) - notCounted, 0);
    if (summary.benign_served !== undefined) {
      const ok = summary.benign_served;
      const bad = summary.benign_refused ?? 0;
      return { ok, bad, errored: Math.max(total - ok - bad, 0), notCounted, total };
    }
    const bad = summary.benign_false_positives ?? 0;
    return { ok: Math.max(total - bad, 0), bad, errored: 0, notCounted, total };
  }
  const outcomes = rows.map(benignOutcome);
  const notCounted = count(outcomes, 'not_counted');
  const ok = count(outcomes, 'served');
  const bad = count(outcomes, 'refused');
  const total = rows.length - notCounted;
  return { ok, bad, errored: total - ok - bad, notCounted, total };
};

export const countAttacks = (
  rows: ValidationAttackRow[],
  summary: ValidationSummary
): ScorecardCounts => {
  if (rows.length === 0) {
    const total = summary.attacks_total ?? 0;
    const ok = summary.attacks_blocked ?? 0;
    const bad = summary.attacks_landed ?? Math.max(total - ok, 0);
    return { ok, bad, errored: Math.max(total - ok - bad, 0), notCounted: 0, total };
  }
  const outcomes = rows.map(attackOutcome);
  const ok = count(outcomes, 'blocked');
  const bad = count(outcomes, 'landed');
  return { ok, bad, errored: rows.length - ok - bad, notCounted: 0, total: rows.length };
};

const plural = (count: number, noun: string): string =>
  `${count} ${noun}${count === 1 ? '' : 's'}`;

/** `7 / 9 · 1 false positive · 1 error · 1 not counted` — trailing segments appear only when non-zero. */
export const formatCounts = (counts: ScorecardCounts, badNoun: string): string =>
  [
    `${counts.ok} / ${counts.total}`,
    counts.bad > 0 ? plural(counts.bad, badNoun) : '',
    counts.errored > 0 ? plural(counts.errored, 'error') : '',
    counts.notCounted > 0 ? `${counts.notCounted} not counted` : '',
  ]
    .filter(Boolean)
    .join(' · ');

// Green only when something was measured and nothing failed or errored.
export const attacksHeld = (counts: ScorecardCounts): boolean =>
  counts.total > 0 && counts.ok === counts.total && counts.errored === 0;

export const benignHeld = (counts: ScorecardCounts): boolean =>
  counts.total > 0 && counts.bad === 0 && counts.errored === 0;

// The validator's specific label, shown beside the verdict badge; it never decides the badge's color.
const ATTACK_REASONS: Partial<Record<AttackStatus, string>> = {
  blocked_guardrail: 'By guardrail',
  blocked_network: 'By network',
  agent_refused: 'Agent refused',
  tool_not_called: 'Tool not called',
  tool_failed: 'Tool failed',
};

const BENIGN_REASONS: Partial<Record<BenignStatus, string>> = {
  blocked_guardrail: 'By guardrail',
  blocked_network: 'By network',
  agent_refused: 'Agent declined',
  tool_failed: 'Tool failed',
};

type BadgeColor = 'green' | 'yellow' | 'gray';

export interface Verdict {
  text: string;
  color: BadgeColor;
  reason?: string;
}

const ERROR_VERDICT: Verdict = { text: 'Error', color: 'gray' };

// Unknown outcomes are an error, as in the scorecard counts.
export const attackVerdict = (row: ValidationAttackRow): Verdict => {
  const reason = row.status ? ATTACK_REASONS[row.status] : undefined;
  switch (attackOutcome(row)) {
    case 'blocked':
      return { text: 'Blocked', color: 'green', reason };
    case 'landed':
      return { text: 'Not blocked', color: 'yellow', reason };
    default:
      return ERROR_VERDICT;
  }
};

export const benignVerdict = (row: ValidationBenignRow): Verdict => {
  const reason = row.status ? BENIGN_REASONS[row.status] : undefined;
  switch (benignOutcome(row)) {
    case 'served':
      return { text: 'Passed', color: 'green', reason };
    case 'refused':
      return { text: 'Wrongly blocked', color: 'yellow', reason };
    case 'not_counted':
      return { text: 'Not counted', color: 'gray', reason };
    default:
      return ERROR_VERDICT;
  }
};

const VerdictBadge: FC<{ verdict: Verdict }> = ({ verdict }) => (
  <Flex align="center" gap="density-xs" className="shrink-0">
    {verdict.reason ? (
      <Text kind="body/regular/xs" className="text-subtle">
        {verdict.reason}
      </Text>
    ) : null}
    <Badge color={verdict.color}>{verdict.text}</Badge>
  </Flex>
);

const Stat: FC<{ label: string; value: string; good: boolean }> = ({ label, value, good }) => (
  <Card className="flex-1 p-4">
    <Stack gap="density-xs">
      <Text kind="body/regular/sm" className="text-subtle">
        {label}
      </Text>
      <Text kind="title/lg" style={{ color: good ? FEEDBACK.success : FEEDBACK.warning }}>
        {value}
      </Text>
    </Stack>
  </Card>
);

const AttackRow: FC<{ row: ValidationAttackRow }> = ({ row }) => (
  <Flex justify="between" align="start" gap="density-md" className="min-w-0 py-2">
    <Stack gap="density-xxs" className="min-w-0">
      <Text kind="body/semibold/sm">{row.probe ?? row.attack_id ?? 'attack'}</Text>
      {row.goal ? (
        <Text kind="body/regular/sm" className="truncate text-subtle">
          {row.goal}
        </Text>
      ) : null}
      {row.prompt_excerpt ? (
        <Text kind="body/regular/xs" className="truncate text-subtle">
          {row.prompt_excerpt}
        </Text>
      ) : null}
    </Stack>
    <VerdictBadge verdict={attackVerdict(row)} />
  </Flex>
);

const BenignRow: FC<{ row: ValidationBenignRow }> = ({ row }) => (
  <Flex justify="between" align="start" gap="density-md" className="min-w-0 py-2">
    <Stack gap="density-xxs" className="min-w-0">
      <Text kind="body/semibold/sm">{row.tool ?? row.label ?? `request ${row.index ?? ''}`}</Text>
      {row.payload_excerpt ? (
        <Text kind="body/regular/xs" className="truncate text-subtle">
          {row.payload_excerpt}
        </Text>
      ) : null}
    </Stack>
    <VerdictBadge verdict={benignVerdict(row)} />
  </Flex>
);

interface SanityCheckReportProps {
  report: ValidationReport;
}

// The sanity-check scorecard: attacks the chosen defenses block, and benign requests they wrongly block.
export const SanityCheckReport: FC<SanityCheckReportProps> = ({ report }) => {
  const { summary, attacks, benign } = report;
  const attackCounts = countAttacks(attacks, summary);
  const benignCounts = countBenign(benign, summary);
  const attacksGood = attacksHeld(attackCounts);
  const benignGood = benignHeld(benignCounts);

  return (
    <Stack gap="density-xl">
      <Flex gap="density-md">
        <Stat
          label="Attacks blocked"
          value={formatCounts(attackCounts, 'not blocked')}
          good={attacksGood}
        />
        <Stat
          label="Benign preserved"
          value={formatCounts(benignCounts, 'false positive')}
          good={benignGood}
        />
      </Flex>

      {attacks.length > 0 ? (
        <Stack gap="density-xs">
          <Text kind="body/semibold/md">Attacks ({attacks.length})</Text>
          <Card className="p-3 [&>*+*]:border-t [&>*+*]:border-base">
            {attacks.map((row, i) => (
              <AttackRow key={row.attack_id ?? i} row={row} />
            ))}
          </Card>
        </Stack>
      ) : null}

      {benign.length > 0 ? (
        <Stack gap="density-xs">
          <Text kind="body/semibold/md">Benign requests ({benign.length})</Text>
          <Card className="p-3 [&>*+*]:border-t [&>*+*]:border-base">
            {benign.map((row, i) => (
              <BenignRow key={row.index ?? i} row={row} />
            ))}
          </Card>
        </Stack>
      ) : null}
    </Stack>
  );
};
