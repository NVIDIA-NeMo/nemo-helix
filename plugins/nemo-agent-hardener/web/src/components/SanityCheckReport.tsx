// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AttackOutcome,
  BenignOutcome,
  ValidationAttackRow,
  ValidationBenignRow,
  ValidationReport,
  ValidationSummary,
} from '@agent-hardener/components/useSanityCheck';
import { FEEDBACK } from '@agent-hardener/theme';
import { Badge, Card, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { FC } from 'react';

/**
 * Scorecard tallies.
 *
 * `ok` is the good outcome (attack blocked / benign still served), `bad` the actionable
 * failure, and `errored` the items that produced neither — the run could not decide. Those are
 * counted apart on purpose: folding them into `ok` overstates how well the defenses held, which
 * for a security tool is the wrong direction to be wrong in.
 *
 * `notCounted` (benign only) are requests the agent declined on its own: they say nothing about the
 * defenses, so they are shown but left out of `total`.
 */
export interface ScorecardCounts {
  ok: number;
  bad: number;
  errored: number;
  notCounted: number;
  total: number;
}

/** How an attack counts. A result written before the traces labels has no `outcome`, only its status. */
export const attackOutcome = (row: ValidationAttackRow): AttackOutcome =>
  row.outcome ??
  (row.status === 'blocked' ? 'blocked' : row.status === 'not_blocked' ? 'landed' : 'error');

/** How a benign request counts, with the same fallback for results without an `outcome`. */
export const benignOutcome = (row: ValidationBenignRow): BenignOutcome =>
  row.outcome ??
  (row.status === 'passed' ? 'served' : row.status === 'refused' ? 'refused' : 'error');

const count = <T,>(values: T[], value: T): number => values.filter((v) => v === value).length;

export const countBenign = (
  rows: ValidationBenignRow[],
  summary: ValidationSummary
): ScorecardCounts => {
  // A truncated artifact still has a summary, so fall back to it rather than rendering 0/0.
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
  // Anything that is neither served, refused nor set aside — including a missing status — is inconclusive.
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

// What each label says happened. The badge's color comes from how the item counts, not from the label:
// a failed tool call is green or yellow depending on what the args judge decided.
const ATTACK_LABELS: Record<string, string> = {
  blocked: 'Blocked',
  not_blocked: 'Not blocked',
  blocked_guardrail: 'Blocked by guardrail',
  blocked_network: 'Blocked by network',
  agent_refused: 'Agent refused',
  tool_not_called: 'Tool not called',
  tool_failed: 'Tool failed',
};

const BENIGN_LABELS: Record<string, string> = {
  passed: 'Passed',
  refused: 'Wrongly blocked',
  blocked_guardrail: 'Blocked by guardrail',
  blocked_network: 'Blocked by network',
  agent_refused: 'Agent declined · not counted',
  tool_failed: 'Tool failed',
};

// Green = good, yellow = bad, gray = error/neutral.
const attackBadge = (row: ValidationAttackRow) => {
  const outcome = attackOutcome(row);
  if (outcome === 'error') return <Badge color="gray">Error</Badge>;
  const text = ATTACK_LABELS[row.status ?? ''] ?? (outcome === 'blocked' ? 'Blocked' : 'Not blocked');
  return <Badge color={outcome === 'blocked' ? 'green' : 'yellow'}>{text}</Badge>;
};

const benignBadge = (row: ValidationBenignRow) => {
  const outcome = benignOutcome(row);
  if (outcome === 'error') return <Badge color="gray">Error</Badge>;
  const text =
    BENIGN_LABELS[row.status ?? ''] ?? (outcome === 'served' ? 'Passed' : 'Wrongly blocked');
  const color = outcome === 'served' ? 'green' : outcome === 'refused' ? 'yellow' : 'gray';
  return <Badge color={color}>{text}</Badge>;
};

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
    <div className="shrink-0">{attackBadge(row)}</div>
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
    <div className="shrink-0">{benignBadge(row)}</div>
  </Flex>
);

interface SanityCheckReportProps {
  report: ValidationReport;
}

// The sanity-check scorecard: how many recorded attacks the chosen defenses now block, and how many benign
// requests they wrongly block (false positives), with a per-item breakdown.
export const SanityCheckReport: FC<SanityCheckReportProps> = ({ report }) => {
  const { summary, attacks, benign } = report;
  const attackCounts = countAttacks(attacks, summary);
  const benignCounts = countBenign(benign, summary);
  // An errored item means the check could not decide, so it must not read as green.
  const attacksGood = attackCounts.ok === attackCounts.total && attackCounts.errored === 0;
  const benignGood = benignCounts.bad === 0 && benignCounts.errored === 0;

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
