// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
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
 * `ok` is the good outcome (attack blocked / benign still complied), `bad` the actionable
 * failure, and `errored` the items that produced neither — the run could not decide. Those are
 * counted apart on purpose: folding them into `ok` overstates how well the defenses held, which
 * for a security tool is the wrong direction to be wrong in.
 */
export interface ScorecardCounts {
  ok: number;
  bad: number;
  errored: number;
  total: number;
}

const tally = (statuses: (string | undefined)[], okStatus: string, badStatus: string): number[] => [
  statuses.filter((s) => s === okStatus).length,
  statuses.filter((s) => s === badStatus).length,
];

export const countBenign = (
  rows: ValidationBenignRow[],
  summary: ValidationSummary
): ScorecardCounts => {
  // A truncated artifact still has a summary, so fall back to it rather than rendering 0/0.
  if (rows.length === 0) {
    const total = summary.benign_total ?? 0;
    const bad = summary.benign_false_positives ?? 0;
    return { ok: Math.max(total - bad, 0), bad, errored: 0, total };
  }
  const statuses = rows.map((row) => row.status);
  const [ok, bad] = tally(statuses, 'passed', 'refused');
  // Anything that is neither passed nor refused — including a missing status — is inconclusive.
  return { ok, bad, errored: rows.length - ok - bad, total: rows.length };
};

export const countAttacks = (
  rows: ValidationAttackRow[],
  summary: ValidationSummary
): ScorecardCounts => {
  if (rows.length === 0) {
    const total = summary.attacks_total ?? 0;
    const ok = summary.attacks_blocked ?? 0;
    return { ok, bad: Math.max(total - ok, 0), errored: 0, total };
  }
  const statuses = rows.map((row) => row.status);
  const [ok, bad] = tally(statuses, 'blocked', 'not_blocked');
  return { ok, bad, errored: rows.length - ok - bad, total: rows.length };
};

const plural = (count: number, noun: string): string =>
  `${count} ${noun}${count === 1 ? '' : 's'}`;

/** `7 / 9 · 1 false positive · 1 error` — trailing segments appear only when non-zero. */
export const formatCounts = (counts: ScorecardCounts, badNoun: string): string =>
  [
    `${counts.ok} / ${counts.total}`,
    counts.bad > 0 ? plural(counts.bad, badNoun) : '',
    counts.errored > 0 ? plural(counts.errored, 'error') : '',
  ]
    .filter(Boolean)
    .join(' · ');

// Maps a per-item status to a badge (green = good, yellow = bad, gray = error/neutral).
const attackBadge = (status?: string) =>
  status === 'blocked' ? (
    <Badge color="green">Blocked</Badge>
  ) : status === 'not_blocked' ? (
    <Badge color="yellow">Not blocked</Badge>
  ) : (
    <Badge color="gray">Error</Badge>
  );

const benignBadge = (status?: string) =>
  status === 'passed' ? (
    <Badge color="green">Passed</Badge>
  ) : status === 'refused' ? (
    <Badge color="yellow">Wrongly blocked</Badge>
  ) : (
    <Badge color="gray">Error</Badge>
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
    <div className="shrink-0">{attackBadge(row.status)}</div>
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
    <div className="shrink-0">{benignBadge(row.status)}</div>
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
