// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Flex, Stack, Text, TextInput } from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/components/DetailPanel';
import { MetricDetails } from '@studio/routes/evaluation/SkillComparisonRoute/MetricDetails';
import { metricLabel } from '@studio/routes/evaluation/SkillComparisonRoute/presentation';
import type { ComparisonReport } from '@studio/routes/evaluation/SkillComparisonRoute/report';
import { useMemo, useState } from 'react';

export function PackageChecks({ checks }: { checks: ComparisonReport['package_metrics'] }) {
  const [query, setQuery] = useState('');
  const indexed = useMemo(
    () =>
      Object.entries(checks ?? {}).map(([name, result]) => ({
        name,
        result,
        title: metricLabel(name),
        search: [
          name,
          metricLabel(name),
          ...result.outputs.map((output) => metricLabel(output.name)),
          JSON.stringify(result.diagnostics),
        ]
          .join(' ')
          .toLowerCase(),
      })),
    [checks]
  );
  const words = query.trim().toLowerCase().split(/\s+/);
  const matches = indexed.filter(({ search }) => words.every((word) => search.includes(word)));

  return (
    <Stack gap="density-sm" className="pt-4 min-w-0 w-full">
      {indexed.length === 0 ? (
        <Text>No package checks were retained.</Text>
      ) : (
        <>
          <Flex align="center" gap="density-md" className="flex-wrap">
            <TextInput
              aria-label="Search package checks"
              type="search"
              placeholder="Search package checks…"
              value={query}
              onChange={(event) => setQuery(event.currentTarget.value)}
              className="w-full max-w-sm"
            />
            <Text kind="body/regular/sm" className="text-secondary" role="status">
              {matches.length} of {indexed.length} checks
            </Text>
          </Flex>
          {matches.length === 0 ? (
            <Text>No package checks match your search.</Text>
          ) : (
            matches.map(({ name, result, title }) => (
              <DetailPanel key={name} title={title} defaultCollapsed>
                <MetricDetails name={name} result={result} />
              </DetailPanel>
            ))
          )}
        </>
      )}
    </Stack>
  );
}
