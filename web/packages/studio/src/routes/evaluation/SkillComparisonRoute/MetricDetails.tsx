// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { KeyValueGrid } from '@nemo/common/src/components/KeyValueGrid';
import { Badge, Flex, Stack, Table, Text } from '@nvidia/foundations-react-core';
import { metricLabel } from '@studio/routes/evaluation/SkillComparisonRoute/presentation';
import type { MetricResult } from '@studio/routes/evaluation/SkillComparisonRoute/report';

const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });

function displayValue(value: unknown) {
  if (value == null) return 'Unmeasured';
  if (typeof value === 'number') return number.format(value);
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'string') return value;
  return (
    <pre className="whitespace-pre-wrap wrap-anywhere text-sm">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export function MetricDetails({ name, result }: { name: string; result: MetricResult }) {
  return (
    <Stack gap="density-md">
      {result.outputs.length > 0 ? (
        <Table
          aria-label={`${metricLabel(name)} outputs`}
          columns={['Output', 'Value']}
          rows={result.outputs.map((output) => ({
            id: output.name,
            cells: [metricLabel(output.name), { children: displayValue(output.value) }],
          }))}
        />
      ) : (
        <Text>No outputs were measured.</Text>
      )}
      {result.diagnostics.map((diagnostic, index) => (
        <Stack key={index} gap="density-sm" className="rounded border border-base p-3">
          <Flex align="center" gap="density-sm">
            <Text>{diagnostic.message}</Text>
            {diagnostic.severity && <Badge color="gray">{diagnostic.severity}</Badge>}
          </Flex>
          <KeyValueGrid
            items={Object.entries(diagnostic.details ?? {}).map(([key, value]) => ({
              key,
              label: metricLabel(key),
              value: displayValue(value),
              wrapValue: true,
            }))}
          />
        </Stack>
      ))}
    </Stack>
  );
}
