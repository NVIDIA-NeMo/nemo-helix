// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Stack, Text } from '@nvidia/foundations-react-core';
import type { InsightsTriggerResult } from '@studio/api/insightsAnalysis';
import type { ImportTraceResult } from '@studio/components/ImportTracesModal/types';
import { CircleCheck, CircleX, Info } from 'lucide-react';
import type { FC, ReactNode } from 'react';

export interface ImportTracesResultListProps {
  results: ImportTraceResult[];
  insightsResults?: InsightsTriggerResult[] | null;
}

interface RowProps {
  slotIcon: ReactNode;
  title: string;
  message?: string;
}

const ResultRow: FC<RowProps> = ({ slotIcon, title, message }) => (
  <Flex gap="density-md" className="items-start">
    {slotIcon}
    <Stack gap="density-xs" className="min-w-0">
      <Text kind="body/regular/sm" className="break-all">
        {title}
      </Text>
      {message && (
        <Text kind="body/regular/xs" color="secondary" className="break-words">
          {message}
        </Text>
      )}
    </Stack>
  </Flex>
);

const successIcon = <CircleCheck className="size-4 shrink-0 text-status-success" aria-hidden />;
const errorIcon = <CircleX className="size-4 shrink-0 text-status-error" aria-hidden />;
const infoIcon = <Info className="size-4 shrink-0 text-status-info" aria-hidden />;

const insightsIcon = (status: InsightsTriggerResult['status']) => {
  if (status === 'started') return successIcon;
  return status === 'not-enabled' ? infoIcon : errorIcon;
};

const insightsTitle = ({ agent, status, jobName }: InsightsTriggerResult) => {
  if (status === 'started') return `${agent} — analysis queued${jobName ? ` (${jobName})` : ''}`;
  return status === 'not-enabled'
    ? `${agent} — analysis not enabled`
    : `${agent} — analysis failed`;
};

export const ImportTracesResultList: FC<ImportTracesResultListProps> = ({
  results,
  insightsResults,
}) => (
  <Stack gap="density-lg" role="status" aria-label="Import results">
    <Stack gap="density-md">
      <Text kind="body/semibold/sm">Results</Text>
      {results.map(({ label, status, message }) => (
        <ResultRow
          key={`${label}-${status}`}
          slotIcon={status === 'success' ? successIcon : errorIcon}
          title={label}
          message={message}
        />
      ))}
    </Stack>

    {insightsResults && insightsResults.length > 0 && (
      <>
        <Divider />
        <Stack gap="density-md">
          <Text kind="body/semibold/sm">Insights</Text>
          {insightsResults.map((result) => (
            <ResultRow
              key={`${result.agent}-${result.status}`}
              slotIcon={insightsIcon(result.status)}
              title={insightsTitle(result)}
              message={result.message}
            />
          ))}
        </Stack>
      </>
    )}
  </Stack>
);
