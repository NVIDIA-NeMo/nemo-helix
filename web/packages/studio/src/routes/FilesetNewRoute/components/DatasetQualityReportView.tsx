// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { DatasetQualityReport } from '@nemo/common/src/utils/datasetQuality';
import { Flex, Stack, Text } from '@nvidia/foundations-react-core';
import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';
import type { FC } from 'react';

interface DatasetQualityReportViewProps {
  report: DatasetQualityReport;
}

export const DatasetQualityReportView: FC<DatasetQualityReportViewProps> = ({ report }) => {
  const partialScanNote = report.scannedLines < report.totalLines && (
    <Text className="text-secondary" kind="body/regular/sm">
      Scanned first {report.scannedLines.toLocaleString()} of {report.totalLines.toLocaleString()}{' '}
      lines.
    </Text>
  );

  if (!report.hasErrors && !report.hasWarnings) {
    return (
      <Stack gap="density-xs">
        <Flex gap="density-sm" align="center">
          <CheckCircle2 size={16} className="text-green-500 shrink-0" />
          <Text kind="body/regular/sm">{report.fileName}: all quality checks passed.</Text>
        </Flex>
        {partialScanNote}
      </Stack>
    );
  }

  return (
    <Stack gap="density-sm">
      <Text kind="label/bold/sm">{report.fileName}</Text>
      {report.issues.map((issue, idx) => (
        <Flex key={idx} gap="density-sm" align="start">
          {issue.severity === 'error' ? (
            <XCircle size={16} className="text-red-500 shrink-0 mt-0.5" />
          ) : (
            <AlertTriangle size={16} className="text-amber-500 shrink-0 mt-0.5" />
          )}
          <Stack gap="density-xs">
            <Text kind="body/regular/sm">{issue.message}</Text>
            {issue.affectedLines && issue.affectedLines.length > 0 && (
              <Text className="text-secondary" kind="body/regular/sm">
                {'Line' + (issue.affectedLines.length > 1 ? 's' : '') + ': '}
                {issue.affectedLines.join(', ')}
                {issue.count && issue.count > issue.affectedLines.length
                  ? ` (+${issue.count - issue.affectedLines.length} more)`
                  : ''}
              </Text>
            )}
          </Stack>
        </Flex>
      ))}
      {partialScanNote}
    </Stack>
  );
};
