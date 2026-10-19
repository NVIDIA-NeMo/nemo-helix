// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { parseFilesetUrl } from '@nemo/common/src/components/DatasetFileSelect/utils';
import { KVPair } from '@nemo/common/src/components/KVPair';
import { ScrollTable } from '@nemo/common/src/components/ScrollTable';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import {
  useSafeSynthesizerDownloadJobResultSyntheticData as useDownloadJobResultSyntheticDataV1beta1GenerateJobsJobIdResultsSyntheticDataDownloadGet,
  useSafeSynthesizerListJobResults as useListJobResultsV1beta1GenerateJobsJobIdResultsGet,
} from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import type { GenerateJob } from '@nemo/sdk/generated/safe-synthesizer/schema';
import {
  Anchor,
  CodeSnippet,
  Stack,
  Text,
  type TableColumnDefinition,
  type TableRowDefinition,
} from '@nvidia/foundations-react-core';
import { FilePreview } from '@studio/components/SafeSynthesizerFilesetPreview/FilePreview';
import { FilesetFilePreviewLink } from '@studio/components/SafeSynthesizerFilesetPreview/FilesetFilePreviewLink';
import { parseFileContent } from '@studio/components/SafeSynthesizerFilesetPreview/util';
import { EMPTY_FIELD_VALUE } from '@studio/constants/constants';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { JobConfigDrawer } from '@studio/routes/SafeSynthesizerJobDetailsRoute/components/JobConfigDrawer';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { isJobSuccessful } from '@studio/util/safeSynthesizer';
import { FC, useCallback, useState } from 'react';

interface SafeSynthesizerFilesetPreviewProps {
  job: GenerateJob;
  showJobId?: boolean;
}

export const SafeSynthesizerFilesetPreview: FC<SafeSynthesizerFilesetPreviewProps> = ({
  job,
  showJobId = true,
}) => {
  const toast = useToast();
  const workspace = useWorkspaceFromPath();
  const [showJobConfig, setShowJobConfig] = useState(false);
  const { safeSynthesizerJobName } = useRequiredPathParams([ROUTE_PARAMS.safeSynthesizerJobName]);

  const isSuccessful = isJobSuccessful(job.status);
  const { data: jobResultsList } = useListJobResultsV1beta1GenerateJobsJobIdResultsGet(
    workspace,
    safeSynthesizerJobName,
    {
      query: {
        enabled: isSuccessful,
      },
    }
  );

  const fileInfo = parseFilesetUrl(job.spec.data_source);

  const { refetch: refetchSyntheticData } =
    useDownloadJobResultSyntheticDataV1beta1GenerateJobsJobIdResultsSyntheticDataDownloadGet(
      workspace,
      safeSynthesizerJobName,
      {
        query: {
          enabled: false,
        },
      }
    );

  const [, , syntheticData] = jobResultsList?.data ?? [];

  const pathUrl = syntheticData ? parseFilesetUrl(syntheticData.artifact_url) : null;

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);
  const [showPreviewPanel, setShowPreviewPanel] = useState(false);
  const [previewTitle, setPreviewTitle] = useState<string>('');
  const [jsonDataPreview, setJsonDataPreview] = useState<string | undefined>(undefined);
  const [tabularDataPreview, setTabularDataPreview] = useState<
    | {
        rows: TableRowDefinition[];
        columns: TableColumnDefinition[];
      }
    | undefined
  >(undefined);

  const handlePreviewFile = useCallback((url: string, content: string) => {
    const parsed = parseFileContent(url, content);

    if (parsed.type === 'csv' && parsed.tabularData) {
      setTabularDataPreview(parsed.tabularData);
    } else if (parsed.type === 'json' && parsed.jsonData) {
      setJsonDataPreview(parsed.jsonData);
    } else if (parsed.type === 'error') {
      setError(parsed.error);
    }
  }, []);

  const handlePreviewSyntheticData = async () => {
    setIsLoading(true);
    setShowPreviewPanel(true);
    setError(undefined);
    setJsonDataPreview(undefined);
    setTabularDataPreview(undefined);

    if (pathUrl) {
      setPreviewTitle(`${pathUrl.workspace}/${pathUrl.name}/${pathUrl.path}`);
    }
    try {
      const response = await refetchSyntheticData();
      if (!response.data || !pathUrl) {
        setError('Error fetching synthetic data');
        return;
      }
      handlePreviewFile(`${pathUrl.path}.csv`, await response.data.text());
    } catch {
      setError('Could not parse synthetic data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClosePanel = () => {
    setIsLoading(false);
    setError(undefined);
    setShowPreviewPanel(false);
    setJsonDataPreview(undefined);
    setTabularDataPreview(undefined);
  };

  const handleDownloadSyntheticData = async () => {
    try {
      const response = await refetchSyntheticData();
      if (response.data) {
        return response.data;
      }
      return null;
    } catch {
      return null;
    }
  };

  return (
    <>
      <JobConfigDrawer job={job} open={showJobConfig} onOpenChange={setShowJobConfig} />
      {showPreviewPanel && (
        <FilePreview
          error={error}
          isLoading={isLoading}
          onClose={handleClosePanel}
          title={previewTitle}
          onDownload={handleDownloadSyntheticData}
          downloadFileName={pathUrl ? `${pathUrl.path}.csv` : 'synthetic_data.csv'}
        >
          {tabularDataPreview && (
            <ScrollTable
              allowHorizontalScroll
              columns={tabularDataPreview.columns}
              rows={tabularDataPreview.rows}
            />
          )}
          {jsonDataPreview && (
            <CodeSnippet
              value={jsonDataPreview}
              language="json"
              kind="block"
              attributes={{ CodeSnippetCode: { className: '[&&]:min-h-auto' } }}
              onCopySuccess={() => toast.success('Copied to clipboard')}
            />
          )}
        </FilePreview>
      )}
      <Stack gap="density-lg">
        {showJobId && (
          <KVPair
            value={
              safeSynthesizerJobName ? (
                <Text kind="label/semibold/md" className="align-left">
                  {safeSynthesizerJobName}
                </Text>
              ) : (
                EMPTY_FIELD_VALUE
              )
            }
            label="Job ID"
          />
        )}
        <KVPair
          value={
            <FilesetFilePreviewLink url={job.spec.data_source}>
              <Text kind="body/semibold/md" className="text-wrap break-words break-all">
                {fileInfo?.workspace}/{fileInfo?.name}/{fileInfo?.path}
              </Text>
            </FilesetFilePreviewLink>
          }
          label="Data Source"
        />
        <KVPair
          value={
            pathUrl ? (
              <Anchor
                onClick={handlePreviewSyntheticData}
                className="truncate max-w-full cursor-pointer"
              >
                <Text kind="body/semibold/md" className="text-wrap break-words break-all">
                  {pathUrl.path}
                </Text>
              </Anchor>
            ) : (
              EMPTY_FIELD_VALUE
            )
          }
          label="Generation Results"
        />
      </Stack>
    </>
  );
};
