// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { HelixJobTerminalStatuses } from '@nemo/common/src/constants/query';
import {
  Banner,
  Button,
  Flex,
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
  Text,
} from '@nvidia/foundations-react-core';
import { DataDesignerJobActionsMenu } from '@studio/components/DataDesignerJobActionsMenu';
import { DataDesignerTransformModal } from '@studio/components/DataDesignerTransformModal';
import { CreateFileSplitsModal } from '@studio/components/FilesTable/CreateFileSplitsModal';
import { Loading } from '@studio/components/Layouts/Loading';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { BUILDER_CONFIG_FILENAME } from '@studio/routes/DataDesignerJobDetailsRoute/builderConfig';
import { DataDesignerConfigPanel } from '@studio/routes/DataDesignerJobDetailsRoute/DataDesignerConfigPanel';
import { DatasetProfilerSection } from '@studio/routes/DataDesignerJobDetailsRoute/DatasetProfilerSection';
import { JobDatasetEditorSection } from '@studio/routes/DataDesignerJobDetailsRoute/JobDatasetEditorSection';
import { JobLogsSection } from '@studio/routes/DataDesignerJobDetailsRoute/JobLogsSection';
import { JobOutputFilesetSection } from '@studio/routes/DataDesignerJobDetailsRoute/JobOutputFilesetSection';
import { useDataDesignerArtifactsFileset } from '@studio/routes/DataDesignerJobDetailsRoute/useDataDesignerArtifactsFileset';
import { useDataDesignerJobFromRoute } from '@studio/routes/DataDesignerJobDetailsRoute/useDataDesignerJobFromRoute';
import { getDataDesignerJobListRoute } from '@studio/routes/utils';
import { formatDateTime } from '@studio/util/date';
import { ArrowLeft } from 'lucide-react';
import { useRef, useState, type FC } from 'react';
import { Link, useNavigate } from 'react-router';

type JobDetailsTab = 'profile' | 'data' | 'output' | 'logs';

export const DataDesignerJobDetailsRoute: FC = () => {
  const {
    workspace,
    jobName: dataDesignerJobName,
    job,
    isLoading,
    isError,
    refetch,
  } = useDataDesignerJobFromRoute();

  const navigate = useNavigate();
  const [isConfigPanelOpen, setIsConfigPanelOpen] = useState(false);
  const [isSplitModalOpen, setIsSplitModalOpen] = useState(false);
  const [isTransformModalOpen, setIsTransformModalOpen] = useState(false);
  const [cancelError, setCancelError] = useState<string | undefined>(undefined);
  const [selectedTab, setSelectedTab] = useState<JobDetailsTab | undefined>(undefined);

  const defaultTabRef = useRef<JobDetailsTab | undefined>(undefined);
  if (!defaultTabRef.current && job?.status) {
    defaultTabRef.current = HelixJobTerminalStatuses.includes(job.status) ? 'profile' : 'logs';
  }
  const activeTab = selectedTab ?? defaultTabRef.current ?? 'profile';

  const { filesetWorkspace, filesetName, files } = useDataDesignerArtifactsFileset();
  const splitDatasetId =
    filesetWorkspace && filesetName ? `${filesetWorkspace}/${filesetName}` : undefined;
  const splitFileOptions = files
    .map((file) => file.path)
    .filter((path) => /\.(json|jsonl|parquet)$/i.test(path));
  const canSplit = Boolean(splitDatasetId) && splitFileOptions.length > 0;

  const transformFileOptions = splitFileOptions.filter(
    (path) => !path.endsWith(BUILDER_CONFIG_FILENAME)
  );
  const canTransform = Boolean(filesetWorkspace && filesetName) && transformFileOptions.length > 0;

  useBreadcrumbs({
    items: [
      {
        href: getDataDesignerJobListRoute(workspace),
        slotLabel: 'Data Designer',
      },
      {
        slotLabel: job?.name ?? dataDesignerJobName,
      },
    ],
  });

  if (isLoading && !job) {
    return <Loading description="Loading job..." />;
  }

  if (isError || !job) {
    return (
      <AccessibleTitle title={`Data Designer Job - ${dataDesignerJobName}`}>
        <Stack className="h-full overflow-auto" gap="density-2xl" padding="density-2xl">
          <ErrorMessage
            header="Failed to load job"
            message="The job could not be loaded. It may have been deleted or you may not have access."
            slotFooter={
              <Button type="button" kind="tertiary" onClick={() => refetch()}>
                Retry
              </Button>
            }
          />
          <Button asChild kind="secondary">
            <Link to={getDataDesignerJobListRoute(workspace)}>
              <ArrowLeft /> Back to Data Designer
            </Link>
          </Button>
        </Stack>
      </AccessibleTitle>
    );
  }

  return (
    <AccessibleTitle title={`Data Designer Job - ${job.name}`}>
      <Stack className="h-full min-h-0" gap="density-2xl" padding="density-2xl">
        <Stack gap="density-md">
          <Flex gap="density-md" align="center" justify="between" className="flex-wrap">
            <Flex gap="density-md" align="center" className="flex-wrap">
              <Text kind="body/bold/2xl">{job.name}</Text>
              {job.status ? <StatusBadge status={job.status} /> : null}
            </Flex>
            <Flex gap="density-md" align="center">
              <DataDesignerJobActionsMenu
                job={job}
                variant="labeled"
                additionalActions={[
                  {
                    label: 'Transform',
                    disabled: !canTransform,
                    onSelect: () => setIsTransformModalOpen(true),
                  },
                  {
                    label: 'Split',
                    disabled: !canSplit,
                    onSelect: () => setIsSplitModalOpen(true),
                    divider: {},
                  },
                ]}
                onViewConfig={() => setIsConfigPanelOpen(true)}
                onDeleted={() => navigate(getDataDesignerJobListRoute(workspace))}
                onCancelError={setCancelError}
              />
            </Flex>
          </Flex>
          {cancelError && (
            <Banner kind="inline" status="error">
              {cancelError}
            </Banner>
          )}
          {job.description && (
            <Text kind="body/regular/md" className="text-muted">
              {job.description}
            </Text>
          )}
          <Flex gap="density-lg" className="flex-wrap">
            {job.created_at && (
              <Text kind="body/regular/sm" className="text-muted">
                Created: {formatDateTime(job.created_at)}
              </Text>
            )}
            {job.updated_at && (
              <Text kind="body/regular/sm" className="text-muted">
                Updated: {formatDateTime(job.updated_at)}
              </Text>
            )}
          </Flex>
        </Stack>

        <TabsRoot
          value={activeTab}
          onValueChange={(value) => setSelectedTab(value as JobDetailsTab)}
          className="flex min-h-0 w-full min-w-0 flex-1 flex-col"
        >
          <TabsList>
            <TabsTrigger value="profile">Profile</TabsTrigger>
            <TabsTrigger value="data">Data</TabsTrigger>
            <TabsTrigger value="output">Output files</TabsTrigger>
            <TabsTrigger value="logs">Logs</TabsTrigger>
          </TabsList>

          <TabsContent value="profile" className="min-h-0 flex-1 overflow-y-auto px-0">
            <DatasetProfilerSection />
          </TabsContent>

          <TabsContent value="data" className="min-h-0 min-w-0 flex-1 overflow-y-auto px-0">
            <JobDatasetEditorSection />
          </TabsContent>

          <TabsContent value="output" className="min-h-0 flex-1 overflow-y-auto px-0">
            <JobOutputFilesetSection />
          </TabsContent>

          <TabsContent value="logs" className="min-h-0 min-w-0 flex-1 overflow-y-auto px-0">
            <JobLogsSection />
          </TabsContent>
        </TabsRoot>
      </Stack>

      <DataDesignerConfigPanel
        open={isConfigPanelOpen}
        onClose={() => setIsConfigPanelOpen(false)}
      />

      {isTransformModalOpen && (
        <DataDesignerTransformModal
          open
          onClose={() => setIsTransformModalOpen(false)}
          workspace={workspace}
          sourceJobName={job.name}
          filesetWorkspace={filesetWorkspace}
          filesetName={filesetName}
          fileOptions={transformFileOptions}
          defaultNumRecords={job.spec?.job_config?.num_records ?? 0}
        />
      )}

      {isSplitModalOpen && (
        <CreateFileSplitsModal
          open
          onClose={() => setIsSplitModalOpen(false)}
          datasetId={splitDatasetId}
          fileOptions={splitFileOptions}
        />
      )}
    </AccessibleTitle>
  );
};
