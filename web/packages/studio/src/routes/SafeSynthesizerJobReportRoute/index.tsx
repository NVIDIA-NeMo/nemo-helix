// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import {
  useSafeSynthesizerDownloadJobResultSummary,
  useSafeSynthesizerGetJobSuspense,
} from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import {
  Flex,
  Spinner,
  Stack,
  TabsContent,
  TabsList,
  TabsRoot,
  TabsTrigger,
} from '@nvidia/foundations-react-core';
import { SafeSynthesizerNavigation } from '@studio/components/SafeSynthesizerNavigation';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { QUERY_PARAMETERS } from '@studio/routes/constants';
import { DownloadReportButton } from '@studio/routes/SafeSynthesizerJobReportRoute/components/DownloadReportButton';
import { OverviewTab } from '@studio/routes/SafeSynthesizerJobReportRoute/components/OverviewTab';
import { DataPrivacyMetrics } from '@studio/routes/SafeSynthesizerJobReportRoute/components/PrivacyMetrics';
import { SyntheticQualityMetrics } from '@studio/routes/SafeSynthesizerJobReportRoute/components/QualityMetrics';
import { DataPrivacyPanel } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/DataPrivacyPanel';
import { SyntheticQualityPanel } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/SyntheticQualityPanel';
import { TrainingColumnsTab } from '@studio/routes/SafeSynthesizerJobReportRoute/components/TrainingColumnsTab';
import {
  isScoreAvailable,
  REPORT_TABS,
  type ReportTab,
  visibleReportTabs,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportTabs';
import { useEvaluationReportData } from '@studio/routes/SafeSynthesizerJobReportRoute/useEvaluationReportData';
import { getSafeSynthesizerRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { isJobSuccessful } from '@studio/util/safeSynthesizer';
import type { FC } from 'react';
import { useSearchParams } from 'react-router';

const SyntheticQualityIcon = REPORT_TABS['synthetic-quality'].Icon;
const DataPrivacyIcon = REPORT_TABS['data-privacy'].Icon;

export const GenerateJobReportRoute: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { safeSynthesizerJobName } = useRequiredPathParams([ROUTE_PARAMS.safeSynthesizerJobName]);
  const [searchParams, setSearchParams] = useSearchParams();

  useBreadcrumbs({
    items: [
      { href: getSafeSynthesizerRoute(workspace), slotLabel: 'Safe Synthesizer' },
      { slotLabel: 'Job Report' },
    ],
  });

  const { data: job } = useSafeSynthesizerGetJobSuspense(workspace, safeSynthesizerJobName);
  const isSuccessful = isJobSuccessful(job.status);
  const { data: reportSummary } = useSafeSynthesizerDownloadJobResultSummary(
    workspace,
    safeSynthesizerJobName,
    { query: { enabled: isSuccessful } }
  );
  const { data: reportDataResult, isLoading: isReportDataLoading } = useEvaluationReportData(
    workspace,
    safeSynthesizerJobName,
    isSuccessful
  );
  const reportData = reportDataResult ?? null;

  const tabs = visibleReportTabs(reportData);
  const requestedTab = searchParams.get(QUERY_PARAMETERS.tab);
  const activeTab = tabs.find((tab) => tab === requestedTab) ?? 'overview';
  const selectTab = (tab: ReportTab) => {
    setSearchParams(
      (previous) => {
        previous.set(QUERY_PARAMETERS.tab, tab);
        return previous;
      },
      { replace: true }
    );
  };

  const sqsScore =
    reportData?.scores.synthetic_quality?.score ?? reportSummary?.synthetic_data_quality_score;
  const dpsScore = reportData?.scores.data_privacy?.score ?? reportSummary?.data_privacy_score;
  const dpEnabled =
    reportData?.differential_privacy.enabled ?? job.spec?.config?.privacy?.dp_enabled ?? false;
  const withPrivacyScore = reportData === null || isScoreAvailable(reportData.scores.data_privacy);

  return (
    <AccessibleTitle title="Safe Synthesizer Job Report">
      <Stack className="h-full w-full overflow-auto" gap="density-2xl" padding="density-2xl">
        <SafeSynthesizerNavigation selected="report" jobName={safeSynthesizerJobName} />
        <TabsRoot value={activeTab} onValueChange={(value) => selectTab(value as ReportTab)}>
          <Flex justify="between" align="end" gap="density-lg">
            <TabsList className="min-w-0 flex-1">
              {tabs.map((tab) => {
                const { Icon, label } = REPORT_TABS[tab];
                return (
                  <TabsTrigger key={tab} value={tab}>
                    <Flex align="center" gap="density-sm">
                      <Icon />
                      {label}
                    </Flex>
                  </TabsTrigger>
                );
              })}
            </TabsList>
            <DownloadReportButton
              workspace={workspace}
              jobName={safeSynthesizerJobName}
              enabled={isSuccessful}
            />
          </Flex>
          {isReportDataLoading ? (
            <Flex justify="center" className="py-density-2xl">
              <Spinner size="medium" aria-label="Loading report" />
            </Flex>
          ) : (
            <>
              <TabsContent value="overview" className="px-0 pt-density-xl">
                <OverviewTab
                  jobId={reportData?.job_id ?? null}
                  sqsScore={sqsScore}
                  dpsScore={dpsScore}
                  showDps={withPrivacyScore}
                  dataset={reportData?.dataset ?? null}
                  onViewQuality={
                    tabs.includes('synthetic-quality')
                      ? () => selectTab('synthetic-quality')
                      : undefined
                  }
                  onViewPrivacy={
                    tabs.includes('data-privacy') ? () => selectTab('data-privacy') : undefined
                  }
                />
              </TabsContent>
              <TabsContent value="synthetic-quality" className="px-0 pt-density-xl">
                <SyntheticQualityPanel
                  reportSummary={reportSummary}
                  score={sqsScore}
                  title={REPORT_TABS['synthetic-quality'].label}
                  icon={<SyntheticQualityIcon />}
                >
                  {reportData ? <SyntheticQualityMetrics data={reportData} /> : null}
                </SyntheticQualityPanel>
              </TabsContent>
              <TabsContent value="data-privacy" className="px-0 pt-density-xl">
                {withPrivacyScore ? (
                  <DataPrivacyPanel
                    reportSummary={reportSummary}
                    score={dpsScore}
                    dpEnabled={dpEnabled}
                    epsilon={reportData?.differential_privacy.epsilon}
                    delta={reportData?.differential_privacy.delta}
                    title={REPORT_TABS['data-privacy'].label}
                    icon={<DataPrivacyIcon />}
                  >
                    {reportData ? <DataPrivacyMetrics data={reportData} withPrivacyScore /> : null}
                  </DataPrivacyPanel>
                ) : null}
                {reportData && !withPrivacyScore ? (
                  <DataPrivacyMetrics data={reportData} withPrivacyScore={false} />
                ) : null}
              </TabsContent>
              {reportData ? (
                <TabsContent value="training-data" className="px-0 pt-density-xl">
                  <TrainingColumnsTab
                    columns={reportData.columns}
                    withDistribution={isScoreAvailable(reportData.scores.synthetic_quality)}
                    withTransform={reportData.any_column_transformed}
                  />
                </TabsContent>
              ) : null}
            </>
          )}
        </TabsRoot>
      </Stack>
    </AccessibleTitle>
  );
};
