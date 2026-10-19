// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AccessibleTitle } from '@nemo/common/src/components/AccessibleTitle';
import {
  useSafeSynthesizerDownloadJobResultSummary as useDownloadJobResultSummaryV1beta1GenerateJobsJobIdResultsSummaryDownloadGet,
  useSafeSynthesizerGetJobSuspense as useGetJobV1beta1GenerateJobsJobIdGetSuspense,
} from '@nemo/sdk/generated/safe-synthesizer/safe-synthesizer';
import { Stack } from '@nvidia/foundations-react-core';
import { SafeSynthesizerNavigation } from '@studio/components/SafeSynthesizerNavigation';
import { SAFE_SYNTHESIZER_ENABLED } from '@studio/constants/environment';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useBreadcrumbs } from '@studio/providers/breadcrumbs/useBreadcrumbs';
import { OverviewPanel } from '@studio/routes/SafeSynthesizerJobReportRoute/components/OverviewPanel';
import { ReportMenu } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ReportMenu';
import { DataPrivacyPanel } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/DataPrivacyPanel';
import { SyntheticQualityPanel } from '@studio/routes/SafeSynthesizerJobReportRoute/components/ScorePanels/SyntheticQualityPanel';
import { getSafeSynthesizerRoute } from '@studio/routes/utils';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import { isJobSuccessful } from '@studio/util/safeSynthesizer';
import { BadgeCheck, File, Lock } from 'lucide-react';
import { FC, useRef } from 'react';

export type ReportSection = 'overview' | 'synthetic-quality' | 'data-privacy';
export interface MenuItem {
  id: ReportSection;
  label: string;
  icon: React.ReactNode;
}

export const GenerateJobReportRoute: FC | null = SAFE_SYNTHESIZER_ENABLED
  ? () => {
      const workspace = useWorkspaceFromPath();
      const { safeSynthesizerJobName } = useRequiredPathParams([
        ROUTE_PARAMS.safeSynthesizerJobName,
      ]);

      useBreadcrumbs({
        items: [
          {
            href: getSafeSynthesizerRoute(workspace),
            slotLabel: 'Safe Synthesizer',
          },
          {
            slotLabel: 'Job Report',
          },
        ],
      });

      // Refs for each section
      const overviewRef = useRef<HTMLDivElement>(null);
      const syntheticQualityRef = useRef<HTMLDivElement>(null);
      const dataPrivacyRef = useRef<HTMLDivElement>(null);

      const { data: job } = useGetJobV1beta1GenerateJobsJobIdGetSuspense(
        workspace,
        safeSynthesizerJobName
      );

      const isSuccessful = isJobSuccessful(job.status);

      const { data: reportSummary } =
        useDownloadJobResultSummaryV1beta1GenerateJobsJobIdResultsSummaryDownloadGet(
          workspace,
          safeSynthesizerJobName,
          {
            query: {
              enabled: isSuccessful,
            },
          }
        );

      const dpEnabled = job?.spec?.config?.privacy?.dp_enabled ?? false;

      const handleSectionChange = (section: ReportSection) => {
        const refMap = {
          overview: overviewRef,
          'synthetic-quality': syntheticQualityRef,
          'data-privacy': dataPrivacyRef,
        };

        const targetRef = refMap[section];
        if (targetRef.current) {
          targetRef.current.scrollIntoView({
            behavior: 'smooth',
            block: 'start',
          });
        }
      };

      const menuItems: MenuItem[] = [
        {
          id: 'overview',
          label: 'Overview',
          icon: <File />,
        },
        {
          id: 'synthetic-quality',
          label: 'Synthetic Quality',
          icon: <BadgeCheck />,
        },
        {
          id: 'data-privacy',
          label: 'Data Privacy',
          icon: <Lock />,
        },
      ];

      return (
        <AccessibleTitle title="Safe Synthesizer Job Report">
          <Stack className="h-full w-full overflow-auto" gap="density-2xl" padding="density-2xl">
            <SafeSynthesizerNavigation selected="report" jobName={safeSynthesizerJobName} />
            <Stack gap="density-2xl">
              <ReportMenu items={menuItems} onSectionChange={handleSectionChange} />
              <Stack gap="density-2xl" className="w-full">
                {menuItems.map((item) => {
                  switch (item.id) {
                    case 'overview':
                      return (
                        <div key={item.id} ref={overviewRef}>
                          <OverviewPanel
                            jobId={safeSynthesizerJobName}
                            title={item.label}
                            icon={item.icon}
                          />
                        </div>
                      );
                    case 'synthetic-quality':
                      return (
                        <div key={item.id} ref={syntheticQualityRef}>
                          <SyntheticQualityPanel
                            reportSummary={reportSummary}
                            title={item.label}
                            icon={item.icon}
                          />
                        </div>
                      );
                    case 'data-privacy':
                      return (
                        <div key={item.id} ref={dataPrivacyRef}>
                          <DataPrivacyPanel
                            reportSummary={reportSummary}
                            dpEnabled={dpEnabled}
                            title={item.label}
                            icon={item.icon}
                          />
                        </div>
                      );
                  }
                  return null;
                })}
              </Stack>
            </Stack>
          </Stack>
        </AccessibleTitle>
      );
    }
  : null;
