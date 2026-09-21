// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CJobTerminalStatuses } from '@nemo/common/src/constants/query';
import { useLiveSeconds } from '@nemo/common/src/hooks/useLiveSeconds';
import { Stack } from '@nvidia/foundations-react-core';
import { GrpoRewardPanel } from '@studio/components/CustomizationOverview/GrpoRewardPanel';
import { GrpoTrainingHealthPanel } from '@studio/components/CustomizationOverview/GrpoTrainingHealthPanel';
import { RunConfigurationPanel } from '@studio/components/CustomizationOverview/RunConfigurationPanel';
import { TrainingHealthPanel } from '@studio/components/CustomizationOverview/TrainingHealthPanel';
import { TrainingLossPanel } from '@studio/components/CustomizationOverview/TrainingLossPanel';
import { ErrorMessageWithRetry } from '@studio/components/ErrorMessageWithRetry';
import { Loading } from '@studio/components/Layouts/Loading';
import { CustomizationConfigSidePanel } from '@studio/components/sidePanels/CustomizationConfigSidePanel';
import { useCustomizationJob } from '@studio/hooks/useCustomizationJob';
import { useCustomizationJobStatus } from '@studio/hooks/useCustomizationJobStatus';
import { hasMetrics } from '@studio/types/customization';
import { isGrpoJob } from '@studio/util/customizationBackend';
import { resolveCustomizationFailure } from '@studio/util/customizationFailure';
import {
  getGrpoProgressTiles,
  getGrpoSummaryTiles,
  getJobDuration,
  getJobStartDate,
  getLossTiles,
  getTrainingProgressTiles,
  getTrainingDiagnosticsTiles,
  getTrainingTelemetry,
} from '@studio/util/customizations';
import { buildRewardChartData } from '@studio/util/grpoMetrics';
import { type FC, useState } from 'react';

type Props = {
  customizationJobName: string;
  workspace?: string;
};

export const CustomizationOverview: FC<Props> = ({ customizationJobName, workspace = '' }) => {
  const [openConfigSidePanel, setOpenConfigSidePanel] = useState(false);
  const {
    job: customization,
    isLoading: isLoadingCustomization,
    isError,
    refetch,
    backend,
  } = useCustomizationJob(workspace, customizationJobName);

  const { steps } = useCustomizationJobStatus(
    workspace,
    customizationJobName,
    backend,
    customization?.status
  );

  const isTerminalStatus = Boolean(
    customization?.status && CJobTerminalStatuses.includes(customization.status)
  );

  const statusDetails = customization?.status_details;
  const telemetry = getTrainingTelemetry(customization);

  const liveSeconds = useLiveSeconds({
    startDate: isTerminalStatus ? undefined : getJobStartDate(steps),
  });

  // The backend reports max_steps once at training start, and it is the same number the
  // Steps Completed tile shows. Estimating it from record counts instead meant downloading
  // every dataset file to scale one axis, and the count was capped at a preview of the
  // file, so the estimate came out low on any dataset past that cap.
  const maxXAxisValue = telemetry.maxSteps ?? 0;
  const isLoading = isLoadingCustomization;

  if (isLoading) {
    return <Loading />;
  }

  if (isError) {
    return <ErrorMessageWithRetry onRetry={refetch} />;
  }

  if (!customization) {
    return null;
  }

  const metrics = hasMetrics(statusDetails) ? statusDetails : undefined;
  const runState = {
    isTerminal: isTerminalStatus,
    duration: getJobDuration(steps, isTerminalStatus, liveSeconds),
    failedAtStepLabel: resolveCustomizationFailure(customization, steps)?.failingStepLabel,
  };

  return (
    <Stack gap="density-xl">
      {isGrpoJob(customization) ? (
        <>
          <GrpoRewardPanel
            reward={buildRewardChartData(metrics)}
            metrics={getGrpoSummaryTiles(metrics, isTerminalStatus)}
            progress={getGrpoProgressTiles(telemetry, runState)}
          />
          <GrpoTrainingHealthPanel statusDetails={metrics} />
        </>
      ) : (
        <>
          <TrainingLossPanel
            trainLoss={metrics?.metrics?.train_loss}
            valLoss={metrics?.metrics?.val_loss}
            maxSteps={maxXAxisValue}
            metrics={[
              ...getLossTiles(metrics, isTerminalStatus),
              ...getTrainingDiagnosticsTiles(telemetry, metrics, runState),
            ]}
            progress={getTrainingProgressTiles(telemetry)}
          />
          <TrainingHealthPanel statusDetails={metrics} />
        </>
      )}

      <RunConfigurationPanel
        customization={customization}
        telemetry={telemetry}
        onViewConfiguration={() => setOpenConfigSidePanel(true)}
      />

      <CustomizationConfigSidePanel
        open={openConfigSidePanel}
        onOpenChange={setOpenConfigSidePanel}
        customizationJobName={customizationJobName}
        workspace={workspace}
      />
    </Stack>
  );
};
