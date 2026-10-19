/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import { KVPair } from '@nemo/common/src/components/KVPair';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import { StatusBadge } from '@nemo/common/src/components/StatusBadge';
import { JOB_POLLING_INTERVAL_MS } from '@nemo/common/src/constants';
import { useLiveSeconds } from '@nemo/common/src/hooks/useLiveSeconds';
import { formatTimeInSeconds, utcToLocalDate } from '@nemo/common/src/utils/date';
import { useModelsGetDeploymentConfigVersion } from '@nemo/sdk/generated/platform/model-deployment-configs';
import { useModelsGetLatestDeployment } from '@nemo/sdk/generated/platform/model-deployments';
import {
  Engine,
  type ModelDeployment,
  ModelDeploymentStatus,
} from '@nemo/sdk/generated/platform/schema';
import { Banner, Button, Flex, SidePanel, Stack, Text } from '@nvidia/foundations-react-core';
import { ErrorMessageWithRetry } from '@studio/components/ErrorMessageWithRetry';
import { Loading } from '@studio/components/Layouts/Loading';
import { useRehydrateDeploymentsListFromDetail } from '@studio/hooks/useRehydrateDeploymentsListFromDetail';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { formatElapsedTime } from '@studio/util/date';
import { type FC, useMemo } from 'react';

function formatDeploymentEngine(engine: string | undefined): string {
  if (!engine) return '-';
  const normalized = engine.toLowerCase();
  if (normalized === Engine.nim) return 'NIM';
  if (normalized === Engine.vllm) return 'vLLM';
  if (normalized === Engine.generic) return 'Generic';
  return engine;
}

function isStatusMessageError(status: ModelDeploymentStatus | undefined): boolean {
  return (
    status === ModelDeploymentStatus.ERROR ||
    status === ModelDeploymentStatus.UNKNOWN ||
    status === ModelDeploymentStatus.LOST
  );
}

/** Statuses where deployment work has finished (show fixed duration, not a live timer). */
function isDeploymentTerminalStatus(status: ModelDeploymentStatus | undefined): boolean {
  if (!status) return false;
  return (
    status === ModelDeploymentStatus.READY ||
    status === ModelDeploymentStatus.ERROR ||
    status === ModelDeploymentStatus.DELETED ||
    status === ModelDeploymentStatus.LOST ||
    status === ModelDeploymentStatus.UNKNOWN
  );
}

export type DeploymentDetailsSidePanelDeployment = ModelDeployment & { description?: string };

export interface DeploymentDetailsSidePanelProps {
  open: boolean;
  onClose: () => void;
  /** Decoded deployment name from the URL (must match route param). */
  deploymentName: string;
  onRequestDelete: (deployment: ModelDeployment) => void;
}

export const DeploymentDetailsSidePanel: FC<DeploymentDetailsSidePanelProps> = ({
  open,
  onClose,
  deploymentName,
  onRequestDelete,
}) => {
  const workspace = useWorkspaceFromPath();
  const queryEnabled = Boolean(open && workspace && deploymentName);

  const { data, isLoading, isPending, isError, refetch } = useModelsGetLatestDeployment(
    workspace,
    deploymentName,
    {
      query: {
        enabled: queryEnabled,
        refetchInterval: (query) =>
          isDeploymentTerminalStatus(query.state.data?.status) ? false : JOB_POLLING_INTERVAL_MS,
      },
    }
  );

  useRehydrateDeploymentsListFromDetail({
    workspace,
    deployment: data,
    enabled: queryEnabled,
  });

  const configName = data?.config ?? '';
  const configVersionKey =
    data?.config_version != null && configName ? String(data.config_version) : '';

  const { data: deploymentConfig, isLoading: isLoadingDeploymentConfig } =
    useModelsGetDeploymentConfigVersion(workspace, configName, configVersionKey, {
      query: {
        enabled: queryEnabled && Boolean(configName && configVersionKey),
      },
    });

  const displayDeployment = useMemo((): DeploymentDetailsSidePanelDeployment | null => {
    if (!data) return null;
    const description = deploymentConfig?.description;
    return {
      ...data,
      ...(description != null && description !== '' ? { description } : {}),
    };
  }, [data, deploymentConfig]);

  const engineDisplay = useMemo(() => {
    return formatDeploymentEngine(deploymentConfig?.engine);
  }, [deploymentConfig]);

  const liveSeconds = useLiveSeconds({
    startDate:
      displayDeployment &&
      !isDeploymentTerminalStatus(displayDeployment.status) &&
      displayDeployment.created_at
        ? utcToLocalDate(displayDeployment.created_at)
        : undefined,
  });

  const statusMessageContent =
    displayDeployment && isStatusMessageError(displayDeployment.status) ? (
      <Banner className="w-full " kind="inline" status="error">
        <Text className="whitespace-pre-wrap" kind="body/regular/sm">
          {displayDeployment.status_message}
        </Text>
      </Banner>
    ) : (
      displayDeployment?.status_message || '-'
    );

  const isAwaitingFirstResult = queryEnabled && !data && (isPending || isLoading);
  const showError = queryEnabled && isError && !data;
  const showBody = Boolean(displayDeployment);

  const statusValue =
    displayDeployment?.status != null ? (
      <Flex align="center" gap="2">
        <StatusBadge status={displayDeployment.status} />
        {isDeploymentTerminalStatus(displayDeployment.status) &&
        displayDeployment.created_at &&
        displayDeployment.updated_at
          ? formatElapsedTime(
              new Date(displayDeployment.created_at),
              new Date(displayDeployment.updated_at)
            )
          : !isDeploymentTerminalStatus(displayDeployment.status)
            ? formatTimeInSeconds(liveSeconds)
            : null}
      </Flex>
    ) : null;

  return (
    <SidePanel
      bordered
      className="w-[600px]"
      modal
      open={open}
      side="right"
      slotFooter={
        showBody ? (
          <Flex className="w-full" justify="end">
            <Button
              color="danger"
              onClick={() => {
                if (displayDeployment) {
                  onRequestDelete(displayDeployment);
                }
              }}
            >
              Delete
            </Button>
          </Flex>
        ) : undefined
      }
      slotHeading={
        <Text className="min-w-0 truncate" kind="label/bold/lg" title={deploymentName}>
          {deploymentName}
        </Text>
      }
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          onClose();
        }
      }}
    >
      <Stack className="flex min-h-0 flex-1 flex-col gap-density-lg">
        {isAwaitingFirstResult ? (
          <Loading />
        ) : showError ? (
          <ErrorMessageWithRetry onRetry={refetch} />
        ) : showBody && displayDeployment ? (
          <Stack className="shrink-0 gap-density-md">
            <KVPair label="Status" value={statusValue} />
            <KVPair
              attributes={{ value: { className: 'flex-1 whitespace-pre-wrap' } }}
              label="Status Message"
              value={statusMessageContent}
            />
            {displayDeployment.description ? (
              <KVPair
                label="Description"
                orientation="horizontal"
                size="medium"
                attributes={{ value: { className: 'whitespace-pre-wrap' } }}
                value={displayDeployment.description}
              />
            ) : null}
            <KVPair
              label="Engine"
              orientation="horizontal"
              size="medium"
              loading={isLoadingDeploymentConfig}
              value={engineDisplay}
            />
            <KVPair
              label="Image Name"
              orientation="horizontal"
              size="medium"
              loading={isLoadingDeploymentConfig}
              value={deploymentConfig?.executor_config?.image_name}
            />
            <KVPair
              label="Image Tag"
              orientation="horizontal"
              size="medium"
              loading={isLoadingDeploymentConfig}
              value={deploymentConfig?.executor_config?.image_tag ?? '—'}
            />
            <KVPair
              label="Model Name"
              orientation="horizontal"
              size="medium"
              loading={isLoadingDeploymentConfig}
              value={deploymentConfig?.model_spec?.model_name ?? '—'}
            />
            <KVPair
              label="Created"
              orientation="horizontal"
              size="medium"
              value={
                displayDeployment.created_at && (
                  <RelativeTime
                    focusableForTooltip={false}
                    datetime={displayDeployment.created_at}
                  />
                )
              }
            />
          </Stack>
        ) : null}
      </Stack>
    </SidePanel>
  );
};
