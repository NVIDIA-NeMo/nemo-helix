// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage } from '@nemo/common/src/api/common/utils';
import { ErrorPanel } from '@nemo/common/src/components/ErrorPanel';
import { KVPair } from '@nemo/common/src/components/KVPair';
import { RelativeTime } from '@nemo/common/src/components/RelativeTime';
import type { MiddlewareCall, VirtualModel } from '@nemo/sdk/generated/platform/schema';
import { useGetVirtualModel } from '@nemo/sdk/generated/platform/virtual-models';
import { Block, Flex, Skeleton, Stack, Text } from '@nvidia/foundations-react-core';
import { ROUTE_PARAMS } from '@studio/constants/routes';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { useRequiredPathParams } from '@studio/util/hooks/useRequiredPathParams';
import type { FC } from 'react';

interface MiddlewareCallViewProps {
  readonly call: MiddlewareCall;
}

const MiddlewareCallView: FC<MiddlewareCallViewProps> = ({ call }) => (
  <Block className="rounded-lg border border-base bg-surface-raised p-density-md">
    <Stack className="gap-density-sm">
      <KVPair label="Plugin" orientation="horizontal" size="narrow" value={call.name} />
      <KVPair label="Config type" orientation="horizontal" size="narrow" value={call.config_type} />
      {call.config_id ? (
        <KVPair
          label="Config ref"
          orientation="horizontal"
          size="narrow"
          truncate={false}
          value={call.config_id}
        />
      ) : null}
      {call.config && Object.keys(call.config).length > 0 ? (
        <Stack className="gap-density-xs">
          <Text kind="label/regular/sm" className="text-secondary">
            Config
          </Text>
          <pre className="overflow-auto whitespace-pre-wrap rounded bg-surface p-density-sm text-sm">
            {JSON.stringify(call.config, null, 2)}
          </pre>
        </Stack>
      ) : null}
    </Stack>
  </Block>
);

interface MiddlewarePipelineProps {
  readonly label: string;
  readonly calls: MiddlewareCall[] | undefined;
}

const MiddlewarePipeline: FC<MiddlewarePipelineProps> = ({ label, calls }) => (
  <Stack className="gap-density-sm">
    <Text kind="label/bold/sm">{label}</Text>
    {calls && calls.length > 0 ? (
      calls.map((call, index) => <MiddlewareCallView key={`${call.name}-${index}`} call={call} />)
    ) : (
      <Text kind="body/regular/sm" className="text-secondary">
        None
      </Text>
    )}
  </Stack>
);

/** Placeholder while the virtual model loads. The page heading and tabs render from the URL. */
const DetailsSkeleton: FC = () => (
  <Stack className="gap-density-lg" data-testid="virtual-model-details-skeleton">
    <Stack className="gap-density-md">
      {['created', 'default-model', 'autoprovisioned', 'models'].map((field) => (
        <Flex key={field} align="center" className="gap-density-md">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 flex-1" />
        </Flex>
      ))}
    </Stack>
    <Stack className="gap-density-md">
      <Skeleton className="h-5 w-24" />
      <Skeleton className="h-16 w-full" />
      <Skeleton className="h-16 w-full" />
    </Stack>
  </Stack>
);

interface VirtualModelDetailsProps {
  readonly virtualModel: VirtualModel;
}

const VirtualModelDetails: FC<VirtualModelDetailsProps> = ({ virtualModel }) => {
  const models = virtualModel.models ?? [];

  return (
    <Stack className="gap-density-lg">
      <Stack className="gap-density-md">
        <KVPair
          label="Created"
          orientation="horizontal"
          size="medium"
          value={
            virtualModel.created_at ? (
              <RelativeTime datetime={virtualModel.created_at} focusableForTooltip={false} />
            ) : undefined
          }
        />
        <KVPair
          label="Default model"
          orientation="horizontal"
          size="medium"
          truncate={false}
          value={virtualModel.default_model_entity}
        />
        <KVPair
          label="Autoprovisioned"
          orientation="horizontal"
          size="medium"
          value={virtualModel.autoprovisioned ? 'Yes' : 'No'}
        />
        {virtualModel.override_proxy ? (
          <KVPair
            label="Override proxy"
            orientation="horizontal"
            size="medium"
            truncate={false}
            value={virtualModel.override_proxy}
          />
        ) : null}
        <KVPair
          attributes={{ value: { className: 'whitespace-pre-wrap' } }}
          label="Models"
          orientation="horizontal"
          size="medium"
          truncate={false}
          value={models
            .map((m) => (m.backend_format ? `${m.model} (${m.backend_format})` : m.model))
            .join('\n')}
        />
      </Stack>

      <Stack className="gap-density-md">
        <Text kind="label/bold/md">Middleware</Text>
        <MiddlewarePipeline label="Request" calls={virtualModel.request_middleware} />
        <MiddlewarePipeline label="Response" calls={virtualModel.response_middleware} />
        <MiddlewarePipeline label="Post-response" calls={virtualModel.post_response_middleware} />
      </Stack>
    </Stack>
  );
};

export const VirtualModelDetailsTab: FC = () => {
  const workspace = useWorkspaceFromPath();
  const { virtualModelName } = useRequiredPathParams([ROUTE_PARAMS.virtualModelName]);

  const { data: virtualModel, error } = useGetVirtualModel(workspace, virtualModelName);

  if (virtualModel) {
    return <VirtualModelDetails virtualModel={virtualModel} />;
  }

  if (error) {
    return (
      <ErrorPanel
        title={`Failed to load virtual model '${virtualModelName}'`}
        errorMessage={getErrorMessage(error)}
      />
    );
  }

  return <DetailsSkeleton />;
};
