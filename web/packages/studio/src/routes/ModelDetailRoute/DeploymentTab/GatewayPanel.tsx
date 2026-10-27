// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { MiddlewareCall, ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { useGetVirtualModel } from '@nemo/sdk/generated/platform/virtual-models';
import { Anchor } from '@nvidia/foundations-react-core';
import { DetailPanel } from '@studio/routes/ModelDetailRoute/DetailsTab/layout/DetailPanel';
import { getVirtualModelDetailRoute } from '@studio/routes/utils';
import type { FC } from 'react';
import { Link } from 'react-router';

const formatMiddleware = (calls: MiddlewareCall[] | undefined) =>
  calls?.length ? calls.map((call) => call.name).join(' → ') : undefined;

/** The VirtualModel named after the model: every chat request goes through it. */
export const GatewayPanel: FC<{ model: ModelEntity }> = ({ model }) => {
  const { data: virtualModel, isError } = useGetVirtualModel(model.workspace, model.name, {
    query: { retry: false },
  });

  if (isError || !virtualModel) return null;

  return (
    <DetailPanel
      heading="Gateway"
      description={
        <>
          Chat requests go through the{' '}
          <Anchor asChild>
            <Link to={getVirtualModelDetailRoute(model.workspace, model.name)}>
              {model.name} virtual model
            </Link>
          </Anchor>{' '}
          before they reach the deployment.
        </>
      }
      tiles={[]}
      fields={[
        { label: 'Guardrails', value: virtualModel.guardrail_config_ids?.join(', ') || 'None' },
        { label: 'Request Middleware', value: formatMiddleware(virtualModel.request_middleware) },
        { label: 'Response Middleware', value: formatMiddleware(virtualModel.response_middleware) },
      ]}
    />
  );
};
