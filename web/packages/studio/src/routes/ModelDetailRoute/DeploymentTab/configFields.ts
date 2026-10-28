// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ModelDeploymentConfig } from '@nemo/sdk/generated/platform/schema';
import { formatDeploymentEngine } from '@studio/routes/DeploymentsListRoute/DeploymentDetailsSidePanel/utils';
import type { DetailField, DetailTile } from '@studio/routes/ModelDetailRoute/DetailsTab/fields';
import { formatMaybe } from '@studio/util/intakeTelemetry';

const formatImage = ({ executor_config: executor }: ModelDeploymentConfig) =>
  executor.image_name
    ? [executor.image_name, executor.image_tag].filter(Boolean).join(':')
    : undefined;

const formatSource = ({ model_spec: spec }: ModelDeploymentConfig) => {
  if (!spec.model_name) return undefined;
  const name = [spec.model_namespace, spec.model_name].filter(Boolean).join('/');
  return spec.model_revision ? `${name} @ ${spec.model_revision}` : name;
};

export const getConfigTiles = (config: ModelDeploymentConfig): DetailTile[] => {
  const { executor_config: executor, model_spec: spec } = config;
  return [
    { label: 'Engine', value: formatDeploymentEngine(config.engine) },
    { label: 'GPUs', value: formatMaybe(executor.gpu) },
    { label: 'Disk', value: formatMaybe(executor.disk_size) },
    { label: 'LoRA', value: spec.lora_enabled ? 'Enabled' : 'Disabled' },
  ];
};

/** Environment variables are listed by name only: their values can hold credentials. */
export const getConfigFields = (config: ModelDeploymentConfig): DetailField[] => {
  const { executor_config: executor, model_spec: spec } = config;
  const envNames = Object.keys(executor.additional_envs ?? {});
  return [
    { label: 'Image', value: formatImage(config) },
    { label: 'Source', value: formatSource(config) },
    { label: 'Health Check Path', value: executor.health_check_path },
    { label: 'Chat Template', value: spec.chat_template ? 'Overridden' : undefined },
    { label: 'Tool Calling', value: spec.tool_call_config ? 'Overridden' : undefined },
    { label: 'Arguments', value: executor.additional_args?.join(' ') },
    { label: 'Environment', value: envNames.length ? envNames.join(', ') : undefined },
  ];
};
