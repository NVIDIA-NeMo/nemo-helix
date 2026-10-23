/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

import Anthropic from '@nemo/common/src/svgs/model_icon_anthropic.svg?react';
import Deepseek from '@nemo/common/src/svgs/model_icon_deepseek.svg?react';
import Mistral from '@nemo/common/src/svgs/model_icon_mistral.svg?react';
import Nvidia from '@nemo/common/src/svgs/model_icon_nvidia.svg?react';
import OpenAi from '@nemo/common/src/svgs/model_icon_openai.svg?react';
import OpenRouter from '@nemo/common/src/svgs/model_icon_openrouter.svg?react';
import { LinkBreak, ChartFlow } from '@nemo/icons/icons';
import { FC, SVGProps } from 'react';

export type InferenceProviderPresetId =
  | 'build'
  | 'openai'
  | 'anthropic'
  | 'deepseek'
  | 'mistral'
  | 'openrouter'
  | 'custom';

export const PRESET_CREDENTIALS: Record<
  Exclude<InferenceProviderPresetId, 'custom'>,
  { name: string; host_url: string }
> = {
  build: { name: 'build', host_url: 'https://integrate.api.nvidia.com/v1' },
  openai: { name: 'openai', host_url: 'https://api.openai.com/v1' },
  anthropic: { name: 'anthropic', host_url: 'https://api.anthropic.com/v1' },
  deepseek: { name: 'deepseek', host_url: 'https://api.deepseek.com/v1' },
  mistral: { name: 'mistral', host_url: 'https://api.mistral.ai/v1' },
  openrouter: { name: 'openrouter', host_url: 'https://openrouter.ai/api/v1' },
};

export const MODEL_PROVIDER_ROWS: {
  id: Exclude<InferenceProviderPresetId, 'custom'>;
  label: string;
  filterText: string;
  Icon: FC<SVGProps<SVGSVGElement>>;
}[] = [
  {
    id: 'build',
    label: 'NVIDIA Build',
    filterText: 'nvidia build integrate',
    Icon: Nvidia,
  },
  {
    id: 'openai',
    label: 'OpenAI',
    filterText: 'openai',
    Icon: OpenAi,
  },
  {
    id: 'anthropic',
    label: 'Anthropic',
    filterText: 'anthropic claude',
    Icon: Anthropic,
  },
  {
    id: 'deepseek',
    label: 'DeepSeek',
    filterText: 'deepseek',
    Icon: Deepseek,
  },
  {
    id: 'mistral',
    label: 'Mistral AI',
    filterText: 'mistral',
    Icon: Mistral,
  },
  {
    id: 'openrouter',
    label: 'OpenRouter',
    filterText: 'openrouter router models api',
    Icon: OpenRouter,
  },
];

export const CUSTOM_ROW = {
  id: 'custom' as const,
  label: 'OpenAI Compatible Endpoint',
  filterText: 'openai compatible endpoint custom',
  Icon: ChartFlow,
};

export function getInferenceProviderPresetLabel(id: InferenceProviderPresetId): string {
  if (id === 'custom') return CUSTOM_ROW.label;
  return MODEL_PROVIDER_ROWS.find((r) => r.id === id)?.label ?? id;
}

export function getInferenceProviderPresetIcon(id: InferenceProviderPresetId) {
  if (id === 'custom') return CUSTOM_ROW.Icon;
  return MODEL_PROVIDER_ROWS.find((r) => r.id === id)?.Icon ?? LinkBreak;
}
