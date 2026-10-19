// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FinetuningType } from '@nemo/sdk/generated/platform/schema';
import { Box, Brain, Toolbox } from 'lucide-react';
import { ReactNode } from 'react';

import DeepSeek from '../svgs/model_icon_deepseek.svg?react';
import Google from '../svgs/model_icon_google.svg?react';
import Meta from '../svgs/model_icon_meta.svg?react';
import Microsoft from '../svgs/model_icon_microsoft.svg?react';
import Mistral from '../svgs/model_icon_mistral.svg?react';
import Nvidia from '../svgs/model_icon_nvidia.svg?react';
import Qwen from '../svgs/model_icon_qwen.svg?react';

export type TagLabel = 'tool-calling' | 'reasoning';

export const tagToIcon = (tag: TagLabel): ReactNode => {
  switch (tag) {
    case 'tool-calling':
      return <Toolbox />;
    case 'reasoning':
      return <Brain />;
    default:
      return null;
  }
};

export const creatorToIcon = (creator: string, props?: { className?: string }): ReactNode => {
  const creatorLower = creator.toLowerCase();
  switch (creatorLower) {
    case 'deepseek':
    case 'deepseek-ai':
      return <DeepSeek {...props} />;
    case 'google':
      return <Google {...props} />;
    case 'meta':
      return <Meta {...props} />;
    case 'microsoft':
      return <Microsoft {...props} />;
    case 'mistral':
    case 'mistralai':
      return <Mistral {...props} />;
    case 'nvidia':
      return <Nvidia {...props} />;
    case 'qwen':
      return <Qwen {...props} />;
    default:
      return <Box {...props} />;
  }
};

export interface ModelMetadata {
  name: string;
  creator: string;
  architecture?: string;
  description?: string;
  'max-io-tokens'?: string;
  parameters?: string;
  'training-data'?: string;
  'fine-tune-options'?: FinetuningType[];
  'recommended-gpus-for-customization'?: { [key in FinetuningType]?: number };
  version?: string;
  tags?: TagLabel[];
}

// Information from Customizer docs:
// https://docs.nvidia.com/nemo/microservices/latest/fine-tune/models/index.html
export const META_MODEL_METADATA: Record<string, ModelMetadata> = {
  'meta/llama-3.3-70b-instruct': {
    name: 'Llama-3.3-70b Instruct',
    creator: 'Meta',
    architecture: 'transformer',
    description:
      'Llama-3.3-70b is a large language AI model optimized for advanced dialogue and reasoning capabilities.',
    'max-io-tokens': '8192',
    parameters: '70 billion',
    'training-data': '15+ trillion tokens (up to 2024)',
    'fine-tune-options': ['lora'],
    'recommended-gpus-for-customization': {
      lora: 16,
    },
    tags: ['tool-calling'],
  },
  'meta/llama-3.2-3b-instruct': {
    name: 'Llama-3.2-3b Instruct',
    creator: 'Meta',
    architecture: 'transformer',
    description:
      'Llama-3.2-3b is a compact yet powerful language model suitable for various dialogue applications.',
    'max-io-tokens': '8192',
    parameters: '3 billion',
    'training-data': '15+ trillion tokens (up to 2024)',
    'fine-tune-options': ['lora'],
    'recommended-gpus-for-customization': {
      lora: 1,
    },
    tags: ['tool-calling'],
  },
  'meta/llama-3.2-1b-instruct': {
    name: 'Llama-3.2-1b Instruct',
    creator: 'Meta',
    architecture: 'transformer',
    description:
      'Llama-3.2-1b is a lightweight language model designed for efficient deployment while maintaining strong capabilities.',
    'max-io-tokens': '8192',
    parameters: '1 billion',
    'training-data': '15+ trillion tokens (up to 2024)',
    'fine-tune-options': ['lora'],
    'recommended-gpus-for-customization': {
      lora: 1,
    },
    tags: ['tool-calling'],
  },
  'meta/llama-3.1-70b-instruct': {
    name: 'Llama-3.1-70b Instruct',
    creator: 'Meta',
    architecture: 'transformer',
    description:
      'Llama-3.1-70b is a large language AI model optimized for multilingual dialogue uses.',
    'max-io-tokens': '8192',
    parameters: '70 billion',
    'training-data': '15 trillion tokens (up to December 2023)',
    'fine-tune-options': ['lora'],
    'recommended-gpus-for-customization': {
      lora: 16,
    },
    tags: ['tool-calling'],
  },
  'meta/llama-3.1-8b-instruct': {
    name: 'Llama-3.1-8b Instruct',
    creator: 'Meta',
    architecture: 'transformer',
    description:
      'Llama-3.1-8b is a large language AI model optimized for multilingual dialogue uses.',
    'max-io-tokens': '8192',
    parameters: '8 billion',
    'training-data': '15 trillion tokens (up to December 2023)',
    'fine-tune-options': ['lora'],
    'recommended-gpus-for-customization': {
      lora: 4,
    },
    tags: ['tool-calling'],
  },
  'meta/llama-3-70b-instruct': {
    name: 'Llama-3-70b Instruct',
    creator: 'Meta',
    architecture: 'transformer',
    description:
      'Llama-3-70b is a large language AI model comprising a collection of models capable of generating text and code in response to prompts.',
    'max-io-tokens': '8192',
    parameters: '70 billion',
    'training-data': '15 trillion tokens (up to December 2023)',
    'fine-tune-options': ['lora'],
    'recommended-gpus-for-customization': {
      lora: 16,
    },
    tags: ['tool-calling'],
  },
};

export const NEMOTRON_MODEL_METADATA: Record<string, ModelMetadata> = {
  'nvidia/nemotron-3.5-lightning-30b-a3b': {
    name: 'NVIDIA Nemotron 3.5 Lightning 30B A3B',
    creator: 'NVIDIA',
    architecture: 'hybrid-mamba-transformer-moe',
    description:
      'Nemotron 3.5 Lightning 30B-A3B is a unified reasoning and chat model with 30B total parameters and 3B active parameters.',
    'max-io-tokens': '1048576',
    parameters: '30 billion (3 billion active)',
    'training-data': 'Not specified',
    'fine-tune-options': ['lora'],
    'recommended-gpus-for-customization': {
      lora: 1,
    },
    tags: ['reasoning', 'tool-calling'],
  },
  'nvidia/nemotron-nano-llama-3.1-8b': {
    name: 'Llama-3.1 Nemotron Nano 8B v1',
    creator: 'NVIDIA',
    architecture: 'transformer',
    description:
      'Llama-3.1 Nemotron Nano 8B v1 is a compact, instruction-tuned model for efficient customization and deployment.',
    'max-io-tokens': '4096',
    parameters: '8 billion',
    'training-data': 'Not specified',
    'fine-tune-options': ['lora', 'all_weights'],
    'recommended-gpus-for-customization': {
      lora: 1,
      all_weights: 8,
    },
    tags: ['reasoning'],
  },
  'nvidia/nemotron-super-llama-3.3-49b': {
    name: 'Llama-3.3 Nemotron Super 49B',
    creator: 'NVIDIA',
    architecture: 'transformer',
    description:
      'Llama-3.3 Nemotron Super 49B v1 is a large, instruction-tuned model for advanced dialogue and reasoning tasks.',
    'max-io-tokens': '4096',
    parameters: '49 billion',
    'training-data': 'Not specified',
    'fine-tune-options': ['lora'],
    'recommended-gpus-for-customization': {
      lora: 4,
    },
    tags: ['reasoning'],
  },
};

export const MICROSOFT_MODEL_METADATA: Record<string, ModelMetadata> = {
  'microsoft/phi-4': {
    name: 'Microsoft Phi-4',
    creator: 'Microsoft',
    architecture: 'decoder-only transformer',
    description:
      "Phi-4 is Microsoft's most advanced small language model, designed to deliver strong reasoning capabilities while being efficient to deploy.",
    'max-io-tokens': '16384',
    parameters: '14 billion',
    'training-data': 'High-quality data with emphasis on reasoning and code',
    'fine-tune-options': ['lora'],
    'recommended-gpus-for-customization': {
      lora: 2,
    },
  },
};

export const MODEL_METADATA = {
  ...META_MODEL_METADATA,
  ...NEMOTRON_MODEL_METADATA,
  ...MICROSOFT_MODEL_METADATA,
};

/**
 * Given the version string of a model, return the version number
 * @param version The version string of the model
 * @returns The version number
 */
export const getModelVersion = (version?: string): string => {
  return version?.split(':')[1] || 'Unknown';
};
