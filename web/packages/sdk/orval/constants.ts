// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const DEFAULT_INPUT = './openapi.yaml';
export const DEFAULT_TARGET = './generated/api.ts';
export const DEFAULT_SCHEMAS = './generated/schema';
export const DEFAULT_CLIENT = 'react-query';
export const headers = {
  'X-Source': 'NeMo Studio',
};
export type ServiceConfig = {
  path: string;
  url: string;
  zod?: boolean; // If true, generate zod client and schemas
  /**
   * Order of priority for API Base URL requests. Vite environment variables are checked first, then Node.js process.env variables.
   */
  apiEnvKeys?: string[];
};

export const serviceConfigs: Record<string, ServiceConfig> = {
  platform: {
    path: 'platform',
    url: `../../../../openapi/ga/individual/platform.openapi.yaml`,
    apiEnvKeys: ['VITE_PLATFORM_BASE_URL'],
    zod: true,
  },
  anonymizer: {
    path: 'anonymizer',
    url: `../../../../plugins/nemo-anonymizer/openapi/openapi.yaml`,
    apiEnvKeys: ['VITE_PLATFORM_BASE_URL'],
    zod: true,
  },
  customizer: {
    path: 'customizer',
    url: `../../../../plugins/nemo-customizer/openapi/openapi.yaml`,
    apiEnvKeys: ['VITE_PLATFORM_BASE_URL'],
    zod: true,
  },
  'data-designer': {
    path: 'data-designer',
    url: `../../../../plugins/nemo-data-designer/openapi/openapi.yaml`,
    apiEnvKeys: ['VITE_PLATFORM_BASE_URL'],
    zod: true,
  },
  agents: {
    path: 'agents',
    url: `../../../../plugins/nemo-agents/openapi/openapi.yaml`,
    apiEnvKeys: ['VITE_PLATFORM_BASE_URL'],
    zod: true,
  },
  'safe-synthesizer': {
    path: 'safe-synthesizer',
    url: `../../../../plugins/nemo-safe-synthesizer/openapi/openapi.yaml`,
    apiEnvKeys: ['VITE_PLATFORM_BASE_URL'],
    zod: true,
  },
  evaluator: {
    path: 'evaluator',
    url: `../../../../plugins/nemo-evaluator/openapi/openapi.yaml`,
    apiEnvKeys: ['VITE_PLATFORM_BASE_URL'],
    zod: true,
  },
  insights: {
    path: 'insights',
    url: `../../../../plugins/nemo-insights/openapi/openapi.yaml`,
    apiEnvKeys: ['VITE_PLATFORM_BASE_URL'],
    zod: true,
  },
  'agent-optimization': {
    path: 'agent-optimization',
    url: `../../../../plugins/nemo-agent-optimization/openapi/openapi.yaml`,
    apiEnvKeys: ['VITE_PLATFORM_BASE_URL'],
    zod: true,
  },
};

export const serviceToConfig = {
  agents: 'nemoMicroservices',
  anonymizer: 'nemoMicroservices',
  customizer: 'nemoMicroservices',
  'data-designer': 'nemoMicroservices',
  'deployment-management': 'nemoMicroservices',
  'entity-store': 'nemoMicroservices',
  evaluation: 'nemoMicroservices',
  guardrails: 'nemoMicroservices',
  intake: 'nemoMicroservices',
  jobs: 'nemoMicroservices',
  'safe-synthesizer': 'nemoMicroservices',
};
