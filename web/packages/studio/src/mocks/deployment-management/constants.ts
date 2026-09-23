// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const getModelDeploymentsListResponse = {
  data: [
    {
      async_enabled: false,
      config: {
        created_at: '2025-09-24T16:42:37.026262644Z',
        model: 'meta/llama-3.2-3b-instruct',
        name: 'llama-3.2-3b-instruct-config',
        namespace: 'ben-test',
        nim_deployment: {
          additional_envs: {
            NIM_GUIDED_DECODING_BACKEND: 'fast_outlines',
            NIM_SERVED_MODEL_NAME: 'ben-test/llama-3.2-3b-instruct',
          },
          disable_lora_support: false,
          gpu: 1,
          image_name: 'nvcr.io/nim/meta/llama-3.2-3b-instruct',
          image_tag: '1.8.5',
        },
      },
      created_at: '2025-09-24T16:42:53.608326298Z',
      deployed: true,
      name: 'codellama-70b',
      namespace: 'meta',
      status_details: {
        description: 'deployment "meta-codellama-70b-deployment" successfully rolled out\n',
        status: 'ready',
      },
      url: '',
    },
    {
      async_enabled: false,
      config: {
        model: 'meta/codellama-13b-instruct',
        nim_deployment: {
          gpu: 1,
          image_name: 'nvcr.io/nim/meta/codellama-13b-instruct',
          image_tag: '1.2.2',
        },
      },
      created_at: '2025-07-10T01:02:36.549374463Z',
      deployed: false,
      name: 'codellama-13b-instruct',
      namespace: 'default',
      status_details: {
        description: 'no associated NIMService found',
        status: 'unknown',
      },
      url: '',
    },
    {
      async_enabled: false,
      config: {
        model: 'meta/codellama-13b-instruct',
        nim_deployment: {
          gpu: 1,
          image_name: 'nvcr.io/nim/meta/codellama-13b-instruct',
          image_tag: 'latest',
        },
      },
      created_at: '2025-07-10T01:30:49.998079539Z',
      deployed: false,
      name: 'codellama-13b-instruct_latest',
      namespace: 'default',
      status_details: {
        description: 'no associated NIMService found',
        status: 'unknown',
      },
      url: '',
    },
    {
      async_enabled: false,
      config: {
        model: 'meta/llama-3.1-8b-instruct',
        nim_deployment: {
          gpu: 1,
          image_name: 'registry.example.com/nemo-helix/llama-3.1-8b-instruct',
          image_tag: '1.6-dev1',
          pvc_size: '25Gi',
        },
      },
      created_at: '2025-02-27T21:07:58.098138359Z',
      deployed: false,
      name: 'llama-3.1-8b-instruct',
      namespace: 'default',
      status_details: {
        description: 'no associated NIMService found',
        status: 'unknown',
      },
      url: '',
    },
  ],
  filter: {},
  object: 'list',
  pagination: {
    current_page_size: 4,
    page: 1,
    page_size: 1000,
    total_pages: 1,
    total_results: 4,
  },
  sort: 'created_at',
};
