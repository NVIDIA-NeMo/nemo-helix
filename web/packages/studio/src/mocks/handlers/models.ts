// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ModelEntitySortField,
  ModelEntitysPage,
  ModelProvider,
  ModelProvidersPage,
  ModelsListModelsParams,
} from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

/**
 * Mock handlers for platform model endpoints
 */
export const modelsHandlers = [
  // List models with query parameters
  http.get<{ workspace: string }, never, ModelEntitysPage>(
    `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models`,
    async ({ request, params: { workspace } }) => {
      const { entityStoreBaseModel1, entityStorePromptTunedModel1, entityStoreCustomizedModel1 } =
        await import('@studio/mocks/entity-store/models');
      const url = new URL(request.url);
      const searchParams = url.searchParams;

      // Parse query parameters
      const query: ModelsListModelsParams = {
        sort: searchParams.get('sort') as ModelEntitySortField | undefined,
        page: Number(searchParams.get('page')) || 1,
        page_size: Number(searchParams.get('page_size')) || 1000,
      };

      const { entityStoreModelWithFileset } = await import('@studio/mocks/entity-store/models');

      // Get all models
      let models = [
        entityStoreBaseModel1,
        entityStorePromptTunedModel1,
        entityStoreCustomizedModel1,
        entityStoreModelWithFileset,
      ];

      // Filter by adapters (models with PEFT/LoRA adapters)
      const adaptersFilter = searchParams.get('filter[adapters]');
      if (adaptersFilter === 'true') {
        models = models.filter((model) => model.adapters && model.adapters.length > 0);
      } else if (adaptersFilter === 'false') {
        models = models.filter((model) => !model.adapters || model.adapters.length === 0);
      }

      // Filter by prompt (prompt-tuned models)
      const promptFilter = searchParams.get('filter[prompt]');
      if (promptFilter === 'true') {
        models = models.filter(
          (model) => model.spec?.num_virtual_tokens && model.spec.num_virtual_tokens > 0
        );
      } else if (promptFilter === 'false') {
        models = models.filter(
          (model) => !model.spec?.num_virtual_tokens || model.spec.num_virtual_tokens === 0
        );
      }

      // Filter by fileset
      const filesetFilter = searchParams.get('filter[fileset]');
      if (filesetFilter) {
        models = models.filter((model) => model.fileset === filesetFilter);
      }

      // Filter by workspace
      const workspaceFilter = searchParams.get('filter[workspace]');
      if (workspaceFilter) {
        models = models.filter((model) => model.workspace === workspaceFilter);
      } else if (!filesetFilter) {
        // If no workspace filter and no fileset filter, filter by the workspace from the path
        models = models.filter((model) => model.workspace === workspace);
      }

      // Sort models
      if (query.sort) {
        switch (query.sort) {
          case '-created_at':
            models = models.sort(
              (a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime()
            );
            break;
          case 'name':
            models = models.sort((a, b) => a.name!.localeCompare(b.name!));
            break;
          case '-name':
            models = models.sort((a, b) => b.name!.localeCompare(a.name!));
            break;
          case 'created_at':
          default:
            models = models.sort(
              (a, b) => new Date(a.created_at!).getTime() - new Date(b.created_at!).getTime()
            );
            break;
        }
      }

      return HttpResponse.json({
        data: models,
      });
    }
  ),

  // OPTIONS preflight request for CORS
  http.options(`${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models`, () => {
    return new HttpResponse(null, { status: 204 });
  }),

  // Get a single model provider by name
  http.get<{ workspace: string; provider: string }, ModelProvider>(
    `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/providers/:provider`,
    ({ params: { workspace, provider } }) => {
      const now = new Date().toISOString();
      return HttpResponse.json<ModelProvider>({
        name: provider,
        workspace,
        host_url: 'https://example.com',
        created_at: now,
        updated_at: now,
      });
    }
  ),

  // List model providers (inference providers)
  http.get(`${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/providers`, () => {
    const providers: ModelProvidersPage['data'] = [];
    return HttpResponse.json({
      data: providers,
      pagination: {
        page: 1,
        page_size: 100,
        current_page_size: 0,
        total_pages: 1,
        total_results: 0,
      },
    } satisfies ModelProvidersPage);
  }),

  // Create model provider
  http.post(
    `${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/providers`,
    async ({ request, params: { workspace } }) => {
      const body = (await request.json()) as {
        name: string;
        host_url: string;
        api_key_secret_name?: string;
        description?: string;
      };
      const now = new Date().toISOString();
      return HttpResponse.json(
        {
          name: body.name,
          workspace,
          host_url: body.host_url,
          api_key_secret_name: body.api_key_secret_name,
          description: body.description,
          created_at: now,
          updated_at: now,
        },
        { status: 201 }
      );
    }
  ),
];
