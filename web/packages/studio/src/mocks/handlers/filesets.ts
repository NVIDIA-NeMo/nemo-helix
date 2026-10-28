// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  FilesetFileOutput,
  FilesetOutput,
  FilesetOutputsPage,
} from '@nemo/sdk/generated/platform/schema';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { http, HttpResponse } from 'msw';

export const filesetsHandlers = [
  http.get<{ workspace: string }, never, FilesetOutputsPage>(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`,
    async ({ request, params: { workspace } }) => {
      const { datasets } = await import('@studio/mocks/datasets');
      // Honor the exact-name filter the create forms use to check whether a name is taken.
      const name = new URL(request.url).searchParams.get('filter[name]');
      if (name === null) return HttpResponse.json(datasets);
      const data = datasets.data.filter((d) => d.workspace === workspace && d.name === name);
      return HttpResponse.json({
        ...datasets,
        data,
        pagination: {
          page: 1,
          page_size: data.length,
          current_page_size: data.length,
          total_pages: 1,
          total_results: data.length,
        },
      });
    }
  ),
  http.get<{ workspace: string; name: string }, never, FilesetOutput>(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name`,
    async ({ params: { workspace, name } }) => {
      const { datasets } = await import('@studio/mocks/datasets');
      const fileset = datasets.data.find((d) => d.workspace === workspace && d.name === name);
      if (!fileset) {
        // Return a generic fileset for testing
        return HttpResponse.json({
          id: `${workspace}/${name}`,
          name,
          workspace,
          description: '',
          purpose: 'dataset',
          storage: { type: 'local' },
          metadata: {},
          custom_fields: {},
          project: workspace,
          created_at: '2024-12-17T16:08:56.880768',
          updated_at: '2024-12-17T16:08:56.880771',
        } as FilesetOutput);
      }
      return HttpResponse.json(fileset);
    }
  ),
  http.head(`${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name/-/*`, () => {
    return new HttpResponse(null, {
      status: 200,
      headers: {
        'Content-Type': 'application/octet-stream',
        'Access-Control-Allow-Origin': '*',
      },
    });
  }),

  http.get<{ workspace: string; name: string }, { path: string }, FilesetFileOutput>(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name/-/*`,
    async ({ request }) => {
      const { customizationFiles } = await import('@studio/mocks/datasets/files');
      const url = new URL(request.url);
      const pathParam = url.searchParams.get('path');

      const filteredFiles = pathParam
        ? customizationFiles.filter((f) => f.path.startsWith(pathParam))
        : customizationFiles;
      return HttpResponse.json(filteredFiles[0]);
    }
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name/files`,
    async ({ request }) => {
      const { customizationFiles } = await import('@studio/mocks/datasets/files');
      const url = new URL(request.url);
      const pathParam = url.searchParams.get('path');

      const filteredFiles = pathParam
        ? customizationFiles.filter((f) => f.path.startsWith(pathParam))
        : customizationFiles;

      return HttpResponse.json({
        data: filteredFiles.map((f) => ({
          path: f.path,
          size: f.size,
          file_ref: f.file_ref,
        })),
        pagination: {
          page: 1,
          page_size: 100,
          current_page_size: filteredFiles.length,
          total_pages: 1,
          total_results: filteredFiles.length,
        },
      });
    }
  ),
  // Download fileset file - wildcard path for nested files
  http.options(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name/files/-/*`,
    () => {
      return new HttpResponse(null, {
        status: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
          'Access-Control-Allow-Headers': '*',
        },
      });
    }
  ),
  http.head(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name/files/-/*`,
    () => {
      return new HttpResponse(null, {
        status: 200,
        headers: {
          'Content-Type': 'application/octet-stream',
          'Access-Control-Allow-Origin': '*',
        },
      });
    }
  ),
  http.get(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name/files/-/*`,
    async () => {
      const { mockLfsObjectResponse } = await import('@studio/mocks/datasets');
      return new HttpResponse(mockLfsObjectResponse, {
        status: 200,
        headers: {
          'Content-Type': 'application/octet-stream',
          'Access-Control-Allow-Origin': '*',
        },
      });
    }
  ),
  http.post<{ workspace: string }, { name: string; description?: string }, FilesetOutput>(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`,
    async ({ params, request }) => {
      const body = await request.json();
      const fileset: FilesetOutput = {
        id: 'test-fileset-id',
        name: body.name,
        workspace: params.workspace,
        description: body.description || '',
        purpose: 'dataset',
        storage: { type: 'local', path: 'local/path' },
        metadata: {},
        custom_fields: {},
        project: params.workspace,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      return HttpResponse.json(fileset);
    }
  ),

  http.patch<{ workspace: string; name: string }>(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name`,
    async ({ params, request }) => {
      const body = (await request.json()) as Record<string, unknown>;
      return HttpResponse.json({
        id: `${params.workspace}/${params.name}`,
        name: params.name,
        workspace: params.workspace,
        purpose: 'dataset',
        storage: { type: 'local', path: '/data' },
        metadata: {},
        custom_fields: {},
        project: params.workspace,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: new Date().toISOString(),
        ...body,
      } as FilesetOutput);
    }
  ),
  http.delete(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name`,
    () => new HttpResponse(null, { status: 200 })
  ),
  http.options(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name/-/*`,
    () =>
      new HttpResponse(null, {
        status: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'PUT, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      })
  ),
  http.put(
    `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets/:name/-/*`,
    () => new HttpResponse(null, { status: 200 })
  ),
];
