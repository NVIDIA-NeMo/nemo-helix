// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MOCKS_DIR } from '@e2e-tests/utils/constants';
import { PLATFORM_BASE_URL, NHX_BASE_URL } from '@e2e-tests/utils/environment';
import { APIRequestContext } from '@playwright/test';
import crypto from 'crypto';
import fs from 'fs/promises';
import path from 'path';

/** Dataset shape used by e2e API (files_url, name, namespace). */
interface Dataset {
  files_url?: string;
  name?: string;
  namespace?: string;
  [key: string]: unknown;
}

export class DatasetsAPI {
  constructor(private request: APIRequestContext) {}

  private getFileNameFromPath(filePath: string) {
    const fileNameParts = filePath.split('/');
    return fileNameParts.length > 1 ? fileNameParts.at(-1) : filePath;
  }

  async createDataset(name: string, namespace: string, project: string, description: string = '') {
    // Create repo via Data Store
    const createRepoResponse = await this.request.post(
      `${PLATFORM_BASE_URL}/v1/hf/api/repos/create`,
      {
        data: {
          name,
          organization: namespace,
          type: 'dataset',
        },
      }
    );
    const createRepoResponseData = (await createRepoResponse.json()) as { url: string };
    const repoUrl = `hf://${createRepoResponseData.url}`;

    // Create dataset in Entity Store
    const createDatasetResponse = await this.request.post(`${NHX_BASE_URL}/v1/datasets`, {
      data: {
        description,
        files_url: repoUrl,
        name,
        namespace,
        project,
      },
    });
    const createDatasetResponseData = (await createDatasetResponse.json()) as Dataset;
    return createDatasetResponseData;
  }

  async deleteDataset(name: string, namespace: string) {
    // Delete dataset in Entity Store
    await this.request.delete(`${NHX_BASE_URL}/v1/datasets/${namespace}/${name}`);
    // Delete repo via Data Store
    await this.request.delete(`${PLATFORM_BASE_URL}/v1/hf/api/repos/delete`, {
      data: {
        name,
        organization: namespace,
        type: 'dataset',
      },
    });
  }

  /**
   * Uploads the file at the given local path to the given dataset.
   * Typically, we'd use the HuggingFace SDK's `uploadFile` function to handle file uploads,
   * but from a Playwright test we need to use the APIRequestContext to make HTTP calls. So,
   * here we need to chain together the necessary calls manually.
   *
   * @param dataset Dataset to upload the file into
   * @param testFilePath Path to test file stored locally in `MOCKS_DIR`
   * @param folder (optional) Folder in the dataset to store the file in
   */
  async uploadFile(dataset: Dataset, testFilePath: string, folder?: string) {
    const localFilePath = path.join(MOCKS_DIR, testFilePath);
    const fileContents = await fs.readFile(localFilePath);

    const fileName = this.getFileNameFromPath(localFilePath);
    // Path at which to store this file in the dataset. HF expects files at root to be formatted
    // as a relative path.
    const datasetFilePath = folder ? `${folder}/${fileName}` : `./${fileName}`;

    // Extract repo info from dataset URL
    const repoUrl = (dataset.files_url ?? '').replace('hf://', '');
    const [, namespace, name] = repoUrl.split('/');

    // 1: Pre-upload
    await this.request.post(
      `${PLATFORM_BASE_URL}/v1/hf/api/datasets/${namespace}/${name}/preupload/main`,
      {
        data: {
          files: [
            {
              path: datasetFilePath,
              size: fileContents.length,
              sample: fileContents.subarray(0, 1024).toString('base64'), // Take first 1KB as sample
            },
          ],
        },
      }
    );

    // 2: Request LFS upload URL
    const fileHash = crypto.createHash('sha256').update(fileContents).digest('hex');
    const lfsBatchResponse = await this.request.post(
      `${PLATFORM_BASE_URL}/v1/hf/datasets/${namespace}/${name}.git/info/lfs/objects/batch`,
      {
        data: {
          operation: 'upload',
          transfers: ['basic', 'multipart'],
          hash_algo: 'sha_256',
          ref: {
            name: 'main',
          },
          objects: [
            {
              oid: fileHash,
              size: fileContents.length,
            },
          ],
        },
        headers: {
          'Content-Type': 'application/vnd.git-lfs+json',
          Accept: 'application/vnd.git-lfs+json',
        },
      }
    );

    // 3: Upload file content to LFS storage
    const lfsBatchData = (await lfsBatchResponse.json()) as {
      objects: Array<{
        oid: string;
        size: number;
        actions?: {
          upload?: {
            href: string;
            header?: Record<string, string>;
          };
        };
      }>;
    };

    // Check if we need to upload (server may already have the file cached)
    const uploadAction = lfsBatchData.objects?.[0]?.actions?.upload;
    if (uploadAction) {
      await this.request.put(uploadAction.href, {
        data: fileContents,
        headers: {
          'Content-Type': 'application/octet-stream',
          ...uploadAction.header,
        },
      });
    }

    // 4: Commit the changes
    // HF expects request body for commits as JSONL
    const commitData = [
      { key: 'header', value: { summary: 'Add 1 files' } },
      {
        key: 'lfsFile',
        value: {
          path: datasetFilePath,
          algo: 'sha256',
          size: fileContents.length,
          oid: fileHash,
        },
      },
    ]
      .map((line) => JSON.stringify(line))
      .join('\n');

    await this.request.post(
      `${PLATFORM_BASE_URL}/v1/hf/api/datasets/${namespace}/${name}/commit/main`,
      {
        data: commitData,
      }
    );
  }
}
