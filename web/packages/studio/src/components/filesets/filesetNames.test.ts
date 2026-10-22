// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { filesRetrieveFileset } from '@nemo/sdk/generated/platform/files';
import type { FilesetOutput } from '@nemo/sdk/generated/platform/schema';
import {
  findAvailableFilesetName,
  findFreeFilesetName,
  retrieveFilesetIfPresent,
} from '@studio/components/filesets/filesetNames';

vi.mock('@nemo/sdk/generated/platform/files');

const mockRetrieve = vi.mocked(filesRetrieveFileset);

const notFound = () => Object.assign(new Error('not found'), { status: 404 });

/** Lookups find exactly the names in `taken`. */
const takenNames = (...taken: string[]) =>
  mockRetrieve.mockImplementation(async (_workspace, name) => {
    if (taken.includes(name)) return { name } as FilesetOutput;
    throw notFound();
  });

describe('filesetNames', () => {
  beforeEach(() => {
    mockRetrieve.mockReset();
  });

  describe('retrieveFilesetIfPresent', () => {
    it('returns null for a 404 and rethrows anything else', async () => {
      takenNames();
      await expect(retrieveFilesetIfPresent('default', 'x')).resolves.toBeNull();

      mockRetrieve.mockRejectedValue(Object.assign(new Error('boom'), { status: 500 }));
      await expect(retrieveFilesetIfPresent('default', 'x')).rejects.toThrow('boom');
    });
  });

  describe('findFreeFilesetName', () => {
    it('keeps the base name when it is free', async () => {
      takenNames();
      await expect(findFreeFilesetName('default', 'data')).resolves.toBe('data');
    });

    it('moves to the first free suffix when the base is taken', async () => {
      takenNames('data', 'data-2');
      await expect(findFreeFilesetName('default', 'data')).resolves.toBe('data-3');
    });

    it('reports no name rather than a taken one when every suffix is taken', async () => {
      takenNames('data', ...Array.from({ length: 19 }, (_, i) => `data-${i + 2}`));
      await expect(findFreeFilesetName('default', 'data')).resolves.toBeNull();
    });
  });

  describe('findAvailableFilesetName', () => {
    it('falls back to the last candidate once the limit is reached', async () => {
      takenNames('data-2', 'data-3');
      await expect(findAvailableFilesetName('default', 'data', 3)).resolves.toBe('data-3');
    });
  });
});
