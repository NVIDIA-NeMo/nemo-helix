// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SamplingStrategy, type CreateJobRequest } from '@nemo/sdk/generated/data-designer/schema';

export interface TransformJobRequestInput {
  readonly jobName: string;
  readonly processorName: string;
  /** Workspace owning the source fileset. */
  readonly filesetWorkspace: string;
  readonly filesetName: string;
  /** Path of the source file inside that fileset. */
  readonly filePath: string;
  /** Rows to read from the source file. */
  readonly numRecords: number;
  readonly template: Record<string, unknown>;
  /**
   * When set, the job declares this column as a UUID sampler so the template can
   * reference it. Used to give each row an identifier the source file lacks.
   */
  readonly generatedIdColumn?: string;
}

/**
 * `FilesetFileSeedSource` addresses a single file as `<workspace>/<fileset>#<path>`.
 * The workspace prefix is optional server-side but we always send it, since the
 * fileset a job wrote to is not necessarily the workspace the user is browsing.
 */
export const buildSeedPath = (
  filesetWorkspace: string,
  filesetName: string,
  filePath: string
): string => `${filesetWorkspace}/${filesetName}#${filePath}`;

/**
 * Builds a transform job: the engine resolves the seed file's own columns into
 * passthrough seed columns, then the `schema_transform` processor rewrites each
 * row into `processors-files/<processorName>/`.
 *
 * Only `generatedIdColumn` is ever declared as a column, and a sampler costs no
 * inference, so the job needs no model either way.
 *
 * `num_records` must not exceed the source file's row count — the seed reader
 * restarts at the top when it runs dry, which silently duplicates rows.
 */
export const buildTransformJobRequest = ({
  jobName,
  processorName,
  filesetWorkspace,
  filesetName,
  filePath,
  numRecords,
  template,
  generatedIdColumn,
}: TransformJobRequestInput): CreateJobRequest => ({
  name: jobName,
  spec: {
    num_records: numRecords,
    config: {
      columns: generatedIdColumn
        ? [
            {
              name: generatedIdColumn,
              column_type: 'sampler',
              sampler_type: 'uuid',
              params: { short_form: true },
            },
          ]
        : [],
      seed_config: {
        source: {
          seed_type: 'nhx',
          path: buildSeedPath(filesetWorkspace, filesetName, filePath),
        },
        sampling_strategy: SamplingStrategy.ordered,
      },
      processors: [
        {
          processor_type: 'schema_transform',
          name: processorName,
          template,
        },
      ],
    },
  },
});
