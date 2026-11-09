// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { asRecord } from '@studio/util/guards';

/**
 * Lightweight reader for the Data Designer `builder_config.json` artifact.
 *
 * The file is the serialized `BuilderConfig` Pydantic model
 * (`{ data_designer: DataDesignerConfig, library_version }`). Rather than mirror
 * the full discriminated-union config (that's the schema-inspector ticket), this
 * extracts just the important fields for an at-a-glance summary, parsing
 * defensively so an unexpected shape degrades gracefully instead of throwing.
 *
 * Filename inside the artifacts fileset.
 */
export const BUILDER_CONFIG_FILENAME = 'builder_config.json';

export interface BuilderConfigColumnSummary {
  readonly name: string;
  readonly type: string;
  readonly modelAlias?: string;
}

export interface BuilderConfigModelSummary {
  readonly alias: string;
  readonly model: string;
  readonly provider?: string;
}

export interface BuilderConfigSeedSummary {
  readonly type: string;
  readonly samplingStrategy?: string;
}

export interface BuilderConfigSummary {
  readonly columnCount: number;
  readonly columns: BuilderConfigColumnSummary[];
  readonly columnTypeBreakdown: Array<{ type: string; count: number }>;
  readonly models: BuilderConfigModelSummary[];
  readonly seed?: BuilderConfigSeedSummary;
  readonly constraintCount: number;
  readonly profilerCount: number;
  readonly processorNames: string[];
  readonly libraryVersion?: string;
}

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined;

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const UNNAMED = '(unnamed)';

/**
 * Parses raw `builder_config.json` contents into a {@link BuilderConfigSummary}.
 * Returns `null` when the payload is not a recognizable builder config (missing
 * the top-level `data_designer` object).
 */
export const summarizeBuilderConfig = (raw: unknown): BuilderConfigSummary | null => {
  const root = asRecord(raw);
  const dataDesigner = asRecord(root?.data_designer);
  if (!root || !dataDesigner) {
    return null;
  }

  const columns: BuilderConfigColumnSummary[] = asArray(dataDesigner.columns).map((column) => {
    const record = asRecord(column) ?? {};
    return {
      name: asString(record.name) ?? UNNAMED,
      type: asString(record.column_type) ?? 'unknown',
      modelAlias: asString(record.model_alias),
    };
  });

  const breakdownCounts = new Map<string, number>();
  for (const column of columns) {
    breakdownCounts.set(column.type, (breakdownCounts.get(column.type) ?? 0) + 1);
  }
  const columnTypeBreakdown = [...breakdownCounts.entries()]
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));

  const models: BuilderConfigModelSummary[] = asArray(dataDesigner.model_configs).map((model) => {
    const record = asRecord(model) ?? {};
    return {
      alias: asString(record.alias) ?? UNNAMED,
      model: asString(record.model) ?? '—',
      provider: asString(record.provider),
    };
  });

  const seedConfig = asRecord(dataDesigner.seed_config);
  const seedSource = asRecord(seedConfig?.source);
  const seed: BuilderConfigSeedSummary | undefined = seedConfig
    ? {
        type: asString(seedSource?.seed_type) ?? 'unknown',
        samplingStrategy: asString(seedConfig.sampling_strategy),
      }
    : undefined;

  const processorNames = asArray(dataDesigner.processors).map(
    (processor) => asString(asRecord(processor)?.name) ?? UNNAMED
  );

  return {
    columnCount: columns.length,
    columns,
    columnTypeBreakdown,
    models,
    seed,
    constraintCount: asArray(dataDesigner.constraints).length,
    profilerCount: asArray(dataDesigner.profilers).length,
    processorNames,
    libraryVersion: asString(root.library_version),
  };
};

/** Formats the column-type breakdown as a compact label (e.g. `3 sampler, 2 llm-text`). */
export const formatColumnTypeBreakdown = (summary: BuilderConfigSummary): string =>
  summary.columnTypeBreakdown.map(({ type, count }) => `${count} ${type}`).join(', ');
