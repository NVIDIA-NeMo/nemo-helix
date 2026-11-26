// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SERIES_COLORS } from '@nemo/common/src/components/charts/tokens';
import {
  SCORE_TIER_COLORS,
  type ScoreTier,
  UNAVAILABLE_COLOR,
} from '@nemo/common/src/components/ScoreGauge/consts';
import type {
  AutocorrelationProfile,
  Distribution,
  PIIReplacement,
  PIIReplacementColumn,
  PIIReplayEntry,
  PIIReplayMetric,
} from '@studio/routes/SafeSynthesizerJobReportRoute/reportDataTypes';

export const DATASET_COLORS = {
  training: SERIES_COLORS[0],
  synthetic: SERIES_COLORS[3],
  holdout: SERIES_COLORS[6],
} as const;

export const DATASET_LABELS = {
  training: 'Training Data',
  synthetic: 'Synthetic Data',
  holdout: 'Holdout Data',
} as const;

export const HEAT_COLORS = {
  training: 'var(--text-color-accent-blue)',
  synthetic: 'var(--text-color-accent-yellow)',
  difference: 'var(--text-color-accent-red)',
} as const;

export const GRADES_BEST_FIRST = Object.keys(SCORE_TIER_COLORS) as Exclude<
  ScoreTier,
  'Unavailable'
>[];

export const gradeColor = (grade: string): string =>
  grade in SCORE_TIER_COLORS
    ? SCORE_TIER_COLORS[grade as keyof typeof SCORE_TIER_COLORS]
    : UNAVAILABLE_COLOR;

/** Shades `color` by `intensity` (0 to 1) against whatever surface the chart sits on, so it reads in both themes. */
export const heatColor = (color: string, intensity: number): string => {
  const clamped = Math.min(Math.max(Number.isFinite(intensity) ? intensity : 0, 0), 1);
  return `color-mix(in srgb, ${color} ${Math.round(clamped * 100)}%, transparent)`;
};

export const finiteValues = (values: readonly (number | null)[]): number[] =>
  values.filter((value): value is number => value !== null && Number.isFinite(value));

export const formatPercent = (value: number | null, digits = 2): string =>
  value === null ? '—' : `${value.toFixed(digits)}%`;

export interface CategoryRow {
  label: string;
  training: number | null;
  synthetic: number | null;
}

export const categoryRows = (distribution: Distribution): CategoryRow[] =>
  distribution.labels.map((label, index) => ({
    label,
    training: distribution.training_percent[index] ?? null,
    synthetic: distribution.synthetic_percent[index] ?? null,
  }));

export interface BinRow extends CategoryRow {
  edge: number;
}

/** Rows for step areas drawn over bin edges: one per bin start, plus the closing edge repeating the last bin. */
export const binnedRows = (distribution: Distribution): BinRow[] => {
  const edges = distribution.bin_edges ?? [];
  const rows = categoryRows(distribution).map((row, index) => ({ ...row, edge: edges[index] }));
  const last = rows.at(-1);
  if (last === undefined || edges.length <= rows.length) {
    return rows;
  }
  return [...rows, { ...last, edge: edges[rows.length] }];
};

export const hasSyntheticValues = (distribution: Distribution): boolean =>
  distribution.synthetic_percent.some((value) => value !== null && value !== 0);

/** Linear-interpolated quantile of sorted values, as numpy's default. */
export const quantile = (sorted: readonly number[], q: number): number => {
  if (sorted.length === 0) {
    return Number.NaN;
  }
  const position = (sorted.length - 1) * q;
  const low = Math.floor(position);
  const high = Math.min(low + 1, sorted.length - 1);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
};

const MAX_AUTO_BINS = 100;

/** Shared bin edges for two samples, like numpy's `bins="auto"` capped at 100 bins. */
export const autoBinEdges = (first: readonly number[], second: readonly number[]): number[] => {
  const values = [...first, ...second].sort((a, b) => a - b);
  if (values.length === 0) {
    return [0, 1];
  }
  const low = values[0];
  const high = values[values.length - 1];
  if (low === high) {
    return [low, low + 1];
  }
  const range = high - low;
  const sturgesWidth = range / (Math.log2(values.length) + 1);
  const iqr = quantile(values, 0.75) - quantile(values, 0.25);
  const fdWidth = (2 * iqr) / Math.cbrt(values.length);
  const width = fdWidth > 0 ? Math.min(fdWidth, sturgesWidth) : sturgesWidth;
  const count = Math.min(Math.max(Math.ceil(range / width), 1), MAX_AUTO_BINS);
  return Array.from({ length: count + 1 }, (_, index) => low + (range * index) / count);
};

export type HistogramNorm = 'percent' | 'density';

/** Share of `values` in each bin; the last bin includes its upper edge, as numpy's histogram. */
export const histogram = (
  values: readonly number[],
  edges: readonly number[],
  norm: HistogramNorm
): number[] => {
  const bins = edges.length - 1;
  const counts = new Array<number>(Math.max(bins, 0)).fill(0);
  for (const value of values) {
    if (value < edges[0] || value > edges[bins]) {
      continue;
    }
    let index = edges.findIndex((edge, i) => i > 0 && value < edge) - 1;
    if (index < 0) {
      index = bins - 1;
    }
    counts[index] += 1;
  }
  const total = values.length;
  return counts.map((count, index) => {
    if (total === 0) {
      return 0;
    }
    if (norm === 'percent') {
      return (count / total) * 100;
    }
    return count / total / (edges[index + 1] - edges[index]);
  });
};

export interface HistogramRow {
  edge: number;
  label: string;
  training: number;
  synthetic: number;
}

export const formatNumber = (value: number, digits = 3): string =>
  Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(digits)));

/** Overlaid histogram rows over shared edges, closed with a repeat of the last bin for step areas. */
export const overlaidHistogramRows = (
  training: readonly number[],
  synthetic: readonly number[],
  edges: readonly number[],
  norm: HistogramNorm
): HistogramRow[] => {
  const trainingBins = histogram(training, edges, norm);
  const syntheticBins = histogram(synthetic, edges, norm);
  const rows = trainingBins.map((value, index) => ({
    edge: edges[index],
    label: `[${formatNumber(edges[index])}, ${formatNumber(edges[index + 1])})`,
    training: value,
    synthetic: syntheticBins[index],
  }));
  const last = rows.at(-1);
  if (last === undefined) {
    return rows;
  }
  return [...rows, { ...last, edge: edges[edges.length - 1] }];
};

export interface ScatterPoint {
  x: number;
  y: number;
}

export const toPoints = (
  xs: readonly (number | null)[],
  ys: readonly (number | null)[]
): ScatterPoint[] => {
  const points: ScatterPoint[] = [];
  xs.forEach((x, index) => {
    const y = ys[index];
    if (x !== null && y !== null && y !== undefined && Number.isFinite(x) && Number.isFinite(y)) {
      points.push({ x, y });
    }
  });
  return points;
};

export const pointsDomain = (
  points: readonly ScatterPoint[],
  axis: 'x' | 'y'
): [number, number] => {
  const values = points.map((point) => point[axis]);
  if (values.length === 0) {
    return [0, 1];
  }
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = (high - low || 1) * 0.05;
  return [low - pad, high + pad];
};

/** Deterministic pseudo-random numbers in [0, 1), so jittered dots stay put across renders. */
export const seededRandom = (seed: number): (() => number) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

const LABEL_MAX_LENGTH = 10;
const LABEL_EDGE_LENGTH = 3;

export const shortenLabel = (name: string): string => {
  if (name.length <= LABEL_MAX_LENGTH) {
    return name;
  }
  return `${name.slice(0, LABEL_EDGE_LENGTH)}...${name.slice(-LABEL_EDGE_LENGTH)}`;
};

export const groupLabel = (group: string | null): string => (group === null ? 'all rows' : group);

const mean = (values: readonly number[]): number =>
  values.reduce((sum, value) => sum + value, 0) / values.length;

/** Columns from lowest to highest mean pair similarity, names breaking ties. */
export const orderAutocorrelationColumns = (
  profiles: readonly AutocorrelationProfile[]
): string[] => {
  const scores = new Map<string, number[]>();
  for (const profile of profiles) {
    scores.set(profile.column, [...(scores.get(profile.column) ?? []), profile.similarity]);
  }
  return [...scores.keys()].sort((a, b) => {
    const difference = mean(scores.get(a) ?? []) - mean(scores.get(b) ?? []);
    if (difference !== 0) {
      return difference;
    }
    return a.localeCompare(b);
  });
};

const lagMatrix = (rows: readonly (readonly (number | null)[])[]): (number | null)[][] => {
  const width = Math.max(0, ...rows.map((row) => row.length));
  return rows.map((row) => Array.from({ length: width }, (_, lag) => row[lag] ?? null));
};

export interface LagQuantileRow {
  lag: number;
  trainingMedian: number;
  trainingBand: [number, number];
  syntheticMedian?: number;
  syntheticBand?: [number, number];
}

const lagQuantiles = (rows: readonly (readonly (number | null)[])[]) =>
  lagMatrix(rows)
    .reduce<number[][]>((columns, row) => {
      row.forEach((value, lag) => {
        columns[lag] = columns[lag] ?? [];
        if (value !== null && Number.isFinite(value)) {
          columns[lag].push(value);
        }
      });
      return columns;
    }, [])
    .map((values, index) => {
      const sorted = [...values].sort((a, b) => a - b);
      return {
        lag: index + 1,
        count: sorted.length,
        lower: quantile(sorted, 0.25),
        median: quantile(sorted, 0.5),
        upper: quantile(sorted, 0.75),
      };
    });

/** Median and interquartile band of training and synthetic autocorrelation per lag for one column. */
export const autocorrelationSummaryRows = (
  profiles: readonly AutocorrelationProfile[],
  column: string
): LagQuantileRow[] => {
  const columnProfiles = profiles.filter((profile) => profile.column === column);
  const training = lagQuantiles(columnProfiles.map((profile) => profile.training_acf));
  const synthetic = lagQuantiles(columnProfiles.map((profile) => profile.synthetic_acf));
  return training
    .filter((row) => row.count > 0)
    .map((row) => {
      const match = synthetic.find((candidate) => candidate.lag === row.lag);
      const hasSynthetic = match !== undefined && match.count > 0;
      return {
        lag: row.lag,
        trainingMedian: row.median,
        trainingBand: [row.lower, row.upper],
        syntheticMedian: hasSynthetic ? match.median : undefined,
        syntheticBand: hasSynthetic ? [match.lower, match.upper] : undefined,
      };
    });
};

export interface LagDifferenceRow {
  lag: number;
  difference: number;
  pairs: number;
}

/** Mean absolute training and synthetic autocorrelation difference per lag, over pairs finite at that lag. */
export const lagDifferenceRows = (
  profiles: readonly AutocorrelationProfile[]
): LagDifferenceRow[] => {
  const training = lagMatrix(profiles.map((profile) => profile.training_acf));
  const synthetic = lagMatrix(profiles.map((profile) => profile.synthetic_acf));
  const width = training[0]?.length ?? 0;
  return Array.from({ length: width }, (_, lag) => {
    const differences = training
      .map((row, index) => {
        const a = row[lag];
        const b = synthetic[index]?.[lag];
        if (a === null || b === null || b === undefined) {
          return null;
        }
        return Math.abs(a - b);
      })
      .filter((value): value is number => value !== null && Number.isFinite(value));
    return {
      lag: lag + 1,
      difference: differences.length ? mean(differences) : Number.NaN,
      pairs: differences.length,
    };
  }).filter((row) => row.pairs > 0);
};

export interface PiiEntityRow {
  piiType: string;
  found: number;
  unique: number | null;
  synthetic: number | null;
  replayed: number | null;
  replayPercentage: number | null;
}

export interface PiiColumnGroup {
  columnName: string;
  replacement: PIIReplacementColumn | null;
  entities: PiiEntityRow[];
}

const piiEntity = (piiType: string, found: number, replay?: PIIReplayEntry): PiiEntityRow => ({
  piiType,
  found,
  unique: replay?.unique_training_data ?? null,
  synthetic: replay?.total_synthetic_data ?? null,
  replayed: replay?.unique_synthetic_data ?? null,
  replayPercentage: replay?.unique_synthetic_data_percentage ?? null,
});

/** Each replaced column joined with its replay results, one entry per PII type found, most frequent first. */
export const piiColumnGroups = (
  pii: PIIReplacement,
  replay: PIIReplayMetric | null
): PiiColumnGroup[] => {
  const replayByColumn = new Map<string, PIIReplayEntry[]>();
  for (const entry of replay?.entries ?? []) {
    replayByColumn.set(entry.column_name, [
      ...(replayByColumn.get(entry.column_name) ?? []),
      entry,
    ]);
  }
  const groups: PiiColumnGroup[] = pii.columns.map((column) => {
    const replayed = replayByColumn.get(column.column_name) ?? [];
    replayByColumn.delete(column.column_name);
    const counts = Object.keys(column.entity_counts).length
      ? Object.entries(column.entity_counts)
      : [[column.entity_type, 0] as const];
    const entities = [...counts]
      .sort((a, b) => b[1] - a[1])
      .map(([piiType, found]) =>
        piiEntity(
          piiType,
          found,
          replayed.find((entry) => entry.pii_type === piiType)
        )
      );
    return { columnName: column.column_name, replacement: column, entities };
  });
  for (const [columnName, entries] of replayByColumn) {
    groups.push({
      columnName,
      replacement: null,
      entities: entries.map((entry) => piiEntity(entry.pii_type, entry.total_training_data, entry)),
    });
  }
  return groups;
};

/** Why PII replay results are missing or partial, or null when replay ran in full. */
export const piiReplayNote = (replay: PIIReplayMetric | null): string | null => {
  if (replay === null) {
    return null;
  }
  if (replay.skipped_reason) {
    if (replay.entries.length) {
      return `Only the columns in evaluation.pii_replay_columns were checked: ${replay.skipped_reason}.`;
    }
    return `PII replay was not run: ${replay.skipped_reason}.`;
  }
  if (replay.enabled === false) {
    return 'PII replay was turned off with evaluation.pii_replay_enabled.';
  }
  return null;
};

export const plural = (count: number, noun: string): string =>
  `${count} ${noun}${count === 1 ? '' : 's'}`;

export const piiHeadline = (
  pii: PIIReplacement,
  groups: readonly PiiColumnGroup[],
  replayNote: string | null
): string => {
  if (pii.columns.length) {
    return `${plural(pii.columns.length, 'column')} replaced`;
  }
  if (groups.length) {
    return `PII in ${plural(groups.length, 'column')}, not replaced`;
  }
  return replayNote ? 'Replay not run' : 'No PII found';
};
