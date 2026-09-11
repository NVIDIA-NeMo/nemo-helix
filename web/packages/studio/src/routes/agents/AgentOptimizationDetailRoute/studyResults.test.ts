// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { agentsListOptimizeJobResults } from '@nemo/sdk/generated/agents/agents';
import { filesDownloadFile, filesListFilesetFiles } from '@nemo/sdk/generated/platform/files';
import {
  fetchStudyResults,
  parseDurationSeconds,
} from '@studio/routes/agents/AgentOptimizationDetailRoute/studyResults';

vi.mock('@nemo/sdk/generated/agents/agents', () => ({
  agentsListOptimizeJobResults: vi.fn(),
}));
vi.mock('@nemo/sdk/generated/platform/files', () => ({
  filesDownloadFile: vi.fn(),
  filesListFilesetFiles: vi.fn(),
}));

const workspace = 'default';
const jobName = 'brevity-sweep-3';
const resultDir = 'results/attempt-1/optimizer_results';

const SUMMARY = JSON.stringify({
  status: 'completed',
  n_trials: 3,
  best_trial: 6,
  best_params: { temperature: 0.2 },
  best_values: [4.19, 742],
  metric_names: ['avg_score', 'avg_tokens'],
});

const TRIALS_CSV = [
  'number,state,datetime_start,datetime_complete,duration,values_avg_score,values_avg_tokens,params_prompt,params_temperature,rep_scores,pareto_optimal',
  '6,COMPLETE,2026-09-11T10:00:00,2026-09-11T10:00:12,0:00:12.500000,4.19,742,v3-concise,0.2,"[4.1, 4.2, 4.27]",True',
  '3,COMPLETE,2026-09-11T10:00:12,2026-09-11T10:00:21,0:00:09,4.14,610,v2-structured,0.2,"[4.1, 4.18]",True',
  '4,FAIL,2026-09-11T10:00:21,,,,,v2-structured,0.9,null,False',
].join('\n');

const mockResults = (artifactUrl: string) =>
  vi.mocked(agentsListOptimizeJobResults).mockResolvedValue({
    data: [
      {
        name: 'optimizer_results',
        job: jobName,
        workspace,
        artifact_url: artifactUrl,
        artifact_storage_type: 'fileset',
      },
    ],
  } as Awaited<ReturnType<typeof agentsListOptimizeJobResults>>);

const mockFiles = (paths: string[]) =>
  vi.mocked(filesListFilesetFiles).mockResolvedValue({
    data: paths.map((path) => ({ path, size: 1, file_ref: path })),
  } as Awaited<ReturnType<typeof filesListFilesetFiles>>);

const mockDownloads = (byPath: Record<string, string>) =>
  vi
    .mocked(filesDownloadFile)
    .mockImplementation(async (_workspace, _fileset, path) => new Blob([byPath[path] ?? '']));

beforeEach(() => {
  vi.clearAllMocks();
});

describe('parseDurationSeconds', () => {
  it.each([
    ['0:00:12.500000', 12.5],
    ['0:01:09', 69],
    ['1:00:00', 3600],
    ['2 days, 0:00:30', 172_830],
  ])('parses %s', (input, expected) => {
    expect(parseDurationSeconds(input)).toBe(expected);
  });

  it('returns null for the empty duration of a trial that never completed', () => {
    expect(parseDurationSeconds('')).toBeNull();
    expect(parseDurationSeconds(undefined)).toBeNull();
  });
});

describe('fetchStudyResults', () => {
  it('parses the summary and every trial row', async () => {
    mockResults(`fileset://${workspace}/study-artifacts#${resultDir}`);
    mockFiles([`${resultDir}/study_summary.json`, `${resultDir}/trials_dataframe_params.csv`]);
    mockDownloads({
      [`${resultDir}/study_summary.json`]: SUMMARY,
      [`${resultDir}/trials_dataframe_params.csv`]: TRIALS_CSV,
    });

    const results = await fetchStudyResults(workspace, jobName);

    expect(results?.summary).toEqual({
      nTrials: 3,
      bestTrial: 6,
      metricNames: ['avg_score', 'avg_tokens'],
      bestValues: [4.19, 742],
    });
    expect(results?.metricNames).toEqual(['avg_score', 'avg_tokens']);
    expect(results?.trials).toHaveLength(3);

    const [best] = results?.trials ?? [];
    expect(best).toEqual({
      number: 6,
      state: 'COMPLETE',
      durationSeconds: 12.5,
      paretoOptimal: true,
      metrics: [
        { name: 'avg_score', value: 4.19 },
        { name: 'avg_tokens', value: 742 },
      ],
      // rep_scores and the datetime columns are not parameters, so they stay out of params.
      params: [
        { name: 'prompt', value: 'v3-concise' },
        { name: 'temperature', value: '0.2' },
      ],
    });
  });

  it('keeps a failed trial, with null metrics and no duration', async () => {
    mockResults(`fileset://${workspace}/study-artifacts#${resultDir}`);
    mockFiles([`${resultDir}/study_summary.json`, `${resultDir}/trials_dataframe_params.csv`]);
    mockDownloads({
      [`${resultDir}/study_summary.json`]: SUMMARY,
      [`${resultDir}/trials_dataframe_params.csv`]: TRIALS_CSV,
    });

    const results = await fetchStudyResults(workspace, jobName);
    const failed = results?.trials.find((trial) => trial.number === 4);

    expect(failed?.state).toBe('FAIL');
    expect(failed?.durationSeconds).toBeNull();
    expect(failed?.paretoOptimal).toBe(false);
    expect(failed?.metrics).toEqual([
      { name: 'avg_score', value: null },
      { name: 'avg_tokens', value: null },
    ]);
  });

  it('returns null when the job has published no study artifacts', async () => {
    mockResults(`fileset://${workspace}/study-artifacts#${resultDir}`);
    mockFiles([`${resultDir}/optimized_config.yml`]);

    await expect(fetchStudyResults(workspace, jobName)).resolves.toBeNull();
    expect(filesDownloadFile).not.toHaveBeenCalled();
  });

  it('reads trials even when the summary is missing, falling back to CSV metric order', async () => {
    mockResults(`fileset://${workspace}/study-artifacts#${resultDir}`);
    mockFiles([`${resultDir}/trials_dataframe_params.csv`]);
    mockDownloads({ [`${resultDir}/trials_dataframe_params.csv`]: TRIALS_CSV });

    const results = await fetchStudyResults(workspace, jobName);

    expect(results?.summary).toBeNull();
    expect(results?.metricNames).toEqual(['avg_score', 'avg_tokens']);
    expect(results?.trials).toHaveLength(3);
  });
});
