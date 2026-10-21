// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cancelRunJob, deleteRunWithJob } from '@agent-hardener/api/runJobs';

const { request } = vi.hoisted(() => ({ request: vi.fn() }));
vi.mock('@agent-hardener/api/fetcher', () => ({ customFetch: request }));

const missing = { isAxiosError: true, response: { status: 404 } };
const base = '/apis/agent-hardener/v2/workspaces/default';
const job = 'job-with-no-kind-in-its-name';
const run = { name: 'run', job_id: job };

beforeEach(() => {
  request.mockReset();
});

it('cancels a war-game job without probing the sibling collection', async () => {
  request.mockResolvedValue({});
  await cancelRunJob({ workspace: 'default', name: job });
  expect(request.mock.calls.map(([config]) => [config.method, config.url])).toEqual([
    ['POST', `${base}/jobs/${job}/cancel`],
  ]);
});

it('cancels a synthesis job through its own collection after a war-game 404', async () => {
  request.mockRejectedValueOnce(missing).mockResolvedValueOnce({});
  await cancelRunJob({ workspace: 'default', name: job });
  expect(request.mock.calls.map(([config]) => [config.method, config.url])).toEqual([
    ['POST', `${base}/jobs/${job}/cancel`],
    ['POST', `${base}/synth-benign/jobs/${job}/cancel`],
  ]);
});

it.each([401, 403, 409, 500])('does not retry a %s mutation failure in another collection', async (status) => {
  const error = { isAxiosError: true, response: { status } };
  request.mockRejectedValue(error);
  await expect(cancelRunJob({ workspace: 'default', name: job })).rejects.toBe(error);
  expect(request).toHaveBeenCalledTimes(1);
});

it('does not retry an ambiguous network failure that may have completed the mutation', async () => {
  const error = new Error('connection reset');
  request.mockRejectedValue(error);
  await expect(cancelRunJob({ workspace: 'default', name: job })).rejects.toBe(error);
  expect(request).toHaveBeenCalledTimes(1);
});

it('deletes a war-game job before its run record', async () => {
  request.mockResolvedValue(undefined);
  await deleteRunWithJob('default', run);
  expect(request.mock.calls.map(([config]) => [config.method, config.url])).toEqual([
    ['DELETE', `${base}/jobs/${job}`],
    ['DELETE', `${base}/runs/run`],
  ]);
});

it('deletes a synthesis job before its run record without swallowing the first 404', async () => {
  request.mockRejectedValueOnce(missing).mockResolvedValue(undefined);
  await deleteRunWithJob('default', run);
  expect(request.mock.calls.map(([config]) => [config.method, config.url])).toEqual([
    ['DELETE', `${base}/jobs/${job}`],
    ['DELETE', `${base}/synth-benign/jobs/${job}`],
    ['DELETE', `${base}/runs/run`],
  ]);
});

it.each([403, 404, 500])('keeps the run record and reports synthesis deletion failure %s', async (status) => {
  const error = { isAxiosError: true, response: { status } };
  request.mockRejectedValueOnce(missing).mockRejectedValueOnce(error);
  await expect(deleteRunWithJob('default', run)).rejects.toBe(error);
  expect(request).toHaveBeenCalledTimes(2);
  expect(request.mock.calls.some(([config]) => config.url.includes('/runs/'))).toBe(false);
});

it('deletes a local run with no linked platform job', async () => {
  request.mockResolvedValue(undefined);
  await deleteRunWithJob('default', { name: 'local-run' });
  expect(request).toHaveBeenCalledWith(expect.objectContaining({ method: 'DELETE', url: `${base}/runs/local-run` }));
});
