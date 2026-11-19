// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { TasksetRunStatus } from '@nemo/sdk/generated/evals/schema';
import {
  readScoringBundle,
  loadSavedComparison,
} from '@studio/routes/evaluation/SkillComparisonRoute/history';
import { Blob as NodeBlob } from 'node:buffer';
import { gzipSync } from 'node:zlib';

function archive(files: Record<string, string>) {
  const parts: Buffer[] = [];
  for (const [name, body] of Object.entries(files)) {
    const header = Buffer.alloc(512);
    header.write(name);
    header.write(Buffer.byteLength(body).toString(8).padStart(11, '0'), 124);
    parts.push(
      header,
      Buffer.from(body),
      Buffer.alloc((512 - (Buffer.byteLength(body) % 512)) % 512)
    );
  }
  parts.push(Buffer.alloc(1024));
  return new NodeBlob([gzipSync(Buffer.concat(parts))]) as Blob;
}

describe('saved comparison bundles', () => {
  it('reads summary and per-trial scores without extracting unrelated members', async () => {
    const result = await readScoringBundle(
      archive({
        'agent-eval-results/summary.json': '{"trial_count":1}',
        'agent-eval-results/scores.jsonl': '{"task_id":"task"}\n',
        '../../ignored.html': '<script>ignored</script>',
      })
    );
    expect(result).toEqual({ summary: { trial_count: 1 }, scores: [{ task_id: 'task' }] });
  });
  it('rejects bundles without per-trial evidence', async () => {
    await expect(readScoringBundle(archive({ 'summary.json': '{}' }))).rejects.toThrow('missing');
  });
  it('rejects duplicate scorer members rather than choosing arbitrary evidence', async () => {
    await expect(
      readScoringBundle(
        archive({
          'one/summary.json': '{}',
          'two/summary.json': '{}',
          'scores.jsonl': '',
        })
      )
    ).rejects.toThrow('Ambiguous');
  });
  it('rejects malformed JSON', async () => {
    await expect(
      readScoringBundle(
        archive({
          'summary.json': '{',
          'scores.jsonl': '{}',
        })
      )
    ).rejects.toThrow();
  });
  it('rejects expanded bundles larger than the limit', async () => {
    await expect(
      readScoringBundle(
        archive({
          'large.txt': 'x'.repeat(32 * 1024 * 1024 + 1),
        })
      )
    ).rejects.toThrow('Expanded scoring bundle');
  });
  it('rejects selecting the same execution twice before making requests', async () => {
    const run = { run_id: 'same' } as TasksetRunStatus;
    await expect(loadSavedComparison('default', run, run)).rejects.toThrow('distinct');
  });
  it('rejects differing pinned tasksets before making requests', async () => {
    await expect(
      loadSavedComparison(
        'default',
        { run_id: 'before', taskset_ref: 'a' } as TasksetRunStatus,
        { run_id: 'after', taskset_ref: 'b' } as TasksetRunStatus
      )
    ).rejects.toThrow('same pinned taskset');
  });
});
