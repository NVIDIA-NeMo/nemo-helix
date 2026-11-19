// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { evalsListTasksetRuns } from '@nemo/sdk/generated/evals/evals-plugin-taskset-run-routes';
import type { TasksetRunStatus } from '@nemo/sdk/generated/evals/schema';
import { Button, Flex, Modal, Stack, Table, Text } from '@nvidia/foundations-react-core';
import { loadSavedComparison } from '@studio/routes/evaluation/SkillComparisonRoute/history';
import type { ComparisonReport } from '@studio/routes/evaluation/SkillComparisonRoute/report';
import { useState } from 'react';

interface SavedRunsProps {
  workspace: string;
  onLoad: (report: ComparisonReport) => void;
}

export function SavedRuns({ workspace, onLoad }: SavedRunsProps) {
  const [open, setOpen] = useState(false);
  const [runs, setRuns] = useState<TasksetRunStatus[]>([]);
  const [baseline, setBaseline] = useState<TasksetRunStatus>();
  const [treatment, setTreatment] = useState<TasksetRunStatus>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const show = async () => {
    setOpen(true);
    setLoading(true);
    setError(undefined);
    try {
      const page = await evalsListTasksetRuns(workspace, { page_size: 100 });
      setRuns(page.data.filter((run) => run.phase === 'completed' && run.result_state === 'ready'));
    } catch {
      setError('Could not load saved executions. Check the connection to this Helix deployment.');
    } finally {
      setLoading(false);
    }
  };
  const load = async () => {
    if (!baseline || !treatment) return;
    setLoading(true);
    setError(undefined);
    try {
      const report = await loadSavedComparison(workspace, baseline, treatment);
      onLoad(report);
      setOpen(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not load this comparison.');
    } finally {
      setLoading(false);
    }
  };
  return (
    <>
      <Button kind="secondary" onClick={() => void show()}>
        Load Saved Runs
      </Button>
      <Modal
        open={open}
        onOpenChange={(value) => {
          if (!loading) setOpen(value);
        }}
        slotHeading="Compare Saved Executions"
        className="max-w-5xl w-[90vw]"
      >
        <Stack gap="density-lg">
          <Text className="text-secondary">
            Choose a baseline without skills and a with-skill execution of the same taskset. This
            loads retained results; it does not start an evaluation. Showing completed runs from the
            latest 100 executions. Uses the latest completed scoring replay matching each trial ID;
            scoring job names are shown in Evidence.
          </Text>
          <Flex gap="density-xl" className="flex-wrap">
            <Text>Baseline: {baseline?.run_id ?? 'Choose below'}</Text>
            <Text>With skill: {treatment?.run_id ?? 'Choose below'}</Text>
          </Flex>
          {error && (
            <Text role="alert" className="text-error">
              {error}
            </Text>
          )}
          {loading && <Text role="status">Loading saved results…</Text>}
          <div className="max-h-[45vh] overflow-auto">
            <Table
              aria-label="Saved executions"
              columns={['Execution', 'Started', 'Taskset', 'Compare as']}
              rows={runs.map((run) => ({
                id: run.run_id,
                cells: [
                  run.run_id,
                  new Date(run.accepted_at).toLocaleString(),
                  {
                    children: <span title={run.taskset_ref}>{run.taskset_ref.split('#')[0]}</span>,
                  },
                  {
                    children: (
                      <Flex gap="density-sm">
                        <Button
                          size="small"
                          kind={baseline?.run_id === run.run_id ? 'primary' : 'secondary'}
                          disabled={loading}
                          onClick={() => setBaseline(run)}
                        >
                          Baseline
                        </Button>
                        <Button
                          size="small"
                          kind={treatment?.run_id === run.run_id ? 'primary' : 'secondary'}
                          disabled={loading}
                          onClick={() => setTreatment(run)}
                        >
                          With Skill
                        </Button>
                      </Flex>
                    ),
                  },
                ],
              }))}
            />
          </div>
          {!loading && !runs.length && (
            <Text>No completed executions with retained results were found.</Text>
          )}
          <Flex justify="end">
            <Button
              disabled={loading || !baseline || !treatment || baseline.run_id === treatment.run_id}
              onClick={() => void load()}
            >
              Load Comparison
            </Button>
          </Flex>
        </Stack>
      </Modal>
    </>
  );
}
