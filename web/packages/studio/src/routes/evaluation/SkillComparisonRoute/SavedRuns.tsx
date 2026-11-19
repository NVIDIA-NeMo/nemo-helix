// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  evalsGetTasksetRun,
  evalsListTasksetRuns,
} from '@nemo/sdk/generated/evals/evals-plugin-taskset-run-routes';
import type { TasksetRunStatus } from '@nemo/sdk/generated/evals/schema';
import { Button, Flex, Modal, Stack, Table, Text } from '@nvidia/foundations-react-core';
import { loadSavedComparison } from '@studio/routes/evaluation/SkillComparisonRoute/history';
import { ARM_LABELS } from '@studio/routes/evaluation/SkillComparisonRoute/presentation';
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
  const [baselineNotice, setBaselineNotice] = useState<string>();
  const show = async () => {
    setOpen(true);
    setBaseline(undefined);
    setTreatment(undefined);
    setBaselineNotice(undefined);
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
  const chooseTreatment = async (run: TasksetRunStatus) => {
    setTreatment(run);
    setBaselineNotice(undefined);
    setError(undefined);
    const linkedId = run.baseline_run_id;
    if (!linkedId) {
      setBaseline((current) =>
        current?.taskset_ref === run.taskset_ref && current.run_id !== run.run_id
          ? current
          : undefined
      );
      return;
    }
    setBaseline(undefined);
    setLoading(true);
    try {
      const linked =
        runs.find((candidate) => candidate.run_id === linkedId) ??
        (await evalsGetTasksetRun(workspace, linkedId));
      if (
        linked.phase !== 'completed' ||
        linked.result_state !== 'ready' ||
        linked.taskset_ref !== run.taskset_ref ||
        linked.run_id === run.run_id
      ) {
        setBaselineNotice(
          'The linked control has no compatible retained results. Choose a control below.'
        );
        return;
      }
      setBaseline(linked);
    } catch {
      setBaselineNotice('The linked control could not be loaded. Choose a control below.');
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
            Choose a variant. If it has a linked control, Studio selects it for you. Otherwise,
            choose a control from the same taskset. The list shows completed runs from the latest
            100 executions.
          </Text>
          <Flex gap="density-xl" className="flex-wrap">
            <Text>
              {ARM_LABELS.baseline}: {baseline?.run_id ?? 'Choose below'}
              {baseline && baseline.run_id === treatment?.baseline_run_id && ' (linked)'}
            </Text>
            <Text>
              {ARM_LABELS['with-skill']}: {treatment?.run_id ?? 'Choose below'}
            </Text>
          </Flex>
          {error && (
            <Text role="alert" className="text-error">
              {error}
            </Text>
          )}
          {baselineNotice && <Text role="status">{baselineNotice}</Text>}
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
                          onClick={() => {
                            setBaseline(run);
                            setBaselineNotice(undefined);
                          }}
                        >
                          {ARM_LABELS.baseline}
                        </Button>
                        <Button
                          size="small"
                          kind={treatment?.run_id === run.run_id ? 'primary' : 'secondary'}
                          disabled={loading}
                          onClick={() => void chooseTreatment(run)}
                        >
                          {ARM_LABELS['with-skill']}
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
              disabled={
                loading ||
                !baseline ||
                !treatment ||
                baseline.run_id === treatment.run_id ||
                baseline.taskset_ref !== treatment.taskset_ref
              }
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
