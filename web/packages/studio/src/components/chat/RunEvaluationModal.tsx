// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { FormModal } from '@nemo/common/src/components/FormModal';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { Block, Stack, Text } from '@nvidia/foundations-react-core';
import { useCallback, useState, type FC } from 'react';

interface RunEvaluationModalProps {
  open: boolean;
  onClose: () => void;
  workspace: string;
  /** URNs of models currently in the Playground panels (deduped, non-null). */
  modelUrns: string[];
}

const EVAL_SETS = [
  { id: 'customer-support-golden', name: 'Customer-Support Golden Set (200 prompts)' },
  { id: 'internal-helpfulness-bench', name: 'Internal Helpfulness Bench (mini)' },
  { id: 'reasoning-subset', name: 'NeMo Eval — Reasoning Subset' },
];

const METRICS = [
  { id: 'llm-judge', name: 'LLM-as-judge (helpfulness)' },
  { id: 'exact-match', name: 'Exact-match accuracy' },
  { id: 'rouge', name: 'ROUGE-L overlap' },
];

/**
 * V1 stub of the Run Evaluation handoff. The backend endpoint exists
 * (`agentsCreateJob`) but Studio isn't wired to it yet. Rather than fake a
 * success, we surface an honest "coming next" toast and close — see Follow-up
 * A in the staged-seahorse plan. A future PR replaces submit() with a real
 * POST and routes the user to the eval-job detail page.
 */
export const RunEvaluationModal: FC<RunEvaluationModalProps> = ({
  open,
  onClose,
  workspace,
  modelUrns,
}) => {
  const toast = useToast();
  const [evalSetId, setEvalSetId] = useState(EVAL_SETS[0].id);
  const [metricId, setMetricId] = useState(METRICS[0].id);
  const [submitting, setSubmitting] = useState(false);

  const submit = useCallback(async () => {
    setSubmitting(true);
    toast.info(
      `Coming next — Studio will POST this evaluation to the agents service in the next release. (Captured: ${modelUrns.length} model${modelUrns.length === 1 ? '' : 's'} · eval-set ${evalSetId} · metric ${metricId})`
    );
    setSubmitting(false);
    onClose();
  }, [evalSetId, metricId, modelUrns.length, onClose, toast]);

  return (
    <FormModal
      open={open}
      onClose={onClose}
      title="Run Evaluation"
      submitButtonText="Submit Evaluation"
      onSubmit={submit}
      disabled={submitting || modelUrns.length === 0}
      loading={submitting}
    >
      <Stack gap="density-xl">
        <Block className="rounded border border-base bg-surface-sunken px-3 py-2">
          <Text className="text-secondary" kind="body/regular/sm">
            Preview only — Submit captures your choices and shows what would be sent. The wire-up to
            the evaluator service lands in the next release.
          </Text>
        </Block>
        <Block>
          <Text className="text-secondary" kind="label/bold/sm">
            Models from this Playground ({modelUrns.length})
          </Text>
          {modelUrns.length === 0 ? (
            <Text className="text-secondary" kind="body/regular/sm">
              Pick at least one model in the Playground first.
            </Text>
          ) : (
            <ul className="mt-2 list-disc pl-5">
              {modelUrns.map((u) => (
                <li key={u} className="text-sm font-mono">
                  {u}
                </li>
              ))}
            </ul>
          )}
        </Block>
        <Block>
          <Text className="text-secondary" kind="label/bold/sm">
            Eval set
          </Text>
          <select
            className="mt-1 w-full rounded border border-base bg-surface-sunken px-2 py-1 text-sm"
            value={evalSetId}
            onChange={(e) => setEvalSetId(e.target.value)}
          >
            {EVAL_SETS.map((e) => (
              <option key={e.id} value={e.id}>
                {e.name}
              </option>
            ))}
          </select>
        </Block>
        <Block>
          <Text className="text-secondary" kind="label/bold/sm">
            Metric
          </Text>
          <select
            className="mt-1 w-full rounded border border-base bg-surface-sunken px-2 py-1 text-sm"
            value={metricId}
            onChange={(e) => setMetricId(e.target.value)}
          >
            {METRICS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </Block>
        <Text className="text-secondary" kind="body/regular/sm">
          Workspace: <Text kind="body/semibold/sm">{workspace}</Text>
        </Text>
      </Stack>
    </FormModal>
  );
};
