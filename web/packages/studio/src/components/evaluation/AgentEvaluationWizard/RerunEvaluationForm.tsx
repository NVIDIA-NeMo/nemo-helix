// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isConflictError } from '@nemo/common/src/api/common/utils';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { evalsCreateEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import type { AgentEvaluateJobRequest, EvaluateJobRequest } from '@nemo/sdk/generated/evals/schema';
import { deleteEvaluation, getEvaluation } from '@nemo/sdk/generated/platform/evaluations';
import { useListExperiments } from '@nemo/sdk/generated/platform/experiments';
import { jobsGetJob } from '@nemo/sdk/generated/platform/jobs';
import { Banner, Stack, Text } from '@nvidia/foundations-react-core';
import { submitAgentEvalJob } from '@studio/api/evaluation/agent-evaluations';
import { evalJobKind, publishedEvaluationName } from '@studio/api/evaluation/utils';
import {
  resolveRunTarget,
  RUN_TARGET_DEFAULTS,
  useRunNameCheck,
} from '@studio/components/evaluation/AgentEvaluationWizard/runTarget';
import { RunTargetFields } from '@studio/components/evaluation/AgentEvaluationWizard/RunTargetFields';
import type { RunTargetFormFields } from '@studio/components/evaluation/AgentEvaluationWizard/types';
import { isDefaultExperiment } from '@studio/components/evaluation/defaultExperiment';
import { EvalWizard } from '@studio/components/evaluation/EvalWizard';
import {
  createRunEvaluation,
  EVAL_CONFIG_FILESET_KEY,
  evaluationFilesetName,
  recordRunJob,
} from '@studio/components/evaluation/experimentEvalConfig';
import { submitErrorMessage } from '@studio/components/evaluation/shared/entityNameField';
import { buildRerunRequestBody } from '@studio/components/evaluation/submitEvaluationJob';
import { Loading } from '@studio/components/Layouts/Loading';
import { DEFAULT_LARGE_PAGE_SIZE } from '@studio/constants/constants';
import { getAgentEvaluationsTabRoute } from '@studio/routes/utils';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { type FC, useEffect } from 'react';
import { FormProvider, useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';

interface RerunEvaluationFormProps {
  workspace: string;
  agent: string;
  /** The evaluator job whose spec is copied. */
  sourceJob: string;
}

const configFilesetLabel = (spec: Record<string, unknown>): string | undefined => {
  const value = (spec.labels as Record<string, unknown> | undefined)?.eval_config_fileset;
  return typeof value === 'string' && value ? value : undefined;
};

/** Re-submit a job's spec as-is under a new evaluation. Only where the results land is asked
 *  for, and both answers are optional, so a re-run is one click away. */
export const RerunEvaluationForm: FC<RerunEvaluationFormProps> = ({
  workspace,
  agent,
  sourceJob,
}) => {
  const toast = useToast();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const methods = useForm<RunTargetFormFields>({ defaultValues: RUN_TARGET_DEFAULTS });
  const { handleSubmit, setValue, getFieldState } = methods;

  const jobQuery = useQuery({
    queryKey: ['rerun-source-job', workspace, sourceJob],
    queryFn: ({ signal }) => jobsGetJob(workspace, sourceJob, signal),
  });
  const job = jobQuery.data;
  const sourceEvaluationName = job ? publishedEvaluationName(job) : null;
  // Optional: a job that never published still re-runs, it just has no lineage to record.
  const { data: sourceEvaluation, isLoading: isLoadingEvaluation } = useQuery({
    queryKey: ['rerun-source-evaluation', workspace, sourceEvaluationName],
    queryFn: ({ signal }) =>
      getEvaluation(workspace, sourceEvaluationName ?? '', signal).catch(() => null),
    enabled: !!sourceEvaluationName,
  });
  const { data: experimentsPage } = useListExperiments(workspace, {
    page_size: DEFAULT_LARGE_PAGE_SIZE,
  });

  // Land the copy beside its source on the same leaderboard, unless the user already chose.
  useEffect(() => {
    if (!sourceEvaluation || !experimentsPage || getFieldState('experimentId').isDirty) return;
    const linked = (experimentsPage.data ?? []).find(
      (experiment) =>
        sourceEvaluation.experiment_ids.includes(experiment.id) && !isDefaultExperiment(experiment)
    );
    if (linked) setValue('experimentId', linked.id);
  }, [sourceEvaluation, experimentsPage, getFieldState, setValue]);

  const { status: nameStatus } = useRunNameCheck(workspace, methods.control);

  const {
    mutateAsync: submit,
    error: submitError,
    isPending,
  } = useMutation({
    mutationFn: async (formData: RunTargetFormFields) => {
      if (!job?.spec) throw new Error(`Job "${sourceJob}" has no spec to copy`);
      const spec = job.spec as Record<string, unknown>;
      const { experimentIds, name } = await resolveRunTarget(workspace, agent, formData);
      const filesetName =
        (sourceEvaluation && evaluationFilesetName(sourceEvaluation)) ?? configFilesetLabel(spec);
      await createRunEvaluation(workspace, {
        experimentIds,
        name,
        filesetName,
        parentEvaluationId: sourceEvaluation?.id,
      }).catch((err: unknown) => {
        if (isConflictError(err)) {
          throw new Error(
            `An evaluation named "${name}" already exists — choose a different evaluation name`
          );
        }
        throw err;
      });
      try {
        const body = buildRerunRequestBody(spec, name, name);
        const created =
          evalJobKind(job) === 'dataset'
            ? await evalsCreateEvaluateJob(workspace, body as EvaluateJobRequest)
            : await submitAgentEvalJob(workspace, body as AgentEvaluateJobRequest);
        if (!created?.name) throw new Error('Submission did not return a job name');
        await recordRunJob(
          workspace,
          name,
          created.name,
          filesetName ? { [EVAL_CONFIG_FILESET_KEY]: filesetName } : {}
        );
        return created.name;
      } catch (err) {
        await deleteEvaluation(workspace, name).catch(() => undefined);
        throw err;
      }
    },
    onSuccess: (name) => {
      toast.success(`Evaluation "${name}" submitted`);
      void queryClient.invalidateQueries({ queryKey: ['evaluator-jobs', workspace] });
      navigate(getAgentEvaluationsTabRoute(workspace, agent), { replace: true });
    },
  });

  if (jobQuery.isLoading) return <Loading description="Loading job..." />;

  const nameBusy = nameStatus === 'checking' || nameStatus === 'conflict';
  const blockedHint = isLoadingEvaluation ? 'Loading the source evaluation...' : undefined;
  const errorMessage = jobQuery.error
    ? `Could not load job "${sourceJob}". It may have been deleted.`
    : submitErrorMessage(submitError);

  const run = () => void handleSubmit((data) => submit(data).catch(() => undefined))();

  return (
    <EvalWizard
      heading="Re-run Agent Evaluation"
      headingDescription="Submit a copy of an earlier evaluation job."
      onSubmit={(event) => {
        event.preventDefault();
        run();
      }}
      slotBanner={
        errorMessage ? (
          <Banner kind="inline" status="error">
            {errorMessage}
          </Banner>
        ) : null
      }
      cancelTo={getAgentEvaluationsTabRoute(workspace, agent)}
      continueLabel="Run Evaluation"
      continueLoading={isPending}
      canContinue={!!job && !nameBusy && !isLoadingEvaluation}
      blockedHint={blockedHint}
      onContinue={run}
    >
      <FormProvider {...methods}>
        <Stack gap="density-xl">
          <Text kind="body/regular/md">
            Copies the dataset, metrics, and target from job{' '}
            <Text kind="body/semibold/md">{sourceJob}</Text>
            {sourceEvaluationName ? ` (evaluation ${sourceEvaluationName})` : ''}. Only where the
            results publish changes.
          </Text>
          <RunTargetFields workspace={workspace} agent={agent} disabled={isPending || !job} />
        </Stack>
      </FormProvider>
    </EvalWizard>
  );
};
