// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { getErrorMessage } from '@nemo/common/src/utils/error';
import { logger } from '@nemo/common/src/utils/logger';
import { useEvaluatorCreateEvaluateJob } from '@nemo/sdk/generated/evaluator/evaluator-plugin-jobs-routes';
import type { EvaluateJobRequest, MetricInline } from '@nemo/sdk/generated/evaluator/schema';
import { ensureEvalConfigFileset } from '@studio/api/evaluation/eval-config-fileset';
import { evalConfigFilename } from '@studio/components/evaluation/experimentEvalConfig';
import {
  buildEvalJobName,
  generateEvalConfigName,
  serializeEvalConfig,
} from '@studio/components/evaluation/submitEvaluationJob';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { buildEvaluationSpec } from '@studio/routes/evaluation/EvaluationNewRoute/buildEvaluationSpec';
import type {
  EvaluationFormValues,
  DatasetBindings,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';
import { getEvaluationResultDetailsRoute } from '@studio/routes/utils';
import { getModelInferenceGatewayUrl } from '@studio/util/models';
import { useNavigate } from 'react-router';

/** Fileset holding a run's reusable configuration. The dataset is NOT copied in:
 *  it already lives in its own fileset and the spec references it, so a config
 *  records which dataset it was built against without duplicating the data. */
const evalConfigFileset = (name: string) => `${name}-eval`;

/** The bare model id the endpoint expects, without its workspace prefix. */
const modelName = (modelRef: string): string =>
  modelRef.includes('/') ? modelRef.split('/').slice(1).join('/') : modelRef;

export function useCreateEvaluation() {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const toast = useToast();
  const { mutateAsync: createEvaluateJob, isPending } = useEvaluatorCreateEvaluateJob();

  const createEvaluation = async (values: EvaluationFormValues, bindings: DatasetBindings) => {
    const spec = buildEvaluationSpec(values, bindings, workspace);

    const name = generateEvalConfigName();
    const fileset = evalConfigFileset(name);

    try {
      // Persist first: a config that cannot be stored is not worth running, and
      // this way a failed upload leaves no orphan job behind.
      await ensureEvalConfigFileset(
        workspace,
        fileset,
        new AbortController().signal,
        [
          {
            path: evalConfigFilename('json'),
            content: serializeEvalConfig(spec, 'json'),
            type: 'application/json',
          },
        ],
        'Model Evaluation Config'
      );

      const request: EvaluateJobRequest = {
        name: buildEvalJobName(name),
        spec: {
          dataset: spec.dataset,
          metrics: spec.metrics as unknown as MetricInline[],
          // A Model target REQUIRES params that resolve to RunConfigOnlineModel.
          // `params` is an untagged union whose base RunConfig sets
          // extra="forbid", so an online-only key is what selects the online
          // member -- omitting params, or sending {}, lands on the base class and
          // `resolve_params` raises TypeError. Pydantic only converts ValueError
          // and AssertionError into a 422, so that escapes as an opaque 500.
          // ignore_request_failure stays false so a model 502 fails the job
          // rather than silently scoring the row NaN.
          //
          // parallelism is lowered from the SDK default of 8 because
          // integrate.api.nvidia.com caps CONCURRENT requests per worker at 16,
          // and every row costs two calls -- one generation, one judge -- so the
          // default fans out to roughly that ceiling on its own and the upstream
          // starts returning 503 "ResourceExhausted: Worker local total request
          // limit reached".
          params: { ignore_request_failure: false, parallelism: 4 },
          // TargetSpec is `Model | Agent` -- unlike the judge's `model` field, a
          // bare ModelRef string is not accepted, so the ref is resolved to its
          // gateway URL here. The SDK strips a trailing /chat/completions before
          // using this as an OpenAI base_url, so the full chat URL is correct.
          target: {
            url: getModelInferenceGatewayUrl(workspace, values.model),
            name: modelName(values.model),
          },
          prompt_template: spec.prompt_template,
          ...(spec.field_mapping ? { field_mapping: spec.field_mapping } : {}),
        },
      };

      const job = await createEvaluateJob({ workspace, data: request });
      toast.success('Evaluation job created');
      navigate(getEvaluationResultDetailsRoute(workspace, job.name), { flushSync: true });
    } catch (error) {
      const message = getErrorMessage(error as Error, 'Failed to create evaluation');
      logger.error(`EvaluationNewRoute: ${message}`);
      toast.error(message);
    }
  };

  return { createEvaluation, isPending };
}
