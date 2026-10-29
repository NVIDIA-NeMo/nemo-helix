// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getErrorMessage, isConflictError } from '@nemo/common/src/api/common/utils';
import { useModelEntity } from '@nemo/common/src/api/models/useModelEntity';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { logger } from '@nemo/common/src/utils/logger';
import { toInferenceModelName } from '@nemo/common/src/utils/models';
import { useEvalsCreateEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import type { EvaluateJobRequest, MetricInline } from '@nemo/sdk/generated/evals/schema';
import { getGatewayProxyGetQueryKey } from '@nemo/sdk/generated/platform/inference-gateway';
import type { ModelEntity } from '@nemo/sdk/generated/platform/schema';
import { createEvalConfigFileset } from '@studio/api/evaluation/eval-config-fileset';
import { evalConfigFilename } from '@studio/components/evaluation/experimentEvalConfig';
import { buildEvaluationSpec } from '@studio/components/evaluation/ModelEvaluationWizard/buildEvaluationSpec';
import {
  type DatasetBindings,
  type EvaluationFormValues,
  NEW_CONFIG,
} from '@studio/components/evaluation/ModelEvaluationWizard/types';
import {
  buildEvalJobName,
  MODEL_EVAL_CONFIG_DESCRIPTION,
  serializeEvalConfig,
} from '@studio/components/evaluation/submitEvaluationJob';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { useWorkspaceFromPath } from '@studio/hooks/useWorkspaceFromPath';
import { getEvaluationResultDetailsRoute } from '@studio/routes/utils';
import { parseEvaluationModelValue } from '@studio/util/evaluations';
import { toWorkspaceModelRef } from '@studio/util/models';
import { useNavigate } from 'react-router';

/**
 * @param selectedModel The current `model` form value, watched by the caller. An
 *   adapter's parent entity has to be fetched before submit, for the workspace
 *   its composite name carries -- and a hook cannot run inside the submit
 *   handler.
 */
export function useCreateEvaluation(selectedModel: string = '') {
  const workspace = useWorkspaceFromPath();
  const navigate = useNavigate();
  const toast = useToast();
  const { mutateAsync: createEvaluateJob, isPending } = useEvalsCreateEvaluateJob();

  const { modelUrn, adapterName } = parseEvaluationModelValue(selectedModel);
  const parentModel = useModelEntity(modelUrn, { enabled: Boolean(adapterName) });

  /** The base model's Inference Gateway route, for the one target a ref cannot
   *  name. Built the same way `useChatCompletion` builds it for an adapter chat:
   *  route on the base model, and let the gateway read the adapter from the
   *  model field. Model targets no longer go near this -- the server resolves
   *  their ref to a URL the job can reach, which a browser origin is not.
   *
   *  Taken off the entity, not the page: the gateway finds the base VirtualModel
   *  by the URL's workspace, and this is the same entity the composite name was
   *  built from, so the two cannot disagree about which model it is. */
  const baseModelGatewayUrl = (model: ModelEntity): string => {
    const [path] = getGatewayProxyGetQueryKey(model.workspace, model.name, 'v1/');
    return `${PLATFORM_BASE_URL}${path}`;
  };

  /** The run's `spec.target`, or undefined once the failure has been reported.
   *
   *  A model is sent as a `workspace/name` ref and resolved server-side, which
   *  is what keeps the browser's origin out of the job's URL.
   *
   *  An adapter cannot be: a ref resolves by looking up a VirtualModel or model
   *  entity of that name, and a LoRA adapter has neither. It stays an inline
   *  model, addressed exactly as chat addresses one -- routed on the BASE
   *  model's gateway URL, with the composite `base&adapters/{workspace}/{adapter}`
   *  in the model field, which the gateway splits to resolve the adapter through
   *  the base model's VirtualModel. So the adapter inherits that VM's middleware,
   *  and no provider has to be resolved -- it rides whatever serves its base.
   *  The job proxies inline key-less gateway models by path, so this URL's origin
   *  is replaced just as a resolved ref's would be.
   */
  const buildTarget = (modelValue: string): string | { url: string; name: string } | undefined => {
    const { modelUrn, adapterName: adapter } = parseEvaluationModelValue(modelValue);
    if (!adapter) {
      return toWorkspaceModelRef(workspace, modelUrn);
    }

    // The composite carries the adapter's OWN workspace, which may differ from
    // its base model's, so it has to come off the entity rather than be assumed.
    const adapterEntity = parentModel?.adapters?.find((entry) => entry.name === adapter);
    if (!parentModel || !adapterEntity) {
      toast.error(
        parentModel
          ? `"${modelUrn}" has no adapter named "${adapter}".`
          : 'Still loading the selected adapter. Try again in a moment.'
      );
      return undefined;
    }
    return {
      url: baseModelGatewayUrl(parentModel),
      name: toInferenceModelName(parentModel, adapterEntity),
    };
  };

  const createEvaluation = async (
    values: EvaluationFormValues,
    bindings: DatasetBindings,
    onNameConflict?: (name: string) => void
  ) => {
    const spec = buildEvaluationSpec(values, bindings, workspace);

    // Reusing a saved config: `configSource` IS the fileset it lives in, so the
    // run takes its name from there and writes nothing. Authoring: the fileset
    // takes the name the user typed, verbatim.
    const reusing = Boolean(values.configSource) && values.configSource !== NEW_CONFIG;
    const fileset = reusing ? values.configSource : values.name;

    // A model goes as a ref for the server to resolve; an adapter, which a ref
    // cannot name, goes inline. See `buildTarget`.
    const target = buildTarget(values.model);
    if (!target) return;

    try {
      if (!reusing) {
        // Persist first: a config that cannot be stored is not worth running, and
        // this way a failed upload leaves no orphan job behind.
        await createEvalConfigFileset(
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
          MODEL_EVAL_CONFIG_DESCRIPTION
        );
      }

      const request: EvaluateJobRequest = {
        name: buildEvalJobName(fileset),
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
          target,
          prompt_template: spec.prompt_template,
          ...(spec.field_mapping ? { field_mapping: spec.field_mapping } : {}),
        },
      };

      const job = await createEvaluateJob({ workspace, data: request });
      toast.success('Evaluation job created');
      navigate(getEvaluationResultDetailsRoute(workspace, job.name), {
        flushSync: true,
        replace: true,
      });
    } catch (error) {
      if (!reusing && isConflictError(error)) {
        onNameConflict?.(fileset);
        return;
      }
      const message = getErrorMessage(error as Error, 'Failed to create evaluation');
      logger.error(`ModelEvaluationWizard: ${message}`);
      toast.error(message);
    }
  };

  return { createEvaluation, isPending };
}
