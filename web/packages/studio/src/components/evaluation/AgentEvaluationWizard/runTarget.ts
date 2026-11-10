// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_DEBOUNCE_MS } from '@nemo/common/src/constants';
import { toValidEntityName } from '@nemo/common/src/utils/entityName';
import { useListEvaluations } from '@nemo/sdk/generated/platform/evaluations';
import type { RunTargetFormFields } from '@studio/components/evaluation/AgentEvaluationWizard/types';
import { ensureDefaultExperiment } from '@studio/components/evaluation/defaultExperiment';
import { entityNameField } from '@studio/components/evaluation/shared/entityNameField';
import { bareName, buildEvalJobName } from '@studio/components/evaluation/submitEvaluationJob';
import { nameCheckStatus } from '@studio/components/form/entityNameCheck';
import { type Control, useWatch } from 'react-hook-form';
import { useDebounce } from 'use-debounce';
import { z } from 'zod';

export const runTargetSchemaShape = {
  experimentId: z.string(),
  evaluationRecordName: entityNameField(),
};

export const RUN_TARGET_DEFAULTS: RunTargetFormFields = {
  experimentId: '',
  evaluationRecordName: '',
};

/** The experiments and name a run is published under, creating the default experiment when no
 *  experiment was picked. */
export const resolveRunTarget = async (
  workspace: string,
  agent: string,
  { experimentId, evaluationRecordName }: RunTargetFormFields
): Promise<{ experimentIds: string[]; name: string }> => ({
  experimentIds: experimentId
    ? [experimentId]
    : [(await ensureDefaultExperiment(workspace, agent)).id],
  name: toValidEntityName(evaluationRecordName, '') || buildEvalJobName(bareName(agent)),
});

/** Whether the typed evaluation name is free. Undefined while the field is blank. */
export const useRunNameCheck = <T extends RunTargetFormFields>(
  workspace: string,
  control: Control<T>
) => {
  const typedControl = control as unknown as Control<RunTargetFormFields>;
  const preview = toValidEntityName(
    useWatch({ control: typedControl, name: 'evaluationRecordName' }),
    ''
  );
  const [debounced] = useDebounce(preview, DEFAULT_DEBOUNCE_MS);
  const query = useListEvaluations(
    workspace,
    { page_size: 1, filter: { name: debounced } },
    { query: { enabled: !!debounced } }
  );
  return { preview, status: nameCheckStatus(preview, debounced, query) };
};
