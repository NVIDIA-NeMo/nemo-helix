// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { buildEvaluationSpec } from '@studio/components/evaluation/EvalWizard/buildEvaluationSpec';
import {
  type DatasetBindings,
  EVALUATION_FORM_DEFAULTS,
} from '@studio/components/evaluation/EvalWizard/types';
import { specToFormValues } from '@studio/components/evaluation/ModelEvaluationWizard/specToFormValues';

const bindings: DatasetBindings = {
  messagesColumn: null,
  input: '{{item.input}}',
  reference: '{{item.reference}}',
  context: null,
  inputPath: 'input',
  referencePath: 'reference',
};

const savedSpec = (judgeModel: string) =>
  buildEvaluationSpec(
    {
      ...EVALUATION_FORM_DEFAULTS,
      model: 'ws/target-model',
      body: { ...EVALUATION_FORM_DEFAULTS.body, judgeModel },
    },
    bindings,
    'ws'
  );

describe('specToFormValues judge model', () => {
  it.each(['ws/judge-model', 'other-workspace/judge-model'])(
    'reopens a saved config with its judge %s selected',
    (judgeModel) => {
      expect(specToFormValues(savedSpec(judgeModel), 'ws').body.judgeModel).toBe(judgeModel);
    }
  );

  it('reopens a config saved with an inline judge as a ref in the current workspace', () => {
    const spec = savedSpec('ws/judge-model');
    for (const bundle of spec.metrics ?? []) {
      const metric = bundle.payload?.metric as Record<string, unknown> | undefined;
      if (metric && 'model' in metric) {
        metric.model = {
          url: 'https://helix.example.com/apis/inference-gateway/x',
          name: 'judge-model',
        };
      }
    }

    expect(specToFormValues(spec, 'ws').body.judgeModel).toBe('ws/judge-model');
  });
});
