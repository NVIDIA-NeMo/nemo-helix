// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  type DatasetBindings,
  EVALUATION_FORM_DEFAULTS,
} from '@studio/components/evaluation/ModelEvaluationWizard/types';
import { useCreateEvaluation } from '@studio/components/evaluation/ModelEvaluationWizard/useCreateEvaluation';
import { server } from '@studio/mocks/node';
import { renderRoute } from '@studio/tests/util/render';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const message = `Evaluation rejected: ${'Please check the dataset and model configuration. '.repeat(8)}End of server message.`;
const bindings: DatasetBindings = {
  messagesColumn: null,
  input: '{{item.input}}',
  reference: '{{item.reference}}',
  context: null,
  inputPath: 'input',
  referencePath: 'reference',
};

function SubmitEvaluation() {
  const { createEvaluation } = useCreateEvaluation();
  return (
    <button
      onClick={() =>
        createEvaluation(
          {
            ...EVALUATION_FORM_DEFAULTS,
            configSource: 'saved-config',
            model: `${DEFAULT_WORKSPACE}/test-model`,
          },
          bindings
        )
      }
    >
      Run Evaluation
    </button>
  );
}

describe('model evaluation submit errors', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it.each(
    [400, 422, 500].flatMap((status) =>
      ['detail', 'message', 'text', 'validation'].map((format) => ({ status, format }))
    )
  )('shows the full $format response for HTTP $status', async ({ status, format }) => {
    const errorLog = vi.spyOn(console, 'error').mockImplementation(() => {});
    server.use(
      http.post('*/apis/evals/v2/workspaces/:workspace/evaluate/jobs', () => {
        if (format === 'text') return HttpResponse.text(message, { status });
        if (format === 'validation') {
          return HttpResponse.json(
            { detail: [{ msg: message, type: 'value_error', loc: ['body', 'model'] }] },
            { status }
          );
        }
        return HttpResponse.json({ [format]: message }, { status });
      })
    );
    renderRoute(undefined, {
      history: `/studio/workspaces/${DEFAULT_WORKSPACE}/evaluation/new`,
      routes: [
        { path: '/studio/workspaces/:workspace/evaluation/new', element: <SubmitEvaluation /> },
      ],
    });

    await userEvent.click(screen.getByRole('button', { name: 'Run Evaluation' }));

    expect(
      await screen.findByText(format === 'validation' ? `model: ${message}` : message)
    ).toBeVisible();
    expect(screen.queryByText(/Request failed with status code/)).not.toBeInTheDocument();
    expect(errorLog).toHaveBeenCalledWith(
      `ModelEvaluationWizard: ${format === 'validation' ? 'model: ' : ''}${message}`
    );
  });
});
