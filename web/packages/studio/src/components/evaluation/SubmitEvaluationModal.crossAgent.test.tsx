// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { evalsCreateEvaluateJob } from '@nemo/sdk/generated/evals/evals-plugin-jobs-routes';
import { createExperiment } from '@nemo/sdk/generated/platform/experiments';
import { filesDownloadFile } from '@nemo/sdk/generated/platform/files';
import {
  createRunEvaluation,
  evaluationConfigError,
  findEvalConfigFile,
} from '@studio/components/evaluation/experimentEvalConfig';
import { SubmitEvaluationModal } from '@studio/components/evaluation/SubmitEvaluationModal';
import { ROUTES } from '@studio/constants/routes';
import { evaluationSourcesHandlers } from '@studio/mocks/handlers/evaluationSources';
import { server } from '@studio/mocks/node';
import {
  experimentFixture,
  reusableEvaluationFixture,
} from '@studio/tests/util/evaluationFixtures';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

vi.mock('@nemo/sdk/generated/evals/evals-plugin-jobs-routes', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/evals/evals-plugin-jobs-routes')>()),
  evalsCreateEvaluateJob: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/experiments', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/experiments')>()),
  createExperiment: vi.fn(),
}));

vi.mock('@nemo/sdk/generated/platform/files', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@nemo/sdk/generated/platform/files')>()),
  filesDownloadFile: vi.fn(),
}));

vi.mock('@studio/components/evaluation/experimentEvalConfig', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@studio/components/evaluation/experimentEvalConfig')>()),
  createRunEvaluation: vi.fn(),
  evaluationConfigError: vi.fn(),
  findEvalConfigFile: vi.fn(),
}));

const BASELINE_AGENT = 'f80-demand-planner';
const TUNED_AGENT = 'f80-demand-planner-tuned';

const BASELINE = reusableEvaluationFixture('baseline-t9', 'grp_planner', BASELINE_AGENT);

const STORED_EVAL_CONFIG = `dataset: default/baseline-t9-data#dataset.jsonl
prompt_template: "{{ item.prompt }}"
metrics:
  - bundle_kind: metric-bundle
    bundle_format_version: v1
    metric_type: string-check
    metadata: {}
    outputs:
      - name: string-check
        value_json_schema:
          type: number
    secrets: {}
    payload:
      kind: inline
      metric:
        type: string-check
        operation: equals
        left_template: "{{ sample.output_text }}"
        right_template: "{{ item.expected }}"
`;

beforeEach(() => {
  server.use(
    ...evaluationSourcesHandlers({
      experiments: [experimentFixture('grp_planner', 'f80-planner-t9')],
      evaluations: [BASELINE],
    })
  );
  vi.mocked(evalsCreateEvaluateJob).mockResolvedValue({ name: 'job-1' } as never);
  vi.mocked(createRunEvaluation).mockResolvedValue('eval_tuned' as never);
  vi.mocked(evaluationConfigError).mockResolvedValue(null);
  vi.mocked(findEvalConfigFile).mockResolvedValue('eval-config.yaml' as never);
  vi.mocked(filesDownloadFile).mockResolvedValue(new Blob([STORED_EVAL_CONFIG]) as never);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('SubmitEvaluationModal across agents', () => {
  it("runs one agent into another agent's experiment as a child of its evaluation", async () => {
    const user = userEvent.setup();
    renderRoute(undefined, {
      history: `/workspaces/${DEFAULT_WORKSPACE}`,
      routes: [
        {
          path: '/workspaces/:workspace',
          element: (
            <SubmitEvaluationModal
              open
              onClose={() => {}}
              workspace={DEFAULT_WORKSPACE}
              agent={TUNED_AGENT}
            />
          ),
        },
        { path: ROUTES.workspace.agentDetail, element: <div>Agent page</div> },
      ],
    });

    await waitFor(() =>
      expect(screen.getByRole('radio', { name: /Re-run an existing evaluation/ })).toBeEnabled()
    );
    await user.click(screen.getByRole('radio', { name: /Re-run an existing evaluation/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.click(await screen.findByRole('combobox', { name: /evaluation to re-run/i }));
    await user.click(await screen.findByRole('option', { name: 'baseline-t9' }));
    const name = await screen.findByLabelText('New Evaluation Name');
    await user.clear(name);
    await user.type(name, 'tuned-t9');

    await user.click(screen.getByRole('button', { name: 'Submit' }));

    await waitFor(() => expect(evalsCreateEvaluateJob).toHaveBeenCalledTimes(1));
    expect(createExperiment).not.toHaveBeenCalled();
    expect(createRunEvaluation).toHaveBeenCalledWith(
      DEFAULT_WORKSPACE,
      expect.objectContaining({
        name: 'tuned-t9',
        experimentIds: BASELINE.experiment_ids,
        parentEvaluationId: BASELINE.id,
        nameStem: 'f80-planner-t9',
      })
    );
    expect(vi.mocked(evalsCreateEvaluateJob).mock.calls[0]?.[1]?.spec?.target).toMatchObject({
      name: TUNED_AGENT,
    });
  });
});
