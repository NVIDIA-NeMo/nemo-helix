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

describe('model evaluation submit body', () => {
  function SubmitWithJudge({ judgeModel }: { judgeModel: string }) {
    const { createEvaluation } = useCreateEvaluation();
    return (
      <button
        onClick={() =>
          createEvaluation(
            {
              ...EVALUATION_FORM_DEFAULTS,
              configSource: 'saved-config',
              model: `${DEFAULT_WORKSPACE}/test-model`,
              body: { ...EVALUATION_FORM_DEFAULTS.body, judgeModel },
            },
            bindings
          )
        }
      >
        Run Evaluation
      </button>
    );
  }

  it.each([`${DEFAULT_WORKSPACE}/judge-model`, 'other-workspace/judge-model'])(
    'sends the target and the judge %s as workspace/name refs for the server to resolve',
    async (judgeModel) => {
      let body:
        | { spec: { target: unknown; metrics: { payload: { metric: { model?: unknown } } }[] } }
        | undefined;
      server.use(
        http.post('*/apis/evals/v2/workspaces/:workspace/evaluate/jobs', async ({ request }) => {
          body = (await request.json()) as typeof body;
          return HttpResponse.json(
            { name: 'eval-job', workspace: DEFAULT_WORKSPACE },
            { status: 201 }
          );
        })
      );
      renderRoute(undefined, {
        history: `/studio/workspaces/${DEFAULT_WORKSPACE}/evaluation/new`,
        routes: [
          {
            path: '/studio/workspaces/:workspace/evaluation/new',
            element: <SubmitWithJudge judgeModel={judgeModel} />,
          },
          // Where a successful submit navigates.
          { path: '*', element: <div /> },
        ],
      });

      await userEvent.click(screen.getByRole('button', { name: 'Run Evaluation' }));

      await vi.waitFor(() => expect(body).toBeDefined());
      expect(body?.spec.target).toBe(`${DEFAULT_WORKSPACE}/test-model`);
      const judges = body?.spec.metrics.filter(
        (metric) => metric.payload.metric.model !== undefined
      );
      expect(judges?.map((metric) => metric.payload.metric.model)).toEqual([judgeModel]);
    }
  );
});

function SubmitAdapterEvaluation({ model }: { model: string }) {
  const { createEvaluation } = useCreateEvaluation(model);
  return (
    <button
      onClick={() =>
        createEvaluation(
          { ...EVALUATION_FORM_DEFAULTS, configSource: 'saved-config', model },
          bindings
        )
      }
    >
      Run Evaluation
    </button>
  );
}

describe('model evaluation target', () => {
  const adapterValue = `${DEFAULT_WORKSPACE}/test-model::my-lora`;

  const renderAt = (model: string) =>
    renderRoute(undefined, {
      history: `/studio/workspaces/${DEFAULT_WORKSPACE}/evaluation/new`,
      routes: [
        {
          path: '/studio/workspaces/:workspace/evaluation/new',
          element: <SubmitAdapterEvaluation model={model} />,
        },
        // A successful submit navigates to the new run; the destination only has
        // to exist for the router, since the assertions are on what was posted.
        { path: '/workspaces/:workspace/evaluation/results/:id', element: <div /> },
      ],
    });

  const submitted = () => {
    const specs: Record<string, unknown>[] = [];
    server.use(
      http.get('*/apis/models/v2/workspaces/:workspace/models/:name', ({ params }) =>
        HttpResponse.json({
          name: 'test-model',
          workspace: params.workspace,
          // Deliberately no `model_providers`: an adapter rides whatever serves
          // its base model, so the target must not depend on one being listed.
          adapters: [{ name: 'my-lora', workspace: 'adapter-ws' }],
        })
      ),
      http.post('*/apis/evals/v2/workspaces/:workspace/evaluate/jobs', async ({ request }) => {
        const body = (await request.json()) as { name: string; spec: Record<string, unknown> };
        specs.push(body.spec);
        return HttpResponse.json({ name: body.name });
      })
    );
    return specs;
  };

  it('targets an adapter inline, routed on its base model', async () => {
    const specs = submitted();
    renderAt(adapterValue);

    await userEvent.click(await screen.findByRole('button', { name: 'Run Evaluation' }));

    await vi.waitFor(() => expect(specs).toHaveLength(1));
    // A ref cannot name an adapter, so this one target stays inline.
    // Routed on the base model, with the composite in the model field -- the
    // same addressing chat uses, so the adapter inherits the base VM's
    // middleware and the run is attributed to the adapter, not to its base.
    expect(specs[0].target).toEqual({
      url: expect.stringContaining(`/workspaces/${DEFAULT_WORKSPACE}/model/test-model/-/v1`),
      name: 'test-model&adapters/adapter-ws/my-lora',
    });
  });

  it("routes on the base model's own workspace, not the page's", async () => {
    const specs = submitted();
    renderAt('other-ws/test-model::my-lora');

    await userEvent.click(await screen.findByRole('button', { name: 'Run Evaluation' }));

    await vi.waitFor(() => expect(specs).toHaveLength(1));
    // The gateway finds the base VirtualModel by the URL's workspace, so routing
    // on the page's would miss it -- or reach a same-named model there instead.
    expect(specs[0].target).toEqual({
      url: expect.stringContaining('/workspaces/other-ws/model/test-model/-/v1'),
      name: 'test-model&adapters/adapter-ws/my-lora',
    });
  });
});
