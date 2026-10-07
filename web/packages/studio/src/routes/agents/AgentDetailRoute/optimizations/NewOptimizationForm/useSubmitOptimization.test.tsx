// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { EVAL_CONFIG_FILESET_KEY } from '@studio/components/evaluation/experimentEvalConfig';
import { PLATFORM_BASE_URL } from '@studio/constants/environment';
import { ROUTES } from '@studio/constants/routes';
import { entityStoreBaseModel1 } from '@studio/mocks/entity-store/models';
import { server } from '@studio/mocks/node';
import { NewOptimizationForm } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm';
import { useSubmitOptimization } from '@studio/routes/agents/AgentDetailRoute/optimizations/NewOptimizationForm/useSubmitOptimization';
import type { AgentEvaluationRow } from '@studio/routes/agents/AgentDetailRoute/useAgentDetails';
import { getAgentOptimizeRoute } from '@studio/routes/utils';
import { renderRoute, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { parse } from 'yaml';

const AGENT = 'react-agent';
const FILESETS_URL = `${PLATFORM_BASE_URL}/apis/files/v2/workspaces/:workspace/filesets`;
const JOBS_URL = `${PLATFORM_BASE_URL}/apis/agent-optimization/v2/workspaces/:workspace/jobs/run-strategy`;
const EVALUATION: AgentEvaluationRow = {
  id: 'eval-baseline',
  name: 'baseline',
  workspace: DEFAULT_WORKSPACE,
  experiment_ids: ['exp-quality'],
  experiment_group_id: 'exp-quality',
  dataset_name: 'eval-data',
  metadata: { [EVAL_CONFIG_FILESET_KEY]: 'eval-data' },
  experiments: [
    {
      id: 'exp-quality',
      name: 'Quality',
      description: null,
      isFavorite: false,
      showsEvaluationsOverTime: false,
    },
  ],
};

const SubmitForm = ({ evaluation = EVALUATION }: { evaluation?: AgentEvaluationRow }) => {
  const onSubmit = useSubmitOptimization({
    workspace: DEFAULT_WORKSPACE,
    agentName: AGENT,
    evals: [evaluation],
  });
  return (
    <NewOptimizationForm
      agentName={AGENT}
      evals={[evaluation]}
      isEvalsPending={false}
      onBack={() => {}}
      onSubmit={onSubmit}
    />
  );
};

describe('optimization form submission', () => {
  it('uploads gateway models and a fixed credential from the selected judge and budget', async () => {
    const uploaded = new Map<string, string>();
    const submitted: unknown[] = [];
    server.use(
      http.get(`${FILESETS_URL}/eval-data/files`, () =>
        HttpResponse.json({ data: [{ path: 'eval-config.json' }] })
      ),
      http.get(`${FILESETS_URL}/eval-data/-/eval-config.json`, () =>
        HttpResponse.text(
          JSON.stringify({
            tasks: [
              {
                id: 'row-1',
                intent: 'Classify this email',
                reference: { answer: 'phishing' },
                metrics: [
                  {
                    metric_type: 'exact-match',
                    payload: { metric: {} },
                  },
                ],
              },
            ],
          })
        )
      ),
      http.get(`${PLATFORM_BASE_URL}/apis/agents/v2/workspaces/:workspace/agents/${AGENT}`, () =>
        HttpResponse.json({
          name: AGENT,
          config: {
            models: {
              default: {
                provider: 'nvidia',
                model: 'agent-model',
                base_url: 'https://integrate.api.nvidia.com/v1',
                api_key_env: 'NVIDIA_API_KEY',
              },
            },
          },
        })
      ),
      http.get(`${PLATFORM_BASE_URL}/apis/models/v2/workspaces/:workspace/models`, () =>
        HttpResponse.json({ data: [entityStoreBaseModel1] })
      ),
      http.put(`${FILESETS_URL}/:name/-/*`, async ({ request }) => {
        const path = decodeURIComponent(new URL(request.url).pathname.split('/-/')[1]);
        uploaded.set(path, await request.text());
        return new HttpResponse(null, { status: 200 });
      }),
      http.post(JOBS_URL, async ({ request }) => {
        submitted.push(await request.json());
        return HttpResponse.json({ name: 'study-1' });
      })
    );
    const user = userEvent.setup();
    renderRoute(undefined, {
      history: getAgentOptimizeRoute(DEFAULT_WORKSPACE, AGENT),
      routes: [
        { path: ROUTES.workspace.agentDetail, element: <SubmitForm /> },
        { path: ROUTES.workspace.agentOptimizationDetail, element: <div>Study detail page</div> },
      ],
    });

    await user.click(await screen.findByRole('button', { name: 'Select a model' }));
    await user.click(await screen.findByText('codellama-70b'));
    await user.click(screen.getByRole('radio', { name: /Quick/ }));
    await user.click(screen.getByRole('button', { name: 'Run optimization' }));

    expect(await screen.findByText('Study detail page')).toBeInTheDocument();
    expect(submitted).toHaveLength(1);
    expect(submitted[0]).toMatchObject({
      spec: {
        agent: AGENT,
        strategy: 'legacy',
        optimize_config: 'optimize.yaml',
        optimize_config_fileset: expect.stringMatching(
          new RegExp(`^${DEFAULT_WORKSPACE}/${AGENT}-optimize-`)
        ),
      },
    });
    const config = parse(uploaded.get('optimize.yaml')!);
    const baseUrl = `\${NHX_BASE_URL}/apis/inference-gateway/v2/workspaces/${DEFAULT_WORKSPACE}/openai/-/v1`;
    expect(config.models.default).toMatchObject({
      model: 'agent-model',
      base_url: baseUrl,
      api_key_env: 'NEMO_AGENTS_IGW_API_KEY',
    });
    expect(config.models.judge).toMatchObject({
      model: `${DEFAULT_WORKSPACE}/${entityStoreBaseModel1.name}`,
      base_url: baseUrl,
    });
    expect(config.models.judge).not.toHaveProperty('api_key_env');
    expect(config.models.judge).not.toHaveProperty('api_key_secret');
    expect(config.optimizer.search_space.gateway_credential).toEqual({
      type: 'fabric',
      path: 'environment.env.NEMO_AGENTS_IGW_API_KEY',
      values: ['not-used'],
    });
    expect(config.optimizer.numeric.n_trials).toBe(4);
    expect(config.optimizer.experiment_id).toBe('exp-quality');
    expect(JSON.parse(uploaded.get('dataset.json')!)).toEqual([
      { id: 'row-1', question: 'Classify this email', answer: 'phishing' },
    ]);
  });

  it('says why the evaluation cannot be staged before the user runs the study', async () => {
    renderRoute(undefined, {
      history: getAgentOptimizeRoute(DEFAULT_WORKSPACE, AGENT),
      routes: [
        {
          path: ROUTES.workspace.agentDetail,
          element: <SubmitForm evaluation={{ ...EVALUATION, metadata: {} }} />,
        },
      ],
    });

    expect(await screen.findByText(/has no stored eval config/)).toBeInTheDocument();
  });
});
