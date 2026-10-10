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
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { useLocation } from 'react-router';
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

const LocationSearch = () => <div data-testid="location-search">{useLocation().search}</div>;

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
  it('stages the optimize config and hands the job the evaluation config to score with', async () => {
    const uploaded = new Map<string, string>();
    const submitted: unknown[] = [];
    server.use(
      http.get(`${FILESETS_URL}/eval-data/files`, () =>
        HttpResponse.json({ data: [{ path: 'eval-config.json' }] })
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
        {
          path: ROUTES.workspace.agentDetail,
          element: (
            <>
              <SubmitForm />
              <LocationSearch />
            </>
          ),
        },
      ],
    });

    await screen.findByText('baseline');
    await user.click(screen.getByRole('radio', { name: /Quick/ }));
    await user.click(screen.getByRole('button', { name: 'Run optimization' }));

    await waitFor(() =>
      expect(screen.getByTestId('location-search')).toHaveTextContent(/^\?tab=optimizations$/)
    );
    expect(submitted).toHaveLength(1);
    expect(submitted[0]).toMatchObject({
      spec: {
        agent: AGENT,
        strategy: 'legacy',
        optimize_config: 'optimize.yaml',
        optimize_config_fileset: expect.stringMatching(
          new RegExp(`^${DEFAULT_WORKSPACE}/${AGENT}-optimize-`)
        ),
        evaluation_config: `${DEFAULT_WORKSPACE}/eval-data#eval-config.json`,
      },
    });
    expect([...uploaded.keys()]).toEqual(['optimize.yaml']);
    const config = parse(uploaded.get('optimize.yaml')!);
    const baseUrl = `\${NHX_BASE_URL}/apis/inference-gateway/v2/workspaces/${DEFAULT_WORKSPACE}/openai/-/v1`;
    expect(config.models).toEqual({
      default: expect.objectContaining({
        model: 'agent-model',
        base_url: baseUrl,
        api_key_env: 'NEMO_AGENTS_IGW_API_KEY',
      }),
    });
    expect(config.eval).toBeUndefined();
    expect(config.optimizer.eval_metrics.average_score.direction).toBe('maximize');
    expect(config.optimizer.search_space.gateway_credential).toEqual({
      type: 'fabric',
      path: 'environment.env.NEMO_AGENTS_IGW_API_KEY',
      values: ['not-used'],
    });
    expect(config.optimizer.numeric.n_trials).toBe(4);
    expect(config.optimizer.experiment_id).toBe('exp-quality');
  });

  it('says why an evaluation without a stored config cannot score the study', async () => {
    const user = userEvent.setup();
    renderRoute(undefined, {
      history: getAgentOptimizeRoute(DEFAULT_WORKSPACE, AGENT),
      routes: [
        {
          path: ROUTES.workspace.agentDetail,
          element: <SubmitForm evaluation={{ ...EVALUATION, metadata: {} }} />,
        },
      ],
    });

    await screen.findByText('baseline');
    await user.click(screen.getByRole('button', { name: 'Run optimization' }));

    expect(await screen.findByText(/has no stored eval config/)).toBeInTheDocument();
  });
});
