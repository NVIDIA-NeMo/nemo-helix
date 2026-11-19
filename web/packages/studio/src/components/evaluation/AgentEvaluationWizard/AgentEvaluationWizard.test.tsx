// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import { getListExperimentsQueryKey } from '@nemo/sdk/generated/platform/experiments';
import { getFilesListFilesetFilesQueryKey } from '@nemo/sdk/generated/platform/files';
import type { EvaluationResponse, ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { AgentEvaluationWizard } from '@studio/components/evaluation/AgentEvaluationWizard';
import { EVAL_CONFIG_FILESET_KEY } from '@studio/components/evaluation/experimentEvalConfig';
import { ROUTES } from '@studio/constants/routes';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { getAgentEvaluationsTabRoute } from '@studio/routes/utils';
import { LOCATION_DISPLAY_TEST_ID } from '@studio/tests/util/constants';
import { LocationDisplay } from '@studio/tests/util/LocationDisplay';
import { renderRoute, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

vi.mock('@studio/components/filesets/hooks/useDownloadFileAsArrayBuffer', () => ({
  useFetchFileAsArrayBuffer: () => async () =>
    new TextEncoder().encode('{"prompt": "hi", "expected": "hi"}\n').buffer,
}));

vi.mock('@nemo/common/src/components/DatasetFileSelect/ControlledDatasetFileSelect', async () =>
  (await import('@studio/tests/util/mockDatasetPick')).datasetFileSelectMock()
);

vi.mock('@studio/components/evaluation/EvalWizard/useDatasetPreview', async (importOriginal) =>
  (await import('@studio/tests/util/mockDatasetPick')).datasetPreviewMock(await importOriginal(), {
    prompt: 'hi',
    expected: 'hi',
  })
);

// The fixture filesets hold no eval config, so the real check would report every run as unusable.
const AGENT = 'my-agent';

const emptyPage = { data: [], pagination: { total: 0, page: 1, page_size: 100 } };

const experiment = (id: string, name: string): ExperimentResponse => ({
  id,
  name,
  workspace: DEFAULT_WORKSPACE,
  default_sort: '-created_at',
  evaluation_count: 1,
});

const evaluation = (name: string, experimentId: string): EvaluationResponse => ({
  id: `eval_${name}`,
  name,
  workspace: DEFAULT_WORKSPACE,
  experiment_ids: [experimentId],
  experiment_group_id: experimentId,
  dataset_name: 'ds',
  agent_names: [AGENT],
  metadata: { [EVAL_CONFIG_FILESET_KEY]: `${name}-data` },
});

/** Two experiments that both contain a run called "baseline", which is what makes the grouped
 *  picker's typeahead worth testing: one term has to reach across sections. */
const EXPERIMENTS = [
  experiment('grp_primary', 'primary-use-cases-benchmark'),
  experiment('grp_regression', 'regression-sweep'),
];

const EVALUATIONS = [
  evaluation('baseline', 'grp_primary'),
  evaluation('nemotron-super-3-temp-point5', 'grp_primary'),
  evaluation('baseline-regression', 'grp_regression'),
];

const mockLists = () => {
  server.use(
    http.get(mockApiUrl(getListExperimentsQueryKey, ':workspace'), ({ request }) => {
      // The name-conflict probe asks for one exact name; everything else is the group lookup.
      const name = new URL(request.url).searchParams.get('filter[name]');
      return HttpResponse.json({
        data: name ? EXPERIMENTS.filter((item) => item.name === name) : EXPERIMENTS,
      });
    }),
    http.get(mockApiUrl(getListEvaluationsQueryKey, ':workspace'), ({ request }) => {
      const url = new URL(request.url);
      // The name-conflict probe asks for one exact name; everything else is the picker's list.
      const name = url.searchParams.get('filter[name]');
      if (name) {
        return HttpResponse.json({
          ...emptyPage,
          data: EVALUATIONS.filter((item) => item.name === name),
        });
      }
      return HttpResponse.json({ data: EVALUATIONS });
    }),
    http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () =>
      HttpResponse.json({ data: [] })
    )
  );
};

describe('AgentEvaluationWizard', () => {
  it("cancels to the agent's evaluations when opened by a direct link", async () => {
    mockLists();
    const user = userEvent.setup();
    renderRoute(undefined, {
      history: `/workspaces/${DEFAULT_WORKSPACE}`,
      routes: [
        {
          path: '/workspaces/:workspace',
          element: <AgentEvaluationWizard workspace={DEFAULT_WORKSPACE} agent={AGENT} />,
        },
        { path: ROUTES.workspace.agentDetail, element: <LocationDisplay /> },
      ],
    });

    await user.click(await screen.findByRole('button', { name: 'Cancel' }));

    expect(await screen.findByTestId(LOCATION_DISPLAY_TEST_ID)).toHaveTextContent(
      getAgentEvaluationsTabRoute(DEFAULT_WORKSPACE, AGENT)
    );
  });
});
