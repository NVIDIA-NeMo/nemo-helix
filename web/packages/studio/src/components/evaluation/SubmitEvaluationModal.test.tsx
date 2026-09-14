// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { getListEvaluationsQueryKey } from '@nemo/sdk/generated/platform/evaluations';
import { getListExperimentsQueryKey } from '@nemo/sdk/generated/platform/experiments';
import type { EvaluationResponse, ExperimentResponse } from '@nemo/sdk/generated/platform/schema';
import { EVAL_CONFIG_FILESET_KEY } from '@studio/components/evaluation/experimentEvalConfig';
import { SubmitEvaluationModal } from '@studio/components/evaluation/SubmitEvaluationModal';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

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
    })
  );
};

const renderModal = (props: Partial<React.ComponentProps<typeof SubmitEvaluationModal>> = {}) =>
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
            agent={AGENT}
            {...props}
          />
        ),
      },
    ],
  });

describe('SubmitEvaluationModal', () => {
  it('starts by asking how to begin, and routes to the experiment form for a new experiment', async () => {
    mockLists();
    const user = userEvent.setup();
    renderModal();

    expect(await screen.findByText('How do you want to start?')).toBeInTheDocument();
    // Neither path's fields are on screen until the choice is made.
    expect(screen.queryByLabelText('Name')).not.toBeInTheDocument();

    await user.click(screen.getByRole('radio', { name: /Create a new experiment/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));

    // The experiment step is the Experiments page's own form: name plus every setting.
    expect(await screen.findByLabelText('Name')).toBeInTheDocument();
    expect(screen.getByLabelText('Description (Optional)')).toBeInTheDocument();
    expect(screen.getByRole('switch', { name: /evaluate over time/i })).toBeInTheDocument();
    expect(screen.getByRole('switch', { name: /favorite/i })).toBeInTheDocument();
  });

  it("does not carry the re-run path's derived name onto a new experiment", async () => {
    mockLists();
    const user = userEvent.setup();
    renderModal();

    // Seed a name on the re-run path first: pick a source, let it prefill, then switch paths.
    await screen.findByText('How do you want to start?');
    await user.click(screen.getByRole('radio', { name: /Re-run an existing evaluation/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));

    await user.click(await screen.findByRole('combobox', { name: /evaluation to re-run/i }));
    await user.click(await screen.findByRole('option', { name: 'nemotron-super-3-temp-point5' }));
    const seeded = await screen.findByLabelText<HTMLInputElement>('New Evaluation Name');
    await waitFor(() => expect(seeded).toHaveValue('nemotron-super-3-temp-point5'));

    await user.click(screen.getByRole('button', { name: 'Back' }));
    await user.click(await screen.findByRole('radio', { name: /Create a new experiment/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));

    await user.type(await screen.findByLabelText('Name'), 'model-update-tests');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Next' }));

    // A borrowed name would say this run is a repeat of some other experiment's run.
    expect(await screen.findByLabelText<HTMLInputElement>('Evaluation Name')).toHaveValue('');
  });

  it('will not advance past the experiment step without a name', async () => {
    mockLists();
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('radio', { name: /Create a new experiment/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));

    await screen.findByLabelText('Name');
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();

    await user.type(screen.getByLabelText('Name'), 'model-update-tests');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
  });

  it('names every step under its dot, not just the one in progress', async () => {
    mockLists();
    const user = userEvent.setup();
    renderModal();

    // Re-running has no experiment to set up, so it is two steps.
    expect(await screen.findByText('Begin')).toBeInTheDocument();
    expect(screen.getByText('Create evaluation')).toBeInTheDocument();
    expect(screen.queryByText('Create experiment')).not.toBeInTheDocument();

    await user.click(screen.getByRole('radio', { name: /Create a new experiment/ }));

    // The new-experiment path gains its own step, and all three are named up front.
    expect(await screen.findByText('Create experiment')).toBeInTheDocument();
    expect(screen.getByText('Begin')).toBeInTheDocument();
    expect(screen.getByText('Create evaluation')).toBeInTheDocument();
  });

  it("puts the new run's name under the picker it is derived from", async () => {
    mockLists();
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('radio', { name: /Re-run an existing evaluation/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));

    // One screen: pick the run to re-run, then name the run that pick produces.
    const picker = await screen.findByRole('combobox', { name: /evaluation to re-run/i });
    const nameField = screen.getByLabelText('New Evaluation Name');
    expect(
      picker.compareDocumentPosition(nameField) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();
  });

  it('routes the re-run path to a picker grouped by experiment', async () => {
    mockLists();
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('radio', { name: /Re-run an existing evaluation/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));

    await user.click(await screen.findByRole('combobox', { name: /evaluation to re-run/i }));
    expect(await screen.findByText('primary-use-cases-benchmark')).toBeInTheDocument();
    expect(await screen.findByText('regression-sweep')).toBeInTheDocument();

    // Searching an experiment's name narrows to that one section, including the runs under it
    // whose own names share nothing with the term.
    await user.type(screen.getByTestId('evaluationName-search'), 'primary-use-cases-benchmark');
    await waitFor(() =>
      expect(screen.queryByRole('option', { name: 'baseline-regression' })).not.toBeInTheDocument()
    );
    expect(
      screen.getByRole('option', { name: 'nemotron-super-3-temp-point5' })
    ).toBeInTheDocument();
  });

  it('opens on the last step for a handed-in source, with an editable derived name', async () => {
    mockLists();
    renderModal({ sourceEvaluation: 'nemotron-super-3-temp-point5' });

    // Straight to naming the run — the first two answers came in with the source.
    const nameField = await screen.findByLabelText<HTMLInputElement>('New Evaluation Name');
    await waitFor(() => expect(nameField.value).toBe('nemotron-super-3-temp-point5'));
    // The picker sits on this step too, so the name appears in its options as well; the point is
    // that the help text under it states which experiment the chosen run belongs to.
    await waitFor(() =>
      expect(
        screen.getByRole('combobox', { name: /evaluation to re-run/i })
      ).toHaveAccessibleDescription(/Experiment: primary-use-cases-benchmark/)
    );

    const user = userEvent.setup();
    await user.clear(nameField);
    await user.type(nameField, 'nemotron-super-3-temp-1');
    expect(nameField.value).toBe('nemotron-super-3-temp-1');
  });

  it('lets Back walk out of a handed-in source so the choice stays reviewable', async () => {
    mockLists();
    const user = userEvent.setup();
    renderModal({ sourceEvaluation: 'baseline' });

    await screen.findByLabelText('New Evaluation Name');
    await user.click(screen.getByRole('button', { name: 'Back' }));

    expect(await screen.findByText('How do you want to start?')).toBeInTheDocument();
  });
});
