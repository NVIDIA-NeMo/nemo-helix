// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import { getFilesListFilesetFilesQueryKey } from '@nemo/sdk/generated/platform/files';
import { SubmitEvaluationModal } from '@studio/components/evaluation/SubmitEvaluationModal';
import { evaluationSourcesHandlers } from '@studio/mocks/handlers/evaluationSources';
import { mockApiUrl } from '@studio/mocks/mockApiUrl';
import { server } from '@studio/mocks/node';
import {
  experimentFixture,
  reusableEvaluationFixture,
} from '@studio/tests/util/evaluationFixtures';
import { renderRoute, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';

const AGENT = 'my-agent';
const OTHER_AGENT = 'baseline-agent';

/** Two experiments that both contain a run called "baseline", which is what makes the grouped
 *  picker's typeahead worth testing: one term has to reach across sections. */
const EXPERIMENTS = [
  experimentFixture('grp_primary', 'primary-use-cases-benchmark'),
  experimentFixture('grp_regression', 'regression-sweep'),
];

const EVALUATIONS = [
  reusableEvaluationFixture('baseline', 'grp_primary', AGENT),
  reusableEvaluationFixture('nemotron-super-3-temp-point5', 'grp_primary', AGENT),
  reusableEvaluationFixture('baseline-regression', 'grp_regression', AGENT),
];

const OTHER_AGENT_EVALUATION = reusableEvaluationFixture(
  'other-agent-baseline',
  'grp_regression',
  OTHER_AGENT
);

const mockLists = (evaluations = EVALUATIONS) => {
  server.use(...evaluationSourcesHandlers({ experiments: EXPERIMENTS, evaluations }));
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

  it('offers a parallel-requests setting that defaults to 4', async () => {
    mockLists();
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole('radio', { name: /Create a new experiment/ }));
    await user.click(screen.getByRole('button', { name: 'Next' }));
    await user.type(await screen.findByLabelText('Name'), 'model-update-tests');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled());
    await user.click(screen.getByRole('button', { name: 'Next' }));

    expect(await screen.findByRole('spinbutton', { name: 'Parallel requests' })).toHaveValue(4);
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
    expect(screen.getByText('Run Evaluation')).toBeInTheDocument();
    expect(screen.queryByText('Create experiment')).not.toBeInTheDocument();

    await user.click(screen.getByRole('radio', { name: /Create a new experiment/ }));

    // The new-experiment path gains its own step, and all three are named up front.
    expect(await screen.findByText('Create experiment')).toBeInTheDocument();
    expect(screen.getByText('Begin')).toBeInTheDocument();
    expect(screen.getByText('Run Evaluation')).toBeInTheDocument();
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

  describe("other agents' evaluations", () => {
    const INCLUDE_OTHERS = /include other agents' evaluations/i;

    const openRerunStep = async (user: ReturnType<typeof userEvent.setup>) => {
      await user.click(await screen.findByRole('radio', { name: /Re-run an existing evaluation/ }));
      await user.click(screen.getByRole('button', { name: 'Next' }));
      return screen.findByRole('checkbox', { name: INCLUDE_OTHERS });
    };

    it('keeps them out of the picker until asked for', async () => {
      mockLists([...EVALUATIONS, OTHER_AGENT_EVALUATION]);
      const user = userEvent.setup();
      renderModal();

      const includeOthers = await openRerunStep(user);
      expect(includeOthers).not.toBeChecked();
      await user.click(screen.getByRole('combobox', { name: /evaluation to re-run/i }));
      expect(await screen.findByRole('option', { name: 'baseline' })).toBeInTheDocument();
      expect(
        screen.queryByRole('option', { name: 'other-agent-baseline' })
      ).not.toBeInTheDocument();
      await user.keyboard('{Escape}');

      await user.click(includeOthers);
      await user.click(screen.getByRole('combobox', { name: /evaluation to re-run/i }));
      expect(
        await screen.findByRole('option', { name: 'other-agent-baseline' })
      ).toBeInTheDocument();
    });

    it('offers them by default to an agent with no evaluations of its own', async () => {
      mockLists([OTHER_AGENT_EVALUATION]);
      server.use(
        http.get(mockApiUrl(getFilesListFilesetFilesQueryKey, ':workspace', ':name'), () =>
          HttpResponse.json({ data: [{ path: 'eval-config.yaml' }] })
        )
      );
      const user = userEvent.setup();
      renderModal();

      await waitFor(() =>
        expect(screen.getByRole('radio', { name: /Re-run an existing evaluation/ })).toBeEnabled()
      );
      const includeOthers = await openRerunStep(user);
      await waitFor(() => expect(includeOthers).toBeChecked());

      const picker = screen.getByRole('combobox', { name: /evaluation to re-run/i });
      await user.click(picker);
      await user.click(await screen.findByRole('option', { name: 'other-agent-baseline' }));
      await waitFor(() =>
        expect(picker).toHaveAccessibleDescription(
          /Experiment: regression-sweep · Agent: baseline-agent/
        )
      );
    });

    it('drops a picked run of another agent once they are hidden again', async () => {
      mockLists([...EVALUATIONS, OTHER_AGENT_EVALUATION]);
      const user = userEvent.setup();
      renderModal();

      const includeOthers = await openRerunStep(user);
      await user.click(includeOthers);
      await user.click(screen.getByRole('combobox', { name: /evaluation to re-run/i }));
      await user.click(await screen.findByRole('option', { name: 'other-agent-baseline' }));
      await waitFor(() =>
        expect(screen.queryByText('Pick an evaluation to re-run.')).not.toBeInTheDocument()
      );

      await user.click(includeOthers);
      expect(await screen.findByText('Pick an evaluation to re-run.')).toBeInTheDocument();
    });
  });
});
