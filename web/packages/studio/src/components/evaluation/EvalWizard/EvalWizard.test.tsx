// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { EvalWizard } from '@studio/components/evaluation/EvalWizard';
import { renderRoute, screen } from '@studio/tests/util/render';
import { fireEvent } from '@testing-library/react';

const mountPage = (props: { continueLoading?: boolean; canContinue?: boolean }) => {
  const onSubmit = vi.fn((event: React.FormEvent) => event.preventDefault());
  renderRoute(undefined, {
    history: '/workspaces/default',
    routes: [
      {
        path: '/workspaces/:workspace',
        element: (
          <EvalWizard
            heading="Run Evaluation"
            headingDescription="Description"
            steps={['Begin', 'Run']}
            activeStep={1}
            cancelTo="/workspaces/default"
            onSubmit={onSubmit}
            continueLabel="Run Evaluation"
            canContinue={props.canContinue ?? true}
            continueLoading={props.continueLoading}
            onContinue={vi.fn()}
          >
            <input aria-label="Name" />
          </EvalWizard>
        ),
      },
    ],
  });
  return { onSubmit };
};

/** What the browser does on Enter in a field: submit the field's form. */
const pressEnter = () => fireEvent.submit(screen.getByRole('form', { name: 'Run Evaluation' }));

describe('EvalWizard', () => {
  it('submits on Enter when Continue is available', () => {
    const { onSubmit } = mountPage({});
    pressEnter();
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it('ignores Enter while a submission is running, so it cannot start a second one', () => {
    const { onSubmit } = mountPage({ continueLoading: true });
    pressEnter();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('ignores Enter while Continue is disabled', () => {
    const { onSubmit } = mountPage({ canContinue: false });
    pressEnter();
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
