// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import {
  BASE_DEPLOYMENT_DEPLOY,
  BASE_DEPLOYMENT_SKIP,
  baseDeploymentDefaults,
  type BaseDeploymentChoice,
} from '@studio/components/NewCustomizationForm/baseDeploymentForm';
import { DeploymentSection } from '@studio/components/NewCustomizationForm/DeploymentSection';
import type { BaseModelDeploymentReadiness } from '@studio/hooks/useBaseModelDeploymentReadiness';
import {
  createDeploymentWizardSchema,
  type WizardFormValues,
} from '@studio/routes/DeploymentsListRoute/CreateDeploymentSidePanel/schema';
import { render, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import type { FC } from 'react';
import { useForm } from 'react-hook-form';

const readiness = (
  over: Partial<BaseModelDeploymentReadiness> = {}
): BaseModelDeploymentReadiness => ({
  state: 'none',
  deploymentName: null,
  status: null,
  isLoading: false,
  ...over,
});

const Harness: FC<{
  r: BaseModelDeploymentReadiness;
  modelRef?: string;
  choice?: BaseDeploymentChoice;
  onChoiceChange?: (c: BaseDeploymentChoice) => void;
}> = ({ r, modelRef = 'ws/base', choice = BASE_DEPLOYMENT_DEPLOY, onChoiceChange = () => {} }) => {
  const f = useForm<WizardFormValues>({
    resolver: zodResolver(createDeploymentWizardSchema),
    defaultValues: baseDeploymentDefaults(modelRef),
  });
  return (
    <DeploymentSection
      readiness={r}
      control={f.control}
      errors={f.formState.errors}
      baseModelRef={modelRef}
      choice={choice}
      onChoiceChange={onChoiceChange}
    />
  );
};

describe('DeploymentSection', () => {
  it('names the existing deployment and offers no controls when the base already serves LoRA', () => {
    render(<Harness r={readiness({ state: 'serving-lora', deploymentName: 'base-deployment' })} />);
    expect(screen.getByText(/base-deployment/)).toBeInTheDocument();
    expect(screen.queryByText('Engine')).not.toBeInTheDocument();
    // Nothing to create, so the timing question does not arise.
    expect(screen.queryByRole('radio', { name: /Now/ })).not.toBeInTheDocument();
  });

  it('offers the deployment fields when the base is not deployed', () => {
    render(<Harness r={readiness({ state: 'none' })} />);
    expect(screen.getByText(/is not deployed/)).toBeInTheDocument();
    expect(screen.getByText('Engine')).toBeInTheDocument();
  });

  // The case that looks fine but is not: base is up, adapter will be refused.
  it('explains that a LoRA-disabled deployment will refuse the adapter', () => {
    render(<Harness r={readiness({ state: 'serving-without-lora' })} />);
    expect(screen.getByText(/LoRA disabled/)).toBeInTheDocument();
    expect(screen.getByText('Engine')).toBeInTheDocument();
  });

  // Pinned true by `baseDeploymentDefaults`. A base deployed without LoRA support
  // refuses the adapter this run produces, so it is not a choice to offer — and a
  // disabled switch would still read as a decision the user could revisit.
  it('does not offer the LoRA Enabled switch, which is fixed for this flow', () => {
    render(<Harness r={readiness({ state: 'none' })} />);
    expect(screen.getByText('GPUs')).toBeInTheDocument();
    expect(screen.queryByText('LoRA Enabled')).not.toBeInTheDocument();
  });

  it('warns about a failed existing deployment without hiding the fields', () => {
    render(<Harness r={readiness({ state: 'unavailable', status: 'ERROR' })} />);
    expect(screen.getByText(/failed state/)).toBeInTheDocument();
    expect(screen.getByText('Engine')).toBeInTheDocument();
  });

  it('prompts for a model before anything else', () => {
    render(<Harness r={readiness()} modelRef="" />);
    expect(screen.getByText(/Select a base model/)).toBeInTheDocument();
  });

  it('shows a loading note while readiness resolves', () => {
    render(<Harness r={readiness({ isLoading: true })} />);
    expect(screen.getByText(/Checking whether/)).toBeInTheDocument();
  });

  describe('opting out', () => {
    it('defaults to deploying', () => {
      render(<Harness r={readiness({ state: 'none' })} />);
      expect(screen.getByRole('radio', { name: /Deploy the base model/ })).toBeChecked();
      expect(screen.getByRole('radio', { name: /Don't deploy/ })).not.toBeChecked();
    });

    it('replaces the fields with a warning when the user opts out', () => {
      render(<Harness r={readiness({ state: 'none' })} choice={BASE_DEPLOYMENT_SKIP} />);
      expect(screen.getByText(/will not be servable until/)).toBeInTheDocument();
      // Nothing to configure once there is no deployment to create.
      expect(screen.queryByText('Engine')).not.toBeInTheDocument();
      expect(screen.queryByText('GPUs')).not.toBeInTheDocument();
    });

    it('reports the choice back to the caller', async () => {
      const onChoiceChange = vi.fn();
      const user = userEvent.setup();
      render(<Harness r={readiness({ state: 'none' })} onChoiceChange={onChoiceChange} />);

      await user.click(screen.getByRole('radio', { name: /Don't deploy/ }));
      expect(onChoiceChange).toHaveBeenCalledWith(BASE_DEPLOYMENT_SKIP);
    });
  });
});
