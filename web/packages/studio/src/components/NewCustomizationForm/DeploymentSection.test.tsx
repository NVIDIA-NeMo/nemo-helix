// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import {
  baseDeploymentDefaults,
  DEFAULT_DEPLOY_BASE_MODEL,
} from '@studio/components/NewCustomizationForm/baseDeploymentForm';
import { DeploymentSection } from '@studio/components/NewCustomizationForm/DeploymentSection';
import type { BaseModelDeploymentReadiness } from '@studio/hooks/useBaseModelDeploymentReadiness';
import {
  createDeploymentWizardSchema,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';
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
  deploy?: boolean;
  onDeployChange?: (deploy: boolean) => void;
}> = ({ r, modelRef = 'ws/base', deploy = true, onDeployChange = () => {} }) => {
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
      deployBaseModel={deploy}
      onDeployBaseModelChange={onDeployChange}
    />
  );
};

describe('DeploymentSection', () => {
  it('names the existing deployment and offers no controls when the base already serves LoRA', () => {
    render(<Harness r={readiness({ state: 'serving-lora', deploymentName: 'base-deployment' })} />);
    expect(screen.getByText(/base-deployment/)).toBeInTheDocument();
    expect(screen.queryByText('Engine')).not.toBeInTheDocument();
    // Nothing to create, so there is no toggle to offer.
    expect(screen.queryByRole('switch')).not.toBeInTheDocument();
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

  // The timing is the whole reason the copy changed: the job creates the deployment
  // after training rather than Studio creating it up front and idling a GPU.
  it('says the deployment is created when training finishes', () => {
    render(<Harness r={readiness({ state: 'none' })} />);
    expect(screen.getByRole('switch', { name: /when training finishes/ })).toBeInTheDocument();
    expect(screen.getByText(/once training\s+completes/)).toBeInTheDocument();
  });

  describe('opting out', () => {
    it('defaults to deploying', () => {
      expect(DEFAULT_DEPLOY_BASE_MODEL).toBe(true);
      render(<Harness r={readiness({ state: 'none' })} />);
      expect(screen.getByRole('switch')).toBeChecked();
    });

    it('replaces the fields with a warning when the user opts out', () => {
      render(<Harness r={readiness({ state: 'none' })} deploy={false} />);
      expect(screen.getByText(/will not be servable until/)).toBeInTheDocument();
      // Nothing to configure once there is no deployment to create.
      expect(screen.queryByText('Engine')).not.toBeInTheDocument();
      expect(screen.queryByText('GPUs')).not.toBeInTheDocument();
    });

    it('reports the choice back to the caller', async () => {
      const onDeployChange = vi.fn();
      const user = userEvent.setup();
      render(<Harness r={readiness({ state: 'none' })} onDeployChange={onDeployChange} />);

      await user.click(screen.getByRole('switch'));
      expect(onDeployChange).toHaveBeenCalledWith(false);
    });
  });
});
