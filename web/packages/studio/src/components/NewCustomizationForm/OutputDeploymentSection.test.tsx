// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { zodResolver } from '@hookform/resolvers/zod';
import {
  DEPLOY_BY_DEFAULT,
  outputDeploymentDefaults,
} from '@studio/components/NewCustomizationForm/baseDeploymentForm';
import { OutputDeploymentSection } from '@studio/components/NewCustomizationForm/OutputDeploymentSection';
import {
  createDeploymentWizardSchema,
  type WizardFormValues,
} from '@studio/routes/NewDeploymentRoute/schema';
import { render, screen } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';
import type { FC } from 'react';
import { useForm } from 'react-hook-form';

const Harness: FC<{
  outputName?: string;
  deploy?: boolean;
  onDeployChange?: (deploy: boolean) => void;
}> = ({ outputName = 'my-adapter', deploy = true, onDeployChange = () => {} }) => {
  const f = useForm<WizardFormValues>({
    resolver: zodResolver(createDeploymentWizardSchema),
    defaultValues: outputDeploymentDefaults('default', outputName),
  });
  return (
    <OutputDeploymentSection
      control={f.control}
      errors={f.formState.errors}
      outputName={outputName}
      deployOutputModel={deploy}
      onDeployOutputModelChange={onDeployChange}
    />
  );
};

describe('OutputDeploymentSection', () => {
  it('names the model the run produces', () => {
    render(<Harness outputName="my-model" />);
    expect(screen.getByText(/This run produces my-model/)).toBeInTheDocument();
  });

  it('offers the deployment fields when deploying', () => {
    render(<Harness />);
    expect(screen.getByText('Engine')).toBeInTheDocument();
    expect(screen.getByText('GPUs')).toBeInTheDocument();
  });

  // The adapter flow hides this switch because a LoRA-disabled base refuses the
  // adapter. Nothing about a full-weight run depends on it, so it stays a choice.
  it('offers the LoRA Enabled switch, which is a real choice here', () => {
    render(<Harness />);
    expect(screen.getByText('LoRA Enabled')).toBeInTheDocument();
  });

  it('says the deployment is created when training finishes', () => {
    render(<Harness />);
    expect(screen.getByRole('switch', { name: /when training finishes/ })).toBeInTheDocument();
    expect(screen.getByText(/once training completes/)).toBeInTheDocument();
  });

  it('prompts for an output name before anything else', () => {
    render(<Harness outputName="" />);
    expect(screen.getByText(/Name the output model/)).toBeInTheDocument();
    expect(screen.queryByRole('switch')).not.toBeInTheDocument();
  });

  // Opt-out, the same as the adapter flow: a user who fine-tunes a model means to
  // use it, so deploying should not be the path that needs extra work.
  it('defaults to deploying', () => {
    expect(DEPLOY_BY_DEFAULT).toBe(true);
  });

  it('replaces the fields with a warning when the user opts out', () => {
    render(<Harness deploy={false} />);
    expect(screen.getByText(/nothing will serve it until/)).toBeInTheDocument();
    expect(screen.queryByText('Engine')).not.toBeInTheDocument();
    expect(screen.queryByText('GPUs')).not.toBeInTheDocument();
  });

  it('reports the choice back to the caller', async () => {
    const onDeployChange = vi.fn();
    const user = userEvent.setup();
    render(<Harness deploy={false} onDeployChange={onDeployChange} />);

    await user.click(screen.getByRole('switch'));
    expect(onDeployChange).toHaveBeenCalledWith(true);
  });
});

describe('outputDeploymentDefaults', () => {
  // The output Model Entity does not exist until the job finishes; the config is
  // created before the job starts. Pointing it forward is what lets the job deploy
  // the moment training ends, and `create_deployment_config` never requires the
  // referenced entity to exist.
  it('points modelRef forward at the not-yet-created output model', () => {
    expect(outputDeploymentDefaults('default', 'my-model').modelRef).toBe('default/my-model');
  });

  it('leaves modelRef empty when the run has no output name yet', () => {
    expect(outputDeploymentDefaults('default').modelRef).toBe('');
  });

  // Pinned true only for the adapter flow, where it is an invariant.
  it('does not pin loraEnabled', () => {
    const values = outputDeploymentDefaults('default', 'my-model');
    expect(values.loraEnabled).toBe(createDeploymentWizardSchema.parse(values).loraEnabled);
  });
});
