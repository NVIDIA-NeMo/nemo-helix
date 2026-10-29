// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { suppressConsoleError } from '@nemo/testing/utils/suppress-console';
import { CustomizationFilesetCreateModal } from '@studio/components/CustomizationFilesetCreateModal';
import { workspace1 } from '@studio/mocks/entity-store/projects';
import { mockUseParams } from '@studio/tests/util/mockUseParams';
import { render, screen, waitFor } from '@studio/tests/util/render';
import userEvent from '@testing-library/user-event';

describe('CustomizationFilesetCreateModal', () => {
  beforeEach(() => {
    mockUseParams({
      workspace: workspace1.workspace,
      projectNamespace: workspace1.workspace,
      projectName: workspace1.name,
    });
  });

  it('requires name, training file, and validation file', async () => {
    const user = userEvent.setup();
    // Form validation errors are logged via WebsiteLogger.error — expected in validation tests
    suppressConsoleError('Customization Fileset Create Form Errors');

    render(<CustomizationFilesetCreateModal onFilesetCreated={vi.fn()} onClose={vi.fn()} open />);

    const dialogSubmitButton = screen.getByRole('button', { name: 'Add to Customization' });
    // Button is always enabled - validation errors appear on submit attempt
    expect(dialogSubmitButton).not.toBeDisabled();

    // Try to submit empty form to trigger validation
    await user.click(dialogSubmitButton);

    // Expect validation errors for required fields
    const nameRequiredErrorMessage = await screen.findByText('Name is required.');
    const trainingFileRequiredErrorMessage = await screen.findByText('Training file is required');
    const validationFileRequiredErrorMessage = await screen.findByText(
      'Validation file is required'
    );
    expect(nameRequiredErrorMessage).toBeInTheDocument();
    expect(trainingFileRequiredErrorMessage).toBeInTheDocument();
    expect(validationFileRequiredErrorMessage).toBeInTheDocument();
  });

  it('requires the fileset name to contain at least one letter or number', async () => {
    const user = userEvent.setup();
    // Form validation errors are logged via WebsiteLogger.error — expected in validation tests
    suppressConsoleError('Customization Fileset Create Form Errors');

    render(<CustomizationFilesetCreateModal onFilesetCreated={vi.fn()} onClose={vi.fn()} open />);

    // Enter a name with no letters or numbers — entity naming strips invalid chars, so
    // only a name that sanitizes to empty triggers the error.
    const nameInput = await screen.findByRole('textbox', { name: 'Name' });
    await user.type(nameInput, '!!!');

    // Try to submit the form to trigger validation
    const submitButton = screen.getByRole('button', { name: 'Add to Customization' });
    await user.click(submitButton);

    const nameValidationErrorMessage = await screen.findByText(
      'Name must start with a letter and be at least 2 characters.'
    );
    expect(nameValidationErrorMessage).toBeInTheDocument();
  });

  it('submits the correct POST requests', async () => {
    const user = userEvent.setup();
    const onCreated = vi.fn();
    render(<CustomizationFilesetCreateModal onFilesetCreated={onCreated} onClose={vi.fn()} open />);

    // Fill out name and description
    const nameInput = await screen.findByRole('textbox', { name: 'Name' });
    await user.type(nameInput, 'test-fileset-name');
    const descriptionInput = screen.getByRole('textbox', { name: 'Description' });
    await user.type(descriptionInput, 'Test fileset description');

    // Upload training file
    const trainingFileInput = screen.getByLabelText('Training File(s)');
    const trainingFile = new File(['{}'], 'training_file.jsonl', {
      type: 'application/json',
    });
    await user.upload(trainingFileInput, trainingFile);

    // Upload validation file
    const validationFileInput = screen.getByLabelText('Validation File(s)');
    const validationFile = new File(['{}'], 'validation_file.jsonl', {
      type: 'application/json',
    });
    await user.upload(validationFileInput, validationFile);

    // Submit the dialog
    const dialogSubmitButton = screen.getByRole('button', { name: 'Add to Customization' });
    await user.click(dialogSubmitButton);
    await waitFor(() => {
      expect(onCreated).toBeCalledTimes(1);
    });
    await screen.findByText('Successfully created dataset!');
  });
});
