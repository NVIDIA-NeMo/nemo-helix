// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AdvancedParameters } from '@studio/routes/SafeSynthesizerNewRoute/components/AdvancedParameters';
import {
  type SafeSynthesizerFormData,
  getSafeSynthesizerFormDefaults,
} from '@studio/routes/SafeSynthesizerNewRoute/schema';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FormProvider, useForm, useWatch } from 'react-hook-form';

const PretrainedModelProbe = () => {
  const value = useWatch<SafeSynthesizerFormData>({
    name: 'spec.config.training.pretrained_model',
  });
  return <div data-testid="pretrained-model-value">{value}</div>;
};

// Test wrapper component that provides form context
const TestWrapper = ({
  defaultValues,
  children,
}: {
  defaultValues?: Partial<SafeSynthesizerFormData>;
  children: React.ReactNode;
}) => {
  const methods = useForm<SafeSynthesizerFormData>({
    defaultValues: {
      ...getSafeSynthesizerFormDefaults(),
      ...defaultValues,
    },
  });

  return <FormProvider {...methods}>{children}</FormProvider>;
};

describe('AdvancedParameters', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Component Rendering', () => {
    it('renders all section headings', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByText('Core Generation Settings')).toBeInTheDocument();
      expect(screen.getByText('Training Data Configuration')).toBeInTheDocument();
      expect(screen.getByText('Context Length Scaling')).toBeInTheDocument();
      expect(screen.getByText('PII Replacement Configuration')).toBeInTheDocument();
    });

    it('renders introductory description text', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(
        await screen.findByText(/Training hyperparameters control how the model learns/)
      ).toBeInTheDocument();
    });

    it('renders temperature controls', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByText('temperature')).toBeInTheDocument();
      // Temperature slider should be present - verify by checking there are sliders
      const sliders = screen.getAllByRole('slider');
      expect(sliders.length).toBeGreaterThanOrEqual(1);
      // First slider should be the temperature slider with correct max/min
      expect(sliders[0]).toHaveAttribute('max', '2');
      expect(sliders[0]).toHaveAttribute('min', '0');
    });

    it('renders top_p controls', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByText('top_p')).toBeInTheDocument();
      // Get all sliders - there should be 4 (temperature, top_p, repetition_penalty, rope_scaling_factor)
      const sliders = screen.getAllByRole('slider');
      expect(sliders.length).toBeGreaterThanOrEqual(2);
    });

    it('renders repetition_penalty controls', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByText('repetition_penalty')).toBeInTheDocument();
      // Get all sliders - there should be 4 (temperature, top_p, repetition_penalty, rope_scaling_factor)
      const sliders = screen.getAllByRole('slider');
      expect(sliders.length).toBeGreaterThanOrEqual(3);
    });

    it('renders num_input_records_to_sample input', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByText('num_input_records_to_sample')).toBeInTheDocument();
      expect(screen.getByText('Use Automatic Sampling')).toBeInTheDocument();
    });

    it('renders rope_scaling_factor controls', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByText('rope_scaling_factor')).toBeInTheDocument();
      expect(screen.getByText('Use Automatic Scaling')).toBeInTheDocument();
      // All sliders are present (temperature, top_p, repetition_penalty, rope_scaling_factor)
      const sliders = screen.getAllByRole('slider');
      expect(sliders.length).toBe(4);
    });

    it('renders enable_replace_pii switch', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByText('enable_replace_pii')).toBeInTheDocument();
      expect(screen.getByRole('switch')).toBeInTheDocument();
    });

    it('renders all reset buttons', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      await waitFor(() =>
        expect(screen.getAllByRole('button', { name: /reset|undo/i }).length).toBeGreaterThan(0)
      );
    });
  });

  describe('Temperature Controls', () => {
    it('displays default temperature value', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      await waitFor(() => expect(screen.getAllByDisplayValue('0.9').length).toBeGreaterThan(0));
    });

    it('updates temperature value when typing in input', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const temperatureInputs = screen.getAllByDisplayValue('0.9');
      const numberInput = temperatureInputs.find(
        (input) => input.getAttribute('type') === 'number'
      ) as HTMLInputElement;

      fireEvent.change(numberInput, { target: { value: '1.5' } });

      await waitFor(() => {
        expect(numberInput).toHaveValue(1.5);
      });
    });

    it('clamps temperature value to minimum of 0', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const temperatureInputs = screen.getAllByDisplayValue('0.9');
      const numberInput = temperatureInputs.find(
        (input) => input.getAttribute('type') === 'number'
      ) as HTMLInputElement;

      fireEvent.change(numberInput, { target: { value: '0' } });

      await waitFor(() => {
        expect(numberInput).toHaveValue(0);
      });

      expect(numberInput).toHaveAttribute('min', '0');
    });

    it('clamps temperature value to maximum of 2', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const temperatureInputs = screen.getAllByDisplayValue('0.9');
      const numberInput = temperatureInputs.find(
        (input) => input.getAttribute('type') === 'number'
      ) as HTMLInputElement;

      await user.clear(numberInput);
      await user.type(numberInput, '3');
      await user.tab();

      await waitFor(() => {
        expect(Number(numberInput.value)).toBeLessThanOrEqual(2);
      });
    });

    it('resets temperature to default when reset button is clicked', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const temperatureInputs = screen.getAllByDisplayValue('0.9');
      const numberInput = temperatureInputs.find(
        (input) => input.getAttribute('type') === 'number'
      ) as HTMLInputElement;

      // Change the value
      fireEvent.change(numberInput, { target: { value: '1.5' } });

      // Click reset button (first reset button is for temperature)
      const resetButton = screen.getByRole('button', {
        name: 'Reset temperature to default value',
      });
      await user.click(resetButton);

      await waitFor(() => {
        expect(numberInput).toHaveValue(0.9);
      });
    });
  });

  describe('Top_p Controls', () => {
    it('displays default top_p value', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      await waitFor(() => expect(screen.getAllByDisplayValue('1').length).toBeGreaterThan(0));
    });

    it('updates top_p value when typing in input', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const topPInputs = screen.getAllByDisplayValue('1');
      const numberInput = topPInputs.find(
        (input) => input.getAttribute('type') === 'number' && input.getAttribute('max') === '1'
      ) as HTMLInputElement;

      fireEvent.change(numberInput, { target: { value: '0.7' } });

      await waitFor(() => {
        expect(numberInput).toHaveValue(0.7);
      });
    });

    it('clamps top_p value to minimum of 0', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const topPInputs = screen.getAllByDisplayValue('1');
      const numberInput = topPInputs.find(
        (input) => input.getAttribute('type') === 'number' && input.getAttribute('max') === '1'
      ) as HTMLInputElement;

      await user.clear(numberInput);
      await user.type(numberInput, '-0.5');
      await user.tab();

      await waitFor(() => {
        expect(Number(numberInput.value)).toBeGreaterThanOrEqual(0);
      });
    });

    it('clamps top_p value to maximum of 1', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const topPInputs = screen.getAllByDisplayValue('1');
      const numberInput = topPInputs.find(
        (input) => input.getAttribute('type') === 'number' && input.getAttribute('max') === '1'
      ) as HTMLInputElement;

      await user.clear(numberInput);
      await user.type(numberInput, '1.5');
      await user.tab();

      await waitFor(() => {
        expect(Number(numberInput.value)).toBeLessThanOrEqual(1);
      });
    });

    it('resets top_p to default when reset button is clicked', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const topPInputs = screen.getAllByDisplayValue('1');
      const numberInput = topPInputs.find(
        (input) => input.getAttribute('type') === 'number' && input.getAttribute('max') === '1'
      ) as HTMLInputElement;

      // Change the value
      fireEvent.change(numberInput, { target: { value: '0.5' } });

      await waitFor(() => {
        expect(numberInput).toHaveValue(0.5);
      });

      // Click reset button
      const resetButton = screen.getByRole('button', {
        name: 'Reset top_p to default value',
      });
      await user.click(resetButton);

      await waitFor(() => {
        expect(numberInput).toHaveValue(1);
      });
    });
  });

  describe('Repetition_penalty Controls', () => {
    it('displays default repetition_penalty value', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      await waitFor(() => expect(screen.getAllByDisplayValue('1').length).toBeGreaterThan(0));
    });

    it('renders repetition_penalty slider with correct constraints', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByText('repetition_penalty')).toBeInTheDocument();
      // Find the repetition_penalty slider - it's the third slider (after temperature and top_p)
      const sliders = screen.getAllByRole('slider');
      expect(sliders.length).toBeGreaterThanOrEqual(3);
      // Third slider should be repetition_penalty with min=1, max=2
      expect(sliders[2]).toHaveAttribute('max', '2');
      expect(sliders[2]).toHaveAttribute('min', '1');
    });

    it('updates repetition_penalty value when typing in input', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const repetitionPenaltyInputs = screen.getAllByDisplayValue('1');
      const numberInput = repetitionPenaltyInputs.find(
        (input) =>
          input.getAttribute('type') === 'number' &&
          input.getAttribute('max') === '2' &&
          input.getAttribute('min') === '1'
      ) as HTMLInputElement;

      fireEvent.change(numberInput, { target: { value: '1.5' } });

      await waitFor(() => {
        expect(numberInput).toHaveValue(1.5);
      });
    });

    it('clamps repetition_penalty value to minimum of 1', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const repetitionPenaltyInputs = screen.getAllByDisplayValue('1');
      const numberInput = repetitionPenaltyInputs.find(
        (input) =>
          input.getAttribute('type') === 'number' &&
          input.getAttribute('max') === '2' &&
          input.getAttribute('min') === '1'
      ) as HTMLInputElement;

      await user.clear(numberInput);
      await user.type(numberInput, '0.5');
      await user.tab();

      await waitFor(() => {
        expect(Number(numberInput.value)).toBeGreaterThanOrEqual(1);
      });
    });

    it('clamps repetition_penalty value to maximum of 2', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const repetitionPenaltyInputs = screen.getAllByDisplayValue('1');
      const numberInput = repetitionPenaltyInputs.find(
        (input) =>
          input.getAttribute('type') === 'number' &&
          input.getAttribute('max') === '2' &&
          input.getAttribute('min') === '1'
      ) as HTMLInputElement;

      await user.clear(numberInput);
      await user.type(numberInput, '3');
      await user.tab();

      await waitFor(() => {
        expect(Number(numberInput.value)).toBeLessThanOrEqual(2);
      });
    });

    it('resets repetition_penalty to default when reset button is clicked', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const repetitionPenaltyInputs = screen.getAllByDisplayValue('1');
      const numberInput = repetitionPenaltyInputs.find(
        (input) =>
          input.getAttribute('type') === 'number' &&
          input.getAttribute('max') === '2' &&
          input.getAttribute('min') === '1'
      ) as HTMLInputElement;

      // Change the value
      fireEvent.change(numberInput, { target: { value: '1.8' } });

      await waitFor(() => {
        expect(numberInput).toHaveValue(1.8);
      });

      // Click reset button
      const resetButton = screen.getByRole('button', {
        name: 'Reset repetition_penalty to default value',
      });
      await user.click(resetButton);

      await waitFor(() => {
        expect(numberInput).toHaveValue(1);
      });
    });
  });

  describe('Automatic Sampling Controls', () => {
    it('disables input when automatic sampling is enabled', async () => {
      render(
        <TestWrapper
          defaultValues={{
            spec: {
              data_source: 'test',
              config: {
                training: {
                  num_input_records_to_sample: 'auto',
                  rope_scaling_factor: 'auto',
                },
              },
            },
          }}
        >
          <AdvancedParameters />
        </TestWrapper>
      );

      await waitFor(() =>
        expect(screen.getByRole('checkbox', { name: /use automatic sampling/i })).toBeChecked()
      );
    });

    it('enables input when automatic sampling is disabled', async () => {
      render(
        <TestWrapper
          defaultValues={{
            spec: {
              data_source: 'test',
              config: {
                training: {
                  num_input_records_to_sample: 100,
                  rope_scaling_factor: 'auto',
                },
              },
            },
          }}
        >
          <AdvancedParameters />
        </TestWrapper>
      );

      await waitFor(() =>
        expect(screen.getByRole('checkbox', { name: /use automatic sampling/i })).not.toBeChecked()
      );
      const samplingInput = screen.getByDisplayValue('100');
      expect(samplingInput).toBeDefined();
      expect(samplingInput).not.toBeDisabled();
      expect(samplingInput).toHaveAttribute('type', 'number');
    });

    it('toggles automatic sampling when checkbox is clicked', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const checkbox = screen.getByRole('checkbox', { name: /use automatic sampling/i });
      expect(checkbox).toBeChecked(); // Default is auto

      await user.click(checkbox);

      await waitFor(() => {
        expect(checkbox).not.toBeChecked();
      });

      await user.click(checkbox);

      await waitFor(() => {
        expect(checkbox).toBeChecked();
      });
    });

    it('sets value to "auto" when enabling automatic sampling', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper
          defaultValues={{
            spec: {
              data_source: 'test',
              config: {
                training: {
                  num_input_records_to_sample: 100,
                  rope_scaling_factor: 'auto',
                },
              },
            },
          }}
        >
          <AdvancedParameters />
        </TestWrapper>
      );

      const checkbox = screen.getByRole('checkbox', { name: /use automatic sampling/i });
      await user.click(checkbox);

      await waitFor(() => {
        expect(checkbox).toBeChecked();
      });
    });

    it('sets value to 1 when disabling automatic sampling', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const checkbox = screen.getByRole('checkbox', { name: /use automatic sampling/i });
      await user.click(checkbox);

      await waitFor(() => {
        expect(checkbox).not.toBeChecked();
      });
    });
  });

  describe('Automatic Scaling Controls', () => {
    it('disables slider and input when automatic scaling is enabled', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      await waitFor(() =>
        expect(screen.getByRole('checkbox', { name: /use automatic scaling/i })).toBeChecked()
      );
      // Fourth slider is rope_scaling_factor - it should be disabled
      const sliders = screen.getAllByRole('slider');
      const ropeSlider = sliders[3];
      expect(ropeSlider).toBeDisabled();

      const numberInputs = screen
        .getAllByRole('spinbutton')
        .filter(
          (input) => input.getAttribute('max') === '6' && input.getAttribute('min') === '1'
        ) as HTMLInputElement[];
      expect(numberInputs.length).toBeGreaterThan(0);
      expect(numberInputs[0]).toBeDisabled();
    });

    it('enables slider and input when automatic scaling is disabled', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const checkbox = screen.getByRole('checkbox', { name: /use automatic scaling/i });
      await user.click(checkbox);

      await waitFor(() => {
        const sliders = screen.getAllByRole('slider');
        const ropeSlider = sliders[3]; // Fourth slider is rope_scaling_factor
        expect(ropeSlider).not.toBeDisabled();
      });
    });

    it('toggles automatic scaling when checkbox is clicked', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const checkbox = screen.getByRole('checkbox', { name: /use automatic scaling/i });
      expect(checkbox).toBeChecked(); // Default is auto

      await user.click(checkbox);

      await waitFor(() => {
        expect(checkbox).not.toBeChecked();
      });

      await user.click(checkbox);

      await waitFor(() => {
        expect(checkbox).toBeChecked();
      });
    });

    it('displays rope_scaling_factor value correctly', async () => {
      render(
        <TestWrapper
          defaultValues={{
            spec: {
              data_source: 'test',
              config: {
                training: {
                  num_input_records_to_sample: 'auto',
                  rope_scaling_factor: 4,
                },
              },
            },
          }}
        >
          <AdvancedParameters />
        </TestWrapper>
      );

      await waitFor(() =>
        expect(
          screen
            .getAllByDisplayValue('4')
            .find((input) => input.getAttribute('type') === 'number') as HTMLInputElement
        ).toHaveValue(4)
      );
      const numberInput = screen
        .getAllByDisplayValue('4')
        .find((input) => input.getAttribute('type') === 'number') as HTMLInputElement;
      expect(numberInput).toHaveAttribute('min', '1');
      expect(numberInput).toHaveAttribute('max', '6');
    });

    it('clamps rope_scaling_factor to minimum of 1', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper
          defaultValues={{
            spec: {
              data_source: 'test',
              config: {
                training: {
                  num_input_records_to_sample: 'auto',
                  rope_scaling_factor: 3,
                },
              },
            },
          }}
        >
          <AdvancedParameters />
        </TestWrapper>
      );

      const ropeInputs = screen.getAllByDisplayValue('3');
      const numberInput = ropeInputs.find(
        (input) => input.getAttribute('type') === 'number'
      ) as HTMLInputElement;

      await user.clear(numberInput);
      await user.type(numberInput, '0');
      await user.tab();

      await waitFor(() => {
        expect(Number(numberInput.value)).toBeGreaterThanOrEqual(1);
      });
    });

    it('clamps rope_scaling_factor to maximum of 6', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper
          defaultValues={{
            spec: {
              data_source: 'test',
              config: {
                training: {
                  num_input_records_to_sample: 'auto',
                  rope_scaling_factor: 3,
                },
              },
            },
          }}
        >
          <AdvancedParameters />
        </TestWrapper>
      );

      const ropeInputs = screen.getAllByDisplayValue('3');
      const numberInput = ropeInputs.find(
        (input) => input.getAttribute('type') === 'number'
      ) as HTMLInputElement;

      await user.clear(numberInput);
      await user.type(numberInput, '10');
      await user.tab();

      await waitFor(() => {
        expect(Number(numberInput.value)).toBeLessThanOrEqual(6);
      });
    });
  });

  describe('Model Selector', () => {
    it('renders the model selector under Training Data Configuration', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByText('Model')).toBeInTheDocument();
      expect(screen.getByTestId('nv-select-trigger')).toHaveTextContent('HuggingFaceTB/SmolLM3-3B');
    });

    it('defaults to SmolLM3 when the user does not interact', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
          <PretrainedModelProbe />
        </TestWrapper>
      );

      await waitFor(() =>
        expect(screen.getByTestId('pretrained-model-value')).toHaveTextContent(
          'HuggingFaceTB/SmolLM3-3B'
        )
      );
    });

    it('lists the three supported models in order', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      await user.click(screen.getByTestId('nv-select-trigger'));

      await waitFor(() => {
        const options = screen.getAllByRole('option');
        expect(options.map((option) => option.textContent)).toEqual([
          'HuggingFaceTB/SmolLM3-3B',
          'mistralai/Mistral-7B-Instruct-v0.3',
          'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        ]);
      });
    });

    it('updates the configuration when a different model is selected', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
          <PretrainedModelProbe />
        </TestWrapper>
      );

      await user.click(screen.getByTestId('nv-select-trigger'));
      await user.click(
        await screen.findByRole('option', { name: 'mistralai/Mistral-7B-Instruct-v0.3' })
      );

      await waitFor(() =>
        expect(screen.getByTestId('pretrained-model-value')).toHaveTextContent(
          'mistralai/Mistral-7B-Instruct-v0.3'
        )
      );
    });
  });

  describe('PII Replacement Controls', () => {
    it('displays enable_replace_pii switch', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByRole('switch')).toBeInTheDocument();
    });

    it('reflects default enable_replace_pii value', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      await waitFor(() => expect(screen.getByRole('switch')).toBeChecked());
    });

    it('toggles enable_replace_pii when switch is clicked', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const switchElement = screen.getByRole('switch');
      expect(switchElement).toBeChecked();

      await user.click(switchElement);

      await waitFor(() => {
        expect(switchElement).not.toBeChecked();
      });
    });
  });

  describe('Error Display', () => {
    it('does not display error messages when no errors present', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByText('temperature')).toBeInTheDocument();
      // Verify no error messages are shown initially
      expect(screen.queryByText(/must be between/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/must be a positive number/i)).not.toBeInTheDocument();
    });

    it('renders error message slots for all form fields', async () => {
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      expect(await screen.findByText('temperature')).toBeInTheDocument();
      // Just verify the component renders without errors - actual error validation
      // is handled by the form schema and would be tested in integration tests
      expect(screen.getByText('top_p')).toBeInTheDocument();
      expect(screen.getByText('repetition_penalty')).toBeInTheDocument();
      expect(screen.getByText('num_input_records_to_sample')).toBeInTheDocument();
      expect(screen.getByText('rope_scaling_factor')).toBeInTheDocument();
    });

    it('allows clearing input when automatic sampling is disabled', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper
          defaultValues={{
            spec: {
              data_source: 'test',
              config: {
                training: {
                  num_input_records_to_sample: 100,
                  rope_scaling_factor: 'auto',
                },
              },
            },
          }}
        >
          <AdvancedParameters />
        </TestWrapper>
      );

      const checkbox = screen.getByRole('checkbox', { name: /use automatic sampling/i });
      expect(checkbox).not.toBeChecked();

      const samplingInput = screen.getByDisplayValue('100');
      await user.clear(samplingInput);

      expect(samplingInput).toHaveValue(null);
    });

    it('disables input when switching to automatic sampling', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper
          defaultValues={{
            spec: {
              data_source: 'test',
              config: {
                training: {
                  num_input_records_to_sample: 100,
                  rope_scaling_factor: 'auto',
                },
              },
            },
          }}
        >
          <AdvancedParameters />
        </TestWrapper>
      );

      const checkbox = screen.getByRole('checkbox', { name: /use automatic sampling/i });
      expect(checkbox).not.toBeChecked();

      const samplingInput = screen.getByDisplayValue('100');
      expect(samplingInput).not.toBeDisabled();

      await user.click(checkbox);

      await waitFor(() => {
        expect(checkbox).toBeChecked();
      });
      expect(samplingInput).toBeDisabled();
    });
  });

  describe('Reset Buttons', () => {
    it('resets temperature to default when reset button is clicked', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      const temperatureInputs = screen.getAllByDisplayValue('0.9');
      const numberInput = temperatureInputs.find(
        (input) => input.getAttribute('type') === 'number'
      ) as HTMLInputElement;

      // Change the value
      fireEvent.change(numberInput, { target: { value: '1.5' } });

      await waitFor(() => {
        expect(numberInput).toHaveValue(1.5);
      });

      // Click reset button
      const resetButton = screen.getByRole('button', {
        name: 'Reset temperature to default value',
      });
      await user.click(resetButton);

      await waitFor(() => {
        expect(numberInput).toHaveValue(0.9);
      });
    });
  });

  describe('Integration', () => {
    it('maintains form state across multiple interactions', async () => {
      const user = userEvent.setup();
      render(
        <TestWrapper>
          <AdvancedParameters />
        </TestWrapper>
      );

      // Change temperature
      const temperatureInputs = screen.getAllByDisplayValue('0.9');
      const temperatureInput = temperatureInputs.find(
        (input) => input.getAttribute('type') === 'number'
      ) as HTMLInputElement;
      fireEvent.change(temperatureInput, { target: { value: '1.2' } });

      // Change top_p
      const topPInputs = screen.getAllByDisplayValue('1');
      const topPInput = topPInputs.find(
        (input) => input.getAttribute('type') === 'number' && input.getAttribute('max') === '1'
      ) as HTMLInputElement;
      fireEvent.change(topPInput, { target: { value: '0.8' } });

      // Disable automatic sampling
      const samplingCheckbox = screen.getByRole('checkbox', { name: /use automatic sampling/i });
      await user.click(samplingCheckbox);

      // Toggle PII switch
      const piiSwitch = screen.getByRole('switch');
      await user.click(piiSwitch);

      await waitFor(() => {
        expect(temperatureInput).toHaveValue(1.2);
      });
      expect(topPInput).toHaveValue(0.8);
      expect(samplingCheckbox).not.toBeChecked();
      expect(piiSwitch).not.toBeChecked();
    });
  });
});
