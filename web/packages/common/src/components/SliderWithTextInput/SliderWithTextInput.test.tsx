// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SliderWithTextInput } from '@nemo/common/src/components/SliderWithTextInput/index';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const createMockField = (value: number | undefined = 0, onChange = vi.fn()) => ({
  value,
  onChange,
  onBlur: vi.fn(),
  name: 'test-field',
  ref: vi.fn(),
});

describe('SliderWithTextInput', () => {
  describe('Basic Rendering', () => {
    it('should render slider and text input', () => {
      const field = createMockField(10);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={0}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      expect(screen.getByRole('slider')).toBeInTheDocument();
      expect(screen.getByRole('spinbutton')).toBeInTheDocument();
    });
  });

  describe('Value Display', () => {
    it('should display the correct value in both slider and input', () => {
      const field = createMockField(25);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={0}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      const slider = screen.getByRole('slider');
      const textInput = screen.getByRole('spinbutton');

      expect(slider).toHaveValue('25');
      expect(textInput).toHaveValue(25);
    });

    it('should display 0 when value is null or undefined', () => {
      const field = createMockField(undefined);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={0}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      const slider = screen.getByRole('slider');
      const textInput = screen.getByRole('spinbutton');

      expect(slider).toHaveValue('0');
      expect(textInput).toHaveValue(0);
    });
  });

  describe('Slider Configuration', () => {
    it('should set correct min/max attributes', () => {
      const field = createMockField(50);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={10}
          max={90}
          step={5}
          disabled={false}
        />
      );

      const slider = screen.getByRole('slider');
      const textInput = screen.getByRole('spinbutton');

      expect(slider).toHaveAttribute('min', '10');
      expect(slider).toHaveAttribute('max', '90');
      expect(textInput).toHaveAttribute('min', '10');
      expect(textInput).toHaveAttribute('max', '90');
      expect(textInput).toHaveAttribute('step', '5');
    });
  });

  describe('Form Field Properties', () => {
    it('should render with label and help text', () => {
      const field = createMockField(50);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={0}
          max={100}
          step={1}
          disabled={false}
          formFieldProps={{
            slotLabel: 'Temperature',
            slotHelp: 'Controls randomness in the output',
          }}
        />
      );

      expect(screen.getByText('Temperature')).toBeInTheDocument();
      expect(screen.getByText('Controls randomness in the output')).toBeInTheDocument();
    });

    it('should render with error message', () => {
      const field = createMockField(150);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={50}
          min={0}
          max={100}
          step={1}
          disabled={false}
          formFieldProps={{
            slotLabel: 'Temperature',
            slotError: 'Value must be between 0 and 100',
          }}
        />
      );
    });

    it('should render as required field', () => {
      const field = createMockField(50);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={0}
          max={100}
          step={1}
          disabled={false}
          formFieldProps={{
            slotLabel: 'Temperature',
            required: true,
          }}
        />
      );

      // Check for required indicator (usually an asterisk)
      expect(screen.getByText('Temperature')).toBeInTheDocument();
    });
  });

  describe('Disabled State', () => {
    it('should disable both slider and input when disabled=true', () => {
      const field = createMockField(50);
      render(
        <SliderWithTextInput field={field} defaultValue={25} min={0} max={100} step={1} disabled />
      );

      const textInput = screen.getByRole('spinbutton');

      expect(textInput).toBeDisabled();
    });

    it('should enable both slider and input when disabled=false', () => {
      const field = createMockField(50);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      const textInput = screen.getByRole('spinbutton');

      expect(textInput).not.toBeDisabled();
    });
  });

  describe('User Interactions', () => {
    it('should call field.onChange when slider value changes', () => {
      const mockOnChange = vi.fn();
      const field = createMockField(50, mockOnChange);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      const slider = screen.getByRole('spinbutton');
      fireEvent.change(slider, { target: { value: '75' } });

      expect(mockOnChange).toHaveBeenCalledWith(75);
    });

    it('should call field.onChange when text input value changes', async () => {
      const mockOnChange = vi.fn();
      const field = createMockField(50, mockOnChange);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      const textInput = screen.getByRole('spinbutton');
      await waitFor(() => expect(textInput).toBeInTheDocument());

      fireEvent.change(textInput, { target: { value: '80' } });

      await waitFor(() => expect(mockOnChange).toHaveBeenCalledWith(80));
    });

    it('should clamp values to min/max bounds when slider changes', () => {
      const mockOnChange = vi.fn();
      const field = createMockField(50, mockOnChange);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={10}
          max={90}
          step={1}
          disabled={false}
        />
      );

      const slider = screen.getByRole('slider');

      // Test value above max
      fireEvent.input(slider, { target: { value: '150' } });
      expect(mockOnChange).toHaveBeenCalledWith(90);

      // Test value below min
      fireEvent.input(slider, { target: { value: '5' } });
      expect(mockOnChange).toHaveBeenCalledWith(10);
    });

    it('should not clamp text input values while the user is still typing', () => {
      const mockOnChange = vi.fn();
      const field = createMockField(50, mockOnChange);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={10}
          max={90}
          step={1}
          disabled={false}
        />
      );

      const textInput = screen.getByRole('spinbutton');

      // '5' is below min but is a valid prefix of '55', so it must survive.
      fireEvent.change(textInput, { target: { value: '5' } });
      expect(mockOnChange).not.toHaveBeenCalled();
      expect(textInput).toHaveValue(5);

      fireEvent.change(textInput, { target: { value: '55' } });
      expect(mockOnChange).toHaveBeenCalledWith(55);
    });

    it('should clamp out-of-range text input values on blur', () => {
      const mockOnChange = vi.fn();
      const field = createMockField(50, mockOnChange);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={10}
          max={90}
          step={1}
          disabled={false}
        />
      );

      const textInput = screen.getByRole('spinbutton');

      // Test value above max
      fireEvent.change(textInput, { target: { value: '150' } });
      fireEvent.blur(textInput);
      expect(mockOnChange).toHaveBeenCalledWith(90);

      // Test value below min
      fireEvent.change(textInput, { target: { value: '5' } });
      fireEvent.blur(textInput);
      expect(mockOnChange).toHaveBeenCalledWith(10);
    });

    it('should call the TextInput onBlur after clamping', () => {
      const calls: string[] = [];
      const field = createMockField(
        50,
        vi.fn(() => calls.push('onChange'))
      );
      render(
        <SliderWithTextInput
          field={field}
          min={10}
          max={90}
          attributes={{ TextInput: { onBlur: () => calls.push('onBlur') } }}
        />
      );

      const textInput = screen.getByRole('spinbutton');
      fireEvent.change(textInput, { target: { value: '150' } });
      fireEvent.blur(textInput);
      expect(calls).toEqual(['onChange', 'onBlur']);
    });

    it('should allow typing a value whose prefix is below min', () => {
      const mockOnChange = vi.fn();
      const field = createMockField(1, mockOnChange);
      const { rerender } = render(
        <SliderWithTextInput
          field={field}
          defaultValue={1}
          min={0.1}
          max={2}
          step={0.1}
          disabled={false}
        />
      );

      const textInput = screen.getByRole('spinbutton');

      // Typing '0' must not be rewritten to the 0.1 minimum.
      fireEvent.change(textInput, { target: { value: '0' } });
      expect(mockOnChange).not.toHaveBeenCalled();
      expect(textInput).toHaveValue(0);

      fireEvent.change(textInput, { target: { value: '0.7' } });
      expect(mockOnChange).toHaveBeenCalledWith(0.7);

      rerender(
        <SliderWithTextInput
          field={createMockField(0.7, mockOnChange)}
          defaultValue={1}
          min={0.1}
          max={2}
          step={0.1}
          disabled={false}
        />
      );
      fireEvent.blur(textInput);
      expect(textInput).toHaveValue(0.7);
    });
  });

  describe('Reset Functionality', () => {
    it('should reset value to default after user has changed it', async () => {
      const mockOnChange = vi.fn();
      const field = createMockField(50, mockOnChange);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={10}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      // Change the value via text input
      const textInput = screen.getByRole('spinbutton');
      fireEvent.change(textInput, { target: { value: '80' } });

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(80);
      });

      // Reset to default
      const resetButton = screen.getByRole('button', { name: 'Reset test-field to default value' });
      fireEvent.click(resetButton);

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(10);
      });
    });

    it('should disable reset button when component is disabled', () => {
      const field = createMockField(75);
      render(
        <SliderWithTextInput field={field} defaultValue={25} min={0} max={100} step={1} disabled />
      );

      const resetButton = screen.getByRole('button', { name: 'Reset test-field to default value' });
      expect(resetButton).toBeDisabled();
    });

    it('should not submit parent form when reset button is clicked', async () => {
      const mockOnChange = vi.fn();
      const mockOnSubmit = vi.fn((e) => e.preventDefault());
      const field = createMockField(75, mockOnChange);

      render(
        <form onSubmit={mockOnSubmit}>
          <SliderWithTextInput
            field={field}
            defaultValue={25}
            min={0}
            max={100}
            step={1}
            disabled={false}
          />
        </form>
      );

      const resetButton = screen.getByRole('button', { name: 'Reset test-field to default value' });
      fireEvent.click(resetButton);

      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(25);
      });

      // Verify that form submit was NOT called
      expect(mockOnSubmit).not.toHaveBeenCalled();
    });

    it('should hide reset button when showReset is false', () => {
      const field = createMockField(75);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={0}
          max={100}
          step={1}
          disabled={false}
          showReset={false}
        />
      );

      const resetButton = screen.queryByRole('button', {
        name: 'Reset test-field to default value',
      });
      expect(resetButton).not.toBeInTheDocument();
    });

    it('should have unique aria-label for each field instance', () => {
      const field1 = { ...createMockField(75), name: 'temperature' };
      const field2 = { ...createMockField(50), name: 'batch_size' };

      const { rerender } = render(
        <SliderWithTextInput
          field={field1}
          defaultValue={25}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      const resetButton1 = screen.getByRole('button', {
        name: 'Reset temperature to default value',
      });
      expect(resetButton1).toBeInTheDocument();

      rerender(
        <SliderWithTextInput
          field={field2}
          defaultValue={25}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      const resetButton2 = screen.getByRole('button', {
        name: 'Reset batch_size to default value',
      });
      expect(resetButton2).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should set correct aria attributes', () => {
      const field = createMockField(50);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={0}
          max={100}
          step={1}
          disabled={false}
          formFieldProps={{
            slotLabel: 'Volume',
          }}
        />
      );

      const slider = screen.getByRole('slider');
      const textInput = screen.getByRole('spinbutton');

      expect(slider).toHaveAttribute('aria-label', 'Controlled slider');
      expect(textInput).toHaveAttribute('aria-label', 'slider_text_input');
    });

    it('should have data-testid for testing', () => {
      const field = createMockField(50);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      expect(screen.getByRole('slider')).toBeInTheDocument();
      expect(screen.getByRole('spinbutton')).toBeInTheDocument();
    });
  });

  describe('Training Parameter Use Cases', () => {
    it('should support batch_size configuration', () => {
      const field = createMockField(8);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={8}
          min={8}
          max={128}
          step={8}
          disabled={false}
          formFieldProps={{
            slotLabel: 'Batch Size',
            slotHelp: 'Batch size is a hyperparameter used for training a customization.',
          }}
        />
      );

      expect(screen.getByText('Batch Size')).toBeInTheDocument();
      expect(screen.getByText(/Batch size is a hyperparameter/)).toBeInTheDocument();

      const slider = screen.getByRole('slider');
      const textInput = screen.getByRole('spinbutton');

      expect(slider).toHaveAttribute('min', '8');
      expect(slider).toHaveAttribute('max', '128');
      expect(textInput).toHaveAttribute('step', '8');
    });

    it('should support learning_rate configuration with scientific notation', () => {
      const field = createMockField(0.0001);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={0.0001}
          min={1e-15}
          max={1e-3}
          step={1e-6}
          disabled={false}
          formFieldProps={{
            slotLabel: 'Learning Rate',
            slotHelp: 'How much to adjust the model parameters in response to the loss gradient.',
          }}
        />
      );

      expect(screen.getByText('Learning Rate')).toBeInTheDocument();

      const textInput = screen.getByRole('spinbutton');
      expect(textInput).toHaveValue(0.0001);
    });
  });

  describe('Model Parameter Use Cases', () => {
    it('should support temperature configuration', () => {
      const field = createMockField(0.7);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={0.5}
          min={0.0}
          max={2.0}
          step={0.1}
          disabled={false}
          formFieldProps={{
            slotLabel: 'Temperature',
            slotHelp: 'Controls randomness in the output.',
          }}
        />
      );

      expect(screen.getByText('Temperature')).toBeInTheDocument();
      expect(screen.getByText('Controls randomness in the output.')).toBeInTheDocument();

      const textInput = screen.getByRole('spinbutton');
      expect(textInput).toHaveValue(0.7);
    });
  });

  describe('Evaluation Parameter Use Cases', () => {
    it('should support max_tokens configuration', () => {
      const field = createMockField(100);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={50}
          min={1}
          max={1000}
          step={1}
          disabled={false}
          formFieldProps={{
            slotHelp: 'The maximum number of tokens to generate.',
          }}
        />
      );

      expect(screen.getByText('The maximum number of tokens to generate.')).toBeInTheDocument();

      const textInput = screen.getByRole('spinbutton');
      expect(textInput).toHaveValue(100);
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero values', () => {
      const field = createMockField(0);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={0}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      const slider = screen.getByRole('slider');
      const textInput = screen.getByRole('spinbutton');

      expect(slider).toHaveValue('0');
      expect(textInput).toHaveValue(0);
    });

    it('should handle negative values', () => {
      const field = createMockField(-5);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={0}
          min={-10}
          max={10}
          step={1}
          disabled={false}
        />
      );

      const slider = screen.getByRole('slider');
      const textInput = screen.getByRole('spinbutton');

      expect(slider).toHaveValue('-5');
      expect(textInput).toHaveValue(-5);
    });

    it('should handle very small decimal values', () => {
      const field = createMockField(0.00001);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={0}
          min={0}
          max={0.001}
          step={0.00001}
          disabled={false}
        />
      );

      const textInput = screen.getByRole('spinbutton');
      expect(textInput).toHaveValue(0.00001);
    });

    it('should handle NaN values in text input gracefully', () => {
      const mockOnChange = vi.fn();
      const field = createMockField(50, mockOnChange);
      render(
        <SliderWithTextInput
          field={field}
          defaultValue={25}
          min={0}
          max={100}
          step={1}
          disabled={false}
        />
      );

      const textInput = screen.getByRole('spinbutton');
      fireEvent.change(textInput, { target: { value: 'invalid' } });

      // Should call onChange with NaN, which will be clamped
      expect(mockOnChange).toHaveBeenCalled();
    });
  });
});
