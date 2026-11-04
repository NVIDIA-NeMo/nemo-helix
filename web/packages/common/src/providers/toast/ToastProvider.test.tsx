// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ToastProvider } from '@nemo/common/src/providers/toast/ToastProvider';
import { useToast } from '@nemo/common/src/providers/toast/useToast';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FC } from 'react';

interface TestProps {
  durationMs?: number;
}
// Mock component to test useToast hook
const TestComponent: FC<TestProps> = ({ durationMs }) => {
  const toast = useToast();
  const options = durationMs !== undefined ? { durationMs } : undefined;
  return (
    <div>
      <button onClick={() => toast.success('Success', options)}>Show Success</button>
      <button onClick={() => toast.error('Error', options)}>Show Error</button>
      <button onClick={() => toast.info('Info', options)}>Show Info</button>
      <button onClick={() => toast.warning('Warning', options)}>Show Warning</button>
      <button onClick={() => toast.neutral('Neutral', options)}>Show Neutral</button>
    </div>
  );
};

describe('ToastProvider', () => {
  const user = userEvent.setup();

  it('renders children', () => {
    render(
      <ToastProvider>
        <div data-testid="child">Child Component</div>
      </ToastProvider>
    );
    expect(screen.getByTestId('child')).toBeInTheDocument();
  });

  it.each([
    ['success', 'Success'],
    ['error', 'Error'],
    ['info', 'Info'],
    ['warning', 'Warning'],
  ])('shows %s toast', async (_label, findText) => {
    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>
    );
    await user.click(screen.getByText(`Show ${findText}`));
    expect(await screen.findByText(findText)).toBeInTheDocument();
  });

  it('shows multiple toasts in order', async () => {
    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>
    );
    await user.click(screen.getByText(`Show Success`));
    await user.click(screen.getByText(`Show Error`));

    const toasts = await screen.findAllByRole('alert');
    expect(toasts).toHaveLength(2);
    expect(toasts[0]).toHaveTextContent('Success');
    expect(toasts[1]).toHaveTextContent('Error');
  });

  it('removes toast when close button is clicked', async () => {
    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>
    );

    await user.click(screen.getByText('Show Info'));

    const toast = await screen.findByText('Info');
    expect(toast).toBeInTheDocument();

    const closeButton = screen.getByRole('button', { name: 'Close' });
    await user.click(closeButton);

    await waitFor(() => {
      expect(screen.queryByText('Info')).not.toBeInTheDocument();
    });
  });

  it('removes toast automatically when called with duration', async () => {
    const durationMs = 3000;
    render(
      <ToastProvider>
        <TestComponent durationMs={durationMs} />
      </ToastProvider>
    );

    await user.click(screen.getByText('Show Info'));

    const toast = await screen.findByText('Info');
    expect(toast).toBeInTheDocument();
    setTimeout(() => {
      expect(screen.queryByText('Info')).not.toBeInTheDocument();
    }, durationMs);
  });

  it.each([
    ['success', 'Success'],
    ['error', 'Error'],
    ['info', 'Info'],
    ['warning', 'Warning'],
    ['neutral', 'Neutral'],
  ])('auto-dismisses %s toasts by default', async (_label, buttonText) => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const fakeUser = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });

    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>
    );

    await fakeUser.click(screen.getByText(`Show ${buttonText}`));
    await act(() => vi.advanceTimersByTimeAsync(50));
    expect(screen.getByRole('alert')).toBeInTheDocument();

    await act(() => vi.runAllTimersAsync());

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();

    vi.useRealTimers();
  });

  it('handles rapid toast additions and removals correctly', async () => {
    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>
    );

    // Add multiple toasts in quick succession
    await user.click(screen.getByText('Show Success'));
    await user.click(screen.getByText('Show Error'));
    await user.click(screen.getByText('Show Info'));

    // Verify all toasts are visible
    const initialToasts = await screen.findAllByRole('alert');
    expect(initialToasts).toHaveLength(3);
    expect(initialToasts[0]).toHaveTextContent('Success');
    expect(initialToasts[1]).toHaveTextContent('Error');
    expect(initialToasts[2]).toHaveTextContent('Info');

    // Close the middle toast
    const closeButtons = screen.getAllByRole('button', { name: 'Close' });
    await user.click(closeButtons[1]);

    // Verify the middle toast is removed while others remain
    await waitFor(() => {
      expect(screen.getAllByRole('alert')).toHaveLength(2);
    });
    const remainingToasts = screen.getAllByRole('alert');
    expect(remainingToasts[0]).toHaveTextContent('Success');
    expect(remainingToasts[1]).toHaveTextContent('Info');

    // Add another toast while others are still visible
    await user.click(screen.getByText('Show Warning'));

    // Verify the new toast is added to the end
    const finalToasts = await screen.findAllByRole('alert');
    expect(finalToasts).toHaveLength(3);
    expect(finalToasts[0]).toHaveTextContent('Success');
    expect(finalToasts[1]).toHaveTextContent('Info');
    expect(finalToasts[2]).toHaveTextContent('Warning');

    // Close all remaining toasts
    const remainingCloseButtons = screen.getAllByRole('button', { name: 'Close' });
    for (const button of remainingCloseButtons) {
      await user.click(button);
    }

    // Verify all toasts are removed
    await waitFor(() => {
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });
  });
});
