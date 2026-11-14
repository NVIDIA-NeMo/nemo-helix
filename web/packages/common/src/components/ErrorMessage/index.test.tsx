// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ErrorMessage } from '@nemo/common/src/components/ErrorMessage';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useNavigate } from 'react-router';
import type { Mock } from 'vitest';

// Mock react-router
vi.mock('react-router', async (importOriginal) => {
  const original = await importOriginal();
  return {
    // @ts-expect-error expect issue here with spread
    ...original,
    useNavigate: vi.fn(),
  };
});

describe('ErrorMessage', () => {
  const mockNavigate = vi.fn();

  beforeEach(() => {
    (useNavigate as unknown as Mock).mockReturnValue(mockNavigate);
  });

  it('Navigates to previous page when clicking Go Back button', async () => {
    const user = userEvent.setup();

    render(<ErrorMessage />);

    const backButton = screen.getByRole('button', { name: /Go back/i });
    await user.click(backButton);

    // Expect that navigate(-1) was called to go back to previous page
    expect(mockNavigate).toHaveBeenCalledWith(-1);
  });

  it('Refreshes page when clicking Refresh Page button', async () => {
    const user = userEvent.setup();
    const mockReload = vi.fn();
    Object.defineProperty(window, 'location', {
      value: { reload: mockReload },
      writable: true,
    });

    render(<ErrorMessage />);

    const refreshButton = screen.getByRole('button', { name: /Refresh Page/i });
    await user.click(refreshButton);

    // Expect that window.location.reload was called
    expect(mockReload).toHaveBeenCalled();
  });

  it('Renders with default error message and header', () => {
    render(<ErrorMessage />);

    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(screen.getByText('An unexpected error occurred')).toBeInTheDocument();
  });

  it('Renders with custom error message and header', () => {
    const customHeader = 'Custom Error';
    const customMessage = 'Something went wrong';

    render(<ErrorMessage header={customHeader} message={customMessage} />);

    expect(screen.getByText(customHeader)).toBeInTheDocument();
    expect(screen.getByText(customMessage)).toBeInTheDocument();
  });
});
