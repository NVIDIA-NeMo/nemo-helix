// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { mockSignoutRedirect } from '@studio/tests/mocks/react-oidc-context';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router';

// Get access to the centralized auth mocks
const { mockUseAuthProfile } = vi.hoisted(() => ({
  mockUseAuthProfile: vi.fn(),
}));

// Mock just the useAuthProfile hook since react-oidc-context is handled centrally
vi.mock('@studio/providers/auth/useAuthProfile', () => ({
  useAuthProfile: mockUseAuthProfile,
}));

vi.mock('@studio/constants/environment', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@studio/constants/environment')>();
  return {
    ...actual,
    TELEMETRY_ENABLED: true,
  };
});

// Mock the ReportTraceModal component
vi.mock('@studio/components/ReportTraceModal', () => ({
  ReportTraceModal: ({ open, onClose }: { open: boolean; onClose: () => void }) => {
    if (!open) return null;
    return (
      <div data-testid="report-trace-modal">
        <button onClick={onClose}>Close Modal</button>
      </div>
    );
  },
}));

describe('UserPopover', () => {
  let user: ReturnType<typeof userEvent.setup>;
  let UserPopover: React.ComponentType;

  const mockProfile = {
    name: 'John Doe',
    email: 'john.doe@example.com',
    workspace: 'john-doe',
  };

  beforeAll(async () => {
    const { UserPopover: UserPopoverComponent } = await import('@studio/components/UserPopover');
    UserPopover = UserPopoverComponent;
  }, 180000);

  beforeEach(() => {
    vi.clearAllMocks();

    // Initialize user for each test
    user = userEvent.setup();

    // Default mock setup - user is authenticated
    mockUseAuthProfile.mockReturnValue(mockProfile);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const renderWithRouter = (ui: React.ReactElement) => {
    const router = createMemoryRouter([{ path: '/', element: ui }], { initialEntries: ['/'] });
    return render(<RouterProvider router={router} />);
  };

  it('should render the user popover trigger with user profile', async () => {
    renderWithRouter(<UserPopover />);

    // Check that the avatar trigger is rendered
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    expect(trigger).toBeInTheDocument();
    expect(trigger).toHaveTextContent('J'); // First letter of John
    expect(trigger).toHaveTextContent('John Doe');
  });

  it('should render fallback avatar when no user profile', async () => {
    mockUseAuthProfile.mockReturnValue(undefined);
    renderWithRouter(<UserPopover />);

    // Check that the avatar trigger is rendered with fallback
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    expect(trigger).toBeInTheDocument();
    expect(trigger).toHaveTextContent('N');
  });

  it('should display email heading when opened', async () => {
    renderWithRouter(<UserPopover />);

    // Click the avatar to open the popover
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    await user.click(trigger);

    // Check that the email is displayed as heading (use testid to be specific)
    const heading = screen.getByTestId('nv-menu-heading');
    expect(heading).toBeInTheDocument();
    expect(heading).toHaveTextContent('john.doe@example.com');
  });

  it('should not display heading when no profile', async () => {
    mockUseAuthProfile.mockReturnValue(undefined);
    renderWithRouter(<UserPopover />);

    // Click the avatar to open the popover
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    await user.click(trigger);

    // Check that no heading is displayed
    expect(screen.queryByTestId('nv-menu-heading')).not.toBeInTheDocument();
  });

  it('should display all menu items when opened with authenticated user', async () => {
    renderWithRouter(<UserPopover />);

    // Click the avatar to open the popover
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    await user.click(trigger);

    // Check that all menu items are present
    expect(screen.getByText('Report a Trace')).toBeInTheDocument();
    expect(screen.getByText('Sign Out')).toBeInTheDocument();
  });

  it('should not display sign out when no user profile', async () => {
    mockUseAuthProfile.mockReturnValue(undefined);
    renderWithRouter(<UserPopover />);

    // Click the avatar to open the popover
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    await user.click(trigger);

    // Check that sign out is not present
    expect(screen.queryByText('Sign Out')).not.toBeInTheDocument();
  });

  it('should hide telemetry menu item when telemetry is disabled', async () => {
    vi.resetModules();
    vi.doMock('@studio/constants/environment', async (importOriginal) => {
      const actual = await importOriginal<typeof import('@studio/constants/environment')>();
      return {
        ...actual,
        TELEMETRY_ENABLED: false,
      };
    });
    mockUseAuthProfile.mockReturnValue(mockProfile);
    const { UserPopover: UserPopoverWithTelemetryDisabled } =
      await import('@studio/components/UserPopover');
    renderWithRouter(<UserPopoverWithTelemetryDisabled />);

    // Click the avatar to open the popover
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    await user.click(trigger);

    // Check that the telemetry menu item is not present
    expect(screen.queryByText('Report a Trace')).not.toBeInTheDocument();
  });

  it('should render nothing when no profile and telemetry disabled', async () => {
    vi.resetModules();
    vi.doMock('@studio/constants/environment', async (importOriginal) => {
      const actual = await importOriginal<typeof import('@studio/constants/environment')>();
      return {
        ...actual,
        TELEMETRY_ENABLED: false,
      };
    });
    mockUseAuthProfile.mockReturnValue(undefined);
    const { UserPopover: UserPopoverEmpty } = await import('@studio/components/UserPopover');
    const { container } = renderWithRouter(<UserPopoverEmpty />);

    expect(screen.queryByTestId('nv-dropdown-trigger')).not.toBeInTheDocument();
    expect(screen.queryByText('N')).not.toBeInTheDocument();
    expect(container).toBeEmptyDOMElement();
  });

  it('should open trace modal when "Report a Trace" is clicked', async () => {
    renderWithRouter(<UserPopover />);

    // Click the avatar to open the popover
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    await user.click(trigger);

    // Click "Report a Trace"
    const reportTraceButton = screen.getByText('Report a Trace');
    await user.click(reportTraceButton);

    // Check that the modal is opened
    expect(screen.getByTestId('report-trace-modal')).toBeInTheDocument();
  });

  it('should close trace modal when close button is clicked', async () => {
    renderWithRouter(<UserPopover />);

    // Click the avatar to open the popover
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    await user.click(trigger);

    // Click "Report a Trace" to open modal
    const reportTraceButton = screen.getByText('Report a Trace');
    await user.click(reportTraceButton);

    // Verify modal is open
    expect(screen.getByTestId('report-trace-modal')).toBeInTheDocument();

    // Click close button
    const closeButton = screen.getByText('Close Modal');
    await user.click(closeButton);

    // Verify modal is closed
    await waitFor(() => {
      expect(screen.queryByTestId('report-trace-modal')).not.toBeInTheDocument();
    });
  });

  it('should close popover when clicking outside', async () => {
    renderWithRouter(
      <div>
        <div data-testid="outside-element">Outside</div>
        <UserPopover />
      </div>
    );

    // Click the avatar to open the popover
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    await user.click(trigger);

    // Verify popover is open by checking for dropdown content
    expect(screen.getByTestId('nv-dropdown-content')).toBeInTheDocument();

    // Click outside the popover
    const outsideElement = screen.getByTestId('outside-element');
    await user.click(outsideElement);

    // Verify popover is closed
    await waitFor(() => {
      expect(screen.queryByTestId('nv-dropdown-content')).not.toHaveAttribute('popover-open');
    });
  });

  it('should call signoutRedirect when Sign Out is clicked', async () => {
    renderWithRouter(<UserPopover />);

    // Click the avatar to open the popover
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    await user.click(trigger);

    // Click Sign Out
    const signOutButton = screen.getByText('Sign Out');
    await user.click(signOutButton);

    // Verify signoutRedirect was called
    expect(mockSignoutRedirect).toHaveBeenCalled();
  });

  it('should handle profile with different name', async () => {
    const customProfile = {
      name: 'Alice Smith',
      email: 'alice.smith@example.com',
      workspace: 'alice-smith',
    };
    mockUseAuthProfile.mockReturnValue(customProfile);

    renderWithRouter(<UserPopover />);

    // Check that the avatar shows first letter of Alice
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    expect(trigger).toHaveTextContent('A');
    expect(trigger).toHaveTextContent('Alice Smith');
  });

  it('should handle profile with empty name', async () => {
    const customProfile = {
      name: '',
      email: 'test@example.com',
      workspace: 'test',
    };
    mockUseAuthProfile.mockReturnValue(customProfile);

    renderWithRouter(<UserPopover />);

    // Check that the avatar shows fallback 'N'
    const trigger = screen.getByTestId('nv-dropdown-trigger');
    expect(trigger).toHaveTextContent('N');
  });
});
