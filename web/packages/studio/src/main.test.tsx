// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable testing-library/no-node-access */
import { screen, act } from '@testing-library/react';

vi.mock('@studio/telemetry/telemetry', () => ({}));

vi.mock('@studio/routes', () => ({
  routes: [
    {
      path: '/',
      element: <div>Projects</div>,
    },
  ],
}));

// This test verifies the main entry point of the app, and that the app is rendered in the DOM.
describe('main', () => {
  const renderApp = async () => {
    await act(async () => await import('@studio/main'));
    expect(document.getElementById('app')).toBeInTheDocument();
  };

  beforeEach(() => {
    vi.resetModules();

    Object.defineProperty(window, 'location', {
      value: {
        origin: 'http://localhost:3000',
        href: 'http://localhost:3000/',
        pathname: '/',
        search: '',
        hash: '',
      },
      writable: true,
    });

    // Create the target element before each test
    const appRoot = document.createElement('div');
    appRoot.id = 'app';
    document.body.appendChild(appRoot);
  });

  afterEach(() => {
    // Clean up after each test
    const appRoot = document.getElementById('app');
    if (appRoot) {
      appRoot.remove();
    }
    vi.unstubAllEnvs();
  });

  it('should render', async () => {
    await renderApp();
    expect(await screen.findByText('Projects')).toBeInTheDocument();
  });

  it.each(['true', 'false'])('should render if TELEMETRY_ENABLED is %s', async (value) => {
    vi.stubEnv('VITE_TELEMETRY_ENABLED', value);
    await renderApp();
    expect(await screen.findByText('Projects')).toBeInTheDocument();
  });
});
