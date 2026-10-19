// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import '@studio/index.css';

import { logger } from '@nemo/common/src/utils/logger';
import { App } from '@studio/App';
import { UI_THEME } from '@studio/util/localStorage';
import ReactDOM from 'react-dom/client';

// OpenTelemetry patches fetch/XHR globally, so this must settle before React
// renders and issues the first requests.
const telemetryReady = import('@studio/telemetry/telemetry').catch((error: unknown) => {
  logger.error('Telemetry failed to initialize', error);
});

const storedTheme = window.localStorage.getItem(UI_THEME);
const theme = storedTheme ? JSON.parse(storedTheme) : 'dark';

if (theme === 'dark') {
  document.documentElement.classList.add('nv-dark');
} else {
  document.documentElement.classList.remove('nv-dark');
}

/** Resolves when the theme stylesheet has loaded to avoid content layout shift. */
function waitForThemeStylesheet(): Promise<void> {
  const link = document.getElementById('theme-stylesheet') as HTMLLinkElement | null;
  if (!link) return Promise.resolve();
  // link.sheet is set when the stylesheet is loaded (avoids CORS issues with cssRules)
  if (link.sheet) return Promise.resolve();
  return new Promise((resolve) => {
    link.addEventListener('load', () => resolve(), { once: true });
    link.addEventListener('error', () => resolve(), { once: true });
  });
}

const rootElement = document.getElementById('app')!;
if (!rootElement.innerHTML) {
  Promise.all([waitForThemeStylesheet(), telemetryReady]).then(() => {
    rootElement.removeAttribute('aria-busy');
    const root = ReactDOM.createRoot(rootElement);
    root.render(<App />);
  });
}
