// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* eslint-disable no-console -- CLI script: console is the correct output mechanism */
import { chromium, type Page } from '@playwright/test';
import { TOUR_SEEN_KEY, UI_THEME } from '@studio/util/localStorage';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { parseArgs } from 'node:util';

const USAGE = `
Capture a headless screenshot of a Studio page or element.

Usage: pnpm screenshot --route <path> [options]

  --route <path>          Route relative to the base URL (e.g. workspaces/default/agents)
  --testid <id>           Screenshot only the element with this data-testid (default: viewport)
  --click <id>            Click a data-testid before capturing; repeatable, runs in order
  --out <file>            Output PNG (default: screenshots/out/<testid|page>-<theme>.png)
  --theme <light|dark>    UI theme (default: light)
  --base-url <url>        Studio base URL (default: $STUDIO_URL or https://localhost:5173/)
  --width <px>            Viewport width (default: 1440)
  --height <px>           Viewport height (default: 900)
  --padding <px>          Extra space around the --testid element (default: 0)
  --full-page             Capture the full scrollable page (ignored with --testid)
  --delay <ms>            Wait after the page settles, for animations (default: 500)
  --timeout <ms>          Max wait for elements (default: 15000)
  --auth <file>           Playwright storage state to reuse (default: screenshots/out/auth.json if present)
  --login                 Open a headed browser to sign in, then save --auth and exit
`;

const OUT_DIR = path.resolve(import.meta.dirname, 'out');

const HIDE_DEVTOOLS_CSS = `
  .tsqd-open-btn-container, .tsqd-parent-container, [class*="ReactQueryDevtools"] {
    display: none !important;
  }
`;

const { values: args } = parseArgs({
  options: {
    route: { type: 'string' },
    testid: { type: 'string' },
    click: { type: 'string', multiple: true, default: [] },
    out: { type: 'string' },
    theme: { type: 'string', default: 'light' },
    'base-url': { type: 'string', default: process.env.STUDIO_URL ?? 'https://localhost:5173/' },
    width: { type: 'string', default: '1440' },
    height: { type: 'string', default: '900' },
    padding: { type: 'string', default: '0' },
    'full-page': { type: 'boolean', default: false },
    delay: { type: 'string', default: '500' },
    timeout: { type: 'string', default: '15000' },
    auth: { type: 'string', default: path.join(OUT_DIR, 'auth.json') },
    login: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
});

// Trailing slash so relative routes resolve under the base path (e.g. /studio/).
const baseUrl = args['base-url'].replace(/\/*$/, '/');
// The dev server uses a local mkcert certificate; keep TLS validation for any other host.
const ignoreHTTPSErrors = ['localhost', '127.0.0.1', '[::1]'].includes(new URL(baseUrl).hostname);

const fail = (message: string): never => {
  console.error(`Error: ${message}\n${USAGE}`);
  process.exit(1);
};

const waitForEnter = (prompt: string) =>
  new Promise<void>((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.question(prompt, () => {
      rl.close();
      resolve();
    });
  });

async function login() {
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext({ ignoreHTTPSErrors });
  await (await context.newPage()).goto(baseUrl);
  await waitForEnter('\nSign in, then press Enter to save the session...\n');
  fs.mkdirSync(path.dirname(args.auth), { recursive: true });
  await context.storageState({ path: args.auth });
  await browser.close();
  console.log(`Saved session to ${args.auth}`);
}

async function capture(page: Page, outPath: string) {
  const timeout = Number(args.timeout);
  const url = new URL(args.route!.replace(/^\//, ''), baseUrl).href;

  console.log(`Navigating to ${url}`);
  const response = await page.goto(url, { waitUntil: 'networkidle', timeout: timeout * 2 });
  if (response && !response.ok()) {
    throw new Error(`${url} responded with HTTP ${response.status()}`);
  }
  await page.addStyleTag({ content: HIDE_DEVTOOLS_CSS });

  for (const id of args.click) {
    console.log(`Clicking [data-testid="${id}"]`);
    await page.getByTestId(id).first().click({ timeout });
  }

  const target = args.testid ? page.getByTestId(args.testid).first() : undefined;
  await target?.waitFor({ state: 'visible', timeout });
  await page.waitForLoadState('networkidle').catch(() => undefined);
  await page.waitForTimeout(Number(args.delay));

  fs.mkdirSync(path.dirname(outPath), { recursive: true });

  if (!target) {
    await page.screenshot({ path: outPath, fullPage: args['full-page'] });
    return;
  }

  const padding = Number(args.padding);
  if (padding === 0) {
    await target.screenshot({ path: outPath });
    return;
  }

  await target.scrollIntoViewIfNeeded();
  const box = await target.boundingBox();
  if (!box) {
    fail(`Element [data-testid="${args.testid}"] has no bounding box`);
    return;
  }
  const x = Math.max(0, box.x - padding);
  const y = Math.max(0, box.y - padding);
  await page.screenshot({
    path: outPath,
    clip: { x, y, width: box.width + padding * 2, height: box.height + padding * 2 },
  });
}

async function main() {
  if (args.help) {
    console.log(USAGE);
    return;
  }
  if (args.login) {
    await login();
    return;
  }
  if (!args.route) {
    fail('--route is required');
  }
  if (args.theme !== 'light' && args.theme !== 'dark') {
    fail('--theme must be "light" or "dark"');
  }

  const outPath = path.resolve(
    args.out ?? path.join(OUT_DIR, `${args.testid ?? 'page'}-${args.theme}.png`)
  );

  const browser = await chromium.launch();
  const context = await browser.newContext({
    ignoreHTTPSErrors,
    viewport: { width: Number(args.width), height: Number(args.height) },
    deviceScaleFactor: 2,
    storageState: fs.existsSync(args.auth) ? args.auth : undefined,
  });
  await context.addInitScript(
    ({ tourKey, themeKey, theme }) => {
      window.localStorage.setItem(tourKey, 'true');
      window.localStorage.setItem(themeKey, JSON.stringify(theme));
    },
    { tourKey: TOUR_SEEN_KEY, themeKey: UI_THEME, theme: args.theme }
  );
  const page = await context.newPage();

  try {
    await capture(page, outPath);
    console.log(`Saved screenshot to ${outPath}`);
  } catch (error) {
    const debugPath = outPath.replace(/\.png$/, '.error.png');
    await page.screenshot({ path: debugPath }).catch(() => undefined);
    console.error(`Capture failed (page state saved to ${debugPath})`);
    throw error;
  } finally {
    await browser.close();
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
