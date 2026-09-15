// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin } from 'vite';

// Keep in sync with the keys of VENDOR_IMPORT_MAP in
// web/packages/studio/vite.config.ts — that map, not VENDOR_EXTERNALS, is what
// the browser resolves at runtime. Studio serves one shared instance of each of
// these, and the plugin bundle must leave them as bare specifiers to reach it.
const STUDIO_SHARED_DEPS = [
  'react',
  'react/jsx-runtime',
  'react-dom',
  'react-dom/client',
  'react-router',
  // Studio's design system, shared via the import map so the plugin's KUI
  // components use Studio's single foundations instance and theme.
  '@nvidia/foundations-react-core',
  // Shared so the plugin's useQuery uses Studio's QueryClient (one cache).
  '@tanstack/react-query',
  // Studio's shared UI. Bare specifier only — deep paths aren't mapped.
  '@nemo/common',
];

// A deep import isn't externalized, so it silently bundles a duplicate.
const rejectDeepSharedImports = (): Plugin => ({
  name: 'reject-deep-shared-imports',
  // Must beat Vite's core resolver, which would resolve the path first.
  enforce: 'pre',
  resolveId(id: string, importer: string | undefined) {
    if (id.startsWith('@nemo/common/')) {
      throw new Error(
        `Deep import "${id}"${importer ? ` from ${importer}` : ''}.\n` +
          `Only the bare "@nemo/common" specifier is in Studio's import map, so ` +
          `a deep path is not externalized and bundles a duplicate copy of the ` +
          `component. Import from "@nemo/common"; if the name is missing, add it ` +
          `to web/packages/common/src/plugin.ts.`
      );
    }
    return null;
  },
});

// Prepended to the built bundle so the emitted artifact keeps an SPDX header —
// minification strips source comments, and CI's copyright-header check
// (script/copyright_fixer.py) requires the header literally in the file.
const LICENSE_BANNER =
  '// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n' +
  '// SPDX-License-Identifier: Apache-2.0';

export default defineConfig({
  plugins: [react(), rejectDeepSharedImports()],
  resolve: {
    alias: {
      '@agent-hardener': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    lib: {
      entry: 'src/index.ts',
      formats: ['es'],
      fileName: () => 'index.js',
    },
    // Build into the Python package so it gets included in wheel installs
    outDir: '../src/nemo_agent_hardener_plugin/web/dist',
    emptyOutDir: true,
    rolldownOptions: {
      external: STUDIO_SHARED_DEPS,
      output: {
        banner: LICENSE_BANNER,
        // StudioSpec registers a single `web/dist/index.js`, and the built
        // artifact is committed. Without this, react-diff-viewer's optional
        // Prism languages split into ~35 hashed chunks that churn on every
        // build — none of which this plugin ever loads.
        inlineDynamicImports: true,
      },
    },
  },
});
