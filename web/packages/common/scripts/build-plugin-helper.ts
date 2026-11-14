// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { build } from 'rolldown';

// Committed, not gitignored: `web-static` in CI regenerates this and fails on a
// diff, so a contract change can't land without the helper out-of-tree plugins
// vendor moving with it.
const OUT_DIR = 'plugin-build';

const LICENSE_BANNER =
  '// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n' +
  '// SPDX-License-Identifier: Apache-2.0';

// Plugin vite configs can't import @nemo/common's TypeScript sources (out-of-tree
// repos don't have them), so roll `studioPlugin()` and the contract it reads up
// into one dependency-free ESM file they can import or vendor. Node builtins stay
// external; `rolldown` is a type-only import and is erased.
await build({
  input: 'src/pluginContract/studioPlugin.ts',
  platform: 'node',
  // Resolves the `@nemo/common/*` imports through tsconfig `paths`.
  tsconfig: 'tsconfig.json',
  output: {
    dir: OUT_DIR,
    format: 'esm',
    entryFileNames: 'studio-plugin.mjs',
    banner: LICENSE_BANNER,
    sourcemap: false,
  },
});

// eslint-disable-next-line no-console
console.log(`Plugin build helper written to ${OUT_DIR}/studio-plugin.mjs`);
