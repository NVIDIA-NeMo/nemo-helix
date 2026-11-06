// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { studioPlugin } from "../../../web/packages/common/plugin-build/studio-plugin.mjs";

// studioPlugin() externalizes the specifiers in Studio's plugin contract (bare
// specifiers the browser resolves to Studio's shared instances via its import
// map), fails the build on deep imports of shared packages (a deep path isn't
// externalized and would bundle a duplicate), and emits plugin.manifest.json.

// Prepended to the built bundle so the emitted artifact keeps an SPDX header —
// minification strips source comments, and CI's copyright-header check
// (script/copyright_fixer.py) requires the header literally in the file.
const LICENSE_BANNER =
  "// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n" +
  "// SPDX-License-Identifier: Apache-2.0";

export default defineConfig({
  plugins: [react(), studioPlugin()],
  build: {
    lib: {
      entry: "src/index.ts",
      formats: ["es"],
      fileName: () => "index.js",
    },
    // Build into the Python package so it gets included in wheel installs
    outDir: "../src/nemo_example_plugin/web/dist",
    emptyOutDir: true,
    rolldownOptions: {
      output: {
        banner: LICENSE_BANNER,
      },
    },
  },
});
