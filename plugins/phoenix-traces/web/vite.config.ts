// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { studioPlugin } from "../../../web/packages/common/plugin-build/studio-plugin.mjs";

export default defineConfig({
  plugins: [react(), studioPlugin()],
  build: {
    lib: { entry: "src/index.ts", formats: ["es"], fileName: () => "index.js" },
    outDir: "../dist",
    emptyOutDir: true,
  },
});
