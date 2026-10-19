#!/usr/bin/env node
/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fern currently prints broken-link errors while still exiting successfully.
 * Treat reported errors as a lint failure so CI can block regressions.
 */

import { spawn } from "node:child_process";

const child = spawn("npx", ["-y", "fern-api@latest", "docs", "broken-links"], {
  stdio: ["ignore", "pipe", "pipe"],
});

let output = "";

function record(chunk, stream) {
  const text = chunk.toString();
  output += text;
  stream.write(text);
}

child.stdout.on("data", (chunk) => record(chunk, process.stdout));
child.stderr.on("data", (chunk) => record(chunk, process.stderr));

child.on("error", (error) => {
  console.error(`check-broken-links: failed to start Fern CLI: ${error.message}`);
  process.exit(1);
});

child.on("close", (code) => {
  if (code !== 0) {
    process.exit(code ?? 1);
  }

  const match = output.match(/Found\s+([1-9][0-9]*)\s+errors?\b/i);
  if (match) {
    console.error(`check-broken-links: Fern reported ${match[1]} broken link error(s)`);
    process.exit(1);
  }
});
