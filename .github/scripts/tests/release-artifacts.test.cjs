// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for release artifact validation.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const { validateReleaseArtifacts } = require("../release-artifacts.cjs");

function createSourceTree() {
  const sourceRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "nemo-release-artifacts-"),
  );
  fs.mkdirSync(path.join(sourceRoot, "packages", "nemo_helix"), {
    recursive: true,
  });
  fs.mkdirSync(
    path.join(sourceRoot, ".github", "assets", "ngc", "containers"),
    { recursive: true },
  );
  fs.writeFileSync(
    path.join(sourceRoot, "packages", "nemo_helix", "pyproject.toml"),
    '[project]\nname = "nemo-helix"\n',
  );
  fs.writeFileSync(
    path.join(sourceRoot, "docker-bake.hcl"),
    'target "nhx-api-docker" {}\n',
  );
  fs.writeFileSync(
    path.join(
      sourceRoot,
      ".github",
      "assets",
      "ngc",
      "containers",
      "nhx-api.md",
    ),
    "# API\n",
  );
  return sourceRoot;
}

function selectedArtifacts() {
  return {
    wheels: [
      {
        id: "nemo-helix",
        package: "nemo-helix",
        path: "packages/nemo_helix",
      },
    ],
    containers: [{ id: "nhx-api", target: "nhx-api-docker" }],
  };
}

test("validates selected wheel and container artifacts", (t) => {
  const sourceRoot = createSourceTree();
  t.after(() => fs.rmSync(sourceRoot, { recursive: true, force: true }));

  assert.deepEqual(
    validateReleaseArtifacts({ ...selectedArtifacts(), sourceRoot }),
    {
      wheels: "nemo-helix",
      containers: "nhx-api",
    },
  );
});

test("rejects a wheel whose project name does not match the release catalog", (t) => {
  const sourceRoot = createSourceTree();
  t.after(() => fs.rmSync(sourceRoot, { recursive: true, force: true }));
  fs.writeFileSync(
    path.join(sourceRoot, "packages", "nemo_helix", "pyproject.toml"),
    '[project]\nname = "other-package"\n',
  );

  assert.throws(
    () => validateReleaseArtifacts({ ...selectedArtifacts(), sourceRoot }),
    /does not declare that project name/,
  );
});

test("rejects a container without matching NGC metadata", (t) => {
  const sourceRoot = createSourceTree();
  t.after(() => fs.rmSync(sourceRoot, { recursive: true, force: true }));
  fs.rmSync(
    path.join(
      sourceRoot,
      ".github",
      "assets",
      "ngc",
      "containers",
      "nhx-api.md",
    ),
  );

  assert.throws(
    () => validateReleaseArtifacts({ ...selectedArtifacts(), sourceRoot }),
    /missing matching NGC metadata/,
  );
});
