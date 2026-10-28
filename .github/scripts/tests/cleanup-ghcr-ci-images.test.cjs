// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const process = require("node:process");
const test = require("node:test");

const script = fs.readFileSync(
  require.resolve("../cleanup-ghcr-ci-images.sh"),
  "utf8",
);
// Exercise the actual jq selector without contacting GitHub or deleting images.
const selector =
  script.match(/^release_tag_pattern=.*$/m)[0] +
  "\n" +
  script.slice(script.indexOf("case "), script.indexOf('\necho "Scanning')) +
  "\n" +
  script.slice(
    script.indexOf("find_stale_versions()"),
    script.indexOf("find_tagged_versions()"),
  );

function candidates(
  scope,
  cutoff,
  tags = [
    "rc-20261021010203",
    "0.9.0-rc-20261021010203",
    "nightly-20261021010203",
    "0.9.0-nightly-20261021010203",
    "0.9.0-night-20261021010203",
    "a".repeat(40),
    "latest",
  ],
) {
  const versions = tags.map((tag, i) => ({
    id: i,
    name: "sha256:" + i,
    updated_at: "2026-10-21T01:02:03Z",
    metadata: { container: { tags: [tag] } },
  }));
  const result = spawnSync(
    "bash",
    ["-eu", "-c", selector + '\nfind_stale_versions "$PACKAGE_VERSIONS"'],
    {
      encoding: "utf8",
      env: {
        PATH: process.env.PATH,
        CLEANUP_SCOPE: scope,
        cutoff,
        PACKAGE_VERSIONS: versions.map((v) => JSON.stringify(v)).join("\n"),
      },
      timeout: 10_000,
    },
  );
  assert.equal(result.status, 0, result.stderr);
  return result.stdout.trim()
    ? result.stdout
        .trim()
        .split("\n")
        .map((line) => Number(line.split("\t")[0]))
    : [];
}

test("CI cleanup preserves RC and historical release artifacts", () => {
  assert.deepEqual(candidates("ci", "2026-10-23T00:00:00Z"), [5]);
});

test("other timestamped tags retain the CI cleanup policy", () => {
  const tags = [
    "main-20261021010203",
    "cache-20261021010203",
    "0.9.0-main-20261021010203",
  ];
  assert.deepEqual(candidates("ci", "2026-10-23T00:00:00Z", tags), [0, 1, 2]);
  assert.deepEqual(candidates("release", "2027-01-23T00:00:00Z", tags), []);
});

test("release cleanup excludes cache tags and unrelated version formats", () => {
  assert.deepEqual(
    candidates("release", "2027-01-23T00:00:00Z", [
      "cache-main-linux-amd64",
      "cache-release-0-9-linux-arm64",
      "cache-rc-20261021010203-linux-amd64",
      "0.9.0",
      "0.9.0-rc1",
      "rc-20261021",
      "rc-20261021010203-amd64",
      "night-20261021010203",
    ]),
    [],
  );
});

test("release cleanup preserves recent releases and selects expired releases", () => {
  assert.deepEqual(candidates("release", "2026-07-25T00:00:00Z"), []);
  assert.deepEqual(
    candidates("release", "2027-01-23T00:00:00Z"),
    [0, 1, 2, 3, 4],
  );
});
