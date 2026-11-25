// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const { Buffer } = require("node:buffer");
const { readFileSync } = require("node:fs");
const test = require("node:test");
const { readReleaseCatalog } = require("../release-plan.cjs");
const {
  addedImages,
  selectImages,
  dispatchBootstrap,
  waitForBootstrap,
  checkPackages,
} = require("../release-package-access.cjs");

function catalog(ids) {
  return `env:\n  RELEASE_WHEELS_JSON: >-\n    []\n  RELEASE_CONTAINERS_JSON: >-\n    ${JSON.stringify(ids.map((id) => ({ id, target: `${id}-docker` })))}\n`;
}

function fixture() {
  const repository = { full_name: "NVIDIA-NeMo/nemo-helix" };
  return {
    context: {
      repo: { owner: "NVIDIA-NeMo", repo: "nemo-helix" },
      runId: 123,
      payload: {
        pull_request: {
          base: { sha: "b".repeat(40), repo: repository },
          head: { sha: "a".repeat(40), repo: repository },
        },
      },
    },
    env: {
      GITHUB_RUN_ATTEMPT: "2",
      DISPATCH_REPO: "NVIDIA-NeMo/Platform-Deploy",
      IMAGE_IDS: '["nhx-example"]',
    },
    core: { info() {}, setOutput() {} },
  };
}

test("only additions and renames need a bootstrap", () => {
  assert.deepEqual(addedImages(catalog(["a", "b"]), catalog(["b", "a"])), []);
  assert.deepEqual(addedImages(catalog(["a", "b"]), catalog(["a"])), []);
  assert.deepEqual(addedImages(catalog(["a"]), catalog(["a", "b"])), ["b"]);
  assert.deepEqual(addedImages(catalog(["a"]), catalog(["renamed"])), [
    "renamed",
  ]);
  assert.throws(() => addedImages("invalid", catalog([])), /missing/);
});

test("forks without additions pass; forks adding release images fail", async () => {
  const input = fixture();
  input.context.payload.pull_request.head.repo = {
    full_name: "contributor/nemo-helix",
  };
  const current = readFileSync(".github/workflows/release.yaml", "utf8");
  let base = current;
  input.github = {
    rest: {
      repos: {
        getContent: async ({ ref }) => {
          assert.equal(ref, "b".repeat(40));
          return { data: { content: Buffer.from(base).toString("base64") } };
        },
      },
    },
  };
  input.core.setOutput = (name, value) =>
    assert.deepEqual([name, value], ["images", "[]"]);
  await selectImages(input);
  base = catalog(
    readReleaseCatalog(current)
      .containers.slice(1)
      .map(({ id }) => id),
  );
  await assert.rejects(selectImages(input), /same-repository PR/);
});

test("dispatch includes the exact PR head and CI attempt", async () => {
  const input = fixture();
  let request;
  input.github = {
    rest: {
      repos: {
        createDispatchEvent: async (value) => {
          request = value;
        },
      },
    },
  };
  await dispatchBootstrap(input);
  assert.deepEqual(request, {
    owner: "NVIDIA-NeMo",
    repo: "Platform-Deploy",
    event_type: "release-package-access",
    client_payload: {
      sha: "a".repeat(40),
      images: ["nhx-example"],
      run_id: "123",
      run_attempt: "2",
    },
  });
  input.github.rest.repos.createDispatchEvent = async () => {
    throw new Error("dispatch denied");
  };
  await assert.rejects(dispatchBootstrap(input), /dispatch denied/);
});

function pollingInput(responses) {
  const input = fixture();
  let elapsed = 0;
  input.now = () => elapsed;
  input.wait = async (milliseconds) => {
    elapsed += milliseconds;
  };
  input.github = {
    rest: { repos: { listCommitStatusesForRef() {} } },
    paginate: async (_method, args) => {
      assert.equal(args.ref, "a".repeat(40));
      return responses.shift() ?? [];
    },
  };
  return input;
}

test("waits for this attempt and ignores an earlier success", async () => {
  const input = pollingInput([
    [{ context: "release-package-bootstrap/123/1", state: "success" }],
    [{ context: "release-package-bootstrap/123/2", state: "pending" }],
    [{ context: "release-package-bootstrap/123/2", state: "success" }],
  ]);
  await waitForBootstrap(input);
  assert.equal(input.now(), 30_000);
});

for (const state of ["failure", "error"]) {
  test(`publisher ${state} fails the Helix job`, async () => {
    const input = pollingInput([
      [
        {
          context: "release-package-bootstrap/123/2",
          state,
          target_url: "https://example.com/run",
        },
      ],
    ]);
    await assert.rejects(
      waitForBootstrap(input),
      /could not publish.*https:\/\/example.com\/run/,
    );
  });
}

test("a missing callback times out after ten minutes", async () => {
  const input = pollingInput([]);
  await assert.rejects(waitForBootstrap(input), /Timed out/);
  assert.equal(input.now(), 600_000);
});

test("checks package settings and the exact bootstrap manifest", async () => {
  const input = fixture();
  let metadata = {
    visibility: "public",
    repository: { full_name: "NVIDIA-NeMo/nemo-helix" },
  };
  const inspected = [];
  input.github = {
    rest: {
      packages: {
        getPackageForOrganization: async (args) => {
          assert.equal(args.package_name, "nemo-helix/nhx-example");
          return { data: metadata };
        },
      },
    },
  };
  input.execute = (command, args) => inspected.push([command, args]);
  await checkPackages(input);
  assert.deepEqual(inspected, [
    [
      "docker",
      [
        "manifest",
        "inspect",
        "ghcr.io/nvidia-nemo/nemo-helix/nhx-example:bootstrap-test",
      ],
    ],
  ]);

  metadata.visibility = "private";
  await assert.rejects(checkPackages(input), /must be public/);
  metadata = {
    visibility: "public",
    repository: { full_name: "NVIDIA-NeMo/Platform-Deploy" },
  };
  await assert.rejects(checkPackages(input), /must be linked/);
  metadata.repository = undefined;
  await assert.rejects(checkPackages(input), /must be linked/);
  metadata.repository = { full_name: "NVIDIA-NeMo/nemo-helix" };
  input.execute = () => {
    throw new Error("unauthorized");
  };
  await assert.rejects(checkPackages(input), /Helix CI cannot read/);
  input.github.rest.packages.getPackageForOrganization = async () => {
    throw new Error("404");
  };
  await assert.rejects(checkPackages(input), /missing or inaccessible/);
});

test("CI wires the path filter and job into its existing merge gate", () => {
  const workflow = readFileSync(".github/workflows/ci.yaml", "utf8");
  const changes = readFileSync(".github/actions/changes/action.yaml", "utf8");
  assert.match(
    changes,
    /release-packages:\n\s+- '\.github\/workflows\/release\.yaml'/,
  );
  assert.match(
    workflow,
    /ci-status:[\s\S]*?needs:[\s\S]*? - release-package-access\n/,
  );
  assert.match(
    workflow,
    /release-package-access:[\s\S]*?github.event_name == 'pull_request' &&\s+needs.changes.outputs.release-packages == 'true'/,
  );
});
