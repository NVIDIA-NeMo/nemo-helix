// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { parse } from "yaml";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const fernDir = resolve(scriptDir, "..");

function git(cwd, ...args) {
  return execFileSync("git", args, {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      GIT_CONFIG_GLOBAL: "/dev/null",
      GIT_CONFIG_SYSTEM: "/dev/null",
      GIT_AUTHOR_NAME: "Docs Test",
      GIT_AUTHOR_EMAIL: "docs@example.com",
      GIT_COMMITTER_NAME: "Docs Test",
      GIT_COMMITTER_EMAIL: "docs@example.com",
    },
  }).trim();
}

function put(root, path, content) {
  mkdirSync(dirname(join(root, path)), { recursive: true });
  writeFileSync(join(root, path), content);
}

function fixture(t, { shallow = false } = {}) {
  const root = mkdtempSync(join(tmpdir(), "helix-branch-docs-test-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const remote = join(root, "origin.git");
  const seed = join(root, "seed");
  const checkout = join(root, "checkout");
  mkdirSync(seed);
  git(root, "init", "--bare", "--initial-branch=main", remote);
  git(seed, "init", "--initial-branch=main");
  git(seed, "config", "core.hooksPath", "/dev/null");
  put(seed, "docs/fern/docs.yml", 'versions:\n  - display-name: Latest\n    path: versions/latest.yml\n    slug: latest\n');
  put(seed, "docs/fern/versions/latest.yml", 'navigation:\n  - page: Overview\n    path: ../../overview.mdx\n');
  put(seed, "docs/overview.mdx", "Branch docs at initial commit\n");
  git(seed, "add", ".");
  git(seed, "commit", "-m", "initial docs");
  git(seed, "tag", "0.8.0");
  git(seed, "remote", "add", "origin", remote);
  git(seed, "push", "origin", "main", "--tags");
  git(root, "clone", ...(shallow ? ["--depth=1"] : []), `file://${remote}`, checkout);
  for (const name of ["materialize-release-versions.mjs", "sync-colab-links.mjs", "sync-helm-docs.mjs"]) {
    const target = join(checkout, "docs/fern/scripts", name);
    mkdirSync(dirname(target), { recursive: true });
    cpSync(join(scriptDir, name), target);
  }
  symlinkSync(join(fernDir, "node_modules"), join(checkout, "docs/fern/node_modules"), "dir");
  const run = (...args) => execFileSync(process.execPath, [
    join(checkout, "docs/fern/scripts/materialize-release-versions.mjs"), ...args,
  ], { cwd: checkout, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  const versions = () => parse(readFileSync(join(checkout, "docs/fern/docs.yml"), "utf8")).versions;
  const branch = (name) => git(seed, "push", "origin", `HEAD:refs/heads/${name}`);
  const commit = (text) => {
    put(seed, "docs/overview.mdx", text);
    git(seed, "add", "docs/overview.mdx");
    git(seed, "commit", "-m", text);
  };
  const redirects = () => parse(readFileSync(join(checkout, "docs/fern/docs.yml"), "utf8")).redirects ?? [];
  return { root, seed, checkout, run, versions, redirects, branch, commit };
}

test("discovers and sorts branch lines, not tag snapshots or local-only branches", (t) => {
  const f = fixture(t);
  for (const branch of ["release/0.9", "release/0.10", "release/0.9.1", "release/not-a-version", "feature/demo"]) {
    f.branch(branch);
  }
  git(f.checkout, "branch", "release/99.0");
  f.run();
  const versions = f.versions().slice(1);
  assert.deepEqual(versions.map((v) => v["display-name"]), ["0.10", "0.9.1", "0.9"]);
  assert.deepEqual(versions.map((v) => v.slug), ["v0.10", "v0.9.1", "v0.9"]);
  assert.ok(versions.every((v) => v.availability === "preview"));
  assert.ok(!existsSync(join(f.checkout, "docs/fern/versions/v0.8.0.yml")));
  assert.match(readFileSync(join(f.checkout, "docs/fern/versions/v0.9.yml"), "utf8"), /generated\/release-versions\/v0.9\/overview.mdx/);
});

for (const basepath of ["", "/nemo-helix"]) {
  test(`links Latest to the newest released branch only (basepath '${basepath}')`, (t) => {
    const f = fixture(t);
    const config = join(f.checkout, "docs/fern/docs.yml");
    const latest = parse(readFileSync(join(fernDir, "docs.yml"), "utf8")).versions[0];
    const warning = latest.announcement.message;
    assert.equal(latest["display-name"], "Latest");
    assert.match(warning, /unreleased development docs/);
    assert.ok(!warning.includes("](")); // Authored fallback makes no release claim.
    writeFileSync(config, readFileSync(config, "utf8")
      + `    announcement:\n      message: ${JSON.stringify(warning)}\n`
      + (basepath ? `instances:\n  - url: example.com/other\n    custom-domain: docs.example.com${basepath}\n` : ""));
    f.branch("release/0.8"); // The fixture's reachable 0.8.0 tag marks this line released.
    f.commit("0.9 development");
    f.branch("release/0.9");
    f.commit("0.10 development");
    f.branch("release/0.10");

    const check = (stable, slugs, availability) => {
      const versions = f.versions();
      const message = warning + (stable
        ? ` See the [latest released docs](${basepath}/${stable}/documentation/home).`
        : "");
      assert.deepEqual(versions[0], { ...latest, announcement: { message } });
      assert.deepEqual(versions.map((v) => v.slug), ["latest", ...slugs]);
      assert.deepEqual(versions.slice(1).map((v) => v.availability), availability);
      assert.ok(versions.every((v) => !("default" in v)));
      assert.ok(versions.slice(1).every((v) => !("announcement" in v)));
    };
    f.run();
    check("v0.8", ["v0.10", "v0.9", "v0.8"], ["preview", "preview", undefined]);
    const first = readFileSync(config, "utf8");
    f.run();
    assert.equal(readFileSync(config, "utf8"), first);

    // Tag the actual 0.9 branch tip, not the later 0.10 development commit.
    git(f.seed, "tag", "0.9.0", "HEAD~1");
    git(f.seed, "push", "origin", "--tags");
    f.run();
    check("v0.9", ["v0.10", "v0.9", "v0.8"], ["preview", undefined, undefined]);

    // Removing all stable candidates must discard the previously generated link.
    git(f.seed, "push", "origin", ":refs/heads/release/0.8", ":refs/tags/0.9.0");
    f.run();
    check(undefined, ["v0.10", "v0.9"], ["preview", "preview"]);
    const fallback = readFileSync(config, "utf8");
    f.run();
    assert.equal(readFileSync(config, "utf8"), fallback);
  });
}

test("only reachable stable tags from the branch's own release line remove preview", (t) => {
  const f = fixture(t);
  f.commit("0.9 release work");
  f.branch("release/0.9");
  git(f.seed, "tag", "v0.9.0-rc.1");
  git(f.seed, "push", "origin", "--tags");
  // A matching stable tag elsewhere in the repo is not enough.
  f.commit("unmerged work");
  git(f.seed, "tag", "0.9.2");
  git(f.seed, "push", "origin", "--tags");
  f.run();
  assert.equal(f.versions()[1].availability, "preview");
  git(f.seed, "tag", "-a", "v0.9.0", "HEAD~1", "-m", "stable release");
  git(f.seed, "push", "origin", "--tags");
  f.run();
  assert.equal(f.versions()[1].availability, undefined);
  // All patch tags collapse into the same branch entry, not separate versions.
  assert.equal(f.versions().length, 2);
});

test("always refreshes branch tips and keeps serving branch docs after a tag", (t) => {
  const f = fixture(t);
  f.branch("release/0.8");
  f.run();
  assert.equal(f.versions()[1].availability, undefined);
  f.commit("Updated docs after the tag\n");
  f.branch("release/0.8");
  f.run();
  assert.equal(readFileSync(join(f.checkout, "docs/fern/generated/release-versions/v0.8/overview.mdx"), "utf8"), "Updated docs after the tag\n");
  assert.equal(f.versions().length, 2);
});

test("ignores local-only tags and prunes deleted remote tags without touching local tags", (t) => {
  const f = fixture(t);
  f.branch("release/0.9");
  git(f.checkout, "tag", "0.9.0");
  f.run();
  assert.equal(f.versions()[1].availability, "preview");
  git(f.seed, "tag", "0.9.1");
  git(f.seed, "push", "origin", "--tags");
  f.run();
  assert.equal(f.versions()[1].availability, undefined);
  git(f.seed, "push", "origin", "--delete", "refs/tags/0.9.1");
  f.run();
  assert.equal(f.versions()[1].availability, "preview");
  assert.equal(git(f.checkout, "tag", "--list", "0.9.0"), "0.9.0");
});

test("patch branches require a reachable stable tag for that exact patch", (t) => {
  const f = fixture(t);
  f.branch("release/0.8.1");
  f.run();
  assert.equal(f.versions()[1].availability, "preview");
  git(f.seed, "tag", "0.8.1+build.1");
  git(f.seed, "push", "origin", "--tags");
  f.run();
  assert.equal(f.versions()[1].availability, undefined);
});

test("prunes deleted branches and old generated tag versions without deleting hand-authored nav", (t) => {
  const f = fixture(t);
  f.branch("release/0.9");
  const header = "# Generated by scripts/materialize-release-versions.mjs; do not commit.\n";
  put(f.checkout, "docs/fern/versions/v0.8.0.yml", header + "navigation: []\n");
  put(f.checkout, "docs/fern/versions/custom.yml", "navigation: []\n");
  put(f.checkout, "docs/fern/docs.yml", readFileSync(join(f.checkout, "docs/fern/docs.yml"), "utf8") + '  - display-name: Old tag\n    path: versions/v0.8.0.yml\n    slug: v0.8.0\n');
  f.run();
  assert.ok(!existsSync(join(f.checkout, "docs/fern/versions/v0.8.0.yml")));
  git(f.seed, "push", "origin", "--delete", "release/0.9");
  f.run();
  assert.equal(f.versions().length, 1);
  assert.ok(!existsSync(join(f.checkout, "docs/fern/versions/v0.9.yml")));
  assert.ok(!existsSync(join(f.checkout, "docs/fern/generated/release-versions/v0.9")));
  assert.ok(existsSync(join(f.checkout, "docs/fern/versions/custom.yml")));
});

test("skips branches without Fern nav and fails on remote errors before touching output", (t) => {
  const f = fixture(t);
  git(f.seed, "rm", "docs/fern/versions/latest.yml");
  git(f.seed, "commit", "-m", "no Fern nav");
  f.branch("release/0.1");
  assert.match(f.run(), /does not contain docs\/fern\/versions\/latest.yml/);
  assert.equal(f.versions().length, 1);
  const before = readFileSync(join(f.checkout, "docs/fern/docs.yml"), "utf8");
  put(f.checkout, "docs/fern/generated/release-versions/sentinel", "keep");
  git(f.checkout, "remote", "set-url", "origin", join(f.root, "missing-origin"));
  assert.throws(() => f.run(), /Command failed/);
  assert.equal(readFileSync(join(f.checkout, "docs/fern/docs.yml"), "utf8"), before);
  assert.ok(existsSync(join(f.checkout, "docs/fern/generated/release-versions/sentinel")));
});

test("unshallows history so older same-line tags still mark a branch as released", (t) => {
  const f = fixture(t, { shallow: true });
  f.commit("docs after the stable tag");
  f.branch("release/0.8");
  assert.equal(git(f.checkout, "rev-parse", "--is-shallow-repository"), "true");
  f.run();
  assert.equal(git(f.checkout, "rev-parse", "--is-shallow-repository"), "false");
  assert.equal(f.versions()[1].availability, undefined);
});

test("redirects legacy 0.5 through 0.8 patch URLs without adding picker entries", (t) => {
  const f = fixture(t);
  for (const version of ["0.4.0", "0.5.0", "0.5.1", "0.6.0", "0.7.2", "v0.8.1", "0.9.0"]) {
    git(f.seed, "tag", version);
  }
  git(f.seed, "push", "origin", "--tags");
  for (const line of ["0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]) {
    f.branch(`release/${line}`);
  }
  const config = join(f.checkout, "docs/fern/docs.yml");
  writeFileSync(config, "instances:\n  - url: example.com/nemo-helix\n    custom-domain: docs.example.com/nemo-helix\n" + readFileSync(config, "utf8"));
  f.run();
  const aliases = ["0.5.0", "0.5.1", "0.6.0", "0.7.2", "0.8.0", "0.8.1"];
  assert.equal(f.redirects().length, aliases.length * 2);
  for (const version of aliases) {
    const line = version.split(".").slice(0, 2).join(".");
    for (const suffix of ["", "/:path*"]) {
      assert.deepEqual(f.redirects().find((r) => r.source === `/v${version}${suffix}`), {
        source: `/v${version}${suffix}`, destination: `/nemo-helix/v${line}${suffix}`,
      });
    }
  }
  assert.deepEqual(f.versions().slice(1).map((v) => v["display-name"]), ["0.9", "0.8", "0.7", "0.6", "0.5", "0.4"]);
  assert.ok(!f.redirects().some((r) => r.source.startsWith("/v0.4") || r.source.startsWith("/v0.9")));
});

test("preserves authored redirects and refreshes generated aliases idempotently", (t) => {
  const f = fixture(t);
  f.branch("release/0.8");
  git(f.seed, "tag", "0.8.1");
  git(f.seed, "push", "origin", "--tags");
  const config = join(f.checkout, "docs/fern/docs.yml");
  writeFileSync(config, readFileSync(config, "utf8") + 'redirects:\n  - source: /v0.8.0\n    destination: /manual\n');
  f.run();
  const first = f.redirects();
  assert.equal(first.find((r) => r.source === "/v0.8.0").destination, "/manual");
  assert.equal(first.length, 4);
  f.run();
  assert.deepEqual(f.redirects(), first);
  git(f.seed, "push", "origin", "--delete", "refs/tags/0.8.1");
  f.run();
  assert.equal(f.redirects().length, 2);
  assert.ok(!f.redirects().some((r) => r.source.startsWith("/v0.8.1")));
  git(f.seed, "push", "origin", "--delete", "release/0.8");
  f.run();
  assert.deepEqual(f.redirects(), [{ source: "/v0.8.0", destination: "/manual" }]);
});

test("mirrors each release branch's retired-page redirects under its version slug", (t) => {
  const f = fixture(t);
  put(f.seed, "docs/fern/docs.yml", [
    "versions:",
    "  - display-name: Latest",
    "    path: versions/latest.yml",
    "    slug: latest",
    "redirects:",
    "  - source: /documentation/old-page",
    "    destination: /documentation/new-page",
    "  - source: /latest/documentation/old-page",
    "    destination: /latest/documentation/new-page",
    "  - source: /latest/documentation/legacy/:path*",
    "    destination: /latest/documentation/current/:path*",
    "  - source: /latest/documentation/elsewhere",
    "    destination: https://example.com/elsewhere",
    "",
  ].join("\n"));
  git(f.seed, "add", ".");
  git(f.seed, "commit", "-m", "0.8 redirects without basepath");
  f.branch("release/0.8");
  put(f.seed, "docs/fern/docs.yml", [
    "versions:",
    "  - display-name: Latest",
    "    path: versions/latest.yml",
    "    slug: latest",
    "redirects:",
    "  - source: /latest/documentation/old-page",
    "    destination: /nemo-helix/latest/documentation/new-page",
    "  - source: /latest/documentation/only-in-09",
    "    destination: /nemo-helix/latest/documentation/renamed-in-09",
    "",
  ].join("\n"));
  git(f.seed, "add", ".");
  git(f.seed, "commit", "-m", "0.9 redirects with basepath");
  f.branch("release/0.9");
  const config = join(f.checkout, "docs/fern/docs.yml");
  writeFileSync(
    config,
    "instances:\n  - url: example.com/nemo-helix\n    custom-domain: docs.example.com/nemo-helix\n"
      + readFileSync(config, "utf8")
      + "redirects:\n  - source: /v0.9/documentation/only-in-09\n    destination: /manual\n",
  );
  f.run();
  const find = (source) => f.redirects().find((r) => r.source === source);
  assert.deepEqual(find("/v0.8/documentation/old-page"), {
    source: "/v0.8/documentation/old-page", destination: "/nemo-helix/v0.8/documentation/new-page",
  });
  assert.deepEqual(find("/v0.8/documentation/legacy/:path*"), {
    source: "/v0.8/documentation/legacy/:path*", destination: "/nemo-helix/v0.8/documentation/current/:path*",
  });
  assert.deepEqual(find("/v0.9/documentation/old-page"), {
    source: "/v0.9/documentation/old-page", destination: "/nemo-helix/v0.9/documentation/new-page",
  });
  // Authored redirects win, unversioned sources are not mirrored, and external targets are skipped.
  assert.equal(find("/v0.9/documentation/only-in-09").destination, "/manual");
  assert.equal(find("/v0.8/documentation/only-in-09"), undefined);
  assert.equal(find("/v0.8/documentation/elsewhere"), undefined);
  assert.ok(!f.redirects().some((r) => r.source.startsWith("/v0.8/latest") || r.source === "/v0.8/old-page"));
  const first = f.redirects();
  f.run();
  assert.deepEqual(f.redirects(), first);
});

test("does not redirect away an existing patch-branch version, even a preview", (t) => {
  const f = fixture(t);
  f.branch("release/0.8.1");
  f.commit("Minor branch release not yet merged to patch branch");
  git(f.seed, "tag", "0.8.1");
  git(f.seed, "push", "origin", "--tags");
  f.branch("release/0.8");
  f.run();
  assert.deepEqual(f.versions().slice(1).map((v) => v["display-name"]), ["0.8.1", "0.8"]);
  assert.ok(!f.redirects().some((r) => r.source.startsWith("/v0.8.1")));
  assert.equal(f.redirects().find((r) => r.source === "/v0.8.0").destination, "/v0.8");
});

test("retains stable-tag publish eligibility without supporting tag snapshots", (t) => {
  const f = fixture(t);
  for (const tag of ["0.9.0", "v0.9.1", "0.9.1+build.1"]) {
    assert.equal(f.run("--check-tag", tag).trim(), "eligible=true");
  }
  for (const tag of ["0.9.0-rc.1", "nightly-20261023", "docs/v0.9.0"]) {
    assert.equal(f.run("--check-tag", tag).trim(), "eligible=false");
  }
  assert.throws(() => f.run("--tag", "0.8.0"), /Command failed/);
});
