// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

function resolveRelease(version) {
  const match = /^(\d+\.\d+\.\d+)(?:-(nightly-\d{14}|(?:a|b|rc)\d+))?$/.exec(
    version,
  );
  if (!match || match[0] !== version)
    throw new Error(
      "Use a full chart version, e.g. 0.8.0 or 0.8.0-nightly-20261021000000",
    );
  const [, base, suffix] = match;
  const nightly = suffix?.startsWith("nightly-");
  return {
    chart_version: version,
    wheel_version: nightly
      ? `${base}.dev${suffix.slice(8)}`
      : `${base}${suffix || ""}`,
    wheel_index: nightly
      ? "https://pypi.nvidia.com"
      : "https://pypi.org/simple",
    image_registry: nightly
      ? "ghcr.io/nvidia-nemo/nemo-helix"
      : "nvcr.io/nvidia/nemo-helix",
    image_tag: nightly ? suffix : version,
    chart_repository: nightly
      ? "oci://ghcr.io/nvidia-nemo/nemo-helix/nemo-helix"
      : "https://helm.ngc.nvidia.com/nvidia/nemo-helix",
  };
}

// The rollout switch stays off until stable/nightly manual validation is complete.
function smokeEligible(env) {
  return (
    env.SMOKE_ENABLED === "true" &&
    env.RELEASE_SCOPE === "all" &&
    ["stable", "nightly"].includes(env.RELEASE_TYPE) &&
    env.POLL_RESULT === "success" &&
    env.DRY_RUN !== "true" &&
    env.ACT !== "true"
  );
}

function summarize(results) {
  if (results.some((result) => result === "failure")) return "failed";
  if (results.length > 0 && results.every((result) => result === "success"))
    return "passed";
  return "cancelled/incomplete";
}

async function finishSmoke({ env, fetchImpl }) {
  const result = summarize([
    env.PREPARE_RESULT,
    env.MACOS_RESULT,
    env.KUBERNETES_RESULT,
  ]);
  const text = [
    `*Release smoke checks ${result}*`,
    `Release: ${env.RELEASE_VERSION}`,
    `macOS: ${env.MACOS_RESULT}${env.MACOS_STAGE ? ` (${env.MACOS_STAGE})` : ""}`,
    `Kubernetes: ${env.KUBERNETES_RESULT}${env.KUBERNETES_STAGE ? ` (${env.KUBERNETES_STAGE})` : ""}`,
    `:link: <${env.RUN_URL}|Smoke check results and diagnostics>`,
  ].join("\n");
  if (env.SEND_NOTIFICATIONS === "true") {
    if (!env.SLACK_RELEASE_WEBHOOK)
      throw new Error(
        "SLACK_RELEASE_WEBHOOK is required when notifications are enabled",
      );
    const response = await fetchImpl(env.SLACK_RELEASE_WEBHOOK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: globalThis.AbortSignal.timeout(30000),
    });
    if (!response.ok)
      throw new Error(`Slack webhook returned ${response.status}`);
  }
  return { result, text };
}

function reportSmoke({ core, env, steps, fs }) {
  const directory = env.SMOKE_RESULTS;
  fs.mkdirSync(directory, { recursive: true });
  // Always sanitize before upload, even if the cleanup step was interrupted.
  const values = ["NGC_REGISTRY_READ_TOKEN", "GH_TOKEN"]
    .map((key) => env[key])
    .filter(Boolean);
  function redact(path) {
    for (const entry of fs.readdirSync(path, { withFileTypes: true })) {
      const file = `${path}/${entry.name}`;
      if (entry.isDirectory()) redact(file);
      else if (entry.isFile() && !file.endsWith(".png")) {
        let contents = fs.readFileSync(file, "utf8");
        for (const value of values)
          contents = contents.replaceAll(value, "[REDACTED]");
        fs.writeFileSync(file, contents);
      }
    }
  }
  redact(directory);
  const file = `${directory}/result.json`;
  const report = fs.existsSync(file)
    ? JSON.parse(fs.readFileSync(file, "utf8"))
    : {
        release: env.RELEASE_VERSION,
        result: "incomplete",
        stage: "setup",
      };
  const failed = Object.entries(steps).find(([, step]) =>
    ["failure", "cancelled"].includes(step.outcome),
  );
  if (failed) {
    if (report.result !== "failed") report.stage = failed[0];
    report.result =
      failed[1].outcome === "failure" ? "failed" : "cancelled/incomplete";
  }
  fs.writeFileSync(file, JSON.stringify(report, null, 2));
  core.setOutput("stage", report.stage);
  return core.summary
    .addRaw(
      `Release ${report.release}: ${report.result}; stage: ${report.stage}`,
    )
    .write();
}

module.exports = {
  resolveRelease,
  smokeEligible,
  summarize,
  finishSmoke,
  reportSmoke,
};
