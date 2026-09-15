// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Release plan parsing is independent of the GitHub Actions runtime.
const SEMVER_CORE_PATTERN = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
// ECMA-compatible SemVer 2.0.0 pattern from https://semver.org/.
const SEMVER_PATTERN = new RegExp(
  "^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)" +
    "(?:-((?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*)" +
    "(?:\\.(?:0|[1-9]\\d*|\\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?" +
    "(?:\\+([0-9a-zA-Z-]+(?:\\.[0-9a-zA-Z-]+)*))?$",
);

function selectArtifacts(value, allowedArtifacts, label, inputName) {
  if (!value.trim()) {
    return [];
  }

  const requestedList = value.split(",").map((id) => id.trim());
  if (requestedList.some((id) => id.length === 0)) {
    throw new Error(`${inputName} contains an empty entry.`);
  }

  const requestedIds = new Set(requestedList);
  if (requestedIds.size !== requestedList.length) {
    throw new Error(`${inputName} contains duplicate entries.`);
  }

  const allowedIds = allowedArtifacts.map((artifact) => artifact.id);
  const unknownIds = [...requestedIds].filter((id) => !allowedIds.includes(id));
  if (unknownIds.length > 0) {
    throw new Error(
      `Unknown ${label} IDs: ${unknownIds.join(", ")}. ` +
        `Allowed ${label} IDs: ${allowedIds.join(", ")}.`,
    );
  }

  return allowedArtifacts.filter((artifact) => requestedIds.has(artifact.id));
}

async function resolveReleasePlan({
  env,
  context,
  listBranches,
  now = () => new Date(),
}) {
  const allWheels = JSON.parse(env.RELEASE_WHEELS_JSON);
  const allContainers = JSON.parse(env.RELEASE_CONTAINERS_JSON);
  const inputs = context.payload.inputs ?? {};
  const isManual = context.eventName === "workflow_dispatch";
  const releaseType = isManual ? inputs["release-type"] : "nightly";
  const releaseScope = isManual ? inputs["release-scope"] || "all" : "all";
  const updateNgcMetadata =
    isManual && inputs["update-ngc-metadata"] === "true";
  const sendNotifications = inputs["send-notifications"] !== "false";
  const dryRun = isManual && inputs["dry-run"] === "true";
  const helmVersionOverride = isManual
    ? (inputs["helm-version"] ?? "").trim()
    : "";
  let sourceSha = isManual ? (inputs["source-sha"] ?? "").trim() : "";
  let sourceBranch = "";
  const version = releaseType === "stable" ? (inputs.version ?? "").trim() : "";

  if (releaseType === "stable") {
    if (!/^[0-9a-f]{40}$/i.test(sourceSha)) {
      throw new Error(
        "Stable releases require an exact 40-character source SHA.",
      );
    }
    if (!SEMVER_CORE_PATTERN.test(version)) {
      throw new Error("Stable releases require a MAJOR.MINOR.PATCH version.");
    }
  } else {
    if (sourceSha && !/^[0-9a-f]{40}$/i.test(sourceSha)) {
      throw new Error(
        "A pinned nightly source must be an exact 40-character SHA.",
      );
    }
    if (!sourceSha) {
      const [branch] = (await listBranches())
        .filter(({ name }) => /^release\/\d+\.\d+$/.test(name))
        .sort((a, b) => {
          const [aMajor, aMinor] = a.name
            .slice("release/".length)
            .split(".")
            .map(Number);
          const [bMajor, bMinor] = b.name
            .slice("release/".length)
            .split(".")
            .map(Number);
          return bMajor - aMajor || bMinor - aMinor;
        });
      if (!branch) {
        throw new Error("No release/X.X branch found for nightly publication.");
      }
      sourceBranch = branch.name;
      sourceSha = branch.commit.sha;
    }
  }

  sourceSha = sourceSha.toLowerCase();
  const presets = {
    all: { wheels: allWheels, containers: allContainers, includeHelm: true },
    wheels: { wheels: allWheels, containers: [], includeHelm: false },
    containers: { wheels: [], containers: allContainers, includeHelm: false },
    helm: { wheels: [], containers: [], includeHelm: true },
  };
  let selection = presets[releaseScope];

  if (releaseScope === "custom") {
    selection = {
      wheels: selectArtifacts(
        inputs["wheel-ids"] || "",
        allWheels,
        "wheel",
        "wheel-ids",
      ),
      containers: selectArtifacts(
        inputs["container-ids"] || "",
        allContainers,
        "container",
        "container-ids",
      ),
      includeHelm: inputs["include-helm"] === "true",
    };
    if (
      selection.wheels.length === 0 &&
      selection.containers.length === 0 &&
      !selection.includeHelm
    ) {
      throw new Error("A custom release must select at least one artifact.");
    }
  }

  if (!selection) {
    throw new Error(`Unknown release scope: ${releaseScope}.`);
  }

  const { wheels, containers, includeHelm } = selection;
  const wheelIds = wheels.map((wheel) => wheel.id);
  const containerIds = containers.map((container) => container.id);
  if (helmVersionOverride) {
    if (
      releaseType !== "stable" ||
      !includeHelm ||
      wheelIds.length > 0 ||
      containerIds.length > 0
    ) {
      throw new Error(
        "helm-version can only be used for stable Helm-only releases.",
      );
    }
    if (!SEMVER_PATTERN.test(helmVersionOverride)) {
      throw new Error("helm-version must be a SemVer chart version.");
    }
  }

  const nightlyTimestamp =
    releaseType === "nightly"
      ? now().toISOString().replace(/\D/g, "").slice(0, 14)
      : "";
  const releaseLabel =
    releaseType === "nightly" ? `nightly-${nightlyTimestamp}` : version;

  return {
    releaseType,
    releaseScope,
    sourceSha,
    sourceBranch,
    version,
    releaseLabel,
    nightlyTimestamp,
    wheels,
    containers,
    wheelIds,
    containerIds,
    includeHelm,
    helmVersionOverride,
    updateNgcMetadata,
    sendNotifications,
    dryRun,
  };
}

module.exports = { resolveReleasePlan };
