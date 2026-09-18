// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

function eventEnvelope(eventType, clientPayload) {
  return { action: eventType, client_payload: clientPayload };
}

async function dispatchOrReport({ core, github, env, eventType, clientPayload }) {
  const envelope = eventEnvelope(eventType, clientPayload);
  if (env.ACT === "true") {
    core.info(
      `ACT=true; repository_dispatch event:\n${JSON.stringify(envelope, null, 2)}`,
    );
    return envelope;
  }

  const destination = /^([^/\s]+)\/([^/\s]+)$/.exec(env.DISPATCH_REPO ?? "");
  if (!destination) {
    throw new Error("DISPATCH_REPO must be an owner/repository.");
  }
  await github.rest.repos.createDispatchEvent({
    owner: destination[1],
    repo: destination[2],
    event_type: eventType,
    client_payload: clientPayload,
  });
  core.info(`Dispatched ${eventType}: ${JSON.stringify(clientPayload)}`);
  return envelope;
}

module.exports = { dispatchOrReport, eventEnvelope };
