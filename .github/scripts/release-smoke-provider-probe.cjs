// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Temporary validation helper: print model IDs, never credentials or the endpoint.
module.exports = async function probe({ env, fetchImpl }) {
  const base = env.NVIDIA_INFERENCE_URL.replace(/\/+$/, "");
  const response = await fetchImpl(
    `${base}${base.endsWith("/v1") ? "" : "/v1"}/models`,
    {
      headers: { Authorization: `Bearer ${env.NVIDIA_INFERENCE_KEY}` },
      signal: globalThis.AbortSignal.timeout(30000),
    },
  );
  if (!response.ok)
    throw new Error(`Model discovery returned ${response.status}`);
  const body = await response.json();
  const available = body.data.map((model) => model.id);
  const preferred = [
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "mistralai/mistral-large",
    "mistralai/mistral-large-2-instruct",
  ];
  return {
    modelCount: available.length,
    candidates: preferred.filter((model) => available.includes(model)),
    haiku: available.filter((model) => model.includes("haiku")),
  };
};
