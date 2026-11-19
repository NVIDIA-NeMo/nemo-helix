// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface TokenPrice {
  inputPerMillion: number;
  outputPerMillion: number;
}

interface ListedPrice extends TokenPrice {
  /** Matches the model name with punctuation folded to dashes, e.g. `glm-5-3-flash`. */
  pattern: RegExp;
}

/** List prices in USD per million tokens. A model that is not listed has no cost estimate. */
const LISTED_PRICES: ListedPrice[] = [
  // Z.ai first-party API.
  { pattern: /glm-(5-3-flash|flash-5-3)/, inputPerMillion: 0.15, outputPerMillion: 0.5 },
  // build.nvidia.com. Only the Lightning 30B is listed.
  { pattern: /nemotron-3-5-lightning/, inputPerMillion: 0.05, outputPerMillion: 0.2 },
];

const fold = (model: string): string => model.toLowerCase().replace(/[^a-z0-9]+/g, '-');

export const listedPrice = (model: string | undefined): TokenPrice | undefined =>
  model ? LISTED_PRICES.find(({ pattern }) => pattern.test(fold(model))) : undefined;

export const estimateCostPer1kTasks = (
  inputTokensPerTask: number,
  outputTokensPerTask: number,
  price: TokenPrice
): number =>
  (1000 * (inputTokensPerTask * price.inputPerMillion + outputTokensPerTask * price.outputPerMillion)) /
  1_000_000;
