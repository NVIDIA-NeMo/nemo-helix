// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A pure helper only — deliberately does NOT call vi.mock(). Vitest's mock hoisting only
// rewrites vi.mock() calls that are lexically present in the file under test; moving it into an
// imported module breaks the hoisting (the mocked hook silently falls back to its real
// implementation, e.g. "No QueryClient set"). Each *.test.tsx must keep its own vi.mock() blocks
// for the 5 SDK hooks — only this response-shape factory is safe to share.

/** Builds a minimal fake return value for one of the 5 mocked list hooks used by the dashboard. */
export const queryResult = (
  totalResults: number | undefined,
  options: { isLoading?: boolean; isError?: boolean } = {}
) =>
  ({
    data: { pagination: { total_results: totalResults } },
    isLoading: options.isLoading ?? false,
    isError: options.isError ?? false,
  }) as never;
