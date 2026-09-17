// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { canApplyMitigation } from '@agent-hardener/components/HardenPanel';

const params = (over: Partial<Parameters<typeof canApplyMitigation>[0]> = {}) => ({
  hasComposedGuardrails: true,
  hasManifestId: true,
  manifestQuerySucceeded: true,
  isProjectSource: false,
  ...over,
});

describe('canApplyMitigation', () => {
  it('allows applying once guardrails are composed and the manifest is confirmed non-project', () => {
    expect(canApplyMitigation(params())).toBe(true);
  });

  it('disables until the manifest query settles, since isProjectSource defaults to false', () => {
    // Covers both loading and error: neither confirms the source type, and enabling on the
    // default would let a project manifest be applied to a same-named registered agent.
    expect(canApplyMitigation(params({ manifestQuerySucceeded: false }))).toBe(false);
  });

  it('disables for a confirmed project-source manifest, which has no agent to adopt onto', () => {
    expect(canApplyMitigation(params({ isProjectSource: true }))).toBe(false);
  });

  it('allows a run with no manifest behind it, matching what the API permits', () => {
    // `run --config` produces a run with no manifest_id; the API's _reject_project_source returns
    // early for those. Gating on a query that never fires would disable Apply forever.
    expect(canApplyMitigation(params({ hasManifestId: false, manifestQuerySucceeded: false }))).toBe(
      true
    );
  });

  it('disables when no guardrails have been composed yet', () => {
    expect(canApplyMitigation(params({ hasComposedGuardrails: false }))).toBe(false);
  });
});
