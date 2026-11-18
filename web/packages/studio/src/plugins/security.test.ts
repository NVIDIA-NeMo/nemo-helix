// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isTrustedBundleUrl } from '@studio/plugins/security';

describe('isTrustedBundleUrl', () => {
  it('accepts valid plugin bundle paths', () => {
    expect(isTrustedBundleUrl('/plugin-ui/my-plugin/index.js')).toBe(true);
    expect(isTrustedBundleUrl('/plugin-ui/nemo-agents/index.js')).toBe(true);
  });

  it('accepts content-addressed installed community bundles', () => {
    expect(isTrustedBundleUrl(`/plugin-ui/tavily/${'a'.repeat(64)}/index.js`)).toBe(true);
  });

  it('rejects malformed revisions and alternate asset paths', () => {
    for (const revision of ['a'.repeat(63), 'A'.repeat(64), '../escape', 'a%2fb']) {
      expect(isTrustedBundleUrl(`/plugin-ui/tavily/${revision}/index.js`)).toBe(false);
    }
    expect(isTrustedBundleUrl(`/plugin-ui/tavily/${'a'.repeat(64)}/index.js?url=evil`)).toBe(false);
    expect(isTrustedBundleUrl(`/apis/plugins/tavily/${'a'.repeat(64)}/index.js`)).toBe(false);
  });

  it('rejects single-character plugin names', () => {
    expect(isTrustedBundleUrl('/plugin-ui/a/index.js')).toBe(false);
  });

  it('rejects absolute https URLs', () => {
    expect(isTrustedBundleUrl('https://evil.com/malicious.js')).toBe(false);
  });

  it('rejects absolute http URLs', () => {
    expect(isTrustedBundleUrl('http://evil.com/bundle.js')).toBe(false);
  });

  it('rejects paths not under /plugin-ui/', () => {
    expect(isTrustedBundleUrl('/other/path/index.js')).toBe(false);
    expect(isTrustedBundleUrl('/studio/plugin-ui/x/index.js')).toBe(false);
  });

  it('rejects paths with non-alphanumeric plugin names', () => {
    expect(isTrustedBundleUrl('/plugin-ui/../../../etc/passwd')).toBe(false);
    expect(isTrustedBundleUrl('/plugin-ui/evil%2F../index.js')).toBe(false);
  });

  it('rejects paths that do not end with /index.js', () => {
    expect(isTrustedBundleUrl('/plugin-ui/my-plugin/bundle.js')).toBe(false);
    expect(isTrustedBundleUrl('/plugin-ui/my-plugin/')).toBe(false);
  });

  it('rejects empty string', () => {
    expect(isTrustedBundleUrl('')).toBe(false);
  });
});
