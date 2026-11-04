// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_WORKSPACE } from '@nemo/common/src/models/constants';
import {
  getEntityReference,
  getURNFromNamedEntityRef,
  type NamedEntity,
  type NamedEntityRef,
} from '@nemo/common/src/namedEntity';

describe('getEntityFullName', () => {
  it('should return empty string for undefined input', () => {
    expect(getEntityReference(undefined)).toBe('');
  });

  it('should return empty string for null input', () => {
    expect(getEntityReference(null as unknown as NamedEntityRef)).toBe('');
  });

  it('should return the string as-is when input is a string', () => {
    const input = 'test/entity';
    expect(getEntityReference(input)).toBe(input);
  });

  it('should encode string input when encode option is true', () => {
    const input = 'test/entity with spaces';
    expect(getEntityReference(input, { encode: true })).toBe(encodeURIComponent(input));
  });

  it('should use default namespace when namespace is not provided', () => {
    const input: NamedEntity = { name: 'test' };
    expect(getEntityReference(input)).toBe(`${DEFAULT_WORKSPACE}/test`);
  });

  it('should combine namespace and name with forward slash', () => {
    const input: NamedEntity = { workspace: 'custom', name: 'test' };
    expect(getEntityReference(input)).toBe('custom/test');
  });

  it('should encode object input when encode option is true', () => {
    const input: NamedEntity = { workspace: 'custom space', name: 'test name' };
    expect(getEntityReference(input, { encode: true })).toBe(
      encodeURIComponent('custom space/test name')
    );
  });

  it('should handle empty name in object input', () => {
    const input: NamedEntity = { workspace: 'custom' };
    expect(getEntityReference(input)).toBe('custom/');
  });

  it('should handle empty namespace in object input', () => {
    const input: NamedEntity = { name: 'test' };
    expect(getEntityReference(input)).toBe(`${DEFAULT_WORKSPACE}/test`);
  });
});

describe('getURNFromNamedEntityRef', () => {
  it('returns a valid workspace/name resource ref', () => {
    expect(getURNFromNamedEntityRef({ workspace: 'default', name: 'test-model' })).toBe(
      'default/test-model'
    );
  });

  it('returns undefined for references that are not workspace/name refs', () => {
    expect(getURNFromNamedEntityRef('urn:model:123')).toBeUndefined();
  });
});
