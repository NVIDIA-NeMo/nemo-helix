// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { getClusterInfoUrl, parseClusterInfo } from '@studio/api/clusterInfo';

describe('parseClusterInfo', () => {
  it('reads /cluster-info on the platform base URL', () => {
    expect(getClusterInfoUrl()).toBe('http://localhost:8080/cluster-info');
  });

  it.each([
    ['0.10.0', '0.10.0'],
    ['v0.10.0', '0.10.0'],
    ['0.11.0rc1', '0.11.0rc1'],
    ['0.10.0.post2', '0.10.0.post2'],
    ['dev', undefined],
    ['0.11.0.dev0', undefined],
    ['0.11.0.post3.dev0+5d62bc7400', undefined],
    ['', undefined],
  ])('treats platform_version %j as release %j', (platformVersion, releaseVersion) => {
    expect(parseClusterInfo({ platform_version: platformVersion }).releaseVersion).toBe(
      releaseVersion
    );
  });

  it('passes the architecture through and tolerates servers that omit it', () => {
    expect(parseClusterInfo({ architecture: 'linux/arm64' }).architecture).toBe('linux/arm64');
    expect(parseClusterInfo({ platform_version: '0.10.0' }).architecture).toBeUndefined();
    expect(parseClusterInfo('<html>')).toEqual({});
  });
});
