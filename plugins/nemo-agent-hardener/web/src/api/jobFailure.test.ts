// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { jobFailure } from '@agent-hardener/api/jobFailure';

it('reads the message and remediation the backend wrote', () => {
  const job = { error_details: { message: ' victim answered HTTP 500 ', remediation: 'Check the victim logs.' } };
  expect(jobFailure(job)).toEqual({ message: 'victim answered HTTP 500', remediation: 'Check the victim logs.' });
});

it('leaves remediation out when the backend gave none', () => {
  expect(jobFailure({ error_details: { message: 'boom', remediation: '' } })).toEqual({
    message: 'boom',
    remediation: undefined,
  });
});

it('is undefined for a job with no usable message', () => {
  expect(jobFailure(undefined)).toBeUndefined();
  expect(jobFailure({ error_details: null })).toBeUndefined();
  expect(jobFailure({ error_details: { message: 42 } })).toBeUndefined();
});
