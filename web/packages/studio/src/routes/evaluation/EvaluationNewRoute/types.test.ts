// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  EMPTY_FIELD_MAPPING,
  isSupportedMappingPath,
  toFieldMapping,
} from '@studio/routes/evaluation/EvaluationNewRoute/types';

/** ``PATH_CASES`` in ``nhx_evals_sdk/tests/dataset_schemas/test_common.py``. */
const ACCEPTED = [
  'question',
  'a.b.c',
  'messages[1].content',
  'messages[0]',
  'turns[0][2].text',
  'messages[role=assistant].content',
  'messages[role=assistant]',
  'a[1][role=x].b',
  'messages[role=a].tool_calls[type=function].id',
];

const REJECTED = [
  'messages[].content',
  'messages[role=assistant][1].text',
  'messages[role=a][role=b].c',
  'messages[role=].content',
  'messages[=assistant].content',
  'messages[role=a=b].c',
  'messages[role = assistant].c',
  'messages[a.b=c].d',
  'messages[role="a"].c',
  'messages[-1].content',
  'messages[x].content',
  'messages[1]content',
  'messages[',
];

describe('isSupportedMappingPath', () => {
  it.each(ACCEPTED)('accepts %s', (path) => {
    expect(isSupportedMappingPath(path)).toBe(true);
  });

  it.each(REJECTED)('rejects %s', (path) => {
    expect(isSupportedMappingPath(path)).toBe(false);
  });
});

describe('toFieldMapping', () => {
  it('forwards role paths for a messages dataset', () => {
    expect(
      toFieldMapping({
        ...EMPTY_FIELD_MAPPING,
        messages: 'conversation',
        input: 'conversation[role=user].content',
        reference: 'conversation[role=assistant].content',
      })
    ).toEqual({
      messages: 'conversation',
      input: 'conversation[role=user].content',
      reference: 'conversation[role=assistant].content',
    });
  });

  it('drops a path the evaluator would refuse', () => {
    expect(toFieldMapping({ ...EMPTY_FIELD_MAPPING, input: 'messages[].content' })).toBeUndefined();
  });
});
