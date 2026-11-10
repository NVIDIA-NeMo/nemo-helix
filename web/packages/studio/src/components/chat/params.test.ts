// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { InferenceParams } from '@nemo/sdk/generated/platform/schema';
import {
  CHAT_INFERENCE_PARAM_METADATA,
  chatParamDefaults,
  chatParamMetadataWithMaxTokens,
  mergeChatParams,
} from '@studio/components/chat/params';

describe('mergeChatParams', () => {
  it('applies the values that are set', () => {
    expect(mergeChatParams({ temperature: 0.7, max_tokens: 512 }, { temperature: 0 })).toEqual({
      temperature: 0,
      max_tokens: 512,
    });
  });

  it('keeps the previous value for a param the edit leaves unset', () => {
    expect(
      mergeChatParams({ temperature: 0.7, max_tokens: 512 }, { temperature: undefined })
    ).toEqual({ temperature: 0.7, max_tokens: 512 });
  });

  it('keeps the previous value over a null', () => {
    expect(
      mergeChatParams({ temperature: 0.7, max_tokens: 512 }, {
        temperature: null,
      } as unknown as Partial<InferenceParams>)
    ).toEqual({ temperature: 0.7, max_tokens: 512 });
  });
});

describe('chatParamMetadataWithMaxTokens', () => {
  it("swaps only the max-tokens default, keeping chat's bounds", () => {
    expect(chatParamMetadataWithMaxTokens(4096)).toEqual({
      ...CHAT_INFERENCE_PARAM_METADATA,
      max_tokens: { ...CHAT_INFERENCE_PARAM_METADATA.max_tokens, default: 4096 },
    });
  });
});

describe('chatParamDefaults', () => {
  it("starts from each param's default", () => {
    expect(chatParamDefaults(chatParamMetadataWithMaxTokens(4096))).toEqual({
      temperature: CHAT_INFERENCE_PARAM_METADATA.temperature.default,
      max_tokens: 4096,
    });
  });
});
