// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { HuggingfaceStorageConfigRepoType } from '@nemo/sdk/generated/platform/schema';
import { getHuggingFaceRepoUrl } from '@studio/util/huggingFaceUrl';

describe('getHuggingFaceRepoUrl', () => {
  it('prefixes datasets and spaces, and leaves models and unset types at the root', () => {
    expect(
      getHuggingFaceRepoUrl(HuggingfaceStorageConfigRepoType.dataset, 'tatsu-lab/alpaca')
    ).toBe('https://huggingface.co/datasets/tatsu-lab/alpaca');
    expect(getHuggingFaceRepoUrl(HuggingfaceStorageConfigRepoType.space, 'gradio/chatbot')).toBe(
      'https://huggingface.co/spaces/gradio/chatbot'
    );
    expect(
      getHuggingFaceRepoUrl(HuggingfaceStorageConfigRepoType.model, 'meta-llama/Llama-2-7b')
    ).toBe('https://huggingface.co/meta-llama/Llama-2-7b');
    expect(getHuggingFaceRepoUrl(undefined, 'meta-llama/Llama-2-7b')).toBe(
      'https://huggingface.co/meta-llama/Llama-2-7b'
    );

    // Hyphens, dots and mixed case in the repo id are kept verbatim
    expect(
      getHuggingFaceRepoUrl(
        HuggingfaceStorageConfigRepoType.dataset,
        'nvidia/Nemotron-Cascade-2-SFT-Data'
      )
    ).toBe('https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data');
    expect(getHuggingFaceRepoUrl(HuggingfaceStorageConfigRepoType.dataset, 'org/data.v2')).toBe(
      'https://huggingface.co/datasets/org/data.v2'
    );
  });

  it('replaces the origin with a custom endpoint, with or without a trailing slash', () => {
    const { dataset, space, model } = HuggingfaceStorageConfigRepoType;
    ['https://hf.example.com', 'https://hf.example.com/'].forEach((endpoint) => {
      expect(getHuggingFaceRepoUrl(dataset, 'o/d', endpoint)).toBe(
        'https://hf.example.com/datasets/o/d'
      );
      expect(getHuggingFaceRepoUrl(space, 'o/s', endpoint)).toBe(
        'https://hf.example.com/spaces/o/s'
      );
      expect(getHuggingFaceRepoUrl(model, 'o/m', endpoint)).toBe('https://hf.example.com/o/m');
      expect(getHuggingFaceRepoUrl(undefined, 'o/m', endpoint)).toBe('https://hf.example.com/o/m');
    });
  });

  it('falls back to huggingface.co for an empty or missing endpoint', () => {
    const { dataset } = HuggingfaceStorageConfigRepoType;
    expect(getHuggingFaceRepoUrl(dataset, 'o/d', undefined)).toBe(
      'https://huggingface.co/datasets/o/d'
    );
    expect(getHuggingFaceRepoUrl(dataset, 'o/d', '')).toBe('https://huggingface.co/datasets/o/d');
  });
});
