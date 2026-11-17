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
});
