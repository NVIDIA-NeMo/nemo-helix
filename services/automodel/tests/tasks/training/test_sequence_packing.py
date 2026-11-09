# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-length estimation used by automatic sequence packing."""

from collections import UserDict

from nhx.automodel.tasks.training.sequence_packing import _get_sample_token_length

SAMPLE = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]}
TOKEN_IDS = list(range(37))


class _ListTokenizer:
    """Older transformers behaviour: ``tokenize=True`` returns a list of ids."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        return list(TOKEN_IDS)


class _BatchEncodingTokenizer:
    """transformers 5 behaviour: a mapping is returned unless ``return_dict=False``."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, return_dict=True, **kwargs):
        if return_dict:
            return UserDict(input_ids=list(TOKEN_IDS), attention_mask=[1] * len(TOKEN_IDS))
        return list(TOKEN_IDS)


class _AlwaysMappingTokenizer:
    """A tokenizer that ignores ``return_dict`` and always returns a mapping."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        return UserDict(input_ids=list(TOKEN_IDS), attention_mask=[1] * len(TOKEN_IDS))


def test_list_result_counts_token_ids():
    assert _get_sample_token_length(SAMPLE, _ListTokenizer()) == len(TOKEN_IDS)


def test_batch_encoding_result_is_not_counted_by_key_count():
    # Before the fix this returned 2 (the number of keys in the mapping).
    assert _get_sample_token_length(SAMPLE, _BatchEncodingTokenizer()) == len(TOKEN_IDS)


def test_mapping_result_is_unwrapped_to_input_ids():
    assert _get_sample_token_length(SAMPLE, _AlwaysMappingTokenizer()) == len(TOKEN_IDS)
