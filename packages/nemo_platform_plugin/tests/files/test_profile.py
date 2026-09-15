# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dataset-profiler stored contract.

Three representative datasets — a conversational prompt/completion set, a conversational preference
pair, and a standard scored-response set — are embedded as YAML fixtures and exercised as executable
expectations: if a field name, alias, or vocabulary value drifts, one of these deserializations
breaks.
"""

from datetime import datetime

import pytest
import yaml
from nemo_platform_plugin.files.profile import (
    PROFILE_SCHEMA_VERSION,
    ColumnStats,
    Coverage,
    DatasetProfile,
    Evidence,
    FeatureSchema,
    FileError,
    MessageStats,
    PartitionClassification,
    PartitionProfile,
    Quantiles,
    SplitProfile,
    Verifiability,
)
from pydantic import ValidationError

# --- Fixture: trl-lib/OpenMathReasoning (conversational prompt_completion, verifiable) ---
OPENMATHREASONING = """
profile_schema_version: "1.0"
created_at: 2026-07-08T22:05:12Z
profiler_info: {name: nemo-dataset-profiler, version: 0.1.0}
coverage: {examples_scanned: 2112, examples_present: 3201061,
           files_read: 33, files_present: 33, bytes_present: 31821490182}
partitions:
  - name: ""
    file_formats: [parquet]
    examples_complete: false
    splits:
      - {name: train, canonical: train, num_examples: 3200861, num_files: 32,
         size_bytes: 31819412254, data_files: 'train*.parquet'}
      - {name: test, canonical: test, num_examples: 200, num_files: 1,
         size_bytes: 2077928, data_files: 'test*.parquet'}
    features:
      - {name: prompt, dtype: messages, semantic_role: prompt, semantic_role_source: detected,
         items: {dtype: struct, fields: [{name: role, dtype: string}, {name: content, dtype: string}]}}
      - {name: completion, dtype: messages, semantic_role: completion, semantic_role_source: detected,
         items: {dtype: struct, fields: [{name: role, dtype: string}, {name: content, dtype: string}]}}
    stats:
      prompt:     {messages: {turns: {p50: 1, p95: 1, p99: 1, max: 1}, content_code_points: {p50: 180, p95: 620, p99: 1100, max: 4800},
                              content_utf8_bytes: {p50: 184, p95: 632, p99: 1122, max: 4896},
                              roles_seen: [user], ends_with_assistant_rate: 0.0, valid_alternation_rate: 1.0}}
      completion: {messages: {turns: {p50: 1, p95: 1, p99: 1, max: 1}, content_code_points: {p50: 2400, p95: 7800, p99: 12000, max: 32000},
                              content_utf8_bytes: {p50: 2448, p95: 7956, p99: 12240, max: 32640},
                              roles_seen: [assistant], ends_with_assistant_rate: 1.0, valid_alternation_rate: 1.0}}
    classification:
      modality: text
      candidates: [prompt_completion]
      format: conversational
      prompt_form: explicit
      verifiability:
        method: extractable_final_answer
        coverage: 0.81
        evidence: [{kind: content_probe, detail: 'completion ends with \\boxed{...} in 81% of 2112 sampled rows'}]
      evidence:
        - {kind: column_name,   detail: "prompt + completion column pair"}
        - {kind: content_probe, detail: "prompt ends on a user turn, completion is a single assistant turn"}
"""

# --- Fixture: trl-lib/hh-rlhf-helpful-base (conversational preference_pair, explicit) -----
HH_RLHF_HELPFUL_BASE = """
profile_schema_version: "1.0"
created_at: 2026-07-08T22:41:37Z
profiler_info: {name: nemo-dataset-profiler, version: 0.1.0}
coverage: {examples_scanned: 1024, examples_present: 46189,
           files_read: 2, files_present: 2, bytes_present: 27055195}
partitions:
  - name: ""
    file_formats: [parquet]
    examples_complete: false
    splits:
      - {name: train, canonical: train, num_examples: 43835, num_files: 1,
         size_bytes: 25670988, data_files: 'train*.parquet'}
      - {name: test, canonical: test, num_examples: 2354, num_files: 1,
         size_bytes: 1384207, data_files: 'test*.parquet'}
    features:
      - {name: prompt, dtype: messages, semantic_role: prompt, semantic_role_source: detected,
         items: {dtype: struct, fields: [{name: role, dtype: string}, {name: content, dtype: string}]}}
      - {name: chosen, dtype: messages, semantic_role: chosen, semantic_role_source: detected,
         items: {dtype: struct, fields: [{name: role, dtype: string}, {name: content, dtype: string}]}}
      - {name: rejected, dtype: messages, semantic_role: rejected, semantic_role_source: detected,
         items: {dtype: struct, fields: [{name: role, dtype: string}, {name: content, dtype: string}]}}
    stats:
      prompt:   {messages: {turns: {p50: 3, p95: 8, p99: 9, max: 9}, content_code_points: {p50: 640, p95: 3200, p99: 5400, max: 9800},
                            content_utf8_bytes: {p50: 646, p95: 3232, p99: 5454, max: 9898},
                            roles_seen: [user, assistant], ends_with_assistant_rate: 0.0, valid_alternation_rate: 1.0}}
      chosen:   {messages: {turns: {p50: 1, p95: 1, p99: 1, max: 1}, content_code_points: {p50: 420, p95: 1400, p99: 2100, max: 3600},
                            content_utf8_bytes: {p50: 424, p95: 1414, p99: 2121, max: 3636},
                            roles_seen: [assistant], ends_with_assistant_rate: 1.0, valid_alternation_rate: 1.0}}
      rejected: {messages: {turns: {p50: 1, p95: 1, p99: 1, max: 1}, content_code_points: {p50: 410, p95: 1380, p99: 2050, max: 3500},
                            content_utf8_bytes: {p50: 414, p95: 1394, p99: 2071, max: 3535},
                            roles_seen: [assistant], ends_with_assistant_rate: 1.0, valid_alternation_rate: 1.0}}
    classification:
      modality: text
      candidates: [preference_pair]
      format: conversational
      prompt_form: explicit
      evidence:
        - {kind: column_name,   detail: "chosen + rejected column pair"}
        - {kind: content_probe, detail: "prompt carries the multi-turn history ending on a user turn"}
"""

# --- Fixture: nvidia/HelpSteer2 (standard scored_response, no verifiability) --------------
HELPSTEER2 = """
profile_schema_version: "1.0"
created_at: 2026-07-09T10:12:45Z
profiler_info: {name: nemo-dataset-profiler, version: 0.1.0}
coverage: {examples_scanned: 1024, examples_present: 21362,
           files_read: 2, files_present: 2, bytes_present: 19459677}
partitions:
  - name: ""
    file_formats: [parquet]
    examples_complete: false
    splits:
      - {name: train, canonical: train, num_examples: 20324, num_files: 1,
         size_bytes: 18495985, data_files: 'train*.parquet'}
      - {name: validation, canonical: validation, num_examples: 1038, num_files: 1,
         size_bytes: 963692, data_files: 'validation*.parquet'}
    features:
      - {name: prompt,      dtype: string, semantic_role: prompt, semantic_role_source: detected}
      - {name: response,    dtype: string, semantic_role: completion, semantic_role_source: detected}
      - {name: helpfulness, dtype: int64,  semantic_role: score, semantic_role_source: detected}
      - {name: correctness, dtype: int64,  semantic_role: score, semantic_role_source: detected}
      - {name: coherence,   dtype: int64,  semantic_role: score, semantic_role_source: detected}
      - {name: complexity,  dtype: int64,  semantic_role: score, semantic_role_source: detected}
      - {name: verbosity,   dtype: int64,  semantic_role: score, semantic_role_source: detected}
    stats:
      prompt:   {text: {code_points: {p50: 320, p95: 2200, p99: 5600, max: 12000},
                        utf8_bytes: {p50: 323, p95: 2222, p99: 5656, max: 12120}}}
      response: {text: {code_points: {p50: 1350, p95: 3900, p99: 6200, max: 10500},
                        utf8_bytes: {p50: 1364, p95: 3939, p99: 6262, max: 10605}}}
      helpfulness: {numeric: {min: 0, max: 4, mean: 2.8}, categorical: {distinct_count: 5}}
      correctness: {numeric: {min: 0, max: 4, mean: 2.9}, categorical: {distinct_count: 5}}
      coherence:   {numeric: {min: 0, max: 4, mean: 3.5}, categorical: {distinct_count: 5}}
      complexity:  {numeric: {min: 0, max: 4, mean: 1.6}, categorical: {distinct_count: 5}}
      verbosity:   {numeric: {min: 0, max: 4, mean: 1.5}, categorical: {distinct_count: 5}}
    classification:
      modality: text
      candidates: [scored_response, prompt_completion]
      format: standard
      prompt_form: explicit
      evidence:
        - {kind: column_name,   detail: "prompt + response pair; five rating columns match score aliases"}
        - {kind: content_probe, detail: "all five ratings bounded 0-4 with 5 distinct values"}
        - {kind: card_metadata, detail: "README tag 'human-feedback' corroborates scored human ratings"}
"""

FIXTURES = {
    "OpenMathReasoning": OPENMATHREASONING,
    "hh-rlhf-helpful-base": HH_RLHF_HELPFUL_BASE,
    "HelpSteer2": HELPSTEER2,
}


def _build_profile() -> DatasetProfile:
    """A hand-built profile exercising every model in the contract."""
    return DatasetProfile(
        created_at=datetime(2026, 7, 13, 12, 0, 0),
        profiler_info={"name": "nemo-dataset-profiler", "version": "0.1.0"},
        coverage=Coverage(
            examples_scanned=1024,
            examples_present=2048,
            files_read=2,
            files_present=2,
        ),
        partitions=[
            PartitionProfile(
                file_formats=["parquet"],
                examples_complete=False,
                splits=[
                    SplitProfile(
                        name="train",
                        canonical="train",
                        num_examples=2048,
                        num_files=1,
                    )
                ],
                features=[
                    FeatureSchema(name="prompt", dtype="string", semantic_role="prompt"),
                    FeatureSchema(name="response", dtype="string", semantic_role="completion"),
                ],
                stats={
                    "prompt": ColumnStats(text=None),
                    "response": ColumnStats(numeric=None),
                },
                classification=PartitionClassification(
                    candidates=["prompt_completion"],
                    format="standard",
                    prompt_form="explicit",
                    verifiability=Verifiability(
                        method="extractable_final_answer",
                        coverage=0.9,
                        evidence=[Evidence(kind="content_probe", detail="ends with #### in 90% of rows")],
                    ),
                    evidence=[Evidence(kind="column_name", detail="prompt + response pair")],
                ),
            )
        ],
    )


def test_schema_version_defaults_to_constant():
    profile = _build_profile()
    assert profile.profile_schema_version == PROFILE_SCHEMA_VERSION == "1.0"


def test_round_trip_json_is_lossless():
    profile = _build_profile()
    restored = DatasetProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile


@pytest.mark.parametrize("name", list(FIXTURES))
def test_fixture_deserializes(name):
    """Every fixture loads into the contract and round-trips."""
    profile = DatasetProfile.model_validate(yaml.safe_load(FIXTURES[name]))
    assert profile.profile_schema_version == "1.0"
    # All three ship their shards at the fileset root, so the shared path prefix is empty.
    assert profile.partitions[0].name == ""
    # Round-trip through JSON is lossless.
    assert DatasetProfile.model_validate_json(profile.model_dump_json()) == profile


def test_openmathreasoning_locks_contract_shape():
    profile = DatasetProfile.model_validate(yaml.safe_load(OPENMATHREASONING))
    part = profile.partitions[0]
    assert part.classification.primary == "prompt_completion"
    assert part.classification.format == "conversational"
    # semantic_role is stacked on the feature node; message struct spelled out under items.
    prompt_feature = part.features[0]
    assert prompt_feature.name == "prompt"
    assert prompt_feature.dtype == "messages"
    assert prompt_feature.semantic_role == "prompt"
    assert prompt_feature.items.fields[0].name == "role"
    # Verifiability carries its own coverage + scoped evidence.
    verify = part.classification.verifiability
    assert verify.method == "extractable_final_answer"
    assert verify.coverage == pytest.approx(0.81)
    # Message stats fold into the messages block.
    assert part.stats["prompt"].messages.ends_with_assistant_rate == 0.0
    assert part.stats["completion"].messages.roles_seen == ["assistant"]


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_split_sizes_account_for_the_whole_fileset(name):
    """On a clean profile the splits weigh the whole fileset, so `bytes_present` is the same number
    reached without going through partitions. That redundancy is the point: it is what lets the
    figure survive a file no partition could group."""
    profile = DatasetProfile.model_validate(yaml.safe_load(FIXTURES[name]))
    assert not profile.file_errors
    from_splits = sum(split.size_bytes for part in profile.partitions for split in part.splits)
    assert from_splits == profile.coverage.bytes_present


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_split_globs_are_one_pattern_each_and_never_cross_a_directory(name):
    """`data_files` is a single pattern, not a manifest, so it cannot reintroduce the per-file growth
    the split-level counts exist to avoid. `**` is never emitted, because its meaning is not shared
    across glob implementations."""
    profile = DatasetProfile.model_validate(yaml.safe_load(FIXTURES[name]))
    for part in profile.partitions:
        for split in part.splits:
            assert isinstance(split.data_files, str)
            assert "**" not in split.data_files


def test_a_split_with_no_expressible_pattern_says_so():
    """None is a first-class answer: shards spread across subdirectories need `**` to cover, and a
    pattern that resolves differently in the reader than in the profiler is worse than none."""
    split = SplitProfile(name="train", num_files=2)
    assert split.data_files is None


def test_a_split_weighs_something_even_when_its_row_count_does_not():
    """Size is read off the file listing and a row count off the data, so they go unknown
    independently — `num_examples` is None-able and `size_bytes` is not."""
    split = SplitProfile(name="train", num_files=3, size_bytes=4096)
    assert split.num_examples is None
    assert split.size_bytes == 4096


def test_helpsteer2_flat_schema_and_no_verifiability():
    profile = DatasetProfile.model_validate(yaml.safe_load(HELPSTEER2))
    part = profile.partitions[0]
    # A scored prompt/completion set is also a plain prompt_completion set. The head is the most
    # specific reading; the tail is what the same columns otherwise support.
    assert part.classification.candidates == ["scored_response", "prompt_completion"]
    assert part.classification.primary == "scored_response"
    assert part.classification.format == "standard"
    # Absence of a verifiability object *is* the "not verifiable" claim.
    assert part.classification.verifiability is None
    # Physical column name != role (response -> completion).
    response_feature = next(f for f in part.features if f.name == "response")
    assert response_feature.semantic_role == "completion"
    # Bounded rating scale corroborated by cardinality.
    assert part.stats["helpfulness"].categorical.distinct_count == 5
    assert part.stats["helpfulness"].numeric.max == 4.0
    # card_metadata evidence survives (declared card tags corroborate, never override).
    kinds = {e.kind for e in part.classification.evidence}
    assert "card_metadata" in kinds


def test_semantic_role_reachable_at_any_depth():
    """A role marker nested inside a response list (e.g. a rank) is reachable; a flat column->role
    dict could not address it."""
    answers = FeatureSchema(
        name="answers",
        dtype="list",
        items=FeatureSchema(
            dtype="struct",
            fields=[
                FeatureSchema(name="answer", dtype="string", semantic_role="completion"),
                FeatureSchema(name="model", dtype="string", semantic_role="provenance"),
                FeatureSchema(name="rank", dtype="int64", semantic_role="rank"),
            ],
        ),
    )
    assert answers.items.fields[2].semantic_role == "rank"
    # Round-trips through JSON without losing the nested marker.
    restored = FeatureSchema.model_validate_json(answers.model_dump_json())
    assert restored.items.fields[2].semantic_role == "rank"


def test_vocabularies_are_open():
    """Unknown vocabulary values must be accepted so the vocabulary can grow."""
    classification = PartitionClassification(
        modality="video_text",
        candidates=["some_future_type"],
        format="mixed",
    )
    assert classification.primary == "some_future_type"
    feature = FeatureSchema(dtype="tensor", semantic_role="a_role_added_next_year")
    assert feature.semantic_role == "a_role_added_next_year"


def test_fields_and_items_are_mutually_exclusive():
    """A node cannot be both a named-field container and a single-element container."""
    with pytest.raises(ValidationError, match="mutually exclusive"):
        FeatureSchema(
            name="broken",
            dtype="struct",
            fields=[FeatureSchema(name="a", dtype="string")],
            items=FeatureSchema(dtype="string"),
        )


def test_container_shape_is_not_pinned_to_known_dtypes():
    """The exclusivity check must not become a dtype whitelist: a container dtype added by a newer
    profiler still loads on an older reader, which is what the open vocabulary buys."""
    future_map = FeatureSchema(name="attrs", dtype="map", fields=[FeatureSchema(name="k", dtype="string")])
    assert [field.name for field in future_map.fields or []] == ["k"]
    future_tensor = FeatureSchema(name="embedding", dtype="tensor", items=FeatureSchema(dtype="float32"))
    assert future_tensor.items is not None and future_tensor.items.dtype == "float32"


def test_unknown_fields_are_ignored_for_forward_compat():
    """A profile written by a newer minor version (extra fields) still loads on an older reader."""
    doc = yaml.safe_load(HELPSTEER2)
    doc["some_future_top_level_field"] = {"anything": 1}
    doc["partitions"][0]["classification"]["future_axis"] = "value"
    profile = DatasetProfile.model_validate(doc)
    assert profile.partitions[0].classification.primary == "scored_response"


def test_file_errors_are_the_only_channel_for_trouble():
    # Healthy files are counted, never listed, so a reader asking "did anything go wrong?" reads one
    # list whose length is the number of problems -- not one that grows with the shard count and is
    # 95% success records at scale.
    doc = yaml.safe_load(HELPSTEER2)
    doc["file_errors"] = [
        {"path": "train-00007-of-00032.parquet", "error": "ArrowInvalid: not a parquet file"},
        {"path": "notes.csv", "error": "no reader for '.csv' files"},
    ]
    profile = DatasetProfile.model_validate(doc)

    assert [e.path for e in profile.file_errors] == ["train-00007-of-00032.parquet", "notes.csv"]
    # A shard the profiler could not read and a format it has no reader for are the same finding,
    # and land in the same place whether or not a partition managed to group the file first.
    assert all(isinstance(e, FileError) and e.error for e in profile.file_errors)
    assert DatasetProfile.model_validate_json(profile.model_dump_json()) == profile


def test_a_clean_profile_names_no_files_at_all():
    profile = DatasetProfile.model_validate(yaml.safe_load(HELPSTEER2))
    assert profile.file_errors == []
    assert [s.num_files for s in profile.partitions[0].splits] == [1, 1]


def test_a_profile_written_before_the_digest_was_dropped_still_loads():
    # `content_digest` was removed rather than repaired: it froze "which files count as inputs" into
    # stored data at write time, and that judgment moves. Profiles already written with it have to
    # keep loading, or removing it would break every one of them at once — the very failure mode the
    # removal exists to avoid.
    doc = yaml.safe_load(HELPSTEER2)
    doc["content_digest"] = "sha256:7be1c0ffee"
    profile = DatasetProfile.model_validate(doc)
    assert not hasattr(profile, "content_digest")
    assert profile.partitions[0].classification.primary == "scored_response"


def test_a_profile_written_before_dataset_type_was_dropped_still_loads():
    # `dataset_type` was removed rather than validated: it restated `candidates[0]` as its own stored
    # field, and the one code path that built a classification without running the classifier let the
    # two disagree. A summary computed from the list cannot contradict the list. Profiles already
    # written with the field have to keep loading, which `extra="ignore"` is what gives us.
    doc = yaml.safe_load(HELPSTEER2)
    doc["partitions"][0]["classification"]["dataset_type"] = "scored_response"
    profile = DatasetProfile.model_validate(doc)
    classification = profile.partitions[0].classification
    assert not hasattr(classification, "dataset_type")
    assert classification.primary == "scored_response"


def test_primary_is_derived_and_never_stored():
    # The whole point of the shape: there is no second field to drift, in memory or on the wire.
    classification = PartitionClassification(candidates=["preference_pair", "prompt_completion"])
    assert classification.primary == "preference_pair"
    assert "primary" not in classification.model_dump()
    assert PartitionClassification().primary is None


def test_quantiles_and_message_stats_construct():
    """Smoke-check the leaf stat models are wired as documented."""
    stats = MessageStats(
        turns=Quantiles(p50=1, p95=3, p99=5, max=9),
        content_code_points=Quantiles(p50=100, p95=500, p99=900, max=2000),
        content_utf8_bytes=Quantiles(p50=104, p95=520, p99=940, max=2100),
        roles_seen=["user", "assistant", "tool"],
        ends_with_assistant_rate=1.0,
        valid_alternation_rate=0.98,
        has_tool_calls=True,
    )
    assert stats.turns.max == 9
    assert stats.has_tool_calls is True
