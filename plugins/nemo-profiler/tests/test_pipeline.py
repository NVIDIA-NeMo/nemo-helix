# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the profiling pipeline: split/partition resolution and envelope assembly."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nemo_platform_plugin.files.profile import DatasetProfile
from nemo_profiler_plugin.dataset.partition import group_partitions
from nemo_profiler_plugin.dataset.pipeline import MIN_ROWS_PER_FILE, _peek_files, _per_file_cap, profile
from nemo_profiler_plugin.dataset.readers.base import FilePreview
from nemo_profiler_plugin.dataset.splits import infer_data_files, resolve_splits
from nemo_profiler_plugin.source import FileEntry, LocalFileSource

FIXED_TIME = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)


def _write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _entries(*paths):
    return [FileEntry(path=p, size_bytes=100) for p in paths]


# --- split resolution ----------------------------------------------------------------------------


def test_resolve_splits_infers_canonical_from_sharded_names():
    entries = _entries(
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
        "validation-00000-of-00001.parquet",
    )
    splits = {s.name: s for s in resolve_splits(entries)}
    assert set(splits) == {"train", "validation"}
    assert splits["train"].canonical == "train"
    assert splits["validation"].canonical == "validation"
    assert len(splits["train"].entries) == 2


def test_resolve_splits_normalizes_aliases():
    splits = {s.name: s.canonical for s in resolve_splits(_entries("val.jsonl", "dev.jsonl"))}
    assert splits == {"val": "validation", "dev": "validation"}


def test_resolve_splits_does_not_mistake_years_for_shard_numbers():
    # A bare trailing number only reads as a shard when it is zero-padded; otherwise dates and
    # versions were being stripped, e.g. covid-19.jsonl -> a "covid" split.
    assert [s.name for s in resolve_splits(_entries("covid-19.jsonl"))] == ["default"]
    names = {s.name for s in resolve_splits(_entries("train-00000-of-00002.parquet", "data-2024.jsonl"))}
    assert names == {"train", "data-2024"}


def test_resolve_splits_falls_back_to_single_default():
    splits = resolve_splits(_entries("shard-00000.parquet", "shard-00001.parquet"))
    assert len(splits) == 1
    assert splits[0].name == "default"
    assert splits[0].canonical is None
    assert len(splits[0].entries) == 2


# --- data_files glob inference --------------------------------------------------------------------


def _globs(*paths):
    """Infer a glob per split over ``paths``, verifying against the full listing (README included)."""
    data = [e for e in _entries(*paths) if e.path.endswith((".parquet", ".jsonl"))]
    return {s.name: infer_data_files(s.name, s.entries, list(paths)) for s in resolve_splits(data)}


@pytest.mark.parametrize(
    "label,paths,expected",
    [
        (
            "shards in one directory",
            (
                "data/train-00000-of-00002.parquet",
                "data/train-00001-of-00002.parquet",
                "data/test-00000-of-00001.parquet",
            ),
            {"train": "data/train*.parquet", "test": "data/test*.parquet"},
        ),
        (
            "a directory per split",
            ("default/train/0000.parquet", "default/train/0001.parquet", "default/test/0000.parquet"),
            {"train": "default/train/*.parquet", "test": "default/test/*.parquet"},
        ),
        (
            "files at the fileset root",
            ("train.jsonl", "validation.jsonl"),
            {"train": "train*.jsonl", "validation": "validation*.jsonl"},
        ),
        (
            "no split detected: the glob covers the partition",
            ("shard-00000.parquet", "shard-00001.parquet"),
            {"default": "*.parquet"},
        ),
        (
            "mixed formats drop the suffix rather than losing a file",
            ("train-00000-of-00002.parquet", "train-00001-of-00002.jsonl"),
            {"train": "train*"},
        ),
    ],
)
def test_data_files_glob_per_layout(label, paths, expected):
    assert _globs(*paths) == expected, label


def test_data_files_glob_excludes_a_readme_beside_the_shards():
    # `data/*` would sweep the card into the split. The suffix-qualified candidate is what survives
    # verification, and verification runs against every listed file, not just the data ones.
    assert _globs("data/train-00000-of-00001.parquet", "data/README.md") == {"train": "data/train*.parquet"}


def test_data_files_glob_keeps_a_separator_to_beat_a_sibling_split():
    # `train*` would also match train_prefs, so the simple form loses verification and the narrower
    # `train-*` is reached. Both splits still get a pattern; neither over-matches the other.
    assert _globs(
        "train-00000-of-00002.parquet", "train-00001-of-00002.parquet", "train_prefs-00000-of-00001.parquet"
    ) == {"train": "train-*.parquet", "train_prefs": "train_prefs*.parquet"}


def test_data_files_glob_refuses_rather_than_sweep_in_a_non_data_file():
    # Mixed suffixes leave no suffix to qualify with, and an unsplit-named set leaves no stem to
    # anchor on, so the only candidate left is `data/*` -- which would hand a reader the README as
    # if it were a shard. Verification is the whole of what stops that, and None is the answer.
    assert _globs("data/shard-00000.parquet", "data/shard-00001.jsonl", "data/README.md") == {"default": None}


def test_data_files_glob_is_none_when_shards_span_subdirectories():
    # Expressing this needs `**`, whose meaning differs between glob implementations. None is the
    # honest answer; a pattern that selects a different set in the reader than here would not be.
    assert _globs("train/part-a/0000.parquet", "train/part-b/0000.parquet") == {"train": None}


def test_data_files_glob_means_the_same_thing_to_pythons_own_glob(tmp_path):
    """The dialect claim, checked against an independent implementation rather than our matcher.

    A pattern is only worth emitting if a consumer resolves it to the files we said it selects.
    """
    for rel in (
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet",
        "data/test-00000-of-00001.parquet",
    ):
        _write_parquet(tmp_path / rel, [{"a": 1}])
    (tmp_path / "data" / "README.md").write_text("card")

    for split in profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0].splits:
        resolved = sorted(p.relative_to(tmp_path).as_posix() for p in Path(tmp_path).glob(split.data_files))
        assert len(resolved) == split.num_files, f"{split.name}: {split.data_files} -> {resolved}"
        assert all(name.startswith(f"data/{split.name}") for name in resolved)


def test_a_path_holding_a_glob_metacharacter_gets_no_pattern(tmp_path):
    # `_glob_matches` escapes everything but `*`, so verification read `[` and `?` as literals --
    # while the pattern it blessed goes to a consumer whose glob reads them as syntax. The two
    # directions of the same bug: `data[a]/train*.jsonl` selects nothing at all, and
    # `what?/train*.jsonl` also selects `whatX/`, which is the near miss the contract promises
    # cannot happen. Escaping is no answer, since the escape syntax is what the dialects do not
    # share, so the honest pattern is no pattern.
    for directory in ("data[a]", "what?"):
        (tmp_path / directory).mkdir()
        _write_parquet(tmp_path / directory / "train.parquet", [{"a": 1}])
        (tmp_path / directory.replace("[a]", "Xa").replace("?", "X")).mkdir()

    for partition in profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions:
        for split in partition.splits:
            assert split.data_files is None, f"{partition.name}: {split.data_files}"


def test_ordinary_shard_numbering_is_one_split(tmp_path):
    # The padded form had to be four characters, so `train-001.parquet` was not a shard suffix and
    # every shard became a split of its own -- each with its own name, its own num_examples and its
    # own glob. The leading zero is what separates a shard from a year, so three characters is
    # enough, and `N-of-M` needs no padding rule at all since nothing else is spelled that way.
    for shard in ("001", "002", "003"):
        _write_parquet(tmp_path / f"train-{shard}.parquet", [{"a": 1}])
    _write_parquet(tmp_path / "test-1-of-1.parquet", [{"a": 1}])

    splits = {s.name: s for s in profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0].splits}

    assert sorted(splits) == ["test", "train"]
    assert splits["train"].num_files == 3
    assert splits["test"].num_files == 1


def test_a_year_in_a_filename_is_still_not_a_shard_number(tmp_path):
    # The other side of the same rule, and the reason it is not simply `-\d+$`: widening it to
    # reach `train-001` must not reach `covid-19` or `data-2024`, neither of which is padded.
    _write_parquet(tmp_path / "covid-19.parquet", [{"a": 1}])
    _write_parquet(tmp_path / "data-2024.parquet", [{"a": 1}])

    splits = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0].splits

    # Neither name is a recognised split, so both land in the single "default" split intact.
    assert [s.name for s in splits] == ["default"]
    assert splits[0].num_files == 2


def test_a_dot_prefixed_file_does_not_make_a_split_named_nothing(tmp_path):
    # `.split(".")[0]` reads everything before the first dot, which for `._train.jsonl` -- the
    # AppleDouble files a macOS-made tarball is full of -- is the empty string. `SplitProfile.name`
    # is documented as the on-disk name, and "" is not one.
    (tmp_path / "train.jsonl").write_text('{"a": 1}\n')
    (tmp_path / "._train.jsonl").write_text('{"a": 1}\n')

    splits = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0].splits

    assert all(s.name for s in splits), [s.name for s in splits]


# --- the fold ------------------------------------------------------------------------------------


def test_a_parquet_footer_declares_enough_to_fold_without_reading_rows(tmp_path):
    # The footer is what makes a fold possible at all: the schema, so accumulators can exist before
    # the first batch, and the exact row count, so a split can report an exact `num_examples` from a
    # run that never read to the end. A line-delimited file declares neither.
    _write_parquet(tmp_path / "train.parquet", [{"a": i} for i in range(7)])
    (tmp_path / "extra.jsonl").write_text('{"a": 1}\n')
    source = LocalFileSource(tmp_path)

    previews, unpeekable = _peek_files(source, source.list_files())

    assert previews["train.parquet"].num_rows == 7
    assert previews["train.parquet"].arrow_schema is not None
    assert previews["extra.jsonl"] == FilePreview()  # declares nothing, so the partition cannot fold
    assert not unpeekable  # both were asked and answered; neither failed


def test_the_folded_and_materialised_paths_measure_the_same_thing(tmp_path):
    # Parquet declares a schema and is folded batch by batch; jsonl declares none and is
    # materialised. The same rows have to measure the same either way, or the batch size -- an
    # implementation detail no reader of a profile can see -- would be visible in the numbers.
    rows = [{"prompt": f"question {i}", "completion": "answer " * (i % 7 + 1), "score": i % 5} for i in range(200)]
    _write_parquet(tmp_path / "pq" / "train.parquet", rows)
    (tmp_path / "jl").mkdir()
    (tmp_path / "jl" / "train.jsonl").write_text("\n".join(json.dumps(row) for row in rows))

    folded = profile(LocalFileSource(tmp_path / "pq"), created_at=FIXED_TIME).partitions[0]
    materialised = profile(LocalFileSource(tmp_path / "jl"), created_at=FIXED_TIME).partitions[0]

    assert folded.stats == materialised.stats
    assert folded.classification == materialised.classification
    assert [f.model_dump() for f in folded.features] == [f.model_dump() for f in materialised.features]


def test_an_exhaustive_fold_does_not_cost_more_than_a_budgeted_one(tmp_path):
    # The point of the whole exercise: reading every row costs what reading some of them costs, so
    # the budget stops being a memory guard. Same measurements, and `examples_complete` finally true.
    _write_parquet(tmp_path / "train.parquet", [{"t": f"row {i}" * (i % 5 + 1)} for i in range(5000)])

    budgeted = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=500)
    exhaustive = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=None)

    assert budgeted.coverage.examples_scanned == 500
    assert exhaustive.coverage.examples_scanned == 5000
    assert budgeted.partitions[0].examples_complete is False
    assert exhaustive.partitions[0].examples_complete is True
    # Exact where it claims to be exact: the longest row is found by reading all of them.
    assert (
        exhaustive.partitions[0].stats["t"].text.code_points.max
        >= budgeted.partitions[0].stats["t"].text.code_points.max
    )


# --- partition grouping --------------------------------------------------------------------------


def test_group_partitions_names_the_root_partition_with_the_empty_prefix():
    # "" is the path prefix root-level files share, and no directory can be named it -- which is what
    # keeps root files distinct from a directory literally called "default".
    assert group_partitions(_entries("train.parquet", "test.parquet")) == [
        ("", _entries("train.parquet", "test.parquet"))
    ]


def test_group_partitions_collapses_single_container_dir():
    # One container directory is still one partition, and it keeps that directory as its name.
    # Reporting "default" here discarded the only thing identifying the partition.
    parts = group_partitions(_entries("data/train.parquet", "data/test.parquet"))
    assert [name for name, _ in parts] == ["data"]


def test_group_partitions_splits_multiple_top_dirs():
    parts = group_partitions(_entries("main/train.parquet", "socratic/train.parquet"))
    assert [name for name, _ in parts] == ["main", "socratic"]


def test_group_partitions_does_not_treat_split_dirs_as_partitions():
    # train/ and test/ are one dataset's splits, not two datasets.
    parts = group_partitions(_entries("train/data.parquet", "test/data.parquet"))
    assert [name for name, _ in parts] == [""]


def test_resolve_splits_reads_the_split_directory():
    # The data/<split>/<shard> layout names every shard the same thing; only the directory carries
    # the split, so reading the stem alone would collapse the dataset into one split.
    splits = {s.name: s for s in resolve_splits(_entries("data/train/0000.parquet", "data/test/0000.parquet"))}
    assert set(splits) == {"train", "test"}
    assert splits["train"].canonical == "train"
    assert splits["test"].canonical == "test"


def test_resolve_splits_prefers_directory_over_stem():
    splits = resolve_splits(_entries("main/train-00000-of-00001.parquet"))
    assert [s.name for s in splits] == ["train"]  # no split dir on the path, so the stem is used


# --- end-to-end profile() ------------------------------------------------------------------------


def test_profile_parquet_dataset_builds_envelope(tmp_path):
    _write_parquet(tmp_path / "train-00000-of-00001.parquet", [{"prompt": "a"}, {"prompt": "b"}])
    _write_parquet(tmp_path / "validation-00000-of-00001.parquet", [{"prompt": "c"}])

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.profiler_info["name"] == "nemo-dataset-profiler"
    assert len(result.partitions) == 1
    partition = result.partitions[0]
    assert partition.name == ""  # root-level files: the empty path prefix
    assert partition.file_formats == ["parquet"]

    splits = {s.name: s for s in partition.splits}
    assert set(splits) == {"train", "validation"}
    assert splits["train"].canonical == "train"
    assert splits["train"].num_examples == 2
    assert splits["validation"].num_examples == 1
    assert splits["train"].num_files == 1

    # Row schema, stats, and classification are all derived now.
    assert [f.name for f in partition.features] == ["prompt"]
    assert partition.features[0].dtype == "string"
    assert partition.features[0].semantic_role == "prompt"
    assert partition.stats["prompt"].text is not None
    assert partition.classification.primary == "prompt_only"  # a lone prompt column, no target

    # A budgeted run over files that all fit under their share is still a complete scan, which is
    # why the budget and the outcome are separate fields.
    assert partition.examples_complete is True
    assert result.coverage.examples_scanned == result.coverage.examples_present  # exhaustive by default
    assert result.coverage.examples_scanned == 3
    assert result.coverage.examples_present == 3
    assert result.coverage.files_read == result.coverage.files_present == 2


def test_profile_jsonl_dataset_counts_rows_exactly(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    partition = result.partitions[0]
    assert partition.file_formats == ["jsonl"]
    assert partition.splits[0].name == "train"
    assert partition.splits[0].num_examples == 3
    assert result.coverage.examples_scanned == 3


def test_profile_multiple_directories_become_partitions(tmp_path):
    _write_parquet(tmp_path / "main" / "train-00000-of-00001.parquet", [{"q": "1"}])
    _write_parquet(tmp_path / "socratic" / "train-00000-of-00001.parquet", [{"q": "2"}])

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert [p.name for p in result.partitions] == ["main", "socratic"]
    assert all(p.file_formats == ["parquet"] for p in result.partitions)


def test_profile_top_level_split_dirs_become_one_partition(tmp_path):
    # train/ + test/ is one dataset with two splits, not two datasets. As separate partitions each
    # would derive its own schema and classification, and the split structure would disappear.
    _write_parquet(tmp_path / "train" / "data.parquet", [{"prompt": "a", "completion": "b"}])
    _write_parquet(tmp_path / "test" / "data.parquet", [{"prompt": "c", "completion": "d"}])

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert [p.name for p in result.partitions] == [""]
    splits = {s.name: s for s in result.partitions[0].splits}
    assert set(splits) == {"train", "test"}
    assert splits["train"].canonical == "train"
    assert splits["test"].num_examples == 1


def test_profile_nested_split_dirs_keep_splits_apart(tmp_path):
    # data/<split>/<shard> shards are all named alike, so only the directory distinguishes them.
    # Reading the stem alone pooled train and test into a single "default" split.
    _write_parquet(tmp_path / "data" / "train" / "0000.parquet", [{"prompt": "a"}, {"prompt": "b"}])
    _write_parquet(tmp_path / "data" / "test" / "0000.parquet", [{"prompt": "c"}])

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert [p.name for p in result.partitions] == ["data"]  # the container directory, not "default"
    splits = {s.name: s for s in result.partitions[0].splits}
    assert set(splits) == {"train", "test"}
    assert splits["train"].num_examples == 2
    assert splits["test"].num_examples == 1


def test_profile_keeps_a_mixed_format_directory_as_one_partition(tmp_path):
    # A stray .jsonl beside .parquet shards is noise, not a second dataset. Splitting the partition
    # to keep a scalar `file_format` true invented structure that is not in the data *and* renamed
    # the real partition (default -> default:parquet). Format is a per-file fact instead.
    _write_parquet(tmp_path / "data" / "train-00000-of-00001.parquet", [{"prompt": "a"}])
    (tmp_path / "data" / "extra.jsonl").write_text('{"question": "b"}\n{"question": "c"}\n')

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert len(result.partitions) == 1
    partition = result.partitions[0]
    assert partition.name == "data"
    assert partition.file_formats == ["jsonl", "parquet"]
    # Both formats' columns reach features. Trusting the declared parquet schema would have erased
    # `question`, which only the schemaless file witnesses -- the defect the split worked around.
    assert sorted(f.name for f in partition.features) == ["prompt", "question"]
    assert result.coverage.examples_scanned == 3  # 1 parquet + 2 jsonl, each counted once
    assert partition.examples_complete is True


def test_root_files_and_a_directory_named_default_stay_distinct():
    # The collision the empty-string sentinel exists to prevent: a derived label collapsed both to
    # "default", leaving two partitions with one name and no way to reference either.
    parts = group_partitions(_entries("root.parquet", "default/inner.parquet"))
    assert [name for name, _ in parts] == ["", "default"]


def test_an_unrelated_file_does_not_rename_a_partition(tmp_path):
    # Dropping a stray .jsonl into main/ used to turn partition "main" into "main:parquet" -- not
    # renamed, *gone*, so a stored reference resolved to nothing.
    _write_parquet(tmp_path / "main" / "train.parquet", [{"q": "a"}])
    _write_parquet(tmp_path / "socratic" / "train.parquet", [{"q": "b"}])
    before = [p.name for p in profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions]

    (tmp_path / "main" / "notes.jsonl").write_text('{"note": "someone dropped this here"}\n')
    after = [p.name for p in profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions]

    assert before == after == ["main", "socratic"]


def test_profile_unions_columns_across_shards(tmp_path):
    # A column that appears only in a later shard must still reach features/stats. Taking the first
    # shard's schema would drop it entirely.
    _write_parquet(tmp_path / "train-00000-of-00002.parquet", [{"prompt": "a", "completion": "b"}])
    _write_parquet(tmp_path / "train-00001-of-00002.parquet", [{"prompt": "c", "completion": "d", "score": 3}])

    partition = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert [f.name for f in partition.features] == ["prompt", "completion", "score"]
    assert partition.stats["score"].null_rate == 0.5  # absent in the first shard, and said so


def test_profile_is_invariant_to_shard_order(tmp_path, tmp_path_factory):
    # The same rows must profile the same way regardless of which shard sorts first. First-wins
    # schema selection made this data classify as prompt_completion or scored_response depending
    # purely on filename order.
    narrow = [{"prompt": "a", "completion": "b"}]
    wide = [{"prompt": "c", "completion": "d", "score": 3}]

    forward = tmp_path_factory.mktemp("forward")
    _write_parquet(forward / "train-00000-of-00002.parquet", narrow)
    _write_parquet(forward / "train-00001-of-00002.parquet", wide)

    reverse = tmp_path_factory.mktemp("reverse")
    _write_parquet(reverse / "train-00000-of-00002.parquet", wide)
    _write_parquet(reverse / "train-00001-of-00002.parquet", narrow)

    first = profile(LocalFileSource(forward), created_at=FIXED_TIME).partitions[0]
    second = profile(LocalFileSource(reverse), created_at=FIXED_TIME).partitions[0]

    assert [(f.name, f.dtype) for f in first.features] == [(f.name, f.dtype) for f in second.features]
    assert first.classification.primary == second.classification.primary == "scored_response"


def test_profile_survives_conflicting_shard_schemas(tmp_path):
    # Two shards disagreeing on a column's type has no right answer at the schema level; fall back to
    # inferring from the rows (which widens to json) rather than asserting one shard over the other.
    _write_parquet(tmp_path / "train-00000-of-00002.parquet", [{"score": 1}])
    _write_parquet(tmp_path / "train-00001-of-00002.parquet", [{"score": "high"}])

    partition = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert [f.name for f in partition.features] == ["score"]
    assert partition.features[0].dtype == "json"  # mixed, and honest about it


def test_an_unreadable_file_says_what_it_actually_is(tmp_path):
    # The payoff for the magic-byte checks, and the reason they cannot live in `peek` alone: `peek`
    # runs first and its failures are discarded, so this message only reaches a FileError because the
    # read path raises it too. Without it both shards report a decode error from inside the parser,
    # which reads like corrupt data rather than a file that was never this format.
    _write_parquet(tmp_path / "train-00000-of-00001.parquet", [{"a": 1}])
    (tmp_path / "test-00000-of-00001.parquet").write_bytes(b"<!DOCTYPE html><html>404</html>")
    (tmp_path / "extra.jsonl").write_bytes(b"\x1f\x8b\x08\x00" + b"\x00" * 32)

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    errors = {e.path: e.error for e in result.file_errors}
    assert "not a parquet file" in errors["test-00000-of-00001.parquet"]
    assert "gzip archive" in errors["extra.jsonl"]
    assert result.partitions[0].stats  # and the shard that was fine is still measured


def test_profile_isolates_unreadable_files(tmp_path):
    _write_parquet(tmp_path / "train-00000-of-00001.parquet", [{"a": 1}])
    (tmp_path / "test-00000-of-00001.parquet").write_bytes(b"not a real parquet file")

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    splits = {s.name: s for s in result.partitions[0].splits}
    assert splits["train"].num_examples == 1
    assert splits["test"].num_examples is None  # unreadable -> count unknown, not a crash
    assert [e.path for e in result.file_errors] == ["test-00000-of-00001.parquet"]  # named, with a reason
    assert result.file_errors[0].error
    assert result.partitions[0].examples_complete is False  # a file could not be fully parsed
    assert result.coverage.examples_present is None
    assert result.coverage.files_read == 1  # one file was actually read; the other never opened
    assert result.coverage.files_present == 2  # ...out of two that were there to read


def test_profile_row_budget_bounds_reads_and_says_so(tmp_path):
    _write_parquet(tmp_path / "train-00000-of-00001.parquet", [{"a": i} for i in range(10)])

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=4)

    assert result.coverage.examples_scanned == 4
    assert result.partitions[0].examples_complete is False  # 4 of 10 rows is not a full scan
    # The footer knows the total even though the cap stopped the read. Gating this on completeness
    # nulled it exactly when it carried information: "4 of 10" is a ratio, "4 of unknown" is not.
    assert result.coverage.examples_present == 10
    assert result.partitions[0].splits[0].num_examples == 10  # the footer count survives sampling


def test_profile_uncapped_read_is_a_full_scan(tmp_path):
    _write_parquet(tmp_path / "train-00000-of-00001.parquet", [{"a": i} for i in range(10)])

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=None)

    assert result.partitions[0].examples_complete is True
    assert result.coverage.examples_scanned == result.coverage.examples_present == 10


def test_profile_cap_larger_than_a_jsonl_file_keeps_it_exhaustive(tmp_path):
    # jsonl has no footer, so a cap could easily cost the exact count on files that never hit it.
    # Reading to EOF under the cap must stay exact, or capping would degrade every small dataset.
    (tmp_path / "train.jsonl").write_text('{"a": 1}\n{"a": 2}\n')

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=1000)

    assert result.partitions[0].splits[0].num_examples == 2
    assert result.partitions[0].examples_complete is True
    assert result.coverage.examples_present == 2


def test_profile_reports_unsupported_data_files(tmp_path):
    # A directory of formats we cannot read must not profile as an exhaustively scanned empty
    # dataset — that is indistinguishable from a dataset that really is empty.
    (tmp_path / "train.csv").write_text("prompt,completion\na,b\n")
    (tmp_path / "test.arrow").write_bytes(b"\x00")

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.partitions == []
    assert result.coverage.examples_present is None  # not 0: "empty" would be a lie
    assert result.coverage.files_read == 0
    assert result.coverage.files_present == 2  # both are data; neither could be read
    # ...and they still weigh what they weigh. This is the case `bytes_present` exists for: no
    # partition grouped these files, so summing the splits reports zero -- the same lie as "empty".
    on_disk = (tmp_path / "train.csv").stat().st_size + (tmp_path / "test.arrow").stat().st_size
    assert result.coverage.bytes_present == on_disk
    assert sum(s.size_bytes for p in result.partitions for s in p.splits) == 0
    # Typed records now, each saying why -- not bare paths tucked into a free-form dict.
    assert [e.path for e in result.file_errors] == ["test.arrow", "train.csv"]
    assert all("no reader" in e.error for e in result.file_errors)


def test_a_compressed_shard_is_data_the_profiler_cannot_read(tmp_path):
    # The worst shape this list guards against, because it is the silent one. `train.jsonl.gz`
    # reports `.gz` as its suffix, so it matched no reader and no unsupported extension and was
    # dropped before anything counted it: no partition, no error, `examples_present` 0, and the
    # completeness test the README documents still answering True. A real dataset profiled as an
    # exhaustively scanned empty one -- byte-identical to profiling an empty directory.
    import gzip

    with gzip.open(tmp_path / "train.jsonl.gz", "wt") as handle:
        handle.write('{"prompt": "a", "completion": "b"}\n')

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.coverage.files_present == 1  # it is there, and it is data
    assert result.coverage.bytes_present == (tmp_path / "train.jsonl.gz").stat().st_size
    assert result.coverage.examples_present is None  # not 0: "empty" would be the lie
    assert [e.path for e in result.file_errors] == ["train.jsonl.gz"]
    # The reason names the wrapper, which is the part that would have to be stripped -- not the
    # empty suffix the old message printed.
    assert result.file_errors[0].error == "no reader for compressed '.gz' files"


def test_a_data_file_with_no_extension_is_data(tmp_path):
    # Same silence, reached the other way: no suffix matches no table. Guessing "data" wrongly costs
    # one FileError; guessing "not data" wrongly hides the whole dataset, so the ambiguous case goes
    # to data and documentation is excluded by name instead.
    (tmp_path / "train").write_text('{"prompt": "a", "completion": "b"}\n')

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.coverage.files_present == 1
    assert result.coverage.examples_present is None
    assert [e.path for e in result.file_errors] == ["train"]
    assert result.file_errors[0].error == "no reader for a file with no extension"


def test_a_dataset_card_does_not_unknow_a_fileset_that_was_read_whole(tmp_path):
    # `.json` is on the unsupported list because it genuinely can be records, and one unreadable
    # data file unknows `examples_present` for the entire fileset. That put the ordinary HuggingFace
    # layout -- shards plus `dataset_infos.json` -- in the position of reporting an unknown size
    # after reading every row of every shard.
    _write_parquet(tmp_path / "train.parquet", [{"a": 1}, {"a": 2}])
    (tmp_path / "dataset_infos.json").write_text('{"default": {"splits": {"train": {}}}}')
    (tmp_path / ".gitattributes").write_text("*.parquet filter=lfs diff=lfs merge=lfs -text\n")

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.file_errors == []
    assert result.coverage.examples_present == 2
    assert result.coverage.files_present == 1  # the shard; the card and the dotfile are not data
    assert result.partitions[0].examples_complete is True


def test_a_json_file_that_is_not_a_known_sidecar_is_still_data(tmp_path):
    # The exclusion is by name, not by extension: `.json` stays on the unsupported list, so a JSON
    # file that is not one of the known sidecars is still records the profiler cannot read.
    _write_parquet(tmp_path / "train.parquet", [{"a": 1}])
    (tmp_path / "extra_rows.json").write_text('[{"a": 2}]')

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert [e.path for e in result.file_errors] == ["extra_rows.json"]
    assert result.coverage.examples_present is None


def test_profile_ignores_non_data_files_without_penalty(tmp_path):
    # A README is genuinely not data, so it must not cost exhaustiveness the way a .csv does.
    _write_parquet(tmp_path / "train-00000-of-00001.parquet", [{"a": 1}])
    (tmp_path / "README.md").write_text("a dataset card")
    (tmp_path / "LICENSE").write_text("apache")

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.partitions[0].examples_complete is True
    assert result.file_errors == []
    assert result.coverage.files_present == 1  # the README and LICENSE are not data, counted nowhere
    # Nor does their weight land on the dataset: a card is not part of what has to be moved.
    assert result.coverage.bytes_present == (tmp_path / "train-00000-of-00001.parquet").stat().st_size


def test_split_size_survives_a_shard_it_could_not_read(tmp_path):
    # Size comes from the listing and a row count from reading, so they go unknown independently.
    # A shard that fails to parse still weighs what it weighs, where the split's row count cannot.
    _write_parquet(tmp_path / "train-00000-of-00002.parquet", [{"a": 1}, {"a": 2}])
    (tmp_path / "train-00001-of-00002.parquet").write_bytes(b"not parquet")

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    on_disk = sum(p.stat().st_size for p in tmp_path.glob("*.parquet"))
    split = result.partitions[0].splits[0]
    assert split.num_files == 2
    assert split.size_bytes == on_disk  # both shards, including the one that would not open
    assert split.num_examples is None  # the broken shard's rows are unknowable...
    assert split.size_bytes > 0  # ...but its bytes are not
    assert result.coverage.bytes_present == on_disk


def test_profile_records_a_partial_jsonl_read(tmp_path):
    # One corrupt line costs that line, not the file — but the profile must still say the file was
    # only partly understood, rather than presenting a clean-looking count.
    (tmp_path / "train.jsonl").write_text('{"a": 1}\n{"a": 2\n{"a": 3}\n')

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    # Not 2: the rows it could parse are not the file's rows, and a count that omits the corrupt
    # line would read low while looking like a fact. `coverage` says what was actually scanned.
    assert result.partitions[0].splits[0].num_examples is None
    assert result.coverage.examples_scanned == 2
    assert result.coverage.examples_present is None
    assert [e.path for e in result.file_errors] == ["train.jsonl"]
    assert "line 2" in result.file_errors[0].error
    assert result.partitions[0].examples_complete is False  # a line was lost, so not a full scan


def test_profile_classifies_roles_type_and_verifiability(tmp_path):
    _write_parquet(
        tmp_path / "train-00000-of-00001.parquet",
        [{"problem": "q1", "solution": "steps #### 5"}, {"problem": "q2", "solution": "steps #### 6"}],
    )
    partition = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert {f.semantic_role for f in partition.features} == {"prompt", "completion"}
    assert partition.classification.primary == "prompt_completion"
    assert partition.classification.verifiability.method == "extractable_final_answer"
    assert partition.classification.verifiability.coverage == 1.0


def test_profile_from_value_dataset_is_a_chat_dataset(tmp_path):
    # End to end: {from, value} must reach the messages dtype, carry stats, and classify as chat
    # rather than falling through to `unknown` with nothing measured.
    conversation = [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]
    (tmp_path / "train.jsonl").write_text(json.dumps({"conversations": conversation}) + "\n")

    partition = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert partition.features[0].dtype == "messages"
    assert partition.features[0].semantic_role == "messages"
    assert partition.classification.primary == "messages"
    assert partition.stats["conversations"].messages.roles_seen == ["human", "gpt"]


def test_profile_degrades_one_partition_when_measurement_fails(tmp_path, monkeypatch):
    # Reads are isolated per file, but schema/stats/classification ran unguarded, so one odd value
    # could abort an otherwise complete profile. Structure must survive a measurement failure.
    from nemo_profiler_plugin.dataset import pipeline as pipeline_module

    _write_parquet(tmp_path / "train-00000-of-00001.parquet", [{"a": 1}, {"a": 2}])
    monkeypatch.setattr(pipeline_module, "classify", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)  # must not raise

    partition = result.partitions[0]
    assert partition.splits[0].num_examples == 2  # structure survives
    assert partition.stats == {}
    assert partition.classification.primary is None
    assert [e.kind for e in partition.classification.evidence] == ["error"]
    assert "RuntimeError" in partition.classification.evidence[0].detail  # says what failed


def test_a_read_failure_does_not_look_like_a_measurement_failure(tmp_path):
    # The two failure domains have to stay distinguishable: a bad *file* is a FileError, and the
    # rows that were readable still measure and classify normally. Folding the read and measure
    # loops together is what would blur this, so it is pinned before that happens.
    _write_parquet(tmp_path / "train-00000-of-00002.parquet", [{"prompt": "q", "completion": "a"}])
    (tmp_path / "train-00001-of-00002.parquet").write_bytes(b"not parquet")

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    part = result.partitions[0]
    assert [e.path for e in result.file_errors] == ["train-00001-of-00002.parquet"]
    assert part.classification.primary == "prompt_completion"  # the readable rows still classify
    assert "error" not in {e.kind for e in part.classification.evidence}
    assert part.stats  # ...and are still measured


def test_a_measurement_failure_does_not_look_like_a_read_failure(tmp_path, monkeypatch):
    from nemo_profiler_plugin.dataset import pipeline as pipeline_module

    _write_parquet(tmp_path / "train-00000-of-00001.parquet", [{"a": 1}])
    monkeypatch.setattr(pipeline_module, "classify", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.file_errors == []  # the file was fine; the data was odd
    assert [e.kind for e in result.partitions[0].classification.evidence] == ["error"]
    # `examples_complete` speaks to rows read, and every row *was* read -- so it stays True even though
    # there are no stats. Pinned as it stands; the field means what it says once Phase 5 renames it.
    assert result.partitions[0].examples_complete is True


def test_one_unmeasurable_column_does_not_cost_the_partition_its_classification(tmp_path, monkeypatch):
    # The narrow guard, end to end. The column's failure reaches the profile as evidence, and
    # everything the partition could still establish -- the other column's stats, the roles, the
    # dataset type -- survives it.
    from nemo_profiler_plugin.dataset import stats as stats_module

    real_observe = stats_module.RoutedAccumulator._observe

    def boom(self, present):
        if self._name == "completion":
            raise RuntimeError("boom")
        return real_observe(self, present)

    monkeypatch.setattr(stats_module.RoutedAccumulator, "_observe", boom)
    _write_parquet(tmp_path / "train.parquet", [{"prompt": "q", "completion": "a"}])

    part = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert "prompt" in part.stats and "completion" not in part.stats
    assert part.classification.primary == "prompt_completion"  # roles come from names, not stats
    assert any(e.kind == "error" and "'completion'" in e.detail for e in part.classification.evidence)
    # The reasoning for the classification still reads first; the failure is a caveat on it.
    assert part.classification.evidence[0].kind != "error"


def test_a_measurement_failure_is_scoped_to_its_own_partition(tmp_path, monkeypatch):
    from nemo_profiler_plugin.dataset import pipeline as pipeline_module

    real_classify = pipeline_module.classify

    def poison_one_partition(features, stats, **kwargs):
        if any(feature.name == "poison" for feature in features):
            raise RuntimeError("boom")
        return real_classify(features, stats, **kwargs)

    _write_parquet(tmp_path / "good" / "train.parquet", [{"prompt": "q", "completion": "a"}])
    _write_parquet(tmp_path / "bad" / "train.parquet", [{"poison": 1}])
    monkeypatch.setattr(pipeline_module, "classify", poison_one_partition)

    partitions = {p.name: p for p in profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions}

    # The wide guard is the one path that never runs the classifier, so it concludes nothing and
    # says so with an empty list -- the same shape as a partition nothing matched, distinguished by
    # the error evidence rather than by a coined type.
    assert partitions["bad"].classification.candidates == []
    assert partitions["bad"].classification.primary is None
    assert any(e.kind == "error" for e in partitions["bad"].classification.evidence)
    assert partitions["good"].classification.primary == "prompt_completion"
    assert partitions["good"].stats  # a neighbour's bad data costs this partition nothing


def test_a_file_that_fails_partway_still_counts_what_it_contributed(tmp_path, monkeypatch):
    # A read used to be all-or-nothing, so a failure meant no rows at all and the envelope could be
    # written after it. A fold cannot give rows back: batches already folded are in the statistics
    # whatever happens next, and counting the file as unread left `examples_scanned` describing fewer
    # rows than the stats were built from.
    from nemo_profiler_plugin.dataset import pipeline as pipeline_module

    _write_parquet(tmp_path / "train.parquet", [{"a": i} for i in range(4000)])
    real_update = pipeline_module._PartitionFolds.update
    calls = {"n": 0}

    def fail_on_the_third_batch(self, rows, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom mid-file")
        return real_update(self, rows, **kwargs)

    monkeypatch.setattr(pipeline_module._PartitionFolds, "update", fail_on_the_third_batch)
    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.coverage.examples_scanned == 2048  # two batches of 1024 were folded before it failed
    assert result.coverage.files_read == 1  # the file *was* read from, just not to its end
    assert result.partitions[0].stats["a"].numeric is not None  # and those rows shaped the stats
    assert [e.path for e in result.file_errors] == ["train.parquet"]
    assert result.partitions[0].examples_complete is False


def test_reading_everything_is_the_default(tmp_path):
    # The point of the whole exercise. The budget existed to keep a materialised partition off the
    # heap; nothing is materialised, so the default should not answer the question worse than it can
    # be answered.
    _write_parquet(tmp_path / "train.parquet", [{"t": f"row {i}"} for i in range(5_000)])

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.coverage.examples_scanned == 5_000 == result.coverage.examples_present
    assert result.partitions[0].examples_complete is True


def test_examples_complete_speaks_to_examples_read_not_to_exactness(tmp_path):
    # It was `stats_complete`, which promised more than it delivered: the length quantiles are
    # estimates by construction, whatever it says. Renamed to what it actually measures.
    _write_parquet(tmp_path / "train.parquet", [{"t": f"row {i}"} for i in range(100)])

    short = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=10)
    whole = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert short.partitions[0].examples_complete is False  # ten of a hundred rows
    assert whole.partitions[0].examples_complete is True
    # True either way, and it is the measurements themselves that say whether they are exact.
    assert whole.partitions[0].stats["t"].text.code_points.max == max(len(f"row {i}") for i in range(100))


def test_profile_is_deterministic(tmp_path):
    _write_parquet(tmp_path / "train-00000-of-00001.parquet", [{"a": 1}, {"a": 2}])
    source = LocalFileSource(tmp_path)
    first = profile(source, created_at=FIXED_TIME)
    second = profile(source, created_at=FIXED_TIME)
    assert first.model_dump_json() == second.model_dump_json()


def test_profile_tolerates_non_object_jsonl_lines(tmp_path):
    # A valid-JSON-but-non-object line parses cleanly, so the reader (not the read) must handle it;
    # otherwise it would poison the unprotected schema/stats stage and abort the whole profile.
    (tmp_path / "train.jsonl").write_text('{"a": 1}\n[1, 2, 3]\n{"a": 2}\n')
    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.partitions[0].splits[0].num_examples == 2  # objects counted, stray array dropped
    assert result.partitions[0].examples_complete is True


def test_a_footer_count_survives_a_failure_part_way_through_the_data(tmp_path, monkeypatch):
    # The footer is read before any row is, so a row group that will not decode does not unknow the
    # count it already gave. Discarding it made a whole split's `num_examples` unknown over one bad
    # shard, where the contract has it counting every file's rows "whether or not that file was read
    # to the end" -- and left the fileset reporting an unknown size whose splits each knew theirs.
    from nemo_profiler_plugin.dataset.readers import parquet as parquet_module

    _write_parquet(tmp_path / "train.parquet", [{"a": i} for i in range(5000)])

    def fails_to_decode(self, source, entry, *, row_cap=None, errors=None):
        raise RuntimeError("row group failed to decode")
        yield  # pragma: no cover  - makes this a generator, as the protocol requires

    monkeypatch.setattr(parquet_module.ParquetReader, "batches", fails_to_decode)

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.partitions[0].splits[0].num_examples == 5000
    assert result.coverage.examples_present == 5000  # known, and the envelope agrees with the split
    assert result.coverage.examples_scanned == 0  # nothing was actually read
    assert result.partitions[0].examples_complete is False  # which is what says the read fell short
    assert len(result.file_errors) == 1


def test_a_json_array_saved_as_jsonl_is_not_an_empty_dataset(tmp_path):
    # The other side of the tolerance above. A stray non-object line costs nothing while some line
    # is a row; when none is, the file is not a dataset with no rows -- it is a file this reader
    # could not use, and saying nothing made the two indistinguishable. A pretty-printed JSON array
    # is exactly that file, on one line, and it profiled as valid and empty with no error at all.
    (tmp_path / "train.jsonl").write_text('[{"a": 1}, {"a": 2}]\n')

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.coverage.examples_present is None
    assert result.partitions[0].splits[0].num_examples is None
    assert result.partitions[0].examples_complete is False
    assert "not be line-delimited JSON" in result.file_errors[0].error


def test_an_empty_file_is_still_an_empty_dataset(tmp_path):
    # And the case the check must not catch: nothing to read is not the same as nothing usable.
    (tmp_path / "train.jsonl").write_text("")

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert result.coverage.examples_present == 0
    assert result.partitions[0].examples_complete is True
    assert result.file_errors == []


def test_profile_survives_a_hostile_directory(tmp_path):
    """Everything that can go wrong at once must still yield a profile that says what went wrong.

    Each of these individually used to either abort the run or vanish silently; this is the shape of
    bug that got through, so it is worth asserting as one scenario rather than only in isolation.
    """
    _write_parquet(tmp_path / "train-00000-of-00002.parquet", [{"prompt": "a", "completion": "b"}])
    (tmp_path / "train-00001-of-00002.parquet").write_bytes(b"not a parquet file")  # corrupt
    (tmp_path / "extra.jsonl").write_text(
        '{"messages": [{"role": 1, "content": "hi"}]}\n'  # non-string role
        '{"messages": [{"role": "user"\n'  # truncated line
        "[1, 2, 3]\n"  # valid JSON, not a row
        '{"messages": [{"role": "user", "content": "ok"}]}\n'
    )
    (tmp_path / "leftovers.csv").write_text("a,b\n1,2\n")  # recognizable data, no reader
    (tmp_path / "README.md").write_text("a dataset card")  # not data at all

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)  # must not raise

    # Nothing here is exhaustive, and the profile says so rather than looking clean.
    assert result.partitions[0].examples_complete is False
    assert result.coverage.examples_present is None
    # One channel for every file the profiler could not use, whether or not a partition grouped it:
    # the .csv it never read, the corrupt shard, and the jsonl it only partly parsed.
    assert [e.path for e in result.file_errors] == [
        "extra.jsonl",
        "leftovers.csv",
        "train-00001-of-00002.parquet",
    ]

    # One partition, not one per format: the stray .jsonl is noise, not a second dataset.
    assert len(result.partitions) == 1
    partition = result.partitions[0]
    assert partition.file_formats == ["jsonl", "parquet"]
    # The readable parquet rows still produced a real classification...
    assert partition.classification.primary == "prompt_completion"
    # ...and the odd jsonl rows were measured rather than aborting the run.
    assert partition.stats["messages"].messages.roles_seen == ["1", "user"]

    # The whole thing still round-trips as a stored profile.
    assert DatasetProfile.model_validate_json(result.model_dump_json()) == result


def test_split_file_counts_account_for_every_data_file(tmp_path):
    # The contract promises split membership is exhaustive and disjoint. With per-file records gone
    # the counts are all that carries it, so the invariant is worth asserting on them directly --
    # a count that silently dropped a file would look exactly like a smaller dataset.
    _write_parquet(tmp_path / "train-00000-of-00002.parquet", [{"a": 1}])
    _write_parquet(tmp_path / "train-00001-of-00002.parquet", [{"a": 2}])
    _write_parquet(tmp_path / "test-00000-of-00001.parquet", [{"a": 3}])
    (tmp_path / "README.md").write_text("a dataset card")  # not data; never becomes a FileRecord

    source = LocalFileSource(tmp_path)
    result = profile(source, created_at=FIXED_TIME)

    counted = sum(split.num_files for partition in result.partitions for split in partition.splits)
    listed = [e.path for e in source.list_files() if e.path.endswith(".parquet")]
    assert counted == len(listed)  # exhaustive and disjoint: each file lands in exactly one split


def test_profile_isolates_detected_format_with_no_reader(tmp_path, monkeypatch):
    # If detect_format recognizes an extension the registry has no reader for, that file must be
    # isolated like a corrupt one, not crash the whole profile.
    from nemo_profiler_plugin.dataset.readers import base

    monkeypatch.setitem(base._EXTENSION_FORMATS, ".xyz", "xyz-no-reader")
    _write_parquet(tmp_path / "train-00000-of-00001.parquet", [{"a": 1}])
    (tmp_path / "extra.xyz").write_text("whatever")

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)  # must not raise

    assert "extra.xyz" in {e.path for e in result.file_errors}  # named, not silently dropped
    assert result.partitions[0].examples_complete is False


def test_a_column_only_a_schemaless_file_witnessed_survives(tmp_path):
    # A group where only some files declare a schema used to trust that schema and erase every column
    # the schemaless files were the sole witness for. Now the partition infers its schema from the
    # rows as it folds them, so the sole witness is heard.
    _write_parquet(tmp_path / "train-00000-of-00002.parquet", [{"prompt": "a"}])
    (tmp_path / "train-00001-of-00002.jsonl").write_text(json.dumps({"prompt": "b", "extra": "only here"}))

    part = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert [f.name for f in part.features] == ["prompt", "extra"]
    assert set(part.stats) <= {f.name for f in part.features}


def test_a_declared_schema_is_trusted_when_it_covers_every_file(tmp_path):
    # The other half: when every file declares one, the schema is authoritative and the rows are not
    # consulted for it. That is what lets the partition fold with its accumulators chosen up front.
    _write_parquet(tmp_path / "train-00000-of-00002.parquet", [{"prompt": "a"}])
    _write_parquet(tmp_path / "train-00001-of-00002.parquet", [{"prompt": "b"}])

    part = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert [f.name for f in part.features] == ["prompt"]


def test_one_unreadable_shard_does_not_discard_the_declared_schema(tmp_path):
    # A file that cannot be peeked cannot be read either, so it contributes no rows and no columns.
    # Holding the whole partition to it re-typed every healthy shard from row inference: the declared
    # int32 widened to int64 and an all-null string column dropped to `json`, which then fails every
    # dtype gate in classification. One bad file in five hundred was enough.
    schema = pa.schema([pa.field("score", pa.int32()), pa.field("empty", pa.string())])
    table = pa.Table.from_arrays(
        [pa.array([1, 2], type=pa.int32()), pa.array([None, None], type=pa.string())], schema=schema
    )
    pq.write_table(table, tmp_path / "train-00000-of-00002.parquet")
    (tmp_path / "train-00001-of-00002.parquet").write_bytes(b"PAR1junk")

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)

    assert [(f.name, f.dtype) for f in result.partitions[0].features] == [("score", "int32"), ("empty", "string")]
    assert len(result.file_errors) == 1  # the bad shard is still named, and still costs the row count
    assert result.coverage.examples_present is None


def test_a_duplicate_column_name_is_described_once(tmp_path):
    # Parquet permits duplicate field names, and the fold keeps the first so that which one wins is
    # deterministic. The caller used to hold its own copy of the declared schema, though, so
    # `features` announced a column no accumulator had measured and disagreed with `stats` about how
    # many there were. The fold reports the schema it actually measured, so the two cannot drift.
    schema = pa.schema([pa.field("prompt", pa.string()), pa.field("prompt", pa.string())])
    table = pa.Table.from_arrays([pa.array(["a", "b"]), pa.array(["c", "d"])], schema=schema)
    pq.write_table(table, tmp_path / "train.parquet")

    part = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert [(f.name, f.dtype) for f in part.features] == [("prompt", "string")]
    assert set(part.stats) == {f.name for f in part.features}


def test_a_file_that_failed_peek_but_read_fine_keeps_its_columns(tmp_path, monkeypatch):
    # `peek` and `batches` are separate opens, so the comment's bet -- "a file that could not be
    # peeked will not read either" -- is an assumption and not a guarantee. When it does not hold,
    # the declared schema is built from the other files and `RowFold` folds `row.get(name)` over
    # declared names only, so every column the unpeekable file alone witnessed vanished: no feature,
    # no stat, and no FileError to point at, because the read itself succeeded.
    from nemo_profiler_plugin.dataset.readers import parquet as parquet_module

    _write_parquet(tmp_path / "a.parquet", [{"prompt": "p", "extra": "only here"}])
    _write_parquet(tmp_path / "b.parquet", [{"prompt": "q"}])

    real_peek = parquet_module.ParquetReader.peek

    def flaky_peek(self, source, entry):
        if entry.path == "a.parquet":
            raise OSError("transient storage hiccup")
        return real_peek(self, source, entry)

    monkeypatch.setattr(parquet_module.ParquetReader, "peek", flaky_peek)

    part = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert {f.name for f in part.features} == {"prompt", "extra"}
    assert part.stats["extra"].null_rate == 0.5  # absent from b.parquet, back-filled as null


def test_a_zero_row_budget_still_checks_the_format(tmp_path):
    # The zero-cap guard sat above the format check in the jsonl reader and below it in parquet, so
    # a gzip archive saved as `.jsonl` was reported at every budget except zero -- where it passed
    # silently as a readable file that happened to hold no rows. A cap says how much to read, not
    # whether to look.
    import gzip

    with gzip.open(tmp_path / "shard.jsonl", "wt") as handle:
        handle.write('{"a": 1}\n')

    for budget in (None, 5, 0):
        result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=budget)
        assert [e.path for e in result.file_errors] == ["shard.jsonl"], budget


def test_discovery_is_scoped_to_the_file_that_needs_it(tmp_path, monkeypatch):
    # A file that fails `peek` but reads fine is not described by the declared schema, so its rows
    # are walked to find the columns it alone witnesses. That flag was set for the whole partition,
    # which put the same per-row key walk on every healthy shard -- one bad file in five hundred
    # paying for all of them. The condition is per file, so the flag is too.
    from nemo_profiler_plugin.dataset import stats as stats_module
    from nemo_profiler_plugin.dataset.readers import parquet as parquet_module

    shards, rows_each = 8, 50
    for i in range(shards):
        rows = [{"prompt": "p"} for _ in range(rows_each)]
        if i == 0:
            for row in rows:
                row["only_here"] = "x"
        _write_parquet(tmp_path / f"train-{i:05d}-of-{shards:05d}.parquet", rows)

    real_peek = parquet_module.ParquetReader.peek

    def flaky_peek(self, source, entry):
        if entry.path.startswith("train-00000"):
            raise OSError("transient storage hiccup")
        return real_peek(self, source, entry)

    monkeypatch.setattr(parquet_module.ParquetReader, "peek", flaky_peek)

    walked = {"rows": 0}
    real_discover = stats_module.RowFold._discover

    def counting_discover(self, batch):
        walked["rows"] += len(batch)
        return real_discover(self, batch)

    monkeypatch.setattr(stats_module.RowFold, "_discover", counting_discover)
    part = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    # Only the unpeekable shard's rows, not the whole partition's.
    assert walked["rows"] == rows_each, f"{walked['rows']} of {shards * rows_each}"
    # ...and the column it alone carries is still found and back-filled.
    assert "only_here" in {f.name for f in part.features}
    assert part.stats["only_here"].null_rate == (shards - 1) / shards


def test_a_zero_row_budget_reads_no_rows_from_either_format(tmp_path):
    # The jsonl reader tested the cap *after* appending, so `len(rows) >= row_cap` could not fire
    # until a row was already in hand -- one row per file, folded and measured, for the one argument
    # `_per_file_cap` exists to define. Parquet already returned nothing for it.
    (tmp_path / "a.jsonl").write_text('{"a": 1, "src": "x"}\n{"a": 2, "src": "y"}\n')
    (tmp_path / "b.jsonl").write_text('{"a": 3, "src": "z"}\n')
    _write_parquet(tmp_path / "c.parquet", [{"a": 4, "src": "w"}])

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=0)

    assert result.coverage.examples_scanned == 0
    assert all(not p.stats for p in result.partitions)


def test_a_file_holding_exactly_the_cap_was_not_cut_short(tmp_path):
    # `scanned >= row_cap` called this truncation, so a partition read from end to end lost its
    # `categorical.values` -- while `examples_complete` said, correctly, that nothing had been missed.
    # The honest test is whether every row of the file was parsed.
    _write_parquet(tmp_path / "train.parquet", [{"source": v} for v in ["a", "b"] * 10])

    for budget, quoted in ((19, False), (20, True), (21, True), (None, True)):
        part = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=budget).partitions[0]
        values = part.stats["source"].categorical.values
        assert (values is not None) is quoted, f"row_budget={budget}"
        if budget != 19:
            assert part.examples_complete is True, f"row_budget={budget}"


def test_a_zero_row_cap_means_zero_however_many_files_a_partition_has():
    # `max(MIN_ROWS_PER_FILE, 0 // n)` floored a zero budget up to ten rows per file as soon as a
    # partition had more than one, so the same argument meant "no rows" for a single-file partition
    # and "ten per file" for a sharded one.
    assert _per_file_cap(0, 1) == 0
    assert _per_file_cap(0, 3) == 0
    assert _per_file_cap(None, 3) is None
    assert _per_file_cap(100, 3) == 33
    # The floor still applies to a real budget, so a thousand shards do not get one row each.
    assert _per_file_cap(100, 10_000) == MIN_ROWS_PER_FILE


def test_a_file_abandoned_mid_read_does_not_quote_the_prefix_it_managed(tmp_path, monkeypatch):
    # `values` is the one place row content reaches the stored profile, so it may only be written
    # when the read proves it is the whole vocabulary. A file that dies part-way through opens,
    # folds a prefix, and raises -- and the gate used to exclude any file that reported an error,
    # on the assumption that a failed file failed *to open*. So the first batch's distinct values
    # were published as the column's controlled vocabulary while `examples_complete` said False.
    from nemo_profiler_plugin.dataset import pipeline as pipeline_module

    # `label` shows only en/fr before the failure point and de/ja/zh after it.
    rows = [{"label": ("en", "fr")[i % 2] if i < 2048 else ("de", "ja", "zh")[i % 3]} for i in range(4000)]
    _write_parquet(tmp_path / "train.parquet", rows)
    real_update = pipeline_module._PartitionFolds.update
    calls = {"n": 0}

    def fail_on_the_third_batch(self, batch, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("row group failed to decode")
        return real_update(self, batch, **kwargs)

    monkeypatch.setattr(pipeline_module._PartitionFolds, "update", fail_on_the_third_batch)
    # No row budget: the gate was `row_cap is not None and ...`, so with nothing capping the read
    # there was no gate at all and every mid-read failure quoted.
    partition = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert partition.examples_complete is False
    assert partition.stats["label"].categorical.values is None  # not ['en', 'fr']
    # The count still describes what was read; only the vocabulary claim is withheld.
    assert partition.stats["label"].categorical.distinct_count == 2


def test_a_shard_lost_before_it_yielded_a_row_still_quotes(tmp_path, monkeypatch):
    # The other side of the same gate, and the reason it cannot simply be "any error suppresses".
    # A file that raised before yielding a row contributed nothing to measure, so the vocabulary of
    # the files that *were* read is entire. Pinned here against the mid-read case above, which is
    # the distinction the two share a line of code for.
    from nemo_profiler_plugin.dataset import pipeline as pipeline_module

    _write_parquet(tmp_path / "train.parquet", [{"label": t} for t in ("en", "fr", "en")])
    _write_parquet(tmp_path / "extra.parquet", [{"label": "de"}])
    real_update = pipeline_module._PartitionFolds.update

    def fail_before_the_first_batch(self, batch, **kwargs):
        if any(row.get("label") == "de" for row in batch):
            raise RuntimeError("storage went away")
        return real_update(self, batch, **kwargs)

    monkeypatch.setattr(pipeline_module._PartitionFolds, "update", fail_before_the_first_batch)
    partition = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert partition.examples_complete is False  # a shard was lost, and that is reported
    assert partition.stats["label"].categorical.values == ["en", "fr"]  # ...but these are entire


def test_a_budgeted_read_does_not_quote_an_unproven_enumeration(tmp_path):
    # `categorical.values` is the one path row content reaches the stored profile, and it claims to
    # be the column's controlled vocabulary. A budgeted read saw a prefix, so what it collected is a
    # *sample* -- here the values are grouped, so the first ten rows witness one of four.
    #
    # Gated on truncation and not on `examples_complete`, which is the distinction
    # `test_examples_completeness_is_per_partition` pins from the other side: a partition that merely
    # lost a shard read every file it could open to the end, and still quotes.
    lines = [json.dumps({"prompt": f"q{i}", "source": ["aaa", "bbb", "ccc", "ddd"][i // 25]}) for i in range(100)]
    (tmp_path / "train.jsonl").write_text("\n".join(lines) + "\n")

    budgeted = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=10).partitions[0]
    assert budgeted.examples_complete is False
    assert budgeted.stats["source"].categorical.values is None
    # The count survives as the lower bound the field documents, rather than vanishing entirely.
    assert budgeted.stats["source"].categorical.distinct_count == 1

    whole = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]
    assert whole.examples_complete is True
    assert whole.stats["source"].categorical.values == ["aaa", "bbb", "ccc", "ddd"]
    assert whole.stats["source"].categorical.distinct_count == 4


def test_a_dictionary_encoded_column_profiles_as_its_value_type(tmp_path):
    # The whole partition turned on this. A dictionary-encoded `prompt` typed as `json`, which has no
    # measurement and passes no role gate, so the column lost its stats, lost the `prompt` role, and
    # with the role gone the partition classified as nothing at all.
    encoded = tmp_path / "encoded"
    plain = tmp_path / "plain"
    encoded.mkdir()
    plain.mkdir()
    pq.write_table(
        pa.table({"prompt": pa.array(["a", "b", "c"]).dictionary_encode(), "completion": pa.array(["x", "y", "z"])}),
        encoded / "train.parquet",
    )
    pq.write_table(
        pa.table({"prompt": pa.array(["a", "b", "c"]), "completion": pa.array(["x", "y", "z"])}),
        plain / "train.parquet",
    )

    encoded_part = profile(LocalFileSource(encoded), created_at=FIXED_TIME).partitions[0]
    plain_part = profile(LocalFileSource(plain), created_at=FIXED_TIME).partitions[0]

    assert [(f.name, f.dtype, f.semantic_role) for f in encoded_part.features] == [
        (f.name, f.dtype, f.semantic_role) for f in plain_part.features
    ]
    assert encoded_part.classification.candidates == plain_part.classification.candidates == ["prompt_completion"]
    assert encoded_part.stats["prompt"].text is not None


def test_a_declared_dtype_measures_every_value_it_was_given(tmp_path):
    # The other half of duplicate field names: `to_pylist` collapses the pair to the *last* one's
    # values, so the schema says `string` while the rows hold ints. A declared column skips the
    # per-type routing -- the dtype already names the measurement that will answer -- so that
    # measurement is handed the batch whole and sees exactly what a directly chosen accumulator
    # would. Lengths need a string and find none; the vocabulary counts values whatever they are.
    schema = pa.schema([pa.field("prompt", pa.string()), pa.field("prompt", pa.int64())])
    table = pa.Table.from_arrays([pa.array(["a", "b"]), pa.array([1, 2])], schema=schema)
    pq.write_table(table, tmp_path / "train.parquet")

    part = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions[0]

    assert [(f.name, f.dtype) for f in part.features] == [("prompt", "string")]
    categorical = part.stats["prompt"].categorical
    assert part.stats["prompt"].text is None  # no value was a string, so no lengths
    assert categorical is not None and categorical.distinct_count == 2


def test_examples_completeness_is_per_partition(tmp_path):
    # A corrupt shard in one partition says nothing about the measurements in another, but a
    # fileset-wide flag downgraded every partition to the worst one. It was never even the value
    # that gated quoting a proven enumeration -- that was decided per partition and never stored.
    rows = [{"label": t} for t in (True, False, True)]
    _write_parquet(tmp_path / "main" / "train.parquet", rows)
    _write_parquet(tmp_path / "socratic" / "train.parquet", rows)
    (tmp_path / "socratic" / "broken.parquet").write_bytes(b"not a parquet file")

    partitions = {p.name: p for p in profile(LocalFileSource(tmp_path), created_at=FIXED_TIME).partitions}

    assert partitions["main"].examples_complete is True
    assert partitions["socratic"].examples_complete is False
    # Quoting is decided by role, not by completeness, so both keep their label vocabulary --
    # examples_complete is what tells a consumer whether socratic's list is the whole of it.
    assert partitions["main"].stats["label"].categorical.values == ["False", "True"]
    assert partitions["socratic"].stats["label"].categorical.values == ["False", "True"]


def test_dataset_wide_completeness_is_one_expression(tmp_path):
    # Coverage no longer carries `exhaustive`; the contract documents this derivation in its
    # place. It has to keep working, or dropping the flag cost consumers something -- and it now
    # says *which* half failed, which the single bit could not.
    _write_parquet(tmp_path / "train.parquet", [{"a": 1}])
    clean = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)
    assert all(p.examples_complete for p in clean.partitions) and not clean.file_errors

    (tmp_path / "extra.csv").write_text("a,b\n1,2\n")
    with_csv = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME)
    assert all(p.examples_complete for p in with_csv.partitions)  # the parquet rows are still complete
    assert with_csv.file_errors  # but there is data here that went unprofiled
    assert with_csv.coverage.files_read == 1 and with_csv.coverage.files_present == 2


def test_row_budget_is_divided_across_a_partitions_files(tmp_path):
    for shard in range(4):
        _write_parquet(tmp_path / f"train-{shard:05d}-of-00004.parquet", [{"a": i} for i in range(200)])

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=400)

    assert result.coverage.examples_scanned == 400  # 400 / 4 files = 100 rows each


def test_rows_read_do_not_grow_when_a_dataset_is_resharded(tmp_path_factory):
    # The property the budget exists for. Under a per-file cap the same data split ten ways further
    # cost ten times the peak memory while describing exactly the same rows.
    def rows_read(shards, per_shard):
        root = tmp_path_factory.mktemp(f"shards{shards}")
        for shard in range(shards):
            _write_parquet(root / f"train-{shard:05d}-of-{shards:05d}.parquet", [{"a": i} for i in range(per_shard)])
        return profile(LocalFileSource(root), created_at=FIXED_TIME, row_budget=400).coverage.examples_scanned

    assert rows_read(4, 200) == rows_read(40, 20) == 400


def test_a_single_file_partition_is_capped_at_the_budget_itself(tmp_path):
    # The floor exists to stop a *share* being whittled to nothing when a budget is divided. A
    # partition of one file divides nothing, so there is no share to protect and the budget is the
    # cap -- below the floor and correctly so. Pinned because the README names this as deliberate,
    # and it reads like the floor failing.
    _write_parquet(tmp_path / "train.parquet", [{"a": i} for i in range(100)])

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=3)

    assert result.coverage.examples_scanned == 3  # not the 10-row floor
    assert result.partitions[0].examples_complete is False


def test_row_budget_keeps_a_floor_under_very_thin_shards(tmp_path):
    # Below the floor a file cannot witness the columns only it holds, which is the reason every file
    # is opened rather than a subset sampled. Overshooting the budget there is the right trade, and
    # the arithmetic share would be 1, so the floor holds and the budget is deliberately exceeded.
    for shard in range(10):
        _write_parquet(tmp_path / f"train-{shard:05d}-of-00010.parquet", [{"a": i} for i in range(50)])

    result = profile(LocalFileSource(tmp_path), created_at=FIXED_TIME, row_budget=10)

    assert result.coverage.examples_scanned == 100  # 10 files x the 10-row floor, over the budget of 10
