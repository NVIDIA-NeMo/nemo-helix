# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stored contract for the dataset profiler.

A ``DatasetProfile`` is machine-owned metadata the profiler computes once, at the Files layer, so
every consumer reads one typed description of a dataset instead of downloading and re-inspecting it.

The profile has two stored layers:

* **structure** — predominantly facts: file layout, splits, the derived row schema (``features``),
  and per-column ``stats``. The one detected attribute living here is the ``FeatureSchema.semantic_role``
  marker stacked on the feature node it describes.
* **classification** — an objective description of what the data *is*: ``dataset_type``, the
  ``format`` / ``prompt_form`` axes, and ``verifiability``.

Vocabularies (``dataset_type``, ``semantic_role``, ``modality``, ...) are open ``str`` values with
documented canonical sets, not closed enums: only known values are emitted, but consumers must
tolerate unknown ones so the vocabulary can grow without a breaking change. Pydantic's default
``extra="ignore"`` gives the same forward-compatibility for unknown *fields*.

This module is pydantic-only — no platform dependencies — so the profiler can import it as a
standalone contract and ``DatasetMetadataContent`` can later carry it as a typed field.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

# Semver of THIS contract. Gates consumer compatibility: new detectors or vocabulary values are a
# minor bump; a change to the fields below is a major bump.
PROFILE_SCHEMA_VERSION = "1.0"


# ---- classification: an objective description of what the data is (computed, stored) -----------


class Evidence(BaseModel):
    """Why the profiler believes what it detected.

    Captured at profile time — the only moment it is cheap and guaranteed to match the stored
    result; once the data or the profiler version moves, a re-run explains the *new* snapshot, not
    the stored one.
    """

    kind: str = Field(
        description="column_name | column_dtype | content_probe | split_name | file_name | card_metadata",
    )
    detail: str = Field(
        description="Self-describing evidence, e.g. \"answer matches '#### <number>' in 100% of 1024 sampled rows\".",
    )


class Verifiability(BaseModel):
    """A found verification target. Present only when one exists; absence *is* the claim (not verifiable)."""

    method: str = Field(
        description="extractable_final_answer | ground_truth_column | constraint | test_cases",
    )
    coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of sampled rows with a usable verification target.",
    )
    evidence: list[Evidence] = Field(default_factory=list)


class PartitionClassification(BaseModel):
    """What the data *is*, described objectively — the stored basis a downstream reader uses to decide
    which tasks the dataset can train.

    Holds the partition-level findings only: column-level semantics are the ``semantic_role`` markers
    on the feature nodes they describe, but the evidence for *why* they were assigned is recorded here.
    """

    modality: str = Field(default="text", description="text | image_text | audio_text | ...")
    dataset_type: str = Field(description="Dataset-type vocabulary (prompt_completion, preference_pair, ...).")
    format: str | None = Field(default=None, description="standard | conversational | mixed")
    prompt_form: str | None = Field(default=None, description="explicit | implicit | n/a")
    verifiability: Verifiability | None = Field(
        default=None,
        description="Present only when a verification target was found; keeps its own coverage-scoped evidence.",
    )
    evidence: list[Evidence] = Field(
        default_factory=list,
        description=(
            "Why the type / roles / axes were assigned: one flat list, detail strings self-describe "
            "what they support. A profile-time snapshot; unrecoverable once the data or profiler version move."
        ),
    )


# ---- structure: facts + the stacked semantic_role detection (computed) -------------------------


class Quantiles(BaseModel):
    """A per-row distribution summary. p99 = long-tail sequence-length signal; max = hard cap."""

    p50: int
    p95: int
    p99: int
    max: int


class TextStats(BaseModel):
    """Measurements for a ``string`` column."""

    chars: Quantiles = Field(description="Per-row character-length distribution.")


class MessageStats(BaseModel):
    """Measurements for a ``messages`` column (a list of ``{role, content}``)."""

    turns: Quantiles = Field(description="Per-row turn count (p99 -> packing long chats).")
    content_chars: Quantiles = Field(description="Per-row total content length -> chat sequence length.")
    roles_seen: list[str] = Field(
        default_factory=list,
        description=(
            'The distinct role strings actually present in the sampled rows, verbatim — e.g. ["system", '
            '"user", "assistant", "tool"], but equally ShareGPT\'s ["human", "gpt"] or a house convention. '
            "A measurement of row content, not a vocabulary the profiler picks from, so it is deliberately "
            "not an enum: an unexpected role is the finding worth reporting, and normalizing or dropping it "
            "would hide exactly what a consumer needs to see before choosing a chat template."
        ),
    )
    ends_with_assistant_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Key signal separating an SFT target (conversation ends on an assistant turn) from a prompt-only row.",
    )
    valid_alternation_rate: float = Field(ge=0.0, le=1.0)
    has_tool_calls: bool = False


class NumericStats(BaseModel):
    """Measurements for a numeric column."""

    min: float
    max: float
    mean: float


class TextQuality(BaseModel):
    """Cheap, single-pass corruption signals for a text column. Flags training-wrecking data, not
    toxicity / PII.
    """

    whitespace_ratio: float = Field(ge=0.0, le=1.0, description="Padding / bad scraping.")
    non_ascii_ratio: float = Field(ge=0.0, le=1.0, description="Encoding / non-Latin signal.")
    repetition_score: float = Field(ge=0.0, le=1.0, description="Degenerate repeated-substring loops.")


class FeatureSchema(BaseModel):
    """One node of the row schema, derived de novo from the data (there is no external JSON-Schema
    store to reference). Carries the measured layout (name, dtype, children) plus at most one
    detected ``semantic_role`` marker stacked on the same node.

    Recursive and fully expanded: a ``struct`` node has child ``fields``; a ``list`` / ``messages``
    node has an element ``items`` — for ``messages`` the per-message ``{role, content}`` struct is
    spelled out, so a vision message whose content is a list of typed parts shows up structurally.
    The column-level chat summary lives in ``MessageStats`` on the stats side. This tree is the
    clean, bridgeable schema artifact (e.g. to a JSON Schema or a UI columns view).
    """

    name: str = Field(default="", description='Column / struct-field name; "" for a list element.')
    dtype: str = Field(
        description=(
            "string | bool | int8..int64 / uint8..uint64 | float16/32/64 | struct | list | messages | "
            "image | audio | video | json | ... — fixed-width numeric widths as the source file reports them."
        ),
    )
    semantic_role: str | None = Field(
        default=None,
        description=(
            "Detected role (from the role vocabulary), valid at any depth of the tree; omitted when nothing "
            "was detected. The only detected attribute in the structure layer — its evidence lands in "
            "PartitionClassification.evidence. Named `semantic_role`, not `role`, so it never collides with a "
            "message struct's `role` key."
        ),
    )
    fixed_length: int | None = Field(
        default=None,
        description=(
            "dtype == list: constant observed element count (e.g. an embedding vector's 768), None when "
            "variable. Multi-dimensional shapes compose via nesting."
        ),
    )
    fields: list[FeatureSchema] | None = Field(default=None, description="dtype == struct: named child fields.")
    items: FeatureSchema | None = Field(default=None, description="dtype in {list, messages}: element schema.")

    @model_validator(mode="after")
    def _fields_and_items_are_exclusive(self) -> FeatureSchema:
        """A node is either a named-field container or has a single element schema, never both.

        Deliberately the only structural check here: it holds for *any* dtype, so it costs no
        forward compatibility. Tying `fields` / `items` / `fixed_length` to specific dtype values
        would instead reject a profile written by a newer profiler that added a container dtype,
        which is exactly what the open vocabulary exists to prevent.
        """
        if self.fields is not None and self.items is not None:
            raise ValueError(f"feature {self.name!r}: `fields` and `items` are mutually exclusive")
        return self


class CategoricalStats(BaseModel):
    """Cardinality signals for string / int columns.

    ``distinct_count`` is always safe to store; the values themselves ARE row data, so they appear
    only when proven to be a small enumeration by an exhaustive scan — the same
    assert-only-what-was-proven rule applied everywhere the profiler would otherwise leak row content.
    """

    distinct_count: int = Field(
        description=(
            "Distinct values among scanned rows: ~=rows_scanned -> id-like; a small bounded set "
            "corroborates score / category roles."
        ),
    )
    values: list[str] | None = Field(
        default=None,
        description="The proven enumeration; only when the scan was exhaustive and distinct_count <= 32.",
    )


class ColumnStats(BaseModel):
    """Measurements for one top-level column (keyed by name in ``PartitionProfile.stats``).

    The kind-specific block is populated by dtype; deep measurements fold into it (e.g.
    ``MessageStats.content_chars``) so stats stay flat — no path addressing to drift against the
    schema tree. Never row values — profiles stay safe to display / export without leaking data —
    with one gated exception: ``categorical.values``, a proven small enumeration.
    """

    null_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    text: TextStats | None = Field(default=None, description="dtype == string")
    numeric: NumericStats | None = Field(default=None, description="dtype in {int*, uint*, float*}")
    messages: MessageStats | None = Field(default=None, description="dtype == messages (list of {role, content})")
    categorical: CategoricalStats | None = Field(default=None, description="low observed cardinality only")
    quality: TextQuality | None = Field(default=None, description="dtype == string: corruption signals")


class FileRecord(BaseModel):
    """One physical file, measured.

    Stores the exact digest inputs (so a profile self-describes its ``content_digest`` and per-file
    staleness is computable) plus what the reader learned cheaply.
    """

    path: str = Field(description="Relative path within the fileset.")
    size_bytes: int
    checksum: str | None = Field(
        default=None,
        description=(
            'As the Files service reports it (e.g. "sha256:..."). None falls back to a (path, size) digest, '
            "which cannot detect a same-size in-place edit."
        ),
    )
    num_rows: int | None = Field(
        default=None,
        description="Exact only (parquet footer / exhaustive scan), else None.",
    )


class SplitProfile(BaseModel):
    """A split within a partition.

    Resolution precedence (declared structure beats detection):

    1. HF card front-matter when the fileset ships a README — ``configs[].data_files`` maps splits to
       file globs explicitly;
    2. best-effort detection from file paths (train/test/validation markers, sharded layouts like
       ``data/train-00000-of-00003.parquet``); path markers are matched against canonical names and
       common aliases (val/valid/dev -> validation), the normalized concept lands in ``canonical``, and
       the split keeps its on-disk ``name``;
    3. otherwise leave it alone: a single "default" split holding all files.

    A split encoded as a *data column* (a value inside each row rather than a file grouping) is not
    resolved here; such files profile as a single split.
    """

    name: str = Field(description="The on-disk name: train | test | train_prefs | ...")
    canonical: str | None = Field(
        default=None,
        description=(
            "Normalized concept: train | validation | test; None when nothing matches. E.g. train_prefs -> "
            "train, with the variant's intent kept in `name`."
        ),
    )
    files: list[FileRecord] = Field(
        description=(
            "Every file resolved into this split, measured. Partitioning is exhaustive and disjoint: each "
            "file of the partition lands in exactly one split, so concatenating `files` across splits "
            "reconstructs the partition's file list with no gaps or repeats."
        ),
    )
    num_examples: int | None = Field(
        default=None,
        description=(
            "Rows in this split, counting every file in `files` whether or not it was scanned. Exact when "
            "read from parquet footers or an exhaustive scan, otherwise extrapolated from the rows sampled — "
            "check `SamplingInfo.exhaustive` before treating it as a fact. None when nothing usable was found."
        ),
    )


class PartitionProfile(BaseModel):
    """A file-group sharing one row schema, one top-level directory, and one split-name variant
    (roughly an HF config); named after the directory / variant, else "default".

    File membership and row counts live on ``splits`` — every file lands in exactly one split, so
    partition-level files / num_examples would be derivable duplication.
    """

    name: str = "default"
    file_format: str = Field(description="jsonl | parquet | csv | arrow")
    splits: list[SplitProfile] = Field(description="card-declared > path-detected > single 'default' split.")
    features: list[FeatureSchema] = Field(
        description="The row schema: measured layout plus detected role markers, derived de novo (nested).",
    )
    stats: dict[str, ColumnStats] = Field(
        default_factory=dict,
        description=(
            "Top-level column name -> measurements; sparse (a column with nothing worth measuring is "
            "omitted); keys are a subset of the top-level `features` names."
        ),
    )
    classification: PartitionClassification

    @model_validator(mode="after")
    def _stats_keys_subset_of_features(self) -> PartitionProfile:
        """``stats`` is keyed by top-level column name, so every key must name a top-level feature
        (the producer keys stats by ``feature.name``); a stray key is a malformed profile."""
        unknown = set(self.stats) - {feature.name for feature in self.features}
        if unknown:
            raise ValueError(f"stats keys must name top-level features; unknown columns: {sorted(unknown)}")
        return self


# ---- envelope ----------------------------------------------------------------------------------


class SamplingInfo(BaseModel):
    """How much of the data the profile is based on.

    Consumers read ``exhaustive`` to decide whether stats are proven facts or estimates (e.g. only an
    exhaustive profile can assert enum / required in a bridged JSON Schema, or that verifiability
    coverage is truly 1.0).
    """

    exhaustive: bool = Field(description="True => every row of every file was parsed.")
    strategy: str = Field(
        description=(
            "full | stratified_probes | random. Kept explicit alongside `exhaustive` because, with an open "
            "strategy vocabulary, consumers can't derive exhaustiveness from the strategy name alone."
        ),
    )
    rows_scanned: int = Field(description="Total rows actually parsed across all files.")
    rows_total: int | None = Field(
        default=None,
        description=(
            "How many rows the whole fileset holds, scanned or not — the denominator `rows_scanned` is a "
            "fraction of, so a consumer can judge how representative the stats are. Populated only when the "
            "count is exact and cheap (summed parquet footers, or an exhaustive scan); None means unknown, "
            "never zero and never an estimate."
        ),
    )
    files_scanned: int = Field(
        description=(
            "How many files were opened and read from (a count, not a list — the files themselves are "
            "`SplitProfile.files`). Every file should be probed, since head-sampling a subset hides columns "
            "that appear only in later shards; expect this to equal the fileset's file count, and be lower "
            "only when scale forces file-level sampling."
        ),
    )
    per_file_row_cap: int | None = Field(default=None, description="Cap that bounded per-file reads, if any.")
    seed: int | None = Field(default=None, description="RNG seed used for row selection, for reproducibility.")


class DatasetProfile(BaseModel):
    """The machine-owned dataset profile — the root of the stored contract."""

    profile_schema_version: str = Field(
        default=PROFILE_SCHEMA_VERSION,
        description='Semver of THIS contract (e.g. "1.0") — gates consumer compatibility.',
    )
    content_digest: str = Field(description="Digest over the stored FileRecords; staleness = mismatch.")
    created_at: datetime
    profiler_info: dict = Field(
        default_factory=dict,
        description="Free-form profiler metadata (name, version, git sha, timings).",
    )
    sampling: SamplingInfo = Field(description="How much data the profile is based on.")
    partitions: list[PartitionProfile] = Field(
        description="Single partition in the common homogeneous case; there is no fileset-level rollup.",
    )


# Resolve the recursive FeatureSchema self-reference (deferred by `from __future__ import annotations`).
FeatureSchema.model_rebuild()
