# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classification: assign column roles, resolve the format/prompt-form axes, and pick a dataset type.

Roles are inferred from column names, gated by dtype, and stacked onto the feature nodes as
``semantic_role`` markers. The dataset type is the most specific structure the assigned roles
satisfy.

Content probes are measured in :mod:`stats` over every column; this module only interprets the
counts. Roles order that interpretation -- a column known to be the ground truth beats one that
merely looks like it -- but do not gate it, so a dataset with unrecognized column names keeps
whatever its content proves.
"""

from __future__ import annotations

from dataclasses import dataclass

from nemo_platform_plugin.files.profile import (
    ColumnStats,
    Evidence,
    FeatureSchema,
    PartitionClassification,
    Verifiability,
)
from nemo_profiler_plugin.dataset.stats import ColumnProbes

# Column-name aliases -> role. Score is handled separately (name alias + numeric dtype gate).
_ALIAS_ROLES = {
    "prompt": "prompt",
    "question": "prompt",
    "instruction": "prompt",
    "problem": "prompt",
    "query": "prompt",
    "context": "context",
    "input": "context",
    "passage": "context",
    "document": "context",
    "system": "system",
    "system_prompt": "system",
    "response": "completion",
    "output": "completion",
    "answer": "completion",
    "completion": "completion",
    "solution": "completion",
    "messages": "messages",
    "conversation": "messages",
    "conversations": "messages",
    "chosen": "chosen",
    "rejected": "rejected",
    "label": "label",
    "rank": "rank",
    "ground_truth": "ground_truth",
    "reference_answer": "ground_truth",
    "verification_info": "ground_truth",
    "test_cases": "ground_truth",
    "completions": "stepwise_completions",
    "labels": "stepwise_labels",
    "tools": "tools",
    "image": "image",
    "images": "image",
    "id": "id",
    "prompt_id": "id",
    "source": "provenance",
    "dataset": "provenance",
    "model": "provenance",
    "category": "meta",
}

_SCORE_ALIASES = {
    "score",
    "score_chosen",
    "score_rejected",
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
    "quality",
    "rating",
    "reward",
}

# Every role this profiler can assign. Derived from the alias table rather than written out, so the
# two cannot drift; `score` is the one role reached by its own alias set rather than by that table.
# A declared role outside this set is a typo, not a vocabulary the caller is ahead of us on: nothing
# downstream reads it, so storing it would put an out-of-vocabulary value in the profile and report
# a match that never happened.
_KNOWN_ROLES = frozenset(_ALIAS_ROLES.values()) | {"score"}

_TEXT_DTYPES = {"string", "messages"}
# A verification target is naturally a container: test_cases (list), verification_info (struct), or a
# plain string answer — but never a bare scalar/number, which is far more likely a label or score.
_GROUND_TRUTH_DTYPES = {"string", "messages", "list", "struct", "json"}
# Roles whose string-vs-messages dtype decides the format axis.
_SHAPE_ROLES = {"prompt", "completion", "chosen", "rejected", "messages"}


def _is_numeric(dtype: str) -> bool:
    return dtype.startswith(("int", "uint", "float"))


def _is_binary(column: ColumnStats | None) -> bool:
    """Whether a column was observed to hold one or two distinct values.

    Two is what a binary label has. One is accepted as well, because a partition is classified on
    its own: a shard whose labels happen to all be one class, or a read that stopped before the
    second class appeared, is still a label column, and must not classify differently from its
    sibling over class balance. `distinct_count` is a lower bound wherever `examples_complete` is false,
    which makes the looser bound the safe one.

    Zero is not that judgment call. A column with no values at all carries no label, and counting it
    as one let an all-null column decide the whole dataset type.
    """
    if column is None or column.categorical is None:
        return False
    return 1 <= column.categorical.distinct_count <= 2


def _is_label_column(feature: FeatureSchema, stats: dict[str, ColumnStats]) -> bool:
    """Whether a column named ``label`` really carries a binary preference label.

    A bool says so outright. Integers and strings are the common on-disk encodings -- ``0``/``1``,
    ``safe``/``unsafe`` -- and both are accepted, but only when the observed values really are
    binary: a wider range is a class index or a rating, which is a different claim, so it stays
    unroled. The encoding is not the question; the number of distinct values is.
    """
    if feature.dtype == "bool":
        return True
    if not _is_numeric(feature.dtype) and feature.dtype != "string":
        return False
    return _is_binary(stats.get(feature.name))


def _dtype_allows(feature: FeatureSchema, role: str, stats: dict[str, ColumnStats]) -> bool:
    """Whether this column's dtype can carry ``role`` at all.

    Applied to detected and declared roles alike. A hint says which column, not what the data is:
    without this, one typo (``{"score": "prompt"}`` on an int column) would silently produce a
    nonsense classification.
    """
    dtype = feature.dtype
    if role == "score" or role == "rank":
        return _is_numeric(dtype)
    if role == "label":
        return _is_label_column(feature, stats)
    if role == "messages":
        return dtype == "messages"
    if role in {"prompt", "completion", "chosen", "rejected", "context", "system"}:
        return dtype in _TEXT_DTYPES
    if role == "ground_truth":
        return dtype in _GROUND_TRUTH_DTYPES
    if role in {"stepwise_completions", "stepwise_labels"}:
        return dtype == "list"
    return True  # id / provenance / meta / tools / image carry no dtype constraint


def _role_for(feature: FeatureSchema, stats: dict[str, ColumnStats]) -> str | None:
    """The role this column's *name* implies, if the dtype does not contradict it."""
    name = feature.name.lower()
    if name in _SCORE_ALIASES and _is_numeric(feature.dtype):
        return "score"
    role = _ALIAS_ROLES.get(name)
    if role is None:
        return None
    return role if _dtype_allows(feature, role, stats) else None


def _assign_roles(
    features: list[FeatureSchema], stats: dict[str, ColumnStats], column_roles: dict[str, str]
) -> list[Evidence]:
    """Stack roles onto ``features`` in place; return evidence for any hint the data could not support.

    A declared role wins over the name-alias table, since the caller knows their schema and the
    table is ~35 English names. It still has to name a role this profiler assigns and pass the dtype
    gate, and a hint rejected on either count is reported rather than dropped.
    """
    rejected: list[Evidence] = []
    for feature in features:
        declared = column_roles.get(feature.name)
        if declared is not None:
            if declared not in _KNOWN_ROLES:
                rejected.append(
                    Evidence(
                        kind="user_hint",
                        detail=(
                            f"hint '{feature.name} -> {declared}' rejected: not a role this profiler "
                            f"assigns; falling back to detection"
                        ),
                    )
                )
            elif _dtype_allows(feature, declared, stats):
                feature.semantic_role = declared
                feature.semantic_role_source = "declared"
                continue
            else:
                rejected.append(
                    Evidence(
                        kind="user_hint",
                        detail=(
                            f"hint '{feature.name} -> {declared}' rejected: a {feature.dtype} column "
                            f"cannot carry that role; falling back to detection"
                        ),
                    )
                )
        role = _role_for(feature, stats)
        if role is not None:
            feature.semantic_role = role
            feature.semantic_role_source = "detected"
    return rejected


def _detect_modality(features: list[FeatureSchema]) -> str:
    if any(feature.semantic_role == "image" or feature.dtype == "image" for feature in features):
        return "image_text"
    return "text"


def _detect_format(features: list[FeatureSchema]) -> str | None:
    dtypes = {feature.dtype for feature in features if feature.semantic_role in _SHAPE_ROLES}
    has_messages = "messages" in dtypes
    has_string = "string" in dtypes
    if has_messages and has_string:
        return "mixed"
    if has_messages:
        return "conversational"
    if has_string:
        return "standard"
    return None


def _detect_prompt_form(roles: set[str]) -> str | None:
    if "prompt" in roles:
        return "explicit"
    if roles & {"chosen", "rejected", "completion"}:
        return "implicit"  # a prompt exists but is embedded in the completions
    return "n/a"


def _messages_stats(features: list[FeatureSchema], stats: dict[str, ColumnStats]):
    for feature in features:
        if feature.semantic_role == "messages":
            column = stats.get(feature.name)
            if column is not None:
                return column.messages
    return None


def _detect_types(features: list[FeatureSchema], stats: dict[str, ColumnStats]) -> list[str]:
    """Every dataset type the assigned roles satisfy, most specific first.

    Ordered by specificity, so the head is the best single answer and the tail is structures the
    same columns also satisfy. Returning only the head made rule order an invisible tie-break:
    prompt + completion + score + label is genuinely both scored_response and unpaired_preference.
    """
    roles = {feature.semantic_role for feature in features if feature.semantic_role}
    targets = roles & {"completion", "chosen", "rejected", "stepwise_completions"}
    candidates: list[str] = []

    def has(*required: str) -> bool:
        return all(role in roles for role in required)

    if has("prompt", "stepwise_completions", "stepwise_labels"):
        candidates.append("stepwise_supervision")
    if has("chosen", "rejected"):
        candidates.append("preference_pair")
    if has("prompt", "completion", "score"):
        candidates.append("scored_response")
    if has("prompt", "completion", "label"):
        candidates.append("unpaired_preference")
    # `rank` is only a dataset type alongside something to rank. On its own it short-circuits every
    # more specific structure above, so a stray numeric column named `rank` would mislabel the set.
    if has("rank") and targets:
        candidates.append("ranked_responses")
    if has("prompt", "completion"):
        candidates.append("prompt_completion")
    if "messages" in roles:
        message_stats = _messages_stats(features, stats)
        if message_stats is not None and message_stats.ends_with_assistant_rate < 0.5:
            candidates.append("prompt_only")  # a chat that ends on a user turn has no training target
        else:
            candidates.append("messages")
    # A prompt with nothing to predict. Guarded on `targets` because with candidates collected rather
    # than returned early, a prompt+completion set would otherwise claim prompt_only as well.
    if "prompt" in roles and not targets and "prompt_only" not in candidates:
        candidates.append("prompt_only")
    if len(features) == 1 and features[0].dtype == "string" and features[0].semantic_role is None:
        candidates.append("text")
    # Empty when nothing matched, rather than a coined "unknown". Absence of a type is not a type,
    # and a vocabulary field naming one had to be special-cased by every reader of it.
    return candidates


# --- interpreting the content probes --------------------------------------------------------------

# A verification target must cover at least this fraction of sampled rows to be asserted. Below it a
# hit is noise: one completion in thousands ending in `#### <number>` does not make a dataset
# verifiable. The coverage itself is still reported on the Verifiability.
_MIN_VERIFIABILITY_COVERAGE = 0.05


def _pct(fraction: float) -> str:
    return f"{round(fraction * 100)}%"


def _detect_verifiability(features: list[FeatureSchema], probes: dict[str, ColumnProbes]) -> Verifiability | None:
    """The strongest verification target the probes found, if any clears the coverage floor.

    Each method wins only if it clears the floor; otherwise fall through to the next, so a sparse
    ground_truth column yields to an extractable-answer signal instead of masking it.
    """
    ground_truth = next((feature for feature in features if feature.semantic_role == "ground_truth"), None)
    if ground_truth is not None:
        probe = probes.get(ground_truth.name)
        if probe is not None and probe.rows:
            coverage = probe.non_empty / probe.rows
            if coverage >= _MIN_VERIFIABILITY_COVERAGE:
                detail = f"'{ground_truth.name}' present in {_pct(coverage)} of {probe.rows} sampled rows"
                return Verifiability(
                    method="ground_truth_column",
                    coverage=coverage,
                    evidence=[Evidence(kind="content_probe", detail=detail)],
                )

    # A named completion is the authoritative place to look. Without one, take whichever column the
    # probes found the strongest signal in and name it — the markers are a fact about that column
    # whether or not its name happened to be in the alias table.
    completion = next((feature for feature in features if feature.semantic_role == "completion"), None)
    searched = [completion] if completion is not None else features
    best_name: str | None = None
    best_coverage = 0.0
    for feature in searched:
        probe = probes.get(feature.name)
        if probe is None or not probe.rows or not probe.texts:
            continue
        # Over every row the column was asked about, never over the rows that happened to hold text.
        # `texts` is each column's own denominator, so it scores a column present in one row out of a
        # thousand at 1.0 -- enough to outrank the real answer column at 0.8 and then be reported as
        # a coverage of 1.0, which the contract says a consumer may read literally. `rows` is the
        # denominator the ground_truth branch above already divides by, and the one this field is
        # documented as: the fraction of sampled rows carrying a usable verification target.
        coverage = probe.extractable_answer / probe.rows
        if coverage > best_coverage:
            best_name, best_coverage = feature.name, coverage

    if best_name is not None and best_coverage >= _MIN_VERIFIABILITY_COVERAGE:
        sampled = probes[best_name].rows
        detail = (
            f"'{best_name}' ends with an extractable answer (#### or \\boxed) in "
            f"{_pct(best_coverage)} of {sampled} sampled rows"
        )
        return Verifiability(
            method="extractable_final_answer",
            coverage=best_coverage,
            evidence=[Evidence(kind="content_probe", detail=detail)],
        )
    return None


# How much text two answers must open with in common before it reads as a shared prompt rather than
# a shared turn of phrase. Short enough that a one-line question counts, long enough that "I think
# that" does not.
_EMBEDDED_PROMPT_PREFIX_CHARS = 16


def _common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


@dataclass(frozen=True)
class PrefixPair:
    """How often two columns of the same row opened with the same long run of text."""

    pairs: int = 0
    shared: int = 0


# The column names the alias table maps to the two sides of a preference pair. Looked up directly
# rather than resolved through roles, because this fold runs before classification has assigned any.
# Matched against lowercased column names, as `_role_for` does. Looking them up with the row's own
# casing meant a `Chosen`/`Rejected` pair took both roles and reached `_implicit_prompt_evidence`
# with no pairs counted, so the "prompt is embedded" finding was dropped for any dataset that
# capitalizes its columns -- silently, since the roles themselves still landed.
# How many rows the pair probe will spend looking for its two columns before concluding a partition
# has none. Well above any batch, and far below a file: a ragged jsonl can introduce a column late,
# and a partition without the pair must not pay a walk per row forever to keep discovering that.
# Rows spent looking for a chosen/rejected pair before concluding a partition has none. Must be
# larger than one batch of rows or the bound means nothing: it was equal to `_BATCH_ROWS`, so the
# first batch consumed all of it and "a column can first appear well into a file" was never true.
# Line-delimited rows are ragged, and an optional column can genuinely arrive thousands of rows in.
_RESOLVE_ROW_BUDGET = 8192

_CHOSEN_NAMES = frozenset(name for name, role in _ALIAS_ROLES.items() if role == "chosen")
_REJECTED_NAMES = frozenset(name for name, role in _ALIAS_ROLES.items() if role == "rejected")


class PrefixPairFold:
    """The one probe that reads two columns against each other rather than each on its own.

    A preference set whose prompt is embedded in both answers shows up as a long shared prefix
    between them, which no per-column measurement can see. Being relational it cannot live on a
    column accumulator, so it folds separately over the same batches.

    It takes no schema, resolving its two columns off each row, which lets it run over a partition
    whose columns are not known yet. Values that are not text contribute nothing.
    """

    def __init__(self) -> None:
        self._pairs = 0
        self._shared = 0
        # How the last row that held the pair spelled it. A hint, not the answer -- see `update`.
        self._left_key: str | None = None
        self._right_key: str | None = None
        self._searched = 0  # rows spent looking for a pair that has never been seen
        self._gave_up = False

    def _spellings(self, row: dict) -> tuple[str | None, str | None]:
        """The pair's two keys as *this* row spells them, matched as `_role_for` matches names."""
        left = right = None
        for name in row:
            lowered = name.lower()
            if left is None and lowered in _CHOSEN_NAMES:
                left = name
            elif right is None and lowered in _REJECTED_NAMES:
                right = name
        return left, right

    def update(self, rows: list[dict]) -> None:
        """Fold a batch, taking the last row's spelling as a guess and checking it row by row.

        Remembering the spelling is what keeps this cheap: on a wide partition, re-deriving the two
        keys from every row cost 68x two dict lookups, paid by every partition, the great majority
        of which hold no such pair at all. But remembering it as *the* answer was wrong -- a column
        name is a fact about the partition only when the partition came from one export. Two shards
        written by different tools put `chosen` and `Chosen` in one directory, and pinning the first
        spelling silently dropped every row using the other. That did not merely lose evidence: 3
        matching pairs out of 303 became 3 out of 3, and the profile asserted an embedded prompt in
        "100% of pairs".

        So the remembered keys are tried first and a miss falls back to scanning that row. A
        homogeneous partition pays two lookups a row, a mixed one pays a scan only on the rows that
        differ, and a partition with no pair at all pays :data:`_RESOLVE_ROW_BUDGET` rows of
        scanning and then nothing.
        """
        for row in rows:
            left = row.get(self._left_key) if self._left_key is not None else None
            right = row.get(self._right_key) if self._right_key is not None else None
            if (left is None or right is None) and not self._gave_up:
                left_key, right_key = self._spellings(row)
                if left_key is not None and right_key is not None:
                    self._left_key, self._right_key = left_key, right_key
                    left, right = row[left_key], row[right_key]
                elif self._left_key is None or self._right_key is None:
                    # Still looking for a first pair. Spend a row of the budget; a partition that
                    # has not shown one by the end of it is taken not to have one.
                    self._searched += 1
                    self._gave_up = self._searched >= _RESOLVE_ROW_BUDGET
            if not isinstance(left, str) or not isinstance(right, str):
                continue
            self._pairs += 1
            if _common_prefix_len(left, right) >= _EMBEDDED_PROMPT_PREFIX_CHARS:
                self._shared += 1

    def result(self) -> PrefixPair:
        return PrefixPair(pairs=self._pairs, shared=self._shared)


def _implicit_prompt_evidence(
    features: list[FeatureSchema], probes: dict[str, ColumnProbes], prefix_pair: PrefixPair
) -> Evidence | None:
    targets = [f for f in features if f.semantic_role in {"chosen", "rejected", "completion"} and f.dtype == "string"]
    counted = [probes[f.name] for f in targets if f.name in probes]
    sampled = sum(probe.texts for probe in counted)
    marked = sum(probe.transcript_marker for probe in counted)
    if sampled and marked:
        detail = f"embedded transcript markers in {_pct(marked / sampled)} of sampled completions - prompt is embedded"
        return Evidence(kind="content_probe", detail=detail)

    if prefix_pair.pairs and prefix_pair.shared / prefix_pair.pairs >= 0.5:
        rate = _pct(prefix_pair.shared / prefix_pair.pairs)
        return Evidence(
            kind="content_probe",
            detail=f"chosen/rejected share a common prefix in {rate} of pairs - prompt is embedded",
        )
    return None


def classify(
    features: list[FeatureSchema],
    stats: dict[str, ColumnStats],
    *,
    probes: dict[str, ColumnProbes] | None = None,
    prefix_pair: PrefixPair | None = None,
    column_roles: dict[str, str] | None = None,
) -> PartitionClassification:
    """Assign roles onto ``features`` in place and return the partition's classification.

    ``probes`` are the per-column content measurements and ``prefix_pair`` the one relational one,
    both folded before this runs. Nothing here reads a row, which is what lets a partition be
    classified without ever having been materialised. Absent, each reads as "nothing was measured".

    ``column_roles`` maps a column name to a role the caller is asserting, taking precedence over the
    name-alias table but still subject to the dtype gates. The table is ~35 English names with no way
    to say "my `q` column is the prompt", and its misses are silent.
    """
    probes = probes or {}
    evidence = _assign_roles(features, stats, column_roles or {})
    roles = {feature.semantic_role for feature in features if feature.semantic_role}
    candidates = _detect_types(features, stats)
    fmt = _detect_format(features)
    prompt_form = _detect_prompt_form(roles) if candidates else None

    role_columns = [f"{feature.name} -> {feature.semantic_role}" for feature in features if feature.semantic_role]
    if role_columns:
        evidence.append(Evidence(kind="column_name", detail=f"columns matched roles: {', '.join(role_columns)}"))
    if fmt is not None:
        evidence.append(Evidence(kind="column_dtype", detail=f"{fmt} format from role column dtypes"))
    if prompt_form == "implicit":
        embedded = _implicit_prompt_evidence(features, probes, prefix_pair or PrefixPair())
        if embedded is not None:
            evidence.append(embedded)

    return PartitionClassification(
        modality=_detect_modality(features),
        candidates=candidates,
        format=fmt,
        prompt_form=prompt_form,
        verifiability=_detect_verifiability(features, probes),
        evidence=evidence,
    )
