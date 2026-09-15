# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-column statistics and content probes.

:class:`RowFold` measures each top-level column by dtype: length quantiles for text, min/max/mean
for numbers, chat-shape signals for messages, and a bounded vocabulary where the column has one.
The result is sparse, and each column is isolated, so one the detectors cannot handle costs only
itself.

The same pass counts each column's *content* -- answer markers, embedded transcripts -- into
:class:`ColumnProbes`. Those are measurements; what they mean is classification's job. Measuring
here, rather than behind a role, keeps a content signal reachable through a column whose name nobody
recognises.

A :class:`ColumnAccumulator` per column does the measuring. It folds batches in and keeps no
reference to them, so a column measured in pieces gives the same answer as one measured whole --
what lets a caller stop materialising a partition before measuring it. Row values are never stored;
:func:`quote_enumerations` adds a small controlled vocabulary afterwards, gated on role.

See the plugin README for the extended reasoning behind the bounds and the estimate."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from nemo_platform_plugin.files.profile import (
    CategoricalStats,
    ColumnStats,
    Evidence,
    FeatureSchema,
    MessageStats,
    NumericStats,
    Quantiles,
    TextStats,
)
from nemo_profiler_plugin.dataset.schema import MAX_COLUMNS, SchemaFold

# A quotable enumeration holds at most this many distinct values.
_MAX_ENUM_VALUES = 32

# Where a column stops being a plausible controlled vocabulary. Three bounds, because a count alone
# bounds cardinality but not bytes; all three sit well above real vocabularies.
#
# They bound a *column*, and there is deliberately no bound across columns. The pipeline's promise is
# that nothing kept grows with the *file*, which these keep: a column's vocabulary stops at 64 KiB
# however many rows it is fed. The working set does still grow with the column *count*, and the
# figure worth stating is not the one this comment first gave.
#
# `_MAX_VOCABULARY_BYTES` counts string *content*. Measured at the bound, a saturated vocabulary is
# about 137 KiB resident -- 2.1x -- once per-`str` overhead and the set's own table are counted. And
# a `RoutedAccumulator` builds one `_Vocabulary` per shape it sees, so a column holding strings,
# numbers and bools holds three. Worst case for a partition wide enough to hit the column cap is
# nearer 1.7 GB than the 256 MB that content alone suggests, held for one partition's read.
#
# Left as a bound rather than turned into a shared budget on purpose. A partition-wide byte budget
# would have to be threaded through every accumulator and consulted per value, and it would make one
# column's cardinality depend on how many other columns happened to be measured first -- an answer
# that changes with column order, for a shape (4,096 columns each holding a large distinct
# vocabulary) that is already the malformed input `MAX_COLUMNS` exists to bound.
_MAX_VOCABULARY_VALUES = 1024
_MAX_VOCABULARY_VALUE_CHARS = 256
_MAX_VOCABULARY_BYTES = 64 * 1024
# What a non-string distinct value is charged against the byte bound. Ints and bools are small and
# uniform, so an exact size buys nothing the count bound does not already give.
_NON_STRING_VALUE_BYTES = 8

# Roles that are controlled vocabularies by construction, and so safe to quote at any dataset size.
# An allowlist, so an unrecognized column fails to silence rather than to exposure.
_QUOTABLE_ROLES = frozenset({"label", "provenance", "meta", "rank"})


@dataclass(frozen=True)
class PartitionMeasurements:
    """Everything one pass over a partition's columns produced.

    A named result rather than a tuple because the vocabularies joined it: they are not part of the
    stored profile, but :func:`quote_enumerations` needs them and can no longer go back to the rows
    for them.
    """

    stats: dict[str, ColumnStats]
    probes: dict[str, ColumnProbes]
    vocabularies: dict[str, set[Any]]  # name -> distinct values, only where the column stayed one
    errors: list[Evidence]


class RowFold:
    """The per-column accumulators for one partition, fed batch by batch.

    A declared schema names the columns before the first batch. Without one they are discovered as
    they appear and back-filled with the rows they were absent for, which makes the result identical
    to inferring the schema first and measuring second. Both measure through the same accumulator, so
    a declared schema buys the schema itself, not a different mechanism.

    Each column is isolated, per column *per batch*: a value no detector anticipated costs that
    column its measurements and nothing else, reported as ``error`` evidence rather than left as a
    silent gap. This is the narrow half of the profiler's two guards; the wide one wraps the whole
    measure stage and catches anything structural that no single column owns.
    """

    def __init__(self, features: list[FeatureSchema] | None = None) -> None:
        # None means the partition declared no schema. It is not the same as an empty list, which is
        # a declared schema that happens to have no columns and must not then discover any.
        self._infer = features is None
        self._accumulators: dict[str, RoutedAccumulator] = {}
        self._declared: dict[str, FeatureSchema] = {}
        self._order: list[str] = []
        self._failed: dict[str, Evidence] = {}
        self._rows_seen = 0
        self._capped = False  # a column this fold declined to open, because the cap was reached
        for feature in features or []:
            # Parquet permits duplicate field names, and every map here is keyed by name. Keeping
            # the first makes which one wins deterministic rather than "whichever came last".
            if feature.name in self._accumulators:
                continue
            self._declared[feature.name] = feature
            self._accumulators[feature.name] = RoutedAccumulator(feature.name, feature)
            self._order.append(feature.name)

    def update(self, rows: list[dict[str, Any]], *, discover: bool = False) -> None:
        """Fold one batch of rows into every column's accumulator.

        ``discover`` turns discovery on for *this batch* alongside a declared schema, for rows the
        declaration is known not to describe. It is per batch because the condition is per file: one
        unpeekable shard in a partition is not a reason to walk the keys of every healthy shard's
        rows. A well-formed declared partition never sets it, and nothing is ever discovered anyway,
        since no row carries a key the schema did not name.
        """
        if self._infer or discover:
            self._discover(rows)
        for name in self._order:
            if name in self._failed:
                continue
            try:
                self._accumulators[name].update([row.get(name) for row in rows])
            except Exception as exc:
                self._failed[name] = Evidence(kind="error", detail=self._detail(name, "measured", exc))
        self._rows_seen += len(rows)

    def _discover(self, rows: list[dict[str, Any]]) -> None:
        """Open an accumulator for every column this batch is the first to mention."""
        for row in rows:
            for name in row:
                if name in self._accumulators:
                    continue
                if len(self._accumulators) >= MAX_COLUMNS:
                    self._capped = True
                    continue
                accumulator = RoutedAccumulator(name)
                accumulator.backfill_nulls(self._rows_seen)
                self._accumulators[name] = accumulator
                self._order.append(name)

    def columns_were_capped(self) -> bool:
        """Whether the cap stopped this fold describing a column, at the top level or nested.

        Reported by the fold that did the stopping rather than inferred from how many columns came
        back: a partition with exactly :data:`MAX_COLUMNS` columns is described completely and is
        indistinguishable by length from one that was truncated.
        """
        if self._capped:
            return True
        return any(accumulator.schema_was_capped() for accumulator in self._accumulators.values())

    def _detail(self, name: str, verb: str, exc: Exception) -> str:
        """Why a column dropped out, naming its declared dtype when it had one."""
        declared = self._declared.get(name)
        column = f"column {name!r}" + (f" ({declared.dtype})" if declared is not None else "")
        return f"{column} could not be {verb}: {type(exc).__name__}: {exc}"

    def finalize(self) -> tuple[list[FeatureSchema], PartitionMeasurements]:
        """The schema that was measured, and the measurements.

        The features come back rather than being assumed from what was handed in, because they are
        not the same list: duplicate names were dropped in the constructor, so a caller's own copy
        would describe a column no accumulator ever measured. An inferred column has no other source
        for its schema at all.
        """
        features: list[FeatureSchema] = []
        stats: dict[str, ColumnStats] = {}
        probes: dict[str, ColumnProbes] = {}
        vocabularies: dict[str, set[Any]] = {}
        errors: list[Evidence] = []
        for name in self._order:
            failure = self._failed.get(name)
            if failure is not None:
                # A declared column exists whether or not it could be measured, so it is still
                # described. An inferred one was never typed, and there is nothing to describe.
                declared = self._declared.get(name)
                if declared is not None:
                    features.append(declared)
                errors.append(failure)
                continue
            accumulator = self._accumulators[name]
            try:
                features.append(accumulator.feature())
                column, probe = accumulator.finalize()
            except Exception as exc:
                errors.append(Evidence(kind="error", detail=self._detail(name, "summarised", exc)))
                continue
            probes[name] = probe
            vocabulary = accumulator.vocabulary()
            if vocabulary is not None:
                vocabularies[name] = vocabulary
            if column is not None:
                stats[name] = column
        return features, PartitionMeasurements(stats=stats, probes=probes, vocabularies=vocabularies, errors=errors)


def quote_enumerations(
    features: list[FeatureSchema], stats: dict[str, ColumnStats], vocabularies: dict[str, set[Any]]
) -> None:
    """Fill in ``categorical.values`` for columns whose role makes them a controlled vocabulary.

    Runs after classification, since it needs the roles it gates on, and reads what the accumulators
    already kept rather than going back to the rows. It fills in rather than redacts: skip this pass
    and no values are stored, where a skipped redaction pass would leak them.

    Cardinality is the size bound, never the permission -- it inverts on small data, where every
    column holds few distinct values, free text included. The role says what a column *is*, at any
    size.
    """
    for feature in features:
        if feature.semantic_role not in _QUOTABLE_ROLES:
            continue
        column = stats.get(feature.name)
        if column is None or column.categorical is None or column.categorical.distinct_count > _MAX_ENUM_VALUES:
            continue
        values = vocabularies.get(feature.name)
        if values is None:
            continue
        column.categorical.values = sorted(str(value) for value in values)


class ColumnAccumulator:
    """Measures one top-level column, over however many batches it is handed.

    ``update`` folds a batch in and keeps no reference to it; ``finalize`` turns what was folded into
    the stored blocks. Splitting a column across calls gives the same answer as one call with all of
    it.

    The base class is the whole measurement for a dtype with no statistics of its own, because the
    content probes run over every column whatever its type. Subclasses add their state by overriding
    ``_observe`` and ``_stat_blocks``.
    """

    def __init__(self) -> None:
        self.rows = 0
        self._nulls = 0
        self._non_empty = 0
        self._texts = 0
        self._extractable_answer = 0
        self._transcript_marker = 0

    def update(self, values: list[Any]) -> None:
        """Fold one batch of this column's values in, one entry per row."""
        present: list[Any] = []
        for value in values:
            self.rows += 1
            # NaN counts as missing, because that is what it means. Parquet and pandas both spell a
            # missing float that way, and testing only for `None` left it in neither camp: not a
            # null, so `null_rate` read 0.0 over a column that was 97% absent, and not a value the
            # vocabulary would keep either, since that already skips NaN. A column holding one real
            # number and nothing else then presented as a fully-populated single-valued one, which
            # `_is_binary` reads as a label -- so a mostly-empty float column took the `label` role
            # and put `unpaired_preference` at the head of the candidates.
            if value is None or (isinstance(value, float) and math.isnan(value)):
                self._nulls += 1
            else:
                present.append(value)
                if value not in ("", [], {}):
                    self._non_empty += 1
            text = _probe_text(value)
            if text is not None:
                self._texts += 1
                if _GSM8K_ANSWER.search(text) or _BOXED_ANSWER.search(text):
                    self._extractable_answer += 1
                if _TRANSCRIPT_MARKER.search(text):
                    self._transcript_marker += 1
        self._observe(present)

    def finalize(self) -> tuple[ColumnStats | None, ColumnProbes]:
        """The column's stored measurements, and its probe counts.

        Stats are None when there was nothing worth measuring, which keeps the map sparse. Probes are
        always returned: a column of nothing is a finding classification is entitled to read.
        """
        blocks = self._stat_blocks()
        null_rate = self._nulls / self.rows if self.rows else 0.0
        column = ColumnStats(null_rate=null_rate, **blocks)
        if not any(blocks.values()) and null_rate == 0.0:
            column = None
        return column, ColumnProbes(
            rows=self.rows,
            non_empty=self._non_empty,
            texts=self._texts,
            extractable_answer=self._extractable_answer,
            transcript_marker=self._transcript_marker,
        )

    def _observe(self, present: list[Any]) -> None:
        """Fold this batch's non-null values into the dtype's own state. The base column has none."""

    def _stat_blocks(self) -> dict[str, Any]:
        """The dtype-specific ``ColumnStats`` blocks. The base column contributes none."""
        return {}

    def backfill_nulls(self, count: int) -> None:
        """Charge this column ``count`` rows in which it was absent.

        A column that first appears in the fiftieth batch was null for every row before it, which is
        what a materialising reader computes with ``row.get(name)``. Counted rather than fed as
        values, so discovering a column late costs two additions and not a pass.
        """
        self.rows += count
        self._nulls += count

    def vocabulary(self) -> set[Any] | None:
        """The distinct values, for a column that is a bounded vocabulary. None for one that is not,
        which is every dtype without a notion of cardinality."""
        return None


class _Vocabulary:
    """Distinct values, for as long as the column still looks like a controlled vocabulary.

    Counting distinct values exactly means *retaining* them, so on a free-text column this set grows
    to hold the column -- unbounded, the one thing that would make the fold O(rows) again. It stops
    the moment the column stops looking like a vocabulary, and drops what it had.

    Three bounds, because a count alone bounds cardinality but not bytes.
    :data:`_MAX_VOCABULARY_VALUE_CHARS` does the real work: it asks what the column *is* rather than
    how many values it holds, so free-text columns stop on their first long value.

    The values are never handed out here. They are row content, gated on role rather than size, by
    :func:`quote_enumerations`.
    """

    def __init__(self) -> None:
        self._values: set[Any] = set()
        self._bytes = 0
        self._saturated = False

    def update(self, present: list[Any]) -> None:
        if self._saturated:
            return
        for value in present:
            if isinstance(value, str) and len(value) > _MAX_VOCABULARY_VALUE_CHARS:
                return self._saturate()
            try:
                if value in self._values:
                    continue
                self._values.add(value)
            except TypeError:
                return self._saturate()  # unhashable values (dicts / lists) have no cardinality signal
            self._bytes += len(value) if isinstance(value, str) else _NON_STRING_VALUE_BYTES
            if len(self._values) > _MAX_VOCABULARY_VALUES or self._bytes > _MAX_VOCABULARY_BYTES:
                return self._saturate()

    def _saturate(self) -> None:
        self._values = set()  # release what was held; holding it is the cost this bound exists to cap
        self._saturated = True

    def finalize(self) -> CategoricalStats | None:
        return None if self._saturated else CategoricalStats(distinct_count=len(self._values))

    def values(self) -> set[Any] | None:
        """What was kept, or None once the column stopped being a vocabulary.

        Handing this out is what lets :func:`quote_enumerations` fill in an enumeration without a
        second pass over the rows -- which it could only do while the rows were still there.
        """
        return None if self._saturated else self._values


class StringAccumulator(ColumnAccumulator):
    """A ``string`` column: length quantiles and a vocabulary if it has one."""

    def __init__(self) -> None:
        super().__init__()
        self._lengths = _LengthHistogram()
        self._byte_lengths = _LengthHistogram()
        self._vocabulary = _Vocabulary()
        self._strings = 0

    def _observe(self, present: list[Any]) -> None:
        for value in present:
            if isinstance(value, str):
                self._lengths.add(len(value))
                self._byte_lengths.add(_utf8_len(value))
                self._strings += 1
        self._vocabulary.update(present)

    def _stat_blocks(self) -> dict[str, Any]:
        text = (
            TextStats(code_points=self._lengths.quantiles(), utf8_bytes=self._byte_lengths.quantiles())
            if self._strings
            else None
        )
        return {"text": text, "categorical": self._vocabulary.finalize()}

    def vocabulary(self) -> set[Any] | None:
        return self._vocabulary.values()


class NumericAccumulator(ColumnAccumulator):
    """An ``int*`` / ``uint*`` / ``float*`` column: running extrema and mean, plus a vocabulary."""

    def __init__(self) -> None:
        super().__init__()
        self._min = math.inf
        self._max = -math.inf
        self._mean = 0.0
        self._count = 0
        self._vocabulary = _Vocabulary()

    def _observe(self, present: list[Any]) -> None:
        countable: list[Any] = []
        for value in present:
            # NaN alone leaves the vocabulary, and only because it compares unequal to itself: a set
            # counts every NaN separately, so a parquet column of them reported its own row count as
            # its cardinality and past `_MAX_VOCABULARY_VALUES` saturated, taking the column's whole
            # `categorical` block with it. The infinities have none of that problem -- each equals
            # itself and costs one entry -- and dropping them made `{0.0, 1.0, inf}` count as two
            # distinct values, which is enough for `_is_binary` to call it a binary label.
            if isinstance(value, float) and math.isnan(value):
                continue
            countable.append(value)
            # bool is an int here, and is not a number. The extrema still skip every non-finite
            # value, infinities included: those serialize to JSON null and fail to re-validate
            # against NumericStats, which is a separate reason from the vocabulary's.
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                number = float(value)
                self._min = min(self._min, number)
                self._max = max(self._max, number)
                self._count += 1
                # Kept incrementally rather than as a running total divided at the end. A sum
                # overflows to `inf` on inputs that are each perfectly finite -- two values of
                # 1e308 are enough -- and `inf` serializes to JSON null, so the published profile
                # then failed to re-validate against its own NumericStats. The extrema already
                # skip non-finite values for that exact reason; the mean was reaching it by
                # arithmetic instead of by input.
                #
                # Weighted rather than the textbook `mean += (x - mean) / n`, which overflows on
                # its difference when two values sit at opposite extremes. Both terms here are
                # bounded by the larger of the two operands, so a mean of finite values stays in
                # range. Measured against `math.fsum`, that costs about 1.5e-14 relative error
                # where the textbook form gives 3e-15 -- both far below anything a profile reports.
                self._mean = self._mean * ((self._count - 1) / self._count) + number / self._count
        self._vocabulary.update(countable)

    def _stat_blocks(self) -> dict[str, Any]:
        numeric = None
        # `isfinite` on the mean as well as on the inputs: the incremental form above cannot
        # overflow from finite values, but a difference of two opposite extremes can, and a profile
        # that will not re-validate is worse than one missing a block it could not state.
        if self._count and math.isfinite(self._mean):
            numeric = NumericStats(min=self._min, max=self._max, mean=self._mean)
        return {"numeric": numeric, "categorical": self._vocabulary.finalize()}

    def vocabulary(self) -> set[Any] | None:
        return self._vocabulary.values()


class BoolAccumulator(ColumnAccumulator):
    """A ``bool`` column. The column that decides unpaired_preference deserves a measured class
    balance rather than no stats at all, and two values is a vocabulary by any reading."""

    def __init__(self) -> None:
        super().__init__()
        self._vocabulary = _Vocabulary()

    def _observe(self, present: list[Any]) -> None:
        self._vocabulary.update(present)

    def _stat_blocks(self) -> dict[str, Any]:
        return {"categorical": self._vocabulary.finalize()}

    def vocabulary(self) -> set[Any] | None:
        return self._vocabulary.values()


class MessageAccumulator(ColumnAccumulator):
    """A ``messages`` column: turn and length distributions, the roles seen, and chat-shape rates."""

    def __init__(self) -> None:
        super().__init__()
        self._conversations = 0
        self._turns = _LengthHistogram()
        self._content_code_points = _LengthHistogram()
        self._content_utf8_bytes = _LengthHistogram()
        self._roles_seen: list[str] = []
        self._ends_with_assistant = 0
        self._valid_alternation = 0
        self._has_tool_calls = False

    def _observe(self, present: list[Any]) -> None:
        for messages in present:
            if not isinstance(messages, list):
                continue
            self._conversations += 1
            self._turns.add(len(messages))
            total_content = 0
            total_content_bytes = 0
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = _role_of(message)
                if role is not None:
                    # Coerced to str for the contract, but reported verbatim: an unexpected role is
                    # the finding worth surfacing, not something to normalize away.
                    role = role if isinstance(role, str) else str(role)
                    role = role[:_MAX_ROLE_CHARS]
                    if role not in self._roles_seen and len(self._roles_seen) < _MAX_ROLES_SEEN:
                        self._roles_seen.append(role)
                code_points, utf8_bytes = _content_lengths(_message_field(message, "content", "value"))
                total_content += code_points
                total_content_bytes += utf8_bytes
                # `.get` truthiness, not `in`: parquet materializes every declared struct field, so a
                # schema that merely declares tool_calls would otherwise report tool use on every row.
                if message.get("tool_calls") or role == "tool":
                    self._has_tool_calls = True
            self._content_code_points.add(total_content)
            self._content_utf8_bytes.add(total_content_bytes)
            if messages and isinstance(messages[-1], dict) and _is_assistant_role(_role_of(messages[-1])):
                self._ends_with_assistant += 1
            if _valid_alternation(messages):
                self._valid_alternation += 1

    def _stat_blocks(self) -> dict[str, Any]:
        if not self._conversations:
            return {"messages": None}
        return {
            "messages": MessageStats(
                turns=self._turns.quantiles(),
                content_code_points=self._content_code_points.quantiles(),
                content_utf8_bytes=self._content_utf8_bytes.quantiles(),
                roles_seen=self._roles_seen,
                ends_with_assistant_rate=self._ends_with_assistant / self._conversations,
                valid_alternation_rate=self._valid_alternation / self._conversations,
                has_tool_calls=self._has_tool_calls,
            )
        }


# The one place a dtype becomes a measurement. Naming the measurement rather than constructing it is
# what lets one table serve every caller: each looks the name up and decides whether to build, reuse
# or skip.
_MEASUREMENTS: dict[str, type[ColumnAccumulator]] = {
    "string": StringAccumulator,
    "numeric": NumericAccumulator,
    "bool": BoolAccumulator,
    "messages": MessageAccumulator,
}


def _measurement_for(dtype: str) -> str | None:
    """Which measurement reports a column of this dtype, or None for a dtype with no statistics."""
    if _is_numeric(dtype):
        return "numeric"
    return dtype if dtype in _MEASUREMENTS else None


class RoutedAccumulator(ColumnAccumulator):
    """A column measured by every shape its values take, reporting the one its dtype selects.

    Choosing one measurement *by* dtype needs the dtype, which a declared schema gives up front and
    an inferred one does not: the observed types are unioned over the whole column and a
    disagreement widens to ``json``, so the choice cannot be made while making it still matters.
    Measuring every shape and picking at the end neither reads the data twice nor decides from a
    prefix. It costs no more per value than choosing would have, since a string only ever reaches the
    string state and an int only the numeric.

    A declared column takes the same route and simply knows the answer already. Measurements are
    built on first sight of a value that needs one, so a column of a single type pays for one.
    """

    def __init__(self, name: str, declared: FeatureSchema | None = None) -> None:
        super().__init__()
        self._name = name
        self._declared = declared
        # A declared column was typed before the first batch. Folding its schema again would charge
        # every value for an answer already in hand.
        self._schema = SchemaFold(name) if declared is None else None
        self._measurements: dict[str, ColumnAccumulator] = {}
        # The folded schema, kept once it has been asked for. `finalize` asks three times -- once
        # for the feature itself, then again inside `_stat_blocks` and `vocabulary`, each of which
        # needs the dtype to pick its measurement -- and every ask is a full recursion over the
        # column's nested struct and list schema. Dropped whenever a batch arrives, so the value can
        # never be older than the data behind it.
        self._folded: FeatureSchema | None = None

    def _measurement(self, key: str) -> ColumnAccumulator:
        """The named measurement, built if this is the first value to call for it."""
        accumulator = self._measurements.get(key)
        if accumulator is None:
            accumulator = self._measurements[key] = _MEASUREMENTS[key]()
        return accumulator

    def _observe(self, present: list[Any]) -> None:
        self._folded = None
        declared = self._declared
        if declared is not None:
            key = _measurement_for(declared.dtype)
            if key is not None:
                self._measurement(key)._observe(present)
            return
        # Routed by python type, tagged once. Where a dtype resolves to something measurable every
        # present value is of that type by construction, so a measurement sees what a single chosen
        # accumulator would.
        #
        # Tagging by exact class is an identity test rather than a walk up the MRO, and the
        # partition it leaves behind is the one the schema fold would otherwise rebuild a value at a
        # time. Between them those two were most of the type checks this fold performed.
        strings: list[Any] = []
        ints: list[Any] = []
        floats: list[Any] = []
        bools: list[Any] = []
        lists: list[Any] = []
        dicts: list[Any] = []
        for value in present:
            cls = value.__class__
            if cls is str:
                strings.append(value)
            # `bool` before `int` only to read like the isinstance route below, where the order
            # is load-bearing. Here it cannot be: `True.__class__` is `bool` and never `int`,
            # so an exact class is one place the bool-is-an-int trap does not exist.
            elif cls is bool:
                bools.append(value)
            elif cls is int:
                ints.append(value)
            elif cls is float:
                floats.append(value)
            elif cls is list:
                lists.append(value)
            elif cls is dict:
                dicts.append(value)
            else:
                # A subclass of a builtin, or something stranger, which an exact class cannot place.
                # Give up the partition for the whole batch rather than fold the stragglers in
                # afterwards: `roles_seen` records the order roles were first seen, so a value
                # measured out of turn is a different answer and not merely a slower one. Nothing
                # has been mutated yet, which is what makes abandoning it here clean.
                self._observe_by_isinstance(present)
                return
        if self._schema is not None:
            self._schema.update_partitioned(strings, ints, floats, bools, lists, dicts)
        if strings:
            self._measurement("string")._observe(strings)
        # Two folds rather than one list concatenated back together. An accumulator handed a column
        # in pieces answers as one handed all of it -- the property every batch here already relies
        # on -- so keeping ints and floats apart saves the join and costs nothing.
        if ints:
            self._measurement("numeric")._observe(ints)
        if floats:
            self._measurement("numeric")._observe(floats)
        if bools:
            self._measurement("bool")._observe(bools)
        if lists:
            self._measurement("messages")._observe(lists)

    def _observe_by_isinstance(self, present: list[Any]) -> None:
        """Route by ``isinstance``, for a batch holding a value no exact class could place.

        The same routing as :meth:`_observe`, kept whole rather than merged into it so that the two
        are visibly one decision taken two ways. A ``str`` subclass reaches the string measurement
        here exactly as it did before there was a fast path.
        """
        if self._schema is not None:
            self._schema.update(present)
        strings = [value for value in present if isinstance(value, str)]
        if strings:
            self._measurement("string")._observe(strings)
        numbers = [value for value in present if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if numbers:
            self._measurement("numeric")._observe(numbers)
        bools = [value for value in present if isinstance(value, bool)]
        if bools:
            self._measurement("bool")._observe(bools)
        lists = [value for value in present if isinstance(value, list)]
        if lists:
            self._measurement("messages")._observe(lists)

    def schema_was_capped(self) -> bool:
        """Whether this column's inferred schema declined a nested key."""
        return self._schema is not None and self._schema.was_capped()

    def feature(self) -> FeatureSchema:
        """The column's schema -- declared before the fold, or folded out of the values."""
        if self._declared is not None:
            return self._declared
        if self._folded is None:
            self._folded = (self._schema or SchemaFold(self._name)).finalize()
        return self._folded

    def _stat_blocks(self) -> dict[str, Any]:
        key = _measurement_for(self.feature().dtype)
        # Built here when no value ever called for it, so an all-null string column still reports the
        # empty blocks a chosen StringAccumulator would have.
        return self._measurement(key)._stat_blocks() if key is not None else {}

    def vocabulary(self) -> set[Any] | None:
        key = _measurement_for(self.feature().dtype)
        return self._measurement(key).vocabulary() if key is not None else None


def _is_numeric(dtype: str) -> bool:
    return dtype.startswith(("int", "uint", "float"))


# How finely a length distribution is recorded. Lengths below the slice count get a counter each and
# are exact; above it each octave is cut into this many slices, a fixed *relative* width of 1/32.
# Reporting a bucket's midpoint puts every estimate within ~1.6% whatever the magnitude.
_HISTOGRAM_SLICE_BITS = 5
_HISTOGRAM_SLICES = 1 << _HISTOGRAM_SLICE_BITS  # 32


def _length_bucket(value: int) -> int:
    """The counter a length belongs to."""
    if value < _HISTOGRAM_SLICES:
        return value  # small lengths get a counter each, so they are recorded exactly
    shift = value.bit_length() - 1 - _HISTOGRAM_SLICE_BITS
    return (shift + 1) * _HISTOGRAM_SLICES + ((value >> shift) - _HISTOGRAM_SLICES)


def _bucket_bounds(bucket: int) -> tuple[int, int]:
    """The half-open range of lengths that land in ``bucket``. Inverse of :func:`_length_bucket`."""
    if bucket < _HISTOGRAM_SLICES:
        return bucket, bucket + 1
    index, slice_index = divmod(bucket, _HISTOGRAM_SLICES)
    shift = index - 1
    low = (_HISTOGRAM_SLICES + slice_index) << shift
    return low, low + (1 << shift)


class _LengthHistogram:
    """A per-row length distribution, held as counters rather than as the lengths themselves.

    This is what keeps an accumulator O(1) in rows. Exact quantiles need every length kept and
    sorted; a reservoir bounds that but pays with an RNG, putting a seed in the contract and letting
    two runs over the same bytes disagree. Fixed buckets bound it with neither, at a hard error of
    half a bucket width -- ~2% measured against exact quantiles, and the cheap error to accept, since
    the number is read to pick a sequence budget.

    ``max`` is kept exactly and separately, as the one value here a reader may treat as a hard bound.
    """

    def __init__(self) -> None:
        self._counts: dict[int, int] = {}
        self._rows = 0
        self._max = 0

    def add(self, value: int) -> None:
        bucket = _length_bucket(value)
        self._counts[bucket] = self._counts.get(bucket, 0) + 1
        self._rows += 1
        if value > self._max:
            self._max = value

    def quantiles(self) -> Quantiles:
        return Quantiles(p50=self._at(50), p95=self._at(95), p99=self._at(99), max=self._max)

    def _at(self, percentile: int) -> int:
        """Nearest-rank percentile: the bucket the p-th row falls in, reported at its midpoint.

        The rank is exact -- every row was counted, none sampled -- so only the value is approximate.
        Midpoint rather than the bucket's low edge, which sits systematically under the truth and
        roughly doubles the average error.
        """
        if not self._rows:
            return 0
        target = math.ceil(percentile / 100 * self._rows)
        seen = 0
        for bucket in sorted(self._counts):
            seen += self._counts[bucket]
            if seen >= target:
                low, high = _bucket_bounds(bucket)
                # Never above `max`: a midpoint can overshoot the largest value actually present,
                # and a p99 above the maximum would be nonsense.
                return min((low + high) // 2, self._max)
        return self._max


# --- messages ------------------------------------------------------------------------------------

# Role strings meaning "the turn the model is trained to produce". Matching only the literal
# "assistant" makes a dataset using another convention look like it ended on a user turn, which
# classification reads as prompt-only with no training target.
_ASSISTANT_ROLES = {"assistant", "gpt", "bot", "model", "chatbot", "ai"}

# Distinct role strings a chat column may show before the list stops growing. Fed from row content,
# so without a bound one malformed column holds a string per message -- quadratic, since membership is
# checked against the list. A column with more roles than this is not a chat column.
_MAX_ROLES_SEEN = 64

# Characters of a role string that reach the profile. A role is a short token by nature -- `user`,
# `assistant`, `gpt` -- so this only ever truncates a value that is not one, and the truncation is
# itself the finding: something that long is not a chat role and the column is not a chat column.
# Without it the only unbounded row content in the profile was here, outside the role gate the
# contract calls its one exception, and a mis-shaped export put whole message bodies in `roles_seen`.
_MAX_ROLE_CHARS = 64


def _message_field(message: dict, *names: str) -> Any:
    """The first present, non-null value among ``names``.

    Chat rows spell the same two fields either ``{role, content}`` or ``{from, value}``. Reading with
    a plain ``.get`` default is not enough: parquet materializes *every* declared struct field, so an
    absent field arrives as an explicit None rather than a missing key.
    """
    for name in names:
        value = message.get(name)
        if value is not None:
            return value
    return None


def _role_of(message: dict) -> Any:
    return _message_field(message, "role", "from")


def _is_assistant_role(role: Any) -> bool:
    return isinstance(role, str) and role.lower() in _ASSISTANT_ROLES


def _utf8_len(value: str) -> int:
    """A string's length in UTF-8 bytes.

    The unit a byte-level BPE tokenizer actually consumes, which is why it is measured alongside code
    points rather than derived from them: the two differ by 3x on CJK and not at all on ASCII, so
    neither is recoverable from the other.

    ASCII answers from a flag CPython already carries, so the common case costs no encode; only
    non-ASCII strings pay for one.
    """
    return len(value) if value.isascii() else len(value.encode())


def _content_lengths(content: Any) -> tuple[int, int]:
    """A message's content in code points and UTF-8 bytes, from one walk over it."""
    if isinstance(content, str):
        return len(content), _utf8_len(content)
    if isinstance(content, list):  # VLM content as a list of typed parts
        code_points = 0
        utf8 = 0
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text = part["text"]
                code_points += len(text)
                utf8 += _utf8_len(text)
        return code_points, utf8
    return 0, 0


def _valid_alternation(messages: list) -> bool:
    """True when user/assistant turns alternate (ignoring any leading system turns)."""
    roles = [_role_of(m) for m in messages if isinstance(m, dict) and _role_of(m) != "system"]
    return all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


# --- content probes ------------------------------------------------------------------------------

# Probes run over *every* column, not only role-assigned ones. Gating them on roles would make a
# content signal reachable only through a recognized column name, losing verifiability on a dataset
# whose answer column is called `a`.
_TRANSCRIPT_MARKER = re.compile(r"\n\n(?:Human|Assistant|User):")
_GSM8K_ANSWER = re.compile(r"####\s*-?[\d.,/]+\s*$")
_BOXED_ANSWER = re.compile(r"\\boxed\{")


@dataclass(frozen=True)
class ColumnProbes:
    """What the content probes saw in one column across the sampled rows.

    Internal to the profiler rather than part of the stored contract: these are inputs to
    classification, and promoting them to durable per-column facts is a separate contract change.
    Counts, not rates — the caller divides, so a zero denominator stays visible instead of becoming
    a silent 0.0.
    """

    rows: int  # rows considered for this column
    non_empty: int  # value present and not "" / [] / {} — a usable target of any dtype
    texts: int  # rows that yielded text: a string, or a chat column's final turn
    extractable_answer: int  # of `texts`, how many carry `#### <number>` or `\boxed{`
    transcript_marker: int  # of `texts`, how many embed a Human:/Assistant: transcript


def _probe_text(value: Any) -> str | None:
    """The text a probe reads from one cell: the string itself, or a chat column's final turn.

    The final turn is read through :func:`_message_field`, so the ``{from, value}`` spelling works
    like ``{role, content}``. Both are handled everywhere else in this module and in schema
    derivation; missing it here cost every dataset spelled that way its verifiability.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[-1], dict):
        content = _message_field(value[-1], "content", "value")
        if isinstance(content, str):
            return content
    return None
