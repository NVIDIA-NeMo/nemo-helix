# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for classification: role assignment, format/prompt-form axes, and dataset type."""

from nemo_platform_plugin.files.profile import (
    CategoricalStats,
    ColumnStats,
    FeatureSchema,
    MessageStats,
    Quantiles,
)
from nemo_profiler_plugin.dataset.classify import PrefixPairFold, classify
from nemo_profiler_plugin.dataset.stats import RowFold


def _probes(features, rows):
    fold = RowFold(features)
    fold.update(rows)
    return fold.finalize()[1].probes


def classify_rows(features, stats, rows, **kwargs):
    """Classify from rows the way the pipeline does: fold first, then interpret the folds."""
    prefix = PrefixPairFold()
    prefix.update(rows)
    return classify(features, stats, probes=_probes(features, rows), prefix_pair=prefix.result(), **kwargs)


def _f(name, dtype):
    return FeatureSchema(name=name, dtype=dtype)


def _binary_column():
    """A column observed to hold two distinct values -- what makes an int/bool a real label."""
    return ColumnStats(categorical=CategoricalStats(distinct_count=2))


def _messages_column(ends_with_assistant_rate):
    q = Quantiles(p50=1, p95=1, p99=1, max=1)
    return ColumnStats(
        messages=MessageStats(
            turns=q,
            content_code_points=q,
            content_utf8_bytes=q,
            roles_seen=["user", "assistant"],
            ends_with_assistant_rate=ends_with_assistant_rate,
            valid_alternation_rate=1.0,
        )
    )


# --- roles ---------------------------------------------------------------------------------------


def test_roles_assigned_by_name_and_dtype():
    features = [_f("prompt", "string"), _f("response", "string"), _f("helpfulness", "int64")]
    classify(features, {})
    assert [f.semantic_role for f in features] == ["prompt", "completion", "score"]


def test_dtype_gate_rejects_mismatched_aliases():
    # "label" only counts as a label when boolean; a string column named "messages" is not messages.
    features = [_f("label", "string"), _f("messages", "string")]
    classify(features, {})
    assert all(f.semantic_role is None for f in features)


def test_physical_name_differs_from_role():
    features = [_f("response", "string")]
    classify(features, {})
    assert features[0].semantic_role == "completion"


# --- format axis ---------------------------------------------------------------------------------


def test_format_standard_conversational_and_mixed():
    assert classify([_f("prompt", "string"), _f("completion", "string")], {}).format == "standard"
    assert classify([_f("prompt", "messages"), _f("completion", "messages")], {}).format == "conversational"
    mixed = [_f("prompt", "string"), _f("chosen", "messages"), _f("rejected", "messages")]
    assert classify(mixed, {}).format == "mixed"


# --- dataset type + prompt form ------------------------------------------------------------------


def test_prompt_completion_with_explicit_prompt():
    result = classify([_f("prompt", "string"), _f("completion", "string")], {})
    assert result.primary == "prompt_completion"
    assert result.prompt_form == "explicit"


def test_preference_pair_is_implicit_without_a_prompt():
    result = classify([_f("chosen", "string"), _f("rejected", "string")], {})
    assert result.primary == "preference_pair"
    assert result.prompt_form == "implicit"


def test_scored_response_beats_prompt_completion():
    features = [
        _f("prompt", "string"),
        _f("response", "string"),
        _f("helpfulness", "int64"),
        _f("correctness", "int64"),
    ]
    assert classify(features, {}).primary == "scored_response"


def test_unpaired_preference_accepts_a_boolean_label():
    features = [_f("prompt", "string"), _f("completion", "string"), _f("label", "bool")]
    assert classify(features, {}).primary == "unpaired_preference"


def test_unpaired_preference_accepts_a_binary_integer_label():
    # 0/1 is the usual on-disk encoding; requiring a bool made unpaired_preference unreachable for
    # most real datasets.
    features = [_f("prompt", "string"), _f("completion", "string"), _f("label", "int64")]
    stats = {"label": ColumnStats(categorical=CategoricalStats(distinct_count=2))}
    assert classify(features, stats).primary == "unpaired_preference"
    assert features[2].semantic_role == "label"


def test_unpaired_preference_accepts_a_binary_string_label():
    # `safe`/`unsafe` is as common an encoding as 0/1, and requiring a number left every
    # content-safety set unroled -- so its values were never quoted either, because quoting is gated
    # on the role. The encoding is not the question; the number of distinct values is.
    features = [_f("prompt", "string"), _f("completion", "string"), _f("label", "string")]
    stats = {"label": ColumnStats(categorical=CategoricalStats(distinct_count=2))}
    assert classify(features, stats).primary == "unpaired_preference"
    assert features[2].semantic_role == "label"


def test_an_all_null_label_column_is_not_a_preference_label():
    # An empty column decided the dataset type: `distinct_count <= 2` is true of zero, so a `label`
    # column holding nothing took the role and carried the partition to `unpaired_preference`.
    features = [_f("prompt", "string"), _f("completion", "string"), _f("label", "string")]
    stats = {"label": ColumnStats(null_rate=1.0, categorical=CategoricalStats(distinct_count=0))}
    result = classify(features, stats)
    assert features[2].semantic_role is None
    assert result.candidates == ["prompt_completion"]


def test_a_single_class_label_column_is_still_a_preference_label():
    # Not the same judgment as zero. A shard whose labels are all one class -- or a read that
    # stopped before the second class appeared -- is still a label column, and classifying it
    # differently from its sibling over class balance would make the type depend on the split.
    features = [_f("prompt", "string"), _f("completion", "string"), _f("label", "string")]
    stats = {"label": ColumnStats(categorical=CategoricalStats(distinct_count=1))}
    result = classify(features, stats)
    assert features[2].semantic_role == "label"
    assert result.candidates == ["unpaired_preference", "prompt_completion"]


def test_wide_string_label_is_not_a_preference_label():
    # Symmetric with the integer rule: three classes is a class label, not a binary preference.
    features = [_f("prompt", "string"), _f("completion", "string"), _f("label", "string")]
    stats = {"label": ColumnStats(categorical=CategoricalStats(distinct_count=3))}
    assert classify(features, stats).primary == "prompt_completion"
    assert features[2].semantic_role is None


def test_wide_integer_label_is_not_a_preference_label():
    # A multi-class index or a rating is a different claim from a binary preference.
    features = [_f("prompt", "string"), _f("completion", "string"), _f("label", "int64")]
    stats = {"label": ColumnStats(categorical=CategoricalStats(distinct_count=7))}
    assert classify(features, stats).primary == "prompt_completion"
    assert features[2].semantic_role is None


# --- rank ------------------------------------------------------------------------------------


def test_rank_needs_something_to_rank():
    # A lone numeric column named "rank" used to short-circuit every more specific structure.
    features = [_f("rank", "int64")]
    assert classify(features, {}).primary is None


def test_rank_does_not_override_a_preference_pair():
    features = [_f("chosen", "string"), _f("rejected", "string"), _f("rank", "int64")]
    assert classify(features, {}).primary == "preference_pair"


def test_rank_does_not_override_scored_responses():
    features = [_f("prompt", "string"), _f("response", "string"), _f("helpfulness", "int64"), _f("rank", "int64")]
    assert classify(features, {}).primary == "scored_response"


def test_rank_alongside_a_completion_is_ranked_responses():
    features = [_f("prompt", "string"), _f("completion", "string"), _f("rank", "int64")]
    assert classify(features, {}).primary == "ranked_responses"


def test_messages_ending_on_assistant_is_messages_type():
    result = classify([_f("messages", "messages")], {"messages": _messages_column(1.0)})
    assert result.primary == "messages"
    assert result.prompt_form == "n/a"


def test_messages_ending_on_user_is_prompt_only():
    result = classify([_f("messages", "messages")], {"messages": _messages_column(0.0)})
    assert result.primary == "prompt_only"


def test_single_text_column_is_text():
    assert classify([_f("text", "string")], {}).primary == "text"


def test_unrecognized_columns_are_unknown():
    result = classify([_f("foo", "int64"), _f("bar", "int64")], {})
    assert result.primary is None
    assert result.prompt_form is None  # no axes asserted for unknown data


# --- evidence ------------------------------------------------------------------------------------


def test_classification_records_evidence():
    result = classify([_f("prompt", "string"), _f("completion", "string")], {})
    assert {e.kind for e in result.evidence} >= {"column_name", "column_dtype"}


# --- verifiability + content probes --------------------------------------------------------------


def test_verifiability_extractable_gsm8k_answer():
    features = [_f("problem", "string"), _f("solution", "string")]
    rows = [{"problem": "q", "solution": "steps #### 18"}, {"problem": "q", "solution": "no final answer"}]
    result = classify_rows(features, {}, rows)
    assert result.verifiability.method == "extractable_final_answer"
    assert result.verifiability.coverage == 0.5


def test_verifiability_boxed_answer():
    features = [_f("prompt", "string"), _f("completion", "string")]
    result = classify_rows(features, {}, [{"prompt": "q", "completion": r"reasoning \boxed{42}"}])
    assert result.verifiability.method == "extractable_final_answer"
    assert result.verifiability.coverage == 1.0


def test_verifiability_ground_truth_column_coverage():
    features = [_f("prompt", "string"), _f("ground_truth", "string")]
    rows = [{"prompt": "q", "ground_truth": "42"}, {"prompt": "q", "ground_truth": None}]
    result = classify_rows(features, {}, rows)
    assert result.verifiability.method == "ground_truth_column"
    assert result.verifiability.coverage == 0.5


def test_no_verifiability_without_a_target():
    features = [_f("prompt", "string"), _f("completion", "string")]
    result = classify_rows(features, {}, [{"prompt": "q", "completion": "just prose, no answer"}])
    assert result.verifiability is None


def test_verifiability_ignores_below_threshold_extractable_noise():
    # One coincidental "#### <n>" in a large sample is noise, not a verifiable dataset (kto-mix-14k).
    features = [_f("prompt", "string"), _f("completion", "string")]
    rows = [{"prompt": "q", "completion": "just prose"} for _ in range(100)]
    rows[0]["completion"] = "the answer is #### 7"  # 1/100 = 1% < 5% floor
    assert classify_rows(features, {}, rows).verifiability is None


def test_verifiability_asserted_above_coverage_floor():
    features = [_f("prompt", "string"), _f("completion", "string")]
    rows = [{"prompt": "q", "completion": "just prose"} for _ in range(10)]
    for row in rows[:2]:
        row["completion"] = "answer #### 7"  # 2/10 = 20% >= 5% floor
    result = classify_rows(features, {}, rows)
    assert result.verifiability.method == "extractable_final_answer"
    assert result.verifiability.coverage == 0.2


def test_sparse_ground_truth_falls_through_to_extractable_answer():
    # A ground_truth column present in too few rows must not mask a strong extractable-answer signal.
    features = [_f("completion", "string"), _f("ground_truth", "string")]
    rows = [{"completion": "reasoning #### 5", "ground_truth": None} for _ in range(100)]
    rows[0]["ground_truth"] = "5"  # 1/100 ground_truth coverage -> below floor, must fall through
    result = classify_rows(features, {}, rows)
    assert result.verifiability.method == "extractable_final_answer"
    assert result.verifiability.coverage == 1.0


def test_a_sparse_column_does_not_outrank_the_real_answer_column():
    """Coverage is a fraction of the column's rows, not of the rows that happened to hold text.

    Scored the other way, a column present in one row out of a thousand rates 1.0 and outranks the
    genuine answer column at 0.8 -- and the profile then reports that 1.0 as the coverage, which the
    contract says a consumer may read literally. It would point a verifier at the wrong column.
    """
    features = [_f("q", "string"), _f("a", "string"), _f("note", "string")]
    rows = [
        {"q": f"question {i}", "a": (f"reasoning #### {i}" if i % 10 < 8 else "no marker"), "note": None}
        for i in range(1000)
    ]
    rows[0]["note"] = "\\boxed{7}"  # the only row this column is present in at all

    result = classify_rows(features, {}, rows)

    assert result.verifiability.method == "extractable_final_answer"
    assert result.verifiability.coverage == 0.8
    assert "'a'" in result.verifiability.evidence[0].detail


def test_a_column_present_in_one_row_is_not_a_verification_target():
    """The same ranking read on its own: one hit in a thousand rows is below the coverage floor."""
    features = [_f("q", "string"), _f("note", "string")]
    rows = [{"q": f"question {i}", "note": None} for i in range(1000)]
    rows[0]["note"] = "\\boxed{7}"

    assert classify_rows(features, {}, rows).verifiability is None


def test_implicit_prompt_evidence_from_embedded_transcript():
    features = [_f("chosen", "string"), _f("rejected", "string")]
    rows = [{"chosen": "\n\nHuman: hi\n\nAssistant: hello", "rejected": "\n\nHuman: hi\n\nAssistant: hey"}]
    result = classify_rows(features, {}, rows)
    assert result.prompt_form == "implicit"
    assert any(e.kind == "content_probe" for e in result.evidence)


def test_ground_truth_may_be_a_container_dtype():
    # test_cases (list) and verification_info (struct) are verification targets, not free text,
    # so the text-only dtype gate must not drop them.
    features = [_f("prompt", "string"), _f("test_cases", "list"), _f("verification_info", "struct")]
    classify(features, {})
    assert features[1].semantic_role == "ground_truth"
    assert features[2].semantic_role == "ground_truth"


def test_container_ground_truth_drives_verifiability():
    features = [_f("prompt", "string"), _f("test_cases", "list")]
    rows = [{"prompt": "q", "test_cases": [{"in": "1", "out": "2"}]}, {"prompt": "q2", "test_cases": []}]
    result = classify_rows(features, {}, rows)
    assert result.verifiability.method == "ground_truth_column"
    assert result.verifiability.coverage == 0.5  # the empty test_cases list is not a usable target


def test_bare_scalar_ground_truth_alias_is_still_rejected():
    # A numeric column named "ground_truth" is far more likely a label/score than a target.
    features = [_f("ground_truth", "int64")]
    classify(features, {})
    assert features[0].semantic_role is None


def test_verifiability_survives_an_unrecognized_column_name():
    # Gating the probes on roles made a content signal reachable only through a recognized column
    # name: the `#### <n>` markers were in the data and the regex would have matched them, but
    # nothing knew where to look. The finding must name the column it came from.
    features = [_f("q", "string"), _f("a", "string")]
    rows = [{"q": "what is 2+2?", "a": f"add them #### {i}"} for i in range(10)]
    result = classify_rows(features, {}, rows)

    assert {f.semantic_role for f in features} == {None}  # still unroled, and honest about it
    assert result.primary is None
    assert result.verifiability.method == "extractable_final_answer"
    assert result.verifiability.coverage == 1.0
    assert "'a'" in result.verifiability.evidence[0].detail


def test_a_named_completion_still_decides_where_to_look():
    # Roles order the interpretation even though they no longer gate it: a column *known* to be the
    # completion is a better answer than one that merely looks like it.
    features = [_f("completion", "string"), _f("notes", "string")]
    rows = [{"completion": "just prose", "notes": "scratch #### 9"} for _ in range(10)]
    assert classify_rows(features, {}, rows).verifiability is None


def test_verifiability_reads_a_from_value_conversational_completion():
    features = [_f("prompt", "string"), _f("completion", "messages")]
    rows = [{"prompt": "q", "completion": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "#### 4"}]}]
    result = classify_rows(features, {}, rows)
    assert result.verifiability.method == "extractable_final_answer"
    assert result.verifiability.coverage == 1.0


def test_classification_without_probes_claims_nothing_rather_than_guessing():
    # classify() used to derive probes from rows when it was handed none. It no longer sees rows at
    # all, so absent probes must read as "nothing was measured" -- never as "nothing is there".
    rows = [{"prompt": "q", "completion": f"steps #### {i}"} for i in range(10)]
    features = [_f("prompt", "string"), _f("completion", "string")]

    blind = classify(features, {})
    assert blind.verifiability is None

    measured = classify([_f("prompt", "string"), _f("completion", "string")], {}, probes=_probes(features, rows))
    assert measured.verifiability.method == "extractable_final_answer"
    assert measured.verifiability.coverage == 1.0


# --- thresholds ----------------------------------------------------------------------------------
#
# Each of these pins a constant from *both* sides, at the value and one step off it. The direction
# of every comparison here was already covered; none of the values were, so any of them could move
# anywhere inside its range without a test noticing -- and each one decides something a consumer
# acts on.


def test_a_chat_needs_half_its_conversations_to_end_on_an_assistant_turn():
    # Under the rate there is no training target and the set is prompt_only. The threshold is 0.5
    # exactly, and it was free to sit anywhere in (0, 1): at 0.9 a set where 60% of conversations
    # end on an assistant turn -- an ordinary SFT set -- reports as having nothing to train on.
    features = [_f("messages", "messages")]
    at = classify(features, {"messages": _messages_column(0.5)})
    below = classify([_f("messages", "messages")], {"messages": _messages_column(0.49)})

    assert "messages" in at.candidates and "prompt_only" not in at.candidates
    assert "prompt_only" in below.candidates and "messages" not in below.candidates


def test_a_verification_target_must_cover_a_twentieth_of_the_rows():
    # `_MIN_VERIFIABILITY_COVERAGE` is 0.05 and the comparison is `>=`, so a column present in
    # exactly a twentieth of the rows clears it. One step under and the signal is noise.
    for present, expected in ((5, "ground_truth_column"), (4, None)):
        rows = [{"prompt": "q", "ground_truth": "a" if i < present else ""} for i in range(100)]
        # Features are rebuilt per case: `classify` assigns `semantic_role` in place.
        features = [_f("prompt", "string"), _f("ground_truth", "string")]
        result = classify_rows(features, {}, rows)
        method = result.verifiability.method if result.verifiability else None
        assert method == expected, f"{present}/100 -> {method}"

    # The extractable-answer arm has a floor of its own, on a different variable, and the two are
    # separate comparisons against the same constant.
    for present, expected in ((5, "extractable_final_answer"), (4, None)):
        rows = [{"out": f"reasoning #### {i}" if i < present else "reasoning, no answer"} for i in range(100)]
        result = classify_rows([_f("out", "string")], {}, rows)
        method = result.verifiability.method if result.verifiability else None
        assert method == expected, f"extractable {present}/100 -> {method}"


def test_a_shared_prefix_has_to_be_long_enough_not_to_be_a_turn_of_phrase():
    # `_EMBEDDED_PROMPT_PREFIX_CHARS` is 16, compared with `>=`. It was unconstrained from below,
    # where the bug is: at 3, two answers sharing only "The " count as an embedded prompt, and the
    # profile asserts a finding about the dataset that is not true of it.
    for shared_chars, expected_pairs in ((16, 1), (15, 0)):
        prefix = "x" * shared_chars
        fold = PrefixPairFold()
        fold.update([{"chosen": prefix + "aaaaaaaaaa", "rejected": prefix + "bbbbbbbbbb"}])
        assert fold.result().pairs == 1
        assert fold.result().shared == expected_pairs, shared_chars


def test_an_embedded_prompt_is_claimed_at_half_the_pairs_and_not_below():
    # The rate that turns counted pairs into a stated finding, `>= 0.5`.
    from nemo_profiler_plugin.dataset.classify import PrefixPair, _implicit_prompt_evidence

    features = [_f("chosen", "string"), _f("rejected", "string")]
    classify(features, {})  # assigns the chosen/rejected roles the evidence function reads
    at = _implicit_prompt_evidence(features, {}, PrefixPair(pairs=10, shared=5))
    below = _implicit_prompt_evidence(features, {}, PrefixPair(pairs=10, shared=4))

    assert at is not None and "prompt is embedded" in at.detail
    assert below is None


def test_the_first_column_wins_a_coverage_tie():
    # `coverage > best_coverage` keeps the first of equals. With `>=` the reported column flips to
    # the last, which changes which name a consumer is told to read without changing the number
    # beside it -- the kind of difference that survives a review. Neither column is in the alias
    # table, so there is no named completion and every column is searched.
    rows = [{"aaa_out": "the answer is #### 42", "zzz_out": "so #### 42"} for _ in range(10)]
    features = [_f("aaa_out", "string"), _f("zzz_out", "string")]

    result = classify_rows(features, {}, rows)

    assert result.verifiability is not None
    assert result.verifiability.method == "extractable_final_answer"
    assert result.verifiability.coverage == 1.0  # both columns tie at 1.0
    assert "aaa_out" in result.verifiability.evidence[0].detail


# --- candidates ------------------------------------------------------------------------------------


def test_the_prefix_probe_reads_capitalised_chosen_and_rejected_columns():
    # `_role_for` lowercases the column name, so `Chosen`/`Rejected` took both roles and reached
    # `_implicit_prompt_evidence` -- but the probe looked the columns up with the row's own casing,
    # found no pairs, and the "prompt is embedded" finding was dropped. Silently, since the roles
    # themselves landed and the classification looked complete.
    shared = "The capital of France is " * 4
    for chosen_name, rejected_name in (("chosen", "rejected"), ("Chosen", "Rejected")):
        fold = PrefixPairFold()
        fold.update([{chosen_name: shared + "Paris", rejected_name: shared + "Lyon"}])
        assert fold.result().pairs == 1, chosen_name
        assert fold.result().shared == 1, chosen_name

        features = [_f(chosen_name, "string"), _f(rejected_name, "string")]
        result = classify(features, {}, prefix_pair=fold.result())
        assert [f.semantic_role for f in features] == ["chosen", "rejected"]
        assert any("prompt is embedded" in e.detail for e in result.evidence), chosen_name


def test_two_shards_spelling_the_pair_differently_are_both_counted():
    # Resolving the columns once and reusing the spelling made a column name a fact about the
    # *partition*, which it only is when the partition came from one export. Two shards written by
    # different tools put `chosen` and `Chosen` in one directory, and the pinned spelling dropped
    # every row using the other -- turning 3 shared prefixes out of 303 pairs into 3 out of 3, so
    # the profile asserted an embedded prompt in "100% of pairs". Losing evidence would have been
    # bad; asserting the opposite of the truth is worse.
    prompt = "Explain the causes of the war in detail. "
    embedded = [{"chosen": prompt + "yes", "rejected": prompt + "no"} for _ in range(3)]
    plain = [{"Chosen": f"answer {i} alpha", "Rejected": f"reply {i} beta"} for i in range(300)]

    for title, batches in (("one batch", [embedded + plain]), ("two shards", [embedded, plain])):
        fold = PrefixPairFold()
        for batch in batches:
            fold.update(batch)
        assert fold.result().pairs == 303, title
        assert fold.result().shared == 3, title

    features = [_f("chosen", "string"), _f("rejected", "string")]
    fold = PrefixPairFold()
    fold.update(embedded + plain)
    result = classify(features, {}, prefix_pair=fold.result())
    assert not any("prompt is embedded" in e.detail for e in result.evidence)  # 1%, not 100%


def test_the_pair_is_found_after_the_first_batch_has_gone_by():
    # The search is bounded by rows rather than by "the first batch" because line-delimited rows are
    # ragged: an optional column can first appear well into a file. That was the stated design and
    # it never worked, because the budget was set to exactly one batch of rows.
    from nemo_profiler_plugin.dataset.classify import _RESOLVE_ROW_BUDGET
    from nemo_profiler_plugin.dataset.readers.jsonl import _BATCH_ROWS

    assert _RESOLVE_ROW_BUDGET > _BATCH_ROWS, "a budget of one batch is not a budget"

    prompt = "Explain the causes of the war in detail. "
    for first_at, expected in ((_BATCH_ROWS + 1, 5), (_RESOLVE_ROW_BUDGET, 0)):
        fold = PrefixPairFold()
        sent = 0
        while sent < first_at:
            take = min(_BATCH_ROWS, first_at - sent)
            fold.update([{"question": "q"} for _ in range(take)])
            sent += take
        fold.update([{"chosen": prompt + "a", "rejected": prompt + "b"} for _ in range(5)])
        # Found when it arrives inside the budget, and deliberately not once the budget is spent.
        assert fold.result().pairs == expected, first_at


def test_candidates_list_every_structure_the_roles_satisfy():
    # prompt + completion + score + label is genuinely both scored_response and unpaired_preference.
    # Reporting only the first made rule order an invisible tie-break.
    features = [_f("prompt", "string"), _f("completion", "string"), _f("score", "float64"), _f("label", "bool")]
    result = classify(features, {"label": _binary_column()})

    assert result.candidates == ["scored_response", "unpaired_preference", "prompt_completion"]
    assert result.primary == "scored_response"  # the head, derived from the list and never stored beside it


def test_candidates_collapse_to_one_when_the_structure_is_unambiguous():
    result = classify([_f("prompt", "string"), _f("completion", "string")], {})
    assert result.candidates == ["prompt_completion"]


def test_nothing_recognised_reports_no_candidates():
    # Absence of a type, not a type named "unknown". The classifier ran and matched no structure.
    result = classify([_f("foo", "int64"), _f("bar", "int64")], {})
    assert result.candidates == []
    assert result.primary is None


def test_prompt_only_is_not_claimed_alongside_a_training_target():
    # Collecting candidates rather than returning early risks a prompt+completion set also claiming
    # prompt_only, which asserts the opposite of what the data holds.
    result = classify([_f("prompt", "string"), _f("completion", "string")], {})
    assert "prompt_only" not in result.candidates


# --- declared roles (hints) ------------------------------------------------------------------------


def test_a_hint_names_a_column_the_alias_table_does_not_know():
    features = [_f("q", "string"), _f("a", "string")]
    result = classify(features, {}, column_roles={"q": "prompt", "a": "completion"})

    assert [(f.semantic_role, f.semantic_role_source) for f in features] == [
        ("prompt", "declared"),
        ("completion", "declared"),
    ]
    assert result.primary == "prompt_completion"


def test_a_hint_takes_precedence_over_the_name_alias():
    # The caller knows their schema; the table is ~35 English names.
    features = [_f("prompt", "string")]
    classify(features, {}, column_roles={"prompt": "context"})
    assert features[0].semantic_role == "context"
    assert features[0].semantic_role_source == "declared"


def test_a_hint_the_dtype_cannot_support_is_rejected_loudly():
    # A hint says which column, not what the data is. Accepting it unconditionally would let one
    # typo produce a nonsense classification, and silence is what made the table's misses costly.
    features = [_f("n", "int64")]
    result = classify(features, {}, column_roles={"n": "prompt"})

    assert features[0].semantic_role is None
    rejections = [e for e in result.evidence if e.kind == "user_hint"]
    assert len(rejections) == 1
    assert "n -> prompt" in rejections[0].detail and "int64" in rejections[0].detail


def test_a_hint_naming_a_role_that_does_not_exist_is_rejected():
    # The dtype gate has no constraint to apply to a role it does not recognize, so it used to wave
    # a typo through: `semantic_role="prmpt"` reached the profile as a `declared` role, evidence
    # claimed the columns matched, and nothing downstream read it -- so the dataset classified as
    # `unknown` with the profile asserting the opposite.
    features = [_f("q", "string")]
    result = classify(features, {}, column_roles={"q": "prmpt"})

    assert features[0].semantic_role is None
    assert features[0].semantic_role_source is None
    rejections = [e for e in result.evidence if e.kind == "user_hint"]
    assert len(rejections) == 1
    assert "q -> prmpt" in rejections[0].detail
    assert not any("prmpt" in e.detail for e in result.evidence if e.kind == "column_name")


def test_a_rejected_hint_falls_back_to_detection():
    # `answer` is a known alias; a bad hint on it must not cost the role the table would have found.
    features = [_f("answer", "string")]
    classify(features, {}, column_roles={"answer": "messages"})  # messages needs the messages dtype
    assert features[0].semantic_role == "completion"
    assert features[0].semantic_role_source == "detected"


def test_detected_roles_are_marked_as_detected():
    features = [_f("prompt", "string")]
    classify(features, {})
    assert features[0].semantic_role_source == "detected"
