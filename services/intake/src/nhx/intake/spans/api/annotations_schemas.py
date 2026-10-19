# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for post-hoc annotations on spans and sessions.

Both read and write surfaces are kind-discriminated. Each annotation kind has
its own typed shape so clients and SDKs see explicit per-kind fields rather
than a flat catch-all with runtime validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, cast

from nhx.common.entities.values import DatetimeFilter, Filter
from nhx.intake.spans.domain import Annotation as DomainAnnotation
from nhx.intake.spans.domain import AnnotationKind
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class NumericFilter(Filter):
    """Range filter for numeric annotation values.

    At least one of `$gte` or `$lte` must be supplied — an empty `{}` is not a
    meaningful filter and is rejected.
    """

    gte: float | None = Field(
        default=None,
        alias="$gte",
        serialization_alias="$gte",
        description="Include only values greater than or equal to this number.",
    )
    lte: float | None = Field(
        default=None,
        alias="$lte",
        serialization_alias="$lte",
        description="Include only values less than or equal to this number.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
        # Surface the at-least-one-bound rule in the generated OpenAPI schema.
        json_schema_extra={"minProperties": 1},
    )

    @model_validator(mode="after")
    def _require_at_least_one_bound(self) -> Self:
        if self.gte is None and self.lte is None:
            raise ValueError("NumericFilter requires at least one of `$gte` or `$lte`")
        return self


class AnnotationSortField(StrEnum):
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"


class AnnotationFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span_id: str | None = Field(default=None, description="Return only annotations attached to this span.")
    session_id: str | None = Field(default=None, description="Return only annotations attached to this session.")
    kind: AnnotationKind | None = Field(
        default=None,
        description="Return only annotations of this kind (`feedback`, `note`, `label`, or `metadata`).",
    )
    name: str | None = Field(
        default=None,
        description="Return only `label` annotations with this `name` (e.g., `severity`, `helpfulness`).",
    )
    value_text: str | None = Field(
        default=None,
        description=(
            "Return only annotations with this text value. For `feedback` annotations this is"
            " `positive` or `negative`; for `label` annotations with `value_type=text` this is the"
            " label's value."
        ),
    )
    value_numeric: NumericFilter | None = Field(
        default=None,
        description=(
            "Return only `label` annotations whose numeric value falls within the given range."
            " Applies to labels with `value_type=numeric`."
        ),
    )
    created_by: str | None = Field(
        default=None,
        description="Return only annotations created by this user.",
    )
    created_at: DatetimeFilter | None = Field(
        default=None,
        description="Return only annotations created within the given time range.",
    )


# ---------------------------------------------------------------------------
# Write shape — POST /annotations
# ---------------------------------------------------------------------------


class _AnnotationInputBase(BaseModel):
    """Target fields shared by all annotation input variants."""

    model_config = ConfigDict(extra="forbid")

    span_id: str | None = Field(
        default=None,
        description=(
            "Id of the span this annotation applies to. Omit to annotate the whole session instead of a specific span."
        ),
    )
    session_id: str = Field(
        description="Id of the session this annotation belongs to. Always required.",
    )


class FeedbackAnnotationInput(_AnnotationInputBase):
    """Thumbs-up / thumbs-down feedback on a span or session."""

    kind: Literal["feedback"] = Field(description="Discriminator. Always `feedback` for this variant.")
    value: Literal["positive", "negative"] = Field(description="Sentiment of the feedback.")


class NoteAnnotationInput(_AnnotationInputBase):
    """Free-text note attached to a span or session."""

    kind: Literal["note"] = Field(description="Discriminator. Always `note` for this variant.")
    text: str = Field(
        min_length=1,
        max_length=10_000,
        description="The note content. 1 to 10,000 characters.",
    )


class MetadataAnnotationInput(_AnnotationInputBase):
    """Structured key/value metadata attached to a span or session."""

    kind: Literal["metadata"] = Field(description="Discriminator. Always `metadata` for this variant.")
    metadata: dict[str, Any] = Field(
        min_length=1,
        description="Arbitrary key/value pairs. Must contain at least one entry.",
    )


class LabelAnnotationInput(_AnnotationInputBase):
    """Categorical or numeric label attached to a span or session.

    Use `value_type=text` for tag-style labels (e.g., `regression`, `needs-review`) and
    `value_type=numeric` for scored labels (e.g., a 1-5 helpfulness rating). Numeric labels
    must include a `name` to identify what the score measures.
    """

    kind: Literal["label"] = Field(description="Discriminator. Always `label` for this variant.")
    value_type: Literal["text", "numeric"] = Field(
        description="Whether `value` should be interpreted as text (`text`) or a number (`numeric`).",
    )
    value: str | float = Field(
        description=(
            "The label's value. Must be a string when `value_type=text` and a number when `value_type=numeric`."
        ),
    )
    name: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "Name identifying what the label measures (e.g., `severity`, `helpfulness`)."
            " Optional for text labels; required for numeric labels."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.value_type == "numeric":
            if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
                raise ValueError("value_type=numeric requires a numeric `value`")
            if self.name is None:
                raise ValueError("value_type=numeric requires `name`")
        elif not isinstance(self.value, str):
            raise ValueError("value_type=text requires a string `value`")
        return self


class AnnotationInput(
    RootModel[
        Annotated[
            FeedbackAnnotationInput | NoteAnnotationInput | MetadataAnnotationInput | LabelAnnotationInput,
            Field(discriminator="kind"),
        ]
    ]
):
    """Discriminated annotation create body. The shape varies by `kind`."""


# ---------------------------------------------------------------------------
# Read shape — GET responses
# ---------------------------------------------------------------------------


class _AnnotationReadBase(BaseModel):
    """Server-set fields shared by all annotation read variants."""

    annotation_id: str
    workspace: str
    span_id: str | None = Field(
        default=None,
        description="Id of the span this annotation applies to, or omitted for session-level annotations.",
    )
    session_id: str
    created_by: str | None = None
    created_at: datetime
    ingested_at: datetime


class FeedbackAnnotation(_AnnotationReadBase):
    """Thumbs-up / thumbs-down feedback on a span or session."""

    kind: Literal["feedback"] = Field(description="Discriminator. Always `feedback` for this variant.")
    value: Literal["positive", "negative"] = Field(description="Sentiment of the feedback.")


class NoteAnnotation(_AnnotationReadBase):
    """Free-text note attached to a span or session."""

    kind: Literal["note"] = Field(description="Discriminator. Always `note` for this variant.")
    text: str = Field(description="The note content.")


class MetadataAnnotation(_AnnotationReadBase):
    """Structured key/value metadata attached to a span or session."""

    kind: Literal["metadata"] = Field(description="Discriminator. Always `metadata` for this variant.")
    metadata: dict[str, Any] = Field(description="The metadata key/value pairs.")


class LabelAnnotation(_AnnotationReadBase):
    """Categorical or numeric label attached to a span or session."""

    kind: Literal["label"] = Field(description="Discriminator. Always `label` for this variant.")
    value_type: Literal["text", "numeric"] = Field(
        description="Whether `value` is a text label (`text`) or a number (`numeric`).",
    )
    value: str | float = Field(
        description="The label's value. A string when `value_type=text`, a number when `value_type=numeric`.",
    )
    name: str | None = Field(
        default=None,
        description="Name identifying what the label measures, when set.",
    )


class Annotation(
    RootModel[
        Annotated[
            FeedbackAnnotation | NoteAnnotation | MetadataAnnotation | LabelAnnotation,
            Field(discriminator="kind"),
        ]
    ]
):
    """Discriminated annotation read response. The shape varies by `kind`."""


# ---------------------------------------------------------------------------
# Domain <-> wire translation
# ---------------------------------------------------------------------------


def annotation_from_domain(domain: DomainAnnotation) -> Annotation:
    """Translate flat-storage DomainAnnotation into the typed read variant for its kind."""

    common: dict[str, Any] = {
        "annotation_id": domain.annotation_id,
        "workspace": domain.workspace,
        "span_id": domain.span_id,
        "session_id": domain.session_id,
        "created_by": domain.created_by,
        "created_at": domain.created_at,
        "ingested_at": domain.ingested_at,
    }
    variant: FeedbackAnnotation | NoteAnnotation | MetadataAnnotation | LabelAnnotation
    if domain.kind == AnnotationKind.FEEDBACK:
        if domain.value_text not in {"positive", "negative"}:
            raise ValueError(f"feedback row has invalid value_text={domain.value_text!r}")
        variant = FeedbackAnnotation(
            kind="feedback",
            value=cast(Literal["positive", "negative"], domain.value_text),
            **common,
        )
    elif domain.kind == AnnotationKind.NOTE:
        if domain.text is None:
            raise ValueError("note row missing `text`")
        variant = NoteAnnotation(kind="note", text=domain.text, **common)
    elif domain.kind == AnnotationKind.METADATA:
        if not domain.metadata:
            raise ValueError("metadata row missing `metadata`")
        variant = MetadataAnnotation(kind="metadata", metadata=domain.metadata, **common)
    elif domain.kind == AnnotationKind.LABEL:
        if domain.value_numeric is not None:
            variant = LabelAnnotation(
                kind="label",
                value_type="numeric",
                value=domain.value_numeric,
                name=domain.name,
                **common,
            )
        elif domain.value_text is not None:
            variant = LabelAnnotation(
                kind="label",
                value_type="text",
                value=domain.value_text,
                name=domain.name,
                **common,
            )
        else:
            raise ValueError("label row missing both `value_text` and `value_numeric`")
    else:
        raise ValueError(f"unknown annotation kind in storage: {domain.kind!r}")
    return Annotation(root=variant)


def annotation_input_to_domain_fields(
    body: FeedbackAnnotationInput | NoteAnnotationInput | MetadataAnnotationInput | LabelAnnotationInput,
) -> dict[str, Any]:
    """Flatten a typed input variant into the columnar shape used by DomainAnnotation."""

    base: dict[str, Any] = {
        "span_id": body.span_id,
        "session_id": body.session_id,
        "name": None,
        "value_text": None,
        "value_numeric": None,
        "text": None,
        "metadata": None,
    }
    if isinstance(body, FeedbackAnnotationInput):
        return {**base, "kind": AnnotationKind.FEEDBACK, "value_text": body.value}
    if isinstance(body, NoteAnnotationInput):
        return {**base, "kind": AnnotationKind.NOTE, "text": body.text}
    if isinstance(body, MetadataAnnotationInput):
        return {**base, "kind": AnnotationKind.METADATA, "metadata": body.metadata}
    if isinstance(body, LabelAnnotationInput):
        if body.value_type == "numeric":
            return {
                **base,
                "kind": AnnotationKind.LABEL,
                "value_numeric": float(body.value),
                "name": body.name,
            }
        return {
            **base,
            "kind": AnnotationKind.LABEL,
            "value_text": str(body.value),
            "name": body.name,
        }
    raise ValueError(f"unknown input type: {type(body).__name__}")
