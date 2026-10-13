# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse storage bags for normalized span attributes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Self

from nhx.intake.spans.span_attribute_catalog import (
    COST_SCALE,
    KNOWN_BAG_KEYS,
    SOURCE_ONLY_KEYS,
    SOURCE_ONLY_PREFIXES,
    AttributeBag,
    AttributeSpec,
    SpanAttributeField,
    bag_keys,
    from_bag,
    scaled_decimal_to_int,
    spec_for_field,
    to_bag,
)

DIRECT_INGEST_RAW_ATTRIBUTES_KEY = "nemo.intake.direct.raw"


@dataclass
class SpanAttributeBags:
    string: dict[str, str] = field(default_factory=dict)
    number: dict[str, float] = field(default_factory=dict)
    boolean: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_domain_maps(
        cls,
        *,
        attributes_string: Mapping[str, str],
        attributes_number: Mapping[str, float],
        attributes_bool: Mapping[str, bool],
    ) -> Self:
        return cls(
            string=dict(attributes_string),
            number={key: float(value) for key, value in attributes_number.items()},
            boolean=dict(attributes_bool),
        )

    def get_field(self, field: SpanAttributeField | str) -> str | int | float | bool | Decimal | None:
        spec = spec_for_field(field)
        bag = self._bag_for_spec(spec)
        for key in bag_keys(spec):
            value = from_bag(bag.get(key), spec)
            if value is not None:
                return value
        return None

    def put_field(self, field: SpanAttributeField | str, value: Any) -> None:
        self.put_spec(spec_for_field(field), value)

    def put_spec(self, spec: AttributeSpec, value: Any) -> None:
        bag_value = to_bag(value, spec)
        if bag_value is None:
            return
        if spec.bag == AttributeBag.STRING:
            self.string[spec.bag_key] = str(bag_value)
        elif spec.bag == AttributeBag.BOOL:
            self.boolean[spec.bag_key] = bool(bag_value)
        else:
            self.number[spec.bag_key] = float(bag_value)

    def put_json(self, key: str, value: Any) -> None:
        if value is not None:
            self.string[key] = json.dumps(value, separators=(",", ":"), ensure_ascii=False)

    def put_unhandled_source_attributes(self, attributes: Mapping[str, Any], *, consumed_keys: set[str]) -> None:
        excluded = consumed_keys | SOURCE_ONLY_KEYS
        for key, value in attributes.items():
            if value is None or key in excluded or _is_source_only_prefix(key):
                continue
            self.put_unhandled_source_attribute(key, value)

    def put_unhandled_source_attribute(self, key: str, value: Any) -> None:
        if isinstance(value, bool):
            self.boolean[key] = value
        elif isinstance(value, int | float):
            if key.startswith("llm.cost."):
                self.number[f"cost.{key.removeprefix('llm.cost.')}"] = float(
                    scaled_decimal_to_int(value, scale=COST_SCALE)
                )
            else:
                self.number[key] = float(value)
        elif isinstance(value, str):
            if value:
                self.string[key] = value
        else:
            self.put_json(key, value)

    def cost_details(self) -> dict[str, float]:
        details: dict[str, float] = {}
        for key, value in self.number.items():
            if not key.startswith("cost.") or key in KNOWN_BAG_KEYS:
                continue
            details[key.removeprefix("cost.")] = float(Decimal(str(int(value))) / Decimal(COST_SCALE))
        return details

    def raw_attributes_json(self) -> str | None:
        raw: dict[str, Any] = {}
        for raw_key in ("atif.raw", DIRECT_INGEST_RAW_ATTRIBUTES_KEY):
            raw_json = self.string.get(raw_key)
            if not raw_json:
                continue
            parsed_raw = json.loads(raw_json)
            if not isinstance(parsed_raw, dict):
                raise TypeError(f"Expected {raw_key} to contain a JSON object")
            # nemo.experiment.metadata is a retired key no longer in KNOWN_BAG_KEYS; keep excluding it so
            # legacy rows that still carry it don't leak it into raw_attributes.
            parsed_raw.pop("nemo.experiment.metadata", None)
            raw.update(parsed_raw)

        for key, value in self.string.items():
            if (
                key not in {"atif.raw", DIRECT_INGEST_RAW_ATTRIBUTES_KEY, "nemo.experiment.metadata"}
                and key not in KNOWN_BAG_KEYS
            ):
                raw[key] = value
        for key, value in self.number.items():
            if key not in KNOWN_BAG_KEYS:
                raw[key] = value
        for key, value in self.boolean.items():
            if key not in KNOWN_BAG_KEYS:
                raw[key] = value
        if not raw:
            return None
        return json.dumps(raw, separators=(",", ":"), ensure_ascii=False)

    def _bag_for_spec(self, spec: AttributeSpec) -> dict[str, str] | dict[str, float] | dict[str, bool]:
        if spec.bag == AttributeBag.STRING:
            return self.string
        if spec.bag == AttributeBag.BOOL:
            return self.boolean
        return self.number


def _is_source_only_prefix(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in SOURCE_ONLY_PREFIXES)
