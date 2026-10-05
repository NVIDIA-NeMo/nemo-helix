# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON payload deserialization helpers for auth code."""

from pydantic import ConfigDict, TypeAdapter, ValidationError

JsonObject = dict[str, object]
JsonPayload = bytes | bytearray | str


class JsonObjectDeserializationError(ValueError):
    """Raised when a JSON payload cannot be decoded as an object."""


class JsonObjectDeserializer:
    """Deserialize JSON payloads that must contain a strict JSON object."""

    _adapter: TypeAdapter[JsonObject] = TypeAdapter(JsonObject, config=ConfigDict(strict=True))

    def deserialize(self, payload: JsonPayload) -> JsonObject:
        """Return the JSON object encoded by payload."""
        try:
            return self._adapter.validate_json(payload)
        except ValidationError as exc:
            raise JsonObjectDeserializationError("JSON payload was not an object") from exc
