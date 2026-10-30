# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert SDK protobuf messages to and from dictionaries for driver request validation."""

from typing import Any

from google.protobuf import json_format
from google.protobuf.message import Message
from openshell._proto import credential_driver_pb2 as pb  # ty: ignore[unresolved-import]

SERVICE = "openshell.credentials.v1.CredentialDriver"
METHODS = {
    "GetCapabilities": ("GetCredentialDriverCapabilitiesRequest", "GetCredentialDriverCapabilitiesResponse"),
    "StoreCredential": ("StoreCredentialRequest", "StoreCredentialResponse"),
    "ResolveCredentials": ("ResolveCredentialsRequest", "ResolveCredentialsResponse"),
    "DeleteCredential": ("DeleteCredentialRequest", "DeleteCredentialResponse"),
    "ListCredentials": ("ListCredentialsRequest", "ListCredentialsResponse"),
}


def message_type(name: str) -> type[Message]:
    return getattr(pb, name)


def message(name: str, data: dict[str, Any] | None = None) -> Message:
    return json_format.ParseDict(data or {}, message_type(name)())


def data(value: Message) -> dict[str, Any]:
    return json_format.MessageToDict(value, preserving_proto_field_name=True)
