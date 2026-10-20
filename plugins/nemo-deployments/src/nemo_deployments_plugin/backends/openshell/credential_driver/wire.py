# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official OpenShell wire messages, isolated from the SDK's descriptor pool.

The 0.1.2 Python SDK does not ship credential-driver stubs. Load the upstream
descriptor set in a private pool so importing the SDK in the same process
cannot collide with its separately generated datamodel definitions.
The README links the upstream SDK change that would let us replace
this compatibility layer with the SDK's generated messages and servicer.
"""

from pathlib import Path
from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, json_format, message_factory
from google.protobuf.message import Message

SERVICE = "openshell.credentials.v1.CredentialDriver"
METHODS = {
    "GetCapabilities": ("GetCredentialDriverCapabilitiesRequest", "GetCredentialDriverCapabilitiesResponse"),
    "StoreCredential": ("StoreCredentialRequest", "StoreCredentialResponse"),
    "ResolveCredentials": ("ResolveCredentialsRequest", "ResolveCredentialsResponse"),
    "DeleteCredential": ("DeleteCredentialRequest", "DeleteCredentialResponse"),
    "ListCredentials": ("ListCredentialsRequest", "ListCredentialsResponse"),
}

# Protobuf constructs these classes dynamically; the runtime wheel lacks stubs.
_pool = descriptor_pool.DescriptorPool()  # ty: ignore[possibly-missing-implicit-call]
_descriptors = descriptor_pb2.FileDescriptorSet.FromString(  # ty: ignore[unresolved-attribute]
    (Path(__file__).parent / "protocol" / "credential_driver.pb").read_bytes()
)
for _descriptor in _descriptors.file:
    _pool.Add(_descriptor)


def message_type(name: str) -> type[Message]:
    return message_factory.GetMessageClass(_pool.FindMessageTypeByName(f"openshell.credentials.v1.{name}"))


def message(name: str, data: dict[str, Any] | None = None) -> Message:
    return json_format.ParseDict(data or {}, message_type(name)(), descriptor_pool=_pool)


def data(value: Message) -> dict[str, Any]:
    return json_format.MessageToDict(value, preserving_proto_field_name=True, descriptor_pool=_pool)
