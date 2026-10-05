# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Protocol (OTLP) models for log ingestion.

These models support the OTLP/HTTP JSON format for logs.
Reference: https://opentelemetry.io/docs/specs/otlp/
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OtelAttributeKeyValue(BaseModel):
    """Represents a key-value pair attribute."""

    key: str
    value: Dict[str, Any] = Field(..., description="AnyValue as defined in OTLP spec")


class OtelResource(BaseModel):
    """Resource information including attributes."""

    attributes: List[OtelAttributeKeyValue] = Field(default_factory=list, description="Resource attributes")
    droppedAttributesCount: Optional[int] = Field(0, ge=0)


class OtelInstrumentationScope(BaseModel):
    """Information about the instrumentation scope."""

    name: str = ""
    version: Optional[str] = None
    attributes: Optional[List[OtelAttributeKeyValue]] = None


class OtelLogRecord(BaseModel):
    """
    A log record as defined in OTLP specification.

    This is a simplified version that captures the essential fields needed
    for job log ingestion.
    """

    timeUnixNano: str = Field(..., description="Timestamp in nanoseconds since Unix epoch")
    observedTimeUnixNano: Optional[str] = Field(None, description="Observation timestamp")
    severityNumber: Optional[int] = Field(None, ge=0, le=24)
    severityText: Optional[str] = None
    body: Optional[Dict[str, Any]] = Field(None, description="AnyValue containing log message")
    attributes: Optional[List[OtelAttributeKeyValue]] = Field(None, description="Log attributes")
    droppedAttributesCount: Optional[int] = Field(0, ge=0)
    flags: Optional[int] = Field(0, description="Trace flags")
    traceId: Optional[str] = Field(None, description="Hex-encoded trace ID")
    spanId: Optional[str] = Field(None, description="Hex-encoded span ID")

    model_config = {"extra": "ignore"}


class OtelScopeLogs(BaseModel):
    """Collection of logs produced by a scope."""

    scope: Optional[OtelInstrumentationScope] = None
    logRecords: List[OtelLogRecord] = Field(default_factory=list)
    schemaUrl: Optional[str] = None


class OtelResourceLogs(BaseModel):
    """
    A collection of logs from a resource.

    Resource attributes should contain workspace which maps to our filtering needs.
    """

    resource: Optional[OtelResource] = Field(None, description="Resource information")
    scopeLogs: List[OtelScopeLogs] = Field(default_factory=list)
    schemaUrl: Optional[str] = None


class OtelExportLogsServiceRequest(BaseModel):
    """
    Top-level OTLP request for exporting logs.

    This is the main payload sent to the /ingest endpoint. Extra fields are ignored.
    """

    resourceLogs: List[OtelResourceLogs] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class OtelExportLogsPartialSuccess(BaseModel):
    """Partial success response details."""

    rejectedLogRecords: int = Field(0, description="Number of rejected log records")
    errorMessage: Optional[str] = Field(None, description="Human-readable error message")


class OtelExportLogsServiceResponse(BaseModel):
    """
    Response for log export requests.

    Per OTLP spec, successful responses should be empty or contain partial_success info.
    """

    partialSuccess: Optional[OtelExportLogsPartialSuccess] = None


def extract_string_value(any_value: Dict[str, Any]) -> str:
    """
    Extract string value from OTLP AnyValue structure.

    AnyValue can be: stringValue, intValue, doubleValue, boolValue, arrayValue, kvlistValue, bytesValue
    """
    if "stringValue" in any_value:
        return any_value["stringValue"]
    elif "intValue" in any_value:
        return str(any_value["intValue"])
    elif "doubleValue" in any_value:
        return str(any_value["doubleValue"])
    elif "boolValue" in any_value:
        return str(any_value["boolValue"])
    else:
        # For complex types, return JSON representation
        return str(any_value)


def extract_attribute(
    attributes: List[OtelAttributeKeyValue], key: str, default: Optional[str] = None
) -> Optional[str]:
    """
    Extract a specific attribute value from a list of KeyValue pairs.

    Args:
        attributes: List of KeyValue attributes
        key: The attribute key to search for
        default: Default value if not found

    Returns:
        The attribute value as a string, or the default value
    """
    if not attributes:
        return default

    for attr in attributes:
        if attr.key == key:
            return extract_string_value(attr.value)

    return default


def extract_resource_attributes(
    resource_logs: OtelResourceLogs,
) -> Dict[str, str]:
    """
    Extract all resource attributes as a dictionary.

    Args:
        resource_logs: The OtelResourceLogs object

    Returns:
        Dictionary of resource attribute key-value pairs
    """
    if not resource_logs.resource or not resource_logs.resource.attributes:
        return {}

    return {attr.key: extract_string_value(attr.value) for attr in resource_logs.resource.attributes}


def extract_log_attributes(log_record: OtelLogRecord) -> Dict[str, str]:
    """
    Extract all log record level attributes as a dictionary.

    Args:
        log_record: The OtelLogRecord object

    Returns:
        Dictionary of log attribute key-value pairs
    """
    if not log_record.attributes:
        return {}

    return {attr.key: extract_string_value(attr.value) for attr in log_record.attributes}


def get_resource_attribute(resource_logs: OtelResourceLogs, key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a specific resource attribute value.

    Args:
        resource_logs: The OtelResourceLogs object
        key: The attribute key to search for
        default: Default value if not found

    Returns:
        The attribute value as a string, or the default value
    """
    if not resource_logs.resource or not resource_logs.resource.attributes:
        return default

    return extract_attribute(resource_logs.resource.attributes, key, default)


def get_log_attribute(log_record: OtelLogRecord, key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a specific log record attribute value.

    Args:
        log_record: The OtelLogRecord object
        key: The attribute key to search for
        default: Default value if not found

    Returns:
        The attribute value as a string, or the default value
    """
    if not log_record.attributes:
        return default

    return extract_attribute(log_record.attributes, key, default)


def convert_proto_any_value(proto_value) -> Dict[str, Any]:
    """
    Convert protobuf AnyValue to dictionary format.

    Args:
        proto_value: Protobuf AnyValue object

    Returns:
        Dictionary representation of the value
    """
    value_type = proto_value.WhichOneof("value")

    if value_type == "string_value":
        return {"stringValue": proto_value.string_value}
    elif value_type == "bool_value":
        return {"boolValue": proto_value.bool_value}
    elif value_type == "int_value":
        return {"intValue": proto_value.int_value}
    elif value_type == "double_value":
        return {"doubleValue": proto_value.double_value}
    elif value_type == "bytes_value":
        return {"bytesValue": proto_value.bytes_value.hex()}
    elif value_type == "array_value":
        return {"arrayValue": {"values": [convert_proto_any_value(v) for v in proto_value.array_value.values]}}
    elif value_type == "kvlist_value":
        return {
            "kvlistValue": {
                "values": [
                    {"key": kv.key, "value": convert_proto_any_value(kv.value)}
                    for kv in proto_value.kvlist_value.values
                ]
            }
        }
    else:
        return {}


def convert_proto_to_request(proto_request) -> OtelExportLogsServiceRequest:
    """
    Convert protobuf ExportLogsServiceRequest to Pydantic model.

    Args:
        proto_request: Protobuf ExportLogsServiceRequest

    Returns:
        Pydantic OtelExportLogsServiceRequest model
    """
    resource_logs = []

    for proto_resource_logs in proto_request.resource_logs:
        # Convert resource attributes
        resource = None
        if proto_resource_logs.HasField("resource"):
            resource = OtelResource(
                attributes=[
                    OtelAttributeKeyValue(key=attr.key, value=convert_proto_any_value(attr.value))
                    for attr in proto_resource_logs.resource.attributes
                ],
                droppedAttributesCount=proto_resource_logs.resource.dropped_attributes_count
                if proto_resource_logs.resource.dropped_attributes_count
                else 0,
            )

        # Convert scope logs
        scope_logs = []
        for proto_scope_logs in proto_resource_logs.scope_logs:
            # Convert scope
            scope = None
            if proto_scope_logs.HasField("scope"):
                scope = OtelInstrumentationScope(
                    name=proto_scope_logs.scope.name,
                    version=proto_scope_logs.scope.version if proto_scope_logs.scope.version else None,
                    attributes=[
                        OtelAttributeKeyValue(key=attr.key, value=convert_proto_any_value(attr.value))
                        for attr in proto_scope_logs.scope.attributes
                    ]
                    if proto_scope_logs.scope.attributes
                    else None,
                )

            # Convert log records
            log_records = []
            for proto_log in proto_scope_logs.log_records:
                log_record = OtelLogRecord(
                    timeUnixNano=str(proto_log.time_unix_nano),
                    observedTimeUnixNano=str(proto_log.observed_time_unix_nano)
                    if proto_log.observed_time_unix_nano
                    else None,
                    severityNumber=proto_log.severity_number if proto_log.severity_number else None,
                    severityText=proto_log.severity_text if proto_log.severity_text else None,
                    body=convert_proto_any_value(proto_log.body) if proto_log.HasField("body") else None,
                    attributes=[
                        OtelAttributeKeyValue(key=attr.key, value=convert_proto_any_value(attr.value))
                        for attr in proto_log.attributes
                    ]
                    if proto_log.attributes
                    else None,
                    droppedAttributesCount=proto_log.dropped_attributes_count
                    if proto_log.dropped_attributes_count
                    else 0,
                    flags=proto_log.flags if proto_log.flags else 0,
                    traceId=proto_log.trace_id.hex() if proto_log.trace_id else None,
                    spanId=proto_log.span_id.hex() if proto_log.span_id else None,
                )
                log_records.append(log_record)

            scope_logs.append(
                OtelScopeLogs(
                    scope=scope,
                    logRecords=log_records,
                    schemaUrl=proto_scope_logs.schema_url if proto_scope_logs.schema_url else None,
                )
            )

        resource_logs.append(
            OtelResourceLogs(
                resource=resource,
                scopeLogs=scope_logs,
                schemaUrl=proto_resource_logs.schema_url if proto_resource_logs.schema_url else None,
            )
        )

    return OtelExportLogsServiceRequest(resourceLogs=resource_logs)
