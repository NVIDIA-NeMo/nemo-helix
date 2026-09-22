# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime and mapping helpers for observability-store import scripts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from uuid import UUID

import httpx
from nemo_helix_ext.client.factory import create_client

JsonObject = dict[str, Any]
SPAN_BATCH_LIMIT = 1000
DEFAULT_BATCH_SIZE = 500
DEFAULT_TIMEOUT_SECONDS = 60
MAX_INTAKE_PAGES = 1000


@dataclass(frozen=True)
class RecordCoverage:
    """Top-level source-field disposition recorded by an adapter."""

    record: str
    all_fields: frozenset[str]
    mapped_fields: frozenset[str]
    preserved_fields: frozenset[str]
    ignored_fields: frozenset[str]

    def validate(self) -> None:
        dispositions = (self.mapped_fields, self.preserved_fields, self.ignored_fields)
        if any(left & right for index, left in enumerate(dispositions) for right in dispositions[index + 1 :]):
            raise ValueError(f"Overlapping field dispositions for {self.record}")
        if set().union(*dispositions) != self.all_fields:
            raise ValueError(f"Incomplete field dispositions for {self.record}")


@dataclass
class ImportBundle:
    source: str
    spans: list[JsonObject] = field(default_factory=list)
    evaluator_results: list[JsonObject] = field(default_factory=list)
    annotations: list[JsonObject] = field(default_factory=list)
    coverage: list[RecordCoverage] = field(default_factory=list)

    def as_json(self) -> JsonObject:
        return {
            "source": self.source,
            "spans": self.spans,
            "evaluator_results": self.evaluator_results,
            "annotations": self.annotations,
        }

    def validate(self) -> None:
        if not self.spans:
            raise ValueError(f"{self.source} export did not contain any spans")
        identities = [(str(item["trace_id"]), str(item["span_id"])) for item in self.spans]
        if len(identities) != len(set(identities)):
            raise ValueError(f"{self.source} export contains duplicate (trace_id, span_id) identities")
        for item in self.coverage:
            item.validate()


def partition_record(
    record: JsonObject,
    *,
    record_name: str,
    mapped_fields: set[str],
    ignored_fields: set[str] | None = None,
) -> tuple[JsonObject, RecordCoverage]:
    """Preserve every unrecognized top-level field and record its disposition."""

    ignored = ignored_fields or set()
    if mapped_fields & ignored:
        raise ValueError(f"Mapped and ignored fields overlap for {record_name}")
    all_fields = set(record)
    mapped_present = all_fields & mapped_fields
    ignored_present = all_fields & ignored
    preserved = all_fields - mapped_present - ignored_present
    raw = {key: to_jsonable(record[key]) for key in record if key in preserved}
    coverage = RecordCoverage(
        record=record_name,
        all_fields=frozenset(all_fields),
        mapped_fields=frozenset(mapped_present),
        preserved_fields=frozenset(preserved),
        ignored_fields=frozenset(ignored_present),
    )
    coverage.validate()
    return raw, coverage


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value, key=repr)]
    if callable(model_dump := getattr(value, "model_dump", None)):
        return to_jsonable(model_dump(mode="json"))
    if callable(to_dict := getattr(value, "to_dict", None)):
        return to_jsonable(to_dict())
    if callable(to_dictionary := getattr(value, "to_dictionary", None)):
        return to_jsonable(to_dictionary())
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, int | float):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError(f"Unsupported timestamp value: {value!r}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def nanoseconds_to_datetime(value: Any) -> str:
    nanoseconds = int(value)
    seconds, remainder = divmod(nanoseconds, 1_000_000_000)
    return (datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(microseconds=remainder // 1000)).isoformat()


def normalize_kind(value: Any) -> str:
    normalized = str(value or "UNKNOWN").upper().removeprefix("SPAN_KIND_")
    aliases = {
        "CHAT_MODEL": "LLM",
        "MODEL": "LLM",
        "PROMPT": "LLM",
        "WORKFLOW": "CHAIN",
        "TASK": "CHAIN",
        "PARSER": "CHAIN",
        "MEMORY": "CHAIN",
        "FUNCTION": "TOOL",
        "SEARCH": "RETRIEVER",
        "SCORE": "EVALUATOR",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {
        "LLM",
        "CHAIN",
        "TOOL",
        "RETRIEVER",
        "EMBEDDING",
        "AGENT",
        "RERANKER",
        "EVALUATOR",
        "GUARDRAIL",
        "UNKNOWN",
    }
    return normalized if normalized in supported else "UNKNOWN"


def normalize_status(value: Any, *, error: Any = None) -> str:
    if error not in (None, "", False):
        return "error"
    normalized = str(value or "unknown").lower()
    if any(token in normalized for token in ("error", "fail")) or normalized in {"2", "status_code_error"}:
        return "error"
    if "cancel" in normalized:
        return "cancelled"
    if normalized in {"ok", "success", "succeeded", "complete", "completed", "1", "status_code_ok"}:
        return "success"
    return "unknown"


def set_if(attributes: JsonObject, key: str, value: Any) -> None:
    if value is not None and value != "":
        attributes[key] = to_jsonable(value)


def add_provider_raw(span: JsonObject, provider: str, raw: JsonObject) -> None:
    if not raw:
        return
    attributes = span.setdefault("attributes", {})
    if not isinstance(attributes, dict):
        raise TypeError("span attributes must be an object")
    existing = attributes.get(f"{provider}.raw")
    if existing is None:
        attributes[f"{provider}.raw"] = raw
    elif isinstance(existing, dict):
        existing.update(raw)
    else:
        raise TypeError(f"{provider}.raw must be an object")


def add_provider_signal_raw(span: JsonObject, provider: str, raw: JsonObject) -> None:
    if not raw:
        return
    attributes = span.setdefault("attributes", {})
    if not isinstance(attributes, dict):
        raise TypeError("span attributes must be an object")
    key = f"{provider}.signals"
    existing = attributes.setdefault(key, [])
    if not isinstance(existing, list):
        raise TypeError(f"{key} must be an array")
    existing.append(raw)


def project_signal(
    *,
    provider: str,
    span_id: str,
    session_id: str,
    name: str,
    value: Any,
    comment: str | None,
    automated: bool,
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Project one native feedback/evaluation signal onto existing Intake APIs."""

    evaluator_results: list[JsonObject] = []
    annotations: list[JsonObject] = []
    metric_name = f"{provider}.{name}"
    if automated:
        if isinstance(value, bool):
            evaluator_results.append(
                {
                    "span_id": span_id,
                    "session_id": session_id,
                    "name": metric_name,
                    "data_type": "BOOLEAN",
                    "value": 1 if value else 0,
                    "comment": comment,
                }
            )
        elif isinstance(value, int | float):
            evaluator_results.append(
                {
                    "span_id": span_id,
                    "session_id": session_id,
                    "name": metric_name,
                    "data_type": "NUMERIC",
                    "value": float(value),
                    "comment": comment,
                }
            )
        elif isinstance(value, str):
            evaluator_results.append(
                {
                    "span_id": span_id,
                    "session_id": session_id,
                    "name": metric_name,
                    "data_type": "CATEGORICAL",
                    "string_value": value,
                    "comment": comment,
                }
            )
        elif value is not None:
            evaluator_results.append(
                {
                    "span_id": span_id,
                    "session_id": session_id,
                    "name": metric_name,
                    "data_type": "TEXT",
                    "string_value": json.dumps(value, sort_keys=True, ensure_ascii=False),
                    "comment": comment,
                }
            )
        return evaluator_results, annotations

    sentiment = _feedback_sentiment(value)
    if sentiment is not None:
        annotations.append({"span_id": span_id, "session_id": session_id, "kind": "feedback", "value": sentiment})
    elif isinstance(value, int | float) and not isinstance(value, bool):
        annotations.append(
            {
                "span_id": span_id,
                "session_id": session_id,
                "kind": "label",
                "name": name[:256],
                "value_type": "numeric",
                "value": float(value),
            }
        )
    elif isinstance(value, str):
        annotations.append(
            {
                "span_id": span_id,
                "session_id": session_id,
                "kind": "label",
                "name": name[:256],
                "value_type": "text",
                "value": value,
            }
        )
    elif value is not None:
        annotations.append(
            {
                "span_id": span_id,
                "session_id": session_id,
                "kind": "metadata",
                "metadata": {metric_name: to_jsonable(value)},
            }
        )
    if comment:
        annotations.append(
            {
                "span_id": span_id,
                "session_id": session_id,
                "kind": "note",
                "text": comment[:10_000],
            }
        )
    return evaluator_results, annotations


def metadata_annotation(*, span_id: str, session_id: str, metadata: JsonObject) -> JsonObject:
    return {
        "span_id": span_id,
        "session_id": session_id,
        "kind": "metadata",
        "metadata": metadata,
    }


def _feedback_sentiment(value: Any) -> str | None:
    if isinstance(value, bool):
        return "positive" if value else "negative"
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"positive", "thumbs-up", "up", "like", "liked"}:
        return "positive"
    if normalized in {"negative", "thumbs-down", "down", "dislike", "disliked"}:
        return "negative"
    return None


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_bound(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO-8601 timestamp: {value}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("time bounds must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", help="Provider project or experiment identifier (required for live imports).")
    parser.add_argument("--since", type=parse_bound, help="Inclusive live-import lower time bound.")
    parser.add_argument("--until", type=parse_bound, help="Exclusive live-import upper time bound.")
    parser.add_argument("--input", type=Path, help="Read a provider JSON export instead of calling its API.")
    parser.add_argument("--nhx-base-url", default=os.environ.get("NHX_BASE_URL"))
    parser.add_argument("--workspace", default=os.environ.get("WORKSPACE"))
    parser.add_argument(
        "--include-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Import provider feedback, annotations, expectations, and scores.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Print the normalized bundle without writing Intake.")


def validate_common_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.input is None:
        if not args.project:
            parser.error("--project is required for live imports")
        if args.since is None or args.until is None:
            parser.error("live imports require both --since and --until")
        if args.since >= args.until:
            parser.error("--since must be earlier than --until")
    if not 1 <= args.batch_size <= SPAN_BATCH_LIMIT:
        parser.error(f"--batch-size must be between 1 and {SPAN_BATCH_LIMIT}")


class IntakeWriter:
    def __init__(
        self,
        *,
        base_url: str | None,
        workspace: str | None,
        access_token: str | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        session: Any | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._sdk_client: Any | None = None
        if session is None:
            self._sdk_client = create_client(
                base_url=base_url,
                access_token=access_token,
                timeout=float(timeout_seconds),
                max_retries=0,
            )
            self.session = _SdkSession(self._sdk_client)
            self.base_url = _validated_base_url(str(self._sdk_client.base_url))
            self.workspace = workspace or self._sdk_client.workspace or "default"
            self.headers: dict[str, str] = {}
        else:
            if base_url is None:
                raise ValueError("base_url is required when injecting an HTTP session")
            self.session = session
            self.base_url = _validated_base_url(base_url)
            self.workspace = workspace or "default"
            token = access_token or os.environ.get("NHX_ACCESS_TOKEN")
            self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.prefix = f"/apis/intake/v2/workspaces/{quote(self.workspace, safe='')}"

    def __enter__(self) -> IntakeWriter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._sdk_client is not None:
            self._sdk_client.close()
            self._sdk_client = None

    def write(self, bundle: ImportBundle, *, batch_size: int) -> JsonObject:
        bundle.validate()
        for start in range(0, len(bundle.spans), batch_size):
            spans = bundle.spans[start : start + batch_size]
            if self._sdk_client is not None:
                self._sdk_client.intake.ingest.spans.create(
                    workspace=self.workspace,
                    source=bundle.source,
                    spans=spans,
                )
            else:
                self._request(
                    "POST",
                    f"{self.prefix}/ingest/spans",
                    json_body={"source": bundle.source, "spans": spans},
                    expected={201},
                )
        for result in bundle.evaluator_results:
            self._request(
                "POST",
                f"{self.prefix}/evaluator-results",
                json_body=result,
                expected={201},
            )
        existing_signatures: set[str] = set()
        fetched_annotation_keys: set[tuple[str, str, str]] = set()
        written_annotations = 0
        for annotation in bundle.annotations:
            signature = _annotation_signature(annotation)
            if signature in existing_signatures:
                continue
            annotation_key = (
                str(annotation["session_id"]),
                str(annotation["kind"]),
                str(annotation.get("span_id") or ""),
            )
            if annotation_key not in fetched_annotation_keys:
                fetched_annotation_keys.add(annotation_key)
                existing_signatures.update(self._existing_annotation_signatures(annotation))
            if signature in existing_signatures:
                continue
            self._request(
                "POST",
                f"{self.prefix}/annotations",
                json_body=annotation,
                expected={201},
            )
            existing_signatures.add(signature)
            written_annotations += 1
        self._verify_spans(bundle.spans, source=bundle.source)
        return {
            "source": bundle.source,
            "spans": len(bundle.spans),
            "evaluator_results": len(bundle.evaluator_results),
            "annotations": written_annotations,
        }

    def _existing_annotation_signatures(self, annotation: JsonObject) -> set[str]:
        filters: list[tuple[str, str | int]] = [
            ("filter[session_id]", str(annotation["session_id"])),
            ("filter[kind]", str(annotation["kind"])),
        ]
        if annotation.get("span_id"):
            filters.append(("filter[span_id]", str(annotation["span_id"])))
        return {
            _annotation_signature(item) for item in self._paginated_data(f"{self.prefix}/annotations", filters=filters)
        }

    def _verify_spans(self, spans: list[JsonObject], *, source: str) -> None:
        expected_by_trace: dict[str, set[str]] = {}
        for item in spans:
            expected_by_trace.setdefault(str(item["trace_id"]), set()).add(str(item["span_id"]))
        for trace_id, expected_ids in expected_by_trace.items():
            filters: list[tuple[str, str | int]] = [
                ("filter[trace_id]", trace_id),
                ("filter[source]", source),
            ]
            found_ids = {str(item["span_id"]) for item in self._paginated_data(f"{self.prefix}/spans", filters=filters)}
            missing = expected_ids - found_ids
            if missing:
                raise RuntimeError(f"Intake verification did not find imported spans: {sorted(missing)}")

    def _paginated_data(
        self,
        path: str,
        *,
        filters: list[tuple[str, str | int]],
    ) -> Iterator[JsonObject]:
        for page in range(1, MAX_INTAKE_PAGES + 1):
            params = [*filters, ("page", page), ("page_size", 1000)]
            payload = self._request("GET", path, params=params, expected={200})
            if not isinstance(payload, dict):
                raise RuntimeError(f"Intake {path} response must be an object")
            data = payload.get("data", [])
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                raise RuntimeError(f"Intake {path} response `data` must be an array of objects")
            yield from data
            pagination = payload.get("pagination") or {}
            if not isinstance(pagination, dict):
                raise RuntimeError(f"Intake {path} response `pagination` must be an object")
            response_page = int(pagination.get("page", page))
            if response_page >= int(pagination.get("total_pages", response_page)):
                return
        raise RuntimeError(f"Intake {path} pagination exceeded {MAX_INTAKE_PAGES} pages")

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: list[tuple[str, str | int]] | None = None,
        json_body: JsonObject | None = None,
        expected: set[int],
    ) -> Any:
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            params=params,
            json=json_body,
            headers=self.headers,
            timeout=self.timeout_seconds,
            follow_redirects=False,
        )
        if response.status_code not in expected:
            body = response.text[:2000]
            raise RuntimeError(f"{method} {path} returned {response.status_code}: {body}")
        if not response.content:
            return None
        return response.json()


def run_import(bundle: ImportBundle, args: argparse.Namespace) -> int:
    bundle.validate()
    if args.dry_run:
        json.dump(bundle.as_json(), sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    with IntakeWriter(base_url=args.nhx_base_url, workspace=args.workspace) as writer:
        summary = writer.write(bundle, batch_size=args.batch_size)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _annotation_signature(annotation: JsonObject) -> str:
    kind = str(annotation["kind"])
    normalized: JsonObject = {
        "kind": kind,
        "span_id": annotation.get("span_id"),
        "session_id": annotation["session_id"],
    }
    if kind == "feedback":
        normalized["value"] = annotation["value"]
    elif kind == "note":
        normalized["text"] = annotation["text"]
    elif kind == "metadata":
        normalized["metadata"] = annotation["metadata"]
    elif kind == "label":
        normalized.update(
            {
                "name": annotation.get("name"),
                "value_type": annotation["value_type"],
                "value": annotation["value"],
            }
        )
    else:
        raise ValueError(f"Unsupported Intake annotation kind: {kind}")
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class _SdkSession:
    """Adapt the SDK factory's OAuth-aware HTTP client to the importer's request interface."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        options = {
            "params": kwargs.pop("params", None),
            "headers": kwargs.pop("headers", None),
            "timeout": kwargs.pop("timeout", None),
            "follow_redirects": bool(kwargs.pop("follow_redirects", False)),
        }
        options = {key: value for key, value in options.items() if value is not None}
        json_body = kwargs.pop("json", None)
        if kwargs:
            raise TypeError(f"Unsupported SDK request options: {sorted(kwargs)}")
        if method == "GET":
            return self.client.get(url, cast_to=httpx.Response, options=options)
        if method == "POST":
            return self.client.post(url, cast_to=httpx.Response, body=json_body, options=options)
        raise ValueError(f"Unsupported Intake request method: {method}")


def _validated_base_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.username or parsed.password:
        raise ValueError("NHX base URL must not contain userinfo")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("NHX base URL must be an origin without a path, query, or fragment")
    if parsed.scheme == "https" and parsed.hostname:
        return value.rstrip("/")
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}:
        return value.rstrip("/")
    raise ValueError("NHX base URL must use HTTPS, except for localhost or 127.0.0.1")


def validated_service_url(value: str, *, label: str) -> str:
    """Validate an authenticated provider endpoint while permitting a deployment path."""

    parsed = urlparse(value)
    if parsed.username or parsed.password:
        raise ValueError(f"{label} must not contain userinfo")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{label} must not contain a query or fragment")
    if parsed.scheme == "https" and parsed.hostname:
        return value.rstrip("/")
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}:
        return value.rstrip("/")
    raise ValueError(f"{label} must use HTTPS, except for localhost or 127.0.0.1")
