# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OTLP capture both Fabric runtimes share.

The receiver is exercised over real HTTP rather than by calling its methods: it exists because
Relay will only talk to a socket, so the socket is the contract.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_receiver import (
    EXPORT_SUFFIX,
    PROTOBUF_MEDIA_TYPE,
    READY_FILENAME,
    TRACES_PATH,
    OTLPReceiver,
)
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_writer import (
    OTLP_FILENAME,
    fold_exports,
    otlp_endpoint_fields,
    otlp_trace_path,
    register_trace_evidence,
    traces_dir,
)
from nemo_evaluator_sdk.values.evidence import EVIDENCE_FORMAT_ATIF, EVIDENCE_FORMAT_OTLP, EVIDENCE_TRACE
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from packages.nemo_evaluator_sdk.tests.agent_eval._otlp_testkit import export, post, span_names


@pytest.fixture
def receiver(tmp_path: Path):
    with OTLPReceiver(traces_dir(tmp_path)) as running:
        yield running


def test_an_export_is_stored_and_folds_into_a_readable_trace(receiver, tmp_path: Path) -> None:
    assert post(receiver.endpoint, export("agent run")) == 200

    assert fold_exports(traces_dir(tmp_path)) == 1
    assert span_names(otlp_trace_path(tmp_path)) == ["agent run"]


def test_repeated_flushes_all_survive_the_fold(receiver, tmp_path: Path) -> None:
    # An exporter flushes on its own schedule over a run; a later flush must not lose an earlier one.
    for name in ("first", "second", "third"):
        assert post(receiver.endpoint, export(name)) == 200

    assert fold_exports(traces_dir(tmp_path)) == 3
    assert span_names(otlp_trace_path(tmp_path)) == ["first", "second", "third"]


def test_folding_again_with_nothing_pending_leaves_the_trace_alone(receiver, tmp_path: Path) -> None:
    # Folding consumes its inputs, so a repeat pass -- a retry, or reprocessed evidence -- is a
    # no-op rather than a second copy of what a span-counting metric already saw.
    post(receiver.endpoint, export("only"))

    assert fold_exports(traces_dir(tmp_path)) == 1
    assert fold_exports(traces_dir(tmp_path)) == 0
    assert span_names(otlp_trace_path(tmp_path)) == ["only"]


def test_folding_new_exports_appends_to_the_trace_already_written(receiver, tmp_path: Path) -> None:
    """A later fold must extend the trace, not replace it with whatever arrived since.

    The rewrite is atomic, so a fold that wrote only the pending exports would ``os.replace`` the
    earlier spans away -- losing the first half of a run to nothing more than a second flush.
    """
    post(receiver.endpoint, export("first"))
    assert fold_exports(traces_dir(tmp_path)) == 1

    post(receiver.endpoint, export("second"))
    # The return counts what this call folded, not the spans now in the trace.
    assert fold_exports(traces_dir(tmp_path)) == 1

    assert span_names(otlp_trace_path(tmp_path)) == ["first", "second"]


def test_a_folded_export_is_not_kept_alongside_its_own_re_encoding(receiver, tmp_path: Path) -> None:
    post(receiver.endpoint, export("only"))

    fold_exports(traces_dir(tmp_path))

    assert not list(traces_dir(tmp_path).glob(f"*{EXPORT_SUFFIX}"))
    assert otlp_trace_path(tmp_path).is_file()


def test_a_large_export_survives_the_round_trip_whole(receiver, tmp_path: Path) -> None:
    # A real capture ran to 722KB. A line torn across writes parses as neither export, and the
    # reader reports the whole trace malformed rather than the one flush that tore.
    for index in range(4):
        assert post(receiver.endpoint, export(f"span-{index}", padding=150_000)) == 200

    assert fold_exports(traces_dir(tmp_path)) == 4
    lines = otlp_trace_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all(json.loads(line) for line in lines)


def test_an_export_carrying_no_spans_leaves_no_trace_to_register(receiver, tmp_path: Path) -> None:
    # An exporter may flush an empty batch on shutdown; that is not a trace, and an empty file
    # would be registered as one.
    assert post(receiver.endpoint, ExportTraceServiceRequest().SerializeToString()) == 200

    assert fold_exports(traces_dir(tmp_path)) == 0
    assert not otlp_trace_path(tmp_path).exists()


def test_an_unreadable_export_costs_the_flush_not_the_trace(tmp_path: Path) -> None:
    directory = traces_dir(tmp_path)
    directory.mkdir(parents=True)
    (directory / f"000001{EXPORT_SUFFIX}").write_bytes(b"\xff\xfe not protobuf")
    (directory / f"000002{EXPORT_SUFFIX}").write_bytes(export("survivor"))

    assert fold_exports(directory) == 1
    assert span_names(otlp_trace_path(tmp_path)) == ["survivor"]


def test_a_post_to_another_path_is_refused(receiver) -> None:
    assert post(receiver.endpoint.replace("/v1/traces", "/v1/metrics"), export("agent run")) == 404


def test_a_non_protobuf_body_is_refused_rather_than_guessed_at(receiver, tmp_path: Path) -> None:
    # OTLP/JSON would decode into something plausible and wrong. Relay never sends it.
    assert post(receiver.endpoint, b'{"resourceSpans": []}', media_type="application/json") == 415

    assert fold_exports(traces_dir(tmp_path)) == 0


def test_a_body_that_is_not_an_export_does_not_stop_the_receiver(receiver, tmp_path: Path) -> None:
    # Stored verbatim, so a bad body is only found at fold time -- but the socket must stay up.
    assert post(receiver.endpoint, b"\xff\xfe not protobuf") == 200

    assert post(receiver.endpoint, export("agent run")) == 200
    assert fold_exports(traces_dir(tmp_path)) == 1
    assert span_names(otlp_trace_path(tmp_path)) == ["agent run"]


def test_concurrent_flushes_each_land_as_their_own_export(receiver, tmp_path: Path) -> None:
    names = [f"span-{index}" for index in range(12)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        statuses = list(pool.map(lambda name: post(receiver.endpoint, export(name)), names))

    assert statuses == [200] * len(names)
    assert fold_exports(traces_dir(tmp_path)) == len(names)
    assert sorted(span_names(otlp_trace_path(tmp_path))) == sorted(names)


def test_the_endpoint_is_loopback_on_a_port_the_os_assigned(receiver) -> None:
    # Port 0 rather than a fixed one: trials run in parallel and would otherwise collide.
    assert receiver.endpoint.startswith("http://127.0.0.1:")
    assert receiver.endpoint.endswith("/v1/traces")
    assert receiver.server_address[1] != 0


def test_leaving_the_context_stops_serving_and_leaves_no_thread(tmp_path: Path) -> None:
    before = threading.active_count()
    with OTLPReceiver(traces_dir(tmp_path)) as running:
        endpoint = running.endpoint
        assert post(endpoint, export("agent run")) == 200

    with pytest.raises(urllib.error.URLError):
        post(endpoint, export("after close"))
    assert threading.active_count() == before


def test_signal_ready_announces_a_bound_socket(tmp_path: Path) -> None:
    # The container runtime waits on this file before starting the harness, so it must mean the
    # socket already accepts exports.
    with OTLPReceiver(traces_dir(tmp_path)) as running:
        running.signal_ready()

        assert (traces_dir(tmp_path) / READY_FILENAME).is_file()
        assert post(running.endpoint, export("agent run")) == 200


def test_the_primary_trace_is_otlp_when_there_is_one_and_atif_otherwise(tmp_path: Path) -> None:
    # ATIF needs no file on disk: Fabric declares it in the artifact manifest and the handle
    # validates it on read.
    atif, otlp = tmp_path / "t.atif.json", tmp_path / OTLP_FILENAME

    atif_only: dict = {}
    register_trace_evidence(atif_only, atif=atif, otlp=otlp)
    assert atif_only[EVIDENCE_TRACE].format == EVIDENCE_FORMAT_ATIF

    otlp.write_text("{}\n", encoding="utf-8")
    both: dict = {}
    register_trace_evidence(both, atif=atif, otlp=otlp)
    assert both[EVIDENCE_TRACE].format == EVIDENCE_FORMAT_OTLP
    # The format decides which view is primary, never which is reachable.
    assert both[f"{EVIDENCE_TRACE}:{EVIDENCE_FORMAT_ATIF}"].format == EVIDENCE_FORMAT_ATIF
    assert both[f"{EVIDENCE_TRACE}:{EVIDENCE_FORMAT_OTLP}"].format == EVIDENCE_FORMAT_OTLP


def test_an_empty_trace_file_is_not_registered(tmp_path: Path) -> None:
    otlp = tmp_path / OTLP_FILENAME
    otlp.touch()

    descriptors: dict = {}
    register_trace_evidence(descriptors, atif=None, otlp=otlp)

    assert descriptors == {}


def test_the_endpoint_fields_are_the_ones_relay_needs() -> None:
    # Shared because both runtimes build this endpoint out of different classes; only the values
    # are common, and a divergence would silently change what Relay exports.
    fields = otlp_endpoint_fields(endpoint="http://127.0.0.1:1/v1/traces", service_name="fabric")

    assert fields == {
        "type": "full",
        "endpoint": "http://127.0.0.1:1/v1/traces",
        "transport": "http_binary",
        "service_name": "fabric",
    }


def test_a_fold_that_dies_partway_leaves_the_previous_trace_intact(receiver, tmp_path: Path, monkeypatch) -> None:
    # Written through a temporary file and moved into place: a half-written trace parses as
    # malformed, and a reader cannot tell it from one the exporter never finished.
    post(receiver.endpoint, export("first"))
    fold_exports(traces_dir(tmp_path))
    post(receiver.endpoint, export("second"))

    def boom(_: bytes) -> str:
        raise RuntimeError("disk full")

    monkeypatch.setattr("nemo_evaluator_sdk.agent_eval.runtimes.fabric.otlp_writer.export_to_json_line", boom)
    with pytest.raises(RuntimeError):
        fold_exports(traces_dir(tmp_path))

    assert span_names(otlp_trace_path(tmp_path)) == ["first"]
    assert not list(traces_dir(tmp_path).glob("*.partial"))


def _post_raw(endpoint: str, headers: str, body: bytes = b"") -> str:
    """POST with hand-written headers, so a header urllib would never emit can be sent."""
    host, _, port = endpoint.split("//", 1)[1].split("/", 1)[0].partition(":")
    with socket.create_connection((host, int(port)), timeout=10) as connection:
        connection.sendall(headers.encode("ascii") + body)
        connection.settimeout(10)
        return connection.recv(256).decode("ascii", errors="replace").splitlines()[0]


def test_a_negative_content_length_is_refused_without_stalling_the_receiver(receiver, tmp_path: Path) -> None:
    """A negative length must be rejected before the read, not passed to ``rfile.read``.

    ``read(-1)`` consumes until EOF, so the receiver -- single-threaded, and reachable by the
    untrusted agent sharing its loopback -- would serve nothing further until that client hung up,
    silently costing the trial its trace.
    """
    status = _post_raw(
        receiver.endpoint,
        f"POST {TRACES_PATH} HTTP/1.1\r\nHost: localhost\r\n"
        f"Content-Type: {PROTOBUF_MEDIA_TYPE}\r\nContent-Length: -1\r\n\r\n",
    )
    assert "400" in status, status

    # The receiver is still serving: a real export after the bad one still lands.
    assert post(receiver.endpoint, export("after")) == 200
    assert fold_exports(traces_dir(tmp_path)) == 1
    assert span_names(otlp_trace_path(tmp_path)) == ["after"]


def test_a_spanless_export_is_consumed_rather_than_re_read(receiver, tmp_path: Path) -> None:
    """An export carrying no spans still counts as read, so the next fold does not see it again.

    Relay can flush an empty export. Left on disk it folds to nothing forever, so every later pass
    reports zero folded — which is the signal the container runtime uses to warn that no OTLP was
    captured at all.
    """
    assert post(receiver.endpoint, ExportTraceServiceRequest().SerializeToString()) == 200
    traces = traces_dir(tmp_path)
    assert list(traces.glob(f"*{EXPORT_SUFFIX}"))

    assert fold_exports(traces) == 0
    assert list(traces.glob(f"*{EXPORT_SUFFIX}")) == []
    assert not otlp_trace_path(tmp_path).exists()


def test_a_spanless_flush_does_not_disturb_a_trace_already_folded(receiver, tmp_path: Path) -> None:
    # Consuming the empty export must not take the earlier trace with it: the partial is discarded,
    # so `os.replace` never runs and what is already on disk stands.
    post(receiver.endpoint, export("real"))
    assert fold_exports(traces_dir(tmp_path)) == 1

    post(receiver.endpoint, ExportTraceServiceRequest().SerializeToString())
    assert fold_exports(traces_dir(tmp_path)) == 0

    assert span_names(otlp_trace_path(tmp_path)) == ["real"]


def test_the_path_check_reads_the_url_not_the_raw_request_target(receiver, tmp_path: Path) -> None:
    """A query string or an absolute-form target still resolves to the traces path.

    RFC 7230 lets a client send the whole URL as the request target, and comparing the raw target
    against the path would 404 it — the export is refused and the trial silently loses its trace.
    """
    body = export("absolute form")
    host = receiver.endpoint.split("//", 1)[1].split("/", 1)[0]
    status = _post_raw(
        receiver.endpoint,
        f"POST http://{host}{TRACES_PATH}?compression=none HTTP/1.1\r\nHost: {host}\r\n"
        f"Content-Type: {PROTOBUF_MEDIA_TYPE}\r\nContent-Length: {len(body)}\r\n\r\n",
        body,
    )
    assert "200" in status, status

    assert fold_exports(traces_dir(tmp_path)) == 1
    assert span_names(otlp_trace_path(tmp_path)) == ["absolute form"]


def test_an_unconsumable_export_does_not_cost_the_committed_trace(
    receiver, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Cleanup runs after the commit, so it must not raise over a trace already on disk.

    Removing an input needs the same directory permission the commit did, so this is narrow --
    an immutable file, or one held open on Windows -- but the trace is safe by then and losing it
    to a cleanup error would be strictly worse than leaving the input behind.
    """
    post(receiver.endpoint, export("kept"))
    original = Path.unlink

    def refuse(self: Path, missing_ok: bool = False) -> None:
        if self.suffix == ".pb":
            raise PermissionError("Operation not permitted")
        original(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", refuse)

    with caplog.at_level("WARNING"):
        assert fold_exports(traces_dir(tmp_path)) == 1

    assert span_names(otlp_trace_path(tmp_path)) == ["kept"]
    assert "duplicate its spans" in caplog.text
