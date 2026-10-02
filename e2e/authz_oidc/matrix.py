# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The authz verification matrix: request + identity -> expected status.

Each case is one HTTP request against the running platform, authenticated by a
real signed JWT (or a deliberately defective one). ``expected`` is either a set
of acceptable status codes, or ``NOT_403`` for rows where the authz decision is
the oracle but the downstream handler's status is environment-dependent (e.g.
the agent-gateway proxy 404s on a nonexistent agent *after* authorization
passes — getting past the PDP is exactly what the row proves).

Identity keys map to tokens minted by the session issuer (see conftest):

- ``admin``      sub=usr-admin,  email matching auth.admin_email -> HelixAdmin@system (seeded)
- ``alice``      sub=usr-alice,  email alice@harness.test  -> Editor@<wsA> (provisioned)
- ``victor``     sub=usr-victor, email victor@harness.test -> Viewer@<wsA> (provisioned)
- ``sam``        sub=usr-sam,    email sam@harness.test    -> Viewer@system (provisioned)
- ``nobody``     sub=usr-nobody, email nobody@harness.test -> no bindings anywhere
- ``service``    sub=service:probe        -> service principal (ServiceSystem '*' default)
- ``provisioner`` sub=service:e2e-harness -> service principal used for setup
- ``anonymous``  no Authorization header at all

Workspace placeholders ``{wsA}``/``{wsB}`` are substituted by the test with the
session's provisioned workspace names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

NOT_403 = "not-403"

# Paths under test (placeholders substituted at runtime).
TARGETS = "/apis/auditor/v2/workspaces/{wsA}/targets"
TARGETS_B = "/apis/auditor/v2/workspaces/{wsB}/targets"
WORKSPACES = "/apis/entities/v2/workspaces"
EVAL_HELLO = "/apis/evaluator/v1/hello/world"
EVAL_HEALTHZ = "/apis/evaluator/v1/healthz"
GATEWAY = "/apis/agents/v2/workspaces/{wsA}/agents/ghost-agent/-/health"
SERVICE_ONLY = "/apis/harness-fixture/probe/service-only"
FIXTURE_OPEN = "/apis/harness-fixture/probe/open"
UNRULED_OK = "/apis/harness-unruled/ruled"
UNRULED_BAD = "/apis/harness-unruled/unruled"
BROKEN_SUB = "/apis/harness-broken/anything"
BROKEN_BARE = "/apis/harness-broken"
UNKNOWN_PATH = "/apis/auditor/v2/path-that-matches-no-rule"
IAM_BINDINGS = "/apis/auth/v2/iam/role-bindings"
PDP_ALLOW = "/apis/auth/v2/authz/allow"

TARGET_BODY = {"name": "e2e-authz-{case}", "type": "openai", "model": "gpt-test"}


@dataclass(frozen=True)
class Case:
    id: str
    group: str
    description: str
    method: str
    path: str
    identity: str
    expected: set[int] | Literal["not-403"]  # exact acceptable codes, or the NOT_403 sentinel
    scopes: list[str] = field(default_factory=list)
    token_defect: str | None = None  # expired|wrong-issuer|wrong-audience|unknown-key|unsigned|garbage
    body: Mapping[str, object] | None = None
    phase: str = "default"
    notes: str = ""


MATRIX: list[Case] = [
    # ------------------------------------------------------------------ #
    # A. Token validity (authentication). Target endpoint is one alice    #
    # is fully authorized for, so ONLY the token defect varies.           #
    # ------------------------------------------------------------------ #
    Case("A1", "authn", "Valid signed token, authorized principal", "GET", TARGETS, "alice", {200}),
    Case("A2", "authn", "No token at all", "GET", TARGETS, "anonymous", {401}),
    Case("A3", "authn", "Expired token", "GET", TARGETS, "alice", {401}, token_defect="expired"),
    Case(
        "A4",
        "authn",
        "Wrong issuer (signed by our key)",
        "GET",
        TARGETS,
        "alice",
        {401},
        token_defect="wrong-issuer",
    ),
    Case("A5", "authn", "Wrong audience", "GET", TARGETS, "alice", {401}, token_defect="wrong-audience"),
    Case(
        "A6",
        "authn",
        "Signed by a key absent from JWKS",
        "GET",
        TARGETS,
        "alice",
        {401},
        token_defect="unknown-key",
    ),
    Case(
        "A7",
        "authn",
        "Unsigned alg=none token (allow_unsigned_jwt=false)",
        "GET",
        TARGETS,
        "alice",
        {401},
        token_defect="unsigned",
    ),
    Case("A8", "authn", "Garbage bearer string", "GET", TARGETS, "alice", {401}, token_defect="garbage"),
    # ------------------------------------------------------------------ #
    # B. Role bindings & workspace isolation (authorization basics).      #
    # ------------------------------------------------------------------ #
    Case("B1", "bindings", "Valid signature, zero role bindings", "GET", TARGETS, "nobody", {403}),
    Case(
        "B2",
        "bindings",
        "Editor can write (control for scope rows)",
        "POST",
        TARGETS,
        "alice",
        {201},
        body=TARGET_BODY,
    ),
    Case("B3", "bindings", "Viewer reads OK", "GET", TARGETS, "victor", {200}),
    Case("B4", "bindings", "Viewer denied on write", "POST", TARGETS, "victor", {403}, body=TARGET_BODY),
    Case("B5", "bindings", "Cross-workspace: Editor in wsA denied in wsB", "GET", TARGETS_B, "alice", {403}),
    # ------------------------------------------------------------------ #
    # C. permission-stamped no-{workspace} GETs require the               #
    # permission (in the system workspace), not mere authentication.      #
    # Requires the seeded wildcard Viewer@system binding to be revoked.   #
    # ------------------------------------------------------------------ #
    Case(
        "C1",
        "no-workspace-get",
        "workspaces.list required: no system role -> deny",
        "GET",
        WORKSPACES,
        "alice",
        {403},
    ),
    Case("C2", "no-workspace-get", "workspaces.list via Viewer@system -> allow", "GET", WORKSPACES, "sam", {200}),
    Case(
        "C3",
        "no-workspace-get",
        "evaluator.hello.read: no system role -> deny",
        "GET",
        EVAL_HELLO,
        "alice",
        {403},
    ),
    Case(
        "C4",
        "no-workspace-get",
        "evaluator.hello.read via Viewer@system -> allow",
        "GET",
        EVAL_HELLO,
        "sam",
        {200},
    ),
    Case(
        "C5",
        "no-workspace-get",
        "Permissionless no-workspace GET stays open to any authenticated user (control)",
        "GET",
        EVAL_HEALTHZ,
        "alice",
        {200},
    ),
    Case(
        "C6",
        "no-workspace-get",
        "Permissionless no-workspace GET still requires authentication",
        "GET",
        EVAL_HEALTHZ,
        "anonymous",
        {401},
    ),
    # ------------------------------------------------------------------ #
    # D. read-scoped vs write-scoped tokens. alice is Editor@wsA          #
    # (holds the permissions) — only the token's scope claim varies.      #
    # ------------------------------------------------------------------ #
    Case(
        "D1",
        "scopes",
        "auditor:read scope allows GET",
        "GET",
        TARGETS,
        "alice",
        {200},
        scopes=["auditor:read"],
    ),
    Case(
        "D2",
        "scopes",
        "auditor:read scope denies POST",
        "POST",
        TARGETS,
        "alice",
        {403},
        scopes=["auditor:read"],
        body=TARGET_BODY,
    ),
    Case(
        "D3",
        "scopes",
        "auditor:write scope allows POST",
        "POST",
        TARGETS,
        "alice",
        {201},
        scopes=["auditor:write"],
        body=TARGET_BODY,
    ),
    Case(
        "D4",
        "scopes",
        "OIDC-only scopes (no area:verb) = full power, documented",
        "POST",
        TARGETS,
        "alice",
        {201},
        scopes=["openid", "profile", "email"],
        body=TARGET_BODY,
        notes="scopes.rego: tokens with no colon-scopes skip the scope gate by design",
    ),
    Case(
        "D5",
        "scopes",
        "Gateway read method passes with agents:read (authz oracle: not 403)",
        "GET",
        GATEWAY,
        "alice",
        NOT_403,
        scopes=["agents:read"],
        notes="proxy 404s on the nonexistent agent AFTER authorization passes",
    ),
    Case(
        "D6",
        "scopes",
        "Gateway write method denied with agents:read",
        "POST",
        GATEWAY,
        "alice",
        {403},
        scopes=["agents:read"],
        body={},
    ),
    Case(
        "D7",
        "scopes",
        "Gateway write method passes with agents:write (authz oracle: not 403)",
        "POST",
        GATEWAY,
        "alice",
        NOT_403,
        scopes=["agents:write"],
        body={},
    ),
    # ------------------------------------------------------------------ #
    # E. symmetric caller-kind enforcement.                               #
    # ------------------------------------------------------------------ #
    Case(
        "E1",
        "caller-kind",
        "Service principal denied on callers=[principal] route",
        "GET",
        TARGETS,
        "service",
        {403},
        notes="a service principal would otherwise pass via the ServiceSystem '*' wildcard",
    ),
    Case(
        "E2",
        "caller-kind",
        "Service principal allowed on service-only route",
        "GET",
        SERVICE_ONLY,
        "service",
        {200},
    ),
    Case(
        "E3",
        "caller-kind",
        "Human denied on service-only route (holds no relevant permission)",
        "GET",
        SERVICE_ONLY,
        "alice",
        {403},
    ),
    Case(
        "E4",
        "caller-kind",
        "HelixAdmin allowed on service-only route (admin global bypass holds)",
        "GET",
        SERVICE_ONLY,
        "admin",
        {200},
    ),
    Case(
        "E5",
        "caller-kind",
        "Fixture plugin mounted + open route works (control)",
        "GET",
        FIXTURE_OPEN,
        "alice",
        {200},
    ),
    Case(
        "E6",
        "caller-kind",
        "Service no-match bypass pinned: unknown path under healthy plugin -> authz passes (404)",
        "GET",
        UNKNOWN_PATH,
        "service",
        NOT_403,
        notes="documents the deliberate service:* bypass for unmatched paths",
    ),
    Case("E7", "caller-kind", "Human denied on same unknown path", "GET", UNKNOWN_PATH, "alice", {403}),
    Case(
        "E8",
        "caller-kind",
        "Human denied on IAM role-bindings API (service-principal-only handler)",
        "POST",
        IAM_BINDINGS,
        "alice",
        {403},
        body={"principal": "x@harness.test", "workspace": "default", "role": "Viewer"},
    ),
    Case(
        "E9",
        "caller-kind",
        "Service JWT (sub=service:*) accepted by IAM API — provisioning ran on signed JWTs",
        "GET",
        IAM_BINDINGS,
        "provisioner",
        {200},
    ),
    Case(
        "E10",
        "caller-kind",
        "PDP entrypoint rejects Bearer identity (header-principal only)",
        "POST",
        PDP_ALLOW,
        "provisioner",
        {401},
        body={"input": {}},
        notes="middleware consults only X-NHX-Principal-Id on /apis/auth/v2/authz/*",
    ),
    # ------------------------------------------------------------------ #
    # F. plugin fence & deny_route containment.                           #
    # ------------------------------------------------------------------ #
    Case(
        "F1",
        "fence",
        "Unenumerable plugin: human denied under fenced namespace",
        "GET",
        BROKEN_SUB,
        "alice",
        {403},
    ),
    Case(
        "F2",
        "fence",
        "Unenumerable plugin: SERVICE principal denied (no-match bypass closed)",
        "GET",
        BROKEN_SUB,
        "service",
        {403},
    ),
    Case("F3", "fence", "Unenumerable plugin: HelixAdmin denied", "GET", BROKEN_SUB, "admin", {403}),
    Case(
        "F4",
        "fence",
        "Bare fenced prefix also denied for service principal",
        "GET",
        BROKEN_BARE,
        "service",
        {403},
    ),
    Case(
        "F5",
        "fence",
        "deny_route containment: ruled sibling route still works",
        "GET",
        UNRULED_OK,
        "alice",
        {200},
    ),
    Case("F6", "fence", "Unruled route denied for human", "GET", UNRULED_BAD, "alice", {403}),
    Case(
        "F7",
        "fence",
        "Unruled route denied for service principal (overrides '*')",
        "GET",
        UNRULED_BAD,
        "service",
        {403},
    ),
    Case(
        "F8",
        "fence",
        "Unruled route denied for HelixAdmin (overrides bypass)",
        "GET",
        UNRULED_BAD,
        "admin",
        {403},
    ),
    # ------------------------------------------------------------------ #
    # G. Knob phase: on_invalid_plugin=quarantine (restarted platform).   #
    # ------------------------------------------------------------------ #
    Case(
        "G1",
        "knobs",
        "Quarantine: ruled route of the offending plugin now fenced too",
        "GET",
        UNRULED_OK,
        "service",
        {403},
        phase="knobs",
    ),
    Case(
        "G2",
        "knobs",
        "Quarantine: HelixAdmin denied on quarantined namespace",
        "GET",
        UNRULED_OK,
        "admin",
        {403},
        phase="knobs",
    ),
    Case(
        "G3",
        "knobs",
        "Platform sanity under knob phase (admin lists workspaces)",
        "GET",
        WORKSPACES,
        "admin",
        {200},
        phase="knobs",
    ),
]
