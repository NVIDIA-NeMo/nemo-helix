# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the trusted episode provisioning broker."""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sandboxed_gym.egress import EgressAllowlist

# Metadata keys the broker owns. A caller may not set them, so a backend label can always be
# trusted to say which job an episode belongs to.
RESERVED_METADATA_PREFIX = "nemo-rl-"
JOB_ID_METADATA_KEY = "nemo-rl-job-id"

# Platform job-pod labels. A sandbox carrying them would match NetworkPolicies that select the
# job's own pods, such as the per-step ingress policy that lets job pods reach each other's Ray ports.
PLATFORM_LABEL_PREFIX = "nhx.nvidia.com/"

# Reserved for anything the broker itself injects into an episode environment. Nothing uses it
# today; reserving it now means a caller cannot pre-empt a name the broker later needs to own.
RESERVED_ENV_PREFIX = "NEMO_RL_"

# Kubernetes label rules. Sandbox metadata becomes pod labels, so both the keys a caller supplies
# and the job id the broker stamps have to satisfy them.
K8S_LABEL_KEY_RE = re.compile(r"^([a-z0-9]([-a-z0-9.]*[a-z0-9])?/)?[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?$")
K8S_LABEL_VALUE_RE = re.compile(r"^[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?$")

# The trusted broker buffers every request body, so this cap protects the training leader pod --
# not the episode. Uploads larger than this must be staged through the episode itself.
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024


class EpisodeBrokerConfig(BaseModel):
    """Policy and placement settings for one job's episode broker.

    Every field is trusted-side policy: the job sandbox can read the broker token but has no way to
    influence any value here. Defaults are deliberately closed -- with no ``approved_images`` or
    ``approved_image_prefixes`` configured, every episode create is refused, so a job that does not
    use episode sandboxes is unaffected and one that does must be granted images explicitly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str = Field(min_length=1)

    backend: Literal["opensandbox", "memory"] = "opensandbox"
    backend_options: dict[str, Any] = Field(default_factory=dict)
    # Second key required to select the in-memory backend, which provisions nothing real.
    allow_insecure_memory_backend: bool = False

    approved_images: tuple[str, ...] = ()
    approved_image_prefixes: tuple[str, ...] = ()

    default_ttl_s: float = Field(default=300.0, gt=0)
    max_ttl_s: float = Field(default=3600.0, gt=0)
    # Caps on how long one broker request may block. A caller that could name its own timeout
    # could otherwise pin trusted-side connections open indefinitely.
    default_exec_timeout_s: float = Field(default=180.0, gt=0)
    max_exec_timeout_s: float = Field(default=1800.0, gt=0)
    max_ready_timeout_s: float = Field(default=300.0, gt=0)
    max_concurrent_episodes: int = Field(default=32, ge=1)
    max_request_bytes: int = Field(default=DEFAULT_MAX_REQUEST_BYTES, gt=0)

    max_cpu: float | None = Field(default=8.0, gt=0)
    max_memory_mib: int | None = Field(default=32768, gt=0)
    max_disk_gib: int | None = Field(default=128, gt=0)
    # Episode sandboxes grade model output on CPU; GPUs are opt-in per deployment.
    max_gpu: int = Field(default=0, ge=0)

    # Egress is always deny-by-default and allow-only. Public internet is an explicit profile
    # that adds DNS suffixes -- never a public CIDR, so reaching public space still requires
    # resolving an allowed name through the sidecar's proxy.
    allow_internet: bool = False
    egress_allow_targets: tuple[str, ...] = ()
    public_dns_allow: tuple[str, ...] | None = None
    # Set when the episode sandbox resolves through a different nameserver than the broker.
    resolver_addresses: tuple[str, ...] | None = None

    # How hard to check that the egress policy the backend reports matches the one requested.
    #
    # ``default_action`` is the default because it is the property that can only be established at
    # create and that decides what happens to unlisted traffic, and because it is the one a backend
    # cannot plausibly reformat. ``strict`` additionally requires every requested rule to come back,
    # which is the stronger guarantee but depends on the backend echoing targets in a comparable
    # form; the OpenSandbox sidecar re-marshals a merged policy, so turn it on only once a
    # deployment has shown it round-trips cleanly.
    egress_verification: Literal["off", "default_action", "strict"] = "default_action"

    # Host advertised to the job sandbox. Left unset the broker uses the local node IP; set it to a
    # headless-Service DNS name where one exists, so the sandbox egress rule survives a reschedule.
    host: str | None = None
    # Full URL published into bootstrap + egress (preferred over ``host`` for cross-job Service DNS).
    # Bind address remains 0.0.0.0:{port}; this is only what the Gym host is told to dial.
    advertise_url: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    port_range_low: int | None = Field(default=None, ge=1, le=65535)
    port_range_high: int | None = Field(default=None, ge=1, le=65535)

    @field_validator("job_id")
    @classmethod
    def _check_job_id(cls, value: str) -> str:
        # The job id is stamped into sandbox metadata, which OpenSandbox turns into a pod label,
        # and NeMo-Gym's provider silently normalizes label values it cannot use. If that
        # normalization ever fired, the stored label would stop matching the id we query by and
        # orphan reconciliation would quietly find nothing. Requiring a label-safe id up front
        # makes the normalization a no-op instead.
        if not K8S_LABEL_VALUE_RE.match(value):
            raise ValueError(
                f"job_id must be a valid Kubernetes label value (<=63 chars, alphanumeric with "
                f"'-', '_', '.', starting and ending alphanumeric): {value!r}"
            )
        return value

    @field_validator("approved_image_prefixes")
    @classmethod
    def _check_prefixes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        # A prefix must end at a path boundary. Without this, granting "registry.example.com/swe"
        # would also grant "registry.example.com/swe-attacker/anything".
        for prefix in value:
            if not prefix or not prefix.endswith("/"):
                raise ValueError(f"approved image prefix must end with '/': {prefix!r}")
        return value

    @model_validator(mode="after")
    def _check_bounds(self) -> "EpisodeBrokerConfig":
        if self.default_ttl_s > self.max_ttl_s:
            raise ValueError("default_ttl_s must not exceed max_ttl_s")
        if self.default_exec_timeout_s > self.max_exec_timeout_s:
            raise ValueError("default_exec_timeout_s must not exceed max_exec_timeout_s")
        low, high = self.port_range_low, self.port_range_high
        if (low is None) != (high is None):
            raise ValueError("port_range_low and port_range_high must be set together")
        if low is not None and high is not None and low >= high:
            raise ValueError("port_range_low must be less than port_range_high")
        return self

    @property
    def egress_allowlist(self) -> EgressAllowlist:
        """This tier's egress inputs, in the shape both sandbox tiers share."""
        return EgressAllowlist(
            targets=self.egress_allow_targets,
            allow_internet=self.allow_internet,
            public_dns_allow=self.public_dns_allow or (),
            resolver_addresses=self.resolver_addresses,
        )


class BrokerEndpoint(BaseModel):
    """Where the job sandbox reaches the broker, and the token it must present.

    ``token`` is excluded from ``repr`` so it does not reach driver logs. It is not a secret from
    the sandbox -- user code can read it -- but it should not leak any further than it must.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    host: str
    port: int
    token: str = Field(repr=False)
