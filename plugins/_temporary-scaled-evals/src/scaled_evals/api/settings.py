# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, Literal
from urllib.parse import quote

from cryptography.fernet import Fernet
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Development default when SCALED_EVALS_DATABASE_URL is unset. Matches upstream.
# Deployments always set the DSN explicitly; leaving this reachable keeps a local
# `docker run postgres` the only setup step, as in the standalone repo.
_STANDALONE_DATABASE_URL = "postgresql://scaled_evals:scaled_evals@localhost:5432/scaled_evals"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    dispatch_worker_health_url: str = ""
    # scaled-evals owns its own Postgres, so this names it. Deliberately not
    # `DATABASE_URL`: the platform injects that into the same process, and reusing
    # it would run these migrations and claim queues inside the platform database.
    database_url: str = Field(default="", validation_alias="SCALED_EVALS_DATABASE_URL")
    # Every scaled-evals table lives here. Redundant in a dedicated database, but
    # it costs one connect-time option and keeps the vendored SQL safe to point at
    # a shared database without touching 19 table names. Unqualified SQL resolves
    # through search_path.
    database_schema: str = Field(default="scaled_evals", validation_alias="SCALED_EVALS_DATABASE_SCHEMA")
    # Applied by the plugin at startup and by `scaled-evals-migrate`. Turn off
    # where an external Job owns schema rollout.
    run_migrations: bool = Field(default=True, validation_alias="SCALED_EVALS_RUN_MIGRATIONS")
    # How long migrations wait for a database that is still starting. Covers
    # `initdb` on a fresh volume, which is what makes an unordered Kubernetes
    # start lose the race. Also the ceiling on how long a wrong DSN delays
    # startup, since the two are indistinguishable at connect time.
    migration_wait_seconds: float = Field(default=30.0, validation_alias="SCALED_EVALS_MIGRATION_WAIT_SECONDS")
    database_ssl_mode: str = "verify-full"
    database_ssl_root_cert: str | None = None
    database_pool_min_size: int = 2
    database_pool_max_size: int = 20
    database_pool_timeout_seconds: float = 10.0
    # Server-sent event guardrails are process-local. The deployment-wide
    # ceiling is this value multiplied by the number of API replicas.
    api_sse_max_connections: int = 20
    api_sse_max_duration_seconds: float = 3600.0
    api_sse_poll_interval_seconds: float = 1.0
    s3_endpoint: str = "http://localhost:9000"
    # Endpoint baked into presigned URLs handed to clients. In the cluster the
    # API reaches RustFS at an internal address (s3_endpoint), but clients hit a
    # different hostname; SigV4 binds Host, so the signed URL must use the host
    # the client actually calls. Falls back to s3_endpoint when unset (prod:
    # same value).
    s3_public_endpoint: str | None = None
    s3_access_key: str = "scaledevals"
    s3_secret_key: str = "scaledevals-dev-secret"
    s3_bucket: str = "scaled-evals"
    # Object-store backend. "s3" covers RustFS, MinIO, AWS S3, and GCS XML API
    # when HMAC credentials are available. "gcs" uses Google Cloud Storage's
    # JSON API with Application Default Credentials / Workload Identity.
    object_store_backend: Literal["s3", "gcs"] = "s3"
    gcs_bucket: str = ""
    gcs_api_base_url: str = "https://storage.googleapis.com"
    # GCS signed URLs use XML API endpoints. They do not require HMAC, but they
    # do require a service account that the Workload Identity principal can sign
    # as through IAM Credentials signBlob. When unset, GCS upload falls back to
    # resumable session URIs.
    gcs_upload_mode: Literal["auto", "signed_url", "resumable_session"] = "auto"
    gcs_signing_service_account: str = ""
    gcs_signed_url_region: str = "auto"
    gcs_iam_credentials_base_url: str = "https://iamcredentials.googleapis.com"
    gcs_token_url: str = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token"
    gcs_access_token: str = ""
    gcs_request_timeout_seconds: float = 30.0
    # Task-pack upload guardrails. Direct-to-object-store uploads are validated
    # server-side before a revision can enter the ready/building path; oversized
    # objects are deleted as a quarantine/cleanup policy so they cannot be
    # accidentally reused by later finalize attempts.
    task_pack_max_size_bytes: int = 20 * 1024 * 1024 * 1024
    task_pack_tenant_storage_quota_bytes: int = 100 * 1024 * 1024 * 1024
    task_pack_max_extracted_size_bytes: int = 20 * 1024 * 1024 * 1024
    task_pack_max_members: int = 100_000
    # Guardrails for server-side results.tar.gz creation. These are source
    # object limits; TODO: add tenant/account quotas and compressed-size caps.
    benchmark_archive_cleanup_interval_seconds: float = Field(default=300.0, gt=0)
    benchmark_archive_max_files: int = 100_000
    benchmark_archive_max_source_bytes: int = 10_000_000_000
    evaluation_archive_max_files: int = 10_000
    evaluation_archive_max_source_bytes: int = 1_000_000_000
    # Optional best-effort publication of completed Harbor job directories to
    # the standalone Harbor Log Viewer. The URL also powers manual upload
    # instructions when automatic upload is disabled by network topology.
    harbor_viewer_base_url: str = ""
    harbor_viewer_auto_upload: bool = True
    harbor_viewer_upload_token: str = ""
    harbor_viewer_upload_timeout_seconds: float = 30.0
    harbor_viewer_upload_overwrite: bool = True
    harbor_dataset_image_prepare_timeout_seconds: float = 7200.0
    harbor_dataset_image_poll_interval_seconds: float = 5.0
    # Public registries from which dataset-only Harbor tasks may import base
    # images. This is independent of the destination task-image allowlist.
    harbor_dataset_upstream_allowed_registries: str = Field(
        default="docker.io,ghcr.io",
        validation_alias="HARBOR_DATASET_UPSTREAM_ALLOWED_REGISTRIES",
    )
    # Disable when the deployment target cannot run the BuildKit daemon. This
    # makes readiness honest and prevents finalize from wedging revisions.
    buildkit_enabled: bool = True
    buildkit_addr: str = "tcp://localhost:1234"
    buildkit_timeout_seconds: float = 2100.0
    # External image-builder service. When set, finalize sends the already-uploaded
    # task pack to the service's uploaded-context mode, which returns an approved,
    # signed image instead of building in-cluster with BuildKit. This is the path
    # for deploy targets that cannot host BuildKit and must run only signed images
    # (restricted-v2 PSA + a signature-enforcing admission controller). Empty
    # disables the service (finalize then uses Cloud Build, BuildKit, or a reuse
    # image_ref). See docs/internals/ARCHITECTURE.md § Container Build.
    image_builder_service_url: str = ""
    # Optional operator-configured non-interactive bearer token. A given builder
    # may not have finalized its service-auth contract, so hosted users must not
    # assume one is provisioned. Empty sends no Authorization header. Never logged;
    # when used, it must come from a secret in the chart.
    image_builder_service_token: str = ""
    # Optional operator-configured immutable source commit for the server-owned
    # builder recipe. This is not a user task source commit; it pins the builder's
    # own service ref when the builder enforces immutable refs.
    image_builder_source_commit: str = ""
    # The resolve call blocks while the builder pipeline builds+signs (minutes), so
    # this must exceed the service's own build-wait timeout.
    image_builder_service_timeout_s: float = 2100.0
    # Optional Google Cloud Build backend for generic GKE deployments. This is
    # selected only when no image-builder service is configured, local BuildKit is
    # disabled, and the task pack already lives in GCS. It builds the uploaded task
    # pack and pushes the resulting task image to IMAGE_REGISTRY, typically GAR.
    cloud_build_enabled: bool = False
    cloud_build_project: str = ""
    cloud_build_location: str = "global"
    cloud_build_api_base_url: str = "https://cloudbuild.googleapis.com"
    cloud_build_docker_builder_image: str = "gcr.io/cloud-builders/docker"
    cloud_build_service_account: str = ""
    cloud_build_timeout_seconds: float = 2100.0
    cloud_build_poll_interval_seconds: float = 5.0
    # A separate durable worker executes queued build jobs. Hosted
    # deployments set this from buildWorker.enabled so readyz catches a chart
    # that deploys the API without the worker.
    build_worker_required: bool = True
    build_worker_stale_seconds: float = 60.0
    # Target platform for finalize image builds. Pinned to the runtime cluster's
    # arch (hosted nodes are amd64) so an arm64 build host (Apple Silicon under
    # Colima) does not produce images that `exec format error` on amd64 nodes.
    # Empty string disables the pin (build for the host arch).
    image_build_platform: str = "linux/amd64"
    # Deployed application version reported by ``GET /version``. The Helm chart
    # sets this from .Chart.AppVersion, generated by CI/release automation.
    app_version: str | None = None
    # Production observability thresholds. These only classify metrics; they do
    # not affect durable dispatch leases or worker recovery behavior.
    observability_stuck_queued_seconds: float = 900.0
    observability_stuck_provisioning_seconds: float = 900.0
    observability_stuck_running_seconds: float = 7200.0
    observability_worker_stale_seconds: float = 120.0
    observability_task_pack_scan_limit: int = 500
    # Evaluation runtime polling. The product can accept more work than the
    # active concurrency limit, while each claimed run has this interval/count
    # baseline (default: about one hour, preserving legacy behavior). A longer
    # finite framework-profile lifecycle timeout extends the terminal wait.
    dispatch_run_poll_interval_seconds: float = 10.0
    resource_usage_sample_interval_seconds: float = 30.0
    dispatch_run_max_polls: int = 360
    # Hosted Kubernetes can move each evaluation into a standalone Job so
    # control-plane Deployment rollouts do not terminate active orchestration.
    dispatch_kubernetes_jobs_enabled: bool = False
    dispatch_job_reconcile_stale_seconds: float = 60.0
    # Platform Jobs migration flags. Postgres remains the admission and
    # compatibility source of truth while these are enabled.
    platform_build_jobs_enabled: bool = Field(
        default=False,
        validation_alias="SCALED_EVALS_PLATFORM_BUILD_JOBS_ENABLED",
    )
    platform_evaluation_jobs_enabled: bool = Field(
        default=False,
        validation_alias="SCALED_EVALS_PLATFORM_EVALUATION_JOBS_ENABLED",
    )
    platform_jobs_workspace: str = Field(
        default="default",
        validation_alias="SCALED_EVALS_PLATFORM_JOBS_WORKSPACE",
    )
    platform_jobs_profile: str = Field(
        default="default",
        validation_alias="SCALED_EVALS_PLATFORM_JOBS_PROFILE",
    )
    platform_jobs_provider: Literal["cpu", "subprocess"] = Field(
        default="cpu",
        validation_alias="SCALED_EVALS_PLATFORM_JOBS_PROVIDER",
    )
    # The image must contain this plugin plus its Harbor/sandbox runtime.
    # Deployments set an immutable image reference before enabling either flag.
    platform_jobs_image: str = Field(
        default="",
        validation_alias="SCALED_EVALS_PLATFORM_JOBS_IMAGE",
    )
    platform_jobs_postgres_password_secret: str = Field(
        default="",
        validation_alias="SCALED_EVALS_PLATFORM_JOBS_POSTGRES_PASSWORD_SECRET",
    )
    platform_jobs_credentials_encryption_key_secret: str = Field(
        default="",
        validation_alias="SCALED_EVALS_PLATFORM_JOBS_CREDENTIALS_ENCRYPTION_KEY_SECRET",
    )
    platform_jobs_registry_auth_secret: str = Field(
        default="",
        validation_alias="SCALED_EVALS_PLATFORM_JOBS_REGISTRY_AUTH_SECRET",
    )
    # Entity Store migration flags. Projection writes a derived read model while
    # Postgres stays authoritative; reads only flip once parity is established,
    # so the two are deliberately separate switches.
    entity_store_projection_enabled: bool = Field(
        default=False,
        validation_alias="SCALED_EVALS_ENTITY_STORE_PROJECTION_ENABLED",
    )
    entity_store_reads_enabled: bool = Field(
        default=False,
        validation_alias="SCALED_EVALS_ENTITY_STORE_READS_ENABLED",
    )
    entity_store_workspace: str = Field(
        default="default",
        validation_alias="SCALED_EVALS_ENTITY_STORE_WORKSPACE",
    )
    entity_store_projection_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        validation_alias="SCALED_EVALS_ENTITY_STORE_PROJECTION_BATCH_SIZE",
    )
    # Registry the finalized sandbox images are pushed to and pulled from.
    # Local compose: the in-stack `registry:2` service. Remote: NGC. Everything
    # that differs local-vs-remote lives here — the build logic has no hardcoded
    # addresses.
    registry_enabled: bool = True
    image_registry: str = "registry:5000"
    # HTTP (no TLS) push, for the local registry only. Secure by default —
    # compose and the chart set this explicitly where a plaintext in-stack
    # registry is intended. MUST be false against NGC.
    registry_insecure: bool = False
    # Optional push credentials (remote/NGC). Empty locally — the insecure local
    # registry needs no auth. When set, a transient docker config is handed to
    # buildctl so BuildKit can authenticate the push.
    registry_username: str | None = None
    registry_password: str | None = None
    # Task-image registry identity is independent of BuildKit's push target.
    # Hosted mode fails settings validation unless registry resolution is
    # explicitly configured.
    task_image_validation_mode: Literal["disabled", "resolve"] = "resolve"
    task_image_allowed_registries: str = ""
    task_image_allowed_repositories: str = ""
    task_image_registry_insecure: bool = False
    task_image_registry_auth_file: str = ""
    task_image_registry_auth_json: str = ""
    task_image_registry_timeout_seconds: float = 10.0
    task_image_hosted_mode: bool = False
    # Primary Fernet key for BYOK credential payloads. There is deliberately no
    # source-code default: every process that reads credentials must receive it
    # from deployment secret configuration. During rotation, comma-separated
    # previous keys remain decrypt-only fallbacks until ciphertext migration is
    # complete. See docs/API.md § Security, BYOK.
    credentials_encryption_key: str
    credentials_encryption_key_previous: str = ""

    @field_validator("credentials_encryption_key")
    @classmethod
    def validate_credentials_encryption_key(cls, value: str) -> str:
        try:
            Fernet(value.encode())
        except (TypeError, ValueError) as exc:
            raise ValueError("must be a valid Fernet key") from exc
        return value

    @model_validator(mode="after")
    def validate_hosted_task_image_policy(self) -> "Settings":
        if not self.task_image_hosted_mode:
            return self
        if self.task_image_validation_mode != "resolve":
            raise ValueError("hosted task images require TASK_IMAGE_VALIDATION_MODE=resolve")
        if not self.task_image_allowed_registries.strip():
            raise ValueError("hosted task images require TASK_IMAGE_ALLOWED_REGISTRIES")
        if self.sandbox_k8s_task_image_reference_mode != "tag":
            raise ValueError("hosted task images require SANDBOX_K8S_TASK_IMAGE_REFERENCE_MODE=tag")
        return self

    @field_validator("image_builder_source_commit")
    @classmethod
    def validate_image_builder_source_commit(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized and not re.fullmatch(r"[0-9a-f]{40}", normalized):
            raise ValueError("must be a full 40-character hex commit")
        return normalized

    @field_validator("credentials_encryption_key_previous")
    @classmethod
    def validate_previous_credentials_encryption_keys(cls, value: str) -> str:
        for key in value.split(","):
            if not key.strip():
                continue
            try:
                Fernet(key.strip().encode())
            except (TypeError, ValueError) as exc:
                raise ValueError("must contain only comma-separated Fernet keys") from exc
        return value

    # Optional live BYOK verification probes. These are low-cost metadata
    # endpoints; set a URL to an empty value to make that provider inconclusive.
    credential_verify_timeout_seconds: float = 10.0
    credential_verify_openai_models_url: str | None = "https://api.openai.com/v1/models"
    # The ``anthropic`` runtime credential is sent to NVIDIA's
    # Anthropic-compatible endpoint by dispatch.credentials.
    credential_verify_anthropic_models_url: str | None = "https://inference-api.nvidia.com/v1/models"
    credential_verify_nvidia_models_url: str | None = "https://inference-api.nvidia.com/v1/models"
    nvidia_inference_base_url: str = "https://inference-api.nvidia.com/v1"
    nvidia_anthropic_base_url: str = "https://inference-api.nvidia.com"

    # --- Control-plane ownership --------------------------------------------
    # The plugin does not authenticate callers; the platform does that before a
    # plugin route runs. Leaving this false resolves every caller to the shared
    # development owner. Setting it true makes scaled_evals.api.auth fail closed
    # with 401 until platform identity is bridged into the ownership model.
    control_plane_auth_enabled: bool = False
    # Comma-separated stable subjects are preferred. Verified email fallback
    # supports initial onboarding while an administrator discovers a colleague's
    # subject via /users/me; neither list grants team membership.
    control_plane_admin_subjects: str = ""
    control_plane_admin_emails: str = ""
    control_plane_admin_groups: str = ""
    control_plane_admin_roles: str = ""
    # Weighted sandbox-slot admission. Submissions may queue beyond these
    # limits; dispatch only claims work when its requested parallelism fits.
    control_plane_cluster_run_limit: int = 500
    control_plane_per_user_run_limit: int = 50

    @field_validator("api_sse_max_connections")
    @classmethod
    def validate_api_sse_max_connections(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be at least 1")
        return value

    @field_validator("api_sse_max_duration_seconds", "api_sse_poll_interval_seconds")
    @classmethod
    def validate_positive_sse_duration(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be greater than 0")
        return value

    # --- Live sandbox_k8s dispatch (Harbor on Kubernetes) -------------------------
    # Off by default: dispatch marks runs failed without contacting a cluster,
    # and unit tests stay cluster-free (they inject a fake backend). Enable to
    # drive `harbor run` -> sandbox_k8s.harbor:K8sSandboxEnvironment against a
    # real target. See scaled_evals.dispatch.sandbox_k8s and the agent-sandbox
    # harness (examples/agent-sandbox).
    sandbox_k8s_enabled: bool = False
    # Harbor checkout with sandbox-k8s[harbor] installed (`harbor run` lives here).
    harbor_dir: str = "~/code/harbor"
    # The Harbor config (e.g. configs/hello-skills-oracle.yaml) and target env
    # file (e.g. targets/<target>.env) from the agent-sandbox harness. Required
    # when enabled.
    sandbox_k8s_config_path: str | None = None
    sandbox_k8s_env_file: str | None = None
    # Where per-evaluation rendered configs + run logs are written.
    sandbox_k8s_work_dir: str = "/tmp"
    # Harbor's `jobs_dir` (relative to harbor_dir), where a run writes its
    # per-job dir `<jobs_dir>/<job_name>/result.json`. Must match the `jobs_dir`
    # in the harbor config; the status reader reads result.json from here.
    # Post-run Intake ATIF upload (dispatch worker) reads trial trajectories from
    # the same tree once the run finishes.
    sandbox_k8s_jobs_dir: str = "jobs/sandbox-k8s"
    # Full parent directory containing per-evaluation Harbor job directories.
    # Hosted runners mount Harbor's ``jobs`` directory somewhere other than
    # ``harbor_dir/jobs``; set this explicitly so post-run artifact/provenance
    # sync reads the same tree as the status reader. Local paths retain the
    # historical harbor_dir + sandbox_k8s_jobs_dir fallback when unset.
    sandbox_k8s_artifact_root: str | None = None
    # Agent-bundle sidecar layout. A bundle image ships an installer that copies
    # the agent into the shared /installed-agent volume and reads its identity
    # from prefixed environment variables. Both are a contract with whichever
    # builder produced the bundle image, not something this service can choose,
    # so point them at that builder's layout when registering bundles.
    agent_bundle_installer_path: str = "/opt/agent-bundle/copy-agent"
    agent_bundle_env_prefix: str = "AGENT_BUNDLE_"
    # Compose: one-shot Linux harbor-runner containers (same pattern as gym-runner).
    harbor_runner_image: str | None = None
    # Signed application/runner artifact identity persisted at submission.
    # Hosted charts set an immutable @sha256 reference; local development uses
    # the mutable dev tag and leaves the digest empty.
    harbor_runner_artifact_ref: str = "scaled-evals-api:dev"
    harbor_runner_artifact_digest: str | None = None
    # Release/signing attestation copied into each run at submission. Hosted CI
    # supplies these for the signed application image; local development leaves
    # them empty and is still explicitly identifiable by the mutable dev ref.
    harbor_runner_source_revision: str | None = None
    harbor_runner_ci_pipeline_id: str | None = None
    harbor_runner_ci_job_id: str | None = None
    harbor_runner_signature_ref: str | None = None
    harbor_runner_signature_digest: str | None = None
    harbor_runner_signature_audit_id: str | None = None
    # Host path for harbor job output (bind-mounted into harbor-runner at /opt/harbor/jobs).
    harbor_jobs_dir: str | None = None
    kube_config_dir: str | None = None
    # Host ``~/.kube`` for compose docker bind mounts (API container → harbor-runner).
    kube_config_dir_host: str | None = None
    sandbox_k8s_docker_volume: str = "scaled-evals-sandbox-k8s-work"
    sandbox_k8s_host_env_file: str | None = None
    # Local-only escape hatch for targets whose CA cannot yet be installed.
    # Hosted charts hard-code this false; user profiles cannot override it.
    sandbox_k8s_allow_insecure_tls: bool = False
    # Fail-closed escape hatch for root-requiring task images (for example,
    # Terminal-Bench). Harbor profiles cannot authorize themselves. Operators
    # must enable the switch and either approve the exact immutable image digest
    # or explicitly opt into the security-sensitive all-images POC mode.
    sandbox_k8s_allow_root: bool = False
    sandbox_k8s_allow_writable_root: bool = False
    sandbox_k8s_root_allow_all_images: bool = False
    sandbox_k8s_root_allowed_image_digests: str = ""
    # Signature-enforcing admission admits the signed tag form; generic
    # Kubernetes should execute the immutable digest form. Hosted values select
    # ``tag`` explicitly.
    sandbox_k8s_task_image_reference_mode: Literal["digest", "tag"] = "digest"

    # Additional runtime backend plugins loaded after the built-in sandbox_k8s
    # plugin. Comma-separated module specs; each module must expose
    # register_runtime_backends(registry), or use module:function.
    runtime_backend_plugins: str = "scaled_evals.dispatch.gym.plugin"
    # compose: probe Docker/kubeconfig from the API process.
    # configured: report runtime wiring from env/config only, for worker-owned
    # dispatch in cluster deployments.
    dispatch_health_mode: Literal["compose", "configured"] = "compose"

    # --- Live gym_daytona dispatch (NeMo Gym harbor_agent + Daytona) ---------
    # Off by default; same pattern as sandbox_k8s. Does not affect sandbox_k8s.
    gym_daytona_enabled: bool = False
    # NeMo Gym checkout (ng_e2e_collect_rollouts / ng_collect_rollouts live here).
    gym_dir: str = "~/src/Gym"
    # Target env from examples/gym-daytona/targets/<target>.env (DAYTONA_API_KEY,
    # GYM_CONFIG_PATHS, GYM_INPUT_JSONL, …).
    gym_daytona_env_file: str | None = None
    gym_daytona_work_dir: str = "/tmp/gym-daytona"
    gym_daytona_host_env_file: str | None = None
    gym_daytona_docker_volume: str = "scaled-evals-gym-daytona-work"

    # --- gym_sandbox_daytona (nemo_gym.sandbox + mini_swe_agent_2) -----------
    # Off by default. Requires Gym with sandbox API (PR #1377) + Daytona
    # provider (PR #1513). Does not affect sandbox_k8s or gym_daytona.
    gym_sandbox_daytona_enabled: bool = False
    gym_sandbox_daytona_env_file: str | None = None
    gym_sandbox_daytona_work_dir: str = "/tmp/gym-sandbox-daytona"
    # Host path to daytona.env for docker bind mounts (compose). When unset,
    # derived from scaled_evals_host_dir + container harness path.
    gym_sandbox_daytona_host_env_file: str | None = None
    # Named Docker volume shared by api and gym-runner containers (compose).
    gym_sandbox_daytona_docker_volume: str = "scaled-evals-gym-sandbox-work"

    # --- gym_sandbox_opensandbox (nemo_gym.sandbox + OpenSandbox) -----------
    # Off by default. NeMo RL sandboxes via an OpenSandbox cell HTTP endpoint.
    # Target env selects dev (port-forward), nrl-colocated, or nrl-remote — see
    # examples/gym-sandbox-opensandbox/targets/*.env.example.
    gym_sandbox_opensandbox_enabled: bool = False
    gym_sandbox_opensandbox_env_file: str | None = None
    gym_sandbox_opensandbox_work_dir: str = "/tmp/gym-sandbox-opensandbox"
    # Host path to opensandbox.env for docker bind mounts (compose). When unset,
    # derived from scaled_evals_host_dir + container harness path.
    gym_sandbox_opensandbox_host_env_file: str | None = None
    # Named Docker volume shared by api and gym-runner containers (compose).
    gym_sandbox_opensandbox_docker_volume: str = "scaled-evals-gym-opensandbox-work"

    # When set, gym dispatch uses the interim compose submitter: one-shot gym-runner
    # containers via the Docker socket. Production should use the same image ref
    # but launch K8s Jobs per evaluation (submitter swap at the RuntimeBackend seam).
    gym_runner_image: str | None = None
    # Compose launches a one-shot Docker container. Hosted evaluation Jobs use
    # the process backend because the Job container is already the isolated,
    # digest-pinned Gym runner and has no Docker socket.
    gym_runner_mode: Literal["docker", "process"] = "docker"
    # Immutable Gym runner identity captured into each gym_* evaluation snapshot.
    # Keep these unset rather than using "unknown" when the build/publish pipeline
    # cannot provide an exact value.
    gym_runner_image_digest: str | None = None
    gym_source_revision: str | None = None
    gym_package_version: str | None = None
    # Optional package-level CycloneDX/SPDX evidence published for the runner
    # image. The run-composition BOM binds it to gym_runner_image_digest.
    gym_runner_image_sbom_ref: str | None = None
    gym_runner_image_sbom_digest: str | None = None
    # Host checkout path for compose docker bind mounts and gym-runner build context.
    scaled_evals_host_dir: str | None = None
    # Shared memory for Ray inside one-shot gym-runner containers (Docker default is 64MB).
    gym_runner_shm_size: str = "2g"
    # Grace period for SIGTERM traps to close remote Daytona/OpenSandbox sandboxes.
    gym_runner_teardown_timeout_seconds: int = 60
    # Runtime-specific evaluation Job sizing. Queue workers keep their small
    # control-plane footprint; only Gym Jobs receive these resources.
    gym_job_cpu_request: str = "1"
    gym_job_cpu_limit: str = "4"
    gym_job_memory_request: str = "2Gi"
    gym_job_memory_limit: str = "8Gi"
    gym_job_shm_size: str = "2Gi"

    # --- Per-evaluation Switchyard dispatch --------------------------------
    # When an evaluation sets switchyard_profile_id, dispatch provisions one
    # Switchyard Deployment/Service/Secret/ConfigMap before launching the runner.
    # The profile should normally supply the image and namespace; these settings
    # are service defaults/fallbacks for local compose and cluster deployments.
    switchyard_image: str | None = None
    # Comma-separated imagePullSecret names automatically applied to every
    # ephemeral Switchyard Deployment. Hosted deployments point this at a
    # platform-provided pull secret so end users never need registry credentials.
    switchyard_image_pull_secrets: str = ""
    switchyard_namespace: str | None = None
    switchyard_kube_context: str | None = None
    switchyard_kube_insecure_skip_tls_verify: bool = False
    switchyard_drain_seconds: float = 300.0
    # Comma-separated exact hosts or leading-wildcard DNS suffixes approved for
    # external Switchyard profiles. Empty disables external endpoints.
    switchyard_external_allowed_hosts: str = ""

    # --- NMP Intake (post-run ATIF upload) ---------------------------------
    # Platform root; the client appends ``/apis/intake/v2/...``. No default: an
    # upload target is deployment-specific, and a wrong one would silently ship
    # trajectories off-site. Uploads are opt-in per evaluation via
    # ``intake_profile_id``, and the profile may carry its own ``intake_base_url``,
    # so this is only the fallback for a profile that omits one.
    # Auth was removed from this endpoint — no bearer token or intake-auth sidecar.
    intake_base_url: str = ""
    intake_source: str = "scaled-evals"
    intake_timeout_seconds: float = 30.0
    # When false (default), Intake upload failures are warnings only.
    intake_fail_on_error: bool = False

    def resolved_s3_public_endpoint(self) -> str:
        return self.s3_public_endpoint or self.s3_endpoint

    def resolved_object_store_bucket(self) -> str:
        return self.gcs_bucket or self.s3_bucket

    def resolved_database_url(self) -> str:
        base = self.database_url or _STANDALONE_DATABASE_URL
        params = [f"sslmode={self.database_ssl_mode}"]
        if self.database_ssl_root_cert:
            params.append(f"sslrootcert={self.database_ssl_root_cert}")
        if self.database_schema:
            # libpq hands this to the backend at connect time, so every pooled
            # connection, worker, and migration run resolves the vendored SQL's
            # unqualified table names into our schema before `public`.
            options = quote(f"-c search_path={self.database_schema},public", safe="")
            params.append(f"options={options}")
        if not params:
            return base
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}{'&'.join(params)}"


class _LazySettings:
    """Defer Settings() until first attribute access.

    The platform loads this plugin through a `nemo.services` entry point and
    swallows any exception raised during import, so an import-time
    ValidationError (a missing CREDENTIALS_ENCRYPTION_KEY, say) makes the whole
    plugin vanish with one log line. Deferring means the plugin always loads and
    misconfiguration surfaces as a real error on the request that needs it.
    """

    _instance: Settings | None = None

    def _resolve(self) -> Settings:
        # ponytail: unsynchronized; a startup race builds Settings twice and
        # discards one. Add a lock if construction ever gains side effects.
        if self._instance is None:
            # Required fields without a source-code default come from the
            # environment, which a type checker reading the signature cannot see.
            self._instance = Settings()  # ty: ignore[missing-argument]
        return self._instance

    # Any, not object: callers use the real Settings surface through this proxy,
    # and `object` makes every field unusable to a type checker.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


settings = _LazySettings()
