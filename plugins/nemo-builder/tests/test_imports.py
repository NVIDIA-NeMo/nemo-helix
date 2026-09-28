# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importing a published image, and the runtime layer -- through the pure half of a submit.

A published benchmark image IS the benchmark: its digest is what the leaderboard ran, and a
rebuild of it is a different image. So an import is a copy when the runtime accepts it as it is,
and a build `FROM` that digest plus the operator's runtime layer when it does not. Which of the two
is never the caller's to say: it follows from the runtime the set names.
"""

from __future__ import annotations

import pytest
from nemo_builder_plugin.compile import compile_build_set
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, JobOrigin
from nemo_builder_plugin.identity import (
    ImageIdentityError,
    ImageReference,
    normalize_pinned_reference,
    require_allowed_registry,
)
from nemo_builder_plugin.plan import BuildCompileError, BuildPlan
from nemo_builder_plugin.registry import ReferenceNotFound
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource, ImageSource
from nemo_builder_plugin.steps import ContextSource, FetchStepConfig, PushStepConfig, SuperviseStepConfig
from nemo_builder_plugin.submit import BuildConflict, InvalidBuildRequest, submit_build_set
from nemo_helix_plugin.entities import EntityConflictError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from pydantic import ValidationError

WORKSPACE = "default"


async def _no_job(name: str) -> dict:
    raise AssertionError(f"no job named {name} should have been looked up")


async def _ignore_job(request: CreateHelixJobRequest) -> None:
    return None


#: What a caller names -- here, an index digest.
INDEX = "sha256:" + "1" * 64
#: What the submit path resolves that index to for linux/amd64. What an import must pull.
AMD64 = "sha256:" + "2" * 64
TB = "docker.io/harborframework/terminal-bench"
LAYER = {"version": "3", "dockerfile": "USER root\nRUN apk add --no-cache bash\nUSER 1000"}


def _config(**overrides: object) -> BuilderConfig:
    settings: dict[str, object] = {
        "registry": "reg.example.com",
        "push_credential_secret": "registry-push-credential",
        "signing_key": "k8s://nhx-builds/cosign-key",
        "import_registries": ["docker.io"],
        "runtime_layers": {"harbor-sandbox": LAYER},
    }
    settings.update(overrides)
    return BuilderConfig.model_validate(settings)


def _import(name: str, *, image: str = f"{TB}@{INDEX}") -> BuildSpec:
    return BuildSpec(
        name=name,
        source=ImageSource(image=image),
        output=BuildOutput(repository=f"tb/{name}", tag="v4.0.0"),
    )


def _build(name: str, *, fileset: str = "fs-a", context_path: str | None = None) -> BuildSpec:
    return BuildSpec(
        name=name,
        source=FileSetSource(fileset=fileset, context_path=context_path),
        output=BuildOutput(repository=f"tb/{name}", tag="v1"),
    )


def _set(*specs: BuildSpec, runtime: str | None = None) -> BuildSet:
    return BuildSet(name="demo", revision=1, runtime=runtime, build_specs=list(specs))


def _plan(build_set: BuildSet, **config: object) -> BuildPlan:
    return BuildPlan.resolve(build_set, config=_config(**config), workspace=WORKSPACE)


def _resolved(build_set: BuildSet, **config: object) -> BuildPlan:
    plan = _plan(build_set, **config)
    return plan.with_upstream_manifests({image.name: AMD64 for image in plan.imports})


class TestThePinnedReference:
    def test_shorthand_expands_like_every_other_reference(self) -> None:
        assert normalize_pinned_reference(f"alpine@{INDEX}").normalized_ref == f"docker.io/library/alpine@{INDEX}"

    def test_a_tag_beside_the_digest_is_dropped(self) -> None:
        """The digest is the identity. Keeping the tag would hand parse_reference two."""
        ref = normalize_pinned_reference(f"{TB}:vpp-loss-divergence-environment-v4.0.0@{INDEX}")
        assert ref.normalized_ref == f"{TB}@{INDEX}"

    def test_a_port_is_not_mistaken_for_a_tag(self) -> None:
        ref = normalize_pinned_reference(f"localhost:5000/team/app@{INDEX}")
        assert (ref.registry, ref.repository) == ("localhost:5000", "team/app")

    @pytest.mark.parametrize("unpinned", [f"{TB}:v4.0.0", "alpine", "alpine:3.20"])
    def test_a_tag_alone_is_refused(self, unpinned: str) -> None:
        """A tag names whatever the publisher pushed last. An import publishes what was pinned."""
        with pytest.raises(ImageIdentityError, match="pinned by digest"):
            normalize_pinned_reference(unpinned)

    def test_the_allowlist_compares_normalized_hosts(self) -> None:
        """`alpine@...` IS docker.io. A string comparison would let the spelling pick the verdict."""
        ref = normalize_pinned_reference(f"alpine@{INDEX}")
        assert require_allowed_registry(ref, ["DOCKER.IO"]) is ref
        with pytest.raises(ImageIdentityError, match="not an allowed import source"):
            require_allowed_registry(ref, ["ghcr.io"])


class TestTheRequest:
    def test_a_source_without_a_type_is_still_a_fileset(self) -> None:
        """Every request written before imports existed omits `type`."""
        spec = BuildSpec.model_validate(
            {"name": "a", "source": {"fileset": "fs-a"}, "output": {"repository": "t/a", "tag": "v1"}}
        )
        assert isinstance(spec.source, FileSetSource)

    def test_an_image_source_is_normalized_when_parsed(self) -> None:
        spec = BuildSpec.model_validate(
            {
                "name": "a",
                "source": {"type": "image", "image": f"alpine:3.20@{INDEX}"},
                "output": {"repository": "t/a", "tag": "v1"},
            }
        )
        assert isinstance(spec.source, ImageSource)
        assert spec.source.image == f"docker.io/library/alpine@{INDEX}"

    def test_an_image_source_survives_a_client_that_drops_its_default_type(self) -> None:
        """`exclude_unset` drops a defaulted discriminator; the arm is inferred from `image`."""
        wire = BuildSpec(
            name="a", source=ImageSource(image=f"{TB}@{INDEX}"), output=BuildOutput(repository="t/a", tag="v1")
        ).model_dump(exclude_unset=True)
        assert "type" not in wire["source"]
        assert isinstance(BuildSpec.model_validate(wire).source, ImageSource)

    def test_an_image_source_by_tag_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="pinned by digest"):
            ImageSource(image=f"{TB}:v4.0.0")

    def test_a_dockerfile_on_an_import_is_refused_not_ignored(self) -> None:
        with pytest.raises(ValidationError, match="does not apply to an image source"):
            BuildSpec(
                name="a",
                source=ImageSource(image=f"{TB}@{INDEX}"),
                output=BuildOutput(repository="t/a", tag="v1"),
                dockerfile="Dockerfile",
            )

    @pytest.mark.parametrize("bad", ["Harbor", "-x", "x-", "a_b", "a" * 64])
    def test_a_runtime_name_is_a_plain_name(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _set(_build("a"), runtime=bad)


class TestTheOperatorsLayer:
    def test_a_layer_may_not_start_a_new_stage(self) -> None:
        """A `FROM` would discard the image the layer is meant to adapt."""
        with pytest.raises(ValidationError, match="must not contain FROM"):
            _config(runtime_layers={"x": {"version": "1", "dockerfile": "RUN true\n  from alpine\n"}})

    def test_a_from_inside_an_instruction_is_not_a_stage(self) -> None:
        _config(runtime_layers={"x": {"version": "1", "dockerfile": "RUN echo FROM here"}})

    def test_runtime_names_are_checked(self) -> None:
        with pytest.raises(ValidationError, match="invalid runtime name"):
            _config(runtime_layers={"Harbor": LAYER})

    @pytest.mark.parametrize("bad", ["docker.io/library", "https://ghcr.io", "", "host name"])
    def test_import_registries_are_hosts(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="registry hosts"):
            _config(import_registries=[bad])

    def test_import_registries_are_normalized_like_references(self) -> None:
        """Compared with normalized references, so an entry spelled any other way would match
        nothing and refuse the imports it was written to allow."""
        config = _config(import_registries=[" GHCR.io ", "index.docker.io", "registry-1.docker.io"])
        assert config.import_registries == ["ghcr.io", "docker.io", "docker.io"]

    def test_layers_arrive_through_the_environment_as_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment beats the config file -- and the config file is a ConfigMap."""
        monkeypatch.setenv("NEMO_BUILDER_RUNTIME_LAYERS", '{"agent": {"version": "7", "dockerfile": "USER 1000"}}')
        monkeypatch.setenv("NEMO_BUILDER_IMPORT_REGISTRIES", '["ghcr.io"]')
        config = BuilderConfig()
        assert config.runtime_layers["agent"].version == "7"
        assert config.import_registries == ["ghcr.io"]


class TestWhatThePlanDecides:
    def test_imports_are_off_until_the_operator_allows_a_registry(self) -> None:
        with pytest.raises(BuildCompileError, match="imports are disabled"):
            _plan(_set(_import("env")), import_registries=[])

    def test_a_registry_not_on_the_list_is_refused_before_anything_exists(self) -> None:
        with pytest.raises(BuildCompileError, match="not an allowed import source"):
            _plan(_set(_import("env", image=f"ghcr.io/org/app@{INDEX}")))

    def test_an_unknown_runtime_is_a_deployment_gap(self) -> None:
        with pytest.raises(BuildCompileError, match="runtime 'astra-skill-eval' is not configured"):
            _plan(_set(_build("a"), runtime="astra-skill-eval"))

    def test_with_no_runtime_an_import_is_a_copy(self) -> None:
        plan = _plan(_set(_import("env")))
        (image,) = plan.images
        assert plan.is_copy(image)
        assert plan.groups() == [], "nothing about a copy runs, so nothing is built"

    def test_with_a_runtime_an_import_is_derived_and_built_last(self) -> None:
        plan = _plan(_set(_import("env"), _build("sidecar"), runtime="harbor-sandbox"))
        assert not plan.is_copy(plan.images[0])
        assert [(source, [i.name for i in images]) for source, images in plan.groups()] == [
            (ContextSource(fileset="default/fs-a"), ["demo-1-1"]),
            (None, ["demo-1-0"]),
        ]

    def test_each_derived_import_gets_a_sandbox_of_its_own(self) -> None:
        """Each build runs its upstream publisher's binaries, which must not reach another's image."""
        plan = _plan(_set(_import("env"), _import("verifier"), runtime="harbor-sandbox"))
        assert [(source, [i.name for i in images]) for source, images in plan.groups()] == [
            (None, ["demo-1-0"]),
            (None, ["demo-1-1"]),
        ]
        assert plan.runtime_layer is not None and plan.runtime_layer.label == "harbor-sandbox@3"

    def test_an_import_pulls_the_platform_manifest_not_what_the_caller_named(self) -> None:
        """For an index those differ, and the row is checked against the manifest."""
        plan = _resolved(_set(_import("env")))
        assert plan.images[0].pinned_upstream == f"{TB}@{AMD64}"

    def test_every_import_must_be_resolved(self) -> None:
        plan = _plan(_set(_import("env"), _import("verifier")))
        with pytest.raises(ValueError, match="demo-1-1"):
            plan.with_upstream_manifests({"demo-1-0": AMD64})


class TestWhatCompiles:
    def _compile(self, build_set: BuildSet, **config: object):
        return compile_build_set(_resolved(build_set, **config), config=_config(**config))

    def test_a_set_of_copies_has_no_build_step(self) -> None:
        """No sandbox, and so no use of the pod-create grant, for a set nothing about runs."""
        spec = self._compile(_set(_import("env"), _import("verifier")))
        assert [step.name for step in spec.steps] == ["fetch", "push"]

    def test_fetch_is_told_to_copy_the_resolved_manifest(self) -> None:
        spec = self._compile(_set(_import("env")))
        fetch = FetchStepConfig.model_validate(spec.steps[0].config)
        assert fetch.sources == []
        assert [(i.image, i.ref, i.platform, i.mode) for i in fetch.imports] == [
            ("demo-1-0", f"{TB}@{AMD64}", "linux/amd64", "copy")
        ]

    def test_a_copy_is_published_like_a_build(self) -> None:
        """Both tags, signed the same way -- push cannot tell a copy from a build."""
        spec = self._compile(_set(_import("env")))
        push = PushStepConfig.model_validate(spec.steps[-1].config)
        assert push.images[0].tags == [
            "reg.example.com/default/tb/env:v4.0.0",
            "reg.example.com/default/tb/env:default--demo-1-0",
        ]

    def test_a_mixed_set_builds_only_what_is_not_copied(self) -> None:
        spec = self._compile(_set(_import("env"), _build("sidecar")))
        build = SuperviseStepConfig.model_validate(spec.steps[1].config)
        assert [[i.image for i in g.images] for g in build.groups] == [["demo-1-1"]]

    def test_a_derived_import_builds_its_own_dockerfile_in_the_import_group(self) -> None:
        spec = self._compile(_set(_import("env"), runtime="harbor-sandbox"))
        fetch = FetchStepConfig.model_validate(spec.steps[0].config)
        build = SuperviseStepConfig.model_validate(spec.steps[1].config)
        assert fetch.imports[0].mode == "derive"
        assert (build.groups[0].source, build.groups[0].images[0].dockerfile) == (None, "Dockerfile")

    def test_the_layer_goes_on_each_fetched_dockerfile_once(self) -> None:
        """Two specs building one Dockerfile get the layer once, not twice."""
        spec = self._compile(_set(_build("a"), _build("b"), runtime="harbor-sandbox"))
        fetch = FetchStepConfig.model_validate(spec.steps[0].config)
        assert fetch.runtime_layer is not None and fetch.runtime_layer.label == "harbor-sandbox@3"
        assert [(d.source.fileset, d.dockerfile) for d in fetch.dockerfiles] == [("default/fs-a", "Dockerfile")]

    def test_without_a_runtime_no_dockerfile_is_touched(self) -> None:
        spec = self._compile(_set(_build("a")))
        fetch = FetchStepConfig.model_validate(spec.steps[0].config)
        assert (fetch.runtime_layer, fetch.dockerfiles) == (None, [])

    def test_an_unresolved_import_does_not_compile(self) -> None:
        """Pulling something other than what the row will be checked against is not an option."""
        with pytest.raises(ValueError, match="not a resolved import"):
            compile_build_set(_plan(_set(_import("env"))), config=_config())


class _Entities:
    def __init__(self) -> None:
        self.rows: dict[str, ContainerImage] = {}

    async def create(self, entity: ContainerImage) -> ContainerImage:
        if entity.name in self.rows:
            raise EntityConflictError(entity.name)
        self.rows[entity.name] = entity
        return entity

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage:
        return self.rows[name]


class TestTheSubmit:
    @staticmethod
    async def _submit(build_set: BuildSet, resolve) -> tuple[_Entities, list[CreateHelixJobRequest]]:
        entities, jobs = _Entities(), []

        async def create_job(request: CreateHelixJobRequest) -> None:
            jobs.append(request)

        await submit_build_set(
            build_set,
            config=_config(),
            workspace=WORKSPACE,
            entity_client=entities,
            create_job=create_job,
            get_job_fields=_no_job,
            resolve_upstream=resolve,
        )
        return entities, jobs

    @pytest.mark.asyncio
    async def test_the_row_records_what_was_resolved_and_for_which_platform(self) -> None:
        asked: list[tuple[str, str]] = []

        async def resolve(reference: ImageReference, platform: str) -> str:
            asked.append((reference.normalized_ref, platform))
            return AMD64

        entities, _ = await self._submit(_set(_import("env")), resolve)

        assert asked == [(f"{TB}@{INDEX}", "linux/amd64")]
        origin = entities.rows["demo-1-0"].provenance.built_by
        assert isinstance(origin, JobOrigin)
        assert origin.upstream is not None
        assert (origin.upstream.image_ref, origin.upstream.manifest_digest) == (f"{TB}@{INDEX}", AMD64)
        assert origin.is_copy and origin.runtime_layer is None

    @pytest.mark.asyncio
    async def test_a_derived_row_names_its_runtime(self) -> None:
        async def resolve(reference: ImageReference, platform: str) -> str:
            return AMD64

        entities, _ = await self._submit(_set(_import("env"), _build("sidecar"), runtime="harbor-sandbox"), resolve)
        origins = [row.provenance.built_by for row in entities.rows.values()]
        assert all(isinstance(o, JobOrigin) and o.runtime_layer == "harbor-sandbox@3" for o in origins)
        assert not any(isinstance(o, JobOrigin) and o.is_copy for o in origins)

    @pytest.mark.asyncio
    async def test_an_upstream_that_does_not_resolve_is_refused_before_any_row(self) -> None:
        async def resolve(reference: ImageReference, platform: str) -> str:
            raise ReferenceNotFound(f"{reference.normalized_ref} does not resolve")

        entities, jobs = _Entities(), []

        async def create_job(request: CreateHelixJobRequest) -> None:
            jobs.append(request)

        with pytest.raises(InvalidBuildRequest, match="does not resolve"):
            await submit_build_set(
                _set(_import("env")),
                config=_config(),
                workspace=WORKSPACE,
                entity_client=entities,
                create_job=create_job,
                get_job_fields=_no_job,
                resolve_upstream=resolve,
            )
        assert entities.rows == {} and jobs == []

    @pytest.mark.asyncio
    async def test_a_different_upstream_at_the_same_revision_is_refused(self) -> None:
        """Adopted, the row would name a base its job was never given."""

        async def resolve(reference: ImageReference, platform: str) -> str:
            return AMD64

        entities, _ = await self._submit(_set(_import("env")), resolve)
        other = _set(_import("env"))
        other.build_specs[0].source = ImageSource(image=f"{TB}@sha256:{'9' * 64}")
        with pytest.raises(BuildConflict):
            await submit_build_set(
                other,
                config=_config(),
                workspace=WORKSPACE,
                entity_client=entities,
                create_job=_ignore_job,
                get_job_fields=_no_job,
                resolve_upstream=resolve,
            )

    @pytest.mark.asyncio
    async def test_a_copys_push_is_held_to_the_upstream_digest(self) -> None:
        async def resolve(reference: ImageReference, platform: str) -> str:
            return AMD64

        _, jobs = await self._submit(_set(_import("env")), resolve)
        push = PushStepConfig.model_validate(jobs[0].platform_spec.steps[-1].config)
        assert push.images[0].expected_digest == AMD64

    @pytest.mark.asyncio
    async def test_a_submit_path_without_a_resolver_cannot_take_imports(self) -> None:
        with pytest.raises(RuntimeError, match="no upstream resolver"):
            await self._submit(_set(_import("env")), None)


class TestSourcesAreUnambiguous:
    """A source is one arm or the other; a field belonging to neither, or to both, is refused."""

    PINNED = f"{TB}@{INDEX}"

    def test_an_import_survives_its_own_dump(self) -> None:
        """The job stores exactly this dump, and an SDK client sends one; neither may be refused."""
        build_set = _set(_import("env"), _build("sidecar"))
        again = BuildSet.model_validate(build_set.model_dump(mode="json"))
        assert again == build_set
        assert again.build_specs[0].dockerfile is None and again.build_specs[1].dockerfile == "Dockerfile"

    @pytest.mark.parametrize(
        "source",
        [
            {"fileset": "fs-a", "image": PINNED},
            {"type": "fileset", "fileset": "fs-a", "image": PINNED},
            {"type": "image", "image": PINNED, "context_path": "env"},
        ],
        ids=["both-fields", "fileset-with-image", "image-with-context-path"],
    )
    def test_a_field_from_the_other_arm_is_refused(self, source: dict) -> None:
        spec = {"name": "a", "source": source, "output": {"repository": "t/a", "tag": "v1"}}
        with pytest.raises(ValidationError, match="Extra inputs"):
            BuildSpec.model_validate(spec)
