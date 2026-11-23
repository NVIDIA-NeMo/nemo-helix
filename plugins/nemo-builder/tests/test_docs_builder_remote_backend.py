# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the operator's guide to the remote backend on the builder's settings, its step driver, what it answers
and the chart's manifests the guide copies.

Offline: the chart's files are read as they are written, not rendered, and no deployment is contacted.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import itertools
import re
from datetime import timedelta
from functools import cache
from pathlib import Path
from typing import Any, get_args

import httpx
import pytest
import yaml
from nemo_builder_plugin import service
from nemo_builder_plugin.backend import BackendRejectedError
from nemo_builder_plugin.backends import BACKENDS
from nemo_builder_plugin.config import BuilderConfig, RemoteConfig
from nemo_builder_plugin.entities import ContainerImage, ExternalImage, ImageRuntimeConfig
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.remote import CANCEL_MARGIN_SECONDS, STEP_NAME, RemoteBackend
from nemo_builder_plugin.run.fetch import MAX_UNPACKED_BYTES, MAX_UNPACKED_ENTRIES
from nemo_builder_plugin.run.remote import (
    COMPLETION_SECONDS,
    START_PATIENCE_SECONDS,
    Clock,
    ImageFailed,
    Stage,
    _complete,
    step_main,
)
from nemo_builder_plugin.schema import CODE_BACKEND_REJECTED, MAX_BUILD_SPECS, BuildSet
from nemo_builder_plugin.settle import JOB_GRACE
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.jobs.execution_profiles import (
    KubernetesJobExecutionProfile,
    KubernetesJobExecutionProfileConfig,
)
from pydantic.fields import FieldInfo

ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs/set-up/helm/builder-remote-backend.mdx"
HARBOR_GUIDE = ROOT / "docs/set-up/helm/harbor-sandbox-profiles.mdx"
README = ROOT / "plugins/nemo-builder/README.md"
PLUGIN = ROOT / "plugins/nemo-builder/src/nemo_builder_plugin"
JOBS_ON_KUBERNETES = ROOT / "services/core/jobs/src/nhx/core/jobs/controllers/backends/kubernetes"
#: Where the sentences the guide quotes are spelt: the plugin, and the Jobs backend the step runs on, whose
#: message for a step it could not schedule, that ran out of time, whose image could not be pulled or whose pod
#: failed, a failed image's row repeats.
SOURCES = (
    *sorted(PLUGIN.rglob("*.py")),
    JOBS_ON_KUBERNETES / "kubernetes_job.py",
    JOBS_ON_KUBERNETES / "common.py",
)
#: Where Jobs' launcher, which starts the step program, spells the lines it ends a step with before the program
#: runs, and the errors those lines wrap. Go, so read as text, and apart from SOURCES.
LAUNCHER = ROOT / "services/core/jobs/jobs-launcher"
LAUNCHER_SOURCES = (LAUNCHER / "cmd/run.go", LAUNCHER / "nhxclient/client.go")
#: Where a deployment that builds with `remote` may raise a refusal a submit answers with `code`
#: `backend_rejected`: anywhere in the plugin but the in-cluster backend, which such a deployment doesn't run.
REFUSALS = tuple(path for path in sorted(PLUGIN.rglob("*.py")) if path.name != "execution.py")
#: Where evals spells the errors the guide's last section names. Apart from SOURCES: nothing evals says may
#: stand for a sentence of the builder's.
EVALS_SOURCES = (
    ROOT / "plugins/nemo-evals/src/nemo_evals/image_builds/runner.py",
    ROOT / "plugins/nemo-evals/src/nemo_evals/agent_layers/runner.py",
)
#: What the guide copies from the chart, for a namespace the chart doesn't create.
CHART_VALUES = ROOT / "k8s/helm/values.yaml"
CHART_BUILD_NAMESPACE = ROOT / "k8s/helm/templates/builder/build-namespace.yaml"
#: The chart, by whose paths the guide names the templates it renders to read a value back.
CHART = CHART_VALUES.parent
#: Each page the guide links into at a heading, by the link's address, and the file that holds its headings.
#: The guide's links into itself have no address.
LINKED = {
    "": GUIDE,
    "/documentation/kubernetes-deployment/setup/helm/harbor-sandbox-profiles": HARBOR_GUIDE,
    "/documentation/kubernetes-deployment/setup/helm/install": ROOT / "docs/set-up/helm/install.mdx",
    "/documentation/access-control/security-model": ROOT / "docs/auth/security-model.mdx",
    "/documentation/access-control/deployment": ROOT / "docs/auth/deployment/configuration.mdx",
    "https://github.com/NVIDIA-NeMo/nemo-helix/blob/main/plugins/nemo-builder/README.md": README,
}

SETTINGS = "## Select the backend"
STEP_IMAGE = "## Build the step image"
NAMESPACE = "## Create the step's namespace"
PROFILE = "## Add the Jobs profile"
CREDENTIAL = "## Give the step its credential"
VERIFY = "## Apply and verify"
TROUBLESHOOTING = "## Troubleshooting"
SUBMIT_REFUSED = "### A submit is refused"
IMAGE_FAILS = "### An image fails"
COMPLETION_REFUSED = "### A completion is refused"
#: Here the guide names evals' errors, which the builder's source doesn't spell.
EVALS_ERRORS = "### An evals build fails after a switch"

#: The `remote` keys whose value in the recipe is an example, not the default: each is unset by default.
EXAMPLES = {"registry", "repository", "image", "credential_secret"}
#: The shortest string of the sources that is taken for part of a quoted sentence.
MIN_SPELT = 12
#: How much of a quoted sentence the sources must spell. Not all of it: where the code fills in a value the
#: guide may name one, such as `kubernetes_job` or `RuntimeError`, and a quoted value keeps its quotation marks.
SPELT_SHARE = 2 / 3
PLACEHOLDER = re.compile(r"<[^<>]+>")
#: How many sentences the troubleshooting tables quote, each in a cell of its own.
QUOTED_SENTENCES = 59
#: What the troubleshooting section quotes outside a cell of its own: the driver's line for an image it didn't
#: build and what may end its reason, what Jobs may add to its message for a pod that failed, and the API's log
#: lines for a completion it refused or put off.
PROSE_QUOTES = (
    "NHX_BUILD_FAILED <image>: <reason>",
    "[build <id>]",
    "the build service could not be told to stop it, it may still be running",
    "a build may be running, started with key <key>",
    "One or more tasks are in error state",
    "completing <workspace>/<image> refused: <detail>",
    "completing <workspace>/<image> put off: its row records no build job yet",
    "could not tell which job is completing <workspace>/<image>",
)
#: The launcher's lines the guide tells an operator to look for at the end of a failed step's container log, and
#: the error one of them ends with when the API answered.
LAUNCHER_LINES = (
    "Error fetching secrets: failed to fetch secret <workspace>/<name>: <error>",
    "failed to get secret: status code <n>",
    "Error starting command: <error>",
)
#: A number as the guide writes it out in a sentence.
NUMBERS = "zero one two three four five six seven eight nine ten eleven twelve".split()
#: What the API answers a completion whose shape it doesn't know with. The framework's answer, not a refusal of
#: the builder's own, so the guide's table of those has no row for it.
UNKNOWN_SHAPE = 422


def _text(path: Path = GUIDE) -> str:
    return path.read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """One second-level section of the guide, from its heading to the next."""
    return heading + _text().split(heading, 1)[1].split("\n## ", 1)[0]


def _subsection(heading: str) -> str:
    """One third-level section of the guide, from its heading to the next heading of either level."""
    return heading + re.split(r"\n#{2,3} ", _text().split(heading, 1)[1], maxsplit=1)[0]


def _yaml_documents(text: str) -> list[dict[str, Any]]:
    """Every YAML document in ``text``'s blocks: a block of manifests holds several, parted by `---`."""
    blocks = re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)
    return [document for block in blocks for document in yaml.safe_load_all(block)]


def _slug(heading: str) -> str:
    return re.sub(r"\s+", "-", re.sub(r"[^a-z0-9 -]", "", heading.lower()).strip())


def _headings(path: Path) -> set[str]:
    return {_slug(heading) for heading in re.findall(r"^#{2,4} (.+)$", _text(path), re.MULTILINE)}


def _settings() -> dict[str, FieldInfo]:
    """Every key the guide's settings table documents: `backend`, and each `remote` key."""
    remote = {f"remote.{name}": field for name, field in RemoteConfig.model_fields.items()}
    return {"backend": BuilderConfig.model_fields["backend"], **remote}


def _backend_recipe() -> dict[str, Any]:
    (recipe,) = _yaml_documents(_section(SETTINGS))
    return recipe["platformConfig"]["builder"]


def _profile_recipe() -> KubernetesJobExecutionProfile:
    (recipe,) = _yaml_documents(_section(PROFILE))
    (executor,) = recipe["platformConfig"]["jobs"]["executors"]
    assert executor["config"].keys() <= KubernetesJobExecutionProfileConfig.model_fields.keys()
    return KubernetesJobExecutionProfile.model_validate(executor)


def _request_plan(config: BuilderConfig) -> BuildPlan:
    """The build the guide's own request asks for, planned as a submit plans it (`submit_build_set`)."""
    section = _section(VERIFY)
    (workspace,) = set(re.findall(r"/workspaces/([a-z0-9-]+)/builds", section))
    (body,) = re.findall(r"-d '(\{.*\})'", section)
    build_set = BuildSet.model_validate_json(body)
    return BuildPlan.resolve(build_set.with_default_platform(config.default_platform), workspace=workspace)


def _api_ingress_peer(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Whom the guide's rule for the API's NetworkPolicy lets in."""
    (values,) = [manifest for manifest in manifests if "networkPolicies" in manifest]
    (rule,) = values["networkPolicies"]["api"]["extraIngress"]
    (peer,) = rule["from"]
    return peer


def _chart_controller_rules() -> list[dict[str, Any]]:
    """The rules of the Role the chart gives the Jobs controller in the build namespace.

    The template is not YAML until Helm renders it, so its documents are told apart as text, and the Role is
    read from its rules on, which hold nothing for Helm to fill in.
    """
    documents = _text(CHART_BUILD_NAMESPACE).split("\n---")
    account = 'name: {{ include "nhx-core.controllerServiceAccountName" . }}'
    (role,) = [document for document in documents if "\nkind: Role\n" in document and account in document]
    return yaml.safe_load(role[role.index("\nrules:") :])["rules"]


def _parts(node: ast.expr) -> tuple[str | None, ...]:
    """A string of the sources in its parts: None where an f-string fills in a value."""
    if isinstance(node, ast.JoinedStr):
        return tuple(
            part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else None
            for part in node.values
        )
    assert isinstance(node, ast.Constant) and isinstance(node.value, str), ast.dump(node)
    return (node.value,)


@cache
def _strings(*paths: Path) -> list[tuple[str | None, ...]]:
    """Every string the files at ``paths`` spell, each in its parts. Not a docstring: it says what the code
    does, and may go on saying it once the code says something else."""
    strings: list[tuple[str | None, ...]] = []
    for path in paths:
        tree = ast.parse(_text(path))
        formatted = [node for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)]
        # A part of an f-string is not a string of its own.
        skipped = {id(part) for node in formatted for part in node.values}
        skipped |= {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Expr)}
        for node in ast.walk(tree):
            if id(node) in skipped:
                continue
            if isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                strings.append(_parts(node))
    return strings


def _spelt_share(sentence: str) -> float:
    """How much of ``sentence``, its placeholders aside, the sources spell: in one string, or in the several
    that a message is put together from."""
    covered = [False] * len(sentence)
    for parts in _strings(*SOURCES):
        literals = [part for part in parts if part]
        # Skipped: a string too short to be taken for part of a sentence, and one whose longest part isn't in the
        # sentence. The second check is cheap, and most strings fail it.
        if sum(map(len, literals)) < MIN_SPELT or max(literals, key=len) not in sentence:
            continue
        pattern = re.compile("".join(".+?" if part is None else f"({re.escape(part)})" for part in parts))
        for found in pattern.finditer(sentence):
            for group in range(1, pattern.groups + 1):
                start, end = found.span(group)
                covered[start:end] = [True] * (end - start)
    placeholders = {position for found in PLACEHOLDER.finditer(sentence) for position in range(*found.span())}
    fixed = [position for position in range(len(sentence)) if position not in placeholders]
    return sum(covered[position] for position in fixed) / len(fixed)


def _said_in_one_string(quote: str, strings: list[tuple[str | None, ...]]) -> bool:
    """Whether one of ``strings`` holds ``quote``, with whatever the code fills in where the quote has a
    placeholder.

    For what the guide quotes a part of, and for a line a logger fills in (`%s`): :func:`_spelt_share` looks for
    a whole string of the sources inside the sentence, and these are inside a string of the sources instead.
    """
    pattern = re.compile(".+?".join(re.escape(piece) for piece in PLACEHOLDER.split(quote)))
    return any(pattern.search("".join("\0" if part is None else part for part in parts)) for parts in strings)


def _quoted_sentences() -> list[str]:
    """Each sentence a troubleshooting table quotes: a cell that is one code span, and not a code or a status."""
    return [cell for cell in re.findall(r"\| `([^`]+)` (?=\|)", _section(TROUBLESHOOTING)) if " " in cell]


@cache
def _launcher_formats() -> list[re.Pattern[str]]:
    """Each format the launcher spells, as a pattern: a group where a verb fills a value in. A format with less
    fixed text than :data:`MIN_SPELT` is left out: `%s=%s` would stand for any line."""
    formats: list[re.Pattern[str]] = []
    for path in LAUNCHER_SOURCES:
        for literal in re.findall(r'"((?:[^"\\]|\\.)*)"', _text(path)):
            fixed = re.split(r"%[a-z]", literal.removesuffix(r"\n"))
            if len(fixed) > 1 and sum(map(len, fixed)) >= MIN_SPELT:
                formats.append(re.compile("(.+)".join(map(re.escape, fixed))))
    return formats


def _launcher_says(line: str) -> bool:
    """Whether the launcher's formats make ``line``: what a format fills in is a placeholder of the guide's or,
    for an error it wraps, what another format makes."""
    if PLACEHOLDER.fullmatch(line):
        return True
    made = [found for pattern in _launcher_formats() if (found := pattern.fullmatch(line))]
    return any(all(map(_launcher_says, found.groups())) for found in made)


def _refusals() -> list[re.Pattern[str]]:
    """What each refusal raised in :data:`REFUSALS` says, as a pattern: anything where the code fills in a value."""
    refusals: list[re.Pattern[str]] = []
    for path in REFUSALS:
        for node in ast.walk(ast.parse(_text(path))):
            raised = node.exc if isinstance(node, ast.Raise) else None
            if isinstance(raised, ast.Call) and isinstance(raised.func, ast.Name):
                if raised.func.id == BackendRejectedError.__name__:
                    (message,) = raised.args
                    parts = _parts(message)
                    refusals.append(re.compile("".join(".+" if part is None else re.escape(part) for part in parts)))
    return refusals


class _Answering:
    """The builder's routes, to a step whose every completion is answered with one status."""

    def __init__(self, status: int) -> None:
        self._answer = httpx.Response(status, request=httpx.Request("POST", "http://builder.test"))

    def row(self, image: str) -> ContainerImage:
        raise NotImplementedError

    def complete(
        self, image: str, digest: str, runtime_config: ImageRuntimeConfig | None, external: ExternalImage | None = None
    ) -> ContainerImage:
        raise NemoHTTPError(self._answer)


def _completion_reason(status: int) -> str:
    """The reason the driver fails a built image with when its completion goes on being answered with
    ``status``: on a clock that leaves it one try before the time for completing is up."""
    ticks = itertools.count(step=COMPLETION_SECONDS)
    clock = Clock(monotonic=lambda: next(ticks), sleep=lambda wait: None)
    built = ExternalImage(image_id="image-1", build_id="build-1")
    with pytest.raises(ImageFailed) as failed:
        _complete(_Answering(status), "image", f"sha256:{'0' * 64}", None, built, clock=clock, cancelled=lambda: False)
    return str(failed.value)


def test_the_settings_table_has_every_key_at_its_default() -> None:
    rows = dict(re.findall(r"^\| `([a-z_.]+)` \| ([^|]+) \|", _section(SETTINGS), re.MULTILINE))
    fields = _settings()
    assert rows.keys() == fields.keys()
    for key, cell in rows.items():
        token = re.search(r"`([^`]+)`", cell)
        assert token or cell.strip() == "unset", f"{key}: no default in {cell!r}"
        documented = yaml.safe_load(token.group(1)) if token else None
        default = fields[key].get_default(call_default_factory=True)
        assert documented == default, f"{key}: the guide says {cell.strip()!r}, the model says {default!r}"


def test_the_backend_recipe_validates_and_shows_the_other_keys_at_their_defaults() -> None:
    recipe = _backend_recipe()
    assert BuilderConfig.model_validate(recipe).backend == "remote"
    shown, defaults = recipe["remote"], RemoteConfig()
    assert shown.keys() == RemoteConfig.model_fields.keys()
    for key in shown.keys() - EXAMPLES:
        assert shown[key] == getattr(defaults, key), key


def test_the_profile_recipe_validates_and_can_run_the_backend_recipe() -> None:
    """Held to the check every submit runs, not to a copy of it: on the build the guide itself submits."""
    config = BuilderConfig.model_validate(_backend_recipe())
    RemoteBackend(config, [_profile_recipe()]).check(_request_plan(config))


def test_the_profile_recipe_takes_its_launcher_image_as_the_charts_profiles_do() -> None:
    """The chart keeps a profile the values list as written, so an image name copied in would stay at the release
    it was copied from. The chart's own expression is filled in again at every upgrade."""
    platform_config = yaml.safe_load(_text(CHART_VALUES))["baseHelixConfig"]
    (image,) = set(re.findall(r'launcher_image: \{\{ include "([\w.-]+)" \. \| quote \}\}', platform_config))
    assert _profile_recipe().config.launcher_image == f'{{{{ include "{image}" . }}}}'


def test_the_build_the_guide_submits_is_named_as_its_commands_name_it() -> None:
    """The commands that follow the submit name its fileset, job, image and step: each as a submit names it."""
    section = _section(VERIFY)
    plan = _request_plan(BuilderConfig.model_validate(_backend_recipe()))
    (image,) = plan.images
    workspace, _, fileset = image.source.fileset.partition("/")
    named = [
        f"nemo files filesets create {fileset} --workspace {workspace}",
        f'nemo files upload "$CTX/" {fileset} --workspace {workspace}',
        f"`job` is `{plan.job_name}`, and the one image, `{image.name}`, is `pending`",
        f'/workspaces/{plan.workspace}/container-images/{image.name}"',
        f"nemo jobs get-status {plan.job_name} --workspace {plan.workspace}",
        f"nemo jobs tail {plan.job_name} --workspace {plan.workspace} --step-id {STEP_NAME}",
    ]
    missing = [text for text in named if text not in section]
    assert not missing, f"the guide's commands don't name what its request makes: {missing}"


def test_the_profile_table_names_real_keys_and_their_defaults() -> None:
    rows = dict(re.findall(r"^\| `config\.([a-z_]+)` \| ([^|]+) \|", _section(PROFILE), re.MULTILINE))
    fields = KubernetesJobExecutionProfileConfig.model_fields
    assert rows and rows.keys() <= fields.keys()
    for key in ("max_active_jobs", "ttl_seconds_active", "ttl_seconds_before_active"):
        assert re.search(rf"If unset, {fields[key].default}\b", rows[key]), key


def test_the_guide_states_the_numbers_the_code_holds() -> None:
    """A floor an operator sizes a limit by, a wait they sit out: each is the code's number, where the guide
    says it."""
    completing = f"{NUMBERS[int(COMPLETION_SECONDS) // 60]} minutes"
    starting = f"{NUMBERS[int(START_PATIENCE_SECONDS) // 60]} minutes"
    pulling = KubernetesJobExecutionProfileConfig.model_fields["ttl_seconds_image_pull"].default
    stated = [
        f"at most {MAX_BUILD_SPECS}), plus {CANCEL_MARGIN_SECONDS} seconds",
        f"at most {MAX_BUILD_SPECS}), plus a {CANCEL_MARGIN_SECONDS}s margin)",
        f"more than {MAX_UNPACKED_BYTES // 1024**3} GiB or {MAX_UNPACKED_ENTRIES:,} entries",
        f"up to {completing} of reporting a built image",
        f"waits that out for up to {completing}",
        f"kept failing for {completing}",
        f"gives up after {completing}",
        f"tried again for up to {starting}",
        f"couldn't start the build for {starting}",
        f"The row is {NUMBERS[JOB_GRACE // timedelta(minutes=1)]} minutes old",
        f"`ttl_seconds_image_pull`, {pulling} if unset",
    ]
    missing = [text for text in stated if text not in _text()]
    assert not missing, f"the guide doesn't say, or says with another number: {missing}"
    # The least a setting takes, in the settings table's row for it.
    least = {
        name: bound.ge
        for name, field in RemoteConfig.model_fields.items()
        for bound in field.metadata
        if hasattr(bound, "ge")
    }
    assert least, "no `remote` setting has a least value"
    for name, value in least.items():
        row = re.search(rf"^\| `remote\.{name}` \|.*$", _section(SETTINGS), re.MULTILINE)
        assert row and f"Below {value}: the section is invalid" in row.group(0), name


def test_the_manifests_the_guide_copies_from_the_chart_are_the_charts() -> None:
    """What the guide copies from the chart for a namespace the chart doesn't create: the Jobs controller's Role
    and a placeholder under the OpenSandbox key's name, which an operator applies there by hand, and the rule in
    the chart's values that lets that namespace's job pods in to the API."""
    manifests = _yaml_documents(_section(NAMESPACE))
    values = yaml.safe_load(_text(CHART_VALUES))
    (role,) = [manifest for manifest in manifests if manifest.get("kind") == "Role"]
    assert role["rules"] == _chart_controller_rules()
    (secret,) = [manifest for manifest in manifests if manifest.get("kind") == "Secret"]
    assert secret["metadata"]["name"] == values["opensandbox"]["apiKeySecret"]
    assert list(secret["stringData"]) == [values["opensandbox"]["apiKeySecretKey"]]
    job_pods = values["networkPolicies"]["api"]["managedJobs"]["podSelector"]
    assert _api_ingress_peer(manifests)["podSelector"] == job_pods


def test_the_manifests_create_what_the_profile_recipe_names() -> None:
    config = _profile_recipe().config
    manifests = _yaml_documents(_section(NAMESPACE))
    created = {manifest["kind"]: manifest["metadata"] for manifest in manifests if "kind" in manifest}
    assert created["Namespace"]["name"] == config.namespace
    assert created["ServiceAccount"]["name"] == config.service_account_name
    assert {metadata["namespace"] for kind, metadata in created.items() if kind != "Namespace"} == {config.namespace}
    selected = _api_ingress_peer(manifests)["namespaceSelector"]["matchLabels"]
    assert selected == {"kubernetes.io/metadata.name": config.namespace}
    # A command the guide runs in a namespace it names, not in the release's, runs where the step does.
    assert set(re.findall(r" -n ([a-z0-9-]+)", _text())) == {config.namespace}


def test_the_templates_the_guide_renders_are_the_charts() -> None:
    """A later values file's list replaces an earlier one's, and only a rendering shows what is left of it. The
    guide names each template it renders by its path in the chart, and a path the chart doesn't hold renders
    nothing to read."""
    rendered = re.findall(r"--show-only (\S+)", _section(VERIFY))
    missing = [path for path in rendered if not (CHART / path).is_file()]
    assert rendered and not missing, f"the chart has no template at: {missing}"
    # The rule the guide adds to the API's policy is read back from a template that renders that policy's list.
    (values,) = [manifest for manifest in _yaml_documents(_section(NAMESPACE)) if "networkPolicies" in manifest]
    (key,) = values["networkPolicies"]["api"]
    templates = [_text(CHART / path) for path in rendered]
    reads = [".Values.networkPolicies.api" in template and f".{key}" in template for template in templates]
    assert any(reads), f"no template the guide renders reads networkPolicies.api.{key}"


def test_the_mounted_credential_recipe_validates_and_is_where_the_step_program_reads() -> None:
    (recipe,) = _yaml_documents(_section(CREDENTIAL))
    config = recipe["platformConfig"]
    assert RemoteConfig.model_validate(config["builder"]["remote"]).credential_secret is None
    (executor,) = config["jobs"]["executors"]
    storage = KubernetesJobExecutionProfile.model_validate(executor).config.storage
    (volume,) = storage.additional_volumes
    (mount,) = storage.additional_volume_mounts
    assert volume.secret is not None and mount.name == volume.name and mount.read_only
    # The Secret the guide creates is the one mounted, where the profile runs, and its key is the file read.
    created = re.search(
        r"kubectl create secret generic (\S+) -n (\S+) \\\n\s+--from-file=([\w.-]+)=", _section(CREDENTIAL)
    )
    assert created, "the guide doesn't create the Secret the profile mounts"
    name, namespace, key = created.groups()
    assert (name, namespace) == (volume.secret.secret_name, _profile_recipe().config.namespace)
    assert f'Path("{mount.mount_path}/{key}")' in _section(STEP_IMAGE)
    assert f"here `{mount.mount_path}/{key}`" in _section(CREDENTIAL)


def test_the_step_program_parses_and_uses_what_the_plugin_has() -> None:
    (program,) = re.findall(r"```python\n(.*?)```", _section(STEP_IMAGE), re.DOTALL)
    imports = [node for node in ast.walk(ast.parse(program)) if isinstance(node, ast.ImportFrom)]
    names = [(node.module or "", alias.name) for node in imports for alias in node.names]
    ours = [(module, name) for module, name in names if module.startswith("nemo_builder_plugin")]
    assert ours, "the step program imports nothing from the plugin"
    for module, name in ours:
        assert hasattr(importlib.import_module(module), name), f"{module} has no {name}"
    assert "step_main(make_service, read_credential=read_credential)" in program
    assert "read_credential" in inspect.signature(step_main).parameters


def test_the_guide_lists_the_stages_the_driver_knows() -> None:
    listed = re.search(r"reports one of (\w+) stages: ([^.]+)\.", _section(STEP_IMAGE))
    assert listed, "the guide doesn't list the stages"
    stages = get_args(Stage)
    assert listed.group(1) == NUMBERS[len(stages)]
    assert tuple(re.findall(r"`([a-z]+)`", listed.group(2))) == stages


def test_each_sentence_the_troubleshooting_tables_quote_is_spelt_by_the_sources() -> None:
    """An operator searches the guide for the sentence an answer or a row shows, so the two must read the same."""
    sentences = _quoted_sentences()
    assert len(sentences) == QUOTED_SENTENCES, (
        f"the troubleshooting tables quote {len(sentences)} sentences, not QUOTED_SENTENCES: update it if a row "
        "was added or removed, otherwise a row is no longer read"
    )
    shares = {sentence: _spelt_share(sentence) for sentence in sentences}
    unspelt = {sentence: f"{share:.0%}" for sentence, share in shares.items() if share < SPELT_SHARE}
    assert not unspelt, f"the guide quotes what the sources don't say: {unspelt}"


def test_the_refusal_tables_hold_each_refusal_of_the_backend_once() -> None:
    """A table the guide files under `backend_rejected` quotes what the code raises as that, and all of it: an
    operator looks a refusal up there, so one the tables lack is one they can't find."""
    tables = re.findall(r"^([^|\n][^\n]*)\n\n((?:\|[^\n]*\n)+)", _subsection(SUBMIT_REFUSED), re.MULTILINE)
    rows: list[str] = []
    for said_of, table in tables:
        if f"`code` `{CODE_BACKEND_REJECTED}`" not in said_of:
            continue
        # Past the table's heading and the rule under it.
        for line in table.splitlines()[2:]:
            sentence = re.match(r"\| `([^`]+)` \| ", line)
            assert sentence, f"a row that quotes no sentence: {line}"
            rows.append(sentence.group(1))
    refusals = _refusals()
    assert rows and refusals
    for row in rows:
        raised = [refusal.pattern for refusal in refusals if refusal.fullmatch(row)]
        assert len(raised) == 1, f"{row!r} is what {len(raised)} refusals say: {raised}"
    for refusal in refusals:
        quoted = [row for row in rows if refusal.fullmatch(row)]
        assert len(quoted) == 1, f"{len(quoted)} rows quote the refusal {refusal.pattern!r}: {quoted}"


def test_each_status_of_a_completion_is_filed_under_the_reason_the_step_gives() -> None:
    """The step tries some answers again, and gives a built image up with another reason than it has for a
    refusal. An operator goes from the reason to the status, so the guide files each status as the step does."""
    put_off, refused = "it was built, but the builder could not be told", "it was built, but the builder refused"
    row = re.search(rf"^\| `{put_off}: <error>` \| (.+) \|$", _subsection(IMAGE_FAILS), re.MULTILINE)
    assert row, "the guide has no row for a completion the step gave up"
    answered = set(re.findall(r"^\| `(\d{3})` \|", _subsection(COMPLETION_REFUSED), re.MULTILINE))
    assert answered, "the guide lists no status a completion is answered with"
    for status in sorted(answered | {str(UNKNOWN_SHAPE)}):
        reason = _completion_reason(int(status))
        assert reason.startswith((put_off, refused)), reason
        tried_again = reason.startswith(put_off)
        assert tried_again == (f"`{status}`" in row.group(1)), f"{status} is filed under another reason than {reason!r}"


def test_what_the_troubleshooting_prose_quotes_is_said_by_the_sources() -> None:
    """A log line to search for, or the end of a reason: quoted in a sentence, where the tables' test doesn't
    read."""
    section = _section(TROUBLESHOOTING)
    for quote in PROSE_QUOTES:
        assert f"`{quote}`" in section, f"the guide no longer quotes {quote!r}"
        assert _said_in_one_string(quote, _strings(*SOURCES)), f"no string of the sources says {quote!r}"


def test_the_launchers_lines_the_guide_quotes_are_the_launchers() -> None:
    """A step that Jobs' launcher ends before the step program runs logs no line of the driver's, so the guide
    names the launcher's, for an operator to find at the end of the container's log."""
    for line in LAUNCHER_LINES:
        assert f"`{line}`" in _subsection(IMAGE_FAILS), f"the guide no longer quotes {line!r}"
        assert _launcher_says(line), f"the launcher's formats don't make {line!r}"


def test_a_row_the_guide_points_at_is_a_row_of_its_tables() -> None:
    """The guide sends a reader on to a row by the sentence the row quotes, which finds the row only while the
    two read the same. Not a row it names by a status, such as a `ready` one: that is an image's row."""
    pointed = {quote for quote in re.findall(r"the `([^`]+)` row\b", _text()) if " " in quote}
    unknown = pointed - set(_quoted_sentences())
    assert pointed and not unknown, f"the troubleshooting tables have no row for: {unknown}"


def test_an_invalid_setting_is_answered_as_the_guide_reads_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The answer is put together from the setting's name and a few words for the problem, so no string of the
    sources holds what the guide quotes of it."""

    def get(cls: type[BuilderConfig]) -> BuilderConfig:
        settings = {"registry": "registry.example.com/team", "repositry": "team/images"}
        return BuilderConfig.model_validate({"remote": settings})

    monkeypatch.setattr(BuilderConfig, "get", classmethod(get))
    with pytest.raises(BackendRejectedError) as refused:
        service._load_config()
    for problem in ("remote.registry: invalid value", "Extra inputs are not permitted"):
        assert f"`{problem}`" in _section(TROUBLESHOOTING), f"the guide no longer quotes {problem!r}"
        assert problem in str(refused.value)


def test_the_evals_errors_the_guide_names_are_spelt_by_evals() -> None:
    quotes = re.findall(r"^- `([^`]+)`:", _subsection(EVALS_ERRORS), re.MULTILINE)
    assert len(quotes) == 3, "the evals errors are not read as they were"
    for quote in quotes:
        # The guide cuts a long error short, and names the two backends where the code fills them in.
        said = re.sub(rf"'({'|'.join(BACKENDS)})'", "<backend>", quote.removesuffix("…"))
        assert _said_in_one_string(said, _strings(*EVALS_SOURCES)), f"evals doesn't say {quote!r}"
    # Wherever else the guide quotes one of them, it quotes it the same.
    assert set(re.findall(r"`(Builder (?:refused|builds)[^`]*)`", _text())) <= set(quotes)


def test_links_name_existing_headings() -> None:
    links = re.findall(r"\]\(([^()#\s]*)#([a-z0-9-]+)\)", _text())
    pages = {page for page, _ in links}
    assert pages == LINKED.keys(), f"not known here, or no longer linked: {pages ^ LINKED.keys()}"
    for page, anchor in links:
        assert anchor in _headings(LINKED[page]), f"{LINKED[page].name} has no heading for #{anchor}"


def test_the_readme_points_at_the_guide() -> None:
    pointers = re.findall(r"\]\((\S*/builder-remote-backend\.mdx)\)", _text(README))
    assert pointers, "the README doesn't link the guide"
    for pointer in pointers:
        assert (README.parent / pointer).resolve() == GUIDE
