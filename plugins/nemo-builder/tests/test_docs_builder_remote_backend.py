# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the operator's guide to the remote backend where it can be checked mechanically: its configuration
recipes validate against the builder's and Jobs' own models, its step program imports what the plugin has, the
manifests it copies are the chart's, and its links lead to headings that exist.

It does not check the guide's prose against the code. Offline: the chart's files are read as they are
written, not rendered, and no deployment is contacted.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import yaml
from nemo_builder_plugin.config import BuilderConfig, RemoteConfig
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.remote import STEP_NAME, RemoteBackend
from nemo_builder_plugin.run.remote import (
    step_main,
)
from nemo_builder_plugin.schema import BuildSet
from nemo_helix_plugin.jobs.execution_profiles import (
    KubernetesJobExecutionProfile,
    KubernetesJobExecutionProfileConfig,
)
from pydantic.fields import FieldInfo

ROOT = Path(__file__).resolve().parents[3]
GUIDE = ROOT / "docs/set-up/helm/builder-remote-backend.mdx"
HARBOR_GUIDE = ROOT / "docs/set-up/helm/harbor-sandbox-profiles.mdx"
README = ROOT / "plugins/nemo-builder/README.md"
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

#: The `remote` keys whose value in the recipe is an example, not the default: each is unset by default.
EXAMPLES = {"registry", "repository", "image", "credential_secret"}


def _text(path: Path = GUIDE) -> str:
    return path.read_text(encoding="utf-8")


def _section(heading: str) -> str:
    """One second-level section of the guide, from its heading to the next."""
    return heading + _text().split(heading, 1)[1].split("\n## ", 1)[0]


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


def _profile_executor() -> dict[str, Any]:
    """The profile the guide has an operator write, as its recipe writes it."""
    (recipe,) = _yaml_documents(_section(PROFILE))
    (executor,) = recipe["platformConfig"]["jobs"]["executors"]
    return executor


def _profile_recipe() -> KubernetesJobExecutionProfile:
    executor = _profile_executor()
    assert executor["config"].keys() <= KubernetesJobExecutionProfileConfig.model_fields.keys()
    return KubernetesJobExecutionProfile.model_validate(executor)


def _request_plan(config: BuilderConfig, *, revision: int | None = None) -> BuildPlan:
    """The build the guide's own request asks for, planned as a submit plans it (`submit_build_set`): at the
    revision the request names, or at ``revision``."""
    section = _section(VERIFY)
    (workspace,) = set(re.findall(r"/workspaces/([a-z0-9-]+)/builds", section))
    (body,) = re.findall(r"-d '(\{.*\})'", section)
    build_set = BuildSet.model_validate_json(body)
    if revision is not None:
        build_set = build_set.model_copy(update={"revision": revision})
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
    config = BuilderConfig.model_validate(_backend_recipe())
    plan = _request_plan(config)
    (image,) = plan.images
    workspace, _, fileset = image.source.fileset.partition("/")
    # A build that failed is built again at the next revision, under other names than the commands above hold.
    again = _request_plan(config, revision=plan.build_set.revision + 1)
    (rebuilt,) = again.images
    named = [
        f"nemo files filesets create {fileset} --workspace {workspace}",
        f'nemo files upload "$CTX/" {fileset} --workspace {workspace}',
        f"`job` is `{plan.job_name}`, and the one image, `{image.name}`, is `pending`",
        f'/workspaces/{plan.workspace}/container-images/{image.name}"',
        f"nemo jobs get-status {plan.job_name} --workspace {plan.workspace}",
        f"nemo jobs tail {plan.job_name} --workspace {plan.workspace} --step-id {STEP_NAME}",
        f'`"revision": {again.build_set.revision}`: its image is then `{rebuilt.name}` and its job `{again.job_name}`',
    ]
    missing = [text for text in named if text not in section]
    assert not missing, f"the guide's commands don't name what its request makes: {missing}"


def test_the_profile_table_names_real_keys_and_their_defaults() -> None:
    rows = dict(re.findall(r"^\| `config\.([a-z_]+)` \| ([^|]+) \|", _section(PROFILE), re.MULTILINE))
    fields = KubernetesJobExecutionProfileConfig.model_fields
    assert rows and rows.keys() <= fields.keys()
    for key in ("max_active_jobs", "ttl_seconds_active", "ttl_seconds_before_active"):
        assert re.search(rf"If unset, {fields[key].default}\b", rows[key]), key


def test_a_profile_key_jobs_does_not_know_is_dropped_and_the_guide_reads_the_profile_back() -> None:
    """Jobs takes a misspelt key without a word and leaves its setting at the default, which for the namespace
    is the release's. No check of the builder's shows that, so the guide says it and has an operator read each
    key it sets back from Jobs."""
    executor = _profile_executor()
    settings = {key if key != "namespace" else "namespce": value for key, value in executor["config"].items()}
    misspelt = KubernetesJobExecutionProfile.model_validate({**executor, "config": settings, "namespce": "beside"})
    assert misspelt.config.namespace is None
    said = "Jobs ignores a key it doesn't know, directly under `config` or beside it, with no warning"
    assert said in _section(PROFILE), "the guide no longer says what Jobs does with a key it doesn't know"
    section = _section(VERIFY)
    profile = _profile_recipe()
    assert f'select(.provider == "{profile.provider}" and .profile == "{profile.profile}")' in section
    read_back = set(re.findall(r"`config\.([a-z_]+)`", section))
    assert "namespace" in read_back and read_back <= executor["config"].keys(), read_back


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


def test_the_platform_secret_the_guide_creates_is_the_one_the_backend_recipe_names() -> None:
    """A secret under another name, or in another workspace, is one the builder never looks for, and every
    submit is then refused as missing it."""
    workspace, _, name = _backend_recipe()["remote"]["credential_secret"].partition("/")
    section = _section(CREDENTIAL)
    assert workspace and name
    assert f"nemo workspaces create {workspace}\n" in section
    assert f"nemo secrets create {name} --workspace {workspace} --from-file " in section


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
