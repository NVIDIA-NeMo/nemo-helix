# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from docs._scripts.lint_helm_snippets import (
    DEFAULT_VALUES_FILE,
    UniqueKeyLoader,
    YamlSnippet,
    check_paths,
    check_snippet,
    extract_yaml_snippets,
    main,
)


@pytest.fixture
def defaults() -> dict:
    with DEFAULT_VALUES_FILE.open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=UniqueKeyLoader)


def messages(source: str, defaults: dict) -> list[str]:
    return [error.message for error in check_snippet(YamlSnippet(Path("example.mdx"), 10, source), defaults)]


def test_extract_fences_and_preserve_line_numbers(tmp_path: Path) -> None:
    doc = tmp_path / "page.mdx"
    doc.write_text(
        '<Tabs>\n   ```YAML title="values.yaml"\n   ingress:\n     enabled: true\n   ```\n'
        "~~~yml\napi: {}\n~~~~\n```sh\n```yaml\nnot a fence\n```\n</Tabs>\n",
        encoding="utf-8",
    )
    snippets = extract_yaml_snippets(doc)
    assert [(item.start_line, item.source, item.indent) for item in snippets] == [
        (3, "ingress:\n  enabled: true", 3),
        (7, "api: {}", 0),
    ]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ingress:\nenabled: true\ndefaultHost: nhx.example.com", ["ingress: expected mapping", "enabled: unknown"]),
        (
            "multinodeNetworking:\n  aws:\n  enabled: true\n  efaDevicesPerGPU: 1",
            ["multinodeNetworking.aws: expected mapping", "multinodeNetworking.enabled: unknown"],
        ),
        (
            "core:\n  storage:\n  size: 200Gi\n  accessModes: [ReadWriteMany]",
            ["core.storage: expected mapping", "core.size: unknown"],
        ),
        (
            "platformConfig:\nfiles:\ndefault_storage_config:\ntype: s3",
            ["platformConfig: expected mapping", "files: unknown"],
        ),
        (
            "platformConfig:\n  files:\n  default_storage_config:\n    type: s3",
            ["platformConfig.files: empty YAML value"],
        ),
        ("api:\n  env:\n  AWS_ACCESS_KEY_ID: example", ["api.env: expected mapping", "api.AWS_ACCESS_KEY_ID: unknown"]),
        ("api:\n  env:\n    AWS_ACCESS_KEY_ID:\n    valueFrom: {}", ["api.env.AWS_ACCESS_KEY_ID: empty YAML value"]),
    ],
)
def test_collapsed_nesting_regressions(source: str, expected: list[str], defaults: dict) -> None:
    # The QA failure is not invalid YAML: a plain YAML parser accepts all of these.
    assert isinstance(yaml.safe_load(source), dict)
    errors = messages(source, defaults)
    for fragment in expected:
        assert any(fragment in error for error in errors), errors


@pytest.mark.parametrize(
    "source",
    [
        "ingress:\n  enabled: true\n  defaultHost: nhx.example.com",
        "multinodeNetworking:\n  aws:\n    enabled: true\n    efaDevicesPerGPU: 1",
        "core:\n  storage:\n    size: 200Gi\n    accessModes: [ReadWriteMany]",
        "platformConfig:\n  files:\n    default_storage_config:\n      type: s3\n      bucket: my-bucket\n      use_sdk_auth: true",
        "platformConfig:\n  new_plugin:\n    arbitrary: [1, {custom: null}]",
        "api:\n  env:\n    AWS_ACCESS_KEY_ID:\n      valueFrom:\n        secretKeyRef:\n          name: s3-credentials\n          key: access_key_id",
        "env:\n  SIMPLE: value\n  FLAG: true\n  COUNT: 2\n  OPTIONAL: null",
        "api:\n  podSecurityContext:\n    runAsUser: 1001\n  resources:\n    requests:\n      nvidia.com/gpu: 1",
        "ingress:\n  annotations:\n    nginx.ingress.kubernetes.io/proxy-body-size: '0'\n  tls:\n    - hosts: [nhx.example.com]\n      secretName: nhx-tls",
        "httpRoute:\n  parentRefs:\n    - name: gateway\n      namespace: gateways",
        "ingress:\n  hosts:\n    - name: nhx.example.com\n      paths:\n        - path: /\n          port: 8080",
        "api:\n  autoscaling:\n    targetMemoryUtilizationPercentage: null",
        "platformConfig: {}\napi: {}\ningress:\n  hosts: []",
    ],
)
def test_valid_partial_and_free_form_overrides(source: str, defaults: dict) -> None:
    assert messages(source, defaults) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ingress:\n  hosts: {}", "ingress.hosts: expected sequence"),
        ("ingress:\n  hosts:\n    - paths: /", "ingress.hosts[0].paths: expected sequence"),
        ("ingress:\n  hosts:\n    - hostname: example.com", "ingress.hosts[0].hostname: unknown chart key"),
        ("api:\n  env: []", "api.env: expected mapping"),
        ("api:\n  enabled: {}", "api.enabled: expected scalar"),
        ("ingress: null", "ingress: expected mapping"),
        ("ingress:\n  enabledd: true", "ingress.enabledd: unknown chart key"),
    ],
)
def test_shape_mismatches(source: str, expected: str, defaults: dict) -> None:
    assert any(expected in error for error in messages(source, defaults))


@pytest.mark.parametrize(
    "source",
    [
        "ingress: {}\ningress: {}",
        "platformConfig:\n  custom:\n    field: 1\n    field: 2",
        "api:\n  env:\n    TOKEN: first\n    TOKEN: second",
        "ingress:\n  hosts:\n    - name: one\n      name: two",
        "apiVersion: v1\nkind: ConfigMap\ndata:\n  key: a\n  key: b",
        "unrelated:\n  field: first\n  field: second",
        "env:\n  true: a\n  yes: b",
        "api:\n  <<: {enabled: true}\n  <<: {enabled: false}",
    ],
)
def test_duplicate_keys_at_any_depth(source: str, defaults: dict) -> None:
    assert any("duplicate key" in error for error in messages(source, defaults))


@pytest.mark.parametrize("source", ["ingress: [", "ingress: !unsafe value", "? [unhashable, key]\n: value"])
def test_invalid_yaml_and_unsafe_tags(source: str, defaults: dict) -> None:
    assert any(error.startswith("YAML:") for error in messages(source, defaults))


def test_diagnostics_map_to_original_doc(defaults: dict) -> None:
    snippet = YamlSnippet(Path("page.mdx"), 20, "ingress:\n  enabled: true\n  enabled: false", indent=3)
    error = check_snippet(snippet, defaults)[0]
    assert (error.path, error.line, error.column) == (Path("page.mdx"), 22, 6)
    assert error.format().startswith("page.mdx:22:6:")


def test_structural_diagnostic_location(defaults: dict) -> None:
    snippet = YamlSnippet(Path("page.mdx"), 20, "ingress:\n  typo: true", indent=3)
    error = check_snippet(snippet, defaults)[0]
    assert (error.line, error.column) == (21, 6)


def test_multi_document_yaml(defaults: dict) -> None:
    source = "ingress: {}\n---\napiVersion: v1\nkind: Secret\n---\ncore:\n  storag: {}"
    assert any("core.storag: unknown" in error for error in messages(source, defaults))


def test_manifests_and_other_chart_examples_are_syntax_only(defaults: dict) -> None:
    assert messages("apiVersion: v1\nkind: Pod\nmetadata: {}\nspec:\n  arbitrary: true", defaults) == []
    assert messages("nodeSelector:\n  gpu: 'true'\nshims:\n  qemu:\n    enabled: true", defaults) == []


def test_yaml_merges_allow_overrides_but_check_shape(defaults: dict) -> None:
    assert messages("api: &api\n  enabled: true\ncore:\n  <<: *api\n  enabled: false", defaults) == []
    assert any(
        "core.bogus: unknown" in error
        for error in messages("platformConfig: &config\n  bogus: true\ncore:\n  <<: *config", defaults)
    )


def test_recursive_alias_does_not_loop(defaults: dict) -> None:
    assert messages("platformConfig: &config\n  self: *config", defaults) == []


def test_unclosed_and_empty_fences(tmp_path: Path, defaults: dict) -> None:
    doc = tmp_path / "page.mdx"
    doc.write_text("```yaml\n```\n```yaml\ningress: {}\n", encoding="utf-8")
    snippets = extract_yaml_snippets(doc)
    assert check_snippet(snippets[0], defaults) == []
    assert any("Unclosed YAML fence" in error.message for error in check_snippet(snippets[1], defaults))


def test_check_paths_uses_supplied_reference(tmp_path: Path) -> None:
    values = tmp_path / "values.yaml"
    values.write_text("custom:\n  nested: {}\n", encoding="utf-8")
    doc = tmp_path / "page.mdx"
    doc.write_text("```yaml\ncustom:\n  nested: []\n```\n", encoding="utf-8")
    results = check_paths([doc], values)
    assert len(results) == 1
    assert not results[0].passed
    assert results[0].snippets == 1
    assert "custom.nested: expected mapping" in results[0].diagnostics[0].message


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    doc = tmp_path / "page.mdx"
    doc.write_text("```yaml\ningress: {}\n```\n", encoding="utf-8")
    assert main([str(doc)]) == 0
    assert "Checked 1 YAML snippet(s); 0 error(s)." in capsys.readouterr().out
    doc.write_text("```yaml\ningress: []\n```\n", encoding="utf-8")
    assert main([str(doc)]) == 1
    assert "expected mapping" in capsys.readouterr().out
    assert main([str(tmp_path / "missing.mdx")]) == 2
    assert "ERROR:" in capsys.readouterr().err
    values = tmp_path / "values.yaml"
    values.write_text("[]", encoding="utf-8")
    assert main([str(doc), "--values-file", str(values)]) == 2
    assert "Expected a nonempty chart values mapping" in capsys.readouterr().err
