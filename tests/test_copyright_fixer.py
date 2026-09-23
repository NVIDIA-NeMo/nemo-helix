# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

from pytest import MonkeyPatch


def _load_copyright_fixer() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "tools" / "lint" / "copyright_fixer.py"
    spec = importlib.util.spec_from_file_location("copyright_fixer", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


copyright_fixer = _load_copyright_fixer()


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _git_add(repo: Path, paths: list[str]) -> None:
    subprocess.run(["git", "add", *paths], cwd=repo, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_supported_file_includes_missing_osrb_file_types(tmp_path: Path) -> None:
    files = {
        "e2e/conftest.py": 'print("ok")\n',
        "k8s/helm/helm-docs-template/nemo-helm-readme.md.gotmpl": "# title\n",
        "sdk/python/nemo-helix/Brewfile": 'brew "uv"\n',
        "sdk/python/nemo-helix/bin/publish-pypi": "#!/usr/bin/env bash\n",
        "services/core/entities/alembic/README": "Generic single-database configuration.\n",
        "services/core/entities/alembic/script.py.mako": '"""${message}"""\n',
        "services/core/entities/src/nhx/core/entities/api/v2/entities/entities.http": "// Example\n",
    }

    for relpath, content in files.items():
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        assert copyright_fixer._is_supported_file(str(path))


def test_supported_file_does_not_treat_shebang_as_universal_override(tmp_path: Path) -> None:
    patch_file = tmp_path / "tarfile.py.fix"
    patch_file.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    assert not copyright_fixer._is_supported_file(str(patch_file))


def test_add_header_uses_syntax_safe_styles_for_missing_osrb_file_types(tmp_path: Path) -> None:
    chart_template = tmp_path / "chart" / "templates" / "serviceaccount.yaml"
    chart_template.parent.mkdir(parents=True)
    (tmp_path / "chart" / "Chart.yaml").write_text("apiVersion: v2\nname: test\nversion: 0.1.0\n", encoding="utf-8")

    files = {
        "test_case.py": ('print("ok")\n', copyright_fixer._HASH_HEADER + "\n"),
        "nemo-helm-readme.md.gotmpl": ("# title\n", copyright_fixer._HTML_HEADER + "\n"),
        "values.yaml": ("apiVersion: v1\n", copyright_fixer._HASH_HEADER + "\n"),
        "chart/templates/serviceaccount.yaml": (
            "{{- if .Values.enabled -}}\napiVersion: v1\n{{- end }}\n",
            copyright_fixer._HELM_TEMPLATE_HEADER + "\n",
        ),
        "Brewfile": ('brew "uv"\n', copyright_fixer._HASH_HEADER + "\n"),
        "publish-pypi": (
            "#!/usr/bin/env bash\nset -e\n",
            "#!/usr/bin/env bash\n" + copyright_fixer._HASH_HEADER + "\n",
        ),
        "README": ("Generic single-database configuration.\n", copyright_fixer._HTML_HEADER + "\n"),
        "script.py.mako": ('"""${message}"""\n', copyright_fixer._HASH_HEADER + "\n"),
        "entities.http": ("// Example\n", copyright_fixer._SLASH_HEADER + "\n"),
    }

    for filename, (content, expected_prefix) in files.items():
        path = tmp_path / filename
        path.write_text(content, encoding="utf-8")

        assert copyright_fixer._add_header(str(path))
        assert path.read_text(encoding="utf-8").startswith(expected_prefix)


def test_fix_style_converts_helm_template_yaml_to_non_rendering_comment(tmp_path: Path) -> None:
    chart_template = tmp_path / "chart" / "templates" / "serviceaccount.yaml"
    chart_template.parent.mkdir(parents=True)
    (tmp_path / "chart" / "Chart.yaml").write_text("apiVersion: v2\nname: test\nversion: 0.1.0\n", encoding="utf-8")
    chart_template.write_text(
        copyright_fixer._HASH_HEADER
        + "\n{{- if .Values.enabled -}}\napiVersion: v1\nkind: ServiceAccount\n{{- end }}\n",
        encoding="utf-8",
    )

    assert copyright_fixer._needs_style_fix(str(chart_template))
    assert copyright_fixer._fix_header_style(str(chart_template))
    assert chart_template.read_text(encoding="utf-8").startswith(copyright_fixer._HELM_TEMPLATE_HEADER + "\n")


def test_fix_style_preserves_markdown_gotmpl_rendered_header(tmp_path: Path) -> None:
    template = tmp_path / "nemo-helm-readme.md.gotmpl"
    template.write_text(copyright_fixer._HTML_HEADER + '\n# {{ template "chart.description" . }}\n', encoding="utf-8")

    assert not copyright_fixer._needs_style_fix(str(template))
    assert not copyright_fixer._fix_header_style(str(template))
    assert template.read_text(encoding="utf-8").startswith(copyright_fixer._HTML_HEADER + "\n")


def test_plain_yaml_keeps_hash_comment_header(tmp_path: Path) -> None:
    values = tmp_path / "chart" / "values.yaml"
    values.parent.mkdir()
    (tmp_path / "chart" / "Chart.yaml").write_text("apiVersion: v2\nname: test\nversion: 0.1.0\n", encoding="utf-8")
    values.write_text("enabled: true\n", encoding="utf-8")

    assert copyright_fixer._add_header(str(values))
    assert values.read_text(encoding="utf-8").startswith(copyright_fixer._HASH_HEADER + "\n")


def test_helm_template_header_keeps_spdx_identifier_on_own_line() -> None:
    assert "\nSPDX-License-Identifier: Apache-2.0\n*/}}\n" in copyright_fixer._HELM_TEMPLATE_HEADER
    assert "SPDX-License-Identifier: Apache-2.0 */}}" not in copyright_fixer._HELM_TEMPLATE_HEADER


def test_inline_helm_template_header_is_not_accepted() -> None:
    inline_header = (
        "{{/* SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved. */}}\n"
        "{{/* SPDX-License-Identifier: Apache-2.0 */}}\n"
    )

    assert not copyright_fixer._has_correct_spdx_header(inline_header)


def test_include_can_target_files_under_copyrightignore_excluded_directories(tmp_path: Path) -> None:
    _git_init(tmp_path)
    ignored_file = tmp_path / "generated" / "conftest.py"
    ignored_file.parent.mkdir()
    ignored_file.write_text('print("ok")\n', encoding="utf-8")
    (tmp_path / ".copyrightignore").write_text("generated/\n", encoding="utf-8")
    _git_add(tmp_path, [".copyrightignore", "generated/conftest.py"])

    default_files = copyright_fixer._collect_files_from_dir(str(tmp_path))
    included_files = copyright_fixer._collect_files_from_dir(str(tmp_path), include=["generated/conftest.py"])

    assert str(ignored_file) not in default_files
    assert str(ignored_file) in included_files


def test_resolve_targets_scans_current_repo_root(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _git_init(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    root_file = tmp_path / "root.md"
    nested_file = nested / "child.md"
    root_file.write_text("# Root\n", encoding="utf-8")
    nested_file.write_text("# Child\n", encoding="utf-8")
    _git_add(tmp_path, ["root.md", "nested/child.md"])

    monkeypatch.chdir(nested)

    files, root = copyright_fixer._resolve_targets()

    assert root == str(tmp_path.resolve())
    assert str(root_file) in files
    assert str(nested_file) in files


def test_proprietary_license_detection_ignores_fixer_source_literals() -> None:
    source = Path(copyright_fixer.__file__).read_text(encoding="utf-8")[:4096]

    assert not copyright_fixer._has_proprietary_license(source)
    assert copyright_fixer._has_proprietary_license(
        "# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: LicenseRef-NvidiaProprietary\n"
    )
