# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for license utility functions."""

import csv
import io
import json
from pathlib import Path
from typing import Any, cast

import pytest
from nemo_helix_sdk_tools.license.license_utils import (
    get_override_key_for_package,
    normalize_package_name,
    normalize_version_for_override,
    resolve_license,
)


class TestNormalizePackageName:
    """Tests for normalize_package_name function."""

    def test_lowercase_conversion(self):
        """Test that package names are converted to lowercase."""
        assert normalize_package_name("MyPackage") == "mypackage"
        assert normalize_package_name("UPPERCASE") == "uppercase"

    def test_underscore_to_dash(self):
        """Test that underscores are converted to dashes."""
        assert normalize_package_name("my_package") == "my-package"
        assert normalize_package_name("test_pkg_name") == "test-pkg-name"

    def test_combined_normalization(self):
        """Test combined lowercase and underscore conversion."""
        assert normalize_package_name("My_Package") == "my-package"
        assert normalize_package_name("TEST_PKG") == "test-pkg"


class TestResolveLicense:
    """Tests for resolve_license function."""

    def test_string_license(self):
        """Test resolving a simple string license."""
        assert resolve_license("MIT") == "MIT"
        assert resolve_license("Apache-2.0") == "Apache-2.0"

    def test_empty_input(self):
        """Test handling of empty input."""
        assert resolve_license("") == ""
        assert resolve_license([]) == ""

    def test_string_with_or_separator(self):
        """Test string with OR separator returns first license."""
        assert resolve_license("Apache-2.0 OR MIT") == "Apache-2.0"
        assert resolve_license("MIT OR BSD-3-Clause") == "MIT"

    def test_string_with_and_separator(self):
        """Test string with AND separator returns first license."""
        assert resolve_license("GPL-2.0 AND MIT") == "GPL-2.0"

    def test_list_without_allowed_licenses(self):
        """Test list returns first license when no allowed licenses provided."""
        assert resolve_license(["Apache-2.0", "MIT"]) == "Apache-2.0"
        assert resolve_license(["non-standard", "MIT"]) == "non-standard"

    def test_list_with_allowed_licenses_prefers_valid(self):
        """Test that allowed licenses are preferred over non-standard ones."""
        allowed = {"MIT", "APACHE-2.0", "BSD-3-CLAUSE"}

        # Should prefer MIT over non-standard
        assert resolve_license(["non-standard", "MIT"], allowed) == "MIT"

        # Should prefer Apache-2.0 (case insensitive)
        assert resolve_license(["non-standard", "Apache-2.0"], allowed) == "Apache-2.0"

        # Should prefer BSD-3-Clause
        assert resolve_license(["unknown", "BSD-3-Clause", "MIT"], allowed) == "BSD-3-Clause"

    def test_list_with_allowed_licenses_case_insensitive(self):
        """Test that license matching is case-insensitive."""
        allowed = {"MIT", "APACHE-2.0"}

        assert resolve_license(["non-standard", "mit"], allowed) == "mit"
        assert resolve_license(["unknown", "Apache-2.0"], allowed) == "Apache-2.0"
        assert resolve_license(["bad", "apache-2.0"], allowed) == "apache-2.0"

    def test_list_no_valid_licenses_returns_first(self):
        """Test that first license is returned when none are in allowed list."""
        allowed = {"MIT", "APACHE-2.0"}

        assert resolve_license(["GPL-3.0", "LGPL-2.1"], allowed) == "GPL-3.0"
        assert resolve_license(["non-standard", "unknown"], allowed) == "non-standard"

    def test_list_with_separator_in_license(self):
        """Test list with licenses containing separators."""
        allowed = {"MIT", "APACHE-2.0"}

        # Should prefer MIT OR BSD-3-Clause because MIT is in allowed (takes first part)
        result = resolve_license(["non-standard", "MIT OR BSD-3-Clause"], allowed)
        assert result == "MIT"

        # Should prefer Apache because it's in allowed (takes first part)
        result = resolve_license(["unknown", "Apache-2.0 OR MIT"], allowed)
        assert result == "Apache-2.0"

    def test_real_world_ormsgpack_case(self):
        """Test the real-world ormsgpack case that motivated this change."""
        allowed = {
            "MIT",
            "BSD-3-CLAUSE",
            "BSD-2-CLAUSE",
            "APACHE-2.0",
            "ISC",
            "ZLIB",
            "NVIDIA PROPRIETARY SOFTWARE",
        }

        # ormsgpack has ["non-standard", "MIT"] from osv-scanner
        # Should prefer MIT since it's in allowed licenses
        result = resolve_license(["non-standard", "MIT"], allowed)
        assert result == "MIT"

    def test_invalid_type_raises_error(self):
        """Test that invalid input types raise appropriate errors."""
        with pytest.raises(TypeError, match="licenses must be a list or str"):
            resolve_license(cast(Any, 123))

        with pytest.raises(TypeError, match="licenses must be a list or str"):
            resolve_license(cast(Any, None))

    def test_single_license_list(self):
        """Test list with single license."""
        assert resolve_license(["MIT"]) == "MIT"
        assert resolve_license(["Apache-2.0"], {"APACHE-2.0"}) == "Apache-2.0"


class TestNormalizeVersionForOverride:
    """Tests for normalize_version_for_override (PEP 440 local version strip)."""

    def test_strips_plus_suffix(self):
        """Version with +cu129 is normalized to base version."""
        assert normalize_version_for_override("0.14.1+cu129") == "0.14.1"
        assert normalize_version_for_override("2.9.0+cu129") == "2.9.0"

    def test_unchanged_without_plus(self):
        """Version without + is unchanged."""
        assert normalize_version_for_override("0.14.1") == "0.14.1"
        assert normalize_version_for_override("13.590.44") == "13.590.44"

    def test_strips_only_first_plus(self):
        """Only the first + and suffix are stripped."""
        assert normalize_version_for_override("0.14.1+cu129+local") == "0.14.1"


class TestGetOverrideKeyForPackage:
    """Tests for get_override_key_for_package (override lookup key)."""

    def test_base_name_unchanged(self):
        """Base name is used as key when no variant suffix."""
        assert get_override_key_for_package("torchao", "0.14.1") == "torchao"
        assert get_override_key_for_package("nvidia-ml-py", "13.590.44") == "nvidia-ml-py"

    def test_cu129_version_still_uses_name(self):
        """Package with +cu129 version still matches override by name."""
        assert get_override_key_for_package("torchao", "0.14.1+cu129") == "torchao"

    def test_name_with_cu129_suffix_stripped(self):
        """Name with -cu129 or _cu129 suffix is normalized for lookup."""
        assert get_override_key_for_package("torchao-cu129", "0.14.1") == "torchao"
        assert get_override_key_for_package("torchao_cu129", "0.14.1+cu129") == "torchao"

    def test_empty_name(self):
        """Empty name returns empty key."""
        assert get_override_key_for_package("", "0.14.1") == ""


class TestOverrideAppliedForCu129Version:
    """Verify override is applied for +cu129-style version (torchao case)."""

    def test_format_licenses_table_keeps_same_name_across_ecosystems(self):
        """PyPI and Go packages with the same normalized name should remain separate rows."""
        from nemo_helix_sdk_tools.license.format_osv_licenses import format_licenses_table

        json_data = {
            "results": [
                {
                    "packages": [
                        {
                            "package": {"name": "shared-name", "version": "1.0.0", "ecosystem": "PyPI"},
                            "licenses": ["MIT"],
                        },
                        {
                            "package": {"name": "shared-name", "version": "1.0.0", "ecosystem": "Go"},
                            "licenses": ["Apache-2.0"],
                        },
                    ]
                }
            ]
        }

        result = format_licenses_table(json_data, overrides={}, local_packages=set())

        assert result.count("shared-name") == 2
        assert "MIT" in result
        assert "APACHE-2.0" in result

    def test_format_licenses_table_applies_override_for_cu129_version(self):
        """Package with version 0.14.1+cu129 and empty licenses gets override license."""
        from nemo_helix_sdk_tools.license.format_osv_licenses import format_licenses_table

        # OSV-style package with +cu129 version and no license from scanner
        json_data = {
            "results": [
                {
                    "packages": [
                        {
                            "package": {"name": "torchao", "version": "0.14.1+cu129"},
                            "licenses": [],
                        }
                    ]
                }
            ]
        }
        overrides = {"torchao": "BSD-3-Clause"}
        result = format_licenses_table(json_data, overrides=overrides, local_packages=set())
        assert "BSD-3-CLAUSE" in result
        assert "torchao" in result
        # Override was applied, so we should not see UNKNOWN for this package
        assert "✘" not in result or "BSD-3-CLAUSE" in result


class TestFormatLicenses:
    """Tests for license report formatting."""

    def test_generate_lockfile_sanitizes_header_without_dropping_export_flags(self, tmp_path, monkeypatch):
        """The generated requirements header keeps export selectors without absolute paths."""
        from nemo_helix_sdk_tools.license import generator

        output_lockfile = tmp_path / "third_party" / "requirements-main.txt"

        def fake_run(cmd: list[str], cwd: Path, text: bool, check: bool) -> None:
            assert cwd == tmp_path
            assert text is True
            assert check is True
            output_lockfile.parent.mkdir()
            output_lockfile.write_text(
                "# This file was autogenerated by uv via the following command:\n"
                f"#    uv export --no-dev --output-file {output_lockfile}\n"
                "aiofiles==25.1.0\n",
                encoding="utf-8",
            )

        monkeypatch.setattr(generator.subprocess, "run", fake_run)

        generator.generate_lockfile_without_dev_dependencies(
            tmp_path,
            output_lockfile,
            extras=["training"],
            packages=["nhx-unsloth"],
            groups=["nhx-api", "nhx-task-runtime"],
        )

        requirements = output_lockfile.read_text(encoding="utf-8")
        assert requirements.splitlines()[1] == (
            "#    uv export --frozen --no-dev --package nhx-unsloth --extra training --group nhx-api "
            "--group nhx-task-runtime --quiet --output-file third_party/requirements-main.txt"
        )
        assert str(tmp_path) not in requirements

    def test_reviewed_license_overrides_cover_corrected_inventory_values(self):
        """Reviewed overrides keep generated license inventory from losing corrected SPDX expressions."""
        import yaml

        repo_root = Path(__file__).parents[4]
        overrides_file = Path(__file__).parents[2] / "src" / "nemo_helix_sdk_tools" / "license" / "overrides.yaml"
        inventory_file = repo_root / "third_party" / "licenses.jsonl"
        overrides = yaml.safe_load(overrides_file.read_text(encoding="utf-8"))["overrides"]
        normalized_overrides = {normalize_package_name(name): license for name, license in overrides.items()}
        inventory_licenses = {
            normalize_package_name(row["name"]): row["license"]
            for row in (json.loads(line) for line in inventory_file.read_text(encoding="utf-8").splitlines())
        }
        # Only packages the workspace still resolves belong here: the inventory
        # lists what we ship, so a name that leaves the dependency set (such as
        # langchain-oci, dropped with the move to NAT 1.9) has no row to check.
        expected_licenses = {
            "certifi": "MPL-2.0",
            "greenlet": "MIT AND PSF-2.0",
            "pathspec": "MPL-2.0",
            "psycopg2-binary": "LGPL-3.0-or-later WITH openvpn-openssl-exception",
            "regex": "Apache-2.0 AND CNRI-Python",
            "tqdm": "MIT AND MPL-2.0",
        }
        normalized_expected = {
            normalize_package_name(name): license_expression for name, license_expression in expected_licenses.items()
        }

        assert {name: normalized_overrides[name] for name in normalized_expected} == normalized_expected
        assert {name: inventory_licenses[name] for name in normalized_expected} == {
            name: license_expression.upper() for name, license_expression in normalized_expected.items()
        }

    def test_format_licenses_fills_missing_osv_package_from_overrides(self, tmp_path):
        """A reviewed override fills an exported requirement omitted by OSV."""
        from nemo_helix_sdk_tools.license.generator import format_licenses

        license_dir = tmp_path / "third_party"
        license_dir.mkdir()
        osv_json = license_dir / "osv-licenses.json"
        osv_json.write_text(json.dumps({"results": [{"packages": []}]}), encoding="utf-8")
        (license_dir / "requirements-main.txt").write_text(
            "cloudpickle==3.1.2 ; python_version >= '3.11' \\\n"
            "    --hash=sha256:7fda9eb655c9c230dab534f1983763de5835249750e85fbcef43aaa30a9a2414\n",
            encoding="utf-8",
        )
        overrides_file = tmp_path / "overrides.yaml"
        overrides_file.write_text("overrides:\n  cloudpickle: BSD-3-Clause\n", encoding="utf-8")
        output_file = license_dir / "licenses.jsonl"

        format_licenses(osv_json, output_file, overrides_file, format_type="jsonl")

        assert output_file.read_text(encoding="utf-8") == (
            '{"name": "cloudpickle", "license": "BSD-3-CLAUSE", "compatible": true}'
        )

    def test_format_licenses_deduplicates_osv_and_override_fallback_rows(self, tmp_path):
        """Reviewed override fallback rows should not duplicate packages already emitted by OSV."""
        from nemo_helix_sdk_tools.license.generator import format_licenses

        license_dir = tmp_path / "third_party"
        license_dir.mkdir()
        osv_json = license_dir / "osv-licenses.json"
        osv_json.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "packages": [
                                {
                                    "package": {
                                        "name": "aiohappyeyeballs",
                                        "version": "2.6.1",
                                        "ecosystem": "PyPI",
                                    },
                                    "licenses": ["PSF-2.0"],
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (license_dir / "requirements-main.txt").write_text(
            "aiohappyeyeballs==2.6.1 ; python_version >= '3.12' \\\n"
            "    --hash=sha256:0000000000000000000000000000000000000000000000000000000000000000\n",
            encoding="utf-8",
        )
        overrides_file = tmp_path / "overrides.yaml"
        overrides_file.write_text("overrides:\n  aiohappyeyeballs: PSF-2.0\n", encoding="utf-8")
        output_file = license_dir / "licenses.jsonl"

        format_licenses(osv_json, output_file, overrides_file, format_type="jsonl")

        assert output_file.read_text(encoding="utf-8") == (
            '{"name": "aiohappyeyeballs", "license": "PSF-2.0", "compatible": true}'
        )

    def test_format_licenses_ignores_osv_packages_missing_from_exported_requirements(self, tmp_path):
        """OSV-only conditional packages should not affect generated license reports."""
        from nemo_helix_sdk_tools.license.generator import format_licenses

        license_dir = tmp_path / "third_party"
        license_dir.mkdir()
        osv_json = license_dir / "osv-licenses.json"
        osv_json.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "packages": [
                                {
                                    "package": {"name": "aiofiles", "version": "25.1.0"},
                                    "licenses": ["Apache-2.0"],
                                },
                                {
                                    "package": {"name": "backports-tarfile", "version": "1.2.0"},
                                    "licenses": ["MIT"],
                                },
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (license_dir / "requirements-main.txt").write_text(
            "aiofiles==25.1.0 ; python_version >= '3.12' \\\n"
            "    --hash=sha256:0000000000000000000000000000000000000000000000000000000000000000\n",
            encoding="utf-8",
        )
        output_file = license_dir / "licenses.jsonl"

        format_licenses(osv_json, output_file, format_type="jsonl")

        assert output_file.read_text(encoding="utf-8") == (
            '{"name": "aiofiles", "license": "APACHE-2.0", "compatible": true}'
        )

    def test_format_licenses_keeps_go_packages_missing_from_python_requirements(self, tmp_path):
        """Only PyPI packages are filtered through the uv-exported requirements file."""
        from nemo_helix_sdk_tools.license.generator import format_licenses

        license_dir = tmp_path / "third_party"
        license_dir.mkdir()
        osv_json = license_dir / "osv-licenses.json"
        osv_json.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "packages": [
                                {
                                    "package": {
                                        "name": "backports-tarfile",
                                        "version": "1.2.0",
                                        "ecosystem": "PyPI",
                                    },
                                    "licenses": ["MIT"],
                                }
                            ]
                        },
                        {
                            "packages": [
                                {
                                    "package": {
                                        "name": "github.com/spf13/cobra",
                                        "version": "1.10.1",
                                        "ecosystem": "Go",
                                    },
                                    "licenses": ["Apache-2.0"],
                                }
                            ]
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        (license_dir / "requirements-main.txt").write_text(
            "aiofiles==25.1.0 ; python_version >= '3.12' \\\n"
            "    --hash=sha256:0000000000000000000000000000000000000000000000000000000000000000\n",
            encoding="utf-8",
        )
        output_file = license_dir / "licenses.jsonl"

        format_licenses(osv_json, output_file, format_type="jsonl")

        assert output_file.read_text(encoding="utf-8") == (
            '{"name": "github.com/spf13/cobra", "license": "APACHE-2.0", "compatible": true}'
        )

    def test_format_licenses_csv_uses_third_party_license_columns(self, tmp_path, monkeypatch):
        """CSV output is Package, License, License URL and supports custom output directories."""
        from nemo_helix_sdk_tools.license import generator

        license_dir = tmp_path / "third_party"
        license_dir.mkdir()
        osv_json = license_dir / "osv-licenses.json"
        osv_json.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "packages": [
                                {
                                    "package": {"name": "aiofiles", "version": "25.1.0"},
                                    "licenses": ["Apache-2.0"],
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_file = tmp_path / "reports" / "licenses.csv"

        monkeypatch.setattr(
            generator,
            "resolve_license_url",
            lambda name, version, license_str: "https://github.com/Tinche/aiofiles/blob/main/LICENSE",
        )

        generator.format_licenses(osv_json, output_file, format_type="csv")

        rows = list(csv.DictReader(io.StringIO(output_file.read_text(encoding="utf-8"))))
        assert rows == [
            {
                "Package": "aiofiles",
                "License": "APACHE-2.0",
                "License URL": "https://github.com/Tinche/aiofiles/blob/main/LICENSE",
            }
        ]

    def test_format_licenses_csv_escapes_formula_license_urls(self, tmp_path, monkeypatch):
        """CSV license URLs are escaped before spreadsheet import can evaluate them."""
        from nemo_helix_sdk_tools.license import generator

        license_dir = tmp_path / "third_party"
        license_dir.mkdir()
        osv_json = license_dir / "osv-licenses.json"
        osv_json.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "packages": [
                                {
                                    "package": {"name": "malicious", "version": "1.0.0"},
                                    "licenses": ["Apache-2.0"],
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_file = tmp_path / "reports" / "licenses.csv"

        monkeypatch.setattr(
            generator,
            "resolve_license_url",
            lambda name, version, license_str: '=HYPERLINK("https://example.com")',
        )

        generator.format_licenses(osv_json, output_file, format_type="csv")

        rows = list(csv.DictReader(io.StringIO(output_file.read_text(encoding="utf-8"))))
        assert rows[0]["License URL"] == '\'=HYPERLINK("https://example.com")'

    def test_get_projects_allows_report_output_file_override(self, tmp_path):
        """The formatted report path can be overridden independently of scan artifacts."""
        from nemo_helix_sdk_tools.license.generator import get_projects

        workspace_root = tmp_path / "repo"
        output_file = tmp_path / "reports" / "licenses.csv"

        projects = get_projects(workspace_root, output_file=output_file)

        assert projects[0]["output_file"] == output_file
        assert projects[0]["osv_json"] == workspace_root / "third_party" / "osv-licenses.json"
        assert projects[0]["output_lockfile"] == workspace_root / "third_party" / "requirements-main.txt"
        assert projects[0]["additional_lockfiles"] == [
            workspace_root / "services/core/jobs/jobs-launcher/go.mod",
            workspace_root / "services/guardrails/callouts/go.mod",
        ]

    def test_generate_project_licenses_keeps_additional_lockfiles_positional_slot(self):
        """additional_lockfiles remains before groups for existing positional callers."""
        import inspect

        from nemo_helix_sdk_tools.license import generator

        parameters = list(inspect.signature(generator.generate_project_licenses).parameters)

        assert parameters.index("additional_lockfiles") < parameters.index("groups")

    def test_generate_project_licenses_merges_additional_lockfile_scans(self, tmp_path, monkeypatch):
        """Python and Go OSV scan results are merged before report formatting."""
        from nemo_helix_sdk_tools.license import generator

        license_dir = tmp_path / "third_party"
        license_dir.mkdir()
        output_lockfile = license_dir / "requirements-main.txt"
        osv_json = license_dir / "osv-licenses.json"
        output_file = license_dir / "licenses.jsonl"
        go_mod = tmp_path / "services" / "core" / "jobs" / "jobs-launcher" / "go.mod"
        go_mod.parent.mkdir(parents=True)
        go_mod.write_text("module example.test/jobs-launcher\n", encoding="utf-8")

        monkeypatch.setattr(
            generator,
            "generate_lockfile_without_dev_dependencies",
            lambda lockfile_dir, output_lockfile, extras=None, packages=None, groups=None: output_lockfile,
        )

        scanned_lockfiles: list[Path] = []

        def fake_run_osv_scanner(lockfile: Path, output_file: Path, cwd: Path | None = None) -> None:
            scanned_lockfiles.append(lockfile)
            if lockfile == output_lockfile:
                package = {
                    "package": {"name": "aiofiles", "version": "25.1.0", "ecosystem": "PyPI"},
                    "licenses": ["Apache-2.0"],
                }
                license_summary = [{"name": "Apache-2.0", "count": 1}]
            else:
                package = {
                    "package": {"name": "github.com/spf13/cobra", "version": "1.10.1", "ecosystem": "Go"},
                    "licenses": ["MIT"],
                }
                license_summary = [{"name": "MIT", "count": 1}]
            output_file.write_text(
                json.dumps({"results": [{"packages": [package]}], "license_summary": license_summary}),
                encoding="utf-8",
            )

        formatted_packages: list[str] = []

        def fake_format_licenses(
            osv_json_path: Path,
            output_file: Path,
            overrides_file: Path | None = None,
            format_type: str = "table",
        ) -> None:
            data = json.loads(osv_json_path.read_text(encoding="utf-8"))
            formatted_packages.extend(
                pkg_entry["package"]["name"]
                for result in data.get("results", [])
                for pkg_entry in result.get("packages", [])
            )

        monkeypatch.setattr(generator, "run_osv_scanner", fake_run_osv_scanner)
        monkeypatch.setattr(generator, "format_licenses", fake_format_licenses)

        generator.generate_project_licenses(
            project_name="main",
            lockfile_dir=tmp_path,
            output_lockfile=output_lockfile,
            osv_json=osv_json,
            output_file=output_file,
            additional_lockfiles=[go_mod],
        )

        assert scanned_lockfiles == [output_lockfile, go_mod]
        assert formatted_packages == ["aiofiles", "github.com/spf13/cobra"]
        assert json.loads(osv_json.read_text(encoding="utf-8"))["license_summary"] == [
            {"name": "Apache-2.0", "count": 1},
            {"name": "MIT", "count": 1},
        ]


class TestPypiMetadata:
    """Tests for PyPI metadata handling."""

    def test_get_pypi_json_skips_invalid_json_and_uses_next_url(self, monkeypatch):
        from nemo_helix_sdk_tools.license import generator

        class FakeResponse:
            def __init__(self, payload: dict[str, Any] | None = None, error: Exception | None = None):
                self.ok = True
                self.payload = payload or {}
                self.error = error
                self.status_code = 200

            def json(self) -> dict[str, Any]:
                if self.error:
                    raise self.error
                return self.payload

        generator._PYPI_JSON_CACHE.clear()
        responses = [
            FakeResponse(error=ValueError("invalid json")),
            FakeResponse(payload={"info": {"name": "demo"}}),
        ]
        calls = []

        def fake_get(url, timeout):
            calls.append((url, timeout))
            return responses.pop(0)

        monkeypatch.setattr(generator.requests, "get", fake_get)

        assert generator._get_pypi_json("demo", "1.2.3") == {"info": {"name": "demo"}}
        assert calls == [
            ("https://pypi.org/pypi/demo/1.2.3/json", 10),
            ("https://pypi.org/pypi/demo/json", 10),
        ]

    def test_license_url_from_pypi_info_accepts_only_http_urls(self):
        from nemo_helix_sdk_tools.license.generator import _license_url_from_pypi_info

        assert (
            _license_url_from_pypi_info(
                {"project_urls": {"License": "https://example.com/LICENSE"}},
                "NON-STANDARD",
            )
            == "https://example.com/LICENSE"
        )
        assert (
            _license_url_from_pypi_info(
                {"project_urls": {"License": "ftp://example.com/LICENSE"}},
                "NON-STANDARD",
            )
            == ""
        )
        assert (
            _license_url_from_pypi_info(
                {"project_urls": {"License": '=HYPERLINK("https://example.com")'}},
                "NON-STANDARD",
            )
            == ""
        )
