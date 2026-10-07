# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests using disposable Git repositories and the public CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/rename/rename_plugins.py"
EVALS = ROOT / "tools/rename/evals.json"


class PluginRenameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        self.profile = Path(self.temp.name) / "profile.json"
        self.profile.write_text(
            json.dumps(
                {
                    "name": "Example rename",
                    "exclude": [],
                    "plugin": {
                        "replacements": {"old_plugin": "new_plugin"},
                        "paths": {"plugins/old_plugin": "plugins/new_plugin"},
                    },
                }
            )
        )

    def write(self, path: str, content: str) -> Path:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def run_rename(self, *args: str, profile: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--profile",
                str(profile or self.profile),
                "--repo-dir",
                str(self.repo),
                *args,
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_plugin_without_library_and_repeat_apply(self) -> None:
        self.write("plugins/old_plugin/code.py", "import old_plugin\n")
        first = self.run_rename("--allow-dirty")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual((self.repo / "plugins/new_plugin/code.py").read_text(), "import new_plugin\n")
        self.assertFalse((self.repo / "plugins/old_plugin").exists())
        self.assertEqual(self.run_rename("--verify").returncode, 0)
        self.assertIn("0 files would change", self.run_rename("--allow-dirty").stdout)

    def test_dry_run_and_verify_do_not_edit(self) -> None:
        original = self.write("plugins/old_plugin/code.py", "import old_plugin\n")
        self.assertEqual(self.run_rename("--dry-run").returncode, 0)
        self.assertEqual(self.run_rename("--verify").returncode, 1)
        self.assertEqual(original.read_text(), "import old_plugin\n")
        self.assertFalse((self.repo / "plugins/new_plugin").exists())

    def test_dirty_worktree_guard(self) -> None:
        original = self.write("code.py", "old_plugin")
        result = self.run_rename()
        self.assertEqual(result.returncode, 1)
        self.assertIn("worktree must be clean", result.stderr)
        self.assertEqual(original.read_text(), "old_plugin")

    def test_collision_preflight_prevents_all_edits(self) -> None:
        original = self.write("a.txt", "old_plugin")
        self.write("plugins/old_plugin/code.py", "old_plugin")
        self.write("plugins/new_plugin/code.py", "existing")
        result = self.run_rename("--allow-dirty")
        self.assertEqual(result.returncode, 1)
        self.assertIn("destination exists", result.stderr)
        self.assertEqual(original.read_text(), "old_plugin")

    def test_two_sources_cannot_share_destination(self) -> None:
        data = json.loads(self.profile.read_text())
        data["plugin"]["paths"]["plugins/other"] = "plugins/new_plugin"
        self.profile.write_text(json.dumps(data))
        original = self.write("plugins/old_plugin/code.py", "old_plugin")
        self.write("plugins/other/code.py", "old_plugin")
        result = self.run_rename("--allow-dirty")
        self.assertEqual(result.returncode, 1)
        self.assertIn("map to", result.stderr)
        self.assertEqual(original.read_text(), "old_plugin")

    def test_destination_ancestor_conflicts_prevent_all_edits(self) -> None:
        for destinations in (("x", "x/y"), ("x/y", "x")):
            data = json.loads(self.profile.read_text())
            data["plugin"]["paths"] = dict(zip(("a", "b"), destinations, strict=True))
            self.profile.write_text(json.dumps(data))
            for mode in ("--dry-run", "--verify", "--allow-dirty"):
                with self.subTest(destinations=destinations, mode=mode):
                    self.repo = Path(self.temp.name) / f"repo-{destinations[0].replace('/', '-')}-{mode}"
                    self.repo.mkdir()
                    subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
                    originals = {name: self.write(name, "old_plugin") for name in ("a", "b", "c")}
                    result = self.run_rename(mode)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("Destination path conflict", result.stderr)
                    for original in originals.values():
                        self.assertEqual(original.read_text(), "old_plugin")
                    self.assertFalse((self.repo / "x").exists())
                    self.assertEqual({path.name for path in self.repo.iterdir()}, {".git", "a", "b", "c"})

    def test_unsafe_profile_path_rejected(self) -> None:
        data = json.loads(self.profile.read_text())
        data["plugin"]["paths"]["plugins/old_plugin"] = "../escape"
        self.profile.write_text(json.dumps(data))
        self.assertEqual(self.run_rename("--dry-run").returncode, 1)

    def test_library_mapping_takes_precedence_over_module_prefix(self) -> None:
        data = json.loads(self.profile.read_text())
        data["library"] = {"replacements": {"old_plugin_sdk": "new_sdk"}}
        self.profile.write_text(json.dumps(data))
        self.write("code.py", "import old_plugin_sdk\nimport old_plugin\n")
        result = self.run_rename("--allow-dirty")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.repo / "code.py").read_text(), "import new_sdk\nimport new_plugin\n")

    def test_evals_wrapper_works_outside_target_checkout(self) -> None:
        self.write("README.md", "NeMo Evaluator")
        result = subprocess.run(
            [str(ROOT / "tools/rename/rename-to-nemo-evals.sh"), "--repo-dir", str(self.repo), "--dry-run"],
            cwd=self.temp.name,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NeMo Evaluator to NeMo Helix Evals", result.stdout)
        self.assertEqual((self.repo / "README.md").read_text(), "NeMo Evaluator")

    def test_symlinks_ignored_files_and_binary_content(self) -> None:
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("old_plugin")
        (self.repo / "link.txt").symlink_to(outside)
        self.write(".gitignore", "ignored/\n")
        ignored = self.write("ignored/code.py", "old_plugin")
        binary = self.repo / "plugins/old_plugin/binary.dat"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"old_plugin\0\xff")
        self.assertEqual(self.run_rename("--allow-dirty").returncode, 0)
        self.assertEqual(outside.read_text(), "old_plugin")
        self.assertEqual(ignored.read_text(), "old_plugin")
        self.assertEqual((self.repo / "plugins/new_plugin/binary.dat").read_bytes(), b"old_plugin\0\xff")

    def test_internal_symlink_moves_and_remaps_its_target(self) -> None:
        data = json.loads(self.profile.read_text())
        data["plugin"]["paths"]["skills/old_plugin"] = "skills/new_plugin"
        self.profile.write_text(json.dumps(data))
        self.write("skills/old_plugin/SKILL.md", "old_plugin")
        link = self.repo / "plugins/old_plugin/skill"
        link.parent.mkdir(parents=True)
        link.symlink_to("../../skills/old_plugin")
        result = self.run_rename("--allow-dirty")
        self.assertEqual(result.returncode, 0, result.stderr)
        moved = self.repo / "plugins/new_plugin/skill"
        self.assertTrue(moved.is_symlink())
        self.assertEqual(moved.readlink(), Path("../../skills/new_plugin"))
        self.assertEqual((moved / "SKILL.md").read_text(), "new_plugin")
        self.assertEqual(self.run_rename("--verify").returncode, 0)

    def test_dangling_symlink_target_is_checked_by_verifier(self) -> None:
        data = json.loads(self.profile.read_text())
        data["plugin"]["paths"]["skills/old_plugin"] = "skills/new_plugin"
        self.profile.write_text(json.dumps(data))
        link = self.repo / "plugins/new_plugin/skill"
        link.parent.mkdir(parents=True)
        link.symlink_to("../../skills/old_plugin")
        self.assertEqual(self.run_rename("--verify").returncode, 1)
        result = self.run_rename("--allow-dirty")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(link.readlink(), Path("../../skills/new_plugin"))
        self.assertEqual(self.run_rename("--verify").returncode, 0)

    def test_include_exclude_and_tracked_ignored_files(self) -> None:
        self.write(".gitignore", "docs/\n")
        self.write("docs/yes.md", "old_plugin")
        excluded = self.write("docs/no.md", "old_plugin")
        other = self.write("other.md", "old_plugin")
        subprocess.run(["git", "-C", str(self.repo), "add", "-f", "docs"], check=True)
        result = self.run_rename("--allow-dirty", "--include-glob", "docs/**", "--exclude-glob", "docs/no.md")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.repo / "docs/yes.md").read_text(), "new_plugin")
        self.assertEqual(excluded.read_text(), "old_plugin")
        self.assertEqual(other.read_text(), "old_plugin")

    def test_evals_preserves_local_evaluator_calls_and_renames_service_lists(self) -> None:
        self.write(
            "packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk/metrics/retrieval.py", "evaluator.evaluate({})\n"
        )
        self.write("plugins/nemo-optimization/driver.py", "evaluator.evaluate(spec)\n")
        self.write(".github/workflows/ci.yaml", "NEMO_PLUGIN_SERVICES_ALLOWLIST: evaluator\n")
        self.write("run.sh", "nemo services run --services auth,entities,evaluator\n")
        self.write(
            "web/consumer.ts", "evaluatorCreateEvaluateJob; useEvaluatorListEvaluateJobs; EvaluatorTaskDefinition;\n"
        )
        self.write("packages/nemo_helix_plugin/src/nemo_helix_plugin/authz_discovery.py", "    owner = service.name\n")
        result = self.run_rename("--allow-dirty", profile=EVALS)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.repo / "packages/nhx_evals_sdk/src/nhx_evals_sdk/metrics/retrieval.py").read_text(),
            "evaluator.evaluate({})\n",
        )
        self.assertEqual((self.repo / "plugins/nemo-optimization/driver.py").read_text(), "evaluator.evaluate(spec)\n")
        self.assertIn("ALLOWLIST: evals", (self.repo / ".github/workflows/ci.yaml").read_text())
        self.assertIn("auth,entities,evals", (self.repo / "run.sh").read_text())
        self.assertEqual(
            (self.repo / "web/consumer.ts").read_text(),
            "evalsCreateEvaluateJob; useEvalsListEvaluateJobs; EvaluatorTaskDefinition;\n",
        )
        self.assertIn(
            '{"evals": "evaluator"}.get(service.name, service.name)',
            (self.repo / "packages/nemo_helix_plugin/src/nemo_helix_plugin/authz_discovery.py").read_text(),
        )
        self.assertEqual(self.run_rename("--verify", profile=EVALS).returncode, 0)

    def test_evals_renames_job_source_consumers_and_existing_overrides(self) -> None:
        self.write(
            "web/packages/studio/src/api/evaluation/evaluator-jobs.ts",
            (ROOT / "web/packages/studio/src/api/evaluation/evaluator-jobs.ts").read_text(),
        )
        self.write(
            "jobs.py",
            'sources = ["nemo-evaluator", "nemo-evaluator.agent-evaluate", "nemo-evaluator.retrieve-eval"]\n'
            'registrations = ["evaluator.evaluate", "evaluator.agent-evaluate", "evaluator.retrieve-eval"]\n'
            'add_job_routes(EvaluateJob, service_name="nemo-evaluator", authz=scope)\n',
        )
        self.assertEqual(self.run_rename("--verify", profile=EVALS).returncode, 1)
        result = self.run_rename("--allow-dirty", profile=EVALS)
        self.assertEqual(result.returncode, 0, result.stderr)
        consumer = (self.repo / "web/packages/studio/src/api/evaluation/evaluator-jobs.ts").read_text()
        self.assertIn("['nemo-evals', 'nemo-evals.agent-evaluate']", consumer)
        self.assertNotIn("nemo-evaluator", consumer)
        self.assertEqual(
            (self.repo / "jobs.py").read_text(),
            'sources = ["nemo-evals", "nemo-evals.agent-evaluate", "nemo-evals.retrieve-eval"]\n'
            'registrations = ["evals.evaluate", "evals.agent-evaluate", "evals.retrieve-eval"]\n'
            'add_job_routes(EvaluateJob, service_name="nemo-evals", authz=scope)\n',
        )
        self.assertEqual(self.run_rename("--verify", profile=EVALS).returncode, 0)
        self.assertIn("0 files would change", self.run_rename("--allow-dirty", profile=EVALS).stdout)

    def test_evals_renames_job_sources_and_preserves_permissions_and_entity_kinds(self) -> None:
        paths = [
            "plugins/nemo-evaluator/src/nemo_evaluator/service.py",
            "plugins/nemo-evaluator/src/nemo_evaluator/authz.py",
            "plugins/nemo-evaluator/src/nemo_evaluator/entities.py",
            "plugins/nemo-evaluator/src/nemo_evaluator/api/v2/tasks.py",
            "plugins/nemo-evaluator/src/nemo_evaluator/api/v2/metrics.py",
            "plugins/nemo-evaluator/src/nemo_evaluator/api/v2/results.py",
            "plugins/nemo-evaluator/src/nemo_evaluator/api/v2/tasksets.py",
            "plugins/nemo-evaluator/src/nemo_evaluator/cli.py",
            "plugins/nemo-evaluator/src/nemo_evaluator/config.py",
            "plugins/nemo-evaluator/src/nemo_evaluator/tasks/evaluate.py",
            "packages/nemo_helix_plugin/src/nemo_helix_plugin/client/client.py",
        ]
        for path in paths:
            self.write(path, (ROOT / path).read_text())
        self.write(
            "plugins/nemo-evaluator/pyproject.toml", (ROOT / "plugins/nemo-evaluator/pyproject.toml").read_text()
        )
        self.write(
            "packages/nemo_evaluator_sdk/src/nemo_evaluator_sdk/metric.py", "from nemo_evaluator_sdk import Metric\n"
        )
        self.write(
            "packages/nemo_helix_plugin/src/nemo_helix_plugin/evaluator/client.py",
            "from nemo_helix_plugin.evaluator import endpoints\n",
        )
        self.write(
            "usage.txt",
            "NeMo Evaluator SDK; nemo evaluator info; /apis/evaluator/v2\n"
            'evaluator.agent-evaluate; nemo-evaluator.agent-evaluate\nkind="evaluator"\n',
        )
        result = self.run_rename("--allow-dirty", profile=EVALS)
        self.assertEqual(result.returncode, 0, result.stderr)
        service = (self.repo / "plugins/nemo-evals/src/nemo_evals/service.py").read_text()
        self.assertIn('name: ClassVar[str] = "evals"', service)
        self.assertIn("add_job_routes(EvaluateJob, authz=scope)", service)
        self.assertIn("from nemo_evals.jobs.evaluate import EvaluateJob", service)
        self.assertIn('AGENT_EVAL_JOB_SOURCE = "nemo-evals.agent-evaluate"', service)
        self.assertIn('RETRIEVE_EVAL_JOB_SOURCE = "nemo-evals.retrieve-eval"', service)
        self.assertNotIn("nemo-evaluator", service)
        self.assertIn('namespace="evaluator"', service)
        authz = (self.repo / "plugins/nemo-evals/src/nemo_evals/authz.py").read_text()
        self.assertIn('AuthzScope("evaluator")', authz)
        tasks = (self.repo / "plugins/nemo-evals/src/nemo_evals/api/v2/tasks.py").read_text()
        self.assertIn('namespace="evaluator.tasks"', tasks)
        cli = (self.repo / "plugins/nemo-evals/src/nemo_evals/cli.py").read_text()
        self.assertIn('name: ClassVar[str] = "evals"', cli)
        config = (self.repo / "plugins/nemo-evals/src/nemo_evals/config.py").read_text()
        self.assertIn('plugin_name: ClassVar[str] = "evals"', config)
        runner = (self.repo / "plugins/nemo-evals/src/nemo_evals/tasks/evaluate.py").read_text()
        self.assertIn('service_name="evals"', runner)
        client = (self.repo / "packages/nemo_helix_plugin/src/nemo_helix_plugin/client/client.py").read_text()
        self.assertIn("def evals(self)", client)
        self.assertIn("from nemo_helix_plugin.evals.client import", client)
        entities = (self.repo / "plugins/nemo-evals/src/nemo_evals/entities.py").read_text()
        self.assertIn('__entity_type__: ClassVar[str] = "metric_bundle"', entities)
        sdk = self.repo / "packages/nhx_evals_sdk/src/nhx_evals_sdk/metric.py"
        self.assertEqual(sdk.read_text(), "from nhx_evals_sdk import Metric\n")
        self.assertTrue((self.repo / "packages/nemo_helix_plugin/src/nemo_helix_plugin/evals/client.py").exists())
        toml = (self.repo / "plugins/nemo-evals/pyproject.toml").read_text()
        self.assertIn('evals = "nemo_evals.service:', toml)
        usage = (self.repo / "usage.txt").read_text()
        self.assertIn("NeMo Helix Evals SDK; nemo evals info; /apis/evals/v2", usage)
        self.assertIn("evals.agent-evaluate; nemo-evals.agent-evaluate", usage)
        self.assertIn('kind="evaluator"', usage)
        self.assertEqual(self.run_rename("--verify", profile=EVALS).returncode, 0)


if __name__ == "__main__":
    unittest.main()
