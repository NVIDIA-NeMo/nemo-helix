# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install published artifacts and exercise them in disposable CI environments.

This driver uses only the standard library. Helix checks run with the installed
wheel's Python; browser checks use a separate venv. No source dependencies are used.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from release_smoke_check import require_completion

SCRIPTS = Path(__file__).resolve().parent


class Smoke:
    def __init__(self) -> None:
        self.root = (
            Path(os.environ["SMOKE_ROOT"])
            if "SMOKE_ROOT" in os.environ
            else Path(os.environ["RUNNER_TEMP"]) / "release-smoke"
        )
        self.results = Path(os.environ["SMOKE_RESULTS"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ)
        self.env.update(
            {
                "SMOKE_ROOT": str(self.root),
                "UV_TOOL_DIR": str(self.root / "tools"),
                "UV_TOOL_BIN_DIR": str(self.root / "bin"),
                "UV_CACHE_DIR": str(self.root / "uv-cache"),
                "XDG_CONFIG_HOME": str(self.root / "config"),
                "XDG_STATE_HOME": str(self.root / "state"),
                "NHX_DATA_DIR": str(self.root / "data"),
                "NHX_BASE_URL": "http://127.0.0.1:8080",
                "DO_NOT_TRACK": "1",
            }
        )
        self.env.pop("PYTHONPATH", None)
        self.env.pop("VIRTUAL_ENV", None)
        self.nemo = str(self.root / "bin/nemo")
        self.python = str(self.root / "tools/nemo-helix/bin/python")
        self.state_file = self.results / "result.json"
        self.state = {"release": self.env["RELEASE_VERSION"], "stage": "setup", "result": "incomplete"}

    def stage(self, name: str) -> None:
        self.state["stage"] = name
        self.state_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        print(f"Smoke stage: {name}", flush=True)

    def command(
        self,
        args: list[str],
        timeout: int = 300,
        *,
        log: str | None = None,
        stdin: str | None = None,
        capture: bool = False,
    ) -> str:
        output = self.results / (log or f"{self.state['stage']}.log")
        with output.open("a", encoding="utf-8") as stream:
            process = subprocess.run(
                args,
                cwd=self.root,
                env=self.env,
                text=True,
                input=stdin,
                stdout=subprocess.PIPE if capture else stream,
                stderr=stream,
                timeout=timeout,
                check=False,
            )
            if capture:
                stream.write(process.stdout)
        if process.returncode:
            raise RuntimeError(f"{args[0]} failed with exit {process.returncode}; see {output.name}")
        return process.stdout if capture else ""

    def install(self, extras: str) -> None:
        args = [
            "uv",
            "tool",
            "install",
            "--python",
            "3.12",
            "--reinstall",
            f"nemo-helix[{extras}]=={self.env['WHEEL_VERSION']}",
        ]
        if self.env["WHEEL_INDEX"] != "https://pypi.org/simple":
            args += ["--index", self.env["WHEEL_INDEX"]]
        self.command(args, timeout=900)
        self.command(
            [
                self.python,
                "-c",
                "import importlib.metadata, sys; assert importlib.metadata.version('nemo-helix') == sys.argv[1]",
                self.env["WHEEL_VERSION"],
            ]
        )

    def start(self) -> None:
        self.command(
            [
                self.nemo,
                "services",
                "start",
                "--instance",
                "release-smoke",
                "--port",
                "8080",
                "--service-group",
                "all",
                "--controller-group",
                "all",
                "--ready-timeout",
                "300",
            ],
            timeout=360,
        )

    def ready(self) -> None:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{self.env['NHX_BASE_URL']}/health/ready", timeout=5) as response:
                    if response.status == 200 and json.load(response)["status"] == "ready":
                        return
            except (urllib.error.URLError, OSError, ValueError, KeyError):
                pass
            time.sleep(3)
        raise RuntimeError("Platform did not become ready within 300 seconds")

    def api(self, operation: str, timeout: int = 150) -> None:
        self.command([self.python, str(SCRIPTS / "release_smoke_api.py"), operation], timeout=timeout)

    def browser(self) -> None:
        browser_env = self.root / "browser"
        browser_python = str(browser_env / "bin/python")
        self.command(["uv", "venv", "--python", "3.12", str(browser_env)])
        self.command(["uv", "pip", "install", "--python", browser_python, "playwright==1.56.0"], timeout=180)
        self.command([browser_python, "-m", "playwright", "install", "--with-deps", "chromium"], timeout=300)
        self.command([browser_python, str(SCRIPTS / "release_smoke_browser.py")], timeout=150)

    def kubernetes(self) -> None:
        self.stage("download-chart")
        repository = self.env["CHART_REPOSITORY"]
        if repository.startswith("oci:"):
            self.command(
                ["helm", "registry", "login", "ghcr.io", "--username", self.env["GITHUB_ACTOR"], "--password-stdin"],
                stdin=self.env["GH_TOKEN"],
            )
        args = [
            "helm",
            "pull",
            repository if repository.startswith("oci:") else "nemo-helix",
            "--version",
            self.env["RELEASE_VERSION"],
            "--destination",
            str(self.root),
        ]
        if not repository.startswith("oci:"):
            args += ["--repo", repository]
        self.command(args)
        self.stage("pull-images")
        registry = self.env["IMAGE_REGISTRY"].split("/", 1)[0]
        token = self.env.get("NGC_REGISTRY_READ_TOKEN", "") if registry == "nvcr.io" else self.env.get("GH_TOKEN", "")
        if token:
            user = "$oauthtoken" if registry == "nvcr.io" else self.env["GITHUB_ACTOR"]
            self.command(["docker", "login", registry, "--username", user, "--password-stdin"], stdin=token)
        for image in ("nhx-api", "nhx-tasks"):
            ref = f"{self.env['IMAGE_REGISTRY']}/{image}:{self.env['IMAGE_TAG']}"
            self.command(["docker", "pull", ref], timeout=600)
            self.command(["kind", "load", "docker-image", "--name", self.env["KIND_CLUSTER_NAME"], ref], timeout=600)
        self.stage("helm-install")
        self.command(
            [
                "helm",
                "upgrade",
                "--install",
                "release-smoke",
                str(self.root / f"nemo-helix-{self.env['RELEASE_VERSION']}.tgz"),
                "--namespace",
                "release-smoke",
                "--create-namespace",
                "--wait",
                "--timeout",
                "8m",
                "-f",
                str(SCRIPTS / "release-smoke-values.yaml"),
                "--set-string",
                f"api.image.repository={self.env['IMAGE_REGISTRY']}/nhx-api",
                "--set-string",
                f"api.image.tag={self.env['IMAGE_TAG']}",
                "--set-string",
                f"core.image.repository={self.env['IMAGE_REGISTRY']}/nhx-api",
                "--set-string",
                f"core.image.tag={self.env['IMAGE_TAG']}",
                "--set-string",
                f"platformConfig.platform.image_registry={self.env['IMAGE_REGISTRY']}",
                "--set-string",
                f"platformConfig.platform.image_tag={self.env['IMAGE_TAG']}",
            ],
            timeout=540,
        )
        with (self.results / "port-forward.log").open("w") as log:
            process = subprocess.Popen(
                ["kubectl", "-n", "release-smoke", "port-forward", "svc/release-smoke-api", "8080:8080"],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self.env,
            )
        (self.root / "port-forward.pid").write_text(str(process.pid))
        self.env["SMOKE_AGENT_BASE_URL"] = "http://release-smoke-api.release-smoke.svc.cluster.local:8080"

    def run(self, target: str) -> None:
        self.stage("configuration")
        for key in ("NVIDIA_INFERENCE_URL", "NVIDIA_INFERENCE_KEY", "SMOKE_INFERENCE_MODEL"):
            if not self.env.get(key):
                raise RuntimeError(f"Missing CI setting: {key}")
        if target == "macos":
            self.stage("colima-install")
            self.command(["brew", "install", "colima", "docker", "qemu"], timeout=600)
            self.stage("colima-start")
            self.command(
                ["colima", "start", "--vm-type", "qemu", "--cpu", "2", "--memory", "6", "--disk", "30"], timeout=300
            )
            self.env["DOCKER_HOST"] = f"unix://{self.root}/config/colima/default/docker.sock"
            self.command(["docker", "info"])
        self.stage("wheel-install")
        self.install("all" if target == "macos" else "all,nemo-agents-plugin-deepagents")
        if target == "macos":
            self.stage("platform-start")
            self.start()
        else:
            self.kubernetes()
        self.stage("platform-ready")
        self.ready()
        self.stage("workspaces")
        self.api("workspaces")
        self.stage("studio")
        self.browser()
        self.stage("provider")
        self.api("provider", timeout=150)
        self.stage("inference")
        self.api("inference", timeout=120)
        if target == "macos":
            self.stage("platform-stop")
            self.command([self.nemo, "services", "stop", "--instance", "release-smoke"], timeout=60)
            self.stage("sample-dependency-install")
            self.install("all,nemo-agents-plugin-deepagents")
            self.stage("platform-restart")
            self.start()
            self.ready()
        self.stage("sample-config")
        self.api("sample-config")
        self.stage("agent-register")
        self.command(
            [
                self.nemo,
                "agents",
                "create",
                "--workspace",
                "release-smoke",
                "--name",
                "smoke-agent",
                "--agent-config",
                str(self.root / "agent.yaml"),
            ]
        )
        self.stage("agent-deploy")
        args = [
            self.nemo,
            "agents",
            "deploy",
            "--workspace",
            "release-smoke",
            "--agent",
            "smoke-agent",
            "--name",
            "smoke-agent",
            "--mode",
            "subprocess" if target == "macos" else "k8s",
            "--timeout",
            "300",
        ]
        if target == "kubernetes":
            args += ["--image", f"{self.env['IMAGE_REGISTRY']}/nhx-tasks:{self.env['IMAGE_TAG']}"]
        self.command(args, timeout=330)
        self.stage("agent-invoke")
        print("Sending to deployment 'release-smoke/smoke-agent'.", flush=True)
        response = self.command(
            [
                self.nemo,
                "agents",
                "invoke",
                "--workspace",
                "release-smoke",
                "--agent-deployment",
                "smoke-agent",
                "--input",
                "Hello, this is a smoke check.",
                "--timeout",
                "120",
                "--no-progress",
            ],
            timeout=120,
            capture=True,
        )
        require_completion(json.loads(response))
        self.state["result"] = "passed"
        self.stage("complete")

    def finalize(self, target: str) -> None:
        # Called by an always() step, including when setup/the driver was cancelled.
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text())
        cleanup_errors = []

        def attempt(args: list[str], *, timeout: int = 60, log: str = "cleanup.log") -> None:
            try:
                self.command(args, timeout=timeout, log=log)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                cleanup_errors.append(str(exc))

        # Preserve pod logs before deleting the agent deployment or its workspace.
        if target == "kubernetes":
            attempt(
                ["bash", str(SCRIPTS.parents[1] / "e2e/k8s/scripts/collect_k8s_logs.sh"), str(self.results / "k8s")],
                timeout=120,
                log="diagnostics.log",
            )
        if Path(self.nemo).exists():
            commands = [
                [self.nemo, "agents", "deployments", "delete", "smoke-agent", "--workspace", "release-smoke", "--yes"]
            ]
            commands += [[self.python, str(SCRIPTS / "release_smoke_api.py"), "cleanup"]]
            if target == "macos":
                self.env["DOCKER_HOST"] = f"unix://{self.root}/config/colima/default/docker.sock"
                commands += [[self.nemo, "services", "stop", "--instance", "release-smoke"]]
            for command in commands:
                attempt(command)
        if target == "macos":
            try:
                for directory in (self.root / "state", self.root / "data"):
                    for source in directory.rglob("*.log"):
                        destination = self.results / "services" / source.relative_to(self.root)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, destination)
            except OSError as exc:
                cleanup_errors.append(str(exc))
            if shutil.which("colima"):
                attempt(["colima", "delete", "--force"], timeout=120)
        else:
            attempt(["kind", "delete", "cluster", "--name", self.env["KIND_CLUSTER_NAME"]], timeout=120)
            pid_file = self.root / "port-forward.pid"
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), 15)
                except ProcessLookupError:
                    pass
        self.state["cleanup_errors"] = cleanup_errors
        self.state_file.write_text(json.dumps(self.state, indent=2))
        if cleanup_errors and self.state["result"] == "passed":
            raise RuntimeError("Smoke checks passed but cleanup failed; see cleanup.log")


if __name__ == "__main__":
    smoke = Smoke()
    if sys.argv[1] == "finalize":
        smoke.finalize(sys.argv[2])
    else:
        try:
            smoke.run(sys.argv[1])
        except Exception:
            smoke.state["result"] = "failed"
            smoke.stage(smoke.state["stage"])
            raise
