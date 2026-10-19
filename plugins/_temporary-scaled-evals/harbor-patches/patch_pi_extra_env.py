# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route Harbor Pi profile values and instructions through safe runtime paths."""

from __future__ import annotations

import sys
from pathlib import Path

COMMAND_PREFIX_REPLACEMENTS = (
    (
        '                f". ~/.nvm/nvm.sh; "\n                f"pi --print --mode json --no-session "',
        '                f". ~/.nvm/nvm.sh; "\n'
        '                f"printf %s {escaped_instruction} | "\n'
        '                f"pi --print --mode json --no-session "',
    ),
    (
        '                f". ~/.nvm/nvm.sh; "\n'
        '                f"pi --print --mode json --session-dir /logs/agent/pi/sessions "',
        '                f". ~/.nvm/nvm.sh; "\n'
        '                f"printf %s {escaped_instruction} | "\n'
        '                f"pi --print --mode json --session-dir /logs/agent/pi/sessions "',
    ),
)

REPLACEMENTS = (
    ("import json\nimport os\nimport shlex\n", "import json\nimport shlex\n"),
    ("val = os.environ.get(key)", "val = self._get_env(key)"),
    ('                f"{escaped_instruction} "\n', ""),
    ("2>&1 </dev/null |", "2>&1 |"),
    (
        """    @with_prompt_template
    async def run(
""",
        """    async def _write_inference_headers_config(
        self, environment: BaseEnvironment, provider: str
    ) -> None:
        raw_headers = self._get_env("SCALED_EVALS_INFERENCE_HEADERS_JSON")
        if not raw_headers:
            return
        headers = json.loads(raw_headers)
        if not isinstance(headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in headers.items()
        ):
            raise ValueError(
                "SCALED_EVALS_INFERENCE_HEADERS_JSON must be an object of strings"
            )
        merge_script = r\'''const fs = require("fs");
const path = require("path");
const target = path.join(process.env.HOME, ".pi", "agent", "models.json");
let config = {};
if (fs.existsSync(target)) config = JSON.parse(fs.readFileSync(target, "utf8"));
if (!config || Array.isArray(config) || typeof config !== "object") {
  throw new Error("Pi models.json must contain an object");
}
if (!config.providers || Array.isArray(config.providers) || typeof config.providers !== "object") {
  config.providers = {};
}
const provider = process.env.SCALED_EVALS_INFERENCE_PROVIDER;
let providerConfig = config.providers[provider];
if (!providerConfig || Array.isArray(providerConfig) || typeof providerConfig !== "object") {
  providerConfig = {};
}
let existing = providerConfig.headers;
if (!existing || Array.isArray(existing) || typeof existing !== "object") existing = {};
const incoming = JSON.parse(process.env.SCALED_EVALS_INFERENCE_HEADERS_JSON);
for (const key of Object.keys(existing)) {
  if (key.toLowerCase() === "x-inference-priority") delete existing[key];
}
providerConfig.headers = {...existing, ...incoming};
config.providers[provider] = providerConfig;
fs.mkdirSync(path.dirname(target), {recursive: true});
fs.writeFileSync(target, JSON.stringify(config, null, 2) + "\\n");
\'''
        await self.exec_as_agent(
            environment,
            command=f"node -e {shlex.quote(merge_script)}",
            env={
                "SCALED_EVALS_INFERENCE_HEADERS_JSON": json.dumps(headers),
                "SCALED_EVALS_INFERENCE_PROVIDER": provider,
            },
        )

    @with_prompt_template
    async def run(
""",
    ),
    (
        """        skills_command = self._build_register_skills_command()
""",
        """        await self._write_inference_headers_config(environment, provider)

        skills_command = self._build_register_skills_command()
""",
    ),
)


def _patch_command_prefix(source: str, path: Path) -> str:
    if any(new in source for _, new in COMMAND_PREFIX_REPLACEMENTS):
        return source

    matches = [(old, new) for old, new in COMMAND_PREFIX_REPLACEMENTS if source.count(old) == 1]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one supported Pi command prefix in {path}; "
            "update this patch for the installed Harbor version"
        )

    old, new = matches[0]
    return source.replace(old, new, 1)


def patch(path: Path) -> None:
    source = path.read_text()
    source = _patch_command_prefix(source, path)
    for old, new in REPLACEMENTS:
        if new:
            if new in source:
                continue
        elif old not in source:
            continue
        if source.count(old) != 1:
            raise RuntimeError(
                f"expected exactly one Pi source anchor in {path}; update this patch for the installed Harbor version"
            )
        source = source.replace(old, new, 1)
    path.write_text(source)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} PI_PY")
    patch(Path(sys.argv[1]))
