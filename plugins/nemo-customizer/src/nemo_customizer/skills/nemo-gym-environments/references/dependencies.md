<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dependency installation

Settle this before the package layout: it decides the format and whether the job needs egress.

In a NeMo Gym checkout, the per-server install is `setup_env_command` in `nemo_gym/cli/setup_command.py`.

Gym builds **one venv per server** at spin-up, under `NEMO_GYM_VENV_DIR` (`/opt/gym_venvs`). That directory ships **empty** — `NEMO_GYM_PREFETCH_CONFIGS` is off by default — so every environment pays this on first use, built-in ones included.

The platform itself installs nothing. `grpo_driver` calls `bootstrap_environment_package(..., install_wheels=False)`: it validates and stops there, because the venvs a server runs from do not exist until `RunHelper.start`.

**There are two installs, and they have different offline guarantees.** This is the single most-confused point about the wheels formats.

| | Step 1 — Gym builds the venv | Step 2 — NeMo-RL installs the package |
|---|---|---|
| When | During `RunHelper.start`, per server | After `RunHelper.start`, per agent/resources venv |
| What | `uv venv --seed` + `uv pip install` of the server's `requirements.txt` / `pyproject.toml`, **plus** `nemo-gym==<image version>`, `ray[default]==<pin>`, `openai==<pin>` | `uv pip install --no-index --find-links=wheels/ <names>` |
| Wheelhouse role | `UV_FIND_LINKS` — a **candidate pool**. Anything it covers installs from local files; anything it misses **goes to an index** | Hard offline. `--no-index` is explicit |
| Requirements | The server's own, resolved normally | One **unpinned** distribution name per wheel in `wheels/` |

So `--no-index` is real, but it is step 2 only. **A job is offline-clean only when `wheels/` also satisfies step 1**, which means the closure must cover:

- everything the server's `requirements.txt` names, and their transitive deps;
- **`nemo-gym` at exactly the version the training image reports** (below);
- `ray[default]` and `openai` at the versions Gym pins as head-server deps — the sub-venv starts from `uv venv --seed`, so nothing is inherited from the image.
- **`pip`** — `uv venv --seed` installs it into each new virtualenv *before* any `uv pip install` runs, resolving it like any other package. Without it in `wheels/` the seed step reaches for an index and the job fails with `Failed to install seed packages into virtual environment` / `No solution found when resolving: pip`. `uv venv` honours `UV_FIND_LINKS`, so vendoring the wheel is all that is needed. Python 3.13 seeds only `pip`.

Completeness of `wheels/` is what makes a sandboxed run work. The format name alone does not.

Step 2's requirements are unpinned on purpose (`wheelhouse_requirements`): the wheelhouse was resolved against the environment alone, while step 1 resolved the venv from the server's requirements. Pinning would overwrite packages the already-running server has imported. An unpinned requirement that is already satisfied is a no-op — which also means step 2 **cannot repair** a step 1 that resolved the wrong version from an index.

| Format | Server deps come from | Needs egress at spin-up |
|---|---|---|
| `native-v1` | An index, per the server's `requirements.txt` | **Yes** |
| `wheels-v1` | The package's `wheels/`, for whatever it vendors | **No**, if the closure covers step 1 as well |
| `adapter-wheels-v1` | The package's complete `wheels/` closure, including the agent harness dependencies | **No** |

The converter resolves the hub environment and `verifiers_agent` together. Source-only
releases are built into wheels on the internet-connected conversion host, and conversion
fails if any pinned distribution is still missing from `wheels/`. The image-bundled agent
harness uses the matching `verifiers` version pin, allowing Gym to resolve it from the
FileSet with `UV_OFFLINE=1` instead of following a Git URL.

Egress is operator config, never a job field: `NHX_RL_SANDBOX_ALLOW_INTERNET`, plus `NHX_RL_SANDBOX_PUBLIC_DNS_ALLOW` for hosts outside NeMo-RL's built-in `*.com` / `*.org` allowance (e.g. `hub.primeintellect.ai`). Check with the operator before committing a user to `native-v1` on a deny-default cluster. Details: `../../nemo-customizer/references/rl-kubernetes-runtime.md` § **Sandboxed Gym (GRPO)**.

## What goes in a server's `requirements.txt`

Gym reads it two different ways depending on where the server sits:

| Situation | Command Gym runs |
|---|---|
| Inside a Gym checkout (`../../pyproject.toml` exists) | `uv pip install -r requirements.txt <head deps>` |
| Staged FileSet (the normal case) | `(echo 'nemo-gym==<image version>' && grep -v -F '../..' requirements.txt) \| uv pip install -r /dev/stdin <head deps>` |

The second form is why a Gym server copies cleanly: its `-e nemo-gym[dev] @ ../../` line, meaningless outside a checkout, is stripped and replaced with `nemo-gym==<image version>`. Vendor a wheel of the `nemo_gym` library at that version, including the `dev` extra (`vllm_model` and `simple_agent` install `nemo-gym[dev]`). Do not `uv build` the Gym checkout: that wheel contains every built-in server.

**Do not ship a literally empty `requirements.txt`.** The file's *existence* is what makes Gym treat the directory as a server, and an empty one does currently install (`grep` finds nothing, `nemo-gym` and the head deps still reach uv). But the resulting venv contains only `nemo-gym`, `ray` and `openai` — so any other import in `app.py` fails at the first rollout — and it leaves the install command's `grep` exiting non-zero, which is harmless only because nothing enables `pipefail` on this path today. List the server's real imports; if it genuinely has none beyond `nemo-gym`, write a comment line rather than leaving the file empty.

## Vendoring a wheel closure

Two consumers, per the table above: Gym's venv build reaches the wheelhouse through `UV_FIND_LINKS` (candidate pool — misses go to an index), and NeMo-RL's later install reads it with `--no-index`. Build the closure for the first, and the second follows for free.

Resolve it from the **`nemo-gym` wheel plus the server's requirements**, not from the requirements alone: the sub-venv starts at `uv venv --seed`, so everything `nemo_gym` itself imports has to be present too.

**Wheels must target the architecture of the training nodes, not the build host.** The training images are published for both `linux/amd64` and `linux/arm64`, so confirm which the cluster runs (`kubectl get nodes -o jsonpath='{.items[*].status.nodeInfo.architecture}'`) and set `ARCH` to `x86_64` or `aarch64` accordingly. The interpreter is **Python 3.13** either way. A closure built for the wrong architecture passes `--validate-only` — which checks layout, never wheel tags — and then fails on the cluster with `has no wheels with a matching platform tag`.

Three versions must match what the **Gym host process** reports, because that is what Gym stamps into every sub-venv install: `nemo-gym` (the sub-venv pin) and `ray[default]` / `openai` (the head-server deps, read as `ray.__version__` / `openai.__version__` in `global_config.py`). All three come from a NeMo-RL checkout at the commit `NEMO_RL_REF` pins in `docker-bake.hcl`, which is what the training image is built from -- no access to the image itself is needed. Take `nemo-gym` from the Gym submodule and `ray`/`openai` from NeMo-RL's `uv.lock`. **Not** from Gym's own `uv.lock`: it pins different versions (`ray 2.55.1` / `openai 2.7.2` against the image's `2.56.1` / `2.6.1`), because the image resolves them with NeMo-RL, not Gym.

```bash
RL=<nemo-rl checkout at NEMO_RL_REF>   # git clone --recurse-submodules
# nemo_gym.__version__ is assembled in package_info.py; exec it rather than importing the
# package, which would pull the whole dependency set.
GYM_VERSION=$(cd "$RL/3rdparty/Gym-workspace/Gym" && python -c \
  "ns={}; exec(open('nemo_gym/package_info.py').read(), ns); print(ns['__version__'])")
read RAY_VERSION OPENAI_VERSION < <(python - "$RL/uv.lock" <<'EOF'
import sys, tomllib
pkgs = tomllib.load(open(sys.argv[1], "rb"))["package"]
pin = lambda n: next(p["version"] for p in pkgs if p.get("name") == n and "version" in p)
print(pin("ray"), pin("openai"))
EOF
)

mkdir -p my-env/wheels

# A wheel of the nemo_gym library alone, at the image's version. Building the Gym checkout
# itself would package every built-in server too, so build a copy that holds only nemo_gym.
GYM="$RL/3rdparty/Gym-workspace/Gym"
LIB="$(mktemp -d)/nemo-gym"
mkdir -p "$LIB" && cp -R "$GYM/nemo_gym" "$LIB/"
python - "$GYM/pyproject.toml" "$GYM_VERSION" > "$LIB/pyproject.toml" <<'EOF'
import json, sys, tomllib
project = tomllib.load(open(sys.argv[1], "rb"))["project"]
dev = project.get("optional-dependencies", {}).get("dev", [])
print(f"""[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "nemo-gym"
version = {json.dumps(sys.argv[2])}
dependencies = {json.dumps(project.get("dependencies", []))}

[project.optional-dependencies]
dev = {json.dumps(dev)}

[tool.setuptools.packages.find]
where = ["."]
include = ["nemo_gym", "nemo_gym.*"]

[tool.setuptools.package-data]
nemo_gym = ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.md", "**/*.txt"]""")
EOF
uv build --wheel --no-config --out-dir my-env/wheels "$LIB"
GYM_WHEEL=$(ls my-env/wheels/nemo_gym-*.whl)

grep -v -e '../..' -e 'nemo-gym' -e '^#' resources_servers/my_env/requirements.txt > server.in

# Resolve and download are separate steps: `--platform` requires `--no-deps`, and resolving
# on the build host would evaluate environment markers for the wrong OS.
cat > closure.in <<EOF
nemo-gym[dev] @ file://$PWD/$GYM_WHEEL
ray[default]==$RAY_VERSION
openai==$OPENAI_VERSION
pip
setuptools>=61,<81
setuptools-scm
hydra-core>=1.3,<1.4
omegaconf>=2.2,<2.4
-r server.in
EOF
uv pip compile closure.in --output-file closure.txt --no-header --no-config \
  --python-platform "$ARCH-unknown-linux-gnu" --python-version 3.13

pip download --dest my-env/wheels --no-cache-dir --no-deps \
  --python-version 3.13 \
  --platform "manylinux_2_39_$ARCH" \
  --platform "manylinux_2_28_$ARCH" \
  --platform "manylinux_2_17_$ARCH" \
  --platform "manylinux2014_$ARCH" \
  -r closure.txt

# omegaconf pins antlr4-python3-runtime, which publishes no wheel on PyPI. `wheels/` must hold
# wheels only, so build whatever arrived as a source artifact. (`--only-binary=:all:` is not an
# option here: it makes the resolve fail outright rather than fall back to the sdist.)
for sdist in my-env/wheels/*.tar.gz; do
  [ -e "$sdist" ] || break
  uv run --no-project --python 3.13 --with pip python -m pip wheel --no-deps \
    --wheel-dir my-env/wheels "$sdist" && rm "$sdist"
done

pi-to-gym-conversion --validate-only ./my-env
```

`setuptools` is capped below 81 — the release that removed `pkg_resources`, which Gym's pinned hydra imports at import time.

The sub-venv is created with `uv venv --seed`, so nothing carries over from the image — omit `ray[default]` / `openai` and the venv build reaches for an index even when the environment's own closure is complete. A `nemo-gym` version mismatch is worse than an omission: uv ignores your wheel and silently resolves upstream from PyPI.

`--platform` is repeated because pip matches tags literally rather than expanding a compatibility range, so a project publishing only against an older glibc floor is missed unless its tag is named.

**Vendor exactly one version of each distribution.** Several versions of one project is legal for a find-links pool, so it only warns locally — but the package then no longer pins what gets installed. Clear `wheels/` before rebuilding into it: copies overwrite by filename, so a wheel from an earlier run survives whenever the new closure resolved that project to a different version.
