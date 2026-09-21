# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]


def divisors(n: int) -> list[int]:
    """Return all divisors of n in ascending order."""
    return [d for d in range(1, n + 1) if n % d == 0]


def compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    """Compile regex patterns for module name matching."""
    return [re.compile(p) for p in patterns]


def name_matches(name: str, include_res, exclude_res):
    """Check if a module name matches include/exclude patterns."""
    if exclude_res and any(r.search(name) for r in exclude_res):
        return False
    if not include_res:
        return True
    return any(r.search(name) for r in include_res)


def get_flat_files_list(parent_dir: str) -> List[str]:
    """
    Get a list of files in a directory
    """
    parent_path = Path(parent_dir).resolve()
    if not parent_path.exists():
        raise ValueError(f"Path {parent_dir} does not exist")
    if not parent_path.is_dir():
        raise ValueError(f"Path {parent_dir} is not a directory")

    return [str(path) for path in parent_path.rglob("*") if path.is_file()]


def is_adapter_file_present(files: List[str]) -> bool:
    """
    Check if the any file is a LoRA adapter file
    """
    for file in files:
        if not file:
            continue
        if any(adapter_file in file.lower() for adapter_file in ADAPTER_FILES):
            return True
    return False


def check_directory_structure(path: Path | str, target: Dict[str, Optional[Dict]]) -> bool:
    if isinstance(path, str):
        path = Path(path)

    if not path.is_dir():
        logger.error(f"Provided path '{path}' is not a directory")
        return False

    try:
        got_files = {f.name for f in path.iterdir()}
    except OSError as e:
        logger.error(f"Cannot read directory '{path}'. Reason: {e}")
        return False

    expected_files = set(target.keys())
    missing = expected_files - got_files
    if missing:
        logger.debug(f"Mismatch in '{path}': Missing items -> {missing}")
        return False

    for name, _target in target.items():
        current_path = path / name
        if isinstance(_target, dict):
            # this is a directory
            if not current_path.is_dir():
                return False
            if not check_directory_structure(current_path, _target):
                return False
        elif _target is None:
            if not current_path.is_file():
                logger.debug(f"Mismatch: '{current_path}' is expected to be a file but is a directory.")
                return False
    return True


def create_tmpdir_with_files(temp_dir: str, paths: List[str], root: Optional[str] = None) -> Path:
    td = Path(temp_dir)
    for p in paths:
        # account for absolute paths
        if root and p[0] == "/":
            p = p[len(root) :]
        if not p:
            continue
        if p[0] == "/":
            p = p[1:]
        file_path = td / Path(p)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
    return td


def is_nemo_model_directory(model_path: Path | str) -> bool:
    nemo_structure = {
        "context": {"nemo_tokenizer": {}, "model.yaml": None},
        "weights": {"metadata.json": None},
    }
    return check_directory_structure(model_path, nemo_structure)


def _has_weight_files_on_disk(model_path: Path) -> bool:
    """Check for model weight files on the local filesystem."""
    safe_tensor_file = model_path / "model.safetensors"
    if safe_tensor_file.is_file() or any(model_path.glob("model-*.safetensors")):
        return True

    logger.debug(f"Missing model weights files in the form of {safe_tensor_file} or {model_path}/model-*.safetensors")

    pytorch_bin_file = model_path / "pytorch_model.bin"
    if pytorch_bin_file.is_file() or any(model_path.glob("pytorch_model-*.bin")):
        return True

    logger.debug(f"Missing model weights files in the form of {pytorch_bin_file} or {model_path}/pytorch_model-*.bin")
    return False


def _has_weight_files_in_listing(file_listing: list[str]) -> bool:
    """Check for model weight files in a remote file listing.

    The listing contains file paths for weight files "model.safetensors" for small models and
    "model-00001-of-00013.safetensors" for large models that may not exist on the
    local filesystem because we explicitly filter the suffixes in method analyze_checkpoint in run.py
    """
    weight_suffixes = (".safetensors", ".bin", ".safetensors.index.json", ".bin.index.json")
    for path in file_listing:
        if path.endswith(weight_suffixes):
            return True

    logger.debug(f"No weight files found in file listing ({len(file_listing)} entries)")
    return False


def is_huggingface_model_directory(
    model_path: Path | str,
    file_listing: list[str] | None = None,
) -> bool:
    """
    Checks if a directory contains the necessary files to be considered a
    Hugging Face model directory.

    Config and tokenizer files are always validated against the local
    filesystem.  Weight files can be validated against either a remote
    file_listing (when only metadata was downloaded locally) or
    the local filesystem (when the full checkpoint is present).

    Args:
        model_path: The path to the local directory to check.
        file_listing: Optional list of file paths (e.g. from a fileset
            API).  When provided, weight-file validation is performed
            against this listing instead of the local filesystem.

    Returns:
        True if the directory contains a config.json file and model weights,
        False otherwise.
    """
    if isinstance(model_path, str):
        model_path = Path(model_path)

    # 1. Check for the mandatory config.json file
    config_file = model_path / "config.json"
    if not config_file.is_file():
        logger.debug(f"Missing {config_file}")
        return False

    tokenizer_files = [
        model_path / "tokenizer.json",
        model_path / "tokenizer_config.json",
        model_path / "vocab.txt",
        model_path / "merges.txt",
    ]
    if not any(tf.is_file() for tf in tokenizer_files):
        logger.debug(f"Missing any tokenizer file: at least one of [{tokenizer_files}] is required")
        return False

    # 2. Check for the presence of model weight files (either safetensors or pytorch bin).
    # We need to add check for both file_listing and on disk because this function can be called from
    # 1) the fileset API in the model-spec task where weight files are intentionally not
    #    downloaded locally.
    # 2) the CLI and other callers that operate on a full checkpoint.
    if file_listing is not None:
        has_weights = _has_weight_files_in_listing(file_listing)
    else:
        has_weights = _has_weight_files_on_disk(model_path)
    if not has_weights:
        logger.info(f"No model weight files found for {model_path} in file listing: {file_listing} or on disk")

    return has_weights


# Kwarg name -> the pair of values probed against each other. A template that
# branches on the kwarg renders these two differently. Booleans cover the
# Qwen3/Nemotron and DeepSeek spellings; gpt-oss-style templates interpolate a
# reasoning-effort string into the system prompt instead, so that one is probed
# with two efforts rather than with on/off.
REASONING_CONTROL_KWARGS: dict[str, tuple[object, object]] = {
    "enable_thinking": (True, False),
    "thinking": (True, False),
    "reasoning_effort": ("high", "low"),
}

_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]
_PROBE_CONTEXT = {
    "messages": _PROBE_MESSAGES,
    "add_generation_prompt": True,
    "bos_token": "",
    "eos_token": "",
    "tools": None,
}


def default_chat_template(chat_template: object) -> str | None:
    """Resolve the template that inference will actually render.

    Transformers exposes ``chat_template`` as a plain string, or as a collection
    of named templates when a model ships more than one. Only the ``default``
    entry serves ordinary requests; a model that puts its reasoning behaviour in
    a separately-named template is switched by template *selection*, not by a
    kwarg, so the others must not be consulted here.
    """
    if isinstance(chat_template, str):
        return chat_template
    if isinstance(chat_template, dict):
        entry = chat_template.get("default")
        return entry if isinstance(entry, str) else None
    if isinstance(chat_template, list):
        for entry in chat_template:
            if isinstance(entry, dict) and entry.get("name") == "default":
                template = entry.get("template")
                return template if isinstance(template, str) else None
    return None


def _render_chat_template(template: str, **kwargs: object) -> str | None:
    """Render a chat template, or return None if it cannot be rendered here.

    jinja2 arrives with transformers, which is only installed in the task image,
    so the import is deliberately lazy — an API-server import of this module must
    not require it.
    """
    try:
        from jinja2 import ChainableUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        logger.info("jinja2 unavailable; cannot determine chat-template reasoning support")
        return None

    def raise_exception(message: str) -> None:
        raise RuntimeError(message)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=ChainableUndefined,
    )
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: ""
    try:
        return env.from_string(template).render(**_PROBE_CONTEXT, **kwargs)
    except Exception as exc:
        logger.info(f"Could not render chat template while probing reasoning support: {exc}")
        return None


def detect_reasoning_control(chat_template: object) -> bool | None:
    """Whether the served chat template gives a caller control over reasoning.

    Renders the template with each candidate kwarg set two different ways and
    compares the output. A template that branches on the kwarg renders
    differently; one that merely mentions it — or hardcodes it, as
    ``{%- set enable_thinking = true %}`` does — renders the same both ways and
    is correctly reported as no control.

    Returns None when the answer is undetermined rather than negative: the
    template could not be resolved or rendered. Absent a template entirely the
    answer is a definite False, since there is nothing to honour the kwarg.

    Scope is the ``chat_template_kwargs`` family only. Reasoning is also
    disabled by means that leave nothing in the template to read:

    * ``reasoning_effort`` sent as a *request parameter*, and Anthropic's
      ``thinking`` config, are properties of the serving API rather than of the
      weights — switchyard handles each separately in its OpenAI and Anthropic
      backends. Only the template-interpolated spelling is visible here.
    * Prompt-level switches such as Qwen3's ``/no_think`` are trained behaviour.
      The template passes the marker through untouched, so rendering cannot see
      it; only an inference probe could.

    A False therefore means "this template will ignore the kwarg", not "reasoning
    cannot be turned off".
    """
    if chat_template is None:
        return False

    template = default_chat_template(chat_template)
    if template is None:
        return None

    candidates = {kwarg: values for kwarg, values in REASONING_CONTROL_KWARGS.items() if kwarg in template}
    if not candidates:
        return False

    undetermined = False
    for kwarg, (one, other) in candidates.items():
        rendered_one = _render_chat_template(template, **{kwarg: one})
        rendered_other = _render_chat_template(template, **{kwarg: other})
        if rendered_one is None or rendered_other is None:
            undetermined = True
            continue
        if rendered_one != rendered_other:
            return True
    return None if undetermined else False
