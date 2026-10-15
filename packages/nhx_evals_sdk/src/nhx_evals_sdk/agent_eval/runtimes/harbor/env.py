# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand ``env_secrets`` and ``env_vars`` to a Harbor agent.

Harbor receives ``${NAME}`` templates naming the env var that already holds each secret
(:func:`harbor_env_templates`), so no value is copied or written to ``os.environ``. ``env_vars`` are
checked against what Harbor would mishandle (:func:`validate_harbor_env`).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from nhx_evals_sdk.agent_eval.runtimes.provenance import require_no_plaintext_credentials
from nhx_evals_sdk.agent_eval.runtimes.secrets import env_secret_vars
from nhx_evals_sdk.resolver_protocols import EnvSecretSource
from nhx_evals_sdk.values.common import SecretRef

logger = logging.getLogger(__name__)

#: Copy of Harbor's ``harbor.utils.env._SENSITIVE_KEY_RE``: Harbor masks ``AgentConfig.env`` values under
#: matching keys in ``config.json`` and scrubs those values from every trial file. Copied because Harbor
#: is an optional SDK extra and ``HarborRuntimeConfig`` must validate without it; a test pins it to
#: Harbor's pattern and flags.
HARBOR_SENSITIVE_ENV_KEY_RE = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH)", re.IGNORECASE)


def harbor_env_templates(env_secrets: Mapping[str, SecretRef], source: EnvSecretSource) -> dict[str, str]:
    """Map each ``env_secrets`` key to a ``${<source var>}`` template Harbor expands when it creates the agent.

    Raises before anything runs when a secret is missing.
    """
    return {name: "${" + var + "}" for name, var in env_secret_vars(env_secrets, source).items()}


def validate_harbor_env(env_vars: Mapping[str, str], env_secrets: Mapping[str, object]) -> None:
    """Refuse ``env_vars`` Harbor would mishandle; shared by the SDK config, the job target and the run.

    ``env_vars`` are non-secret literals handed only to the agent.
    """
    overlap = sorted(set(env_vars) & set(env_secrets))
    if overlap:
        raise ValueError(f"{overlap} appear in both env_vars and env_secrets; name each variable once")
    for key, value in env_vars.items():
        # Harbor expands a `${NAME}` value from the process environment, which would forward the
        # job's environment into the agent.
        if "${" in value:
            raise ValueError(
                f"env_vars[{key!r}] looks like a ${{NAME}} template; env_vars are literals. "
                "Use env_secrets for values from the environment."
            )
        if HARBOR_SENSITIVE_ENV_KEY_RE.search(key):
            raise ValueError(
                f"env_vars[{key!r}]: Harbor treats keys matching KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH "
                "(case-insensitive substring) as secrets: it masks them in config.json and scrubs their values "
                "from every trial file. A credential belongs in env_secrets. For a non-secret setting, pass it "
                "through agent_kwargs if the agent supports it, or use a key that doesn't match."
            )
    require_no_plaintext_credentials(env_vars, field="env_vars", alternative="env_secrets")


def warn_unscrubbed_secret_keys(env_secrets: Mapping[str, object]) -> None:
    """Warn about ``env_secrets`` keys Harbor won't scrub from trial files.

    Harbor redacts secret values from trial files only under keys matching its sensitive-key pattern.
    Not refused: some agents read a fixed variable name the caller can't rename.
    """
    for key in env_secrets:
        if not HARBOR_SENSITIVE_ENV_KEY_RE.search(key):
            logger.warning(
                "env_secrets[%r] doesn't match Harbor's sensitive-key pattern (KEY|SECRET|TOKEN|PASSWORD|"
                "CREDENTIAL|AUTH), so Harbor won't scrub its value from trial logs. Rename it if the agent allows.",
                key,
            )
