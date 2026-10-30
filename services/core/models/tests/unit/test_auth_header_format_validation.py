# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelProvider.auth_header_format validation.

The validator (`_validate_auth_header_format`) enforces three contracts beyond
Jinja2 syntax:
1. Exactly one `{{ auth_secret }}` substitution (duplicates would leak the
   resolved secret into the rendered header twice).
2. The template renders to `<Header-Name>: <Header-Value>` so the runtime
   proxy can split on `": "` cleanly.
3. `None` is accepted (provider falls back to the default `Authorization:
   Bearer {{ auth_secret }}` at request time).
"""

from pathlib import Path

import pytest
from nhx.core.models.schemas import _validate_auth_header_format


class TestAuthHeaderFormatValidation:
    @pytest.mark.parametrize(
        "value",
        [
            "Authorization: Bearer {{ auth_secret }}",
            "X-Api-Key: {{ auth_secret }}",
            "Authorization: Token {{ auth_secret }}",
            "X-Custom-Auth: prefix-{{ auth_secret }}-suffix",
        ],
    )
    def test_accepts_valid_templates(self, value: str) -> None:
        assert _validate_auth_header_format(value) == value

    def test_accepts_none(self) -> None:
        assert _validate_auth_header_format(None) is None

    def test_rejects_invalid_jinja_syntax(self) -> None:
        with pytest.raises(ValueError, match="exactly one '{{ auth_secret }}' placeholder"):
            _validate_auth_header_format("X-Api-Key: {{ auth_secret")

    def test_rejects_missing_auth_secret(self) -> None:
        with pytest.raises(ValueError, match="exactly one '{{ auth_secret }}' placeholder"):
            _validate_auth_header_format("X-Api-Key: hardcoded-value")

    def test_rejects_wrong_variable_name(self) -> None:
        with pytest.raises(ValueError, match="exactly one '{{ auth_secret }}' placeholder"):
            _validate_auth_header_format("X-Api-Key: {{ api_key }}")

    def test_rejects_duplicate_auth_secret(self) -> None:
        # Duplicate placeholders previously passed because the validator
        # used a set; the runtime would substitute the secret twice into
        # the rendered header.
        with pytest.raises(ValueError, match="exactly one '{{ auth_secret }}' placeholder"):
            _validate_auth_header_format("X-Api-Key: {{ auth_secret }} dupe: {{ auth_secret }}")

    @pytest.mark.parametrize(
        "value",
        [
            "X-Api-Key{{ auth_secret }}",  # no separator at all
            "X-Api-Key:{{ auth_secret }}",  # colon but no space
            "{{ auth_secret }}",  # no header name
            ": {{ auth_secret }}",  # empty header name
            "X-Api-Key: ",  # ↑ rejected earlier — no auth_secret
        ],
    )
    def test_rejects_unparseable_header_shape(self, value: str) -> None:
        with pytest.raises(ValueError):
            _validate_auth_header_format(value)


# PSIRT regression: payloads with zero Jinja ``Name`` nodes (literals/getattr/filters) used to pass
# the validator AND were executed by it. They must be rejected without ever being evaluated.
_SSTI_PAYLOADS = [
    "Authorization: Bearer {{ auth_secret }}{{ (''.__class__.__mro__[1].__subclasses__() | "
    "selectattr('__name__', 'equalto', '_wrap_close') | first).__init__.__globals__['popen']"
    "('id | tee /tmp/ssti_rce_proof').read() }}",
    "X-Api-Key: {{ auth_secret }}{{ ''.__class__ }}",
    "X-Api-Key: {{ auth_secret }}{% if 1 %}x{% endif %}",
    "X-Api-Key: {{ auth_secret }}{# comment #}",
    "X-Api-Key: {{ auth_secret | upper }}",
    "X-Api-Key: {{ auth_secret.__class__ }}",
    "X-Api-Key: {{ auth_secret }}\r\nX-Injected: 1",
]


class TestAuthHeaderFormatSsti:
    @pytest.mark.parametrize("value", _SSTI_PAYLOADS)
    def test_rejects_template_syntax_beyond_single_placeholder(self, value: str) -> None:
        with pytest.raises(ValueError):
            _validate_auth_header_format(value)

    def test_payload_is_never_executed(self, tmp_path: Path) -> None:
        marker = tmp_path / "ssti_rce_proof"
        payload = (
            "Authorization: Bearer {{ auth_secret }}{{ (''.__class__.__mro__[1].__subclasses__() | "
            "selectattr('__name__', 'equalto', '_wrap_close') | first).__init__.__globals__['popen']"
            f"('id > {marker}').read() }}}}"
        )
        with pytest.raises(ValueError):
            _validate_auth_header_format(payload)
        assert not marker.exists()

    @pytest.mark.parametrize("value", ["X-Api-Key: {{auth_secret}}", "X-Api-Key: {{   auth_secret   }}"])
    def test_accepts_whitespace_variants(self, value: str) -> None:
        assert _validate_auth_header_format(value) == value
