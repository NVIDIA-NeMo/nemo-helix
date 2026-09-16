# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

try:
    from scaled_evals.api.redaction import redact_json_text, redact_secret_text
except ImportError as exc:
    pytest.skip(f"scaled-evals plugin not installed: {exc}", allow_module_level=True)


def test_redact_secret_text_masks_assignments_and_known_token_shapes() -> None:
    text = "api_key=sk-secret-value bearer nvapi-secret.value password: hunter2"

    redacted = redact_secret_text(text)

    assert redacted == "api_key=<redacted> bearer <redacted> password: <redacted>"


def test_redact_secret_text_preserves_environment_references() -> None:
    assert redact_secret_text("api_key=$OPENAI_API_KEY") == "api_key=$OPENAI_API_KEY"
    assert redact_secret_text("api_key=${OPENAI_API_KEY}") == "api_key=${OPENAI_API_KEY}"
    assert redact_secret_text("api_key=$uperSecret") == "api_key=<redacted>"


def test_redact_secret_text_masks_openshift_jwt_and_database_credentials() -> None:
    database_url = "".join(("postgresql://", "scaled", ":", "super-secret", "@", "db.example", "/", "scaled"))
    text = (
        "SANDBOX_OC_TOKEN=sha256~abcdefghijklmnop "
        "jwt eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signaturevalue "
        f"{database_url}"
    )

    redacted = redact_secret_text(text)

    assert "sha256~" not in redacted
    assert "eyJ" not in redacted
    assert "super-secret" not in redacted
    assert redacted.count("<redacted>") == 3


def test_json_redaction_preserves_escaped_observation_structure() -> None:
    observation = json.dumps({"output": "password=synthetic-value", "metadata": {"exit_code": 0}})
    original = json.dumps({"steps": [{"observation": observation}], "reward": 1.0}, indent=2)
    redacted = redact_json_text(original)
    assert "synthetic-value" not in redacted
    parsed = json.loads(redacted)
    assert parsed["reward"] == 1.0
    assert json.loads(parsed["steps"][0]["observation"]) == {
        "output": "password=<redacted>",
        "metadata": {"exit_code": 0},
    }


def test_json_redaction_covers_secret_fields_and_preserves_nonsecret_values() -> None:
    source = {
        "nested": [{"API_KEY": "synthetic-value", "password": "another-value"}],
        "reference": {"api_key": "$OPENAI_API_KEY"},
        "literal": {"api_key": "$uperSecret"},
        "credential_id": "cred_example",
        "enabled": True,
        "count": 123456789012345678901234567890,
        "message": 'literal \\"quote\\" and unicode café',
    }
    result = json.loads(redact_json_text(json.dumps(source)))
    assert result == {
        **source,
        "nested": [{"API_KEY": "<redacted>", "password": "<redacted>"}],
        "literal": {"api_key": "<redacted>"},
    }


def test_json_redaction_preserves_unchanged_bytes_and_jsonl_records() -> None:
    unchanged = '{\n  "value": 42\n}\n'
    assert redact_json_text(unchanged) == unchanged
    lines = '\n{"value":"sk-synthetic-token"}\r\n{"value":42}\n'
    result = redact_json_text(lines, lines=True)
    assert result.startswith("\n")
    assert "\r\n" in result
    assert [json.loads(line) for line in result.splitlines() if line] == [
        {"value": "<redacted>"},
        {"value": 42},
    ]
    assert redact_json_text("broken api_key=sk-synthetic-token") == "broken api_key=<redacted>"


@pytest.mark.parametrize(
    "value",
    ["synthetic-value", "value with spaces, punctuation !@#$%^&*()", 'escaped "quote" and \\ slash'],
)
def test_json_redaction_masks_quoted_secret_fields_in_malformed_json(value: str) -> None:
    malformed = json.dumps({"password": value})[:-1]

    assert redact_json_text(malformed) == json.dumps({"password": "<redacted>"})[:-1]


def test_json_redaction_masks_quoted_secret_fields_in_malformed_jsonl() -> None:
    malformed = '{"password":"value with spaces"\n{"api_key":"unterminated value\n'

    assert redact_json_text(malformed, lines=True) == '{"password":"<redacted>"\n{"api_key":"<redacted>\n'


def test_json_redaction_preserves_colliding_keys_and_values() -> None:
    source = {
        "sk-synthetic-first": {"password": "first-secret", "index": 1},
        "<redacted>": {"index": 2},
        "sk-synthetic-second": {"index": 3},
        "<redacted>#2": {"index": 4},
    }
    redacted = redact_json_text(json.dumps(source))
    result = json.loads(redacted)
    assert len(result) == len(source)
    assert sorted(item["index"] for item in result.values()) == [1, 2, 3, 4]
    assert result["<redacted>"] == {"index": 2}
    assert result["<redacted>#2"] == {"index": 4}
    assert "sk-synthetic" not in redacted
    assert "first-secret" not in redacted
    assert redact_json_text(redacted) == redacted


def test_changed_json_keeps_multiline_presentation() -> None:
    source = json.dumps({"password": "synthetic-value", "nested": {"count": 2}}, indent=4)
    redacted = redact_json_text(source)
    assert redacted.count("\n") > 1
    assert json.loads(redacted) == {"password": "<redacted>", "nested": {"count": 2}}


def test_deep_json_redacts_within_capacity() -> None:
    source = "[" * 600 + '{"password": "synthetic-value"}' + "]" * 600
    redacted = redact_json_text(source)
    assert "synthetic-value" not in redacted
    value = json.loads(redacted)
    for _ in range(600):
        value = value[0]
    assert value == {"password": "<redacted>"}


def test_deep_json_does_not_fall_back_to_unredacted_structured_fields() -> None:
    source = "[" * 2000 + '{"password": "synthetic-value"}' + "]" * 2000
    with pytest.raises(ValueError, match="nesting exceeds redaction capacity"):
        redact_json_text(source)
