-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Credentials (BYOK): this is the secrets store. Every row holds a
-- user-supplied secret (model-provider API key or intake workspace token),
-- encrypted at rest in encrypted_payload; reads return metadata + fingerprint
-- only, never plaintext.
-- See docs/API.md sections "Credentials (BYOK)", "Tenancy", "Identifiers".

-- provider is the single category and states what the secret is for: the
-- model providers carry an API key; 'nhx' is NHX Intake (workspace token,
-- supplied as a `yaml` blob); 'openshift' is a user's OpenShift bearer token
-- (a `key`) that dispatch turns into a per-eval kubeconfig so the run acts as
-- that user. Callers already know this mapping, so there is no separate `type`
-- column.
CREATE TYPE credential_provider AS ENUM (
    'openai', 'anthropic', 'nvidia', 'nhx', 'openshift', 'switchyard'
);

-- Which write-once secret was supplied: a single-string `key` (model API
-- key) or a structured `yaml` blob (intake workspace token). Both are secret;
-- reported as metadata, plaintext is never returned.
CREATE TYPE credential_payload_kind AS ENUM ('key', 'yaml');

CREATE TABLE credentials (
    id                TEXT PRIMARY KEY,
    owner_id          TEXT REFERENCES users(id),
    name              TEXT NOT NULL,
    provider          credential_provider NOT NULL,
    payload_kind      credential_payload_kind NOT NULL,
    -- Fernet ciphertext (envelope encryption lands later, see API.md:239).
    -- Decrypted only at dispatch into ephemeral runner/Switchyard Secrets.
    encrypted_payload BYTEA NOT NULL,
    -- Non-reversible digest of the plaintext so callers can tell which secret
    -- is loaded without revealing it (GET returns this, never plaintext).
    fingerprint       TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at        TIMESTAMPTZ
);

CREATE INDEX credentials_created_at_idx
    ON credentials (created_at DESC);

-- List filter is provider over live rows (API.md:53).
CREATE INDEX credentials_provider_live_idx
    ON credentials (provider)
    WHERE deleted_at IS NULL;
