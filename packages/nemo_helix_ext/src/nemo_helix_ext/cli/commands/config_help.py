# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared help text for the config command group."""

CONFIG_APP_HELP = """\
Manage NeMo Helix CLI configuration.

Examples:
# Connect the current context to a remote deployment.
nemo config set --base-url https://nhx.example.com
# Keep contexts separate.
nemo config set --context staging --base-url https://nhx.staging.example.com
nemo config set --context production --base-url https://nhx.example.com --activate
# Configure a local TLS endpoint.
nemo config set --context local-tls --base-url https://localhost:8443 --certificate-authority /path/to/ca.crt
# Switch the current context and inspect its configuration.
nemo config use-context staging
nemo config view
# Delete a context without deleting its cluster or user records.
nemo config delete-context staging

NHX_BASE_URL and NHX_CURRENT_CONTEXT override saved configuration."""
