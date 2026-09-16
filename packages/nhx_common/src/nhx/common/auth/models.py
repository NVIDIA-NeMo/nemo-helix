# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth models for NeMo Helix services."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Self
from urllib.parse import quote

from pydantic import BaseModel, Field, ValidationError

from .exceptions import InvalidPrincipalHeader

logger = logging.getLogger(__name__)

NHX_PRINCIPAL_ENVVAR = "NHX_PRINCIPAL"

MAX_PRINCIPAL_ID_LENGTH = 256
MAX_EMAIL_LENGTH = 320
MAX_GROUP_LENGTH = 256
MAX_GROUPS_COUNT = 100

_PRINCIPAL_ID_RE = re.compile(r"^[a-zA-Z0-9@._\-:+/]+$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+$")


def _validate_principal_id(value: str, header_name: str) -> None:
    """Validate a principal identifier value (used for principal-id and on-behalf-of)."""
    if len(value) > MAX_PRINCIPAL_ID_LENGTH:
        raise InvalidPrincipalHeader(f"{header_name} exceeds maximum length of {MAX_PRINCIPAL_ID_LENGTH} characters")
    if not _PRINCIPAL_ID_RE.match(value):
        raise InvalidPrincipalHeader(
            f"{header_name} contains invalid characters; allowed: alphanumeric, @, ., -, _, :, +, /"
        )


def _principal_email_from_headers(headers: Dict[str, str], header_key: str, header_label: str) -> Optional[str]:
    """Parse and validate a principal email header; returns None if absent or empty after strip."""
    raw = headers.get(header_key)
    if raw is None:
        return None
    email = raw.strip()
    if not email:
        return None
    if len(email) > MAX_EMAIL_LENGTH:
        raise InvalidPrincipalHeader(f"{header_label} exceeds maximum length of {MAX_EMAIL_LENGTH} characters")
    if not _EMAIL_RE.match(email):
        raise InvalidPrincipalHeader(f"{header_label} is not a valid email address")
    return email


def _principal_groups_from_headers(headers: Dict[str, str], header_key: str, header_label: str) -> list[str]:
    """Parse and validate a comma-separated principal groups header."""
    groups_header = headers.get(header_key, "")
    if not groups_header:
        return []
    groups = [g.strip() for g in groups_header.split(",") if g.strip()]
    if len(groups) > MAX_GROUPS_COUNT:
        raise InvalidPrincipalHeader(f"{header_label} exceeds maximum of {MAX_GROUPS_COUNT} groups")
    for group in groups:
        if len(group) > MAX_GROUP_LENGTH:
            raise InvalidPrincipalHeader(f"{header_label} contains a group exceeding {MAX_GROUP_LENGTH} characters")
        if not _PRINCIPAL_ID_RE.match(group):
            raise InvalidPrincipalHeader(
                f"{header_label} contains a group with invalid characters; allowed: alphanumeric, @, ., -, _, :, +, /"
            )
    return groups


class Principal(BaseModel):
    """Represents an authenticated principal (user, service account, or group).

    This model encapsulates the identity and authorization context for a request.
    """

    id: str = Field(default="", description="The principal's unique identifier")
    email: Optional[str] = Field(None, description="The principal's email address")
    groups: List[str] = Field(default_factory=list, description="Groups the principal belongs to")
    on_behalf_of: Optional[str] = Field(
        None,
        description="If acting on behalf of another principal, their principal ID",
    )
    on_behalf_of_groups: Optional[List[str]] = Field(
        default=None,
        description="Groups the on-behalf-of principal belongs to",
    )
    on_behalf_of_email: Optional[str] = Field(
        None,
        description="The on-behalf-of principal's email address",
    )

    @property
    def effective_id(self) -> str:
        """The acting user's ID: on_behalf_of if delegated, otherwise the principal's own ID."""
        return self.on_behalf_of or self.id

    @property
    def effective_groups(self) -> List[str]:
        """The acting user's groups: on_behalf_of groups if delegated, otherwise the principal's own groups."""
        if self.on_behalf_of:
            return list(self.on_behalf_of_groups or [])
        return self.groups

    @property
    def effective_email(self) -> Optional[str]:
        """The acting user's email: on_behalf_of email if delegated, otherwise the principal's own email."""
        if self.on_behalf_of:
            return self.on_behalf_of_email
        return self.email

    @property
    def effective_principal(self) -> Principal:
        """The acting user's principal: on_behalf_of principal if delegated, otherwise the principal's own principal."""
        return Principal(
            id=self.effective_id,
            groups=self.effective_groups,
            email=self.effective_email,
        )

    @property
    def is_delegated(self) -> bool:
        """Whether the principal is delegated."""
        return self.on_behalf_of is not None

    @property
    def is_privileged(self) -> bool:
        """Whether the principal is privileged."""
        return self.id.startswith("service:")

    def is_service_identity(self) -> bool:
        """Whether the principal is an internal service or service account."""
        return self.is_privileged or self.id.startswith("service-account:")

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional[Principal]:
        """Create a Principal from request headers.

        Args:
            headers: Dictionary of HTTP headers

        Returns:
            Principal instance if principal ID is found, None otherwise

        Raises:
            InvalidPrincipalHeader: If any header value fails validation
        """
        principal_id = headers.get("x-nhx-principal-id", "").strip()
        if not principal_id:
            return None

        _validate_principal_id(principal_id, "X-NHX-Principal-Id")

        email = _principal_email_from_headers(headers, "x-nhx-principal-email", "X-NHX-Principal-Email")
        groups = _principal_groups_from_headers(headers, "x-nhx-principal-groups", "X-NHX-Principal-Groups")

        on_behalf_of_groups: Optional[List[str]] = None
        on_behalf_of_email: Optional[str] = None

        on_behalf_of = headers.get("x-nhx-principal-on-behalf-of")
        if on_behalf_of is not None:
            on_behalf_of = on_behalf_of.strip()
            if on_behalf_of:
                _validate_principal_id(on_behalf_of, "X-NHX-Principal-On-Behalf-Of")
            else:
                on_behalf_of = None

            on_behalf_of_groups = _principal_groups_from_headers(
                headers,
                "x-nhx-principal-on-behalf-of-groups",
                "X-NHX-Principal-On-Behalf-Of-Groups",
            )
            on_behalf_of_email = _principal_email_from_headers(
                headers,
                "x-nhx-principal-on-behalf-of-email",
                "X-NHX-Principal-On-Behalf-Of-Email",
            )

        return cls(
            id=principal_id,
            email=email,
            groups=groups,
            on_behalf_of=on_behalf_of,
            on_behalf_of_groups=on_behalf_of_groups,
            on_behalf_of_email=on_behalf_of_email,
        )

    def get_headers(self) -> Dict[str, str]:
        """Get headers for service-to-service communication.

        Returns:
            Dictionary of headers to forward the principal context
        """
        headers = {"X-NHX-Principal-Id": self.id}

        if self.email:
            headers["X-NHX-Principal-Email"] = self.email

        if self.groups:
            headers["X-NHX-Principal-Groups"] = ",".join(self.groups)

        if self.on_behalf_of:
            headers["X-NHX-Principal-On-Behalf-Of"] = self.on_behalf_of

        if self.on_behalf_of_groups:
            headers["X-NHX-Principal-On-Behalf-Of-Groups"] = ",".join(self.on_behalf_of_groups)

        if self.on_behalf_of_email:
            headers["X-NHX-Principal-On-Behalf-Of-Email"] = self.on_behalf_of_email

        return headers

    def to_on_behalf_of(self) -> Principal:
        """Create a delegated principal acting on behalf of this principal.

        Returns:
            A new Principal instance representing the delegate
        """
        if not self.on_behalf_of:
            raise ValueError("Cannot create on-behalf-of principal when 'on_behalf_of' is not set")

        return Principal(
            id=self.on_behalf_of,
            groups=list(self.on_behalf_of_groups or []),
            email=self.on_behalf_of_email,
        )

    def get_env_var(self, env_var_name: str = NHX_PRINCIPAL_ENVVAR) -> Dict[str, str]:
        """Get environment variables for passing the principal when launching a task.

        Returns:
            Dictionary of environment variables to forward the principal context
        """
        return {env_var_name: self.model_dump_json(exclude_none=True)}

    def get_otlp_headers_value(self) -> str:
        """Get OTLP headers value for the principal.

        Returns headers in the OTLP format: key=value,key2=value2
        Values are URL-encoded to handle special characters like commas in groups.

        Returns:
            String formatted for OTEL_EXPORTER_OTLP_HEADERS env var
        """
        parts = [f"X-NHX-Principal-Id={quote(self.id, safe='')}"]

        if self.email:
            parts.append(f"X-NHX-Principal-Email={quote(self.email, safe='')}")

        if self.groups:
            # Groups are comma-separated, so URL-encode the whole value
            groups_value = ",".join(self.groups)
            parts.append(f"X-NHX-Principal-Groups={quote(groups_value, safe='')}")

        if self.on_behalf_of:
            parts.append(f"X-NHX-Principal-On-Behalf-Of={quote(self.on_behalf_of, safe='')}")

        if self.on_behalf_of_groups:
            groups_value = ",".join(self.on_behalf_of_groups)
            parts.append(f"X-NHX-Principal-On-Behalf-Of-Groups={quote(groups_value, safe='')}")

        if self.on_behalf_of_email:
            parts.append(f"X-NHX-Principal-On-Behalf-Of-Email={quote(self.on_behalf_of_email, safe='')}")

        return ",".join(parts)

    @classmethod
    def from_env_var(cls, env_var_name: str = NHX_PRINCIPAL_ENVVAR) -> Optional[Principal]:
        """Create a Principal from an environment variable containing JSON.

        Args:
            env_var_name: Name of the environment variable to read

        Returns:
            Principal instance if env var is set and valid, None otherwise

        Raises:
            ValueError: If the env var contains malformed JSON
        """
        principal_json = os.environ.get(env_var_name)
        if not principal_json:
            return None

        # Validate JSON syntax first
        try:
            json.loads(principal_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {env_var_name}: {e}") from e

        # Parse as Principal - returns None if validation fails (e.g., missing required fields)
        try:
            principal = cls.model_validate_json(principal_json)
            # Return None if id is empty (required for a valid principal)
            if not principal.id:
                return None
            return principal
        except ValidationError:
            return None


class AuthContext(BaseModel):
    """Auth context captured at resource creation for delegated access.

    Stores a snapshot of the creating principal's identity so that controllers
    can later act on their behalf (e.g., accessing secrets).
    """

    principal_id: str = Field(..., description="The principal's unique identifier")
    principal_email: Optional[str] = Field(default=None, description="The principal's email address")
    principal_groups: List[str] = Field(default_factory=list, description="Groups the principal belongs to")
    principal_on_behalf_of: Optional[str] = Field(
        default=None, description="If acting on behalf of another principal, their principal ID"
    )
    principal_on_behalf_of_groups: Optional[List[str]] = Field(
        default=None, description="Groups the on-behalf-of principal belongs to"
    )
    principal_on_behalf_of_email: Optional[str] = Field(
        default=None, description="The on-behalf-of principal's email address"
    )

    @classmethod
    def from_principal(cls, principal: Principal) -> Self:
        """Create from a runtime Principal."""
        return cls(
            principal_id=principal.id,
            principal_email=principal.email,
            principal_groups=principal.groups,
            principal_on_behalf_of=principal.on_behalf_of,
            principal_on_behalf_of_groups=principal.on_behalf_of_groups,
            principal_on_behalf_of_email=principal.on_behalf_of_email,
        )

    def to_principal(self) -> Principal:
        """Convert back to a Principal for SDK calls."""
        return Principal(
            id=self.principal_id,
            email=self.principal_email,
            groups=self.principal_groups,
            on_behalf_of=self.principal_on_behalf_of,
            on_behalf_of_groups=self.principal_on_behalf_of_groups,
            on_behalf_of_email=self.principal_on_behalf_of_email,
        )
