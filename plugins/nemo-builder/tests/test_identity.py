# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry hosts, tags, and the system tag."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.identity import (
    ImageIdentityError,
    compose_system_tag,
    split_system_tag,
    validate_caller_tag,
    validate_registry_host,
)


class TestSystemTag:
    def test_composes_and_splits(self) -> None:
        tag = compose_system_tag("scaled-evals", "freight-dispatch-shift", 1, 2)
        assert tag == "scaled-evals--freight-dispatch-shift-1-2"
        assert split_system_tag(tag) == ("scaled-evals", "freight-dispatch-shift", 1, 2)

    def test_each_image_in_a_set_gets_its_own_tag(self) -> None:
        """Otherwise two specs publishing to one repository would replace each other's system tag."""
        tags = {compose_system_tag("ws", "app", 1, index) for index in range(10)}
        assert len(tags) == 10

    def test_the_double_dash_is_what_makes_the_split_unambiguous(self) -> None:
        """Under a single `-`, these two pairs would compose to the same tag."""
        a = compose_system_tag("scaled-evals", "freight", 1, 0)
        b = compose_system_tag("scaled", "evals-freight", 1, 0)
        assert a != b
        assert split_system_tag(a) == ("scaled-evals", "freight", 1, 0)
        assert split_system_tag(b) == ("scaled", "evals-freight", 1, 0)

    def test_a_set_name_ending_in_digits_still_splits(self) -> None:
        """Revision and index are the last two numeric fields, so a set name like `app-2` is safe."""
        assert split_system_tag(compose_system_tag("ws", "app-2", 3, 4)) == ("ws", "app-2", 3, 4)

    def test_rejects_names_that_are_legal_entities_but_illegal_tags(self) -> None:
        """NAME_PATTERN admits `@` and `+`; a tag does not."""
        with pytest.raises(ImageIdentityError):
            compose_system_tag("ws", "build+set", 1, 0)
        with pytest.raises(ImageIdentityError):
            compose_system_tag("ws", "build@set", 1, 0)

    def test_rejects_a_name_already_containing_the_separator(self) -> None:
        with pytest.raises(ImageIdentityError):
            compose_system_tag("we--ird", "set", 1, 0)

    def test_revision_is_one_based_and_index_zero_based(self) -> None:
        with pytest.raises(ImageIdentityError):
            compose_system_tag("ws", "set", 0, 0)
        with pytest.raises(ImageIdentityError):
            compose_system_tag("ws", "set", 1, -1)


class TestCallerTags:
    """A caller tag shares a repository with tags the control plane trusts."""

    def test_an_ordinary_tag_is_allowed(self) -> None:
        assert validate_caller_tag("v1.2-rc_3") == "v1.2-rc_3"

    @pytest.mark.parametrize(
        "reserved", [compose_system_tag("default", "demo", 2, 0), "a--b", "sha256-" + "a" * 64 + ".sig"]
    )
    def test_system_and_signature_shapes_are_reserved(self, reserved: str) -> None:
        with pytest.raises(ImageIdentityError, match="reserved"):
            validate_caller_tag(reserved)

    @pytest.mark.parametrize("bad", ["café", "v１", "日本"])
    def test_a_tag_is_ascii(self, bad: str) -> None:
        """Python's word class is Unicode; the OCI grammar is ASCII."""
        with pytest.raises(ImageIdentityError):
            validate_caller_tag(bad)


class TestRegistryHost:
    @pytest.mark.parametrize("good", ["registry.example.com", "localhost:5000", "Registry.Example.com"])
    def test_a_host_is_lowercased(self, good: str) -> None:
        assert validate_registry_host(good) == good.lower()

    @pytest.mark.parametrize("bad", ["", "https:", "user@host", "host name", "reg?x=1", "host:port"])
    def test_anything_else_is_refused(self, bad: str) -> None:
        with pytest.raises(ImageIdentityError):
            validate_registry_host(bad)
