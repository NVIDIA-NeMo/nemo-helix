# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def merge_version_specifiers(spec1: str, spec2: str) -> str:
    """Merge two specifier strings, keeping highest lower bound and lowest upper bound."""
    combined = SpecifierSet(spec1) & SpecifierSet(spec2)

    # Parse all specifiers into categories
    lower_bounds: list[tuple[str, Version]] = []  # (operator, version) for >=, >
    upper_bounds: list[tuple[str, Version]] = []  # (operator, version) for <, <=
    exact: list[Version] = []  # for ==
    other: list[str] = []  # for !=, ~=, ===

    for spec in combined:
        op = spec.operator
        ver = Version(spec.version)
        if op in (">=", ">"):
            lower_bounds.append((op, ver))
        elif op in ("<", "<="):
            upper_bounds.append((op, ver))
        elif op == "==":
            exact.append(ver)
        else:
            other.append(str(spec))

    result_parts = []

    # Keep highest lower bound
    if lower_bounds:
        # Sort by version, prefer > over => for same version
        lower_bounds.sort(key=lambda x: (x[1], x[0] == ">"), reverse=True)
        op, ver = lower_bounds[0]
        result_parts.append(f"{op}{ver}")

    # Keep lowest upper bound
    if upper_bounds:
        # Sort by version, prefer < over <= for same version
        upper_bounds.sort(key=lambda x: (x[1], x[0] == "<="))
        op, ver = upper_bounds[0]
        result_parts.append(f"{op}{ver}")

    # Handle exact versions (if any)
    if exact:
        # Just keep one (they should all be the same if valid)
        result_parts.append(f"=={exact[0]}")

    # Keep other specifiers as-is
    result_parts.extend(other)

    return ",".join(result_parts)


def merge_dependencies(existing_deps: list[str], new_deps: list[str]) -> list[str]:
    """Merge two dependency lists, keeping the highest version for each package.

    Preserves original string format when possible; only re-serializes when merging conflicts.
    """
    deps_map: dict[str, tuple[str, Requirement]] = {}
    for dep_str in existing_deps + new_deps:
        req_to_add = Requirement(dep_str)
        normalized_name = req_to_add.name.lower()

        if normalized_name not in deps_map:
            # first time we see the dep - just add
            deps_map[normalized_name] = (dep_str, req_to_add)
        else:
            _, existing_req = deps_map[normalized_name]

            # If specs are the same, keep original
            if req_to_add == existing_req:
                continue

            # Merge and simplify specifiers
            merged_specifier = merge_version_specifiers(str(existing_req.specifier), str(req_to_add.specifier))
            merged_extras = existing_req.extras | req_to_add.extras
            extras_str = f"[{','.join(sorted(merged_extras))}]" if merged_extras else ""
            new_req_str = f"{req_to_add.name}{extras_str}{merged_specifier}"
            deps_map[normalized_name] = (new_req_str, Requirement(new_req_str))

    return [dep_str for dep_str, _ in deps_map.values()]
