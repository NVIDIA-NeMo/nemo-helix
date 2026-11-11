#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the RL package-copies table from a scan_rl_image_packages.py JSON file.

python package_dump_to_xslx.py --json PATH --output PATH
"""

import argparse
import json
import re
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

# CUDA builds of one library are published as separate names (nixl-cu12, nixl-cu13).
# Group those as one library and keep the build in the Version cell.
# -base and -core are left alone: hydra-core, megatron-core, and poetry-core are
# different libraries, not CUDA builds.
CU_VARIANT = re.compile(r"-(cu\d+)$")
VARIANT_SUFFIX = re.compile(r"-(cu\d+|base|core)$")


def library_name(package):
    return CU_VARIANT.sub("", package or "")


def gigabytes(size_bytes):
    if size_bytes is None:
        return None
    return size_bytes / 1_000_000_000


def display_version(package, version):
    if not version:
        return version
    match = VARIANT_SUFFIX.search(package or "")
    if match:
        return f"{version} {match.group(1)}"
    return version


# Excel refuses a row taller than 409 points. Each line of Where takes LINE_HEIGHT points.
EXCEL_MAX_ROW_HEIGHT = 409
LINE_HEIGHT = 14


def where_column_width(rows):
    longest = 0
    for row in rows:
        for line in row["where"].split("\n"):
            longest = max(longest, len(line))
    # A line that fills the column wraps, which hides the bottom of Where.
    return min(255, longest + 15)


def where_line_count(text):
    if not text:
        return 1
    return str(text).count("\n") + 1


HEADERS = [
    "Package",
    "Version",
    "Copies in Image",
    "Package Size (GB)",
    "Package Total Size (GB)",
    "Total Copies",
    "Times Present",
    "Where",
]


def location_label(path):
    marker = "/opt/uv_cache/archive-v0/"
    if path.startswith(marker):
        return "unreferenced cache " + path.rsplit("/", 1)[-1]
    return path


def version_row(package, version, count_key):
    """One spreadsheet row for a package version.

    Symlink installs of one version share a single uv-cache copy. Each private
    copy and each unreferenced cache archive counts as its own copy. Mixed
    installs add their private file bytes on top of that shared copy.
    """
    locations = version["locations"]
    size = version.get("size_bytes")
    symlinkish = private = cache_only = unknown = extra_private = 0
    where = []
    for location in locations:
        storage = location.get("storage")
        payload = location.get("payload_files") or {}
        where.append(location_label(location["path"]))
        if storage == "private-copy":
            private += 1
        elif storage == "uv-cache-only":
            cache_only += 1
        elif storage in ("uv-cache-symlink", "mixed"):
            symlinkish += 1
            if storage == "mixed":
                extra_private += payload.get("regular_bytes") or 0
        else:
            unknown += 1
    copies = (1 if symlinkish else 0) + private + cache_only
    if unknown and copies == 0:
        copies = 1
    total = None
    if size is not None:
        total = private * size + cache_only * size
        if symlinkish:
            total += size + extra_private
        if unknown and not symlinkish and not private and not cache_only:
            total += size
    return {
        "package": package["name"],
        "version": version["version"],
        "copies": copies,
        "size": size,
        "total": total,
        "times": version.get(count_key, len(locations)),
        "where": "\n".join(where),
        "unreferenced": bool(cache_only and not symlinkish and not private),
    }


def format_gb(size_bytes):
    if size_bytes is None:
        return ""
    return f"{gigabytes(size_bytes):.3f}"


def group_by_package(rows):
    """One row per package. Version lines stay aligned with that version's copy count.

    Copies in Image is the on-disk copy count of each version, not how many venvs
    link to it. Different versions of one package are separate copies.
    """
    grouped = {}
    order = []
    for row in rows:
        name = library_name(row["package"])
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append(row)
    result = []
    for name in order:
        versions = grouped[name]
        labels = [display_version(version["package"], version["version"]) for version in versions]
        result.append(
            {
                "package": name,
                "version": "\n".join(labels),
                "copies": "\n".join(str(version["copies"]) for version in versions),
                "size": "\n".join(format_gb(version["size"]) for version in versions),
                "total": gigabytes(sum(version["total"] or 0 for version in versions)),
                "total_copies": sum(version["copies"] for version in versions),
                "times": "\n".join(str(version["times"]) for version in versions),
                "where": "\n\n".join(f"{label}\n{version['where']}" for label, version in zip(labels, versions)),
                "unreferenced": all(version["unreferenced"] for version in versions),
                "sort_total": sum(version["total"] or 0 for version in versions),
            }
        )

    def sort_key(row):
        total = row["sort_total"]
        name = row["package"]
        total_bytes = total if isinstance(total, int) and not isinstance(total, bool) else 0
        package = name.lower() if isinstance(name, str) else ""
        return (-total_bytes, package)

    result.sort(key=sort_key)
    return result


def rows_from_inventory(inventory):
    rows = []
    for package in inventory.get("packages") or []:
        for version in package["versions"]:
            rows.append(version_row(package, version, "download_count"))
    for package in inventory.get("editable_packages") or []:
        for version in package["versions"]:
            rows.append(version_row(package, version, "install_count"))
    rows.sort(key=lambda row: (-(row["total"] or 0), row["package"].lower(), str(row["version"])))
    return group_by_package(rows)


def image_notes(inventory):
    image = inventory.get("image") or {}
    repository = image.get("repository")
    tag = image.get("tag")
    reference = f"{repository}:{tag}" if repository and tag else repository or ""
    return reference, image.get("id") or "", inventory.get("scanned_at") or ""


def build_workbook(rows, image, image_id, scanned_at):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Package copies"
    sheet.append(HEADERS)
    for row in rows:
        sheet.append(
            [
                row["package"],
                row["version"],
                row["copies"],
                row["size"],
                row["total"],
                row["total_copies"],
                row["times"],
                row["where"],
            ]
        )

    header_font = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="E8E8E8")
    for cell in sheet[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)

    longest_package = max((len(row["package"]) for row in rows), default=20)
    widths = {
        "A": max(42, longest_package + 2),
        "B": 24,
        "C": 18,
        "D": 24,
        "E": 28,
        "F": 16,
        "G": 16,
        "H": where_column_width(rows),
    }
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width

    for row in sheet.iter_rows(min_row=2):
        for cell in row[:6]:
            cell.alignment = Alignment(vertical="top")
        wrapped = Alignment(vertical="top", wrap_text=True)
        wrapped_right = Alignment(vertical="top", horizontal="right", wrap_text=True)
        row[0].alignment = wrapped
        row[1].alignment = wrapped
        row[2].alignment = wrapped_right
        row[3].alignment = wrapped_right
        row[4].alignment = Alignment(vertical="top", horizontal="right")
        row[4].number_format = "0.000"
        row[5].alignment = Alignment(vertical="top", horizontal="right")
        row[6].alignment = wrapped_right
        row[7].alignment = wrapped
        # Height follows Where. The column is wide enough that each path is one line.
        line_count = where_line_count(row[7].value)
        sheet.row_dimensions[row[0].row].height = min(EXCEL_MAX_ROW_HEIGHT, line_count * LINE_HEIGHT)

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:H{sheet.max_row}"

    notes = workbook.create_sheet("Notes")
    notes.append(["Image", image or ""])
    notes.append(["Image ID", image_id or ""])
    notes.append(["Scanned", scanned_at or ""])
    notes.append(["Package versions", len(rows)])
    notes.append(["Unreferenced cache versions", sum(1 for row in rows if row["unreferenced"])])
    notes.append(
        [
            "Copies in Image",
            "On-disk copies of each version, one line per version in the Version column. "
            "Symlink installs of one version share a single /opt/uv_cache copy. "
            "A different version is another copy. This is not the number of venvs.",
        ]
    )
    notes.append(
        ["Package Size (GB)", "Size of one copy, from the distribution RECORD, in decimal gigabytes (bytes / 1e9)."]
    )
    notes.append(
        [
            "Package Total Size (GB)",
            "Sum of each version's on-disk total: copies of that version times its size, plus private files inside mixed installs. Decimal gigabytes (bytes / 1e9).",
        ]
    )
    notes.append(["Total Copies", "Sum of Copies in Image across the versions of this package."])
    notes.append(["Times Present", "Number of venvs or cache archives that hold each version, one line per version."])
    notes.append(
        [
            "Version",
            "nixl-cu12 and nixl-cu13 are one library. The cu12 or cu13 build is on the version line. "
            "The same applies to other -cuN package names. -base and -core stay separate packages.",
        ]
    )
    notes.append(
        [
            "Where",
            "Each version is labeled, then its install paths. Cache archives that no venv links to are labeled 'unreferenced cache'.",
        ]
    )
    for cell in notes["A"]:
        cell.font = header_font
    notes.column_dimensions["A"].width = 32
    notes.column_dimensions["B"].width = 110
    for row in notes.iter_rows(min_col=2, max_col=2):
        row[0].alignment = Alignment(wrap_text=True, vertical="top")
    # Body text is 10 pt. Row heights are set separately from LINE_HEIGHT and do not change.
    for sheet_in_book in workbook.worksheets:
        for cell_row in sheet_in_book.iter_rows():
            for cell in cell_row:
                cell.font = Font(bold=cell.font.bold, size=10)
    return workbook


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Write the package inventory JSON to an xlsx workbook.")
    parser.add_argument("--json", type=Path, required=True, help="JSON written by scan_rl_image_packages.py.")
    parser.add_argument("--output", type=Path, required=True, help="xlsx file to write.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    inventory = json.loads(args.json.read_text())
    rows = rows_from_inventory(inventory)
    image, image_id, scanned_at = image_notes(inventory)
    workbook = build_workbook(rows, image, image_id, scanned_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.output)
    check = load_workbook(args.output)
    written = check["Package copies"].max_row - 1
    if written != len(rows):
        raise SystemExit(f"wrote {written} rows, expected {len(rows)}")
    print(f"packages {len(rows)} wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
