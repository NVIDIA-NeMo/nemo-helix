#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

usage() {
    echo "usage: $0 OUTPUT_DIR [--installed] [--changed-from DPKG_MANIFEST] [PACKAGE ...]" >&2
}

source_collection_enabled() {
    case "${NHX_COLLECT_SOURCES:-0}" in
        1 | true | TRUE | True | yes | YES | Yes | on | ON | On)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if [ "$#" -lt 1 ]; then
    usage
    exit 2
fi

output_dir="$1"
shift

include_installed=0
changed_from_file=""
while [ "$#" -gt 0 ]; do
    case "${1}" in
        --installed)
            include_installed=1
            shift
            ;;
        --changed-from)
            if [ "$#" -lt 2 ]; then
                usage
                exit 2
            fi
            changed_from_file="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        --*)
            usage
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

if ! source_collection_enabled; then
    mkdir -p "${output_dir}/manifests"
    echo "source collection disabled; set NHX_COLLECT_SOURCES=1 to enable" > \
        "${output_dir}/manifests/source-collection-disabled.txt"
    exit 0
fi

mkdir -p "${output_dir}/sources" "${output_dir}/manifests"
missing_file="${output_dir}/manifests/missing-apt-sources.txt"
downloaded_file="${output_dir}/manifests/downloaded-apt-sources.txt"
log_file="${output_dir}/manifests/apt-source.log"
: > "${missing_file}"
: > "${downloaded_file}"
: > "${log_file}"

if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get is not available" >> "${missing_file}"
    exit 0
fi

packages_file="$(mktemp)"
source_packages_file="$(mktemp)"
trap 'rm -f "${packages_file}" "${source_packages_file}"' EXIT
: > "${packages_file}"
: > "${source_packages_file}"

if command -v dpkg-query >/dev/null 2>&1; then
    # shellcheck disable=SC2016
    dpkg_query_format='${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n'
    current_packages_file="${output_dir}/manifests/installed-dpkg-packages.txt"
    dpkg-query -W -f="${dpkg_query_format}" > "${current_packages_file}" 2>/dev/null || true
    if [ "${include_installed}" -eq 1 ]; then
        awk -F '\t' '
            length($1) > 0 {
                source_package = $3
                if (source_package == "") {
                    source_package = $1
                    sub(/:[^:]+$/, "", source_package)
                }
                source_version = $4
                if (source_version == "") {
                    source_version = $2
                }
                if (source_version != "") {
                    printf "%s=%s\n", source_package, source_version
                } else {
                    print source_package
                }
            }
        ' "${current_packages_file}" >> "${source_packages_file}" 2>/dev/null || true
    fi
    if [ -n "${changed_from_file}" ]; then
        if [ ! -f "${changed_from_file}" ]; then
            echo "baseline dpkg manifest not found: ${changed_from_file}" >> "${missing_file}"
            exit 2
        fi
        cp "${changed_from_file}" "${output_dir}/manifests/baseline-dpkg-packages.txt"
        awk -F '\t' '
            function print_source_package() {
                source_package = $3
                if (source_package == "") {
                    source_package = $1
                    sub(/:[^:]+$/, "", source_package)
                }
                source_version = $4
                if (source_version == "") {
                    source_version = $2
                }
                if (source_version != "") {
                    printf "%s=%s\n", source_package, source_version
                } else {
                    print source_package
                }
            }
            NR == FNR {
                baseline[$1] = $2
                next
            }
            length($1) > 0 && (!($1 in baseline) || baseline[$1] != $2) {
                print_source_package()
            }
        ' "${changed_from_file}" "${current_packages_file}" >> "${source_packages_file}"
    fi
fi

for package in "$@"; do
    printf '%s\n' "${package}" >> "${packages_file}"
done

if [ ! -s "${packages_file}" ] && [ ! -s "${source_packages_file}" ]; then
    echo "no apt packages selected for source collection" >> "${log_file}"
    exit 0
fi

enable_deb_src_for_official_repos() {
    local file tmp

    for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do
        [ -f "${file}" ] || continue
        tmp="${file}.src-tmp"
        awk '
            {
                print $0
                line = $0
                if (line ~ /^[[:space:]]*deb[[:space:]]+/ &&
                    line !~ /^[[:space:]]*deb-src[[:space:]]+/ &&
                    line ~ /(deb\.debian\.org|security\.debian\.org|archive\.ubuntu\.com|security\.ubuntu\.com|ports\.ubuntu\.com)/) {
                    sub(/^[[:space:]]*deb[[:space:]]+/, "deb-src ", line)
                    print line
                }
            }
        ' "${file}" > "${tmp}" && mv "${tmp}" "${file}"
    done

    for file in /etc/apt/sources.list.d/*.sources; do
        [ -f "${file}" ] || continue
        if grep -Eq 'URIs:.*(deb\.debian\.org|security\.debian\.org|archive\.ubuntu\.com|security\.ubuntu\.com|ports\.ubuntu\.com)' "${file}"; then
            sed -i -E '/^Types:/ {/deb-src/! s/$/ deb-src/}' "${file}" || true
        fi
    done
}

enable_deb_src_for_official_repos

if [ -d /etc/apt ]; then
    tar -C /etc/apt -czf "${output_dir}/manifests/apt-sources-config.tar.gz" sources.list sources.list.d 2>/dev/null || true
fi

if ! apt-get update >> "${log_file}" 2>&1; then
    echo "apt-get update failed after enabling deb-src; skipping apt source collection" >> "${missing_file}"
    exit 0
fi

while IFS= read -r package; do
    [ -n "${package}" ] || continue
    source_package="$(
        apt-cache show --no-all-versions "${package}" 2>/dev/null |
            awk -F': ' '
                /^Source:/ {
                    value = $2
                    sub(/[[:space:]]*\(.*/, "", value)
                    print value
                    found = 1
                    exit
                }
                END { if (!found) exit 1 }
            '
    )"
    if [ -z "${source_package}" ]; then
        source_package="${package}"
    fi
    printf '%s\n' "${source_package}" >> "${source_packages_file}"
done < "${packages_file}"

sort -u "${source_packages_file}" | while IFS= read -r source_package; do
    [ -n "${source_package}" ] || continue
    if (
        cd "${output_dir}/sources" &&
            apt-get source --download-only --only-source "${source_package}" >> "${log_file}" 2>&1
    ); then
        printf '%s\n' "${source_package}" >> "${downloaded_file}"
    else
        printf '%s\n' "${source_package}" >> "${missing_file}"
    fi
done

apt-get clean >/dev/null 2>&1 || true
rm -rf /var/lib/apt/lists/* 2>/dev/null || true
