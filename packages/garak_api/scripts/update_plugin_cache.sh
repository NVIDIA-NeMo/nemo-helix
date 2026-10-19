#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


usage() {
	echo "$0 - Download the garak plugin cache for the requested version and update garak API package version." 1>&2
	echo 1>&2
	echo "Usage: $0 <garak version>" 1>&2
	echo 1>&2
	echo "By default, script operates on plugin cache under same package root directory as script." 1>&2
	echo "To operate on alternative garak_api package directory, set NHX_GARAK_PACKAGE_ROOT to the package root eg ~/src/alt_dir/package/garak_api" 1>&2
}

if [[ ! $NHX_GARAK_PACKAGE_ROOT ]]; then
	NHX_GARAK_PACKAGE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd ../ >/dev/null 2>&1 && pwd )"
fi

RESOURCES_DIR="${NHX_GARAK_PACKAGE_ROOT}/garakapi/resources"
if [[ ! -d "$RESOURCES_DIR" ]]; then
	echo "ERROR: Target resources dir '${RESOURCES_DIR}' not found; check NHX_GARAK_PACKAGE_ROOT." 1>&2
	echo 1>&2
	usage
	exit 1;
fi

if [[ ! $1 ]]; then
	echo "ERROR: Missing garak version" 1>&2
	echo 1>&2
	usage
	exit 1;
fi

garak_ver="$1"

cache_url="https://raw.githubusercontent.com/NVIDIA/garak/refs/tags/v${garak_ver}/garak/resources/plugin_cache.json"

curl -f -H "Accept: application/json" -o "${RESOURCES_DIR}/plugin_cache.json" "${cache_url}"

if [ $? -ne 0 ]; then
	echo "ERROR: Failed to download plugin cache at URL ${cache_url}. Check that the version is valid and the URL exists." 1>&2
	exit 1;
fi

CODE_DIR="${NHX_GARAK_PACKAGE_ROOT}/garakapi"

declare -a pyfiles=("_selection.py" "_spec.py" "_config.py" "_plugins.py" "exception.py")

for pyfile in "${pyfiles[@]}"; do
	pyfile_url="https://raw.githubusercontent.com/NVIDIA/garak/refs/tags/v${garak_ver}/garak/${pyfile}"
	curl -f -H "Accept: text/plain" -o "${CODE_DIR}/${pyfile}" "${pyfile_url}"
	if [ $? -ne 0 ]; then
		echo "ERROR: Failed to download python code file at URL ${pyfile_url}. Check that the version is valid and the URL exists." 1>&2
		echo "If python file is missing in upstream garak repo, local garak API package may need to be modified for compatibility with upstream garak version." 1>&2
		exit 1;
	fi
done


echo $garak_ver > "${RESOURCES_DIR}/VERSION"

echo "Downloaded plugin cache for version ${garak_ver} and updated garak API version."
