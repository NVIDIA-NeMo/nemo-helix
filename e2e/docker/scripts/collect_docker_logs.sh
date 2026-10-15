#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script: collect_docker_logs.sh
# Description: Collects Docker container logs and diagnostics, writing to files for archiving.
#              Safe to call even if some containers have already exited — individual failures are ignored.
# Usage: ./collect_docker_logs.sh [output-dir]
#
# Environment Variables:
#   LOG_DIR      (optional) - Directory to write logs to (defaults to docker/logs)
#   JOB_LOGS_DIR (optional) - E2E job log directory used when LOG_DIR is unset.

# Note: intentionally no -e so individual docker failures don't abort collection
set -uo pipefail

LOG_DIR="${1:-${LOG_DIR:-${JOB_LOGS_DIR:-docker/logs}}}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

log_info "Collecting Docker diagnostics to: ${LOG_DIR}"
mkdir -p "${LOG_DIR}"

# Host and Docker runtime summaries
docker version                                                     > "${LOG_DIR}/docker-version.txt"       2>&1 || true
docker info                                                        > "${LOG_DIR}/docker-info.txt"          2>&1 || true
docker system df                                                   > "${LOG_DIR}/docker-system-df.txt"     2>&1 || true
docker image ls                                                    > "${LOG_DIR}/images.txt"               2>&1 || true
docker volume ls                                                   > "${LOG_DIR}/volumes.txt"              2>&1 || true
docker network ls                                                  > "${LOG_DIR}/networks.txt"             2>&1 || true
docker stats --no-stream --all                                     > "${LOG_DIR}/container-stats.txt"      2>&1 || true
docker events --since 2h --until "$(date -u +%Y-%m-%dT%H:%M:%SZ)"  > "${LOG_DIR}/docker-events.txt"       2>&1 || true

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi                                                     > "${LOG_DIR}/nvidia-smi.txt"           2>&1 || true
    nvidia-smi -L                                                  > "${LOG_DIR}/nvidia-smi-list.txt"      2>&1 || true
fi
if command -v nvidia-container-cli >/dev/null 2>&1; then
    nvidia-container-cli info                                      > "${LOG_DIR}/nvidia-container-cli.txt" 2>&1 || true
fi
ls -l /dev/nvidia*                                                 > "${LOG_DIR}/nvidia-devices.txt"       2>&1 || true
ldconfig -p | grep -E 'libcuda|nvidia'                             > "${LOG_DIR}/nvidia-libraries.txt"     2>&1 || true

# Summary of all containers (running and stopped)
docker ps -a                                                       > "${LOG_DIR}/containers.txt"           2>&1 || true

# Per-container log collection
CONTAINERS=$(docker ps -aq 2>/dev/null) || true

if [ -z "${CONTAINERS}" ]; then
    log_warn "No containers found."
else
    for CONTAINER in ${CONTAINERS}; do
        NAME=$(docker inspect --format '{{.Name}}' "${CONTAINER}" 2>/dev/null | sed 's|^/||') || NAME="${CONTAINER}"
        log_info "  Collecting container: ${NAME} (${CONTAINER})"

        docker inspect \
            --format '{{json (dict "Id" .Id "Name" .Name "Image" .Image "Created" .Created "Path" .Path "Args" .Args "State" .State "HostConfig" .HostConfig "Mounts" .Mounts "NetworkSettings" .NetworkSettings)}}' \
            "${CONTAINER}" > "${LOG_DIR}/inspect-${NAME}.json" 2>&1 || true
        docker logs "${CONTAINER}"                      > "${LOG_DIR}/logs-${NAME}.txt"     2>&1 || true
    done
fi

log_info "Log collection complete — files written to: ${LOG_DIR}"
