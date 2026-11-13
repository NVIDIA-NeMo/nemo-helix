#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Serves the docker-socket-proxy filter on a root-only Unix socket instead of the
# image's TCP port, so only containers that mount the shared socket volume can reach
# the Docker API. The upstream entrypoint always binds TCP, so render the stock
# haproxy template with a Unix bind and start haproxy the way the upstream one does.
set -eu

SOCKET_DIR="$(dirname "${PROXY_SOCKET_PATH:?}")"
mkdir -p "${SOCKET_DIR}"
chmod 700 "${SOCKET_DIR}"

sed "s|bind \${BIND_CONFIG}|bind unix@${PROXY_SOCKET_PATH} mode 600|" \
  /usr/local/etc/haproxy/haproxy.cfg.template > /usr/local/etc/haproxy/haproxy.cfg
grep -q "bind unix@${PROXY_SOCKET_PATH} mode 600" /usr/local/etc/haproxy/haproxy.cfg

exec haproxy -W -db -f /usr/local/etc/haproxy/haproxy.cfg
