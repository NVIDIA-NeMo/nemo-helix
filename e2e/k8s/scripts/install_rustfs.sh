#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Install RustFS in standalone mode for E2E tests.
# Expects NAMESPACE to be set (e.g. current kubectl context namespace).
#
# See https://github.com/rustfs/helm for parameter reference.
# Uses default credentials (rustfsadmin/rustfsadmin).
#
set -e

STORAGECLASS="${STORAGECLASS:-standard}"

if [ -z "${NAMESPACE}" ]; then
  echo "NAMESPACE must be set"
  exit 1
fi

helm repo add rustfs https://charts.rustfs.com
helm repo update rustfs

helm upgrade -i -n "${NAMESPACE}" rustfs rustfs/rustfs \
  --version 0.0.85 \
  --set mode.standalone.enabled=true \
  --set mode.distributed.enabled=false \
  --set storageclass.name="${STORAGECLASS}" \
  --timeout 5m
