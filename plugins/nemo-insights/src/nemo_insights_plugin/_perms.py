# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed permissions for Insights HTTP resources."""

from nemo_helix_plugin.authz import PermissionSet, perm


class InsightPerms(PermissionSet, namespace="insights.insights"):
    CREATE = perm("Create insights")
    LIST = perm("List insights")
    READ = perm("Read insights")
    UPDATE = perm("Update insights")
    DELETE = perm("Delete insights")


class AnalysisConfigPerms(PermissionSet, namespace="insights.analysis-configs"):
    ENABLE = perm("Enable periodic analysis")
    DISABLE = perm("Disable periodic analysis")
    LIST = perm("List periodic analysis configuration")
    READ = perm("Read periodic analysis configuration")
    UPDATE = perm("Update periodic analysis configuration")


class AnalysisRunPerms(PermissionSet, namespace="insights.analysis-runs"):
    CREATE = perm("Submit an insights analysis run")
    LIST = perm("List insights analysis runs")
    READ = perm("Read an insights analysis run")


class AnalysisRunStatusPerms(PermissionSet, namespace="insights.analysis-run-statuses"):
    LIST = perm("List analysis run status")
    READ = perm("Read analysis run status")
    UPDATE = perm("Update analysis run status")
