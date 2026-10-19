# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-exports; the error handlers live in :mod:`nemo_helix_plugin.cli_error_handling`."""

from nemo_helix_plugin.cli_error_handling import REMOTE_ERROR_EXIT_CODE as REMOTE_ERROR_EXIT_CODE
from nemo_helix_plugin.cli_error_handling import InvalidSearchPatternError as InvalidSearchPatternError
from nemo_helix_plugin.cli_error_handling import MissingRequiredFieldsError as MissingRequiredFieldsError
from nemo_helix_plugin.cli_error_handling import UnknownInputFieldsError as UnknownInputFieldsError
from nemo_helix_plugin.cli_error_handling import _format_api_error as _format_api_error
from nemo_helix_plugin.cli_error_handling import handle_errors as handle_errors
from nemo_helix_plugin.cli_error_handling import handle_exception as handle_exception
