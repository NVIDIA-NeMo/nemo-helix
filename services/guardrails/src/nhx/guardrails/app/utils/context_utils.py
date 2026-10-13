# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextvars
import json
import logging
from collections import defaultdict
from typing import Optional

from nhx.guardrails.entities.values._private import Model

logger = logging.getLogger(__name__)

_response_headers = defaultdict(str)

api_key_var = contextvars.ContextVar("api_key", default=None)
response_header_var = contextvars.ContextVar("response_header", default=None)
http_request_uid_var = contextvars.ContextVar("request_uid", default=None)
request_default_headers_var = contextvars.ContextVar[dict[str, str]]("request_default_headers", default={})
# Model with type `main` in the incoming request
request_main_model_var = contextvars.ContextVar[Optional[Model]]("request_main_model", default=None)


def set_request_default_headers_into_context(headers: dict[str, str] | None):
    """Set default headers for the request using contextvars."""
    if headers is None:
        headers = {}
    request_default_headers_var.set(headers)


def get_request_default_headers_from_context() -> dict[str, str]:
    """Extract and return all headers that start with 'x' or 'X', excluding 'X-Model-Authorization'."""
    headers = request_default_headers_var.get()

    return headers


def get_x_model_response_headers_from_context():
    logger.debug("Getting custom header from response")
    request_uid = get_http_request_uid_from_context()
    if request_uid is None:
        logger.debug("Request UID is None, returning None")
        return None
    logger.debug(f"Getting custom header for request {request_uid}")
    # We pop the header from the dict to avoid potential security issues
    response_header = _response_headers.pop(request_uid, None)
    return response_header


def set_x_model_response_headers_into_context(header):
    logger.debug("Setting custom header")
    if header is None:
        json_header = "{}"
    elif not isinstance(header, str):
        try:
            json_header = json.dumps(dict(header))
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid header type: {e}")
            json_header = "{}"
    else:
        json_header = header

    request_uid = get_http_request_uid_from_context()
    logger.debug(f"Setting custom header for request {request_uid}")

    _response_headers[request_uid] = json_header


def get_x_model_auth_token_from_context():
    api_key = api_key_var.get()
    return api_key


def set_x_model_auth_token_into_context(api_key):
    api_key_var.set(api_key)
    masked_api_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
    logger.debug(f"Setting api key {masked_api_key}")
    return api_key


def get_http_request_uid_from_context():
    http_request_uid = http_request_uid_var.get()
    logger.debug(f"Getting HTTP request UID: {http_request_uid}")
    return http_request_uid


def set_http_request_uid_into_context(http_request_uid):
    http_request_uid_var.set(http_request_uid)
    logger.debug(f"Setting HTTP request UID: {http_request_uid}")
    return http_request_uid


def set_main_model_into_context(main_model: Model):
    request_main_model_var.set(main_model)
    logger.debug(f"Setting main model into context: {main_model}")
    return main_model


def get_main_model_from_context():
    return request_main_model_var.get()
