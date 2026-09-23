# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional, Type

import yaml
from fastapi import HTTPException, Request, status
from nemoguardrails import LLMRails
from nhx.common.service.headers import build_downstream_service_headers
from nhx.guardrails.app.constants import X_MODEL_AUTHORIZATION_HEADER
from nhx.guardrails.app.handlers.utils import (
    get_main_model_from_config,
    get_main_model_from_rails_config,
    get_merged_custom_headers_token,
    get_model_config_object,
    get_rail_types_from_config,
    get_rails_name_from_config,
    run_generate_async,
    set_main_model_merged_custom_headers_into_context,
    to_internal_rails_config,
)
from nhx.guardrails.app.schemas.utils.generation_options import (
    get_activated_rails_logging_options,
    is_activated_rails_logging_enabled,
    update_generation_options,
)
from nhx.guardrails.app.schemas.utils.request_converters import convert_check_request_to_guardrails
from nhx.guardrails.app.schemas.utils.response_transformers import (
    create_guardrail_check_response_from_generation_response,
)
from nhx.guardrails.app.services.rails.service import RailsService
from nhx.guardrails.app.services.utils import normalize_config_ids
from nhx.guardrails.app.utils.config_utils import configure_rails_config
from nhx.guardrails.app.utils.context_utils import (
    set_http_request_uid_into_context,
    set_main_model_into_context,
    set_request_default_headers_into_context,
    set_x_model_auth_token_into_context,
)
from nhx.guardrails.app.utils.hash_utils import compute_token_headers_hash
from nhx.guardrails.config import settings
from nhx.guardrails.entities.values._private import RailsConfig
from nhx.guardrails.entities.values.check import GuardrailCheckRequest, GuardrailCheckResponse

log = logging.getLogger(__name__)


class CheckRequestHandler:
    """Handle an incoming request.

    Forwards the request to the guardrail layer and returns the response.

    TODO: There's a lot of duplication between CheckRequestHandler and CompletionRequestHandler.
      To fix.
    """

    def __init__(
        self,
        rails_service: RailsService,
        request: Request,
        request_body: GuardrailCheckRequest,
        response_model: Type[GuardrailCheckResponse],
        workspace: str,
    ):
        self.request_body = request_body
        self.llm_rails = None
        self.response_model = response_model
        self.request = request
        self.body = None
        self.workspace = workspace

        self._rails_service = rails_service
        self.body = convert_check_request_to_guardrails(self.request_body)

        # The current exception ...
        self.current_rails_exceptions = None

        self.current_activated_rails_logging_options = None

    async def handle_request(self):
        self.set_api_request_headers()
        self.ensure_request_id()
        self.set_custom_headers()
        token = self.record_auth_token_in_context()

        messages = self._get_messages(self.body)
        try:
            config_ids, config = self.get_guardrails_config()
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)) from e

        if config_ids:
            log.info("Got request for config(s) %s", config_ids)

            self.llm_rails = await self.instantiate_llm_rails(config_ids, token)

            self.current_activated_rails_logging_options = get_activated_rails_logging_options(
                self.request_body.guardrails
            )
            # We need to set if we are dealing with input/output/both rails.
            rail_type_options = get_rail_types_from_config(self.llm_rails.config)  # type: ignore

            log_options = {
                "log": {
                    **self.request_body.guardrails.options.log.model_dump(exclude_unset=True, exclude_defaults=True),
                    "activated_rails": True,
                }
            }

            combined_rails_options = {**rail_type_options, **log_options}

            self.request_body.guardrails = update_generation_options(
                self.request_body.guardrails,
                options=combined_rails_options,  # type: ignore
            )

            # as we have updated the options
            # TODO: change to
            # self.body = body_to_guardrails(self.request.body)

            self.body = convert_check_request_to_guardrails(self.request_body)

            # we enable exceptions here so that we do not override the cache
            self.current_rails_exceptions = self.llm_rails.config.enable_rails_exceptions
            self.llm_rails.config.enable_rails_exceptions = True

        elif config:
            log.debug("No config_id provided in the request")
            config_content = self.request_body.guardrails.config.model_dump(exclude_unset=True)

            yaml_content = yaml.dump(config_content)

            rails_config = RailsConfig.from_content(yaml_content=yaml_content)

            self.current_activated_rails_logging_options = get_activated_rails_logging_options(
                self.request_body.guardrails
            )

            # Ensure that the request_model is the same as the model in the passed config
            # get the main model name
            main_model_name = self.request_body.model
            model_config = get_model_config_object(
                schema_type=type(self.request_body),
                model_name=main_model_name,
                type="main",
            )
            rails_config = configure_rails_config(rails_config, model_config)

            # Set the main model into context so we can extract relevant information (ex. base URL) at inference time
            main_model = get_main_model_from_rails_config(rails_config)
            set_main_model_into_context(main_model)

            rail_type_options = get_rail_types_from_config(rails_config)

            log_options = {"log": {"activated_rails": True}}

            combined_rails_options = {**rail_type_options, **log_options}

            self.request_body.guardrails = update_generation_options(
                self.request_body.guardrails,
                options=combined_rails_options,  # type: ignore
            )

            self.body = convert_check_request_to_guardrails(self.request_body)

            try:
                self.llm_rails = LLMRails(to_internal_rails_config(rails_config))
            except Exception as e:
                log.error(
                    f"Failed to instantiate LLMRails instance for config {config_ids} and model {main_model_name}"
                )
                raise e

            # if we get here then the `enable_rails_exceptions` is set to True
            # so we must control it in the `handle_normal` method
            self.current_rails_exceptions = self.llm_rails.config.enable_rails_exceptions

            # we enable exceptions here so that we do not override the cache
            self.llm_rails.config.enable_rails_exceptions = True

        else:
            raise ValueError("No config_id or config provided in the request")

        if self.llm_rails is None:
            raise ValueError("The LLMRails, the rails entry point, is not instantiated.")

        set_main_model_merged_custom_headers_into_context(get_main_model_from_config(self.llm_rails.config.models))

        result = await self._handle_non_streaming(messages=messages)
        return result

    async def _handle_non_streaming(
        self,
        messages: Optional[List[dict]] = None,
    ) -> GuardrailCheckResponse:
        try:
            options = self.body.options.model_dump(exclude_unset=True)

            res = await run_generate_async(
                self.llm_rails,
                messages=messages,
                options=options,
                state=self.body.state,
            )

            self.llm_rails.config.enable_rails_exceptions = self.current_rails_exceptions
        except Exception as e:
            # we need to reset the config to its original state
            # even when an exception occurs
            self.llm_rails.config.enable_rails_exceptions = self.current_rails_exceptions
            raise e

        rail_names = get_rails_name_from_config(self.llm_rails.config)
        exclude_activated_rails_options = not is_activated_rails_logging_enabled(
            self.current_activated_rails_logging_options
        )

        result: GuardrailCheckResponse = create_guardrail_check_response_from_generation_response(
            res,
            rails=rail_names,
            exclude_activated_rails_options=exclude_activated_rails_options,
        )

        return result

    async def instantiate_llm_rails(self, config_ids, token, enable_rails_exceptions: bool = True) -> LLMRails:
        # get the main model name
        model_name = self.request_body.model

        # generate the request model: type of the llm model to be used
        # specially for nim engine
        model_config = get_model_config_object(
            schema_type=type(self.request_body),
            model_name=model_name,
            type="main",
        )

        # we use the token hash to get the rails instance
        # we use the merged token and custom headers as one of the parts of the cache key for the rails instance
        req_headers_cache_key = compute_token_headers_hash(get_merged_custom_headers_token(token))

        llm_rails = await self._rails_service.get_rails(
            config_ids=config_ids,
            model=model_config,
            req_headers_cache_key=req_headers_cache_key,
        )

        if llm_rails is None:
            log.error("Failed to instantiate llm_rails")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to instantiate llm_rails",
            )
        return llm_rails

    def _get_messages(self, body):
        messages = getattr(body, "messages", None)
        return messages

    def set_api_request_headers(self):
        api_request_headers = settings.api_request_headers
        api_request_headers.set(self.request.headers)

    def ensure_request_id(self):
        request_id = self.request.state.request_id
        if not request_id:
            log.error("Request ID not set in the request, middleware is not working")
        set_http_request_uid_into_context(request_id)

    def set_custom_headers(self):
        custom_headers = self._get_custom_headers()
        # Inject default NeMo Helix headers in addition to user-supplied custom headers.
        # These are required to ensure the Langchain HTTP client, used for inference
        # with non-main models, propagates the correct NeMo Helix auth and OTEL headers to IGW.
        merged = {**custom_headers, **build_downstream_service_headers("guardrails")}
        if merged:
            set_request_default_headers_into_context(merged)

    def _get_custom_headers(self) -> dict:
        """Extract and return all headers that start with 'x' or 'X', excluding 'X-Model-Authorization'."""
        custom_headers = {
            k: v
            for k, v in self.request.headers.items()
            if k.lower().startswith("x-") and k.lower() != X_MODEL_AUTHORIZATION_HEADER.lower()
        }
        return custom_headers

    def record_auth_token_in_context(self):
        """Records the authorization token in the async context.

        This will be used later on, when making the actual request
        """
        token = self._get_auth_token()
        if token:
            set_x_model_auth_token_into_context(token)
        return token

    def _get_auth_token(self) -> str | None:
        """Extract the authorization token from the request headers if present.

        Returns:
            str | None: The authorization token if found, otherwise None.
        """

        token = None
        model_auth_header = self.request.headers.get("X-Model-Authorization")
        if model_auth_header:
            token = model_auth_header
        return token

    def auth(self):
        token = self._get_auth_token()

        # Set the API key for the request
        if token:
            set_x_model_auth_token_into_context(token)

        return token

    def get_guardrails_config(self):
        config_ids = getattr(self.request_body.guardrails, "config_ids", None)
        config = getattr(self.request_body.guardrails, "config", None)
        if config_ids:
            config_ids = normalize_config_ids(config_ids, default_workspace=self.workspace)
        return config_ids, config
