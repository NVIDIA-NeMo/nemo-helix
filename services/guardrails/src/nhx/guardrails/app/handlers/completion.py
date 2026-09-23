# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from typing import List, Optional, Type, Union

import yaml
from fastapi import HTTPException, Request, status
from nemoguardrails import LLMRails
from nhx.common.service.headers import build_downstream_service_headers
from nhx.guardrails.api.schemas import ChatCompletionResponseChoice, CompletionResponseChoice
from nhx.guardrails.app.constants import X_MODEL_AUTHORIZATION_HEADER
from nhx.guardrails.app.handlers.utils import (
    get_main_model_from_config,
    get_main_model_from_rails_config,
    get_merged_custom_headers_token,
    get_model_config_object,
    run_generate_async,
    set_main_model_merged_custom_headers_into_context,
    to_internal_rails_config,
)
from nhx.guardrails.app.schemas.utils.request_converters import (
    convert_chat_completion_request_to_guardrails,
    convert_completion_request_to_guardrails,
)
from nhx.guardrails.app.schemas.utils.response_transformers import (
    create_guardrail_chat_completion_response_from_generation_response,
    create_guardrail_completion_response_from_generation_response,
)
from nhx.guardrails.app.schemas.utils.stream_resonse_transformers import (
    create_guardrail_chat_completion_stream_response_from_chunk,
    create_guardrail_completion_stream_response_from_chunk,
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
from nhx.guardrails.entities.values.chat import (
    GuardrailChatCompletionRequest,
    GuardrailChatCompletionResponse,
    GuardrailChatCompletionStreamResponse,
)
from nhx.guardrails.entities.values.completions import (
    GuardrailCompletionRequest,
    GuardrailCompletionResponse,
    GuardrailCompletionStreamResponse,
)
from pydantic import BaseModel, ValidationError
from starlette.responses import StreamingResponse

log = logging.getLogger(__name__)


class CompletionRequestHandler:
    """Handle an incoming request.

    Forwards the request to the guardrail layer and returns the response.
    """

    def __init__(
        self,
        rails_service: RailsService,
        request: Request,
        request_body: Union[
            GuardrailChatCompletionRequest,
            GuardrailCompletionRequest,
        ],
        normal_response_model: Union[
            Type[GuardrailChatCompletionResponse],
            Type[GuardrailCompletionResponse],
        ],
        streaming_response_model: Union[
            Type[GuardrailChatCompletionStreamResponse],
            Type[GuardrailCompletionStreamResponse],
        ],
        workspace: str,
    ):
        self.request_body = request_body
        self.llm_rails = None
        self.normal_response_model = normal_response_model
        self.streaming_response_model = streaming_response_model
        self.request = request
        self.rails_service = rails_service
        self.workspace = workspace

        # self.body = self.request_body.to_guardrails()  # type: ignore
        # the unified interface has changed to the following
        if isinstance(self.request_body, GuardrailChatCompletionRequest):
            self.body = convert_chat_completion_request_to_guardrails(self.request_body)
        elif isinstance(self.request_body, GuardrailCompletionRequest):
            self.body = convert_completion_request_to_guardrails(self.request_body)
        else:
            raise ValueError("Invalid request body type")

        self._raw_request_body = request_body.model_dump(exclude_unset=True)  # type: ignore

    async def handle_request(self):
        log.info("Got request for config(s) %s", getattr(self.body, "config_ids", None))

        self.set_api_request_headers()
        self.ensure_request_id()
        self.set_custom_headers()
        token = self.authenticate_and_set_headers()

        try:
            config_ids, config = self.get_guardrails_config()
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)) from e
        prompt, messages = self._get_prompt_or_messages(self.body)

        if config_ids:
            _t0 = asyncio.get_event_loop().time()
            await self.instantiate_llm_rails(config_ids, token)
            log.debug(
                "instantiate_llm_rails took %.3fs for config_ids=%s",
                asyncio.get_event_loop().time() - _t0,
                config_ids,
            )

        elif config:
            log.debug("No config_id provided in the request")

            config_content = self.request_body.guardrails.config.model_dump(exclude_unset=True)

            yaml_content = yaml.dump(config_content)

            rails_config = RailsConfig.from_content(yaml_content=yaml_content)

            # Ensure that the request_model is the same as the model in the passed config
            # get the main model name
            main_model_name = self.request_body.model
            model_config = get_model_config_object(
                schema_type=type(self.request_body),
                model_name=main_model_name,
                type="main",
            )
            rails_config = configure_rails_config(rails_config, model_config)

            # Set the main model into context, so we can extract relevant information (ex. base URL) at inference time
            main_model = get_main_model_from_rails_config(rails_config)
            set_main_model_into_context(main_model)

            if isinstance(self.request_body, GuardrailChatCompletionRequest):
                self.body = convert_chat_completion_request_to_guardrails(self.request_body)
            elif isinstance(self.request_body, GuardrailCompletionRequest):
                self.body = convert_completion_request_to_guardrails(self.request_body)
            else:
                raise ValueError("Invalid request body type")

            prompt, messages = self._get_prompt_or_messages(self.body)

            try:
                # Run LLMRails init in a thread to avoid blocking the event loop.
                # LLMRails.__init__ has blocking file I/O and a blocking thread.join() for KB init.
                self.llm_rails = await asyncio.to_thread(LLMRails, to_internal_rails_config(rails_config))
            except Exception as e:
                log.error(
                    f"Failed to instantiate LLMRails instance for config {config_ids} and model {main_model_name}"
                )
                raise e

        else:
            raise ValueError("No config_id or config provided in the request")

        if self.llm_rails is None:
            raise ValueError("The LLMRails, the rails entry point, is not instantiated.")

        set_main_model_merged_custom_headers_into_context(get_main_model_from_config(self.llm_rails.config.models))

        if getattr(self.body, "stream", False):
            result = await self._handle_streaming(messages=messages, prompt=prompt)
            result = StreamingResponse(result, media_type="text/event-stream")
        else:
            result = await self._handle_non_streaming(messages=messages, prompt=prompt)
            result = self._post_process_response(result)

        return result

    async def _handle_non_streaming(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        response_model: Optional[
            Union[Type[GuardrailChatCompletionResponse], Type[GuardrailCompletionResponse]]
        ] = None,
    ):
        if response_model is None:
            response_model = self.normal_response_model

        options_dict = self.body.options.model_dump(exclude_unset=True) if self.body.options else {}

        # Capture a copy of the log options from the incoming request before we mutate options_dict
        # below. We always set `activated_rails=True` internally, but that must not affect what the
        # transformer thinks the user originally requested.
        request_log_options: dict | None = options_dict["log"].copy() if "log" in options_dict else None

        # Explicitly set `activated_rails=True` internally so we can extract the model name and
        # usage from the generation rail.
        if "log" not in options_dict:
            options_dict["log"] = {}
        options_dict["log"]["activated_rails"] = True

        _t0 = asyncio.get_event_loop().time()
        res = await run_generate_async(
            self.llm_rails,
            messages=messages,
            prompt=prompt,
            options=options_dict,
            state=self.body.state,
        )
        log.debug("run_generate_async (non-streaming) took %.3fs", asyncio.get_event_loop().time() - _t0)

        if isinstance(self.request_body, GuardrailChatCompletionRequest):
            result = create_guardrail_chat_completion_response_from_generation_response(
                response=res,
                config_ids=self.body.config_ids,
                log_options=request_log_options,
            )
        elif isinstance(self.request_body, GuardrailCompletionRequest):
            result = create_guardrail_completion_response_from_generation_response(
                response=res,
                config_ids=self.body.config_ids,
                log_options=request_log_options,
            )

        else:
            raise ValueError("Unsupported request body")

        return result

    async def _handle_streaming(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        response_model: Optional[
            Union[
                Type[GuardrailChatCompletionStreamResponse],
                Type[GuardrailCompletionStreamResponse],
            ]
        ] = None,
    ):
        if response_model is None:
            response_model = self.streaming_response_model

        async def streaming_handler(messages: Optional[List[dict]] = None, prompt: Optional[str] = None):
            index = 0

            options_dict = self.body.options.model_dump(exclude_unset=True) if self.body.options else {}

            # create a stream
            stream_iter = self._stream_async(
                prompt=prompt,
                messages=messages,
                options=options_dict,
                state=self.body.state,
            )

            try:
                async for chunk in stream_iter:
                    processed_chunk = process_chunk(chunk)
                    if isinstance(processed_chunk, ErrorData):
                        # Yield the error and stop streaming
                        yield f"data: {processed_chunk.model_dump_json()}\n\n"
                        return

                    kwargs = {
                        "index": index,
                        "chunk": chunk,
                        "model_name": self.request_body.model,
                        "text": chunk,
                        "cumlogprobs": None,
                        "finish_reason": None,
                    }

                    if messages:
                        kwargs["role"] = "assistant"

                    if prompt and "logprobs" not in kwargs:
                        kwargs["logprobs"] = None

                    if issubclass(response_model, GuardrailChatCompletionStreamResponse):
                        response = create_guardrail_chat_completion_stream_response_from_chunk(
                            **kwargs
                        ).model_dump_json()
                    elif issubclass(response_model, GuardrailCompletionStreamResponse):
                        response = create_guardrail_completion_stream_response_from_chunk(**kwargs).model_dump_json()
                    else:
                        raise ValueError("Unsupported response model")

                    # response = response_model.from_chunk(**kwargs).model_dump_json()
                    response = f"data: {response}\n\n"
                    yield response
                    index += 1
            except Exception as e:
                # Exception raised during streaming, before chunk generation. Convert to error chunk.
                log.error(f"Exception during streaming: {type(e).__name__}: {str(e)}", exc_info=True)
                error_data = ErrorData(error=ErrorDetails(message=str(e), type=type(e).__name__, param="", code="500"))
                yield f"data: {error_data.model_dump_json()}\n\n"

        generator = streaming_handler(messages=messages, prompt=prompt)

        return generator

    def _stream_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options=None,
        state=None,
    ):
        return self.llm_rails.stream_async(
            prompt=prompt,
            messages=messages,
            options=options,
            state=state,
        )

    def _get_prompt_or_messages(self, body):
        prompt = getattr(body, "prompt", None)
        messages = getattr(body, "messages", None)
        return prompt, messages

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

    def authenticate_and_set_headers(self):
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
        model_auth_header = self.request.headers.get(X_MODEL_AUTHORIZATION_HEADER)
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

    async def instantiate_llm_rails(self, config_ids, token) -> None:
        model_name = self.request_body.model

        # For NIM, we decide on the type of LLM provider based on the request type
        # (chat vs. completion, OpenAI vs. NeMoLLM format)

        model_config = get_model_config_object(
            schema_type=type(self.request_body),
            model_name=model_name,
            type="main",
        )

        # we use the merged token and custom headers as one of the parts of the cache key for the rails instance
        req_headers_cache_key = compute_token_headers_hash(get_merged_custom_headers_token(token))
        self.llm_rails = await self.rails_service.get_rails(
            config_ids=config_ids,
            model=model_config,
            req_headers_cache_key=req_headers_cache_key,
        )

        if self.llm_rails is None:
            log.error("Failed to instantiate llm_rails")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to instantiate llm_rails",
            )

    def _post_process_response(
        self,
        response: Union[GuardrailCompletionResponse, GuardrailChatCompletionResponse],
    ):
        """Post process the response.

        Removes the guardrail data we the request did not contain the `guardrail` field.
        """

        fields_to_exclude = set()

        guardrails_mode = "guardrails" in self._raw_request_body
        # If we're not in guardrail mode (i.e., no guardrail-related fields
        # were explicitly specified), we remove the `guardrail_data`.

        if not guardrails_mode:
            if isinstance(response, GuardrailCompletionResponse):
                response.guardrails_data = None
            elif isinstance(response, GuardrailChatCompletionResponse):
                response.guardrails_data = None
            else:
                raise ValueError("Invalid response type for post-processing.")
        else:
            if isinstance(response, GuardrailCompletionResponse):
                if (self._raw_request_body.get("guardrails") or {}).get("return_choice"):
                    response.choices.append(
                        CompletionResponseChoice.model_construct(
                            index=len(response.choices),
                            text=response.guardrails_data.model_dump_json(),
                        )
                    )
                    response.guardrails_data = None

            elif isinstance(response, GuardrailChatCompletionResponse):
                if (self._raw_request_body.get("guardrails") or {}).get("return_choice"):
                    response.choices.append(
                        ChatCompletionResponseChoice.model_construct(
                            index=len(response.choices),
                            message={
                                "role": "guardrails_data",
                                "content": response.guardrails_data.model_dump_json(),
                            },
                            finish_reason=None,
                        )
                    )
                    response.guardrails_data = None

        if response.guardrails_data is None:
            fields_to_exclude.add("guardrails_data")
        # we can also use exclude_none=True but it will exclude all None values
        return response.model_dump(exclude=fields_to_exclude)


class ErrorDetails(BaseModel):
    message: str
    type: str = ""
    param: str = ""
    code: str = ""


class ErrorData(BaseModel):
    error: ErrorDetails


def process_chunk(chunk: str) -> Union[str, ErrorData]:
    """
    Processes a single chunk from the stream.

    Args:
        chunk (str): A single chunk from the stream.

    Returns:
        Union[str, ErrorData]: ErrorData instance for errors or the original chunk.
    """
    try:
        validated_data = ErrorData.model_validate_json(chunk)
        log.info("Received error chunk")
        log.info(validated_data.model_dump_json())
        return validated_data  # Return the ErrorData instance directly
    except ValidationError:
        # Not an error, just a normal token
        pass
    except json.JSONDecodeError:
        # Invalid JSON format, treat as normal token
        pass
    except Exception as e:
        # Unexpected error
        log.warning(f"Unexpected error processing stream chunk: {type(e).__name__}: {str(e)}", extra={"chunk": chunk})

    return chunk
