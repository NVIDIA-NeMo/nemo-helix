# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrapper around NIM OpenAI API."""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import openai
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import pre_init
from nhx.guardrails.api.schemas import BaseRequest
from nhx.guardrails.app.constants import (
    FALLBACK_DEFAULT_ENDPOINT_URL,
    X_MODEL_AUTHORIZATION_HEADER,
)
from nhx.guardrails.app.llms.chat.base import FlexibleChatModelBase
from nhx.guardrails.app.llms.utils import (
    determine_main_model_base_url,
    get_main_model_api_key,
    get_provider_from_context,
)
from nhx.guardrails.app.utils.context_utils import (
    get_main_model_from_context,
    get_request_default_headers_from_context,
    set_x_model_response_headers_into_context,
)
from pydantic import SecretStr
from pydantic_core import PydanticUndefined

log = logging.getLogger(__name__)

_FALLBACK_DEFAULT_ENDPOINT_URL = FALLBACK_DEFAULT_ENDPOINT_URL


def _convert_dict_to_message(d: Dict[str, Any]) -> BaseMessage:
    """Convert a role/content dict from the OpenAI API response to a langchain BaseMessage."""
    role = d.get("role", "")
    content = d.get("content") or ""
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        additional_kwargs: Dict[str, Any] = {}
        if function_call := d.get("function_call"):
            additional_kwargs["function_call"] = function_call
        if tool_calls := d.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        return ChatMessage(role=role, content=content)


def _convert_chunk_to_generation_chunk(
    chunk: Dict[str, Any],
    default_chunk_class: type,
    base_generation_info: Dict[str, Any],
) -> Optional[ChatGenerationChunk]:
    """Convert an OpenAI streaming chunk dict to a ChatGenerationChunk."""
    choices = chunk.get("choices", [])
    if not choices:
        return None
    choice = choices[0]
    delta = choice.get("delta", {})
    content = delta.get("content") or ""
    generation_info = {**base_generation_info}
    if finish_reason := choice.get("finish_reason"):
        generation_info["finish_reason"] = finish_reason
    if logprobs := choice.get("logprobs"):
        generation_info["logprobs"] = logprobs
    message_chunk = default_chunk_class(content=content)
    return ChatGenerationChunk(message=message_chunk, generation_info=generation_info or None)


class ChatNIM(FlexibleChatModelBase, BaseRequest):
    """NGM Chat large language models API.

    This class is used to interact with the NIM Chat API. It provides methods for generating chat responses and streaming chat responses.

    It also works with any model provider that implements OpenAI's API.

    Attributes:
        client (Any): The OpenAI client used to interact with the API.
        async_client (Any): The asynchronous OpenAI client used to interact with the API.
        endpoint_url (Optional[str]): The API endpoint to use for inference.
        api_key (Optional[SecretStr]): The API key to use in the OpenAI client.
        api_version (Optional[str]): The version of the API to use. Defaults to "v1".
        include_response_headers (Optional[bool]): Whether to include response headers in the output. Defaults to True.

    Ref: https://python.langchain.com/docs/how_to/custom_chat_model/
    """

    client: Any
    async_client: Any
    endpoint_url: Optional[str] = None
    api_key: Optional[SecretStr] = None
    max_retries: int = 3
    api_version: Optional[str] = "v1"
    include_response_headers: Optional[bool] = True

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """
        This function is called at request time for a completion request for the `main` model, before invoking the model.
        It determines the API key and base URL, and instantiates the OpenAI client to use for inference.
        """
        try:
            import openai
        except ImportError:
            raise ImportError("Could not import openai python package. Please install it with `pip install openai`.")

        # check if api_key was passed directly
        if values.get("api_key") is not None:
            raise ValueError(
                "API keys cannot be passed directly to ChatNIM. "
                "Use Inference Gateway for credentials management, or set the "
                f"'{X_MODEL_AUTHORIZATION_HEADER}' header in the request."
            )

        main_model = get_main_model_from_context()

        try:
            # Determine Base URL for model
            inference_base_url = determine_main_model_base_url(values)
            values["endpoint_url"] = inference_base_url
            cls._endpoint_url = inference_base_url

            # Determine provider for model
            cls._endpoint_provider = get_provider_from_context()

            # Determine API key for model
            api_key = get_main_model_api_key()

            if api_key:
                values["api_key"] = api_key

            log.info(f"Setting base URL to {cls._endpoint_url} for model {main_model.model if main_model else ''}")

        except Exception as e:
            log.error(f"Failed to determine base URL and API key for {main_model.model if main_model else ''}: {e}")
            raise

        client_params = {
            "api_key": values["api_key"].get_secret_value() if values["api_key"] else None,
            "base_url": inference_base_url,
            "max_retries": values["max_retries"],
        }

        if client_params["base_url"] == _FALLBACK_DEFAULT_ENDPOINT_URL and not client_params["api_key"]:
            raise Exception(
                f"Failed to find API key for {values['model']} at URL {inference_base_url}. "
                f"Please ensure the '{X_MODEL_AUTHORIZATION_HEADER}' header is set with your credentials in the request, "
                "or use Inference Gateway for credentials management."
            )

        if not client_params["api_key"]:
            client_params["api_key"] = "EMPTY"

        values["client"] = openai.OpenAI(**client_params)
        values["async_client"] = openai.AsyncOpenAI(**client_params)

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        base_request_params = BaseRequest.__fields__.keys()
        params = {
            k: getattr(self, k)
            for k in base_request_params
            if getattr(self, k) is not None and getattr(self, k) is not PydanticUndefined
        }

        # pop "streaming" from params
        params.pop("streaming", None)
        # remove endpoint from values as the NIM API forbis extra params in the most recent version
        params.pop("endpoint", None)

        return params

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _invocation_params(self) -> Mapping[str, Any]:
        """Get the parameters used to invoke the model."""
        return {**self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "nimchat"

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        default_chunk_class = AIMessageChunk

        if stop:
            params["stop"] = stop
        # make sure to pop "stream" from params
        params.pop("stream", None)

        custom_extra_headers = get_request_default_headers_from_context()

        try:
            model = params.pop("model")
            self.client.chat.completions.create(
                messages=message_dicts,
                model=model,
                stream=True,
                extra_body={**params},
                extra_headers=custom_extra_headers,
            )

            if self.include_response_headers:
                raw_response = self.client.with_raw_response.create(
                    messages=message_dicts,
                    model=model,
                    stream=True,
                    extra_headers=custom_extra_headers,
                    extra_body={**params},
                )
                response = raw_response.parse()
                base_generation_info = {"headers": dict(raw_response.headers)}
                response_headers = raw_response.headers
                set_x_model_response_headers_into_context(response_headers)
            else:
                response = self.client.create(
                    messages=message_dicts,
                    model=model,
                    stream=True,
                    extra_body={**params},
                    extra_headers=custom_extra_headers,
                )
        except openai.NotFoundError as e:
            log.error(f"NotFoundError generating chat completions: {e}")
            raise Exception("Model not found. Please check if the model exists at this endpoint.")
        except Exception as e:
            log.error(f"Error streaming chat completions: {e}")
            raise

        with response:
            is_first_chunk = True
            for chunk in response:
                if not isinstance(chunk, dict):
                    chunk = chunk.model_dump()
                generation_chunk = _convert_chunk_to_generation_chunk(
                    chunk,
                    default_chunk_class,
                    base_generation_info if is_first_chunk else {},
                )
                if generation_chunk is None:
                    continue
                default_chunk_class = generation_chunk.message.__class__
                logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                if run_manager:
                    run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk, logprobs=logprobs)
                is_first_chunk = False
                yield generation_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if kwargs.get("stream", False):
            stream_iter = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            if stream_iter:
                return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}

        custom_extra_headers = get_request_default_headers_from_context()

        try:
            model = params.pop("model")
            if self.include_response_headers:
                raw_response = self.client.chat.completions.with_raw_response.create(
                    messages=message_dicts,
                    model=model,
                    extra_body={**params},
                    extra_headers=custom_extra_headers,
                )
                response = raw_response.parse()

                response_headers = raw_response.headers
                set_x_model_response_headers_into_context(response_headers)
            else:
                response = self.client.chat.completions.create(
                    messages=message_dicts,
                    model=model,
                    extra_body={**params},
                    extra_headers=custom_extra_headers,
                )
        except openai.NotFoundError as e:
            log.error(f"NotFoundError generating chat completions: {e}")
            raise Exception("Model not found. Please check if the model exists at this endpoint.")
        except Exception as e:
            log.error(f"Error generating chat completions: {e}")
            error_message = str(e)
            if "EMPTY" in error_message:
                endpoint_provider = self._endpoint_provider.upper() if self._endpoint_provider else "UNKNOWN"
                raise Exception(
                    f"Missing API Key for {self.model} model at {self._endpoint_url}."
                    f"You must set the API Key via {endpoint_provider}_API_KEY environment variable."
                )
            else:
                # pass the exception as is
                raise

        return self._create_chat_result(response)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        default_chunk_class = AIMessageChunk

        if stop:
            params["stop"] = stop
        # make sure to pop "stream" from params
        params.pop("stream", None)

        custom_extra_headers = get_request_default_headers_from_context()

        try:
            model = params.pop("model")
            if self.include_response_headers:
                raw_response = await self.async_client.chat.completions.with_raw_response.create(
                    messages=message_dicts,
                    model=model,
                    stream=True,
                    extra_body={**params},
                    extra_headers=custom_extra_headers,
                )
                response = raw_response.parse()
                base_generation_info = {"headers": dict(raw_response.headers)}
                response_headers = raw_response.headers
                set_x_model_response_headers_into_context(response_headers)
            else:
                response = await self.async_client.chat.completions.create(
                    messages=message_dicts,
                    model=model,
                    stream=True,
                    extra_body={**params},
                    extra_headers=custom_extra_headers,
                )
        except openai.NotFoundError as e:
            log.error(f"NotFoundError generating chat completions: {e}")
            raise Exception("Model not found. Please check if the model exists at this endpoint.")
        except Exception as e:
            log.error(f"Error streaming chat completions: {e}")
            raise

        async with response:
            is_first_chunk = True
            async for chunk in response:
                if not isinstance(chunk, dict):
                    chunk = chunk.model_dump()
                generation_chunk = _convert_chunk_to_generation_chunk(
                    chunk,
                    default_chunk_class,
                    base_generation_info if is_first_chunk else {},
                )
                if generation_chunk is None:
                    continue
                default_chunk_class = generation_chunk.message.__class__
                logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                if run_manager:
                    await run_manager.on_llm_new_token(generation_chunk.text, chunk=generation_chunk, logprobs=logprobs)
                is_first_chunk = False
                yield generation_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if kwargs.get("stream", False):
            stream_iter = self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
            if stream_iter:
                return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}

        custom_extra_headers = get_request_default_headers_from_context()

        try:
            model = params.pop("model")
            if self.include_response_headers:
                raw_response = await self.async_client.chat.completions.with_raw_response.create(
                    messages=message_dicts,
                    model=model,
                    extra_body={**params},
                    extra_headers=custom_extra_headers,
                )
                response = raw_response.parse()
                response_headers = raw_response.headers
                set_x_model_response_headers_into_context(response_headers)

            else:
                response = await self.async_client.chat.completions.create(
                    messages=message_dicts,
                    model=model,
                    extra_body={**params},
                    extra_headers=custom_extra_headers,
                )
        except openai.NotFoundError as e:
            log.error(f"NotFoundError generating chat completions: {e}")
            raise Exception("Model not found. Please check if the model exists at this endpoint.")
        except Exception as e:
            log.error(f"Error generating chat completions: {e}")
            # If it looks like we have enough info, we keep the exception as is.
            error_message = str(e)
            if "EMPTY" in error_message:
                endpoint_provider = self._endpoint_provider.upper() if self._endpoint_provider else "UNKNOWN"
                raise Exception(
                    f"Missing API Key for {self.model} model at {self._endpoint_url}."
                    f"You must set the API Key via {endpoint_provider}_API_KEY environment variable."
                )
            else:
                # pass the exception as is
                raise

        return self._create_chat_result(response)

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        else:
            raise TypeError(f"Got unknown type {message}")
        return message_dict

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = dict(self._invocation_params)
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    # FIXME: Any : was BaseModel to fix
    def _create_chat_result(self, response: Union[dict, Any]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.model_dump()
        for choice in response["choices"]:
            # remove Nones like function_call or tool_calls when exist from server
            choice["message"] = {k: v for k, v in choice["message"].items() if v is not None}
            message = _convert_dict_to_message(choice["message"])
            generation_info = dict(finish_reason=choice.get("finish_reason"))
            if "logprobs" in choice:
                generation_info["logprobs"] = choice["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        logprobs = generations[0].generation_info.get("logprobs")
        if logprobs:
            logprobs.pop("content", None)
        finish_reason = generations[-1].generation_info.get("finish_reason")
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model,
            "system_fingerprint": response.get("system_fingerprint", ""),
            "logprobs": logprobs,
            "finish_reason": finish_reason,
        }

        llm_output.update(response)

        return ChatResult(generations=generations, llm_output=llm_output)
