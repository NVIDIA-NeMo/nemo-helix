# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx
from fastapi import HTTPException
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.utils import pre_init
from nhx.guardrails.api.schemas import BaseRequest
from nhx.guardrails.app.constants import FALLBACK_DEFAULT_ENDPOINT_URL, NIM_LLM
from nhx.guardrails.app.llms.completion.base import FlexibleLLMBase
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

_FALLBACK_DEFAULT_ENDPOINT_URL = FALLBACK_DEFAULT_ENDPOINT_URL

log = logging.getLogger(__name__)


class NIM(FlexibleLLMBase, BaseRequest):
    """NGM LLM models

    It is used to interact with the NIM LLM models and also other model providers that support OpenAI's api.

    Attributes:
        api_host (Optional[str]): The API endpoint to use for inference.
        api_key (Optional[SecretStr]): The API key to use in the OpenAI client.
        api_version (Optional[str]): The version of the API to use. Defaults to "v1".
    """

    api_host: Optional[str] = None
    api_version: Optional[str] = "v1"
    api_key: Optional[SecretStr] = None

    @pre_init
    def _fetch_model_endpoint_details(cls, values: Dict):
        """
        This function is called at request time for a completion request for the `main` model, before invoking the model.
        It determines the API key and base URL to use for inference.
        """
        # Check if api_key was passed directly
        if values.get("api_key") is not None:
            raise ValueError(
                "API keys cannot be passed directly to NIM. "
                "Use Inference Gateway for credentials management, or set the credentials in the "
                "'X-Model-Authorization' request header."
            )

        main_model = get_main_model_from_context()

        try:
            # Determine Base URL for model
            inference_base_url = determine_main_model_base_url(values)
            values["api_host"] = inference_base_url
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

    def get_api_key(self) -> Optional[str]:
        api_key = self.api_key
        return api_key.get_secret_value() if api_key is not None else "EMPTY"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""

        return NIM_LLM

    def _get_request_url(self) -> str:
        api_host = self._get_api_host()

        return f"{api_host}/completions"

    def _get_api_host(self) -> str:
        if self._endpoint_url is None:
            return self.api_host
        return self._endpoint_url

    def _append_v1_if_not_present(self, api_host: str) -> str:
        return api_host if api_host.endswith("/v1") else f"{api_host}/v1"

    def _get_request_headers(self, **kwargs: Any) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        api_key = self.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if kwargs.get("stream"):
            headers["x-stream"] = "true"

        custom_headers = get_request_default_headers_from_context()

        if custom_headers:
            headers.update(custom_headers)

        return headers

    def _get_request_json(self, prompt: str, stop: Optional[List[str]] = None) -> Dict:
        # if stop is None:
        #     stop = []

        return {
            "prompt": prompt,
            # "stop": stop,
            **self._default_params,
        }

    def _get_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(60.0, connect=10.0)

    def _handle_response_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses with consistent error messages.

        Args:
            response: The httpx Response object to check for errors

        Raises:
            HTTPException: For all non-200 responses with appropriate status codes and messages
        """
        if response.is_success:
            return

        if response.status_code == 401:
            detail = (
                "Authentication failed. Please verify your API key is valid and configured to be used by this endpoint."
            )
            raise HTTPException(status_code=401, detail=detail)
        elif response.status_code == 404:
            detail = (
                "The endpoint was not found. This can occur when the /completions endpoint is not supported for this model. "
                "Please try the /chat/completions endpoint instead."
            )
            raise HTTPException(status_code=404, detail=detail)
        else:
            raise HTTPException(status_code=response.status_code, detail=response.json())

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        stop = self.stop if stop is None else stop
        with httpx.Client(timeout=self._get_timeout()) as client:
            with client.stream(
                "POST",
                url=self._get_request_url(),
                headers=self._get_request_headers(**kwargs),
                json=self._get_request_json(prompt, stop),
            ) as response:
                # Access the headers
                headers = response.headers

                # we set the response headers so that we can access them later
                set_x_model_response_headers_into_context(headers)

                for json_line in response.iter_lines():
                    if not json_line:
                        continue

                    try:
                        json_line = json_line.lstrip("data: ").strip()

                        if json_line.strip() == "[DONE]":
                            break

                        generation_chunk = _stream_response_to_generation_chunk(json.loads(json_line))
                    except json.JSONDecodeError:
                        log.error(f"Failed to decode JSON: {json_line}")
                        continue

                    if len(generation_chunk.text) == 0:
                        continue

                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            verbose=self.verbose,
                            logprobs=(
                                generation_chunk.generation_info["logprobs"]
                                if generation_chunk.generation_info
                                else None
                            ),
                        )

                    yield generation_chunk

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        stop = self.stop if stop is None else stop

        completion = ""

        if kwargs.get("stream"):
            for chunk in self._stream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs):
                completion += chunk.text

            return completion

        with httpx.Client(timeout=self._get_timeout()) as client:
            response = client.post(
                url=self._get_request_url(),
                headers=self._get_request_headers(**kwargs),
                json=self._get_request_json(prompt, stop),
            )

        self._handle_response_error(response)
        return response.json()["choices"][0]["text"]

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        stop = self.stop if stop is None else stop
        generation: Optional[GenerationChunk] = None
        token_usage: Dict[str, int] = {}
        choices = []
        response = None

        for prompt in prompts:
            if kwargs.get("stream", False):
                for chunk in self._stream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk

                    assert generation is not None

                    choices.append(
                        {
                            "text": generation.text,
                            "finish_reason": (
                                generation.generation_info.get("finish_reason") if generation.generation_info else None
                            ),
                            "logprobs": (
                                generation.generation_info.get("logprobs") if generation.generation_info else None
                            ),
                        }
                    )
            else:
                with httpx.Client(timeout=self._get_timeout()) as client:
                    response = client.post(
                        url=self._get_request_url(),
                        headers=self._get_request_headers(**kwargs),
                        json=self._get_request_json(prompt, stop),
                    )

                # Access the headers
                headers = response.headers

                # we set the response headers so that we can access them later
                set_x_model_response_headers_into_context(headers)
                self._handle_response_error(response)

                response = response.json()

                update_token_usage(response, token_usage)

                choices.extend(response["choices"])

        return self._create_llm_result(choices, prompts, token_usage, response=response)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        stop = self.stop if stop is None else stop
        async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
            async with client.stream(
                "POST",
                url=self._get_request_url(),
                headers=self._get_request_headers(**kwargs),
                json=self._get_request_json(prompt, stop),
            ) as response:
                # Access the headers
                headers = response.headers
                # we set the response headers so that we can access them later
                set_x_model_response_headers_into_context(headers)

                async for json_line in response.aiter_lines():
                    if not json_line:
                        continue

                    try:
                        json_line = json_line.lstrip("data: ").strip()
                        if json_line.strip() == "[DONE]":
                            break
                        chunk = _stream_response_to_generation_chunk(json.loads(json_line))
                    except json.JSONDecodeError:
                        log.error(f"Failed to decode JSON: {json_line}")
                        continue

                    # json_line = json_line.lstrip("data: ").strip()
                    # chunk = _stream_response_to_generation_chunk(json.loads(json_line))

                    # We make sure we don't sent chunks of length 0 as this will end the streaming
                    if len(chunk.text) == 0:
                        continue

                    if run_manager:
                        await run_manager.on_llm_new_token(
                            chunk.text,
                            chunk=chunk,
                            verbose=self.verbose,
                            logprobs=(chunk.generation_info["logprobs"] if chunk.generation_info else None),
                        )
                    yield chunk

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Call out to NeMoLLM completion endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        stop = self.stop if stop is None else stop
        generation: Optional[GenerationChunk] = None

        if kwargs.get("stream"):
            async for chunk in self._astream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk

            assert generation is not None

            return generation.text

        async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
            response = await client.post(
                url=self._get_request_url(),
                headers=self._get_request_headers(**kwargs),
                json=self._get_request_json(prompt, stop),
            )

            # Access the headers
            headers = response.headers

            # we set the response headers so that we can access them later
            set_x_model_response_headers_into_context(headers)

        self._handle_response_error(response)
        return response.json()["choices"][0]["text"]

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        stop = self.stop if stop is None else stop
        generation: Optional[GenerationChunk] = None
        choices = []
        response = None
        token_usage: Dict[str, int] = {}

        for prompt in prompts:
            if kwargs.get("stream", False):
                async for chunk in self._astream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk

                assert generation is not None

                choices.append(
                    {
                        "text": generation.text,
                        "finish_reason": (
                            generation.generation_info.get("finish_reason") if generation.generation_info else None
                        ),
                        "logprobs": (
                            generation.generation_info.get("logprobs") if generation.generation_info else None
                        ),
                    }
                )
            else:
                async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
                    response = await client.post(
                        url=self._get_request_url(),
                        headers=self._get_request_headers(**kwargs),
                        json=self._get_request_json(prompt, stop),
                    )

                # Access the headers
                headers = response.headers

                # we set the response headers so that we can access them later
                set_x_model_response_headers_into_context(headers)
                self._handle_response_error(response)

                response = response.json()

                choices.extend(response["choices"])
                update_token_usage(response, token_usage)

        return self._create_llm_result(choices, prompts, token_usage, response=response)

    def _create_llm_result(
        self,
        choices: List[Dict[str, Any]],
        prompts: List[str],
        token_usage: Dict[str, int],
        response: Optional[Dict[str, Any]] = None,
    ) -> LLMResult:
        """Create an LLMResult from the given choices and prompts.

        Args:
            choices (list): A list of dictionaries containing the generated text and additional information.
            prompts (list): A list of prompt strings used to generate the choices.
            token_usage (dict): A dictionary with token usage statistics.

        Returns:
            LLMResult: The result containing the generations and token usage.
        """
        generations = []

        if response and not isinstance(response, dict):
            response = response.model_dump()
        for prompt_index in range(len(prompts)):
            sub_choices = choices[prompt_index : prompt_index + 1]

            generation_list = [
                Generation(
                    text=choice["text"],
                    generation_info={
                        "finish_reason": choice.get("finish_reason"),
                        "logprobs": choice.get("logprobs"),
                    },
                )
                for choice in sub_choices
            ]

            generations.append(generation_list)

        logprobs = generations[0][0].generation_info.get("logprobs")
        if logprobs:
            logprobs.pop("content", None)
        finish_reason = generations[0][0].generation_info.get("finish_reason")
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model,
            "logprobs": logprobs,
            "finish_reason": finish_reason,
        }
        if response:
            llm_output.update(response)
        # Return the LLMResult object
        return LLMResult(generations=generations, llm_output=llm_output)


def _stream_response_to_generation_chunk(
    stream_response: Dict[str, Any],
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""

    if not stream_response["choices"]:
        return GenerationChunk(text="")
    return GenerationChunk(
        text=stream_response["choices"][0]["text"],  # .split(" ")[-1],
        generation_info=dict(
            finish_reason=stream_response["choices"][0].get("finish_reason", None),
            logprobs=stream_response["choices"][0].get("logprobs", None),
        ),
    )


def update_token_usage(response: Dict[str, Any], token_usage: Dict[str, Any]) -> None:
    """Update token usage."""
    keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
    _keys_to_use = keys.intersection(response["usage"])
    for _key in _keys_to_use:
        if _key not in token_usage:
            token_usage[_key] = response["usage"][_key]
        else:
            token_usage[_key] += response["usage"][_key]
