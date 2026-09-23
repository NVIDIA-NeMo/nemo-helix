<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix Python SDK

[![PyPI version](https://img.shields.io/pypi/v/nemo-helix.svg?label=pypi%20(stable))](https://pypi.org/project/nemo-helix/)

The NeMo Helix Python SDK provides convenient access to the NeMo Helix REST API from any Python 3.11+
application. The SDK includes type definitions for all request parameters and response fields,
and offers both synchronous and asynchronous clients powered by [httpx](https://github.com/encode/httpx).

It is generated with [Stainless](https://www.stainless.com/) from the [NeMo Helix REST API](https://docs.nvidia.com/nemo/microservices/latest/api/index.html).

## Documentation

You can find the platform documentation at [NeMo Helix Documentation](https://docs.nvidia.com/nemo/microservices/latest/about/index.html).
The [NVIDIA NeMo Helix APIs](https://docs.nvidia.com/nemo/microservices/latest/api/index.html) section documents the REST API.

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Installation

This project downloads and installs additional third-party open source software projects. Review the license terms of these open source projects before use.

```sh
pip install nemo-helix[all]
```

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Usage

This section describes how to use the NeMo Helix Python SDK.

### Import the Main Client Class

Import the main client class from the `nemo_helix` package and create a client instance as follows:

```python
from nemo_helix import NeMoHelix

client = NeMoHelix(
    base_url="http://nemo.test",
)

# Sample API call 
page = client.workspaces.list()
print(page.data)
```

After creating the client instance, you can use the client to interact with the NeMo Helix APIs.

## Async Usage

If you want to use the asynchronous client, simply import `AsyncNeMoHelix` instead of `NeMoHelix` and use `await` with each API call:

```python
import asyncio
from nemo_helix import AsyncNeMoHelix

client = AsyncNeMoHelix(
    base_url="http://nemo.test",
)

# Sample API call
async def main() -> None:
    page = await client.workspaces.list()
    print(page.data)


asyncio.run(main())
```

Functionality between the synchronous and asynchronous clients is otherwise identical.

### With aiohttp

By default, the async client uses `httpx` for HTTP requests. However, for improved concurrency performance you may also use `aiohttp` as the HTTP backend.

You can enable this by installing `aiohttp`:

```sh
pip install 'nemo-helix[aiohttp]'
```

Then you can enable it by instantiating the client with `http_client=DefaultAioHttpClient()`:

```python
import asyncio
from nemo_helix import DefaultAioHttpClient
from nemo_helix import AsyncNeMoHelix


async def main() -> None:
    async with AsyncNeMoHelix(
        base_url="http://nemo.test",
        http_client=DefaultAioHttpClient(),
    ) as client:
        page = await client.workspaces.list()
        print(page.data)


asyncio.run(main())
```

## Using Types

Nested request parameters are [TypedDicts](https://docs.python.org/3/library/typing.html#typing.TypedDict). Responses are [Pydantic models](https://docs.pydantic.dev) which also provide helper methods for things like:

- Serializing back into JSON, `model.to_json()`
- Converting to a dictionary, `model.to_dict()`

Typed requests and responses provide autocomplete and documentation within your editor. If you would like to see type errors in VS Code to help catch bugs, set `python.analysis.typeCheckingMode` to `basic`.

## Pagination

List methods in the NeMo Helix API are paginated.

This library provides auto-paginating iterators with each list response, so you do not have to request successive pages manually:

```python
from nemo_helix import NeMoHelix

client = NeMoHelix(
    base_url="http://nemo.test",
)

all_jobs = []
# Automatically fetches more pages as needed.
for job in client.jobs.list(workspace="my-workspace"):
    # Do something with job here
    all_jobs.append(job)
print(all_jobs)
```

Or, asynchronously:

```python
import asyncio
from nemo_helix import AsyncNeMoHelix

client = AsyncNeMoHelix(
    base_url="http://nemo.test",
)

async def main() -> None:
    all_jobs = []
    # Iterate through items across all pages, issuing requests as needed.
    async for job in client.jobs.list(workspace="my-workspace"):
        all_jobs.append(job)
    print(all_jobs)


asyncio.run(main())
```

Alternatively, you can use the `.has_next_page()`, `.next_page_info()`, or `.get_next_page()` methods for more granular control working with pages:

```python
first_page = await client.jobs.list(workspace="my-workspace")
if first_page.has_next_page():
    print(f"will fetch next page using these details: {first_page.next_page_info()}")
    next_page = await first_page.get_next_page()
    print(f"number of items we just fetched: {len(next_page.data)}")

# Remove `await` for non-async usage.
```

Or just work directly with the returned data:

```python
first_page = await client.jobs.list(workspace="my-workspace")
for job in first_page.data:
    print(job.id)

# Remove `await` for non-async usage.
```

## Nested Parameters

Nested parameters are dictionaries, typed using `TypedDict`, for example:

```python
from nemo_helix import NeMoHelix

client = NeMoHelix(
    base_url="http://nemo.test",
)

guardrail_config = client.guardrail.configs.create(
    workspace="my-workspace",
    name="self-check-input",
    data={
        "models": [
            {
                "type": "main",
                "engine": "nim",
                "model": "default/mock-llm",
            }
        ],
        "rails": {"input": {"flows": ["self check input"]}},
        "prompts": [
            {
                "task": "self_check_input",
                "content": "Should this user message be blocked? {{ user_input }}",
            }
        ],
    },
    description="Self-check input rail",
)
print(guardrail_config.data)
```

## Handling Errors

The library raises errors when it cannot connect to the API or when the API returns a non-success status code.

When the library cannot connect to the API (for example, due to network connection problems or a timeout), it raises a subclass of `nemo_helix.APIConnectionError`.

When the API returns a non-success status code (that is, 4xx or 5xx
response), it raises a subclass of `nemo_helix.APIStatusError`, containing `status_code` and `response` properties.

All errors inherit from `nemo_helix.APIError`.

```python
import nemo_helix
from nemo_helix import NeMoHelix

client = NeMoHelix()

try:
    client.workspaces.list()
except nemo_helix.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
except nemo_helix.RateLimitError as e:
    print("A 429 status code was received; we should back off a bit.")
except nemo_helix.APIStatusError as e:
    print("Another non-200-range status code was received")
    print(e.status_code)
    print(e.response)
```

Error codes are as follows:

| Status Code | Error Type                 |
| ----------- | -------------------------- |
| 400         | `BadRequestError`          |
| 401         | `AuthenticationError`      |
| 403         | `PermissionDeniedError`    |
| 404         | `NotFoundError`            |
| 422         | `UnprocessableEntityError` |
| 429         | `RateLimitError`           |
| >=500       | `InternalServerError`      |
| N/A         | `APIConnectionError`       |

## Retries

Certain errors are automatically retried 2 times by default, with a short exponential backoff.
Connection errors (for example, due to a network connectivity problem), 408 Request Timeout, 409 Conflict,
429 Rate Limit, and >=500 Internal errors are all retried by default.

You can use the `max_retries` option to configure or disable retry settings:

```python
from nemo_helix import NeMoHelix

# Configure the default for all requests:
client = NeMoHelix(
    base_url="http://nemo.test",
    # default is 2
    max_retries=0,
)

# Or, configure per-request:
client.with_options(max_retries=5).workspaces.list()
```

## Timeouts

By default, requests time out after 1 minute. You can configure this with a `timeout` option,
which accepts a float or an [`httpx.Timeout`](https://www.python-httpx.org/advanced/timeouts/#fine-tuning-the-configuration) object:

```python
from nemo_helix import NeMoHelix

# Configure the default for all requests:
client = NeMoHelix(
    base_url="http://nemo.test",
    # 20 seconds (default is 1 minute)
    timeout=20.0,
)

# More granular control:
client = NeMoHelix(
    timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
)

# Override per-request:
client.with_options(timeout=5.0).workspaces.list()
```

On timeout, an `APITimeoutError` is thrown.

Note that requests that time out are [retried twice by default](#retries).

## Advanced Usage

### Logging

We use the standard library [`logging`](https://docs.python.org/3/library/logging.html) module.

You can enable logging by setting the environment variable `NHX_LOG` to `info`.

```shell
$ export NHX_LOG=info
```

Or to `debug` for more verbose logging.

#### How to Tell Whether `None` Means `null` or Missing

In an API response, a field may be explicitly `null`, or missing entirely; in either case, its value is `None` in this library. You can differentiate the two cases with `.model_fields_set`:

```py
if response.my_field is None:
  if 'my_field' not in response.model_fields_set:
    print('Got json like {}, without a "my_field" key present at all.')
  else:
    print('Got json like {"my_field": null}.')
```

### Accessing Raw Response Data (e.g. Headers)

You can access the "raw" response object by prefixing `.with_raw_response.` to any HTTP method call, for example:

```py
from nemo_helix import NeMoHelix

client = NeMoHelix(base_url="http://nemo.test")
response = client.workspaces.with_raw_response.list()
print(response.headers.get('X-My-Header'))

workspace = response.parse()  # get the object that `workspaces.list()` would have returned
print(workspace.id)
```

These methods return an `APIResponse` object.

The async client returns an `AsyncAPIResponse` with the same structure, the only difference being `await`able methods for reading the response content.

#### `.with_streaming_response`

The above interface eagerly reads the full response body when you make the request, which may not always be what you want.

To stream the response body, use `.with_streaming_response` instead, which requires a context manager and only reads the response body once you call `.read()`, `.text()`, `.json()`, `.iter_bytes()`, `.iter_text()`, `.iter_lines()` or `.parse()`. In the async client, these are async methods.

```python
with client.workspaces.with_streaming_response.list() as response:
    print(response.headers.get("X-My-Header"))

    for line in response.iter_lines():
        print(line)
```

The context manager is required so that the response will reliably be closed.

### Making Custom/Undocumented Requests

This library is typed for convenient access to the documented API.

If you need to access undocumented endpoints, params, or response properties, you can still use the library.

#### Undocumented Endpoints

To make requests to undocumented endpoints, you can make requests using `client.get`, `client.post`, and other
http verbs. The client will respect options (such as retries) when making this request.

```py
import httpx

response = client.post(
    "/foo",
    cast_to=httpx.Response,
    body={"my_param": True},
)

print(response.headers.get("x-foo"))
```

#### Undocumented Request Params

If you want to explicitly send an extra param, you can do so with the `extra_query`, `extra_body`, and `extra_headers` request
options.

#### Undocumented Response Properties

To access undocumented response properties, you can access the extra fields like `response.unknown_prop`. You
can also get all the extra fields on the Pydantic model as a dict with
[`response.model_extra`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_extra).

### Configuring the HTTP Client

You can directly override the [httpx client](https://www.python-httpx.org/api/#client) to customize it for your use case, including:

- Support for [proxies](https://www.python-httpx.org/advanced/proxies/)
- Custom [transports](https://www.python-httpx.org/advanced/transports/)
- Additional [advanced](https://www.python-httpx.org/advanced/clients/) functionality

```python
import httpx
from nemo_helix import NeMoHelix, DefaultHttpxClient

client = NeMoHelix(
    base_url="http://nemo.test",
    http_client=DefaultHttpxClient(
        proxy="http://my.test.proxy.example.com",
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)
```

You can also customize the client on a per-request basis by using `with_options()`:

```python
client.with_options(http_client=DefaultHttpxClient(...))
```

### Managing HTTP Resources

By default the library closes underlying HTTP connections whenever the client is [garbage collected](https://docs.python.org/3/reference/datamodel.html#object.__del__). You can manually close the client using the `.close()` method if desired, or use a context manager that closes when exiting.

```py
from nemo_helix import NeMoHelix

with NeMoHelix() as client:
  # make requests here
  ...

# HTTP client is now closed
```

<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Versioning

This package generally follows [SemVer](https://semver.org/spec/v2.0.0.html) conventions, while it might release certain backwards-incompatible changes as minor versions:

1. Changes that only affect static types, without breaking runtime behavior.
2. Changes to library internals which are technically public but not intended or documented for external use. Open a GitHub issue to let us know if you are relying on such internals.
3. Changes that we do not expect to impact the vast majority of users in practice.

We take backwards-compatibility seriously and work hard to ensure you can rely on a smooth upgrade experience.

We welcome your feedback; please contact us with questions, bugs, or suggestions.

### Determining the Installed Version

If you've upgraded to the latest version but can't find any new features you were expecting, your Python environment is likely still using an older version.

You can determine the version that is being used at runtime with:

```py
import nemo_helix
print(nemo_helix.__version__)
```

## Requirements

Python 3.11 or higher.