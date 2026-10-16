# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anonymous tag → digest lookups against registry v2."""

import httpx
import pytest
from nemo_evals.harbor.remote.registry import ImageDigestUnavailableError, ImageName, resolve_image_digest

DIGEST = "sha256:" + "d" * 64


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_pinned_reference_is_returned_without_a_lookup():
    ref = f"registry.invalid/x@{DIGEST}"
    async with client(lambda request: pytest.fail("no lookup expected")) as http:
        assert await resolve_image_digest(ref, http_client=http) == ref


async def test_docker_hub_library_tag_uses_the_challenge_token():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "auth.docker.io":
            assert request.url.params["scope"] == "repository:library/ubuntu:pull"
            return httpx.Response(200, json={"access_token": "t"})
        if request.headers.get("authorization") == "Bearer t":
            return httpx.Response(200, headers={"docker-content-digest": DIGEST})
        challenge = 'Bearer realm="https://auth.docker.io/token",service="registry.docker.io",scope="repository:library/ubuntu:pull"'
        return httpx.Response(401, headers={"www-authenticate": challenge})

    async with client(handler) as http:
        assert await resolve_image_digest("ubuntu:24.04", http_client=http) == f"ubuntu@{DIGEST}"
    assert seen[0] == "https://registry-1.docker.io/v2/library/ubuntu/manifests/24.04"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404),
        httpx.Response(401, headers={"www-authenticate": 'Basic realm="private"'}),
        httpx.Response(200, headers={"docker-content-digest": "md5:nope"}),
    ],
)
async def test_unresolvable_tags_raise(response):
    async with client(lambda request: response) as http:
        with pytest.raises(ImageDigestUnavailableError):
            await resolve_image_digest("registry.invalid/team/x:1", http_client=http)


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("ubuntu", ("docker.io", "library/ubuntu", "latest")),
        ("team/app:2", ("docker.io", "team/app", "2")),
        ("localhost:5000/app", ("localhost:5000", "app", "latest")),
        ("public.ecr.aws/d3j8x8q7/swe:kh77", ("public.ecr.aws", "d3j8x8q7/swe", "kh77")),
    ],
)
def test_image_names_split_like_docker(image, expected):
    ref = ImageName.parse(image)
    assert (ref.registry, ref.repository, ref.tag) == expected
