# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for HuggingFace Hub endpoint compatibility.

These tests verify that the files service HuggingFace-compatible endpoints
work correctly with the huggingface_hub client library.

Note: These tests are for the HuggingFace Hub API endpoints, not for
HuggingFace storage backends.
"""

from fastapi.testclient import TestClient
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, snapshot_download
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import FilesetOutput
from nhx.core.files.testing.utils import create_fileset


class TestHuggingFaceClientLibrary:
    """Test HuggingFace Hub client library compatibility with the files service."""

    def test_hf_hub_download_nested_files(self, client: NemoClient, fileset: FilesetOutput, tmp_path, hf_asgi_client):
        """Test downloading nested files using huggingface_hub client.

        This test:
        1. Uploads a nested directory structure using the NeMo Helix SDK
        2. Downloads all files using huggingface_hub's snapshot_download
        3. Verifies all files match the originals
        """
        # Upload nested directory structure using SDK
        test_files = {
            "README.md": b"# Test Model\n\nThis is a test model.",
            "config.json": b'{"version": "1.0", "type": "test"}',
            "data/train.txt": b"training data line 1\ntraining data line 2\ntraining data line 3",
            "data/test.txt": b"test data line 1\ntest data line 2",
            "data/validation.txt": b"validation data",
            "weights/model.bin": b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09" * 100,
            "tokenizer/vocab.txt": b"word1\nword2\nword3\nword4\nword5",
            "tokenizer/config.json": b'{"vocab_size": 5}',
        }

        # Upload all files
        for path, content in test_files.items():
            FilesClient.from_client(client).upload_file(
                content=content,
                path=path,
                name=fileset.name,
                workspace=fileset.workspace,
            )

        # Configure HuggingFace Hub to use our files service
        hf_endpoint = f"{client.base_url}/apis/files/v2/hf"
        repo_id = f"{fileset.workspace}/{fileset.name}"

        # Download all files using snapshot_download (model is the default repo_type)
        local_dir = tmp_path / "downloaded"
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            endpoint=hf_endpoint,
            token="service:test",  # Required but not validated in our implementation
        )

        # Verify all files were downloaded correctly
        for path, expected_content in test_files.items():
            downloaded_file = local_dir / path
            assert downloaded_file.exists(), f"File {path} was not downloaded"
            assert downloaded_file.read_bytes() == expected_content, f"Content mismatch for {path}"

    def test_hf_hub_download_single_file(self, client: NemoClient, fileset: FilesetOutput, tmp_path, hf_asgi_client):
        """Test downloading a single file using hf_hub_download."""
        test_content = b"This is a test file for single download"
        test_path = "single_file.txt"

        FilesClient.from_client(client).upload_file(
            content=test_content,
            path=test_path,
            name=fileset.name,
            workspace=fileset.workspace,
        )

        hf_endpoint = f"{client.base_url}/apis/files/v2/hf"
        repo_id = f"{fileset.workspace}/{fileset.name}"

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=test_path,
            local_dir=str(tmp_path),
            endpoint=hf_endpoint,
            token="service:test",
        )

        # Verify file content
        with open(local_path, "rb") as f:
            assert f.read() == test_content

    def test_hf_api_list_repo_files(self, client: NemoClient, fileset: FilesetOutput, hf_asgi_client):
        """Test listing repository files using HfApi."""
        test_files = {
            "file1.txt": b"content1",
            "dir/file2.txt": b"content2",
            "dir/subdir/file3.txt": b"content3",
        }

        for path, content in test_files.items():
            FilesClient.from_client(client).upload_file(
                content=content,
                path=path,
                name=fileset.name,
                workspace=fileset.workspace,
            )

        hf_endpoint = f"{client.base_url}/apis/files/v2/hf"
        repo_id = f"{fileset.workspace}/{fileset.name}"

        api = HfApi(endpoint=hf_endpoint, token="service:test")

        repo_info = api.repo_info(repo_id=repo_id)

        # Extract filenames from siblings
        listed_files = {sibling.rfilename for sibling in repo_info.siblings}
        assert listed_files == set(test_files.keys())

    def test_hf_hub_url_generates_valid_download_url(
        self, client: NemoClient, fileset: FilesetOutput, test_client: TestClient
    ):
        """Test that hf_hub_url generates a valid URL for file download.

        This test verifies that:
        1. hf_hub_url correctly constructs a download URL for our HF-compatible endpoint
        2. The generated URL can be used to download file content directly
        """
        test_content = b"Content for hf_hub_url test"
        test_path = "url_test_file.txt"

        FilesClient.from_client(client).upload_file(
            content=test_content,
            path=test_path,
            name=fileset.name,
            workspace=fileset.workspace,
        )

        hf_endpoint = f"{client.base_url}/apis/files/v2/hf"
        repo_id = f"{fileset.workspace}/{fileset.name}"

        # Generate the URL using hf_hub_url
        url = hf_hub_url(
            repo_id=repo_id,
            filename=test_path,
            endpoint=hf_endpoint,
        )

        # Verify the URL has the expected structure
        assert f"{fileset.workspace}/{fileset.name}" in url
        assert test_path in url
        assert "resolve" in url

        # Verify the URL can be used to download the file
        response = test_client.get(url, headers={"Authorization": "Bearer service:test"})
        assert response.status_code == 200
        assert response.content == test_content


class TestHfFileDownload:
    """Tests for /v2/hf/{workspace}/{name}/resolve/... endpoints."""

    def test_head_file_returns_metadata(self, client: NemoClient, test_client: TestClient, hf_auth_headers):
        """Test HEAD request returns correct headers."""
        with create_fileset(FilesClient.from_client(client)) as fileset:
            content = b"test content"
            FilesClient.from_client(client).upload_file(
                content=content,
                path="data.txt",
                name=fileset.name,
                workspace=fileset.workspace,
            )

            response = test_client.head(
                f"/apis/files/v2/hf/{fileset.workspace}/{fileset.name}/resolve/main/data.txt",
                headers=hf_auth_headers,
            )

            assert response.status_code == 200
            assert response.headers["Content-Length"] == str(len(content))
            assert "X-Repo-Commit" in response.headers
            assert "ETag" in response.headers
            assert response.headers["Accept-Ranges"] == "bytes"

    def test_revision_is_ignored(self, client: NemoClient, test_client: TestClient, hf_auth_headers):
        """Test that revision parameter is ignored (we don't version filesets)."""
        with create_fileset(FilesClient.from_client(client)) as fileset:
            FilesClient.from_client(client).upload_file(
                content=b"content",
                path="data.txt",
                name=fileset.name,
                workspace=fileset.workspace,
            )

            # Different revisions should all work
            for revision in [
                "main",
                "v1.0",
                "abc123def456abc123def456abc123def456abc1",
            ]:
                response = test_client.get(
                    f"/apis/files/v2/hf/{fileset.workspace}/{fileset.name}/resolve/{revision}/data.txt",
                    headers=hf_auth_headers,
                )
                assert response.status_code == 200

    def test_range_request(self, client: NemoClient, test_client: TestClient, hf_auth_headers):
        """Test Range header is respected."""
        with create_fileset(FilesClient.from_client(client)) as fileset:
            FilesClient.from_client(client).upload_file(
                content=b"0123456789",
                path="data.txt",
                name=fileset.name,
                workspace=fileset.workspace,
            )

            response = test_client.get(
                f"/apis/files/v2/hf/{fileset.workspace}/{fileset.name}/resolve/main/data.txt",
                headers={**hf_auth_headers, "Range": "bytes=0-4"},
            )

            assert response.status_code == 206
            assert response.content == b"01234"

    def test_file_not_found(self, client: NemoClient, test_client: TestClient, hf_auth_headers):
        """Test 404 for missing file."""
        with create_fileset(FilesClient.from_client(client)) as fileset:
            response = test_client.get(
                f"/apis/files/v2/hf/{fileset.workspace}/{fileset.name}/resolve/main/missing.txt",
                headers=hf_auth_headers,
            )
            assert response.status_code == 404

    def test_repo_not_found(self, test_client: TestClient, hf_auth_headers):
        """Test 404 for missing repo."""
        response = test_client.get(
            "/apis/files/v2/hf/nonexistent/missing/resolve/main/file.txt",
            headers=hf_auth_headers,
        )
        assert response.status_code == 404

    def test_service_principal_bearer_token(self, client: NemoClient, test_client: TestClient):
        """Test that service principal Bearer tokens work for HF endpoints.

        This verifies the HF_TOKEN=service:<name> authentication flow works,
        allowing huggingface-hub clients to authenticate via Bearer token.
        """
        with create_fileset(FilesClient.from_client(client)) as fileset:
            content = b"model weights"
            FilesClient.from_client(client).upload_file(
                content=content,
                path="model.bin",
                name=fileset.name,
                workspace=fileset.workspace,
            )

            response = test_client.get(
                f"/apis/files/v2/hf/{fileset.workspace}/{fileset.name}/resolve/main/model.bin",
                headers={"Authorization": "Bearer service:nim"},
            )

            assert response.status_code == 200
            assert response.content == content


class TestHfRepoInfo:
    """Tests for /v2/hf/api/models/... endpoints."""

    def test_get_repo_info_at_revision(self, client: NemoClient, test_client: TestClient, hf_auth_headers):
        """Test getting repository info with explicit revision."""
        with create_fileset(FilesClient.from_client(client)) as fileset:
            FilesClient.from_client(client).upload_file(
                content=b"content",
                path="file.txt",
                name=fileset.name,
                workspace=fileset.workspace,
            )

            # This endpoint is called by HfApi.model_info(repo_id, revision="main")
            response = test_client.get(
                f"/apis/files/v2/hf/api/models/{fileset.workspace}/{fileset.name}/revision/main",
                headers=hf_auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == f"{fileset.workspace}/{fileset.name}"
            assert data["modelId"] == f"{fileset.workspace}/{fileset.name}"
            assert len(data["sha"]) == 40
            assert len(data["siblings"]) == 1
            assert data["siblings"][0]["rfilename"] == "file.txt"

    def test_get_tree(self, client: NemoClient, test_client: TestClient, hf_auth_headers):
        """Test getting file tree."""
        with create_fileset(FilesClient.from_client(client)) as fileset:
            FilesClient.from_client(client).upload_file(
                content=b"content",
                path="file.txt",
                name=fileset.name,
                workspace=fileset.workspace,
            )

            response = test_client.get(
                f"/apis/files/v2/hf/api/models/{fileset.workspace}/{fileset.name}/tree/main",
                headers=hf_auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["path"] == "file.txt"
            assert data[0]["type"] == "file"
            assert "oid" in data[0]

    def test_paths_info(self, client: NemoClient, test_client: TestClient, hf_auth_headers):
        """Test paths-info endpoint."""
        with create_fileset(FilesClient.from_client(client)) as fileset:
            FilesClient.from_client(client).upload_file(
                content=b"content",
                path="exists.txt",
                name=fileset.name,
                workspace=fileset.workspace,
            )

            response = test_client.post(
                f"/apis/files/v2/hf/api/models/{fileset.workspace}/{fileset.name}/paths-info/main",
                json={"paths": ["exists.txt", "missing.txt"]},
                headers=hf_auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            # Only existing file should be returned
            assert len(data) == 1
            assert data[0]["path"] == "exists.txt"


class TestCommitHashConsistency:
    """Tests for commit hash and ETag stability."""

    def test_commit_hash_stable_for_same_fileset(self, client: NemoClient, test_client: TestClient, hf_auth_headers):
        """Test same fileset returns same commit hash."""
        with create_fileset(FilesClient.from_client(client)) as fileset:
            FilesClient.from_client(client).upload_file(
                content=b"content",
                path="file.txt",
                name=fileset.name,
                workspace=fileset.workspace,
            )

            response1 = test_client.head(
                f"/apis/files/v2/hf/{fileset.workspace}/{fileset.name}/resolve/main/file.txt",
                headers=hf_auth_headers,
            )
            response2 = test_client.head(
                f"/apis/files/v2/hf/{fileset.workspace}/{fileset.name}/resolve/main/file.txt",
                headers=hf_auth_headers,
            )

            assert response1.headers["X-Repo-Commit"] == response2.headers["X-Repo-Commit"]

    def test_etag_stable_for_same_file(self, client: NemoClient, test_client: TestClient, hf_auth_headers):
        """Test same file returns same ETag."""
        with create_fileset(FilesClient.from_client(client)) as fileset:
            FilesClient.from_client(client).upload_file(
                content=b"content",
                path="file.txt",
                name=fileset.name,
                workspace=fileset.workspace,
            )

            response1 = test_client.head(
                f"/apis/files/v2/hf/{fileset.workspace}/{fileset.name}/resolve/main/file.txt",
                headers=hf_auth_headers,
            )
            response2 = test_client.head(
                f"/apis/files/v2/hf/{fileset.workspace}/{fileset.name}/resolve/main/file.txt",
                headers=hf_auth_headers,
            )

            assert response1.headers["ETag"] == response2.headers["ETag"]
