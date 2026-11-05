# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image references in training rows.

A row names an image the way a chat request to the inference gateway does, so one
dataset trains, evaluates, and serves unchanged:

- ``fileset://<workspace>/<fileset>#<path>`` (or ``fileset://<fileset>#<path>`` in the
  job's workspace), resolved with the job creator's permissions.
- An inline base64 ``data:`` URL.

``http(s)`` URLs are rejected: training nodes may have no internet access, and a URL
can change after the job reads it, so the dataset would not be reproducible.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from urllib.parse import urlsplit

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoClientError, NotFoundError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.refs import FILESET_REF_PATTERN
from nhx.customization_common.schemas.file_io import FileSetRef

FILESET_IMAGE_PREFIX = "fileset://"
MAX_IMAGE_BYTES = 25 * 1024 * 1024
IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "gif", "webp"})
ACCEPTED_IMAGE_FORMS = (
    "Use 'fileset://<workspace>/<fileset>#<path>' (the workspace defaults to the job's) or a base64 data: URL."
)

_FILESET_REF = re.compile(FILESET_REF_PATTERN)
_CHUNK_BYTES = 1024 * 1024


class ImageReferenceError(ValueError):
    """An image URL in a dataset row cannot be used for training."""


@dataclass(frozen=True)
class FilesetImageRef:
    """``fileset://[workspace/]fileset#path``. ``workspace`` is None when the ref omits it."""

    workspace: str | None
    fileset: str
    path: str


def parse_fileset_image_ref(url: str) -> FilesetImageRef:
    """Parse a ``fileset://`` image URL. The path must stay inside the fileset."""
    ref = url.removeprefix(FILESET_IMAGE_PREFIX)
    location, _, path = ref.partition("#")
    workspace: str | None = None
    if "/" in location:
        workspace, _, fileset = location.partition("/")
    else:
        fileset = location
    if not _FILESET_REF.fullmatch(f"{workspace or '_'}/{fileset}#{path}"):
        raise ImageReferenceError(f"Invalid fileset image reference {url!r}. {ACCEPTED_IMAGE_FORMS}")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise ImageReferenceError(f"Fileset image reference {url!r} must name a file inside the fileset.")
    basename = path.rpartition("/")[2]
    if "." not in basename or basename.rpartition(".")[2].lower() not in IMAGE_EXTENSIONS:
        raise ImageReferenceError(
            f"Unsupported image file type for {url!r}. Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}."
        )
    return FilesetImageRef(workspace=workspace, fileset=fileset, path=path)


def check_image_url(url: str) -> None:
    """Reject an image URL training cannot resolve. Training does not read the URL here.

    A legacy path relative to the dataset passes: the trainer resolves it against the
    downloaded dataset.
    """
    if url.startswith("data:"):
        return
    if url.startswith(FILESET_IMAGE_PREFIX):
        parse_fileset_image_ref(url)
        return
    scheme = urlsplit(url).scheme.lower()
    if scheme in ("http", "https"):
        raise ImageReferenceError(
            f"Image URL {url!r} is not accepted: training does not download {scheme} URLs. {ACCEPTED_IMAGE_FORMS}"
        )
    # ``C:\...`` parses with a one-letter scheme, so it is a path too.
    if scheme and len(scheme) > 1 and scheme != "file":
        raise ImageReferenceError(f"Image URL {url!r} is not accepted. {ACCEPTED_IMAGE_FORMS}")
    if scheme or Path(url).is_absolute():
        raise ImageReferenceError(
            f"Image URL {url!r} is a host path, which training cannot read. {ACCEPTED_IMAGE_FORMS}"
        )


class FilesetImageResolver:
    """Map ``fileset://`` image refs onto local files for one training job.

    A ref into the job's dataset fileset maps onto the copy the download step wrote
    under ``dataset_root``. Any other file is downloaded into ``cache_dir`` with a
    client from ``client_factory``; a task client acts for the job creator, so Files
    applies their permissions. The client is created on the first download.
    """

    def __init__(
        self,
        *,
        job_workspace: str,
        dataset_fileset: str | None,
        dataset_root: Path,
        cache_dir: Path,
        client_factory: Callable[[], NemoClient],
    ) -> None:
        self._job_workspace = job_workspace
        self._dataset = FileSetRef.model_validate(dataset_fileset) if dataset_fileset else None
        self._dataset_root = dataset_root.resolve()
        self._cache_dir = cache_dir
        self._client_factory = client_factory
        self._client: NemoClient | None = None
        self._resolved: dict[tuple[str, str, str], Path] = {}

    def __enter__(self) -> FilesetImageResolver:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def local_path(self, url: str) -> Path:
        """The local file a ``fileset://`` image URL names, downloading it when needed."""
        ref = parse_fileset_image_ref(url)
        key = (ref.workspace or self._job_workspace, ref.fileset, ref.path)
        if key not in self._resolved:
            self._resolved[key] = self._dataset_copy(*key) or self._download(*key)
        return self._resolved[key]

    def _dataset_copy(self, workspace: str, fileset: str, path: str) -> Path | None:
        dataset = self._dataset
        if dataset is None or (dataset.workspace or self._job_workspace, dataset.name) != (workspace, fileset):
            return None
        # The download step strips the dataset's directory prefix, if it has one.
        if dataset.path is not None:
            if not path.startswith(dataset.path):
                return None
            path = path.removeprefix(dataset.path)
        local = (self._dataset_root / path).resolve()
        if not local.is_relative_to(self._dataset_root):
            raise ImageReferenceError(f"Image {workspace}/{fileset}#{path} resolves outside the dataset directory.")
        return local if local.is_file() else None

    def _download(self, workspace: str, fileset: str, path: str) -> Path:
        source = f"{workspace}/{fileset}#{path}"
        if self._client is None:
            self._client = self._client_factory()
        files_client = FilesClient.from_client(self._client)
        # A content-free name: the fileset path never becomes a local path.
        dest = self._cache_dir / f"{hashlib.sha256(source.encode()).hexdigest()[:24]}.{path.rpartition('.')[2].lower()}"
        if dest.is_file():
            return dest
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self._cache_dir, prefix=".fileset-image.", delete=False) as handle:
                temp_path = Path(handle.name)
                size = 0
                response = files_client.download_file(workspace=workspace, name=fileset, path=path)
                with response.stream(_CHUNK_BYTES) as chunks:
                    for chunk in chunks:
                        size += len(chunk)
                        if size > MAX_IMAGE_BYTES:
                            raise ImageReferenceError(
                                f"Image file {source} is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MiB."
                            )
                        handle.write(chunk)
            temp_path.replace(dest)
        except NotFoundError as exc:
            raise ImageReferenceError(f"Image file {source} not found.") from exc
        except NemoClientError as exc:
            raise ImageReferenceError(f"Could not download image file {source}: {exc}") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return dest
