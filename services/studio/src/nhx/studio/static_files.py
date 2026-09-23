# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SPA-aware static file serving for the Studio UI."""

import base64
import hashlib
import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

logger = logging.getLogger(__name__)

# Pattern to match any STUDIO_UI_* markers for cleanup
STUDIO_UI_MARKER_PATTERN = re.compile(r"STUDIO_UI_[A-Z_]+")

# Finds inline <script type="importmap">…</script> blocks so their content
# hash can be authorized in script-src without weakening the policy with
# 'unsafe-inline'. The DOTALL flag lets . span newlines; non-greedy capture
# avoids merging multiple script tags.
_INLINE_IMPORTMAP_PATTERN = re.compile(
    r'<script\s+type=["\']importmap["\']\s*>(.*?)</script>',
    re.DOTALL,
)


def _origin(url: str) -> str | None:
    """scheme://host[:port] of an absolute http(s) URL; None for empty/relative values."""
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return None


def _origins(urls: Iterable[str]) -> list[str]:
    """Unique origins of the absolute URLs in *urls*, preserving order."""
    result: list[str] = []
    for url in urls:
        origin = _origin(url)
        if origin is not None and origin not in result:
            result.append(origin)
    return result


def build_csp(
    connect_src_urls: Iterable[str] = (),
    frame_src_urls: Iterable[str] = (),
    script_src_urls: Iterable[str] = (),
) -> str:
    """Content-Security-Policy for the Studio UI, built from configured endpoints.

    script-src 'self' plus per-content SHA-256 hashes (appended at startup):
      Covers Studio JS chunks and plugin bundles (/plugin-ui/…) plus the inline
      <script type="importmap"> block that wires plugin bundles to their shared
      React vendor copy. Hashes are computed from the served HTML so the policy
      stays tight without relying on 'unsafe-inline'.

    Each configured absolute URL contributes its origin (scheme://host[:port])
    to the named directive so cross-origin deployments work: the OIDC issuer in
    connect-src (token endpoint fetches) and frame-src (silent-renew iframe), a
    cross-origin platform base URL in connect-src (API/plugin-manifest fetches)
    and script-src (dynamic import() of plugin bundles). Empty or relative URLs
    contribute nothing; with no arguments the policy is fully same-origin.
    """
    connect_src = " ".join(["'self'", *_origins(connect_src_urls)])
    script_src = " ".join(["'self'", *_origins(script_src_urls)])
    frame_origins = _origins(frame_src_urls)
    frame_src = " ".join(["'self'", *frame_origins]) if frame_origins else "'none'"
    return (
        "default-src 'self'; "
        f"script-src {script_src}; "
        "style-src 'self' 'unsafe-inline' https://webassets.nvidia.com; "
        "font-src 'self' https://webassets.nvidia.com https://brand-assets.cne.ngc.nvidia.com data:; "
        "img-src 'self' data: blob: https:; "
        f"connect-src {connect_src}; "
        f"frame-src {frame_src}; "
        "frame-ancestors 'none'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )


# Same-origin default (standard NHX deployment behind a single ingress).
DEFAULT_CSP = build_csp()


def _sha256_script_source(content: str) -> str:
    """CSP script-src token for an inline script's exact content.

    The browser hashes the verbatim text between <script>…</script>,
    including whitespace, so callers must pass the unmodified inner text.
    """
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    return f"'sha256-{base64.b64encode(digest).decode('ascii')}'"


def _augment_csp_for_inline_scripts(csp: str, html: str) -> str:
    """Add sha256 sources to script-src for each inline import-map script."""
    hashes = [_sha256_script_source(match.group(1)) for match in _INLINE_IMPORTMAP_PATTERN.finditer(html)]
    if not hashes:
        return csp
    # Inject the hashes into the script-src directive. Preserve the rest of
    # the policy verbatim so any other directive tweaks survive.
    addition = " " + " ".join(hashes)
    replaced, count = re.subn(
        r"(script-src[^;]*)",
        lambda m: m.group(1) + addition,
        csp,
        count=1,
    )
    return replaced if count else csp + "; script-src 'self'" + addition


class SPAStaticFiles(StaticFiles):
    """
    Static files handler with SPA (Single Page Application) support.

    This handler serves static files from the Vite build output and implements
    SPA routing by falling back to index.html for routes that don't match
    existing files. This enables client-side routing to work correctly.

    Features:
    - Serves static assets (JS, CSS, images, etc.) directly
    - Falls back to index.html for non-file routes (SPA routing)
    - Handles .html extension stripping for clean URLs
    - Injects runtime environment variables from platform config (pre-processed once at startup)
    - Attaches Content-Security-Policy header to HTML responses
    """

    def __init__(
        self,
        *args,
        env_replacements: dict[str, str] | None = None,
        csp_header: str | None = DEFAULT_CSP,
        **kwargs,
    ):
        """Initialize SPA static files handler.

        Args:
            env_replacements: Optional dict of STUDIO_UI_* markers to replacement values.
                              These will be applied to HTML and JS files once at startup.
            csp_header: Content-Security-Policy header value to attach to HTML responses.
                        Pass None to disable CSP (not recommended in production).
                        Defaults to DEFAULT_CSP.
        """
        super().__init__(*args, **kwargs)
        self._env_replacements = env_replacements or {}
        self._csp_header = csp_header
        # Cache for pre-processed file contents (path -> processed content)
        self._processed_cache: dict[str, str] = {}
        # Pre-process files that need env var replacement
        self._preprocess_files()
        # After preprocessing, derive per-response CSPs that authorize the
        # HTML's inline <script type="importmap"> blocks via SHA-256 hashes.
        self._csp_by_path: dict[str, str] = {}
        if self._csp_header:
            self._csp_by_path = self._compute_csp_by_path()

    def _compute_csp_by_path(self) -> dict[str, str]:
        """Map each HTML file to a CSP that authorizes its inline scripts.

        Uses the preprocessed cache when available (so STUDIO_UI_* markers
        are already replaced — that matters for hash computation if any
        marker lives inside an inline script) and falls back to reading the
        file from disk.
        """
        assert self._csp_header is not None
        result: dict[str, str] = {}
        directory = Path(str(self.directory))
        for html_path in directory.glob("**/*.html"):
            rel = str(html_path.relative_to(directory))
            content = self._processed_cache.get(rel)
            if content is None:
                try:
                    content = html_path.read_text(encoding="utf-8")
                except Exception:
                    continue
            augmented = _augment_csp_for_inline_scripts(self._csp_header, content)
            if augmented != self._csp_header:
                result[rel] = augmented
        return result

    def _apply_env_replacements(self, content: str) -> str:
        """Replace STUDIO_UI_* markers with actual values.

        First replaces known markers from env_replacements dict,
        then clears any remaining STUDIO_UI_* markers with empty strings.

        Args:
            content: File content to process

        Returns:
            Content with all STUDIO_UI_* markers replaced
        """
        # Replace known markers with their configured values
        for marker, value in self._env_replacements.items():
            content = content.replace(marker, value)

        # Clear any remaining STUDIO_UI_* markers (replace with empty string)
        # This prevents unmapped markers from appearing as literal text
        content = STUDIO_UI_MARKER_PATTERN.sub("", content)

        return content

    def _preprocess_files(self) -> None:
        """Pre-process HTML and JS files with env var replacements.

        Called once at startup to cache processed file contents.
        Only processes files that may contain STUDIO_UI_* markers.
        """
        if not self._env_replacements:
            logger.debug("No env replacements configured, skipping preprocessing")
            return

        directory = Path(str(self.directory))
        if not directory.exists():
            logger.warning(f"Static files directory does not exist: {directory}")
            return

        processed_count = 0
        for pattern in ["**/*.html", "**/*.js"]:
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        # Only cache if there are markers to replace
                        if STUDIO_UI_MARKER_PATTERN.search(content):
                            processed_content = self._apply_env_replacements(content)
                            # Store with relative path as key
                            rel_path = str(file_path.relative_to(directory))
                            self._processed_cache[rel_path] = processed_content
                            processed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to preprocess {file_path}: {e}")

        logger.info(f"Pre-processed {processed_count} files with env replacements")

    def _with_csp(self, response: Response, rel_path: str | None = None) -> Response:
        """Attach the CSP header to an HTML response if CSP is configured.

        When *rel_path* names an HTML file whose inline scripts were hashed
        at startup, attach the per-path augmented policy instead of the
        base one so script-src authorizes those inline scripts.
        """
        if self._csp_header:
            csp = self._csp_by_path.get(rel_path, self._csp_header) if rel_path else self._csp_header
            response.headers["Content-Security-Policy"] = csp
        return response

    async def get_response(self, path: str, scope: Scope) -> Response:
        """
        Override to implement SPA fallback routing and environment injection.

        1. Try to serve the requested file directly
        2. If not found and path doesn't have an extension, try adding .html
        3. If still not found, serve index.html for client-side routing
        4. For pre-processed files (HTML/JS with env vars), serve from cache
        """
        # Normalize path for cache lookup
        rel_path = path.lstrip("/")

        # Try the original path first
        try:
            response = await super().get_response(path, scope)
            if response.status_code != 404:
                # Check if this JS file was pre-processed
                if path.endswith(".js") and rel_path in self._processed_cache:
                    return Response(
                        content=self._processed_cache[rel_path],
                        media_type="application/javascript",
                    )
                # Check if this HTML file was pre-processed
                if path.endswith(".html"):
                    if rel_path in self._processed_cache:
                        return self._with_csp(
                            Response(
                                content=self._processed_cache[rel_path],
                                media_type="text/html",
                            ),
                            rel_path,
                        )
                    # Not preprocessed (no markers) — still needs CSP
                    return self._with_csp(response, rel_path)
                # Handle index.html when path is "." or "" (root directory request)
                # StaticFiles serves index.html for directory requests with path="."
                if rel_path in ("", ".", "/"):
                    if "index.html" in self._processed_cache:
                        return self._with_csp(
                            Response(
                                content=self._processed_cache["index.html"],
                                media_type="text/html",
                            ),
                            "index.html",
                        )
                    # index.html had no markers so wasn't preprocessed — still needs CSP
                    return self._with_csp(response, "index.html")
                return response
        except Exception:
            pass

        # If original path failed and doesn't have a file extension
        if not self._has_file_extension(path):
            # Try adding .html extension
            html_path = rel_path.rstrip("/") + ".html"
            full_path = Path(str(self.directory)) / html_path

            if full_path.is_file():
                return self._serve_file(html_path, full_path, "text/html")

            # Fall back to index.html for SPA routing
            index_path = Path(str(self.directory)) / "index.html"
            if index_path.is_file():
                return self._serve_file("index.html", index_path, "text/html")

        # Return the original response (likely 404)
        return await super().get_response(path, scope)

    def _serve_file(self, rel_path: str, file_path: Path, media_type: str) -> Response:
        """Serve a file, using cached pre-processed content if available.

        Args:
            rel_path: Relative path for cache lookup
            file_path: Full path to the file
            media_type: MIME type for the response

        Returns:
            Response with file content (pre-processed if it was cached)
        """
        # Use pre-processed cache if available
        if rel_path in self._processed_cache:
            response = Response(content=self._processed_cache[rel_path], media_type=media_type)
        else:
            # Otherwise read from disk (file has no markers to replace)
            content = file_path.read_text(encoding="utf-8")
            response = Response(content=content, media_type=media_type)

        if media_type == "text/html":
            return self._with_csp(response, rel_path)
        return response

    @staticmethod
    def _has_file_extension(path: str) -> bool:
        """Check if the path appears to have a file extension."""
        # Get the last component of the path
        basename = os.path.basename(path.rstrip("/"))
        # Check if it has a dot followed by characters (file extension)
        if "." in basename:
            parts = basename.rsplit(".", 1)
            # Make sure there's actually an extension (not just a hidden file)
            return len(parts) == 2 and len(parts[1]) > 0
        return False
