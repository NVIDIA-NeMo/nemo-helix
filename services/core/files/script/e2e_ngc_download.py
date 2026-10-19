#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to test NGC storage backend integration with the filesets API.

This tests that the files service can:
1. Create a secret with NGC API key
2. Create a fileset with NGC storage backend referencing the secret
3. List files from an NGC resource
4. Download files from NGC

Prerequisites:
- NGC API key (via --api-key flag or NGC_CLI_API_KEY environment variable)
- Access to NVIDIA NGC resources

Usage:
    python script/e2e_ngc_download.py [--api-key <key>] [--base-url http://localhost:8080]

Example:
    python script/e2e_ngc_download.py --api-key your-key-here

    # Or using environment variable
    export NGC_CLI_API_KEY=your-key-here
    python script/e2e_ngc_download.py
"""

import argparse
import logging
import os
import sys

import httpx
from nhx.common.entities import DEFAULT_WORKSPACE

logger = logging.getLogger(__name__)

# Test configuration
WORKSPACE = DEFAULT_WORKSPACE
SECRET_NAME = "ngc-e2e-api-key"
FILESET_NAME = "ngc-e2e-quickstart"


def test_ngc_integration(base_url: str, api_key: str | None = None):
    """Test NGC storage backend integration with filesets API."""
    print("=" * 70)
    print("NGC Storage Backend Integration Test (v2 API)")
    print(f"Base URL: {base_url}")
    print(f"Workspace: {WORKSPACE}")
    print("=" * 70)

    # Get API key from argument or environment
    if not api_key:
        api_key = os.environ.get("NGC_CLI_API_KEY")
    if not api_key:
        print("✗ Error: NGC API key required. Use --api-key or set NGC_CLI_API_KEY")
        return False

    client = httpx.Client(timeout=60.0)
    results = {"passed": 0, "failed": 0, "tests": []}

    # Test 1: Create secret with NGC API key
    print("\n=== Test 1: Create secret with NGC API key ===")
    secret_data = {
        "name": SECRET_NAME,
        "data": api_key,
    }
    response = client.post(
        f"{base_url}/v2/workspaces/{WORKSPACE}/secrets",
        json=secret_data,
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 201:
        secret = response.json()
        print(f"✓ Created secret: {secret['name']}")
        results["passed"] += 1
        results["tests"].append(("Create NGC API key secret", "PASS"))
    elif response.status_code == 409:
        print("✓ Secret already exists, continuing...")
        results["passed"] += 1
        results["tests"].append(("Create NGC API key secret", "PASS (exists)"))
    else:
        print(f"✗ Failed: {response.text}")
        results["failed"] += 1
        results["tests"].append(("Create NGC API key secret", "FAIL"))
        print_summary(results)
        return False

    # Test 2: Create fileset with NGC storage backend
    print("\n=== Test 2: Create NGC fileset ===")
    storage_config = {
        "type": "ngc",
        "org": "nvidian",
        "team": "nemo-llm",
        "resource": "nemo-helix-quickstart",
        "api_key_secret": SECRET_NAME,
    }

    fileset_data = {
        "name": FILESET_NAME,
        "description": "Test fileset for NGC resource",
        "storage": storage_config,
    }
    response = client.post(
        f"{base_url}/v2/workspaces/{WORKSPACE}/filesets",
        json=fileset_data,
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        fileset = response.json()
        print(f"✓ Created NGC fileset: {fileset['name']}")
        print(f"  Storage type: {fileset.get('storage', {}).get('type')}")
        results["passed"] += 1
        results["tests"].append(("Create NGC fileset", "PASS"))
    else:
        print(f"✗ Failed: {response.text}")
        results["failed"] += 1
        results["tests"].append(("Create NGC fileset", "FAIL"))
        print_summary(results)
        return False

    # Test 3: List files in NGC resource
    print("\n=== Test 3: List files from NGC resource ===")
    response = client.get(f"{base_url}/v2/workspaces/{WORKSPACE}/filesets/{FILESET_NAME}/files")
    print(f"Status: {response.status_code}")
    files = []
    if response.status_code == 200:
        files_data = response.json()
        files = files_data.get("files", [])
        print(f"✓ Listed {len(files)} files from NGC resource")
        if files:
            print("\nFirst few files:")
            for file_info in files[:5]:
                size_mb = file_info["size"] / (1024 * 1024)
                print(f"  - {file_info['path']} ({size_mb:.2f} MB)")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
        results["passed"] += 1
        results["tests"].append(("List NGC files", "PASS"))
    else:
        print(f"✗ Failed: {response.text}")
        results["failed"] += 1
        results["tests"].append(("List NGC files", "FAIL"))
        print_summary(results)
        return False

    # Test 4: Download a file from NGC
    if files:
        print("\n=== Test 4: Download file from NGC ===")
        # Pick a small file if possible, otherwise use the first one
        test_file = min(files, key=lambda f: f["size"]) if len(files) > 1 else files[0]
        file_path = test_file["path"]
        file_size = test_file["size"]
        print(f"Downloading: {file_path} ({file_size / (1024 * 1024):.2f} MB)")

        try:
            with client.stream(
                "GET",
                f"{base_url}/v2/workspaces/{WORKSPACE}/filesets/{FILESET_NAME}/-/{file_path}",
            ) as response:
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    # Stream and count bytes
                    bytes_downloaded = 0
                    chunk_count = 0
                    for chunk in response.iter_bytes(chunk_size=64 * 1024):
                        bytes_downloaded += len(chunk)
                        chunk_count += 1
                        if chunk_count % 100 == 0:
                            progress = (bytes_downloaded / file_size) * 100
                            print(
                                f"  Progress: {bytes_downloaded / (1024 * 1024):.2f} MB / {file_size / (1024 * 1024):.2f} MB ({progress:.1f}%)"
                            )

                    print(f"✓ Downloaded {bytes_downloaded} bytes in {chunk_count} chunks")
                    if bytes_downloaded == file_size:
                        print("✓ File size matches expected size")
                    else:
                        print(f"⚠ Warning: Downloaded size ({bytes_downloaded}) doesn't match expected ({file_size})")

                    results["passed"] += 1
                    results["tests"].append(("Download NGC file", "PASS"))
                else:
                    print(f"✗ Failed: {response.text}")
                    results["failed"] += 1
                    results["tests"].append(("Download NGC file", "FAIL"))
        except Exception as e:
            logger.exception("Failed to download file")
            print(f"✗ Download failed: {e}")
            results["failed"] += 1
            results["tests"].append(("Download NGC file", "FAIL - Exception"))

    # Test 5: Test byte range request
    if files:
        print("\n=== Test 5: Test byte range request ===")
        test_file = files[0]
        file_path = test_file["path"]
        file_size = test_file["size"]

        # Request first 1KB
        range_start = 0
        range_end = min(1023, file_size - 1)  # 1KB or file size, whichever is smaller
        print(f"Requesting bytes {range_start}-{range_end} of {file_path}")

        try:
            response = client.get(
                f"{base_url}/v2/workspaces/{WORKSPACE}/filesets/{FILESET_NAME}/-/{file_path}",
                headers={"Range": f"bytes={range_start}-{range_end}"},
            )
            print(f"Status: {response.status_code}")
            if response.status_code == 206:  # Partial Content
                bytes_received = len(response.content)
                expected_bytes = range_end - range_start + 1
                print(f"✓ Received {bytes_received} bytes (expected {expected_bytes})")
                if bytes_received == expected_bytes:
                    print("✓ Byte range request successful")
                    results["passed"] += 1
                    results["tests"].append(("Byte range request", "PASS"))
                else:
                    print("⚠ Warning: Size mismatch")
                    results["failed"] += 1
                    results["tests"].append(("Byte range request", "FAIL - Size mismatch"))
            else:
                print(f"✗ Expected 206 Partial Content, got {response.status_code}")
                results["failed"] += 1
                results["tests"].append(("Byte range request", "FAIL"))
        except Exception as e:
            logger.exception("Byte range request failed")
            print(f"✗ Byte range request failed: {e}")
            results["failed"] += 1
            results["tests"].append(("Byte range request", "FAIL - Exception"))

    # Cleanup
    cleanup(client, base_url)
    client.close()
    print_summary(results)
    return results["failed"] == 0


def cleanup(client: httpx.Client, base_url: str):
    """Clean up test resources."""
    print("\n=== Cleanup: Delete test resources ===")

    # Delete fileset
    response = client.delete(f"{base_url}/v2/workspaces/{WORKSPACE}/filesets/{FILESET_NAME}")
    if response.status_code == 200:
        print(f"✓ Deleted test fileset: {FILESET_NAME}")
    else:
        print(f"⚠ Warning: Could not delete fileset: {response.status_code}")

    # Delete secret
    response = client.delete(f"{base_url}/v2/workspaces/{WORKSPACE}/secrets/{SECRET_NAME}")
    if response.status_code == 200:
        print(f"✓ Deleted test secret: {SECRET_NAME}")
    else:
        print(f"⚠ Warning: Could not delete secret: {response.status_code}")


def print_summary(results):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for test_name, status in results["tests"]:
        symbol = "✓" if "PASS" in status else "✗"
        print(f"{symbol} {test_name}: {status}")

    print(f"\nTotal: {results['passed']} passed, {results['failed']} failed")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test NGC storage backend integration with filesets API (v2)")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080",
        help="Base URL of the files service (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--api-key",
        help="NGC API key (if not provided, will use NGC_CLI_API_KEY environment variable)",
    )

    args = parser.parse_args()
    success = test_ngc_integration(args.base_url, args.api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
