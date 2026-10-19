"""Register the base model the post-training plugin fine-tunes and the untuned-model deployment uses.

Mirrors the "Import and Fine-Tune Private Hugging Face Models" tutorial in the
Helix docs: a Hugging Face-backed fileset, then a model entity that points to it.
The platform downloads the weights when a job or deployment needs them.

    python training/register_base_model.py                    # Qwen/Qwen3.5-4B-Base -> default/qwen3.5-4b-base
    python training/register_base_model.py --repo Qwen/Qwen3.5-0.8B-Base --name qwen3.5-0.8b-base
"""

from __future__ import annotations

import argparse
import os
import time

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.files.storage_config import HuggingfaceStorageConfig
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.models.types import CreateModelEntityRequest, UpdateModelEntityRequest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--name", default="qwen3.5-4b-base")
    ap.add_argument("--workspace", default="default")
    ap.add_argument("--base-url", default=os.environ.get("NHX_BASE_URL", "http://localhost:8080"))
    args = ap.parse_args()

    # Remote clusters with auth: export NHX_ACCESS_TOKEN=$(nemo auth token)
    client = NemoClient(base_url=args.base_url, workspace=args.workspace, auth=os.environ.get("NHX_ACCESS_TOKEN") or None)
    storage = HuggingfaceStorageConfig(repo_id=args.repo, repo_type="model")
    try:
        fs = client.files.create_fileset(body=CreateFilesetRequest(name=args.name, description=f"Hugging Face model {args.repo}", storage=storage, cache=True)).data()
        print("created fileset", fs.name)
    except ConflictError:
        print("fileset exists:", args.name)
    try:
        model = client.models.create_model(body=CreateModelEntityRequest(name=args.name, fileset=f"{args.workspace}/{args.name}", trust_remote_code=False)).data()
        print("created model entity", model.name)
    except ConflictError:
        model = client.models.update_model(name=args.name, body=UpdateModelEntityRequest(fileset=f"{args.workspace}/{args.name}", trust_remote_code=False)).data()
        print("model entity exists, fileset updated:", model.name)

    deadline = time.time() + 600
    while not model.spec and time.time() < deadline:
        time.sleep(10)
        model = client.models.get_model(name=args.name).data()
    print("model spec resolved:", bool(model.spec))
    print(f"use in training/job.json as model.name: {args.workspace}/{args.name}")


if __name__ == "__main__":
    main()
