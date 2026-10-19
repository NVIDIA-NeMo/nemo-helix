"""Serve a model entity with vLLM through Helix Deployments, so the inference gateway
exposes it like any other model. Mirrors step 8 of the Helix LoRA tutorial.

One deployment serves the untuned base (arm B) and every LoRA adapter trained on
that base (arm C), so pass --lora-rank. vLLM returns next-token logprobs on
/completions, which is what the decision-model readout needs.

    python training/deploy_model.py --model qwen3.5-4b-base --name meridian-qwen --lora-rank 16
    python training/deploy_model.py --model nemotron-nano-9b-v2 --name meridian-nemotron --lora-rank 16
      # base: default/qwen3.5-4b-base, adapter: default--meridian-router-qwen3.5-4b-seed29,
      # both through the provider route of deployment meridian-qwen
"""

from __future__ import annotations

import argparse
import os
import time

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.models.types import (
    ContainerExecutorConfig,
    CreateModelDeploymentConfigRequest,
    CreateModelDeploymentRequest,
    Engine,
    ModelDeploymentConfigModelSpec,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model entity name in the workspace")
    ap.add_argument("--name", required=True, help="deployment name; becomes the gateway provider name")
    ap.add_argument("--workspace", default="default")
    ap.add_argument("--image", default="docker.io/vllm/vllm-openai")
    ap.add_argument("--tag", default="v0.22.1")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--max-model-len", default="4096")
    ap.add_argument("--lora-rank", type=int, default=0, help="serve LoRA adapters on this base (sets lora_enabled and --max-lora-rank); 0 = plain model")
    ap.add_argument("--base-url", default=os.environ.get("NHX_BASE_URL", "http://localhost:8080"))
    ap.add_argument("--timeout-minutes", type=int, default=40)
    args = ap.parse_args()

    # Remote clusters with auth: export NHX_ACCESS_TOKEN=$(nemo auth token)
    client = NemoClient(base_url=args.base_url, workspace=args.workspace, auth=os.environ.get("NHX_ACCESS_TOKEN") or None)
    cfg_name = f"{args.name}-cfg"
    try:
        cfg = client.models.create_deployment_config(body=CreateModelDeploymentConfigRequest(
            name=cfg_name,
            engine=Engine.VLLM,
            model_spec=ModelDeploymentConfigModelSpec(model_namespace=args.workspace, model_name=args.model, lora_enabled=args.lora_rank > 0),
            executor_config=ContainerExecutorConfig(
                gpu=args.gpu, image_name=args.image, image_tag=args.tag,
                additional_args=["--max-model-len", args.max_model_len, "--dtype", "bfloat16"] + (["--max-lora-rank", str(args.lora_rank)] if args.lora_rank else []),
            ),
        )).data()
        print("created deployment config", cfg.name)
    except ConflictError:
        print("deployment config exists:", cfg_name)
    try:
        dep = client.models.create_deployment(body=CreateModelDeploymentRequest(name=args.name, config=cfg_name)).data()
        print("created deployment", dep.name, dep.status)
    except ConflictError:
        print("deployment exists:", args.name)

    deadline = time.time() + args.timeout_minutes * 60
    while time.time() < deadline:
        time.sleep(15)
        dep = client.models.get_deployment(name=args.name).data()
        print("status:", dep.status)
        if dep.status in ("RUNNING", "READY"):
            ok = client.models.wait_for_gateway(args.name, workspace=args.workspace, timeout=120)
            print("gateway ready:", ok)
            print(f"base model served as {args.workspace}/{args.model}; adapters as {args.workspace}--<adapter output name>  (provider: {args.name})")
            print(f"check: nemo inference providers get {args.name} -f json | jq -r '.served_models[].served_model_name'")
            return
        if dep.status in ("FAILED", "ERROR", "TERMINATED"):
            raise SystemExit(f"deployment failed: {dep.status}")
    raise SystemExit("timed out waiting for deployment")


if __name__ == "__main__":
    main()
