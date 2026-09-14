<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fallback: deploy through the raw deployments API

Use this only when the platform predates the two fixes named in the skill: `nemo agents deploy --mode k8s` refuses the `openshell` executor with a 400, or the deployment reaches `running` but `nemo agents invoke` cannot reach the endpoint. This path creates the two deployments-plugin entities by hand and talks to the served agent through the OpenShell gateway port-forward. It skips the agents service, so the agent config must be fully resolved here: concrete model name and the in-cluster Inference Gateway URL.

Everything before Step 8 of the skill still applies: the image is built and reachable by the cluster, the platform is port-forwarded on 8080, and the OpenShell gateway on 18080.

## 1. Write the resolved agent config

Take the model name `nemo setup` chose (it is in the registered agent: `nemo agents get "$AGENT_NAME" | grep -A 3 default:`), then write a config the Fabric server can use without the agents service:

```bash
export MODEL_NAME=<model name from nemo models list>
export IGW_URL="http://${NMP_RELEASE}-api.${NMP_NS}.svc.cluster.local:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1"

cat > "$WORKDIR/agent-resolved.yaml" <<EOF
config_format: nemo-agents-spec-v1
name: $AGENT_NAME
description: Sandbox demo agent (raw deployments API path)
instructions:
  system:
    content: You are a concise assistant running inside a NeMo Platform sandbox. Answer in one or two sentences.
default_harness: deepagents
harnesses:
  deepagents:
    kind: deepagents
    settings:
      deepagents: {}
models:
  default:
    provider: nvidia
    model: $MODEL_NAME
    base_url: $IGW_URL
    api_key_env: NEMO_AGENTS_IGW_API_KEY
skills:
  paths: []
mcp:
  servers: {}
tools:
  blocked: []
environment:
  workspace: ./workspace
EOF
```

`api_key_env` and the `not-used` value below are the same placeholder binding the agents service injects on the normal path; the Inference Gateway ignores the value when platform auth is off.

## 2. Create the DeploymentConfig

The OpenShell backend runs `containers[0].command` inside the sandbox (it does not run the image entrypoint) and writes `configFiles` into the sandbox before starting it.

```bash
export BASE=http://localhost:8080/apis/deployments/v2/workspaces/default
python3 - "$WORKDIR/agent-resolved.yaml" "$AGENT_NAME" "$IMAGE_TAG" > "$WORKDIR/deployment-config.json" <<'EOF'
import json, sys
content = open(sys.argv[1]).read()
name, image = sys.argv[2], sys.argv[3]
print(json.dumps({
  "name": f"{name}-cfg",
  "configFiles": [{"path": "/tmp/nemo/agent.yaml", "content": content}],
  "containers": [{
    "name": "agent",
    "image": image,
    "command": ["/workspace/.venv/bin/python", "-m", "nemo_agents_plugin.fabric.server",
                "--agent-config", "/tmp/nemo/agent.yaml", "--host", "0.0.0.0", "--port", "8000"],
    "env": [
      {"name": "NMP_BASE_URL", "value": "http://" + "NMP_API_HOST_PLACEHOLDER" + ":8080"},
      {"name": "NMP_WORKSPACE", "value": "default"},
      {"name": "NEMO_AGENTS_IGW_API_KEY", "value": "not-used"}
    ],
    "ports": [{"containerPort": 8000, "name": "http"}],
    "readinessProbe": {"httpGet": {"path": "/health", "port": 8000}, "periodSeconds": 5, "failureThreshold": 12}
  }]
}, indent=2))
EOF
sed -i.bak "s#NMP_API_HOST_PLACEHOLDER#${NMP_RELEASE}-api.${NMP_NS}.svc.cluster.local#" "$WORKDIR/deployment-config.json"
curl -sf -X POST "$BASE/deployment-configs" -H 'content-type: application/json' \
  --data-binary @"$WORKDIR/deployment-config.json" | head -c 400
```

Expected: the created entity echoed back with `"name": "<agent>-cfg"`.

## 3. Create the Deployment on the openshell executor

```bash
curl -sf -X POST "$BASE/deployments" -H 'content-type: application/json' -d "{
  \"name\": \"$AGENT_NAME-sandbox\",
  \"deployment_config\": \"$AGENT_NAME-cfg\",
  \"executor\": \"openshell\"
}" | head -c 400
```

Poll until `READY`:

```bash
for i in $(seq 1 60); do
  status=$(curl -sf "$BASE/deployments/$AGENT_NAME-sandbox" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  echo "$status"
  [ "$status" = "READY" ] && break
  [ "$status" = "FAILED" ] && { curl -sf "$BASE/deployments/$AGENT_NAME-sandbox"; break; }
  sleep 5
done
curl -sf "$BASE/deployments/$AGENT_NAME-sandbox" | python3 -c 'import json,sys; print(json.load(sys.stdin)["endpoints"][0]["url"])'
```

Expected: `READY`, then a URL like `http://default--nmp-<hash>--http.openshell.localhost:8080/`. The part between `default--` and `--http` is the sandbox name for `openshell sandbox exec`.

## 4. Reach the agent through the gateway

The URL above is minted by the gateway and only resolves on the gateway host. Reach it through the gateway port-forward on 18080 and send the minted hostname as the `Host` header:

```bash
export MINTED_HOST=$(curl -sf "$BASE/deployments/$AGENT_NAME-sandbox" | python3 -c 'import json,sys; from urllib.parse import urlparse; print(urlparse(json.load(sys.stdin)["endpoints"][0]["url"]).netloc)')
curl -sf -H "Host: $MINTED_HOST" http://127.0.0.1:18080/health
curl -sf -H "Host: $MINTED_HOST" -H 'content-type: application/json' \
  http://127.0.0.1:18080/v1/chat/completions \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, where are you running?\"}]}"
```

Expected: `/health` returns 200 and the chat completion returns a non-empty assistant message. [UNVERIFIED] whether the gateway requires the `:8080` suffix in the `Host` value; try without it if routing returns 404.

Continue with Step 9 of the skill (egress proof) using the sandbox name derived above.

## 5. Cleanup for this path

```bash
curl -sf -X DELETE "$BASE/deployments/$AGENT_NAME-sandbox"
curl -sf -X DELETE "$BASE/deployment-configs/$AGENT_NAME-cfg"
```
