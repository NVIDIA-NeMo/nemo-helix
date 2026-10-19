"""Generate the fleet fixture the get_status tool reads. Deterministic."""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fixtures" / "fleet.json"


def main() -> None:
    rng = random.Random(7)
    nodes: dict[str, dict] = {}
    for pod in ("pod-a", "pod-b", "pod-c"):
        for i in range(1, 25):
            name = f"dgx-{pod[-1]}{i:02d}"
            state = rng.choices(["healthy", "degraded", "drained"], weights=[80, 15, 5])[0]
            nodes[name] = {
                "pod": pod,
                "state": state,
                "gpus": 8,
                "gpu_temp_c": rng.randint(45, 83),
                "xid_errors_24h": rng.choice([0, 0, 0, 1, 3, 12]) if state != "healthy" else 0,
                "nvlink_errors_24h": rng.choice([0, 0, 2, 40]) if pod == "pod-b" else 0,
                "last_reboot": f"2026-10-{rng.randint(1, 6):02d}T{rng.randint(0, 23):02d}:00Z",
            }
    services = {
        "Loom": {"state": "degraded", "detail": "pod-b fabric: 3 links flapping on switch lm-b-04"},
        "Vault": {"state": "healthy", "detail": "mds p99 18ms, capacity 71%"},
        "Dispatch": {"state": "healthy", "detail": "queue depth 412, 0 stuck"},
        "Node Doctor": {"state": "healthy", "detail": "4 drains in last 24h"},
        "Backbone": {"state": "healthy", "detail": "all BGP sessions up"},
        "Gatekeeper": {"state": "degraded", "detail": "group sync lag 14 min"},
        "Kite": {"state": "healthy", "detail": "apiserver p99 220ms"},
        "Beacon": {"state": "degraded", "detail": "endpoint llm-chat-prod 5xx 2.1% since deploy 14:02Z"},
        "Conveyor": {"state": "healthy", "detail": "2 of 61 pipelines late"},
        "Lantern": {"state": "healthy", "detail": "ingest lag 4s"},
        "Ledger": {"state": "healthy", "detail": "team ml-research at 97% of quota"},
        "Sentinel": {"state": "healthy", "detail": "0 open findings"},
    }
    clusters = {
        "pod-a": {"nodes": 24, "healthy": sum(1 for n in nodes.values() if n["pod"] == "pod-a" and n["state"] == "healthy")},
        "pod-b": {"nodes": 24, "healthy": sum(1 for n in nodes.values() if n["pod"] == "pod-b" and n["state"] == "healthy")},
        "pod-c": {"nodes": 24, "healthy": sum(1 for n in nodes.values() if n["pod"] == "pod-c" and n["state"] == "healthy")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"nodes": nodes, "services": services, "clusters": clusters}, indent=2))
    print(f"wrote {OUT} ({len(nodes)} nodes, {len(services)} services)")


if __name__ == "__main__":
    main()
