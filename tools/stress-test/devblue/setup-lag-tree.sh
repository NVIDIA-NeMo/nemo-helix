#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Bring up one build of the platform on :8090 against its own database, seed it, and create five
# long-sleeping jobs for the reconciler sweep.
#   setup-lag-tree.sh <tree> <database>      e.g. setup-lag-tree.sh /work/nmp-lag-main nhx_main
# Each build gets its own database so one build's migrations never touch another's schema.
# Run inside the dev pod after start-lag-platform.sh and lag-config.yaml are in /work.
set -eu
TREE="$1"
DB="$2"
export HOME=/work
unset UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
export NHX_DATA_DIR=/work/.nemo-data-lag
export POD_NAMESPACE=tbray-dev

"$TREE/.venv/bin/python" - "$DB" <<'PY'
import asyncio
import sys

import asyncpg


async def main(db: str) -> None:
    conn = await asyncpg.connect("postgresql://nhx:nhx-lag@pg-lag:5432/postgres")
    try:
        if not await conn.fetchval("select 1 from pg_database where datname = $1", db):
            await conn.execute(f'create database "{db}"')
            print("created database", db)
        else:
            print("database exists", db)
    finally:
        await conn.close()


asyncio.run(main(sys.argv[1]))
PY

export DATABASE_URL="postgresql://nhx:nhx-lag@pg-lag:5433/$DB"
bash /work/start-lag-platform.sh "$TREE"
for _ in $(seq 1 90); do curl -sf http://localhost:8090/health/ready >/dev/null && break; sleep 2; done
curl -s http://localhost:8090/health/ready
echo

# The admin email must go through --env: as a plain process variable the seed task does not see it.
"$TREE/.venv/bin/nemo-helix" run task --task nhx.platform_seed \
  --env NHX_AUTH_ADMIN_EMAIL=lag-admin@example.com \
  --env NHX_CONFIG_FILE_PATH=/work/nmp-lag-config.yaml > /work/logs/lag-seed.log 2>&1
echo "seed exit $?"

H=(-H "X-NHX-Principal-Id: lag-user" -H "X-NHX-Principal-Email: lag-admin@example.com" -H "Content-Type: application/json")
for n in 1 2 3 4 5; do
  body="{\"name\":\"lag-sleep-$n\",\"spec\":{},\"source\":\"lag\",\"platform_spec\":{\"steps\":[{\"name\":\"run\",\"environment\":[],\"executor\":{\"provider\":\"cpu\",\"profile\":\"default\",\"container\":{\"image\":\"docker.io/library/python:3.12-alpine\",\"entrypoint\":[\"/bin/sh\",\"-c\"],\"command\":[\"sleep 7200\"]},\"resources\":{\"requests\":{\"cpu\":\"100m\",\"memory\":\"128Mi\"},\"limits\":{\"cpu\":\"500m\",\"memory\":\"256Mi\"}}}}]}}"
  curl -s -o /tmp/j.json -w "%{http_code} " "${H[@]}" -X POST http://localhost:8090/apis/jobs/v2/workspaces/default/jobs -d "$body"
  head -c 120 /tmp/j.json
  echo
done
