<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Notif Plugin

Ephemeral notification events. Publishing an event fans it out to every SSE listener on the same
workspace. Nothing is stored, so a listener only sees events published while it is connected.

## Install and run

```bash
uv pip install -e plugins/nemo-notif
nemo services run          # notif is discovered through the nemo.services entry point
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/apis/notif/v2/workspaces/{workspace}/events` | Publish an event. Returns the event and how many listeners received it. |
| `GET` | `/apis/notif/v2/workspaces/{workspace}/events/stream` | SSE stream. Repeat `?types=<type>` to filter by event type. |

Publish body:

```json
{
  "type": "jobs.job.failed",
  "title": "Fine-tune failed",
  "message": "optional, up to 2000 chars",
  "severity": "info | success | warning | error",
  "link": "/workspaces/default/jobs/abc",
  "data": {}
}
```

Each SSE frame carries `id:`, `event:` (the type) and `data:` (the full event JSON, including `type`,
`id`, `workspace`, `created_at`, `created_by`). The type is repeated inside `data` because Studio's
`util/sseStream.ts` only parses `id:` and `data:`. The server sends `: keepalive` comments every 15s.

```bash
curl -N "http://localhost:8080/apis/notif/v2/workspaces/default/events/stream"
curl -X POST http://localhost:8080/apis/notif/v2/workspaces/default/events \
  -H 'content-type: application/json' -d '{"type":"jobs.job.failed","title":"Fine-tune failed"}'
```

## Authz

`POST` requires `notif.events.create` (Editor) and `GET .../stream` requires `notif.events.read`
(Viewer and Editor). Both accept users and service principals, so other services can publish.

## Limitations

- Fan-out is in-process. With several API replicas, a listener only sees events published to its
  own replica. That is exact for local `nemo services run`, which is one process.
- No replay: `Last-Event-ID` is ignored and events published while disconnected are lost.
- A listener more than 256 events behind is disconnected.
