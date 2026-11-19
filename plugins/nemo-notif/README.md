<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Notif Plugin

Workspace notification events. Publishing an event stores it as a `notif_event` entity and fans it
out to every SSE listener on the same workspace. Each caller has their own read state, so events
published while you were away show up as unread on your next load.

Client flow: load `GET /events` (with `unread_count`) on first render, open the SSE stream for live
events, and call `/read` or `/read-all` as the user reads them. The stream does not replay, so
reload `GET /events` after a reconnect.

## Install and run

```bash
uv pip install -e plugins/nemo-notif
nemo services run          # notif is discovered through the nemo.services entry point
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/apis/notif/v2/workspaces/{workspace}/events` | Store and publish an event. Returns the event and how many listeners received it live. |
| `GET` | `/apis/notif/v2/workspaces/{workspace}/events` | Recent events, newest first, each with the caller's `read` flag, plus `unread_count`. Query: `limit` (default 50, max 1000), `unread_only`. |
| `POST` | `/apis/notif/v2/workspaces/{workspace}/events/{event_id}/read` | Mark one event read for the caller. Returns the event. 404 if unknown. |
| `POST` | `/apis/notif/v2/workspaces/{workspace}/events/{event_id}/unread` | Mark one event unread for the caller. |
| `POST` | `/apis/notif/v2/workspaces/{workspace}/events/read-all` | Mark every recent event read for the caller. Returns `{"marked": n}`. |
| `GET` | `/apis/notif/v2/workspaces/{workspace}/events/stream` | SSE stream of newly published events. Repeat `?types=<type>` to filter by event type. |

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
`id`, `workspace`, `created_at`, `created_by` and `read`, which is always `false` for a new event). The type is repeated inside `data` because Studio's
`util/sseStream.ts` only parses `id:` and `data:`. The server sends `: keepalive` comments every 15s.

```bash
curl -N "http://localhost:8080/apis/notif/v2/workspaces/default/events/stream"
curl -X POST http://localhost:8080/apis/notif/v2/workspaces/default/events \
  -H 'content-type: application/json' -d '{"type":"jobs.job.failed","title":"Fine-tune failed"}'
```

## Email

Published events can also be emailed. This is off by default, and when turned on it starts as a
dry run that logs each email instead of sending it. Configure it in the platform config:

```yaml
notif:
  email:
    enabled: true
    dry_run: true                     # set to false to open SMTP connections
    smtp_host: smtp.example.com       # unauthenticated relay; port 25, no STARTTLS by default
    sender: nemo-notif@example.com
    recipients: [oncall@example.com]  # receive every matching event
    allowed_recipient_domains: [example.com]
    severities: [warning, error]      # default
    type_prefixes: [openshell.]       # default: every type
    studio_base_url: https://nemo.example.com/studio
```

A publisher can address specific people with `data.email_to` (a list of addresses); only addresses
in `allowed_recipient_domains` are used. The OpenShell draft policy events set it to the job
creator's email when the job has one.

Each matching event is one plain-text email, sent in the background after the event is stored.
There are no per-user preferences, batching or retries: a failed send is logged and dropped.

## Authz

Publishing requires `notif.events.create` (Editor). Listing, streaming and the read/unread routes
require `notif.events.read` (Viewer and Editor): read state only changes the caller's own view. All
routes accept users and service principals, so other services can publish.

## Limitations

- Fan-out is in-process. With several API replicas, a listener only sees events published to its
  own replica. That is exact for local `nemo services run`, which is one process.
- The stream does not replay (`Last-Event-ID` is ignored). Events are stored, so reload
  `GET /events` after reconnecting.
- A listener more than 256 events behind is disconnected.
- Listing and unread counts only consider the 1000 most recent events per workspace, because the
  entity store cannot filter on read state. Nothing is deleted yet.
- Read state is keyed by the caller's principal id. With auth disabled (the local default) every
  caller is the same `anonymous` reader, so read state is shared.
- Read-all updates events one at a time.
