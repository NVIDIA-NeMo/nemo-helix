// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MOCK of the platform side of community plugins, until the real gateway exists.
// Run with `bun mock/gateway.mjs` and point Studio at it
// (VITE_PLATFORM_BASE_URL=http://localhost:8090). It:
//   - adds the plugin in plugin.yaml to GET /apis/plugins,
//   - serves the built bundle at /plugin-ui/<name>/{index.js,plugin.manifest.json},
//   - proxies ONLY the declared endpoints at /apis/plugins/<name>/<endpoint>,
//     never forwarding the caller's Authorization/Cookie,
//   - forwards everything else to the real platform.
// Skipped on purpose: auth, per-plugin permissions, upstream credentials, redirects policy.

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, "..");
const manifest = Bun.YAML.parse(readFileSync(path.join(root, "plugin.yaml"), "utf8"));
const PORT = Number(process.env.GATEWAY_PORT ?? 8090);
const PLATFORM = process.env.PLATFORM_URL ?? "http://127.0.0.1:8080";
const distDir = path.dirname(path.resolve(root, manifest.bundle));
const ASSETS = new Set(["index.js", "plugin.manifest.json"]);

const compile = (template) =>
  new RegExp(`^${template.replace(/\{[^}]+\}/g, "([^/]+)")}$`);
const endpoints = manifest.endpoints.map((e) => ({
  ...e,
  re: compile(e.path),
  names: [...e.path.matchAll(/\{([^}]+)\}/g)].map((m) => m[1]),
}));

const DROP = new Set(["authorization", "cookie", "host", "origin", "referer", "content-length"]);
const cors = (req) => ({
  "access-control-allow-origin": req.headers.get("origin") ?? "*",
  "access-control-allow-headers": "*",
  "access-control-allow-methods": "*",
  vary: "origin",
});
const json = (req, status, body) =>
  Response.json(body, { status, headers: cors(req) });

async function proxyDeclared(req, url) {
  const rest = url.pathname.slice(`/apis/plugins/${manifest.name}`.length);
  for (const e of endpoints) {
    if (e.method !== req.method && !(e.method === "GET" && req.method === "HEAD")) continue;
    const m = rest.match(e.re);
    if (!m) continue;
    let upstream = e.upstream;
    e.names.forEach((n, i) => (upstream = upstream.replace(`{${n}}`, m[i + 1])));
    const headers = new Headers();
    for (const [k, v] of req.headers) if (!DROP.has(k)) headers.set(k, v);
    const res = await fetch(manifest.upstream + upstream + url.search, {
      method: req.method,
      headers,
      body: ["GET", "HEAD"].includes(req.method) ? undefined : req.body,
      redirect: "manual",
    });
    const out = new Headers(res.headers);
    out.delete("content-encoding");
    out.delete("content-length");
    for (const [k, v] of Object.entries(cors(req))) out.set(k, v);
    return new Response(res.body, { status: res.status, headers: out });
  }
  return json(req, 404, { code: "ENDPOINT_NOT_DECLARED", detail: `${req.method} ${rest}` });
}

Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(req) });

    if (url.pathname === "/apis/plugins") {
      const res = await fetch(PLATFORM + url.pathname);
      const list = (await res.json()).filter((p) => p.name !== manifest.name);
      list.push({ name: manifest.name, bundleUrl: `/plugin-ui/${manifest.name}/index.js` });
      return json(req, 200, list);
    }
    if (url.pathname.startsWith(`/apis/plugins/${manifest.name}/`)) return proxyDeclared(req, url);

    const asset = url.pathname.match(new RegExp(`^/plugin-ui/${manifest.name}/([^/]+)$`));
    if (asset && ASSETS.has(asset[1])) {
      const file = Bun.file(path.join(distDir, asset[1]));
      return (await file.exists())
        ? new Response(file, { headers: { ...cors(req), "cache-control": "no-cache" } })
        : json(req, 404, { code: "ASSET_NOT_BUILT", detail: asset[1] });
    }

    // Everything else is the real platform, untouched.
    return fetch(PLATFORM + url.pathname + url.search, {
      method: req.method,
      headers: req.headers,
      body: ["GET", "HEAD"].includes(req.method) ? undefined : req.body,
      redirect: "manual",
    });
  },
});
console.log(`mock gateway on :${PORT} -> platform ${PLATFORM}, plugin '${manifest.name}' -> ${manifest.upstream}`);
