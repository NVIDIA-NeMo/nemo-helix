// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { Button as e, Flex as t, Stack as n, Text as r } from "@nvidia/foundations-react-core";
import { useQuery as i } from "@tanstack/react-query";
import { useEffect as a, useRef as o, useState as s } from "react";
import { NavLink as c, Navigate as l, Route as u, Routes as d, useParams as f } from "react-router";
import { jsx as p, jsxs as m } from "react/jsx-runtime";
//#region src/api.ts
var h = (e) => `${e.apiBaseUrl}/apis/observability/v2/workspaces/${encodeURIComponent(e.workspaceId)}`;
async function g(e, t, n = "GET") {
	let r = e.auth.getAccessToken(), i = await fetch(`${h(e)}${t}`, {
		method: n,
		credentials: "same-origin",
		headers: r ? { Authorization: `Bearer ${r}` } : {}
	});
	if (!i.ok) {
		let e = await i.json().then((e) => e?.detail, () => void 0);
		throw Error(typeof e == "string" ? e : `${n} ${t} failed (${i.status})`);
	}
	return await i.json();
}
var _ = (e) => g(e, "/connections").then((e) => e.data), v = (e, t) => g(e, `/connections/${encodeURIComponent(t)}/session`, "POST"), y = (e, t) => g(e, `/connections/${encodeURIComponent(t)}/otlp-snippet`), b = (e, t = "") => `/workspaces/${e}/plugin/observability/${t}`, x = 6e4, S = 3e4;
function C({ host: e }) {
	let t = i({
		queryKey: [
			"observability",
			e.workspaceId,
			"connections"
		],
		queryFn: () => _(e)
	});
	if (t.isPending) return /* @__PURE__ */ p(D, { text: "Loading trace explorers…" });
	if (t.isError) return /* @__PURE__ */ p(D, { text: `Could not load trace explorers: ${t.error.message}` });
	let n = t.data;
	return n.length === 0 ? /* @__PURE__ */ p(D, { text: "No trace explorer is configured. An operator adds one under observability.connections in the platform config; see the plugin README." }) : /* @__PURE__ */ m(d, { children: [/* @__PURE__ */ p(u, {
		index: !0,
		element: /* @__PURE__ */ p(l, {
			to: b(e.workspaceId, n[0].name),
			replace: !0
		})
	}), /* @__PURE__ */ p(u, {
		path: ":connection",
		element: /* @__PURE__ */ p(w, {
			host: e,
			connections: n
		})
	})] });
}
function w({ host: i, connections: o }) {
	let { connection: l } = f(), u = o.find((e) => e.name === l), { ready: d, error: h } = T(i, u?.name), [g, _] = s(!1), { set: v } = i.breadcrumbs;
	if (a(() => (v([{
		label: "Trace Explorer",
		href: b(i.workspaceId)
	}, { label: u?.display_name ?? l ?? "" }]), () => v([])), [
		v,
		i.workspaceId,
		u?.display_name,
		l
	]), !u) return /* @__PURE__ */ p(D, { text: `Unknown trace explorer "${l}".` });
	let y = `${i.apiBaseUrl}${u.frame_url}`;
	return /* @__PURE__ */ m(n, {
		gap: "2",
		className: "h-full p-4",
		children: [
			/* @__PURE__ */ m(t, {
				gap: "3",
				align: "center",
				className: "border-b border-subtle pb-2",
				children: [o.length > 1 ? o.map((e) => /* @__PURE__ */ p(c, {
					to: b(i.workspaceId, e.name),
					className: ({ isActive: e }) => `px-3 py-1 rounded text-sm font-medium ${e ? "text-primary bg-surface-hover" : "text-subtle hover:text-primary"}`,
					children: e.display_name
				}, e.name)) : /* @__PURE__ */ p(r, {
					kind: "title/sm",
					children: u.display_name
				}), /* @__PURE__ */ m(t, {
					gap: "2",
					className: "ml-auto",
					children: [/* @__PURE__ */ p(e, {
						kind: "tertiary",
						onClick: () => _((e) => !e),
						children: g ? "Hide agent export" : "Send agent traces here"
					}), /* @__PURE__ */ p(e, {
						kind: "secondary",
						disabled: !d,
						onClick: () => window.open(y, "_blank", "noopener"),
						children: "Open in new tab"
					})]
				})]
			}),
			g && /* @__PURE__ */ p(E, {
				host: i,
				connection: u
			}),
			h ? /* @__PURE__ */ p(D, { text: `Could not open ${u.display_name}: ${h}` }) : d ? /* @__PURE__ */ p("iframe", {
				src: y,
				title: u.display_name,
				className: "flex-1 w-full rounded border border-subtle bg-surface-base"
			}, u.name) : /* @__PURE__ */ p(D, { text: `Opening ${u.display_name}…` })
		]
	});
}
function T(e, t) {
	let n = o(e);
	n.current = e;
	let [r, i] = s(!1), [c, l] = s(null);
	return a(() => {
		if (!t) return;
		let e = !1, r, a = null;
		i(!1), l(null);
		let o = () => (a ??= v(n.current, t).then((t) => {
			if (e) return;
			i(!0), l(null);
			let n = Math.max(S, t.expires_at * 1e3 - Date.now() - x);
			r = setTimeout(o, n);
		}).catch((t) => {
			e || l(t instanceof Error ? t.message : String(t));
		}).finally(() => {
			a = null;
		}), a);
		return o(), () => {
			e = !0, clearTimeout(r);
		};
	}, [t]), {
		ready: r,
		error: c
	};
}
function E({ host: a, connection: o }) {
	let s = i({
		queryKey: [
			"observability",
			a.workspaceId,
			"otlp-snippet",
			o.name
		],
		queryFn: () => y(a, o.name)
	});
	return s.isPending ? /* @__PURE__ */ p(r, {
		kind: "body/regular/sm",
		children: "Loading export settings…"
	}) : s.isError ? /* @__PURE__ */ p(r, {
		kind: "body/regular/sm",
		children: s.error.message
	}) : /* @__PURE__ */ m(n, {
		gap: "2",
		className: "rounded border border-subtle bg-surface-raised p-3",
		children: [
			/* @__PURE__ */ m(r, {
				kind: "body/regular/sm",
				children: [
					"Add this to an agent's agent.yaml to export its traces straight to ",
					o.display_name,
					". Helix keeps sending its own telemetry to Intake as well."
				]
			}),
			/* @__PURE__ */ p("pre", {
				className: "overflow-auto rounded bg-surface-sunken p-2 text-sm",
				children: s.data.yaml
			}),
			/* @__PURE__ */ p(t, {
				gap: "2",
				children: /* @__PURE__ */ p(e, {
					kind: "secondary",
					onClick: () => navigator.clipboard.writeText(s.data.yaml).then(() => a.notifications.notify("Copied telemetry stanza", "success"), () => a.notifications.notify("Copy failed; select the text instead", "error")),
					children: "Copy"
				})
			}),
			/* @__PURE__ */ m(r, {
				kind: "body/regular/sm",
				className: "text-subtle",
				children: [
					o.display_name,
					" must be served under ",
					o.ui_prefix,
					": ",
					s.data.upstream_setup.join(", ")
				]
			})
		]
	});
}
function D({ text: e }) {
	return /* @__PURE__ */ p(t, {
		className: "h-full p-6",
		align: "center",
		justify: "center",
		children: /* @__PURE__ */ p(r, {
			kind: "body/regular/md",
			className: "text-subtle",
			children: e
		})
	});
}
//#endregion
//#region src/Nav.tsx
var O = (e) => [{
	group: "Observability",
	items: [{
		id: "observability-trace-explorer",
		iconName: "chart-flow",
		label: "Trace Explorer",
		href: b(e)
	}]
}];
//#endregion
export { C as Root, O as navItems };
