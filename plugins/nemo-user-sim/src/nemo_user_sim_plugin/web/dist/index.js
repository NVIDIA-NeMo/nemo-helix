// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { Route as e, Routes as t, useNavigate as n, useParams as r } from "react-router";
import { createContext as i, useCallback as a, useContext as o, useEffect as s, useMemo as c, useState as l } from "react";
import { Fragment as u, jsx as d, jsxs as f } from "react/jsx-runtime";
import { DeleteConfirmationModal as p, FormModal as m, HelixJobTerminalStatuses as h, JOB_POLLING_INTERVAL_MS as g, RelativeTime as _, StatusBadge as v, StudioDataView as y, TableEmptyState as b, useStudioDataViewState as x } from "@nemo/common";
import { Anchor as ee, Button as S, Card as C, Checkbox as w, CodeSnippet as T, ComboboxContent as te, ComboboxItem as E, ComboboxListbox as ne, ComboboxRoot as re, ComboboxTrigger as ie, Flex as D, FormField as O, Modal as ae, SegmentedControl as oe, SelectContent as se, SelectItem as ce, SelectListbox as k, SelectRoot as A, SelectTrigger as le, SidePanel as ue, Skeleton as de, Slider as fe, Spinner as j, Stack as M, Tag as N, Text as P, TextArea as F, TextInput as I } from "@nvidia/foundations-react-core";
import { useMutation as L, useQuery as R, useQueryClient as z } from "@tanstack/react-query";
//#region src/host.tsx
var B = i(null), pe = ({ host: e, children: t }) => /* @__PURE__ */ d(B.Provider, {
	value: e,
	children: t
}), V = () => {
	let e = o(B);
	if (!e) throw Error("useHost must be used within the user-sim plugin Root");
	return e;
}, H = class extends Error {
	status;
	constructor(e, t) {
		super(`${e}: ${t}`), this.status = e;
	}
}, U = (e) => h.some((t) => t === e), W = (e, t) => `/apis/user-sim/v2/workspaces/${encodeURIComponent(e)}/jobs/${t}`, G = () => {
	let { apiBaseUrl: e, auth: t } = V();
	return a(async (n, r) => {
		let i = t.getAccessToken(), a = await fetch(`${e}${n}`, {
			...r,
			headers: {
				"Content-Type": "application/json",
				...i ? { Authorization: `Bearer ${i}` } : {}
			}
		});
		if (!a.ok) throw new H(a.status, await a.text());
		return a.status === 204 ? void 0 : await a.json();
	}, [e, t]);
}, me = () => {
	let e = G();
	return R({
		queryKey: ["user-sim", "probes"],
		queryFn: () => e("/apis/user-sim/v1/probes"),
		staleTime: Infinity
	});
}, K = () => {
	let { workspaceId: e } = V(), t = G();
	return R({
		queryKey: [
			"user-sim",
			e,
			"simulate"
		],
		queryFn: async () => [...(await t(`${W(e, "simulate")}?page_size=100`)).data].sort((e, t) => (t.created_at ?? "").localeCompare(e.created_at ?? "")),
		refetchInterval: (e) => e.state.data?.some((e) => !U(e.status)) ? g : !1
	});
}, he = (e) => {
	let { workspaceId: t } = V(), n = G();
	return R({
		queryKey: [
			"user-sim",
			t,
			"simulate",
			e
		],
		queryFn: () => n(`${W(t, "simulate")}/${encodeURIComponent(e)}`),
		refetchInterval: (e) => !U(e.state.data?.status) && g
	});
}, q = (e, t, n) => {
	let { workspaceId: r } = V(), i = G();
	return R({
		queryKey: [
			"user-sim",
			r,
			"simulate",
			e,
			t,
			n
		],
		queryFn: () => i(`${W(r, "simulate")}/${encodeURIComponent(e)}/results/${t}/download`).catch((e) => {
			if (e instanceof H && e.status === 404) return null;
			throw e;
		}),
		refetchInterval: !n && g,
		staleTime: n ? Infinity : 0
	});
}, J = (e, t) => ({
	summary: q(e, "summary", t),
	scores: q(e, "scores", t),
	transcripts: q(e, "transcripts", t),
	scoringError: q(e, "scoring_error", t)
}), ge = (e) => {
	let { workspaceId: t } = V(), n = G(), r = z();
	return L({
		mutationFn: (r) => n(W(t, e), {
			method: "POST",
			body: JSON.stringify({ spec: r })
		}),
		onSuccess: () => r.invalidateQueries({ queryKey: [
			"user-sim",
			t,
			e
		] })
	});
}, _e = (e, t) => {
	let { workspaceId: n } = V(), r = G(), i = z();
	return L({
		mutationFn: async (i) => {
			let a = await Promise.allSettled(i.map((i) => r(`${W(n, "simulate")}/${encodeURIComponent(i)}${t}`, { method: e }))), o = i.filter((e, t) => a[t].status === "rejected");
			if (o.length) throw Error(`${o.length} of ${i.length} failed: ${o.join(", ")}`);
		},
		onSettled: () => i.invalidateQueries({ queryKey: [
			"user-sim",
			n,
			"simulate"
		] })
	});
}, ve = () => _e("DELETE", ""), ye = () => _e("POST", "/cancel"), be = () => {
	let { workspaceId: e } = V(), t = G();
	return R({
		queryKey: [
			"user-sim",
			e,
			"agents"
		],
		queryFn: async () => (await t(`/apis/agents/v2/workspaces/${encodeURIComponent(e)}/agents?page_size=100`)).data
	});
}, xe = () => {
	let { workspaceId: e } = V(), t = G();
	return R({
		queryKey: [
			"user-sim",
			e,
			"models"
		],
		queryFn: async () => (await t(`/apis/inference-gateway/v2/workspaces/${encodeURIComponent(e)}/openai/-/v1/models`)).data.map((e) => e.id.split("/").slice(1).join("/") || e.id).sort()
	});
}, Se = (e) => {
	let { workspaceId: t } = V(), n = G();
	return R({
		queryKey: [
			"user-sim",
			t,
			"probe-suggestions",
			e
		],
		queryFn: () => n(`/apis/user-sim/v2/workspaces/${encodeURIComponent(t)}/probe-suggestions`, {
			method: "POST",
			body: JSON.stringify({ agent: e })
		}),
		enabled: !!e,
		staleTime: Infinity
	});
}, Ce = () => {
	let { workspaceId: e } = V(), t = G();
	return R({
		queryKey: [
			"user-sim",
			e,
			"scenario-sets"
		],
		queryFn: () => t(`/apis/user-sim/v2/workspaces/${encodeURIComponent(e)}/scenario-sets`)
	});
}, we = () => {
	let { workspaceId: e } = V(), t = G();
	return L({ mutationFn: (n) => t(`/apis/user-sim/v2/workspaces/${encodeURIComponent(e)}/persona-preview`, {
		method: "POST",
		body: JSON.stringify(n)
	}) });
}, Te = (e) => `/apis/intake/v2/workspaces/${encodeURIComponent(e)}/annotations`, Ee = (e) => {
	let { workspaceId: t } = V(), n = G();
	return R({
		queryKey: [
			"user-sim",
			t,
			"feedback",
			e
		],
		queryFn: async () => {
			let r = encodeURIComponent(JSON.stringify({
				session_id: e,
				kind: "feedback"
			}));
			return (await n(`${Te(t)}?filter=${r}&sort=-created_at&page_size=1`)).data[0]?.value ?? null;
		},
		enabled: !!e
	});
}, De = () => {
	let { workspaceId: e } = V(), t = G(), n = z();
	return L({
		mutationFn: (n) => t(Te(e), {
			method: "POST",
			body: JSON.stringify(n)
		}),
		onSuccess: (t, r) => n.invalidateQueries({ queryKey: [
			"user-sim",
			e,
			"feedback",
			r.session_id
		] })
	});
}, Oe = {
	Strict: "strict",
	Generous: "generous"
}, ke = ({ jobs: e, open: t, onClose: n }) => {
	let { notifications: r } = V(), i = ge("export"), [a, o] = l(e.length === 1 ? `${e[0]}-training` : "user-sim-training"), [s, c] = l("Strict"), [u, p] = l(!1), h = () => i.mutate({
		jobs: e,
		fileset: a.trim(),
		quality: Oe[s],
		require_review: u
	}, {
		onSuccess: (e) => {
			n(), r.notify(`Export ${e.name} started. Training data goes to fileset ${a.trim()}.`, "success");
		},
		onError: (e) => r.notify(`Couldn't start the export. ${e.message}`, "error")
	});
	return /* @__PURE__ */ d(m, {
		open: t,
		title: "Export training data",
		submitButtonText: "Export",
		loading: i.isPending,
		submitDisabled: !a.trim(),
		onSubmit: (e) => {
			e.preventDefault(), h();
		},
		onClose: n,
		children: /* @__PURE__ */ f(M, {
			gap: "density-md",
			children: [
				/* @__PURE__ */ f(P, {
					kind: "body/regular/sm",
					className: "text-subtle",
					children: [
						"Writes the conversations that passed scoring in",
						" ",
						e.length === 1 ? "this run" : `these ${e.length} runs`,
						" as one chat JSONL dataset, ready for the post-training plugin."
					]
				}),
				/* @__PURE__ */ d(O, {
					name: "fileset",
					slotLabel: "Dataset fileset",
					slotHelp: "Created if it doesn't exist.",
					children: /* @__PURE__ */ d(I, {
						value: a,
						onChange: (e) => o(e.target.value)
					})
				}),
				/* @__PURE__ */ d(O, {
					name: "quality",
					slotLabel: "Quality gate",
					slotHelp: "Strict drops any conversation a judge flagged.",
					children: /* @__PURE__ */ f(A, {
						value: s,
						onValueChange: (e) => c(e),
						children: [/* @__PURE__ */ d(le, { className: "w-full" }), /* @__PURE__ */ d(se, {
							className: "w-(--radix-popper-anchor-width)",
							children: /* @__PURE__ */ d(k, { children: Object.keys(Oe).map((e) => /* @__PURE__ */ d(ce, {
								value: e,
								children: e
							}, e)) })
						})]
					})
				}),
				/* @__PURE__ */ f(D, {
					align: "center",
					gap: "density-sm",
					children: [/* @__PURE__ */ d(w, {
						checked: u,
						onCheckedChange: (e) => p(e === !0),
						attributes: { CheckboxInput: { "aria-label": "Require review" } }
					}), /* @__PURE__ */ d(P, {
						kind: "body/regular/sm",
						children: "Only conversations a reviewer gave a thumbs up"
					})]
				})
			]
		})
	});
}, Ae = (e) => `/workspaces/${e}/plugin/user-sim`, je = Ae, Me = (e, t) => `${Ae(e)}/${encodeURIComponent(t)}`, Ne = (e, t) => `/workspaces/${e}/intake/sessions/${encodeURIComponent(t)}`, Y = (e) => (e ?? "").replaceAll("_", " "), Pe = {
	cooperative: "green",
	neutral: "gray",
	impatient: "yellow",
	frustrated: "red",
	confrontational: "red"
}, X = (e) => Pe[e ?? ""] ?? "gray", Fe = [
	["persona_age_bin", "Age"],
	["persona_education_level", "Education"],
	["user_interaction_style", "Interaction style"],
	["probe_type", "Probe"]
], Ie = {
	"agentic.boundary_held": "called a tool it shouldn't have",
	"agentic.blast_radius_contained": "took an action with too wide a reach",
	"agentic.consequence_disclosure": "didn't explain what the action would change",
	"agentic.confirmation_sought": "didn't ask before acting",
	"agentic.injection_resistance": "followed instructions planted in a tool result",
	helpfulness: "didn't help enough",
	accuracy: "got facts wrong",
	coherence: "hard to follow",
	safety: "said something unsafe",
	role_adherence: "stepped out of its role",
	language_appropriateness: "wrong language or tone"
}, Le = (e) => Ie[e] ?? Y(e.slice(e.indexOf(".") + 1)), Re = (e) => e ? e.split(", ") : [], ze = (e, t) => e.map((e) => {
	let n = t?.[e.trajectory_id];
	return {
		...e,
		passed: n?.passed ?? null,
		reason: n?.reason ?? null,
		tested: n?.tested ?? null,
		why: n?.why ?? {},
		picked: n?.picked ?? null
	};
}), Be = (e) => `${e.label}: ${e.tool}`, Ve = (e) => Re(e.reason).filter((t) => !(e.picked && t === "agentic.boundary_held")), He = (e, t, n = (e) => !e.passed) => {
	let r = /* @__PURE__ */ new Map();
	for (let i of e) {
		if (i.passed === null) continue;
		let e = String(i[t] ?? "unknown"), a = r.get(e) ?? {
			value: e,
			failed: 0,
			scored: 0
		};
		a.scored += 1, a.failed += +!!n(i), r.set(e, a);
	}
	return [...r.values()].sort((e, t) => t.failed / t.scored - e.failed / e.scored);
}, Ue = (e) => {
	try {
		return JSON.stringify(JSON.parse(e ?? ""), null, 2);
	} catch {
		return e ?? "";
	}
}, Z = ({ text: e, tag: t, real: n }) => /* @__PURE__ */ f(D, {
	align: "center",
	gap: "density-xs",
	children: [/* @__PURE__ */ d(P, {
		kind: "body/semibold/xs",
		className: "text-subtle",
		children: e
	}), /* @__PURE__ */ d(N, {
		kind: "solid",
		color: n ? "green" : "purple",
		density: "compact",
		readOnly: !0,
		children: t
	})]
}), We = ({ right: e, label: t, text: n }) => /* @__PURE__ */ d(D, {
	justify: e ? "end" : "start",
	children: /* @__PURE__ */ f(M, {
		gap: "density-xs",
		className: `rounded-lg p-3 ${e ? "bg-surface-raised" : "bg-surface-sunken"}`,
		style: { maxWidth: "80%" },
		children: [t, /* @__PURE__ */ d(P, {
			kind: "body/regular/sm",
			className: "whitespace-pre-wrap",
			children: n
		})]
	})
}), Ge = ({ call: e, result: t, real: n }) => /* @__PURE__ */ d(C, {
	density: "compact",
	style: { maxWidth: "80%" },
	children: /* @__PURE__ */ f(M, {
		gap: "density-sm",
		children: [
			/* @__PURE__ */ d(Z, {
				text: e.function.name,
				tag: "called by your agent",
				real: !0
			}),
			/* @__PURE__ */ d(P, {
				kind: "body/regular/xs",
				className: "text-subtle",
				children: "What it sent"
			}),
			/* @__PURE__ */ d(T, {
				value: Ue(e.function.arguments),
				language: "json",
				copyButton: "none"
			}),
			/* @__PURE__ */ d(Z, {
				text: "What came back",
				tag: n ? "real result" : "simulated result",
				real: n
			}),
			t === void 0 ? /* @__PURE__ */ d(P, {
				kind: "body/regular/xs",
				className: "text-subtle",
				children: "No result was recorded for this call."
			}) : /* @__PURE__ */ d(T, {
				value: Ue(t),
				language: "json",
				copyButton: "none"
			})
		]
	})
}), Ke = ({ messages: e, persona: t, agent: n, ownCalls: r = [] }) => {
	let i = new Map(e.filter((e) => e.role === "tool" && e.tool_call_id).map((e) => [e.tool_call_id, e.content]));
	return /* @__PURE__ */ d(M, {
		gap: "density-sm",
		children: e.map((e, a) => {
			if (e.role === "tool") return e.tool_call_id ? null : /* @__PURE__ */ d(We, {
				right: !1,
				label: /* @__PURE__ */ d(Z, {
					text: "Tool",
					tag: "simulated result",
					real: !1
				}),
				text: e.content ?? ""
			}, a);
			let o = e.role === "user" ? /* @__PURE__ */ d(Z, {
				text: t,
				tag: "simulated user",
				real: !1
			}) : e.role === "system" ? /* @__PURE__ */ d(Z, {
				text: "System prompt",
				tag: "set by the probe",
				real: !1
			}) : /* @__PURE__ */ d(Z, {
				text: n,
				tag: "your agent",
				real: !0
			});
			return /* @__PURE__ */ f(M, {
				gap: "density-sm",
				children: [e.content ? /* @__PURE__ */ d(We, {
					right: e.role === "user",
					label: o,
					text: e.content
				}) : null, e.tool_calls?.map((e, t) => /* @__PURE__ */ d(Ge, {
					call: e,
					result: e.id ? i.get(e.id) : void 0,
					real: !!e.id && r.includes(e.id)
				}, e.id ?? t))]
			}, a);
		})
	});
}, qe = ({ agent: e, persona: t, probePrompt: n, ownTools: r }) => /* @__PURE__ */ f(P, {
	kind: "body/regular/xs",
	className: "text-subtle",
	children: [n ? `Real: ${e}'s model, replies and tool calls, under the probe's system prompt shown below. ` : `Real: ${e}, with its own prompt, model and tool calls. `, r ? `Its own tools' results are real too. Simulated: ${t}.` : `Simulated: ${t} and the tool results.`]
}), Q = ({ label: e, children: t }) => /* @__PURE__ */ f(M, {
	gap: "0",
	children: [/* @__PURE__ */ d(P, {
		kind: "body/semibold/xs",
		className: "text-subtle",
		children: e
	}), t]
}), Je = ({ row: e, probe: t, messages: n }) => {
	let r = n?.find((e) => e.role === "user")?.content, i = Ve(e), [a, o] = l(!1);
	return /* @__PURE__ */ d(C, {
		density: "compact",
		children: /* @__PURE__ */ f(M, {
			gap: "density-sm",
			children: [
				/* @__PURE__ */ f(P, {
					kind: "body/semibold/sm",
					children: ["What this conversation tested", t ? `: ${t.label}` : ""]
				}),
				r ? /* @__PURE__ */ d(Q, {
					label: "The user asked",
					children: /* @__PURE__ */ d(P, {
						kind: "body/regular/sm",
						children: r
					})
				}) : null,
				e.tested ? /* @__PURE__ */ f(u, { children: [e.tested.expected ? /* @__PURE__ */ d(Q, {
					label: "What the agent should do",
					children: /* @__PURE__ */ d(P, {
						kind: "body/regular/sm",
						children: e.tested.expected
					})
				}) : null, /* @__PURE__ */ f(D, {
					gap: "density-md",
					wrap: "wrap",
					children: [e.tested.should_call ? /* @__PURE__ */ d(Q, {
						label: "Should call",
						children: /* @__PURE__ */ d(N, {
							kind: "solid",
							color: "green",
							density: "compact",
							readOnly: !0,
							children: e.tested.should_call
						})
					}) : null, e.tested.should_not_call ? /* @__PURE__ */ d(Q, {
						label: "Shouldn't call",
						children: /* @__PURE__ */ d(N, {
							kind: "solid",
							color: "red",
							density: "compact",
							readOnly: !0,
							children: e.tested.should_not_call
						})
					}) : null]
				})] }) : t ? /* @__PURE__ */ d(Q, {
					label: "What this probe looks for",
					children: /* @__PURE__ */ d(P, {
						kind: "body/regular/sm",
						children: t.tests
					})
				}) : null,
				/* @__PURE__ */ d(Q, {
					label: "Result",
					children: e.passed === null ? /* @__PURE__ */ d(P, {
						kind: "body/regular/sm",
						children: "Not scored."
					}) : e.passed && !e.picked ? /* @__PURE__ */ d(P, {
						kind: "body/regular/sm",
						children: "Passed every check."
					}) : /* @__PURE__ */ f(M, {
						gap: "density-xs",
						children: [
							/* @__PURE__ */ d(P, {
								kind: "body/regular/sm",
								children: /* @__PURE__ */ f("strong", { children: [[...e.picked ? [Be(e.picked)] : [], ...i.length ? [`Failed: ${i.map(Le).join(", ")}`] : []].join(". "), "."] })
							}),
							i.length ? /* @__PURE__ */ d("div", { children: /* @__PURE__ */ d(S, {
								kind: "tertiary",
								size: "small",
								onClick: () => o(!a),
								children: a ? "Hide why" : "Why"
							}) }) : null,
							a ? i.map((t) => /* @__PURE__ */ f(P, {
								kind: "body/regular/sm",
								children: [
									/* @__PURE__ */ f("strong", { children: [Le(t), ":"] }),
									" ",
									e.why[t] ?? ""
								]
							}, t)) : null
						]
					})
				})
			]
		})
	});
}, Ye = ({ rows: e, index: t, transcripts: n, outcome: r, agent: i, model: a, probes: o, onIndex: s, onClose: c }) => {
	let { workspaceId: u, navigation: p, notifications: m } = V(), h = t === null ? null : e[t], g = h?.status === "ok", _ = Ee(g ? h.trajectory_id : null), v = De(), [y, b] = l("");
	if (!h || t === null) return null;
	let x = n?.[h.trajectory_id], C = h.persona_name ?? "Simulated user", w = Ne(u, h.trajectory_id), T = (e, t) => v.mutate(e, {
		onSuccess: () => m.notify(t, "success"),
		onError: (e) => m.notify(`Couldn't save. ${e.message}`, "error")
	});
	return /* @__PURE__ */ d(ae, {
		open: !0,
		onOpenChange: (e) => !e && c(),
		className: "w-[90vw] max-w-[1000px]",
		slotHeading: /* @__PURE__ */ f(D, {
			align: "center",
			justify: "between",
			className: "w-full",
			gap: "density-md",
			children: [/* @__PURE__ */ f(M, {
				gap: "0",
				children: [/* @__PURE__ */ d(P, {
					kind: "body/semibold/md",
					children: h.persona_name
				}), /* @__PURE__ */ f(P, {
					kind: "body/regular/xs",
					className: "text-subtle",
					children: [
						h.persona_age_bin,
						" · ",
						h.user_interaction_style,
						" ·",
						" ",
						o.get(h.probe_type ?? "")?.label ?? Y(h.probe_type),
						" · ",
						r(h)
					]
				})]
			}), /* @__PURE__ */ f(D, {
				align: "center",
				gap: "density-sm",
				children: [
					/* @__PURE__ */ d(S, {
						kind: "secondary",
						size: "small",
						disabled: t === 0,
						onClick: () => s(t - 1),
						children: "‹"
					}),
					/* @__PURE__ */ f(P, {
						kind: "body/regular/sm",
						className: "text-subtle",
						children: [
							t + 1,
							" of ",
							e.length
						]
					}),
					/* @__PURE__ */ d(S, {
						kind: "secondary",
						size: "small",
						disabled: t === e.length - 1,
						onClick: () => s(t + 1),
						children: "›"
					})
				]
			})]
		}),
		slotFooter: g ? /* @__PURE__ */ f(M, {
			gap: "density-sm",
			className: "w-full",
			children: [/* @__PURE__ */ f(D, {
				align: "center",
				gap: "density-sm",
				children: [
					/* @__PURE__ */ d(P, {
						kind: "body/regular/sm",
						children: "Use for training?"
					}),
					/* @__PURE__ */ d(S, {
						kind: _.data === "positive" ? "primary" : "secondary",
						size: "small",
						onClick: () => T({
							session_id: h.trajectory_id,
							kind: "feedback",
							value: "positive"
						}, "Marked for training."),
						children: "👍 Yes"
					}),
					/* @__PURE__ */ d(S, {
						kind: _.data === "negative" ? "primary" : "secondary",
						size: "small",
						onClick: () => T({
							session_id: h.trajectory_id,
							kind: "feedback",
							value: "negative"
						}, "Left out of training."),
						children: "👎 No"
					}),
					/* @__PURE__ */ d("div", { className: "flex-1" }),
					/* @__PURE__ */ d(ee, {
						href: w,
						onClick: (e) => {
							e.preventDefault(), p.navigate(w);
						},
						children: "Open trace in Intake"
					})
				]
			}), /* @__PURE__ */ f(D, {
				gap: "density-sm",
				align: "end",
				children: [/* @__PURE__ */ d(F, {
					className: "flex-1",
					rows: 2,
					placeholder: "Add a note for whoever reviews this next",
					value: y,
					onChange: (e) => b(e.target.value)
				}), /* @__PURE__ */ d(S, {
					kind: "secondary",
					disabled: !y.trim() || v.isPending,
					onClick: () => {
						T({
							session_id: h.trajectory_id,
							kind: "note",
							text: y.trim()
						}, "Note saved."), b("");
					},
					children: "Save note"
				})]
			})]
		}) : null,
		children: /* @__PURE__ */ d("div", {
			className: "max-h-[70vh] overflow-auto",
			children: x ? /* @__PURE__ */ f(M, {
				gap: "density-md",
				children: [
					/* @__PURE__ */ d(qe, {
						agent: a && a !== i ? `${i} (${a})` : i,
						persona: C,
						probePrompt: x.some((e) => e.role === "system"),
						ownTools: (h.own_tool_calls ?? []).length > 0
					}),
					/* @__PURE__ */ d(Je, {
						row: h,
						probe: o.get(h.probe_type ?? ""),
						messages: x
					}, h.trajectory_id),
					/* @__PURE__ */ d(Ke, {
						messages: x,
						persona: C,
						agent: i,
						ownCalls: h.own_tool_calls ?? []
					})
				]
			}) : /* @__PURE__ */ d(P, {
				kind: "body/regular/sm",
				className: "text-subtle",
				children: h.status === "running" ? "This conversation is still running." : "No transcript was recorded for this conversation."
			})
		})
	});
}, Xe = [{
	value: "agent",
	children: "Agent"
}, {
	value: "model",
	children: "Model"
}], Ze = "built-in", Qe = (e) => `${e.fileset.replace(/^user-sim-scenarios-/, "")} · ${e.size ? `${e.size} topics` : e.kind}`, $e = [
	{
		value: "10",
		children: "Small · 10"
	},
	{
		value: "50",
		children: "Medium · 50"
	},
	{
		value: "200",
		children: "Large · 200"
	}
], et = ({ persona: e, removed: t, disabled: n, onToggle: r }) => /* @__PURE__ */ d(C, {
	density: "compact",
	className: t ? "opacity-50" : void 0,
	children: /* @__PURE__ */ f(D, {
		justify: "between",
		align: "center",
		gap: "density-md",
		children: [/* @__PURE__ */ f(M, {
			gap: "0",
			children: [
				/* @__PURE__ */ f(D, {
					align: "center",
					gap: "density-sm",
					children: [/* @__PURE__ */ d(P, {
						kind: "body/semibold/sm",
						children: e.name
					}), /* @__PURE__ */ d(N, {
						kind: "solid",
						color: X(e.style),
						density: "compact",
						readOnly: !0,
						children: e.style
					})]
				}),
				/* @__PURE__ */ f(P, {
					kind: "body/regular/xs",
					className: "text-subtle",
					children: [
						e.age,
						" · ",
						Y(e.education),
						" · ",
						Y(e.occupation),
						e.location ? ` · ${e.location}` : ""
					]
				}),
				/* @__PURE__ */ f(P, {
					kind: "body/regular/xs",
					className: "text-subtle",
					children: [
						Y(e.probe),
						" · patience ",
						e.traits.patience ?? "–",
						" · tech literacy",
						" ",
						e.traits.tech_literacy ?? "–"
					]
				})
			]
		}), /* @__PURE__ */ d(S, {
			kind: "tertiary",
			size: "small",
			disabled: n,
			onClick: r,
			children: t ? "Restore" : "Remove"
		})]
	})
}), tt = ({ open: e, onClose: t, from: r }) => {
	let { workspaceId: i, notifications: a } = V(), o = n(), [s, u] = l(r?.spec.target.kind ?? "agent"), [p, m] = l(r ? r.spec.target.agent ?? r.spec.target.model ?? null : null), [h, g] = l(r?.spec.probes ?? null), [_, v] = l(String(r?.spec.users ?? 10)), [y, b] = l(r?.spec.max_turns ?? 5), [x, ee] = l(r?.spec.locale ?? "en_US"), [T, ae] = l(r?.spec.scenarios ?? null), [se, ce] = l(r?.spec.target.use_agent_tools ?? !1), [k, A] = l(/* @__PURE__ */ new Set()), [le, F] = l(!1), [L, R] = l(null), z = be(), B = xe(), { data: pe = [] } = me(), H = Se(s === "agent" ? p : null), U = Ce(), W = we(), G = ge("simulate"), K = c(() => new Map((H.data?.probes ?? []).map((e) => [e.name, e.reason])), [H.data]), he = (U.data ?? []).find((e) => e.fileset === T), q = h ?? he?.probes ?? [...K.keys()], J = [...pe].sort((e, t) => Number(K.has(t.name)) - Number(K.has(e.name))), _e = s === "agent" && !!p && !le, ve = _e ? J.filter((e) => K.has(e.name) || q.includes(e.name)) : J, ye = s === "agent" ? (z.data ?? []).map((e) => e.name) : B.data ?? [], Te = s === "agent" ? z : B, Ee = {
		target: {
			kind: s,
			[s]: p ?? "",
			...s === "agent" && se ? { use_agent_tools: !0 } : {}
		},
		probes: q,
		users: Number(_),
		locale: x.trim(),
		max_turns: y,
		...T ? { scenarios: T } : {}
	}, De = !!p && q.length > 0 && Number(_) >= 1 && Number(_) <= 2e3, Oe = (e) => {
		m(e), g(null), F(!1), R(null);
	}, ke = (e) => {
		ae(e === Ze ? null : e), g(null), R(null);
	}, Ae = (e, t) => g(t ? [...q, e] : q.filter((t) => t !== e)), je = (e) => A((t) => {
		let n = new Set(t);
		return n.has(e) ? n.delete(e) : n.add(e), n;
	}), Ne = (e) => G.mutate(e, {
		onSuccess: (e) => {
			t(), o(Me(i, e.name));
		},
		onError: (e) => a.notify(`Couldn't start the run. ${e.message}`, "error")
	}), Y = (e) => W.mutate(e, {
		onSuccess: (e) => {
			R(e), A(/* @__PURE__ */ new Set());
		},
		onError: (e) => a.notify(`Couldn't pick personas. ${e.message}`, "error")
	}), Pe = L?.personas ?? [], X = Pe.length - k.size, Fe = Number(_) - X, Ie = L ? {
		preview: L.preview,
		exclude_personas: [...k]
	} : {}, Le = L ? /* @__PURE__ */ f(D, {
		justify: "between",
		gap: "density-sm",
		className: "w-full",
		children: [/* @__PURE__ */ d(S, {
			kind: "tertiary",
			disabled: W.isPending,
			onClick: () => R(null),
			children: "Back"
		}), /* @__PURE__ */ f(D, {
			gap: "density-sm",
			children: [Fe > 0 ? /* @__PURE__ */ f(S, {
				kind: "secondary",
				disabled: W.isPending,
				onClick: () => Y({
					...Ee,
					...Ie
				}),
				children: [W.isPending ? /* @__PURE__ */ d(j, {
					kind: "ring",
					"aria-label": "Picking personas"
				}) : null, W.isPending ? `Picking ${Fe} more…` : `Pick ${Fe} more`]
			}) : null, /* @__PURE__ */ f(S, {
				disabled: X === 0 || W.isPending || G.isPending,
				onClick: () => Ne({
					...Ee,
					...Ie
				}),
				children: [
					"Run with ",
					X,
					" ",
					X === 1 ? "persona" : "personas"
				]
			})]
		})]
	}) : /* @__PURE__ */ f(D, {
		justify: "end",
		gap: "density-sm",
		className: "w-full",
		children: [/* @__PURE__ */ d(S, {
			kind: "tertiary",
			disabled: !De || G.isPending,
			onClick: () => Ne(Ee),
			children: "Start without preview"
		}), /* @__PURE__ */ f(S, {
			disabled: !De || W.isPending,
			onClick: () => Y(Ee),
			children: [W.isPending ? /* @__PURE__ */ d(j, {
				kind: "ring",
				"aria-label": "Picking personas"
			}) : null, W.isPending ? `Picking ${_} personas…` : "Preview personas"]
		})]
	});
	return /* @__PURE__ */ d(ue, {
		open: e,
		onOpenChange: (e) => !e && t(),
		side: "right",
		bordered: !0,
		modal: !0,
		className: "w-[560px]",
		slotHeading: L ? "Who this run will meet" : r ? "Run again" : "New simulation",
		slotFooter: Le,
		children: L ? /* @__PURE__ */ f(M, {
			gap: "density-sm",
			children: [
				/* @__PURE__ */ d(P, {
					kind: "body/regular/sm",
					className: "text-subtle",
					children: "Picked from census-grounded personas. Remove any that don't fit, then run with the personas left or pick more to replace them."
				}),
				Pe.map((e) => W.isPending && k.has(e.persona_uuid) ? null : /* @__PURE__ */ d(et, {
					persona: e,
					removed: k.has(e.persona_uuid),
					disabled: W.isPending,
					onToggle: () => je(e.persona_uuid)
				}, e.persona_uuid)),
				W.isPending ? Array.from({ length: Fe }, (e, t) => /* @__PURE__ */ d(C, {
					density: "compact",
					children: /* @__PURE__ */ f(M, {
						gap: "density-xs",
						children: [/* @__PURE__ */ d(de, { className: "w-1/3" }), /* @__PURE__ */ d(de, { className: "w-2/3" })]
					})
				}, t)) : null
			]
		}) : W.isPending ? /* @__PURE__ */ d(D, {
			justify: "center",
			className: "py-16",
			children: /* @__PURE__ */ d(j, { slotDescription: `Picking ${_} personas. This takes about 20 seconds.` })
		}) : /* @__PURE__ */ f(M, {
			gap: "density-lg",
			children: [
				r ? /* @__PURE__ */ f(P, {
					kind: "body/regular/sm",
					className: "text-subtle",
					children: [
						"Same settings as ",
						r.job,
						T ? `, with its topics from ${T}` : "",
						". New simulated users each time."
					]
				}) : null,
				/* @__PURE__ */ d(O, {
					name: "target",
					slotLabel: "What to test",
					children: /* @__PURE__ */ f(M, {
						gap: "density-sm",
						children: [/* @__PURE__ */ d(oe, {
							className: "w-full",
							items: Xe,
							value: s,
							onValueChange: (e) => {
								u(e), Oe(null);
							}
						}), /* @__PURE__ */ f(re, {
							value: p ?? "",
							onValueChange: (e) => {
								typeof e == "string" && Oe(e || null);
							},
							children: [/* @__PURE__ */ d(ie, { placeholder: s === "agent" ? "Search agents…" : "Search models…" }), /* @__PURE__ */ d(te, { children: /* @__PURE__ */ d(ne, {
								emptyStateMessage: Te.isPending ? `Loading ${s}s…` : Te.isError ? `Couldn't load ${s}s. ${Te.error.message}` : `No ${s}s match`,
								children: ye.map((e) => /* @__PURE__ */ d(E, {
									value: e,
									children: e
								}, e))
							}) })]
						}, s)]
					})
				}),
				s === "agent" ? /* @__PURE__ */ d(O, {
					name: "agent-tools",
					slotLabel: "Agent's tools",
					children: /* @__PURE__ */ f(D, {
						align: "start",
						gap: "density-sm",
						children: [/* @__PURE__ */ d(w, {
							checked: se,
							onCheckedChange: (e) => ce(e === !0),
							attributes: { CheckboxInput: { "aria-label": "Use the agent's own tools" } }
						}), /* @__PURE__ */ f(M, {
							gap: "0",
							children: [
								/* @__PURE__ */ d(P, {
									kind: "body/regular/sm",
									children: "Use the agent's own tools"
								}),
								/* @__PURE__ */ d(P, {
									kind: "body/regular/xs",
									className: "text-subtle",
									children: "Only turn this on if they're safe to call for real, like search."
								}),
								/* @__PURE__ */ d(P, {
									kind: "body/regular/xs",
									className: "text-subtle",
									children: "Off: the agent only gets UserSim's simulated tools."
								})
							]
						})]
					})
				}) : null,
				r ? null : /* @__PURE__ */ d(O, {
					name: "topics",
					slotLabel: "Topics",
					slotHelp: "What the simulated users ask about. Upload your own with nemo user-sim run --scenarios.",
					children: /* @__PURE__ */ f(re, {
						value: T ?? Ze,
						onValueChange: (e) => {
							typeof e == "string" && ke(e || Ze);
						},
						children: [/* @__PURE__ */ d(ie, { placeholder: "UserSim's built-in topics" }), /* @__PURE__ */ d(te, { children: /* @__PURE__ */ f(ne, {
							emptyStateMessage: U.isError ? `Couldn't load topic sets. ${U.error.message}` : "No topic sets match",
							children: [/* @__PURE__ */ d(E, {
								value: Ze,
								children: "UserSim's built-in topics"
							}), (U.data ?? []).map((e) => /* @__PURE__ */ d(E, {
								value: e.fileset,
								children: Qe(e)
							}, e.fileset))]
						}) })]
					})
				}),
				/* @__PURE__ */ d(O, {
					name: "probes",
					slotLabel: "Probes",
					slotHelp: s === "agent" ? "Chosen for you from the agent. Change them if you like." : "What the simulated users try to do.",
					children: /* @__PURE__ */ f(M, {
						gap: "density-sm",
						children: [
							H.isFetching ? /* @__PURE__ */ f(P, {
								kind: "body/regular/sm",
								className: "text-subtle",
								children: [
									"Reading ",
									p,
									"'s config and ETHOS.md…"
								]
							}) : H.data ? /* @__PURE__ */ f(P, {
								kind: "body/regular/xs",
								className: "text-subtle",
								children: [
									"Inferred from ",
									p,
									"'s ",
									H.data.read.join(" and "),
									"."
								]
							}) : null,
							ve.map((e) => /* @__PURE__ */ f(D, {
								align: "start",
								gap: "density-sm",
								children: [/* @__PURE__ */ d(w, {
									checked: q.includes(e.name),
									onCheckedChange: (t) => Ae(e.name, t === !0),
									attributes: { CheckboxInput: { "aria-label": e.label } }
								}), /* @__PURE__ */ f(M, {
									gap: "0",
									children: [/* @__PURE__ */ f(D, {
										align: "center",
										gap: "density-sm",
										children: [/* @__PURE__ */ d(P, {
											kind: "body/regular/sm",
											children: e.label
										}), K.has(e.name) ? /* @__PURE__ */ d(N, {
											kind: "solid",
											color: "green",
											readOnly: !0,
											children: "Suggested"
										}) : null]
									}), /* @__PURE__ */ d(P, {
										kind: "body/regular/xs",
										className: "text-subtle",
										children: K.get(e.name) ?? e.tests
									})]
								})]
							}, e.name)),
							_e && ve.length < J.length ? /* @__PURE__ */ f(S, {
								kind: "tertiary",
								size: "small",
								onClick: () => F(!0),
								children: [
									"More probes (",
									J.length - ve.length,
									")"
								]
							}) : null
						]
					})
				}),
				/* @__PURE__ */ d(O, {
					name: "users",
					slotLabel: "Size",
					slotHelp: "How many simulated users, up to 2000.",
					children: /* @__PURE__ */ f(M, {
						gap: "density-sm",
						children: [/* @__PURE__ */ d(oe, {
							className: "w-full",
							items: $e,
							value: $e.some((e) => e.value === _) ? _ : "",
							onValueChange: v
						}), /* @__PURE__ */ d(I, {
							value: _,
							onChange: (e) => v(e.target.value.replace(/[^0-9]/g, ""))
						})]
					})
				}),
				/* @__PURE__ */ d(O, {
					name: "turns",
					slotLabel: `Conversation length: ${y} user turns`,
					slotHelp: "How many times each simulated user speaks before the conversation ends.",
					children: /* @__PURE__ */ d(fe, {
						value: y,
						min: 2,
						max: 10,
						step: 1,
						onValueChange: (e) => b(Array.isArray(e) ? e[0] : e),
						"aria-label": "Conversation length"
					})
				}),
				/* @__PURE__ */ d(O, {
					name: "locale",
					slotLabel: "Persona locale",
					slotHelp: "Personas for this locale must be installed with nemo user-sim setup.",
					children: /* @__PURE__ */ d(I, {
						value: x,
						onChange: (e) => ee(e.target.value)
					})
				})
			]
		})
	});
}, nt = (e) => {
	if (e.status === "running") return "Running";
	if (e.status === "failed") return `Failed to run: ${e.error?.message ?? "no reason recorded"}`;
	if (e.passed === null) return "Not scored";
	let t = Ve(e).map(Le);
	return e.picked ? [Be(e.picked), ...t].join("; also ") : e.passed ? "Passed" : t.length ? `Failed: ${t.join("; ")}` : "Failed scoring";
}, rt = (e) => e.find((e) => e.picked?.pick === "should_not_call")?.picked?.label ?? null, it = ({ label: e, value: t }) => /* @__PURE__ */ d(C, {
	className: "flex-1 p-4",
	children: /* @__PURE__ */ f(M, {
		gap: "density-xs",
		children: [/* @__PURE__ */ d(P, {
			kind: "body/regular/sm",
			className: "text-subtle",
			children: e
		}), /* @__PURE__ */ d(P, {
			kind: "body/semibold/xl",
			children: t
		})]
	})
}), at = ({ rows: e, probeLabel: t, byWrongPick: n }) => /* @__PURE__ */ d("div", {
	style: {
		display: "grid",
		gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
		gap: 12
	},
	children: Fe.map(([r, i]) => /* @__PURE__ */ d(C, {
		className: "p-4",
		children: /* @__PURE__ */ f(M, {
			gap: "density-xs",
			children: [/* @__PURE__ */ d(P, {
				kind: "body/semibold/sm",
				children: i
			}), He(e, r, n ? (e) => e.picked?.pick === "should_not_call" : void 0).map((e) => /* @__PURE__ */ f(D, {
				justify: "between",
				gap: "density-md",
				children: [/* @__PURE__ */ d(P, {
					kind: "body/regular/sm",
					className: "truncate",
					children: r === "probe_type" ? t(e.value) : Y(e.value)
				}), /* @__PURE__ */ f(P, {
					kind: "body/regular/sm",
					className: "whitespace-nowrap text-subtle",
					children: [
						e.failed,
						" of ",
						e.scored,
						n ? "" : " failed"
					]
				})]
			}, e.value))]
		})
	}, r))
}), ot = () => {
	let { job: e = "" } = r(), { workspaceId: t, breadcrumbs: i, notifications: a } = V(), { data: o, isError: m } = he(e), h = U(o?.status), { summary: g, scores: _, transcripts: ee, scoringError: C } = J(e, h), w = c(() => ze(g.data?.conversations ?? [], _.data ?? null), [g.data, _.data]), T = x(), [te, E] = l(!1), [ne, re] = l(!1), [ie, O] = l(!1), ae = n(), oe = ve(), se = ye(), ce = ge("simulate"), [k, A] = l(null), { data: le = [] } = me(), ue = c(() => new Map(le.map((e) => [e.name, e])), [le]), de = (e) => ue.get(e)?.label ?? Y(e);
	s(() => (i.set([{
		label: "User Simulation",
		href: je(t)
	}, { label: e }]), () => i.set([])), [
		i,
		t,
		e
	]);
	let fe = c(() => (e, { rowSelectionColumn: t }) => [
		t({ size: 48 }),
		e.accessor("persona_name", {
			header: "Persona",
			size: 180
		}),
		e.accessor("persona_occupation", {
			header: "Occupation",
			size: 180,
			cell: ({ getValue: e }) => Y(e())
		}),
		e.accessor("user_interaction_style", {
			header: "Style",
			size: 160,
			cell: ({ getValue: e }) => /* @__PURE__ */ d(N, {
				kind: "solid",
				color: X(e()),
				density: "compact",
				readOnly: !0,
				children: e()
			})
		}),
		e.accessor("probe_type", {
			header: "Probe",
			size: 180,
			cell: ({ getValue: e }) => ue.get(e() ?? "")?.label ?? Y(e())
		}),
		e.display({
			id: "outcome",
			header: "Outcome",
			size: 320,
			cell: ({ row: e }) => nt(e.original)
		})
	], [ue]);
	if (m) return /* @__PURE__ */ d(b, {
		header: "Run not found",
		emptyMessage: `No simulation named ${e}.`
	});
	let j = w.filter((e) => e.passed).length, F = w.filter((e) => e.passed !== null).length, I = g.data?.totals, L = g.data?.target ?? o?.spec.target, R = g.data?.assistant_model ?? L?.model, z = L?.agent ?? L?.model ?? "The agent", B = rt(w);
	return /* @__PURE__ */ f(M, {
		gap: "density-lg",
		className: "p-4",
		children: [
			/* @__PURE__ */ f(D, {
				justify: "between",
				align: "center",
				children: [/* @__PURE__ */ f(M, {
					gap: "density-xs",
					children: [
						/* @__PURE__ */ f(D, {
							align: "center",
							gap: "density-md",
							children: [/* @__PURE__ */ d(P, {
								kind: "body/semibold/lg",
								children: e
							}), /* @__PURE__ */ d(v, { status: o?.status })]
						}),
						I ? /* @__PURE__ */ f(P, {
							kind: "body/regular/sm",
							className: "text-subtle",
							children: [
								z,
								L?.agent && R ? ` (${R})` : "",
								" talked to ",
								I.conversations,
								" ",
								"simulated ",
								I.conversations === 1 ? "user" : "users",
								"."
							]
						}) : null,
						L?.use_agent_tools ? /* @__PURE__ */ d(P, {
							kind: "body/regular/sm",
							className: "text-subtle",
							children: "Agent's own tools: on"
						}) : null
					]
				}), /* @__PURE__ */ f(D, {
					gap: "density-sm",
					children: [
						o && !h ? /* @__PURE__ */ d(S, {
							kind: "tertiary",
							onClick: () => se.mutate([e], {
								onSuccess: () => a.notify(`Cancelling ${e}.`, "success"),
								onError: (e) => a.notify(e.message, "error")
							}),
							children: "Cancel run"
						}) : null,
						/* @__PURE__ */ d(S, {
							kind: "tertiary",
							color: "danger",
							onClick: () => re(!0),
							children: "Delete"
						}),
						o ? /* @__PURE__ */ d(S, {
							kind: "secondary",
							onClick: () => O(!0),
							children: "Run again"
						}) : null,
						/* @__PURE__ */ d(S, {
							kind: "secondary",
							disabled: j === 0,
							onClick: () => E(!0),
							children: "Export training data"
						})
					]
				})]
			}),
			C.data?.message || o?.error_details?.message ? /* @__PURE__ */ d(P, {
				kind: "body/regular/sm",
				color: "danger",
				children: C.data?.message ?? o?.error_details?.message
			}) : null,
			g.isError ? /* @__PURE__ */ d(b, {
				header: "Couldn't load results",
				emptyMessage: g.error.message
			}) : g.isPending ? /* @__PURE__ */ d(P, {
				kind: "body/regular/sm",
				className: "text-subtle",
				children: "Loading results…"
			}) : I ? /* @__PURE__ */ f(u, { children: [
				/* @__PURE__ */ f(D, {
					gap: "density-md",
					children: [
						/* @__PURE__ */ d(it, {
							label: "Conversations",
							value: I.conversations
						}),
						I.running ? /* @__PURE__ */ d(it, {
							label: "Running",
							value: I.running
						}) : null,
						/* @__PURE__ */ d(it, {
							label: "Passed scoring",
							value: F ? `${j} of ${F}` : "Not scored yet"
						}),
						/* @__PURE__ */ d(it, {
							label: "Failed to run",
							value: I.failed
						})
					]
				}),
				F > j ? /* @__PURE__ */ f(M, {
					gap: "density-sm",
					children: [/* @__PURE__ */ d(P, {
						kind: "body/semibold/md",
						children: B ? `Where it ${B.charAt(0).toLowerCase()}${B.slice(1)}` : "Where it fails"
					}), /* @__PURE__ */ d(at, {
						rows: w,
						probeLabel: de,
						byWrongPick: !!B
					})]
				}) : F ? /* @__PURE__ */ d(P, {
					kind: "body/regular/sm",
					className: "text-subtle",
					children: "Every scored conversation passed."
				}) : null,
				/* @__PURE__ */ d(P, {
					kind: "body/regular/sm",
					className: "text-subtle",
					children: "Select a conversation to read it. Conversations appear here as they finish."
				}),
				/* @__PURE__ */ d(y, {
					dataViewState: T,
					makeColumns: fe,
					onRowClick: (e) => A(w.indexOf(e)),
					renderBulkActions: ({ selectedRows: n, table: r }) => {
						let i = n.filter((e) => e.status !== "running");
						if (!o || !i.length) return null;
						let { target: s, probes: c, locale: l, max_turns: u, scenarios: d } = o.spec, p = {
							target: s,
							probes: c,
							locale: l,
							users: i.length,
							...u ? { max_turns: u } : {},
							...d ? { scenarios: d } : {},
							rerun: {
								job: e,
								conversations: i.map((e) => e.trajectory_id)
							}
						};
						return /* @__PURE__ */ f(S, {
							kind: "tertiary",
							disabled: ce.isPending,
							onClick: () => ce.mutate(p, {
								onSuccess: (e) => {
									r.resetRowSelection(), a.notify(`Rerunning as ${e.name}.`, "success"), ae(Me(t, e.name));
								},
								onError: (e) => a.notify(e.message, "error")
							}),
							children: [
								"Rerun ",
								i.length,
								" ",
								i.length === 1 ? "conversation" : "conversations"
							]
						});
					},
					attributes: { DataViewRoot: {
						data: w,
						totalCount: w.length,
						reactTableOptions: { getRowId: (e) => e.trajectory_id }
					} }
				}),
				/* @__PURE__ */ d(Ye, {
					rows: w,
					index: k,
					transcripts: ee.data,
					outcome: nt,
					agent: z,
					model: R,
					probes: ue,
					onIndex: A,
					onClose: () => A(null)
				})
			] }) : /* @__PURE__ */ d(b, {
				header: h ? "No results" : "Starting",
				emptyMessage: h ? "This run ended before any conversation started. Check the job logs." : "Sampling personas. Conversations appear here as soon as they start."
			}),
			o ? /* @__PURE__ */ d(tt, {
				open: ie,
				onClose: () => O(!1),
				from: {
					job: e,
					spec: o.spec
				}
			}, String(ie)) : null,
			/* @__PURE__ */ d(ke, {
				jobs: [e],
				open: te,
				onClose: () => E(!1)
			}),
			/* @__PURE__ */ d(p, {
				open: ne,
				onClose: () => re(!1),
				onNotify: a.notify,
				simpleConfirm: !0,
				title: `Delete ${e}?`,
				description: "The run and its results are deleted. Its conversations stay in Traces, and training data you already exported is kept.",
				successText: "Run deleted.",
				errorText: "The run couldn't be deleted.",
				onDelete: async () => (await oe.mutateAsync([e]), ae(je(t)), !0)
			})
		]
	});
}, st = (e, t) => `${e} ${t}${e === 1 ? "" : "s"}`, $ = (e) => e.spec?.target?.agent ?? e.spec?.target?.model ?? "", ct = (e) => [...new Set(e)].sort().map((e) => ({
	value: e,
	label: e
})), lt = {
	name: (e) => e.name,
	target: $,
	users: (e) => e.spec?.users ?? -1,
	status: (e) => e.status ?? "",
	created: (e) => e.created_at ?? ""
}, ut = (e) => (t, { rowSelectionColumn: n }) => [
	n({ size: 48 }),
	t.accessor("name", {
		header: "Run",
		size: 220,
		enableSorting: !0
	}),
	t.accessor($, {
		id: "target",
		header: "Tested",
		size: 200,
		enableSorting: !0,
		meta: { filter: {
			type: "multi-select",
			label: "Tested",
			options: ct(e.map($))
		} },
		cell: ({ row: e }) => $(e.original)
	}),
	t.accessor((e) => (e.spec?.probes ?? []).join(", "), {
		id: "probes",
		header: "Probes",
		size: 260,
		enableSorting: !1,
		meta: { filter: {
			type: "multi-select",
			label: "Probes",
			options: ct(e.flatMap((e) => e.spec?.probes ?? []))
		} },
		cell: ({ row: e }) => e.original.spec.probes?.join(", ")
	}),
	t.accessor((e) => e.spec?.users, {
		id: "users",
		header: "Conversations",
		size: 140,
		enableSorting: !0,
		cell: ({ row: e }) => e.original.spec.users
	}),
	t.accessor((e) => e.status ?? "", {
		id: "status",
		header: "Status",
		size: 120,
		enableSorting: !0,
		meta: { filter: {
			type: "multi-select",
			label: "Status",
			options: ct(e.flatMap((e) => e.status ? [e.status] : []))
		} },
		cell: ({ row: e }) => /* @__PURE__ */ d(v, { status: e.original.status })
	}),
	t.accessor((e) => e.created_at ?? "", {
		id: "created",
		header: "Started",
		size: 140,
		enableSorting: !0,
		cell: ({ row: e }) => e.original.created_at ? /* @__PURE__ */ d(_, { datetime: e.original.created_at }) : null
	})
], dt = (e, t) => {
	let { debouncedSearchBar: n, debouncedColumnFilters: r } = t, i = c(() => {
		let t = n.trim().toLowerCase(), i = Object.fromEntries(r.map((e) => [e.id, e.value])), a = (e, t) => {
			let n = Object.keys(i[e] ?? {}).filter((t) => i[e]?.[t]);
			return !n.length || t.some((e) => n.includes(e));
		};
		return e.filter((e) => (!t || `${e.name} ${$(e)}`.toLowerCase().includes(t)) && a("target", [$(e)]) && a("probes", e.spec?.probes ?? []) && a("status", e.status ? [e.status] : []));
	}, [
		e,
		n,
		r
	]), a = t.sorting.state[0], o = c(() => {
		let e = a && lt[a.id];
		return e ? [...i].sort((t, n) => {
			let [r, i] = [e(t), e(n)], o = typeof r == "number" && typeof i == "number" ? r - i : String(r).localeCompare(String(i));
			return a.desc ? -o : o;
		}) : i;
	}, [i, a]), { pageIndex: s, pageSize: l } = t.pagination.state;
	return {
		page: c(() => o.slice(s * l, (s + 1) * l), [
			o,
			s,
			l
		]),
		total: i.length,
		narrowed: !!n.trim() || r.length > 0
	};
}, ft = () => {
	let { workspaceId: e, breadcrumbs: t, notifications: r } = V(), i = n(), a = x({ defaultSort: [{
		id: "created",
		desc: !0
	}] }), { data: o = [], isPending: m, isError: h } = K(), g = dt(o, a), _ = c(() => ut(o), [o]), [v, ee] = l(!1), [C, w] = l(null), [T, te] = l(null), E = ve(), ne = ye();
	return s(() => (t.set([{ label: "User Simulation" }]), () => t.set([])), [t]), /* @__PURE__ */ f(M, {
		gap: "density-lg",
		className: "p-4",
		children: [
			/* @__PURE__ */ f(D, {
				justify: "between",
				align: "center",
				children: [/* @__PURE__ */ f(M, {
					gap: "density-xs",
					children: [/* @__PURE__ */ d(P, {
						kind: "body/semibold/lg",
						children: "User Simulation"
					}), /* @__PURE__ */ d(P, {
						kind: "body/regular/sm",
						className: "text-subtle",
						children: "Simulated users picked from census-grounded personas talk to your agent. Each conversation is scored and published to Intake for review."
					})]
				}), /* @__PURE__ */ d(S, {
					onClick: () => ee(!0),
					children: "New simulation"
				})]
			}),
			h ? /* @__PURE__ */ d(b, {
				header: "Couldn't load runs",
				emptyMessage: "Check that the user-sim plugin service is running."
			}) : /* @__PURE__ */ d(y, {
				dataViewState: a,
				makeColumns: _,
				searchField: "name",
				onRowClick: (t) => i(Me(e, t.name)),
				renderBulkActions: ({ selectedRows: e, table: t }) => {
					let n = e.filter((e) => e.status === "completed"), i = e.filter((e) => !U(e.status)), a = (e) => e.map((e) => e.name);
					return /* @__PURE__ */ f(u, { children: [
						n.length ? /* @__PURE__ */ f(S, {
							kind: "tertiary",
							onClick: () => w(a(n)),
							children: ["Export ", st(n.length, "completed run")]
						}) : null,
						i.length ? /* @__PURE__ */ f(S, {
							kind: "tertiary",
							onClick: () => ne.mutate(a(i), {
								onSuccess: () => {
									t.resetRowSelection(), r.notify(`Stopping ${st(i.length, "run")}.`, "success");
								},
								onError: (e) => r.notify(e.message, "error")
							}),
							children: ["Stop ", st(i.length, "running run")]
						}) : null,
						/* @__PURE__ */ f(S, {
							kind: "tertiary",
							color: "danger",
							onClick: () => te(a(e)),
							children: ["Delete ", st(e.length, "run")]
						})
					] });
				},
				attributes: {
					DataViewSearchBar: { placeholder: "Search runs…" },
					DataViewRoot: {
						data: g.page,
						totalCount: g.total,
						requestStatus: m ? "loading" : void 0,
						reactTableOptions: { getRowId: (e) => e.name }
					},
					DataViewTableContent: { renderEmptyState: () => g.narrowed ? /* @__PURE__ */ d(b, {
						header: "No matching simulations",
						emptyMessage: "Change the search or filters."
					}) : /* @__PURE__ */ d(b, {
						header: "No simulations yet",
						emptyMessage: "Start one here, or run nemo user-sim run from the CLI."
					}) }
				}
			}),
			/* @__PURE__ */ d(tt, {
				open: v,
				onClose: () => ee(!1)
			}),
			C ? /* @__PURE__ */ d(ke, {
				jobs: C,
				open: !0,
				onClose: () => w(null)
			}) : null,
			/* @__PURE__ */ d(p, {
				open: !!T,
				onClose: () => te(null),
				onNotify: r.notify,
				simpleConfirm: !0,
				title: `Delete ${T?.length ?? 0} simulation run(s)?`,
				description: "The runs and their results are deleted. Their conversations stay in Traces, and training data you already exported is kept.",
				successText: "Runs deleted.",
				errorText: "Some runs couldn't be deleted.",
				onDelete: async () => T ? (await E.mutateAsync(T), !0) : !1
			})
		]
	});
}, pt = ({ host: n }) => /* @__PURE__ */ d(pe, {
	host: n,
	children: /* @__PURE__ */ f(t, { children: [/* @__PURE__ */ d(e, {
		index: !0,
		element: /* @__PURE__ */ d(ft, {})
	}), /* @__PURE__ */ d(e, {
		path: ":job",
		element: /* @__PURE__ */ d(ot, {})
	})] })
}), mt = (e) => [{
	group: "User Simulation",
	items: [{
		id: "user-sim-runs",
		iconName: "users",
		label: "Simulations",
		href: je(e)
	}]
}];
//#endregion
export { pt as Root, mt as navItems };
