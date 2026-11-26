// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import * as e from "react";
import t, { createContext as n, createElement as r, forwardRef as i, useCallback as a, useContext as o, useEffect as s, useLayoutEffect as c, useMemo as l, useRef as u, useState as d } from "react";
import { Fragment as f, jsx as p, jsxs as m } from "react/jsx-runtime";
import { AccessibleTitle as h, AccordionSection as g, CJobCancellableStatuses as _, CJobTerminalStatuses as v, ConfirmationModal as y, ControlledSelect as b, ControlledTextInput as x, CreateSecretModal as S, DeleteConfirmationModal as C, ExpandableMessage as w, FileUpload as T, FormModal as E, JOB_POLLING_INTERVAL_MS as D, QuickActionsMenuRoot as O, RadioCard as ee, RelativeTime as k, StatusBadge as A, StudioDataView as te, TableEmptyState as j, getApiErrorMessage as ne, getJobRefetchInterval as re, getSortParam as M, toCsv as ie, toError as N, triggerDownload as ae, useStudioDataViewState as oe, withOperators as se } from "@nemo/common";
import { AccordionRoot as ce, Badge as le, Banner as ue, Button as P, Card as de, Checkbox as fe, Flex as F, FormField as pe, Grid as me, PageHeader as he, Panel as ge, RadioGroupRoot as _e, SegmentedControl as ve, SelectContent as ye, SelectItem as be, SelectListbox as xe, SelectRoot as Se, SelectTrigger as Ce, Spinner as we, Stack as I, Switch as Te, TabsContent as Ee, TabsList as De, TabsRoot as Oe, TabsTrigger as ke, Text as L, TextArea as Ae, TextInput as je } from "@nvidia/foundations-react-core";
import { Link as Me, Outlet as Ne, Route as Pe, Routes as Fe, useNavigate as Ie, useParams as Le } from "react-router";
import { keepPreviousData as Re, useMutation as ze, useQuery as Be, useQueryClient as Ve } from "@tanstack/react-query";
import { createPortal as He } from "react-dom";
//#region \0rolldown/runtime.js
var Ue = Object.create, We = Object.defineProperty, Ge = Object.getOwnPropertyDescriptor, Ke = Object.getOwnPropertyNames, qe = Object.getPrototypeOf, Je = Object.prototype.hasOwnProperty, R = (e, t, n) => () => {
	if (n) throw n[0];
	try {
		return e && (t = e(e = 0)), t;
	} catch (e) {
		throw n = [e], e;
	}
}, Ye = (e, t) => () => (t || (e((t = { exports: {} }).exports, t), e = null), t.exports), z = (e, t) => {
	let n = {};
	for (var r in e) We(n, r, {
		get: e[r],
		enumerable: !0
	});
	return t || We(n, Symbol.toStringTag, { value: "Module" }), n;
}, Xe = (e, t, n, r) => {
	if (t && typeof t == "object" || typeof t == "function") for (var i = Ke(t), a = 0, o = i.length, s; a < o; a++) s = i[a], !Je.call(e, s) && s !== n && We(e, s, {
		get: ((e) => t[e]).bind(null, s),
		enumerable: !(r = Ge(t, s)) || r.enumerable
	});
	return e;
}, Ze = (e, t, n) => (n = e == null ? {} : Ue(qe(e)), Xe(t || !e || !e.__esModule || !Je.call(e, "default") ? We(n, "default", {
	value: e,
	enumerable: !0
}) : n, e));
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/bind.js
function Qe(e, t) {
	return function() {
		return e.apply(t, arguments);
	};
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/utils.js
var { toString: $e } = Object.prototype, { getPrototypeOf: et } = Object, { iterator: tt, toStringTag: nt } = Symbol, rt = (({ hasOwnProperty: e }) => (t, n) => e.call(t, n))(Object.prototype), it = (e, t) => {
	let n = e, r = [];
	for (; n != null && n !== Object.prototype;) {
		if (r.indexOf(n) !== -1) return !1;
		if (r.push(n), rt(n, t)) return !0;
		n = et(n);
	}
	return !1;
}, at = (e, t) => e != null && it(e, t) ? e[t] : void 0, ot = ((e) => (t) => {
	let n = $e.call(t);
	return e[n] || (e[n] = n.slice(8, -1).toLowerCase());
})(Object.create(null)), st = (e) => (e = e.toLowerCase(), (t) => ot(t) === e), ct = (e) => (t) => typeof t === e, { isArray: lt } = Array, ut = ct("undefined");
function dt(e) {
	return e !== null && !ut(e) && e.constructor !== null && !ut(e.constructor) && ht(e.constructor.isBuffer) && e.constructor.isBuffer(e);
}
var ft = st("ArrayBuffer");
function pt(e) {
	let t;
	return t = typeof ArrayBuffer < "u" && ArrayBuffer.isView ? ArrayBuffer.isView(e) : e && e.buffer && ft(e.buffer), t;
}
var mt = ct("string"), ht = ct("function"), gt = ct("number"), _t = (e) => typeof e == "object" && !!e, vt = (e) => e === !0 || e === !1, yt = (e) => {
	if (!_t(e)) return !1;
	let t = et(e);
	return (t === null || t === Object.prototype || et(t) === null) && !it(e, nt) && !it(e, tt);
}, bt = (e) => {
	if (!_t(e) || dt(e)) return !1;
	try {
		return Object.keys(e).length === 0 && Object.getPrototypeOf(e) === Object.prototype;
	} catch {
		return !1;
	}
}, xt = st("Date"), St = st("File"), Ct = (e) => !!(e && e.uri !== void 0), wt = (e) => e && e.getParts !== void 0, Tt = st("Blob"), Et = st("FileList"), Dt = (e) => _t(e) && ht(e.pipe);
function Ot() {
	return typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : typeof window < "u" ? window : typeof global < "u" ? global : {};
}
var kt = Ot(), At = kt.FormData === void 0 ? void 0 : kt.FormData, jt = (e) => {
	if (!e) return !1;
	if (At && e instanceof At) return !0;
	let t = et(e);
	if (!t || t === Object.prototype || !ht(e.append)) return !1;
	let n = ot(e);
	return n === "formdata" || n === "object" && ht(e.toString) && e.toString() === "[object FormData]";
}, Mt = st("URLSearchParams"), [Nt, Pt, Ft, It] = [
	"ReadableStream",
	"Request",
	"Response",
	"Headers"
].map(st), Lt = (e) => e.trim ? e.trim() : e.replace(/^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, "");
function Rt(e, t, { allOwnKeys: n = !1 } = {}) {
	if (e == null) return;
	let r, i;
	if (typeof e != "object" && (e = [e]), lt(e)) for (r = 0, i = e.length; r < i; r++) t.call(null, e[r], r, e);
	else {
		if (dt(e)) return;
		let i = n ? Object.getOwnPropertyNames(e) : Object.keys(e), a = i.length, o;
		for (r = 0; r < a; r++) o = i[r], t.call(null, e[o], o, e);
	}
}
function zt(e, t) {
	if (dt(e)) return null;
	t = t.toLowerCase();
	let n = Object.keys(e), r = n.length, i;
	for (; r-- > 0;) if (i = n[r], t === i.toLowerCase()) return i;
	return null;
}
var Bt = typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : typeof window < "u" ? window : global, Vt = (e) => !ut(e) && e !== Bt;
function Ht(...e) {
	let { caseless: t, skipUndefined: n } = Vt(this) && this || {}, r = {}, i = (e, i) => {
		if (i === "__proto__" || i === "constructor" || i === "prototype") return;
		let a = t && typeof i == "string" && zt(r, i) || i, o = rt(r, a) ? r[a] : void 0;
		yt(o) && yt(e) ? r[a] = Ht(o, e) : yt(e) ? r[a] = Ht({}, e) : lt(e) ? r[a] = e.slice() : (!n || !ut(e)) && (r[a] = e);
	};
	for (let t = 0, n = e.length; t < n; t++) {
		let n = e[t];
		if (!n || dt(n) || (Rt(n, i), typeof n != "object" || lt(n))) continue;
		let r = Object.getOwnPropertySymbols(n);
		for (let e = 0; e < r.length; e++) {
			let t = r[e];
			en.call(n, t) && i(n[t], t);
		}
	}
	return r;
}
var Ut = (e, t, n, { allOwnKeys: r } = {}) => (Rt(t, (t, r) => {
	n && ht(t) ? Object.defineProperty(e, r, {
		__proto__: null,
		value: Qe(t, n),
		writable: !0,
		enumerable: !0,
		configurable: !0
	}) : Object.defineProperty(e, r, {
		__proto__: null,
		value: t,
		writable: !0,
		enumerable: !0,
		configurable: !0
	});
}, { allOwnKeys: r }), e), Wt = (e) => (e.charCodeAt(0) === 65279 && (e = e.slice(1)), e), Gt = (e, t, n, r) => {
	e.prototype = Object.create(t.prototype, r), Object.defineProperty(e.prototype, "constructor", {
		__proto__: null,
		value: e,
		writable: !0,
		enumerable: !1,
		configurable: !0
	}), Object.defineProperty(e, "super", {
		__proto__: null,
		value: t.prototype
	}), n && Object.assign(e.prototype, n);
}, Kt = (e, t, n, r) => {
	let i, a, o, s = {};
	if (t ||= {}, e == null) return t;
	do {
		for (i = Object.getOwnPropertyNames(e), a = i.length; a-- > 0;) o = i[a], (!r || r(o, e, t)) && !s[o] && (t[o] = e[o], s[o] = !0);
		e = n !== !1 && et(e);
	} while (e && (!n || n(e, t)) && e !== Object.prototype);
	return t;
}, qt = (e, t, n) => {
	e = String(e), (n === void 0 || n > e.length) && (n = e.length), n -= t.length;
	let r = e.indexOf(t, n);
	return r !== -1 && r === n;
}, Jt = (e) => {
	if (!e) return null;
	if (lt(e)) return e;
	let t = e.length;
	if (!gt(t)) return null;
	let n = Array(t);
	for (; t-- > 0;) n[t] = e[t];
	return n;
}, Yt = ((e) => (t) => e && t instanceof e)(typeof Uint8Array < "u" && et(Uint8Array)), Xt = (e, t) => {
	let n = (e && e[tt]).call(e), r;
	for (; (r = n.next()) && !r.done;) {
		let n = r.value;
		t.call(e, n[0], n[1]);
	}
}, Zt = (e, t) => {
	let n, r = [];
	for (; (n = e.exec(t)) !== null;) r.push(n);
	return r;
}, Qt = st("HTMLFormElement"), $t = (e) => e.toLowerCase().replace(/[-_\s]([a-z\d])(\w*)/g, function(e, t, n) {
	return t.toUpperCase() + n;
}), { propertyIsEnumerable: en } = Object.prototype, tn = st("RegExp"), nn = (e, t) => {
	let n = Object.getOwnPropertyDescriptors(e), r = {};
	Rt(n, (n, i) => {
		let a;
		(a = t(n, i, e)) !== !1 && (r[i] = a || n);
	}), Object.defineProperties(e, r);
}, rn = (e) => {
	nn(e, (t, n) => {
		if (ht(e) && [
			"arguments",
			"caller",
			"callee"
		].includes(n)) return !1;
		let r = e[n];
		if (ht(r)) {
			if (t.enumerable = !1, "writable" in t) {
				t.writable = !1;
				return;
			}
			t.set ||= () => {
				throw Error("Can not rewrite read-only method '" + n + "'");
			};
		}
	});
}, an = (e, t) => {
	let n = {}, r = (e) => {
		e.forEach((e) => {
			n[e] = !0;
		});
	};
	return lt(e) ? r(e) : r(String(e).split(t)), n;
}, on = () => {}, sn = (e, t) => e != null && Number.isFinite(e = +e) ? e : t;
function cn(e) {
	return !!(e && ht(e.append) && e[nt] === "FormData" && e[tt]);
}
var ln = (e) => {
	let t = /* @__PURE__ */ new WeakSet(), n = (e) => {
		if (_t(e)) {
			if (t.has(e)) return;
			if (dt(e)) return e;
			if (!("toJSON" in e)) {
				t.add(e);
				let r = lt(e) ? [] : {};
				return Rt(e, (e, t) => {
					let i = n(e);
					!ut(i) && (r[t] = i);
				}), t.delete(e), r;
			}
		}
		return e;
	};
	return n(e);
}, un = st("AsyncFunction"), dn = (e) => e && (_t(e) || ht(e)) && ht(e.then) && ht(e.catch), fn = ((e, t) => e ? setImmediate : t ? ((e, t) => (Bt.addEventListener("message", ({ source: n, data: r }) => {
	n === Bt && r === e && t.length && t.shift()();
}, !1), (n) => {
	t.push(n), Bt.postMessage(e, "*");
}))(`axios@${Math.random()}`, []) : (e) => setTimeout(e))(typeof setImmediate == "function", ht(Bt.postMessage)), pn = typeof queueMicrotask < "u" ? queueMicrotask.bind(Bt) : typeof process < "u" && process.nextTick || fn, mn = (e) => e != null && ht(e[tt]), B = {
	isArray: lt,
	isArrayBuffer: ft,
	isBuffer: dt,
	isFormData: jt,
	isArrayBufferView: pt,
	isString: mt,
	isNumber: gt,
	isBoolean: vt,
	isObject: _t,
	isPlainObject: yt,
	isEmptyObject: bt,
	isReadableStream: Nt,
	isRequest: Pt,
	isResponse: Ft,
	isHeaders: It,
	isUndefined: ut,
	isDate: xt,
	isFile: St,
	isReactNativeBlob: Ct,
	isReactNative: wt,
	isBlob: Tt,
	isRegExp: tn,
	isFunction: ht,
	isStream: Dt,
	isURLSearchParams: Mt,
	isTypedArray: Yt,
	isFileList: Et,
	forEach: Rt,
	merge: Ht,
	extend: Ut,
	trim: Lt,
	stripBOM: Wt,
	inherits: Gt,
	toFlatObject: Kt,
	kindOf: ot,
	kindOfTest: st,
	endsWith: qt,
	toArray: Jt,
	forEachEntry: Xt,
	matchAll: Zt,
	isHTMLForm: Qt,
	hasOwnProperty: rt,
	hasOwnProp: rt,
	hasOwnInPrototypeChain: it,
	getSafeProp: at,
	reduceDescriptors: nn,
	freezeMethods: rn,
	toObjectSet: an,
	toCamelCase: $t,
	noop: on,
	toFiniteNumber: sn,
	findKey: zt,
	global: Bt,
	isContextDefined: Vt,
	isSpecCompliantForm: cn,
	toJSONObject: ln,
	isAsyncFn: un,
	isThenable: dn,
	setImmediate: fn,
	asap: pn,
	isIterable: mn,
	isSafeIterable: (e) => e != null && it(e, tt) && mn(e)
}, hn = B.toObjectSet([
	"age",
	"authorization",
	"content-length",
	"content-type",
	"etag",
	"expires",
	"from",
	"host",
	"if-modified-since",
	"if-unmodified-since",
	"last-modified",
	"location",
	"max-forwards",
	"proxy-authorization",
	"referer",
	"retry-after",
	"user-agent"
]), gn = (e) => {
	let t = {}, n, r, i;
	return e && e.split("\n").forEach(function(e) {
		i = e.indexOf(":"), n = e.substring(0, i).trim().toLowerCase(), r = e.substring(i + 1).trim(), !(!n || t[n] && hn[n]) && (n === "set-cookie" ? t[n] ? t[n].push(r) : t[n] = [r] : t[n] = t[n] ? t[n] + ", " + r : r);
	}), t;
};
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/sanitizeHeaderValue.js
function _n(e) {
	let t = 0, n = e.length;
	for (; t < n;) {
		let n = e.charCodeAt(t);
		if (n !== 9 && n !== 32) break;
		t += 1;
	}
	for (; n > t;) {
		let t = e.charCodeAt(n - 1);
		if (t !== 9 && t !== 32) break;
		--n;
	}
	return t === 0 && n === e.length ? e : e.slice(t, n);
}
var vn = /* @__PURE__ */ RegExp("[\\u0000-\\u0008\\u000a-\\u001f\\u007f]+", "g"), yn = /* @__PURE__ */ RegExp("[^\\u0009\\u0020-\\u007e\\u0080-\\u00ff]+", "g");
function bn(e, t) {
	return B.isArray(e) ? e.map((e) => bn(e, t)) : _n(String(e).replace(t, ""));
}
var xn = (e) => bn(e, vn), Sn = (e) => bn(e, yn);
function Cn(e) {
	let t = Object.create(null);
	return B.forEach(e.toJSON(), (e, n) => {
		t[n] = Sn(e);
	}), t;
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/core/AxiosHeaders.js
var wn = Symbol("internals");
function Tn(e) {
	return e && String(e).trim().toLowerCase();
}
function En(e) {
	return e === !1 || e == null ? e : B.isArray(e) ? e.map(En) : xn(String(e));
}
function Dn(e) {
	let t = Object.create(null), n = /([^\s,;=]+)\s*(?:=\s*([^,;]+))?/g, r;
	for (; r = n.exec(e);) t[r[1]] = r[2];
	return t;
}
var On = (e) => /^[-_a-zA-Z0-9^`|~,!#$%&'*+.]+$/.test(e.trim());
function kn(e, t, n, r, i) {
	if (B.isFunction(r)) return r.call(this, t, n);
	if (i && (t = n), B.isString(t)) {
		if (B.isString(r)) return t.indexOf(r) !== -1;
		if (B.isRegExp(r)) return r.test(t);
	}
}
function An(e) {
	return e.trim().toLowerCase().replace(/([a-z\d])(\w*)/g, (e, t, n) => t.toUpperCase() + n);
}
function jn(e, t) {
	let n = B.toCamelCase(" " + t);
	[
		"get",
		"set",
		"has"
	].forEach((r) => {
		Object.defineProperty(e, r + n, {
			__proto__: null,
			value: function(e, n, i) {
				return this[r].call(this, t, e, n, i);
			},
			configurable: !0
		});
	});
}
var Mn = class {
	constructor(e) {
		e && this.set(e);
	}
	set(e, t, n) {
		let r = this;
		function i(e, t, n) {
			let i = Tn(t);
			if (!i) return;
			let a = B.findKey(r, i);
			(!a || r[a] === void 0 || n === !0 || n === void 0 && r[a] !== !1) && (r[a || t] = En(e));
		}
		let a = (e, t) => B.forEach(e, (e, n) => i(e, n, t));
		if (B.isPlainObject(e) || e instanceof this.constructor) a(e, t);
		else if (B.isString(e) && (e = e.trim()) && !On(e)) a(gn(e), t);
		else if (B.isObject(e) && B.isSafeIterable(e)) {
			let n = Object.create(null), r, i;
			for (let t of e) {
				if (!B.isArray(t)) throw TypeError("Object iterator must return a key-value pair");
				i = t[0], B.hasOwnProp(n, i) ? (r = n[i], n[i] = B.isArray(r) ? [...r, t[1]] : [r, t[1]]) : n[i] = t[1];
			}
			a(n, t);
		} else e != null && i(t, e, n);
		return this;
	}
	get(e, t) {
		if (e = Tn(e), e) {
			let n = B.findKey(this, e);
			if (n) {
				let e = this[n];
				if (!t) return e;
				if (t === !0) return Dn(e);
				if (B.isFunction(t)) return t.call(this, e, n);
				if (B.isRegExp(t)) return t.exec(e);
				throw TypeError("parser must be boolean|regexp|function");
			}
		}
	}
	has(e, t) {
		if (e = Tn(e), e) {
			let n = B.findKey(this, e);
			return !!(n && this[n] !== void 0 && (!t || kn(this, this[n], n, t)));
		}
		return !1;
	}
	delete(e, t) {
		let n = this, r = !1;
		function i(e) {
			if (e = Tn(e), e) {
				let i = B.findKey(n, e);
				i && (!t || kn(n, n[i], i, t)) && (delete n[i], r = !0);
			}
		}
		return B.isArray(e) ? e.forEach(i) : i(e), r;
	}
	clear(e) {
		let t = Object.keys(this), n = t.length, r = !1;
		for (; n--;) {
			let i = t[n];
			(!e || kn(this, this[i], i, e, !0)) && (delete this[i], r = !0);
		}
		return r;
	}
	normalize(e) {
		let t = this, n = {};
		return B.forEach(this, (r, i) => {
			let a = B.findKey(n, i);
			if (a) {
				t[a] = En(r), delete t[i];
				return;
			}
			let o = e ? An(i) : String(i).trim();
			o !== i && delete t[i], t[o] = En(r), n[o] = !0;
		}), this;
	}
	concat(...e) {
		return this.constructor.concat(this, ...e);
	}
	toJSON(e) {
		let t = Object.create(null);
		return B.forEach(this, (n, r) => {
			n != null && n !== !1 && (t[r] = e && B.isArray(n) ? n.join(", ") : n);
		}), t;
	}
	[Symbol.iterator]() {
		return Object.entries(this.toJSON())[Symbol.iterator]();
	}
	toString() {
		return Object.entries(this.toJSON()).map(([e, t]) => e + ": " + t).join("\n");
	}
	getSetCookie() {
		return this.get("set-cookie") || [];
	}
	get [Symbol.toStringTag]() {
		return "AxiosHeaders";
	}
	static from(e) {
		return e instanceof this ? e : new this(e);
	}
	static concat(e, ...t) {
		let n = new this(e);
		return t.forEach((e) => n.set(e)), n;
	}
	static accessor(e) {
		let t = (this[wn] = this[wn] = { accessors: {} }).accessors, n = this.prototype;
		function r(e) {
			let r = Tn(e);
			t[r] || (jn(n, e), t[r] = !0);
		}
		return B.isArray(e) ? e.forEach(r) : r(e), this;
	}
};
Mn.accessor([
	"Content-Type",
	"Content-Length",
	"Accept",
	"Accept-Encoding",
	"User-Agent",
	"Authorization"
]), B.reduceDescriptors(Mn.prototype, ({ value: e }, t) => {
	let n = t[0].toUpperCase() + t.slice(1);
	return {
		get: () => e,
		set(e) {
			this[n] = e;
		}
	};
}), B.freezeMethods(Mn);
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/core/AxiosError.js
var Nn = "[REDACTED ****]";
function Pn(e) {
	if (B.hasOwnProp(e, "toJSON")) return !0;
	let t = Object.getPrototypeOf(e);
	for (; t && t !== Object.prototype;) {
		if (B.hasOwnProp(t, "toJSON")) return !0;
		t = Object.getPrototypeOf(t);
	}
	return !1;
}
function Fn(e, t) {
	let n = new Set(t.map((e) => String(e).toLowerCase())), r = [], i = (e) => {
		if (typeof e != "object" || !e || B.isBuffer(e)) return e;
		if (r.indexOf(e) !== -1) return;
		e instanceof Mn && (e = e.toJSON()), r.push(e);
		let t;
		if (B.isArray(e)) t = [], e.forEach((e, n) => {
			let r = i(e);
			B.isUndefined(r) || (t[n] = r);
		});
		else {
			if (!B.isPlainObject(e) && Pn(e)) return r.pop(), e;
			t = Object.create(null);
			for (let [r, a] of Object.entries(e)) {
				let e = n.has(r.toLowerCase()) ? Nn : i(a);
				B.isUndefined(e) || (t[r] = e);
			}
		}
		return r.pop(), t;
	};
	return i(e);
}
var V = class e extends Error {
	static from(t, n, r, i, a, o) {
		let s = new e(t.message, n || t.code, r, i, a);
		return s.cause = t, s.name = t.name, t.status != null && s.status == null && (s.status = t.status), o && Object.assign(s, o), s;
	}
	constructor(e, t, n, r, i) {
		super(e), Object.defineProperty(this, "message", {
			__proto__: null,
			value: e,
			enumerable: !0,
			writable: !0,
			configurable: !0
		}), this.name = "AxiosError", this.isAxiosError = !0, t && (this.code = t), n && (this.config = n), r && (this.request = r), i && (this.response = i, this.status = i.status);
	}
	toJSON() {
		let e = this.config, t = e && B.hasOwnProp(e, "redact") ? e.redact : void 0, n = B.isArray(t) && t.length > 0 ? Fn(e, t) : B.toJSONObject(e);
		return {
			message: this.message,
			name: this.name,
			description: this.description,
			number: this.number,
			fileName: this.fileName,
			lineNumber: this.lineNumber,
			columnNumber: this.columnNumber,
			stack: this.stack,
			config: n,
			code: this.code,
			status: this.status
		};
	}
};
V.ERR_BAD_OPTION_VALUE = "ERR_BAD_OPTION_VALUE", V.ERR_BAD_OPTION = "ERR_BAD_OPTION", V.ECONNABORTED = "ECONNABORTED", V.ETIMEDOUT = "ETIMEDOUT", V.ECONNREFUSED = "ECONNREFUSED", V.ERR_NETWORK = "ERR_NETWORK", V.ERR_FR_TOO_MANY_REDIRECTS = "ERR_FR_TOO_MANY_REDIRECTS", V.ERR_DEPRECATED = "ERR_DEPRECATED", V.ERR_BAD_RESPONSE = "ERR_BAD_RESPONSE", V.ERR_BAD_REQUEST = "ERR_BAD_REQUEST", V.ERR_CANCELED = "ERR_CANCELED", V.ERR_NOT_SUPPORT = "ERR_NOT_SUPPORT", V.ERR_INVALID_URL = "ERR_INVALID_URL", V.ERR_FORM_DATA_DEPTH_EXCEEDED = "ERR_FORM_DATA_DEPTH_EXCEEDED";
function In(e) {
	return B.isPlainObject(e) || B.isArray(e);
}
function Ln(e) {
	return B.endsWith(e, "[]") ? e.slice(0, -2) : e;
}
function Rn(e, t, n) {
	return e ? e.concat(t).map(function(e, t) {
		return e = Ln(e), !n && t ? "[" + e + "]" : e;
	}).join(n ? "." : "") : t;
}
function zn(e) {
	return B.isArray(e) && !e.some(In);
}
var Bn = B.toFlatObject(B, {}, null, function(e) {
	return /^is[A-Z]/.test(e);
});
function Vn(e, t, n) {
	if (!B.isObject(e)) throw TypeError("target must be an object");
	t ||= new FormData(), n = B.toFlatObject(n, {
		metaTokens: !0,
		dots: !1,
		indexes: !1
	}, !1, function(e, t) {
		return !B.isUndefined(t[e]);
	});
	let r = n.metaTokens, i = n.visitor || m, a = n.dots, o = n.indexes, s = n.Blob || typeof Blob < "u" && Blob, c = n.maxDepth === void 0 ? 100 : n.maxDepth, l = s && B.isSpecCompliantForm(t), u = [];
	if (!B.isFunction(i)) throw TypeError("visitor must be a function");
	function d(e) {
		if (e === null) return "";
		if (B.isDate(e)) return e.toISOString();
		if (B.isBoolean(e)) return e.toString();
		if (!l && B.isBlob(e)) throw new V("Blob is not supported. Use a Buffer instead.");
		return B.isArrayBuffer(e) || B.isTypedArray(e) ? l && typeof Blob == "function" ? new Blob([e]) : Buffer.from(e) : e;
	}
	function f(e) {
		if (e > c) throw new V("Object is too deeply nested (" + e + " levels). Max depth: " + c, V.ERR_FORM_DATA_DEPTH_EXCEEDED);
	}
	function p(e, t) {
		if (c === Infinity) return JSON.stringify(e);
		let n = [];
		return JSON.stringify(e, function(e, r) {
			if (!B.isObject(r)) return r;
			for (; n.length && n[n.length - 1] !== this;) n.pop();
			return n.push(r), f(t + n.length - 1), r;
		});
	}
	function m(e, n, i) {
		let s = e;
		if (B.isReactNative(t) && B.isReactNativeBlob(e)) return t.append(Rn(i, n, a), d(e)), !1;
		if (e && !i && typeof e == "object") {
			if (B.endsWith(n, "{}")) n = r ? n : n.slice(0, -2), e = p(e, 1);
			else if (B.isArray(e) && zn(e) || (B.isFileList(e) || B.endsWith(n, "[]")) && (s = B.toArray(e))) return n = Ln(n), s.forEach(function(e, r) {
				!(B.isUndefined(e) || e === null) && t.append(o === !0 ? Rn([n], r, a) : o === null ? n : n + "[]", d(e));
			}), !1;
		}
		return In(e) ? !0 : (t.append(Rn(i, n, a), d(e)), !1);
	}
	let h = Object.assign(Bn, {
		defaultVisitor: m,
		convertValue: d,
		isVisitable: In
	});
	function g(e, n, r = 0) {
		if (!B.isUndefined(e)) {
			if (f(r), u.indexOf(e) !== -1) throw Error("Circular reference detected in " + n.join("."));
			u.push(e), B.forEach(e, function(e, a) {
				(!(B.isUndefined(e) || e === null) && i.call(t, e, B.isString(a) ? a.trim() : a, n, h)) === !0 && g(e, n ? n.concat(a) : [a], r + 1);
			}), u.pop();
		}
	}
	if (!B.isObject(e)) throw TypeError("data must be an object");
	return g(e), t;
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/AxiosURLSearchParams.js
function Hn(e) {
	let t = {
		"!": "%21",
		"'": "%27",
		"(": "%28",
		")": "%29",
		"~": "%7E",
		"%20": "+"
	};
	return encodeURIComponent(e).replace(/[!'()~]|%20/g, function(e) {
		return t[e];
	});
}
function Un(e, t) {
	this._pairs = [], e && Vn(e, this, t);
}
var Wn = Un.prototype;
Wn.append = function(e, t) {
	this._pairs.push([e, t]);
}, Wn.toString = function(e) {
	let t = e ? function(t) {
		return e.call(this, t, Hn);
	} : Hn;
	return this._pairs.map(function(e) {
		return t(e[0]) + "=" + t(e[1]);
	}, "").join("&");
};
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/buildURL.js
function Gn(e) {
	return encodeURIComponent(e).replace(/%3A/gi, ":").replace(/%24/g, "$").replace(/%2C/gi, ",").replace(/%20/g, "+");
}
function Kn(e, t, n) {
	if (!t) return e;
	let r = B.isFunction(n) ? { serialize: n } : n, i = B.getSafeProp(r, "encode") || Gn, a = B.getSafeProp(r, "serialize"), o;
	if (o = a ? a(t, r) : B.isURLSearchParams(t) ? t.toString() : new Un(t, r).toString(i), o) {
		let t = e.indexOf("#");
		t !== -1 && (e = e.slice(0, t)), e += (e.indexOf("?") === -1 ? "?" : "&") + o;
	}
	return e;
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/core/InterceptorManager.js
var qn = class {
	constructor() {
		this.handlers = [];
	}
	use(e, t, n) {
		return this.handlers.push({
			fulfilled: e,
			rejected: t,
			synchronous: n ? n.synchronous : !1,
			runWhen: n ? n.runWhen : null
		}), this.handlers.length - 1;
	}
	eject(e) {
		this.handlers[e] && (this.handlers[e] = null);
	}
	clear() {
		this.handlers &&= [];
	}
	forEach(e) {
		B.forEach(this.handlers, function(t) {
			t !== null && e(t);
		});
	}
}, Jn = {
	silentJSONParsing: !0,
	forcedJSONParsing: !0,
	clarifyTimeoutError: !1,
	legacyInterceptorReqResOrdering: !0,
	advertiseZstdAcceptEncoding: !1,
	validateStatusUndefinedResolves: !0
}, Yn = {
	isBrowser: !0,
	classes: {
		URLSearchParams: typeof URLSearchParams < "u" ? URLSearchParams : Un,
		FormData: typeof FormData < "u" ? FormData : null,
		Blob: typeof Blob < "u" ? Blob : null
	},
	protocols: [
		"http",
		"https",
		"file",
		"blob",
		"url",
		"data"
	]
}, Xn = /* @__PURE__ */ z({
	hasBrowserEnv: () => Zn,
	hasStandardBrowserEnv: () => $n,
	hasStandardBrowserWebWorkerEnv: () => er,
	navigator: () => Qn,
	origin: () => tr
}), Zn = typeof window < "u" && typeof document < "u", Qn = typeof navigator == "object" && navigator || void 0, $n = Zn && (!Qn || [
	"ReactNative",
	"NativeScript",
	"NS"
].indexOf(Qn.product) < 0), er = typeof WorkerGlobalScope < "u" && self instanceof WorkerGlobalScope && typeof self.importScripts == "function", tr = Zn && window.location.href || "http://localhost", nr = {
	...Xn,
	...Yn
};
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/toURLEncodedForm.js
function rr(e, t) {
	return Vn(e, new nr.classes.URLSearchParams(), {
		visitor: function(e, t, n, r) {
			return nr.isNode && B.isBuffer(e) ? (this.append(t, e.toString("base64")), !1) : r.defaultVisitor.apply(this, arguments);
		},
		...t
	});
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/formDataToJSON.js
var ir = 100;
function ar(e) {
	if (e > ir) throw new V("FormData field is too deeply nested (" + e + " levels). Max depth: " + ir, V.ERR_FORM_DATA_DEPTH_EXCEEDED);
}
function or(e) {
	let t = [], n = /\w+|\[(\w*)]/g, r;
	for (; (r = n.exec(e)) !== null;) ar(t.length), t.push(r[0] === "[]" ? "" : r[1] || r[0]);
	return t;
}
function sr(e) {
	let t = {}, n = Object.keys(e), r, i = n.length, a;
	for (r = 0; r < i; r++) a = n[r], t[a] = e[a];
	return t;
}
function cr(e) {
	function t(e, n, r, i) {
		ar(i);
		let a = e[i++];
		if (a === "__proto__") return !0;
		let o = Number.isFinite(+a), s = i >= e.length;
		return a = !a && B.isArray(r) ? r.length : a, s ? (B.hasOwnProp(r, a) ? r[a] = B.isArray(r[a]) ? r[a].concat(n) : [r[a], n] : r[a] = n, !o) : ((!B.hasOwnProp(r, a) || !B.isObject(r[a])) && (r[a] = []), t(e, n, r[a], i) && B.isArray(r[a]) && (r[a] = sr(r[a])), !o);
	}
	if (B.isFormData(e) && B.isFunction(e.entries)) {
		let n = {};
		return B.forEachEntry(e, (e, r) => {
			t(or(e), r, n, 0);
		}), n;
	}
	return null;
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/defaults/index.js
var lr = (e, t) => e != null && B.hasOwnProp(e, t) ? e[t] : void 0;
function ur(e, t, n) {
	if (B.isString(e)) try {
		return (t || JSON.parse)(e), B.trim(e);
	} catch (e) {
		if (e.name !== "SyntaxError") throw e;
	}
	return (n || JSON.stringify)(e);
}
var dr = {
	transitional: Jn,
	adapter: [
		"xhr",
		"http",
		"fetch"
	],
	transformRequest: [function(e, t) {
		let n = t.getContentType() || "", r = n.indexOf("application/json") > -1, i = B.isObject(e);
		if (i && B.isHTMLForm(e) && (e = new FormData(e)), B.isFormData(e)) return r ? JSON.stringify(cr(e)) : e;
		if (B.isArrayBuffer(e) || B.isBuffer(e) || B.isStream(e) || B.isFile(e) || B.isBlob(e) || B.isReadableStream(e)) return e;
		if (B.isArrayBufferView(e)) return e.buffer;
		if (B.isURLSearchParams(e)) return t.setContentType("application/x-www-form-urlencoded;charset=utf-8", !1), e.toString();
		let a;
		if (i) {
			let t = lr(this, "formSerializer");
			if (n.indexOf("application/x-www-form-urlencoded") > -1) return rr(e, t).toString();
			if ((a = B.isFileList(e)) || n.indexOf("multipart/form-data") > -1) {
				let n = lr(this, "env"), r = n && n.FormData;
				return Vn(a ? { "files[]": e } : e, r && new r(), t);
			}
		}
		return i || r ? (t.setContentType("application/json", !1), ur(e)) : e;
	}],
	transformResponse: [function(e) {
		let t = lr(this, "transitional") || dr.transitional, n = t && t.forcedJSONParsing, r = lr(this, "responseType"), i = r === "json";
		if (B.isResponse(e) || B.isReadableStream(e)) return e;
		if (e && B.isString(e) && (n && !r || i)) {
			let n = !(t && t.silentJSONParsing) && i;
			try {
				return JSON.parse(e, lr(this, "parseReviver"));
			} catch (e) {
				if (n) throw e.name === "SyntaxError" ? V.from(e, V.ERR_BAD_RESPONSE, this, null, lr(this, "response")) : e;
			}
		}
		return e;
	}],
	timeout: 0,
	xsrfCookieName: "XSRF-TOKEN",
	xsrfHeaderName: "X-XSRF-TOKEN",
	maxContentLength: -1,
	maxBodyLength: -1,
	env: {
		FormData: nr.classes.FormData,
		Blob: nr.classes.Blob
	},
	validateStatus: function(e) {
		return e >= 200 && e < 300;
	},
	headers: { common: {
		Accept: "application/json, text/plain, */*",
		"Content-Type": void 0
	} }
};
B.forEach([
	"delete",
	"get",
	"head",
	"post",
	"put",
	"patch",
	"query"
], (e) => {
	dr.headers[e] = {};
});
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/core/transformData.js
function fr(e, t) {
	let n = this || dr, r = t || n, i = Mn.from(r.headers), a = r.data;
	return B.forEach(e, function(e) {
		a = e.call(n, a, i.normalize(), t ? t.status : void 0);
	}), i.normalize(), a;
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/cancel/isCancel.js
function pr(e) {
	return !!(e && e.__CANCEL__);
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/cancel/CanceledError.js
var mr = class extends V {
	constructor(e, t, n) {
		super(e ?? "canceled", V.ERR_CANCELED, t, n), this.name = "CanceledError", this.__CANCEL__ = !0;
	}
};
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/core/settle.js
function hr(e, t, n) {
	let r = n.config.validateStatus;
	!n.status || !r || r(n.status) ? e(n) : t(new V("Request failed with status code " + n.status, n.status >= 400 && n.status < 500 ? V.ERR_BAD_REQUEST : V.ERR_BAD_RESPONSE, n.config, n.request, n));
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/parseProtocol.js
function gr(e) {
	let t = /^([-+\w]{1,25}):(?:\/\/)?/.exec(e);
	return t && t[1] || "";
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/speedometer.js
function _r(e, t) {
	e ||= 10;
	let n = Array(e), r = Array(e), i = 0, a = 0, o;
	return t = t === void 0 ? 1e3 : t, function(s) {
		let c = Date.now(), l = r[a];
		o ||= c, n[i] = s, r[i] = c;
		let u = a, d = 0;
		for (; u !== i;) d += n[u++], u %= e;
		if (i = (i + 1) % e, i === a && (a = (a + 1) % e), c - o < t) return;
		let f = l && c - l;
		return f ? Math.round(d * 1e3 / f) : void 0;
	};
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/throttle.js
function vr(e, t) {
	let n = 0, r = 1e3 / t, i, a, o = (t, r = Date.now()) => {
		n = r, i = null, a &&= (clearTimeout(a), null), e(...t);
	};
	return [(...e) => {
		let t = Date.now(), s = t - n;
		s >= r ? o(e, t) : (i = e, a ||= setTimeout(() => {
			a = null, o(i);
		}, r - s));
	}, () => i && o(i)];
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/progressEventReducer.js
var yr = (e, t, n = 3) => {
	let r = 0, i = _r(50, 250);
	return vr((n) => {
		if (!n || typeof n.loaded != "number") return;
		let a = n.loaded, o = n.lengthComputable ? n.total : void 0, s = o == null ? a : Math.min(a, o), c = Math.max(0, s - r), l = i(c);
		r = Math.max(r, s), e({
			loaded: s,
			total: o,
			progress: o ? s / o : void 0,
			bytes: c,
			rate: l || void 0,
			estimated: l && o ? (o - s) / l : void 0,
			event: n,
			lengthComputable: o != null,
			[t ? "download" : "upload"]: !0
		});
	}, n);
}, br = (e, t) => {
	let n = e != null;
	return [(r) => t[0]({
		lengthComputable: n,
		total: e,
		loaded: r
	}), t[1]];
}, xr = (e) => (...t) => B.asap(() => e(...t)), Sr = nr.hasStandardBrowserEnv ? ((e, t) => (n) => (n = new URL(n, nr.origin), e.protocol === n.protocol && e.host === n.host && (t || e.port === n.port)))(new URL(nr.origin), nr.navigator && /(msie|trident)/i.test(nr.navigator.userAgent)) : () => !0, Cr = nr.hasStandardBrowserEnv ? {
	write(e, t, n, r, i, a, o) {
		if (typeof document > "u") return;
		let s = [`${e}=${encodeURIComponent(t)}`];
		B.isNumber(n) && s.push(`expires=${new Date(n).toUTCString()}`), B.isString(r) && s.push(`path=${r}`), B.isString(i) && s.push(`domain=${i}`), a === !0 && s.push("secure"), B.isString(o) && s.push(`SameSite=${o}`), document.cookie = s.join("; ");
	},
	read(e) {
		if (typeof document > "u") return null;
		let t = document.cookie.split(";");
		for (let n = 0; n < t.length; n++) {
			let r = t[n].replace(/^\s+/, ""), i = r.indexOf("=");
			if (i !== -1 && r.slice(0, i) === e) return decodeURIComponent(r.slice(i + 1));
		}
		return null;
	},
	remove(e) {
		this.write(e, "", Date.now() - 864e5, "/");
	}
} : {
	write() {},
	read() {
		return null;
	},
	remove() {}
};
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/isAbsoluteURL.js
function wr(e) {
	return typeof e == "string" && /^([a-z][a-z\d+\-.]*:)?\/\//i.test(e);
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/combineURLs.js
function Tr(e, t) {
	return t ? e.replace(/\/?\/$/, "") + "/" + t.replace(/^\/+/, "") : e;
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/core/buildFullPath.js
var Er = /^https?:(?!\/\/)/i, Dr = /[\t\n\r]/g;
function Or(e) {
	let t = 0;
	for (; t < e.length && e.charCodeAt(t) <= 32;) t++;
	return e.slice(t);
}
function kr(e) {
	return Or(e).replace(Dr, "");
}
function Ar(e, t) {
	if (typeof e == "string" && Er.test(kr(e))) throw new V("Invalid URL: missing \"//\" after protocol", V.ERR_INVALID_URL, t);
}
function jr(e, t, n, r) {
	Ar(t, r);
	let i = !wr(t);
	return e && (i || n === !1) ? (Ar(e, r), Tr(e, t)) : t;
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/core/mergeConfig.js
var Mr = (e) => e instanceof Mn ? { ...e } : e;
function Nr(e, t) {
	t ||= {};
	let n = Object.create(null);
	Object.defineProperty(n, "hasOwnProperty", {
		__proto__: null,
		value: Object.prototype.hasOwnProperty,
		enumerable: !1,
		writable: !0,
		configurable: !0
	});
	function r(e, t, n, r) {
		return B.isPlainObject(e) && B.isPlainObject(t) ? B.merge.call({ caseless: r }, e, t) : B.isPlainObject(t) ? B.merge({}, t) : B.isArray(t) ? t.slice() : t;
	}
	function i(e, t, n, i) {
		if (!B.isUndefined(t)) return r(e, t, n, i);
		if (!B.isUndefined(e)) return r(void 0, e, n, i);
	}
	function a(e, t) {
		if (!B.isUndefined(t)) return r(void 0, t);
	}
	function o(e, t) {
		if (!B.isUndefined(t)) return r(void 0, t);
		if (!B.isUndefined(e)) return r(void 0, e);
	}
	function s(n) {
		let r = B.hasOwnProp(t, "transitional") ? t.transitional : void 0;
		if (!B.isUndefined(r)) {
			if (B.isPlainObject(r)) {
				if (B.hasOwnProp(r, n)) return r[n];
			} else return;
		}
		let i = B.hasOwnProp(e, "transitional") ? e.transitional : void 0;
		if (B.isPlainObject(i) && B.hasOwnProp(i, n)) return i[n];
	}
	function c(n, i, a) {
		if (B.hasOwnProp(t, a)) return r(n, i);
		if (B.hasOwnProp(e, a)) return r(void 0, n);
	}
	let l = {
		url: a,
		method: a,
		data: a,
		baseURL: o,
		transformRequest: o,
		transformResponse: o,
		paramsSerializer: o,
		timeout: o,
		timeoutMessage: o,
		withCredentials: o,
		withXSRFToken: o,
		adapter: o,
		responseType: o,
		xsrfCookieName: o,
		xsrfHeaderName: o,
		onUploadProgress: o,
		onDownloadProgress: o,
		decompress: o,
		maxContentLength: o,
		maxBodyLength: o,
		beforeRedirect: o,
		transport: o,
		httpAgent: o,
		httpsAgent: o,
		cancelToken: o,
		socketPath: o,
		allowedSocketPaths: o,
		responseEncoding: o,
		validateStatus: c,
		headers: (e, t, n) => i(Mr(e), Mr(t), n, !0)
	};
	return B.forEach(Object.keys({
		...e,
		...t
	}), function(r) {
		if (r === "__proto__" || r === "constructor" || r === "prototype") return;
		let a = B.hasOwnProp(l, r) ? l[r] : i, o = a(B.hasOwnProp(e, r) ? e[r] : void 0, B.hasOwnProp(t, r) ? t[r] : void 0, r);
		B.isUndefined(o) && a !== c || (n[r] = o);
	}), B.hasOwnProp(t, "validateStatus") && B.isUndefined(t.validateStatus) && s("validateStatusUndefinedResolves") === !1 && (B.hasOwnProp(e, "validateStatus") ? n.validateStatus = r(void 0, e.validateStatus) : delete n.validateStatus), n;
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/resolveConfig.js
var Pr = ["content-type", "content-length"];
function Fr(e, t, n) {
	if (n !== "content-only") {
		e.set(t);
		return;
	}
	Object.entries(t).forEach(([t, n]) => {
		Pr.includes(t.toLowerCase()) && e.set(t, n);
	});
}
var Ir = (e) => encodeURIComponent(e).replace(/%([0-9A-F]{2})/gi, (e, t) => String.fromCharCode(parseInt(t, 16)));
function Lr(e) {
	let t = Nr({}, e), n = (e) => B.hasOwnProp(t, e) ? t[e] : void 0, r = n("data"), i = n("withXSRFToken"), a = n("xsrfHeaderName"), o = n("xsrfCookieName"), s = n("headers"), c = n("auth"), l = n("baseURL"), u = n("allowAbsoluteUrls"), d = n("url");
	if (t.headers = s = Mn.from(s), t.url = Kn(jr(l, d, u, t), n("params"), n("paramsSerializer")), c) {
		let e = B.getSafeProp(c, "username") || "", t = B.getSafeProp(c, "password") || "";
		s.set("Authorization", "Basic " + btoa(e + ":" + (t ? Ir(t) : "")));
	}
	if (B.isFormData(r) && (nr.hasStandardBrowserEnv || nr.hasStandardBrowserWebWorkerEnv || B.isReactNative(r) ? s.setContentType(void 0) : B.isFunction(r.getHeaders) && Fr(s, r.getHeaders(), n("formDataHeaderPolicy"))), nr.hasStandardBrowserEnv && (B.isFunction(i) && (i = i(t)), i === !0 || i == null && Sr(t.url))) {
		let e = a && o && Cr.read(o);
		e && s.set(a, e);
	}
	return t;
}
var Rr = typeof XMLHttpRequest < "u" && function(e) {
	return new Promise(function(t, n) {
		let r = Lr(e), i = r.data, a = Mn.from(r.headers).normalize(), { responseType: o, onUploadProgress: s, onDownloadProgress: c } = r, l, u, d, f, p;
		function m() {
			f && f(), p && p(), r.cancelToken && r.cancelToken.unsubscribe(l), r.signal && r.signal.removeEventListener("abort", l);
		}
		let h = new XMLHttpRequest();
		h.open(r.method.toUpperCase(), r.url, !0), h.timeout = r.timeout;
		function g() {
			if (!h) return;
			let r = Mn.from("getAllResponseHeaders" in h && h.getAllResponseHeaders());
			hr(function(e) {
				t(e), m();
			}, function(e) {
				n(e), m();
			}, {
				data: !o || o === "text" || o === "json" ? h.responseText : h.response,
				status: h.status,
				statusText: h.statusText,
				headers: r,
				config: e,
				request: h
			}), h = null;
		}
		"onloadend" in h ? h.onloadend = g : h.onreadystatechange = function() {
			!h || h.readyState !== 4 || h.status === 0 && !(h.responseURL && h.responseURL.startsWith("file:")) || setTimeout(g);
		}, h.onabort = function() {
			h &&= (n(new V("Request aborted", V.ECONNABORTED, e, h)), m(), null);
		}, h.onerror = function(t) {
			let r = new V(t && t.message ? t.message : "Network Error", V.ERR_NETWORK, e, h);
			r.event = t || null, n(r), m(), h = null;
		}, h.ontimeout = function() {
			let t = r.timeout ? "timeout of " + r.timeout + "ms exceeded" : "timeout exceeded", i = r.transitional || Jn;
			r.timeoutErrorMessage && (t = r.timeoutErrorMessage), n(new V(t, i.clarifyTimeoutError ? V.ETIMEDOUT : V.ECONNABORTED, e, h)), m(), h = null;
		}, i === void 0 && a.setContentType(null), "setRequestHeader" in h && B.forEach(Cn(a), function(e, t) {
			h.setRequestHeader(t, e);
		}), B.isUndefined(r.withCredentials) || (h.withCredentials = !!r.withCredentials), o && o !== "json" && (h.responseType = r.responseType), c && ([d, p] = yr(c, !0), h.addEventListener("progress", d)), s && h.upload && ([u, f] = yr(s), h.upload.addEventListener("progress", u), h.upload.addEventListener("loadend", f)), (r.cancelToken || r.signal) && (l = (t) => {
			h &&= (n(!t || t.type ? new mr(null, e, h) : t), h.abort(), m(), null);
		}, r.cancelToken && r.cancelToken.subscribe(l), r.signal && (r.signal.aborted ? l() : r.signal.addEventListener("abort", l)));
		let _ = gr(r.url);
		if (_ && !nr.protocols.includes(_)) {
			n(new V("Unsupported protocol " + _ + ":", V.ERR_BAD_REQUEST, e));
			return;
		}
		h.send(i || null);
	});
}, zr = (e, t) => {
	if (e = e ? e.filter(Boolean) : [], !t && !e.length) return;
	let n = new AbortController(), r = !1, i = function(e) {
		if (!r) {
			r = !0, o();
			let t = e instanceof Error ? e : this.reason;
			n.abort(t instanceof V ? t : new mr(t instanceof Error ? t.message : t));
		}
	}, a = t && setTimeout(() => {
		a = null, i(new V(`timeout of ${t}ms exceeded`, V.ETIMEDOUT));
	}, t), o = () => {
		e &&= (a && clearTimeout(a), a = null, e.forEach((e) => {
			e.unsubscribe ? e.unsubscribe(i) : e.removeEventListener("abort", i);
		}), null);
	};
	e.forEach((e) => e.addEventListener("abort", i));
	let { signal: s } = n;
	return s.unsubscribe = () => B.asap(o), s;
}, Br = function* (e, t) {
	let n = e.byteLength;
	if (!t || n < t) {
		yield e;
		return;
	}
	let r = 0, i;
	for (; r < n;) i = r + t, yield e.slice(r, i), r = i;
}, Vr = async function* (e, t) {
	for await (let n of Hr(e)) yield* Br(n, t);
}, Hr = async function* (e) {
	if (e[Symbol.asyncIterator]) {
		yield* e;
		return;
	}
	let t = e.getReader();
	try {
		for (;;) {
			let { done: e, value: n } = await t.read();
			if (e) break;
			yield n;
		}
	} finally {
		await t.cancel();
	}
}, Ur = (e, t, n, r) => {
	let i = Vr(e, t), a = 0, o, s = (e) => {
		o || (o = !0, r && r(e));
	};
	return new ReadableStream({
		async pull(e) {
			try {
				let { done: t, value: r } = await i.next();
				if (t) {
					s(), e.close();
					return;
				}
				let o = r.byteLength;
				n && n(a += o), e.enqueue(new Uint8Array(r));
			} catch (e) {
				throw s(e), e;
			}
		},
		cancel(e) {
			return s(e), i.return();
		}
	}, { highWaterMark: 2 });
}, Wr = (e) => e >= 48 && e <= 57 || e >= 65 && e <= 70 || e >= 97 && e <= 102, Gr = (e, t, n) => t + 2 < n && Wr(e.charCodeAt(t + 1)) && Wr(e.charCodeAt(t + 2));
function Kr(e) {
	if (!e || typeof e != "string" || !e.startsWith("data:")) return 0;
	let t = e.indexOf(",");
	if (t < 0) return 0;
	let n = e.slice(5, t), r = e.slice(t + 1);
	if (/;base64/i.test(n)) {
		let e = r.length, t = r.length;
		for (let n = 0; n < t; n++) if (r.charCodeAt(n) === 37 && n + 2 < t) {
			let t = r.charCodeAt(n + 1), i = r.charCodeAt(n + 2);
			Wr(t) && Wr(i) && (e -= 2, n += 2);
		}
		let n = 0, i = t - 1, a = (e) => e >= 2 && r.charCodeAt(e - 2) === 37 && r.charCodeAt(e - 1) === 51 && (r.charCodeAt(e) === 68 || r.charCodeAt(e) === 100);
		i >= 0 && (r.charCodeAt(i) === 61 ? (n++, i--) : a(i) && (n++, i -= 3)), n === 1 && i >= 0 && (r.charCodeAt(i) === 61 || a(i)) && n++;
		let o = Math.floor(e / 4) * 3 - (n || 0);
		return o > 0 ? o : 0;
	}
	let i = 0;
	for (let e = 0, t = r.length; e < t; e++) {
		let n = r.charCodeAt(e);
		if (n === 37 && Gr(r, e, t)) i += 1, e += 2;
		else if (n < 128) i += 1;
		else if (n < 2048) i += 2;
		else if (n >= 55296 && n <= 56319 && e + 1 < t) {
			let t = r.charCodeAt(e + 1);
			t >= 56320 && t <= 57343 ? (i += 4, e++) : i += 3;
		} else i += 3;
	}
	return i;
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/env/data.js
var qr = "1.18.0", Jr = 65536, { isFunction: Yr } = B, Xr = (e) => encodeURIComponent(e).replace(/%([0-9A-F]{2})/gi, (e, t) => String.fromCharCode(parseInt(t, 16))), Zr = (e) => {
	if (!B.isString(e)) return e;
	try {
		return decodeURIComponent(e);
	} catch {
		return e;
	}
}, Qr = (e, ...t) => {
	try {
		return !!e(...t);
	} catch {
		return !1;
	}
}, $r = (e) => {
	let t = e.indexOf("://"), n = e;
	return t !== -1 && (n = n.slice(t + 3)), n.includes("@") || n.includes(":");
}, ei = (e) => {
	let t = B.global !== void 0 && B.global !== null ? B.global : globalThis, { ReadableStream: n, TextEncoder: r } = t;
	e = B.merge.call({ skipUndefined: !0 }, {
		Request: t.Request,
		Response: t.Response
	}, e);
	let { fetch: i, Request: a, Response: o } = e, s = i ? Yr(i) : typeof fetch == "function", c = Yr(a), l = Yr(o);
	if (!s) return !1;
	let u = s && Yr(n), d = s && (typeof r == "function" ? ((e) => (t) => e.encode(t))(new r()) : async (e) => new Uint8Array(await new a(e).arrayBuffer())), f = c && u && Qr(() => {
		let e = !1, t = new a(nr.origin, {
			body: new n(),
			method: "POST",
			get duplex() {
				return e = !0, "half";
			}
		}), r = t.headers.has("Content-Type");
		return t.body != null && t.body.cancel(), e && !r;
	}), p = l && u && Qr(() => B.isReadableStream(new o("").body)), m = { stream: p && ((e) => e.body) };
	s && [
		"text",
		"arrayBuffer",
		"blob",
		"formData",
		"stream"
	].forEach((e) => {
		!m[e] && (m[e] = (t, n) => {
			let r = t && t[e];
			if (r) return r.call(t);
			throw new V(`Response type '${e}' is not supported`, V.ERR_NOT_SUPPORT, n);
		});
	});
	let h = async (e) => {
		if (e == null) return 0;
		if (B.isBlob(e)) return e.size;
		if (B.isSpecCompliantForm(e)) return (await new a(nr.origin, {
			method: "POST",
			body: e
		}).arrayBuffer()).byteLength;
		if (B.isArrayBufferView(e) || B.isArrayBuffer(e)) return e.byteLength;
		if (B.isURLSearchParams(e) && (e += ""), B.isString(e)) return (await d(e)).byteLength;
	}, g = async (e, t) => B.toFiniteNumber(e.getContentLength()) ?? h(t);
	return async (e) => {
		let { url: t, method: n, data: s, signal: l, cancelToken: d, timeout: _, onDownloadProgress: v, onUploadProgress: y, responseType: b, headers: x, withCredentials: S = "same-origin", fetchOptions: C, maxContentLength: w, maxBodyLength: T } = Lr(e), E = B.isNumber(w) && w > -1, D = B.isNumber(T) && T > -1, O = (t) => B.hasOwnProp(e, t) ? e[t] : void 0, ee = i || fetch;
		b = b ? (b + "").toLowerCase() : "text";
		let k = zr([l, d && d.toAbortSignal()], _), A = null, te = k && k.unsubscribe && (() => {
			k.unsubscribe();
		}), j, ne = null, re = () => new V("Request body larger than maxBodyLength limit", V.ERR_BAD_REQUEST, e, A);
		try {
			let i, l = O("auth");
			if (l && (i = {
				username: B.getSafeProp(l, "username") || "",
				password: B.getSafeProp(l, "password") || ""
			}), $r(t)) {
				let e = new URL(t, nr.origin);
				!i && (e.username || e.password) && (i = {
					username: Zr(e.username),
					password: Zr(e.password)
				}), (e.username || e.password) && (e.username = "", e.password = "", t = e.href);
			}
			if (i && (x.delete("authorization"), x.set("Authorization", "Basic " + btoa(Xr((i.username || "") + ":" + (i.password || ""))))), E && typeof t == "string" && t.startsWith("data:") && Kr(t) > w) throw new V("maxContentLength size of " + w + " exceeded", V.ERR_BAD_RESPONSE, e, A);
			if (D && n !== "get" && n !== "head") {
				let e = await h(s);
				if (typeof e == "number" && isFinite(e) && (j = e, e > T)) throw re();
			}
			let d = D && (B.isReadableStream(s) || B.isStream(s)), _ = (e, t, n) => Ur(e, Jr, (e) => {
				if (D && e > T) throw ne = re();
				t && t(e);
			}, n);
			if (f && n !== "get" && n !== "head" && (y || d)) {
				if (j ??= await g(x, s), j !== 0 || d) {
					let e = new a(t, {
						method: "POST",
						body: s,
						duplex: "half"
					}), n;
					if (B.isFormData(s) && (n = e.headers.get("content-type")) && x.setContentType(n), e.body) {
						let [t, n] = y && br(j, yr(xr(y))) || [];
						s = _(e.body, t, n);
					}
				}
			} else if (d && !c && u && n !== "get" && n !== "head") s = _(s);
			else if (d && c && !f && n !== "get" && n !== "head") throw new V("Stream request bodies are not supported by the current fetch implementation", V.ERR_NOT_SUPPORT, e, A);
			B.isString(S) || (S = S ? "include" : "omit");
			let M = c && "credentials" in a.prototype;
			if (B.isFormData(s)) {
				let e = x.getContentType();
				e && /^multipart\/form-data/i.test(e) && !/boundary=/i.test(e) && x.delete("content-type");
			}
			x.set("User-Agent", "axios/" + qr, !1);
			let ie = {
				...C,
				signal: k,
				method: n.toUpperCase(),
				headers: Cn(x.normalize()),
				body: s,
				duplex: "half",
				credentials: M ? S : void 0
			};
			A = c && new a(t, ie);
			let N = await (c ? ee(A, C) : ee(t, ie)), ae = Mn.from(N.headers);
			if (E) {
				let t = B.toFiniteNumber(ae.getContentLength());
				if (t != null && t > w) throw new V("maxContentLength size of " + w + " exceeded", V.ERR_BAD_RESPONSE, e, A);
			}
			let oe = p && (b === "stream" || b === "response");
			if (p && N.body && (v || E || oe && te)) {
				let t = {};
				[
					"status",
					"statusText",
					"headers"
				].forEach((e) => {
					t[e] = N[e];
				});
				let n = B.toFiniteNumber(ae.getContentLength()), [r, i] = v && br(n, yr(xr(v), !0)) || [], a = 0;
				N = new o(Ur(N.body, Jr, (t) => {
					if (E && (a = t, a > w)) throw new V("maxContentLength size of " + w + " exceeded", V.ERR_BAD_RESPONSE, e, A);
					r && r(t);
				}, () => {
					i && i(), te && te();
				}), t);
			}
			b ||= "text";
			let se = await m[B.findKey(m, b) || "text"](N, e);
			if (E && !p && !oe) {
				let t;
				if (se != null && (typeof se.byteLength == "number" ? t = se.byteLength : typeof se.size == "number" ? t = se.size : typeof se == "string" && (t = typeof r == "function" ? new r().encode(se).byteLength : se.length)), typeof t == "number" && t > w) throw new V("maxContentLength size of " + w + " exceeded", V.ERR_BAD_RESPONSE, e, A);
			}
			return !oe && te && te(), await new Promise((t, n) => {
				hr(t, n, {
					data: se,
					headers: Mn.from(N.headers),
					status: N.status,
					statusText: N.statusText,
					config: e,
					request: A
				});
			});
		} catch (t) {
			if (te && te(), k && k.aborted && k.reason instanceof V) {
				let n = k.reason;
				throw n.config = e, A && (n.request = A), t !== n && (n.cause = t), n;
			}
			throw ne ? (A && !ne.request && (ne.request = A), ne) : t instanceof V ? (A && !t.request && (t.request = A), t) : t && t.name === "TypeError" && /Load failed|fetch/i.test(t.message) ? Object.assign(new V("Network Error", V.ERR_NETWORK, e, A, t && t.response), { cause: t.cause || t }) : V.from(t, t && t.code, e, A, t && t.response);
		}
	};
}, ti = /* @__PURE__ */ new Map(), ni = (e) => {
	let t = e && e.env || {}, { fetch: n, Request: r, Response: i } = t, a = [
		r,
		i,
		n
	], o = a.length, s, c, l = ti;
	for (; o--;) s = a[o], c = l.get(s), c === void 0 && l.set(s, c = o ? /* @__PURE__ */ new Map() : ei(t)), l = c;
	return c;
};
ni();
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/adapters/adapters.js
var ri = {
	http: null,
	xhr: Rr,
	fetch: { get: ni }
};
B.forEach(ri, (e, t) => {
	if (e) {
		try {
			Object.defineProperty(e, "name", {
				__proto__: null,
				value: t
			});
		} catch {}
		Object.defineProperty(e, "adapterName", {
			__proto__: null,
			value: t
		});
	}
});
var ii = (e) => `- ${e}`, ai = (e) => B.isFunction(e) || e === null || e === !1;
function oi(e, t) {
	e = B.isArray(e) ? e : [e];
	let { length: n } = e, r, i, a = {};
	for (let o = 0; o < n; o++) {
		r = e[o];
		let n;
		if (i = r, !ai(r) && (i = ri[(n = String(r)).toLowerCase()], i === void 0)) throw new V(`Unknown adapter '${n}'`);
		if (i && (B.isFunction(i) || (i = i.get(t)))) break;
		a[n || "#" + o] = i;
	}
	if (!i) {
		let e = Object.entries(a).map(([e, t]) => `adapter ${e} ` + (t === !1 ? "is not supported by the environment" : "is not available in the build"));
		throw new V("There is no suitable adapter to dispatch the request " + (n ? e.length > 1 ? "since :\n" + e.map(ii).join("\n") : " " + ii(e[0]) : "as no adapter specified"), "ERR_NOT_SUPPORT");
	}
	return i;
}
var si = {
	getAdapter: oi,
	adapters: ri
};
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/core/dispatchRequest.js
function ci(e) {
	if (e.cancelToken && e.cancelToken.throwIfRequested(), e.signal && e.signal.aborted) throw new mr(null, e);
}
function li(e) {
	return ci(e), e.headers = Mn.from(e.headers), e.data = fr.call(e, e.transformRequest), [
		"post",
		"put",
		"patch"
	].indexOf(e.method) !== -1 && e.headers.setContentType("application/x-www-form-urlencoded", !1), si.getAdapter(e.adapter || dr.adapter, e)(e).then(function(t) {
		ci(e), e.response = t;
		try {
			t.data = fr.call(e, e.transformResponse, t);
		} finally {
			delete e.response;
		}
		return t.headers = Mn.from(t.headers), t;
	}, function(t) {
		if (!pr(t) && (ci(e), t && t.response)) {
			e.response = t.response;
			try {
				t.response.data = fr.call(e, e.transformResponse, t.response);
			} finally {
				delete e.response;
			}
			t.response.headers = Mn.from(t.response.headers);
		}
		return Promise.reject(t);
	});
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/validator.js
var ui = {};
[
	"object",
	"boolean",
	"number",
	"function",
	"string",
	"symbol"
].forEach((e, t) => {
	ui[e] = function(n) {
		return typeof n === e || "a" + (t < 1 ? "n " : " ") + e;
	};
});
var di = {};
ui.transitional = function(e, t, n) {
	function r(e, t) {
		return "[Axios v" + qr + "] Transitional option '" + e + "'" + t + (n ? ". " + n : "");
	}
	return (n, i, a) => {
		if (e === !1) throw new V(r(i, " has been removed" + (t ? " in " + t : "")), V.ERR_DEPRECATED);
		return t && !di[i] && (di[i] = !0, console.warn(r(i, " has been deprecated since v" + t + " and will be removed in the near future"))), !e || e(n, i, a);
	};
}, ui.spelling = function(e) {
	return (t, n) => (console.warn(`${n} is likely a misspelling of ${e}`), !0);
};
function fi(e, t, n) {
	if (typeof e != "object") throw new V("options must be an object", V.ERR_BAD_OPTION_VALUE);
	let r = Object.keys(e), i = r.length;
	for (; i-- > 0;) {
		let a = r[i], o = Object.prototype.hasOwnProperty.call(t, a) ? t[a] : void 0;
		if (o) {
			let t = e[a], n = t === void 0 || o(t, a, e);
			if (n !== !0) throw new V("option " + a + " must be " + n, V.ERR_BAD_OPTION_VALUE);
			continue;
		}
		if (n !== !0) throw new V("Unknown option " + a, V.ERR_BAD_OPTION);
	}
}
var pi = {
	assertOptions: fi,
	validators: ui
}, mi = pi.validators, hi = class {
	constructor(e) {
		this.defaults = e || {}, this.interceptors = {
			request: new qn(),
			response: new qn()
		};
	}
	async request(e, t) {
		try {
			return await this._request(e, t);
		} catch (e) {
			if (e instanceof Error) {
				let t = {};
				Error.captureStackTrace ? Error.captureStackTrace(t) : t = /* @__PURE__ */ Error();
				let n = (() => {
					if (!t.stack) return "";
					let e = t.stack.indexOf("\n");
					return e === -1 ? "" : t.stack.slice(e + 1);
				})();
				try {
					if (!e.stack) e.stack = n;
					else if (n) {
						let t = n.indexOf("\n"), r = t === -1 ? -1 : n.indexOf("\n", t + 1), i = r === -1 ? "" : n.slice(r + 1);
						String(e.stack).endsWith(i) || (e.stack += "\n" + n);
					}
				} catch {}
			}
			throw e;
		}
	}
	_request(e, t) {
		typeof e == "string" ? (t ||= {}, t.url = e) : t = e || {}, t = Nr(this.defaults, t);
		let { transitional: n, paramsSerializer: r, headers: i } = t;
		n !== void 0 && pi.assertOptions(n, {
			silentJSONParsing: mi.transitional(mi.boolean),
			forcedJSONParsing: mi.transitional(mi.boolean),
			clarifyTimeoutError: mi.transitional(mi.boolean),
			legacyInterceptorReqResOrdering: mi.transitional(mi.boolean),
			advertiseZstdAcceptEncoding: mi.transitional(mi.boolean),
			validateStatusUndefinedResolves: mi.transitional(mi.boolean)
		}, !1), r != null && (B.isFunction(r) ? t.paramsSerializer = { serialize: r } : pi.assertOptions(r, {
			encode: mi.function,
			serialize: mi.function
		}, !0)), t.allowAbsoluteUrls !== void 0 || (this.defaults.allowAbsoluteUrls === void 0 ? t.allowAbsoluteUrls = !0 : t.allowAbsoluteUrls = this.defaults.allowAbsoluteUrls), pi.assertOptions(t, {
			baseUrl: mi.spelling("baseURL"),
			withXsrfToken: mi.spelling("withXSRFToken")
		}, !0), t.method = (t.method || this.defaults.method || "get").toLowerCase();
		let a = i && B.merge(i.common, i[t.method]);
		i && B.forEach([
			"delete",
			"get",
			"head",
			"post",
			"put",
			"patch",
			"query",
			"common"
		], (e) => {
			delete i[e];
		}), t.headers = Mn.concat(a, i);
		let o = [], s = !0;
		this.interceptors.request.forEach(function(e) {
			if (typeof e.runWhen == "function" && e.runWhen(t) === !1) return;
			s &&= e.synchronous;
			let n = t.transitional || Jn;
			n && n.legacyInterceptorReqResOrdering ? o.unshift(e.fulfilled, e.rejected) : o.push(e.fulfilled, e.rejected);
		});
		let c = [];
		this.interceptors.response.forEach(function(e) {
			c.push(e.fulfilled, e.rejected);
		});
		let l, u = 0, d;
		if (!s) {
			let e = [li.bind(this), void 0];
			for (e.unshift(...o), e.push(...c), d = e.length, l = Promise.resolve(t); u < d;) l = l.then(e[u++], e[u++]);
			return l;
		}
		d = o.length;
		let f = t;
		for (; u < d;) {
			let e = o[u++], t = o[u++];
			try {
				f = e(f);
			} catch (e) {
				t.call(this, e);
				break;
			}
		}
		try {
			l = li.call(this, f);
		} catch (e) {
			return Promise.reject(e);
		}
		for (u = 0, d = c.length; u < d;) l = l.then(c[u++], c[u++]);
		return l;
	}
	getUri(e) {
		return e = Nr(this.defaults, e), Kn(jr(e.baseURL, e.url, e.allowAbsoluteUrls, e), e.params, e.paramsSerializer);
	}
};
B.forEach([
	"delete",
	"get",
	"head",
	"options"
], function(e) {
	hi.prototype[e] = function(t, n) {
		return this.request(Nr(n || {}, {
			method: e,
			url: t,
			data: n && B.hasOwnProp(n, "data") ? n.data : void 0
		}));
	};
}), B.forEach([
	"post",
	"put",
	"patch",
	"query"
], function(e) {
	function t(t) {
		return function(n, r, i) {
			return this.request(Nr(i || {}, {
				method: e,
				headers: t ? { "Content-Type": "multipart/form-data" } : {},
				url: n,
				data: r
			}));
		};
	}
	hi.prototype[e] = t(), e !== "query" && (hi.prototype[e + "Form"] = t(!0));
});
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/cancel/CancelToken.js
var gi = class e {
	constructor(e) {
		if (typeof e != "function") throw TypeError("executor must be a function.");
		let t;
		this.promise = new Promise(function(e) {
			t = e;
		});
		let n = this;
		this.promise.then((e) => {
			if (!n._listeners) return;
			let t = n._listeners.length;
			for (; t-- > 0;) n._listeners[t](e);
			n._listeners = null;
		}), this.promise.then = (e) => {
			let t, r = new Promise((e) => {
				n.subscribe(e), t = e;
			}).then(e);
			return r.cancel = function() {
				n.unsubscribe(t);
			}, r;
		}, e(function(e, r, i) {
			n.reason || (n.reason = new mr(e, r, i), t(n.reason));
		});
	}
	throwIfRequested() {
		if (this.reason) throw this.reason;
	}
	subscribe(e) {
		if (this.reason) {
			e(this.reason);
			return;
		}
		this._listeners ? this._listeners.push(e) : this._listeners = [e];
	}
	unsubscribe(e) {
		if (!this._listeners) return;
		let t = this._listeners.indexOf(e);
		t !== -1 && this._listeners.splice(t, 1);
	}
	toAbortSignal() {
		let e = new AbortController(), t = (t) => {
			e.abort(t);
		};
		return this.subscribe(t), e.signal.unsubscribe = () => this.unsubscribe(t), e.signal;
	}
	static source() {
		let t;
		return {
			token: new e(function(e) {
				t = e;
			}),
			cancel: t
		};
	}
};
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/spread.js
function _i(e) {
	return function(t) {
		return e.apply(null, t);
	};
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/isAxiosError.js
function vi(e) {
	return B.isObject(e) && e.isAxiosError === !0;
}
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/helpers/HttpStatusCode.js
var yi = {
	Continue: 100,
	SwitchingProtocols: 101,
	Processing: 102,
	EarlyHints: 103,
	Ok: 200,
	Created: 201,
	Accepted: 202,
	NonAuthoritativeInformation: 203,
	NoContent: 204,
	ResetContent: 205,
	PartialContent: 206,
	MultiStatus: 207,
	AlreadyReported: 208,
	ImUsed: 226,
	MultipleChoices: 300,
	MovedPermanently: 301,
	Found: 302,
	SeeOther: 303,
	NotModified: 304,
	UseProxy: 305,
	Unused: 306,
	TemporaryRedirect: 307,
	PermanentRedirect: 308,
	BadRequest: 400,
	Unauthorized: 401,
	PaymentRequired: 402,
	Forbidden: 403,
	NotFound: 404,
	MethodNotAllowed: 405,
	NotAcceptable: 406,
	ProxyAuthenticationRequired: 407,
	RequestTimeout: 408,
	Conflict: 409,
	Gone: 410,
	LengthRequired: 411,
	PreconditionFailed: 412,
	PayloadTooLarge: 413,
	UriTooLong: 414,
	UnsupportedMediaType: 415,
	RangeNotSatisfiable: 416,
	ExpectationFailed: 417,
	ImATeapot: 418,
	MisdirectedRequest: 421,
	UnprocessableEntity: 422,
	Locked: 423,
	FailedDependency: 424,
	TooEarly: 425,
	UpgradeRequired: 426,
	PreconditionRequired: 428,
	TooManyRequests: 429,
	RequestHeaderFieldsTooLarge: 431,
	UnavailableForLegalReasons: 451,
	InternalServerError: 500,
	NotImplemented: 501,
	BadGateway: 502,
	ServiceUnavailable: 503,
	GatewayTimeout: 504,
	HttpVersionNotSupported: 505,
	VariantAlsoNegotiates: 506,
	InsufficientStorage: 507,
	LoopDetected: 508,
	NotExtended: 510,
	NetworkAuthenticationRequired: 511,
	WebServerIsDown: 521,
	ConnectionTimedOut: 522,
	OriginIsUnreachable: 523,
	TimeoutOccurred: 524,
	SslHandshakeFailed: 525,
	InvalidSslCertificate: 526
};
Object.entries(yi).forEach(([e, t]) => {
	yi[t] = e;
});
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/lib/axios.js
function bi(e) {
	let t = new hi(e), n = Qe(hi.prototype.request, t);
	return B.extend(n, hi.prototype, t, { allOwnKeys: !0 }), B.extend(n, t, null, { allOwnKeys: !0 }), n.create = function(t) {
		return bi(Nr(e, t));
	}, n;
}
var xi = bi(dr);
xi.Axios = hi, xi.CanceledError = mr, xi.CancelToken = gi, xi.isCancel = pr, xi.VERSION = qr, xi.toFormData = Vn, xi.AxiosError = V, xi.Cancel = xi.CanceledError, xi.all = function(e) {
	return Promise.all(e);
}, xi.spread = _i, xi.isAxiosError = vi, xi.mergeConfig = Nr, xi.AxiosHeaders = Mn, xi.formToJSON = (e) => cr(B.isHTMLForm(e) ? new FormData(e) : e), xi.getAdapter = si.getAdapter, xi.HttpStatusCode = yi, xi.default = xi;
//#endregion
//#region node_modules/.pnpm/axios@1.18.0_debug@4.4.3/node_modules/axios/index.js
var { Axios: Si, AxiosError: Ci, CanceledError: wi, isCancel: Ti, CancelToken: Ei, VERSION: Di, all: Oi, Cancel: ki, isAxiosError: Ai, spread: ji, toFormData: Mi, AxiosHeaders: Ni, HttpStatusCode: Pi, formToJSON: Fi, getAdapter: Ii, mergeConfig: Li, create: Ri } = xi, zi = null, Bi = (e) => {
	zi = e;
}, Vi = (e) => `${zi?.baseUrl ?? ""}${e}`, Hi = () => {
	let e = zi?.getAccessToken();
	return e ? { Authorization: `Bearer ${e}` } : {};
}, Ui = async (e) => (await xi({
	...e,
	url: Vi(e.url ?? ""),
	headers: {
		"X-Source": "NeMo Studio",
		...Hi(),
		...e.headers
	},
	paramsSerializer: { indexes: null }
})).data, Wi = n(null), Gi = ({ host: e, children: t }) => /* @__PURE__ */ p(Wi.Provider, {
	value: e,
	children: t
}), Ki = () => {
	let e = o(Wi);
	if (!e) throw Error("useHost must be used within the agent-hardener plugin Root");
	return e;
}, qi = () => Ki().workspaceId, Ji = () => Ki().notifications.notify, Yi = () => {
	let { notifications: e } = Ki();
	return l(() => ({
		success: (t) => e.notify(t, "success"),
		error: (t) => e.notify(t, "error"),
		info: (t) => e.notify(t, "info"),
		warning: (t) => e.notify(t, "warning")
	}), [e]);
}, Xi = ({ items: e } = {}) => {
	let { breadcrumbs: t } = Ki(), n = l(() => (e ?? []).filter((e) => typeof e.slotLabel == "string").map((e) => ({
		label: e.slotLabel,
		href: e.href
	})), [e]), r = JSON.stringify(n);
	s(() => {
		if (n.length !== 0) return t.set(n), () => t.set([]);
	}, [r, t]);
}, Zi = (e) => `/workspaces/${e}/plugin/agent-hardener`, Qi = {
	runList: "",
	runDetails: ":agentHardenerRunName",
	manifestList: "manifests",
	manifestNew: "manifests/new",
	manifestDetail: "manifests/:agentHardenerManifestName"
}, $i = (e) => Zi(e), ea = (e, t) => `${Zi(e)}/${encodeURIComponent(t)}`, ta = (e) => `${Zi(e)}/manifests`, na = (e) => `${Zi(e)}/manifests/new`, ra = (e, t) => `${Zi(e)}/manifests/${encodeURIComponent(t)}`, ia = (e, t) => `/workspaces/${e}/jobs/${encodeURIComponent(t)}`, aa = (e, t) => `${e}: ${ne(N(t), "no reason given")}`, oa = (e, t) => (n) => e.error(aa(t, n)), sa = ({ interview: e }) => e.length === 0 ? null : /* @__PURE__ */ p(I, {
	gap: "density-md",
	children: e.map((e, t) => /* @__PURE__ */ m(I, {
		gap: "density-xs",
		className: "rounded-md border border-base p-3",
		children: [/* @__PURE__ */ p(L, {
			kind: "body/semibold/sm",
			children: e.question || e.gap || `Question ${t + 1}`
		}), /* @__PURE__ */ p(w, {
			message: e.answer || "(no answer)",
			characterLimit: 220
		})]
	}, t))
}), ca = (...e) => e.filter((e, t, n) => !!e && e.trim() !== "" && n.indexOf(e) === t).join(" ").trim(), la = (e) => e.replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase(), ua = (e) => e.replace(/^([A-Z])|[\s-_]+(\w)/g, (e, t, n) => n ? n.toUpperCase() : t.toLowerCase()), da = (e) => {
	let t = ua(e);
	return t.charAt(0).toUpperCase() + t.slice(1);
}, fa = {
	xmlns: "http://www.w3.org/2000/svg",
	width: 24,
	height: 24,
	viewBox: "0 0 24 24",
	fill: "none",
	stroke: "currentColor",
	strokeWidth: 2,
	strokeLinecap: "round",
	strokeLinejoin: "round"
}, pa = (e) => {
	for (let t in e) if (t.startsWith("aria-") || t === "role" || t === "title") return !0;
	return !1;
}, ma = n({}), ha = () => o(ma), ga = i(({ color: e, size: t, strokeWidth: n, absoluteStrokeWidth: i, className: a = "", children: o, iconNode: s, ...c }, l) => {
	let { size: u = 24, strokeWidth: d = 2, absoluteStrokeWidth: f = !1, color: p = "currentColor", className: m = "" } = ha() ?? {}, h = i ?? f ? Number(n ?? d) * 24 / Number(t ?? u) : n ?? d;
	return r("svg", {
		ref: l,
		...fa,
		width: t ?? u ?? fa.width,
		height: t ?? u ?? fa.height,
		stroke: e ?? p,
		strokeWidth: h,
		className: ca("lucide", m, a),
		...!o && !pa(c) && { "aria-hidden": "true" },
		...c
	}, [...s.map(([e, t]) => r(e, t)), ...Array.isArray(o) ? o : [o]]);
}), _a = (e, t) => {
	let n = i(({ className: n, ...i }, a) => r(ga, {
		ref: a,
		iconNode: t,
		className: ca(`lucide-${la(da(e))}`, `lucide-${e}`, n),
		...i
	}));
	return n.displayName = da(e), n;
}, va = _a("check", [["path", {
	d: "M20 6 9 17l-5-5",
	key: "1gmf2c"
}]]), ya = _a("loader-circle", [["path", {
	d: "M21 12a9 9 0 1 1-6.219-8.56",
	key: "13zald"
}]]), ba = _a("maximize-2", [
	["path", {
		d: "M15 3h6v6",
		key: "1q9fwt"
	}],
	["path", {
		d: "m21 3-7 7",
		key: "1l2asr"
	}],
	["path", {
		d: "m3 21 7-7",
		key: "tjx5ai"
	}],
	["path", {
		d: "M9 21H3v-6",
		key: "wtvkvv"
	}]
]), xa = _a("minus", [["path", {
	d: "M5 12h14",
	key: "1ays0h"
}]]), Sa = _a("pencil", [["path", {
	d: "M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z",
	key: "1a8usu"
}], ["path", {
	d: "m15 5 4 4",
	key: "1mk7zo"
}]]), Ca = _a("plus", [["path", {
	d: "M5 12h14",
	key: "1ays0h"
}], ["path", {
	d: "M12 5v14",
	key: "s699le"
}]]), wa = _a("trash", [
	["path", {
		d: "M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6",
		key: "miytrc"
	}],
	["path", {
		d: "M3 6h18",
		key: "d0wm0j"
	}],
	["path", {
		d: "M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2",
		key: "e791ji"
	}]
]), Ta = _a("x", [["path", {
	d: "M18 6 6 18",
	key: "1bl5f8"
}], ["path", {
	d: "m6 6 12 12",
	key: "d8bk6v"
}]]), Ea = {
	tool: "",
	payload: "",
	label: "benign",
	persona: "",
	rationale: ""
}, Da = [
	{
		key: "tool",
		label: "Tool",
		width: "w-[14%]"
	},
	{
		key: "payload",
		label: "Payload",
		width: "w-[26%]"
	},
	{
		key: "label",
		label: "Label",
		width: "w-[12%]"
	},
	{
		key: "rationale",
		label: "Rationale",
		width: "w-[26%]"
	},
	{
		key: "persona",
		label: "Persona",
		width: "w-[14%]"
	}
], Oa = (e, t) => String(e[t] ?? ""), ka = ({ value: e, onChange: t, disabled: n }) => {
	let [r, i] = d(null), a = n || r !== null, o = r !== null && r.index === e.length, s = (e, t) => i((n) => n && {
		...n,
		draft: {
			...n.draft,
			[e]: t
		}
	}), c = () => i(null), l = () => {
		r && (t(r.index < e.length ? e.map((e, t) => t === r.index ? r.draft : e) : [...e, r.draft]), i(null));
	}, u = (n) => {
		t(e.filter((e, t) => t !== n)), i(null);
	}, h = (e) => Da.map((t) => /* @__PURE__ */ p("td", {
		className: "px-3 py-2 align-top",
		children: /* @__PURE__ */ p(je, {
			value: Oa(e, t.key),
			disabled: n,
			onChange: (e) => s(t.key, e.target.value)
		})
	}, t.key)), g = (t, r) => /* @__PURE__ */ p("td", {
		className: "px-3 py-2 align-top",
		children: /* @__PURE__ */ p(F, {
			gap: "density-xs",
			children: t ? /* @__PURE__ */ m(f, { children: [/* @__PURE__ */ p(P, {
				kind: "tertiary",
				size: "small",
				"aria-label": "Save row",
				disabled: n,
				onClick: l,
				children: /* @__PURE__ */ p(va, { className: "h-4 w-4" })
			}), /* @__PURE__ */ p(P, {
				kind: "tertiary",
				size: "small",
				"aria-label": "Cancel edit",
				onClick: c,
				children: /* @__PURE__ */ p(Ta, { className: "h-4 w-4" })
			})] }) : /* @__PURE__ */ m(f, { children: [/* @__PURE__ */ p(P, {
				kind: "tertiary",
				size: "small",
				"aria-label": "Edit row",
				disabled: a,
				onClick: () => i({
					index: r,
					draft: { ...e[r] }
				}),
				children: /* @__PURE__ */ p(Sa, { className: "h-4 w-4" })
			}), /* @__PURE__ */ p(P, {
				kind: "tertiary",
				color: "danger",
				size: "small",
				"aria-label": "Delete row",
				disabled: a,
				onClick: () => u(r),
				children: /* @__PURE__ */ p(wa, { className: "h-4 w-4" })
			})] })
		})
	});
	return /* @__PURE__ */ m(I, {
		gap: "density-md",
		children: [/* @__PURE__ */ p("div", {
			className: "max-h-[360px] overflow-auto rounded-md border border-base",
			children: /* @__PURE__ */ m("table", {
				className: "w-full table-fixed border-collapse text-sm",
				children: [/* @__PURE__ */ p("thead", { children: /* @__PURE__ */ m("tr", {
					className: "border-b border-base",
					children: [Da.map((e) => /* @__PURE__ */ p("th", {
						className: `${e.width} px-3 py-2 text-left`,
						children: /* @__PURE__ */ p(L, {
							kind: "body/semibold/sm",
							className: "text-subtle",
							children: e.label
						})
					}, e.key)), /* @__PURE__ */ p("th", {
						className: "px-3 py-2",
						style: { width: 88 }
					})]
				}) }), /* @__PURE__ */ m("tbody", { children: [
					e.length === 0 && !o ? /* @__PURE__ */ p("tr", { children: /* @__PURE__ */ p("td", {
						colSpan: Da.length + 1,
						className: "px-3 py-4",
						children: /* @__PURE__ */ p(L, {
							kind: "body/regular/sm",
							className: "text-subtle",
							children: "No benign requests yet. Add one, or generate the suite."
						})
					}) }) : null,
					e.map((e, t) => {
						let n = r?.index === t;
						return /* @__PURE__ */ m("tr", {
							className: "border-b border-base",
							children: [n ? h(r.draft) : Da.map((t) => /* @__PURE__ */ p("td", {
								className: "px-3 py-2 align-top",
								children: /* @__PURE__ */ p("span", {
									className: "whitespace-pre-wrap break-words",
									children: Oa(e, t.key) || "—"
								})
							}, t.key)), g(!!n, t)]
						}, t);
					}),
					o ? /* @__PURE__ */ m("tr", {
						className: "border-b border-base",
						children: [h(r.draft), g(!0, r.index)]
					}) : null
				] })]
			})
		}), /* @__PURE__ */ p(F, { children: /* @__PURE__ */ m(P, {
			kind: "secondary",
			disabled: a,
			onClick: () => i({
				index: e.length,
				draft: { ...Ea }
			}),
			children: [/* @__PURE__ */ p(Ca, { className: "h-4 w-4" }), " Add request"]
		}) })]
	});
}, Aa = (e) => e?.find((e) => e.recommended)?.description ?? e?.[0]?.description ?? "", ja = "__other__", Ma = ({ prompt: e, loading: t, onSubmit: n }) => {
	let [r, i] = d(() => Object.fromEntries(e.questions.map((e) => [e.gap, Aa(e.options)]))), [a, o] = d({}), s = (e, t) => i((n) => ({
		...n,
		[e]: t
	})), c = (e, t) => o((n) => ({
		...n,
		[e]: t
	})), l = (e) => r[e] === ja ? a[e] ?? "" : r[e] ?? "";
	return /* @__PURE__ */ m("form", {
		onSubmit: (t) => {
			t.preventDefault(), n(e.questions.map((e) => ({
				gap: e.gap,
				question: e.question,
				answer: l(e.gap)
			})));
		},
		className: "flex h-full flex-col",
		children: [
			/* @__PURE__ */ m(I, {
				gap: "density-xs",
				className: "mb-4 shrink-0",
				children: [/* @__PURE__ */ p(L, {
					kind: "body/semibold/lg",
					children: "Answer the synth interview"
				}), /* @__PURE__ */ m(L, {
					kind: "body/regular/md",
					className: "text-subtle",
					children: [
						"Your answers shape the benign test suite the war-game replays against the agent.",
						" ",
						e.questions.length,
						" question",
						e.questions.length === 1 ? "" : "s",
						"."
					]
				})]
			}),
			/* @__PURE__ */ p(I, {
				gap: "density-lg",
				className: "min-h-0 flex-1 overflow-auto pr-density-xs",
				children: e.questions.map((e, t) => /* @__PURE__ */ p(de, {
					className: "p-4",
					children: /* @__PURE__ */ m(I, {
						gap: "density-md",
						children: [/* @__PURE__ */ m(L, {
							kind: "body/semibold/md",
							children: [/* @__PURE__ */ m("span", {
								className: "text-subtle",
								children: [t + 1, ". "]
							}), e.question]
						}), e.options && e.options.length > 0 ? /* @__PURE__ */ m(I, {
							gap: "density-md",
							children: [/* @__PURE__ */ p(_e, {
								name: e.gap,
								value: r[e.gap],
								onValueChange: (t) => s(e.gap, t),
								className: "w-full",
								children: /* @__PURE__ */ m(I, {
									gap: "3",
									children: [e.options.map((e) => /* @__PURE__ */ p(ee, {
										value: e.description,
										label: e.label || e.description,
										description: e.recommended ? `${e.description} (recommended)` : e.description
									}, e.description)), /* @__PURE__ */ p(ee, {
										value: ja,
										label: "Other",
										description: "Write your own answer"
									})]
								})
							}), r[e.gap] === ja ? /* @__PURE__ */ p(pe, {
								name: `${e.gap}_other`,
								slotLabel: "Your Answer",
								children: /* @__PURE__ */ p(je, {
									value: a[e.gap] ?? "",
									onChange: (t) => c(e.gap, t.target.value)
								})
							}) : null]
						}) : /* @__PURE__ */ p(pe, {
							name: e.gap,
							children: /* @__PURE__ */ p(je, {
								value: r[e.gap] ?? "",
								onChange: (t) => s(e.gap, t.target.value)
							})
						})]
					})
				}, e.gap))
			}),
			/* @__PURE__ */ p(F, {
				className: "mt-4 shrink-0 justify-end",
				children: /* @__PURE__ */ p(P, {
					color: "brand",
					type: "submit",
					disabled: t,
					children: t ? "Submitting…" : "Submit answers"
				})
			})
		]
	});
}, Na = ({ title: e, failure: t, workspace: n, jobName: r }) => /* @__PURE__ */ p(ue, {
	kind: "inline",
	status: "error",
	children: /* @__PURE__ */ m(I, {
		gap: "density-xs",
		children: [
			/* @__PURE__ */ m(L, {
				kind: "body/semibold/md",
				children: [
					e,
					": ",
					t.message
				]
			}),
			t.remediation ? /* @__PURE__ */ p(L, {
				kind: "body/regular/sm",
				children: t.remediation
			}) : null,
			/* @__PURE__ */ p(L, {
				kind: "body/regular/sm",
				children: /* @__PURE__ */ m(Me, {
					to: ia(n, r),
					children: ["View job ", r]
				})
			})
		]
	})
}), Pa = {
	blue: "var(--text-color-accent-blue)",
	gray: "var(--text-color-accent-gray)",
	green: "var(--text-color-accent-green)",
	purple: "var(--text-color-accent-purple)",
	red: "var(--text-color-accent-red)",
	teal: "var(--text-color-accent-teal)",
	yellow: "var(--text-color-accent-yellow)"
}, Fa = {
	danger: "var(--text-color-feedback-danger)",
	success: "var(--text-color-feedback-success)",
	warning: "var(--text-color-feedback-warning)"
}, Ia = (e, t = 12) => `color-mix(in srgb, ${e} ${t}%, transparent)`, La = ({ steps: e, busy: t, activity: n }) => e.length === 0 && !t ? null : /* @__PURE__ */ m(I, {
	gap: "2",
	children: [
		/* @__PURE__ */ p(L, {
			kind: "body/semibold/sm",
			className: "uppercase tracking-wide text-subtle",
			children: "Recon"
		}),
		e.map((e) => /* @__PURE__ */ m(F, {
			gap: "density-sm",
			align: "center",
			children: [/* @__PURE__ */ p(va, {
				size: 16,
				style: { color: Fa.success }
			}), /* @__PURE__ */ p(L, {
				kind: "body/regular/sm",
				children: e.label
			})]
		}, e.phase)),
		t ? /* @__PURE__ */ m(F, {
			gap: "density-sm",
			align: "center",
			children: [/* @__PURE__ */ p(ya, {
				size: 16,
				className: "animate-spin text-subtle"
			}), /* @__PURE__ */ p(L, {
				kind: "body/regular/sm",
				className: "text-subtle",
				children: n ?? "Working…"
			})]
		}) : null
	]
}), Ra = {
	tool: "",
	payload: "",
	label: "benign",
	persona: "",
	rationale: ""
}, za = ({ value: e, onChange: t, disabled: n }) => {
	let r = (n, r) => t(e.map((e, t) => t === n ? {
		...e,
		...r
	} : e)), i = (n) => t(e.filter((e, t) => t !== n));
	return /* @__PURE__ */ m(I, {
		gap: "density-lg",
		children: [e.length === 0 ? /* @__PURE__ */ p(L, {
			kind: "body/regular/md",
			className: "text-subtle",
			children: "No benign requests yet. Add rows manually, or generate the suite to populate it."
		}) : e.map((e, t) => /* @__PURE__ */ m(I, {
			gap: "density-sm",
			className: "rounded-md border border-base p-3",
			children: [
				/* @__PURE__ */ m(F, {
					gap: "density-md",
					children: [
						/* @__PURE__ */ p(pe, {
							name: `tool-${t}`,
							slotLabel: "Tool",
							className: "flex-1",
							children: /* @__PURE__ */ p(je, {
								value: e.tool,
								disabled: n,
								onChange: (e) => r(t, { tool: e.target.value })
							})
						}),
						/* @__PURE__ */ p(pe, {
							name: `persona-${t}`,
							slotLabel: "Persona",
							className: "flex-1",
							children: /* @__PURE__ */ p(je, {
								value: e.persona ?? "",
								disabled: n,
								onChange: (e) => r(t, { persona: e.target.value })
							})
						}),
						/* @__PURE__ */ p(pe, {
							name: `label-${t}`,
							slotLabel: "Label",
							className: "flex-1",
							children: /* @__PURE__ */ p(je, {
								value: e.label ?? "",
								disabled: n,
								onChange: (e) => r(t, { label: e.target.value })
							})
						})
					]
				}),
				/* @__PURE__ */ p(pe, {
					name: `payload-${t}`,
					slotLabel: "Payload",
					children: /* @__PURE__ */ p(Ae, {
						value: e.payload,
						rows: 2,
						disabled: n,
						onChange: (e) => r(t, { payload: e.target.value })
					})
				}),
				/* @__PURE__ */ p(pe, {
					name: `rationale-${t}`,
					slotLabel: "Rationale",
					children: /* @__PURE__ */ p(Ae, {
						value: e.rationale ?? "",
						rows: 2,
						disabled: n,
						onChange: (e) => r(t, { rationale: e.target.value })
					})
				}),
				/* @__PURE__ */ p(F, { children: /* @__PURE__ */ m(P, {
					kind: "tertiary",
					color: "danger",
					disabled: n,
					onClick: () => i(t),
					children: [/* @__PURE__ */ p(wa, {}), " Remove"]
				}) })
			]
		}, t)), /* @__PURE__ */ p(F, { children: /* @__PURE__ */ m(P, {
			kind: "secondary",
			disabled: n,
			onClick: () => t([...e, { ...Ra }]),
			children: [/* @__PURE__ */ p(Ca, {}), " Add request"]
		}) })]
	});
}, Ba = ({ suite: e, loading: t, onSubmit: n }) => {
	let [r, i] = d(e);
	return /* @__PURE__ */ m("div", {
		className: "flex h-full flex-col",
		children: [
			/* @__PURE__ */ m(I, {
				gap: "density-xs",
				className: "mb-4 shrink-0",
				children: [/* @__PURE__ */ p(L, {
					kind: "body/semibold/lg",
					children: "Review the benign suite"
				}), /* @__PURE__ */ p(L, {
					kind: "body/regular/md",
					className: "text-subtle",
					children: "Edit or drop the generated requests. The approved suite is replayed against the agent to confirm it still works after hardening."
				})]
			}),
			/* @__PURE__ */ p("div", {
				className: "min-h-0 flex-1 overflow-auto pr-density-xs",
				children: /* @__PURE__ */ p(za, {
					value: r,
					onChange: i,
					disabled: t
				})
			}),
			/* @__PURE__ */ p(F, {
				className: "mt-4 shrink-0 justify-end",
				children: /* @__PURE__ */ p(P, {
					color: "brand",
					disabled: t,
					onClick: () => n(r),
					children: t ? "Approving…" : `Approve ${r.length} request${r.length === 1 ? "" : "s"}`
				})
			})
		]
	});
}, Va = () => Ki().sdk.platform, Ha = (e, t) => Va().useJobsGetJob(e, t, { query: {
	enabled: !!t,
	refetchInterval: (e) => re(e.state.data?.status)
} });
//#endregion
//#region src/api/filesets.ts
async function Ua(e, { workspace: t, manifestName: n, file: r }, i, a) {
	let o = `${n}-${i}-${Date.now().toString(36)}`, s = await e.filesCreateFileset(t, {
		name: o,
		purpose: "generic"
	}), c = new Blob([await r.arrayBuffer()], { type: r.type || a });
	return await e.filesUploadFile(s.workspace, s.name, r.name, c), `${s.workspace}/${s.name}`;
}
var Wa = () => {
	let e = Va();
	return ze({ mutationFn: (t) => Ua(e, t, "hitlog", "application/jsonl") });
}, Ga = () => {
	let e = Va();
	return ze({ mutationFn: (t) => Ua(e, t, "benign-suite", "text/csv") });
}, Ka = () => {
	let e = Va();
	return ze({ mutationFn: (t) => Ua(e, t, "project", "application/zip") });
}, qa = () => ze({ mutationFn: ({ workspace: e, projectFileset: t, dockerfile: n }) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(e)}/manifests/inspect-project`,
	method: "POST",
	headers: { "Content-Type": "application/json" },
	data: {
		project_fileset: t,
		...n ? { dockerfile: n } : {}
	}
}) }), Ja = () => ze({ mutationFn: ({ workspace: e, agent: t }) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(e)}/manifests/inspect-agent`,
	method: "POST",
	headers: { "Content-Type": "application/json" },
	data: { agent: t }
}) }), Ya = (e, t) => {
	let n = { queryKey: t };
	for (let t of Object.keys(e)) t !== "queryKey" && Object.defineProperty(n, t, {
		enumerable: !0,
		configurable: !0,
		get: () => e[t]
	});
	return n;
}, Xa = (e, t) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/execution`,
	method: "GET",
	signal: t
}), Za = (e) => [`/apis/agent-hardener/v2/workspaces/${e}/execution`], Qa = (e, t) => {
	let { query: n } = t ?? {};
	return {
		queryKey: n?.queryKey ?? Za(e),
		queryFn: ({ signal: t }) => Xa(e, t),
		enabled: e != null,
		...n
	};
};
function $a(e, t, n) {
	let r = Qa(e, t);
	return Ya(Be(r, n), r.queryKey);
}
var eo = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/jobs`,
	method: "POST",
	headers: { "Content-Type": "application/json" },
	data: t,
	signal: n
}), to = (e) => {
	let t = ["agentHardenerCreateJob"], { mutation: n } = e ? e.mutation && "mutationKey" in e.mutation && e.mutation.mutationKey ? e : {
		...e,
		mutation: {
			...e.mutation,
			mutationKey: t
		}
	} : { mutation: { mutationKey: t } };
	return {
		mutationFn: (e) => {
			let { workspace: t, data: n } = e ?? {};
			return eo(t, n);
		},
		...n
	};
}, no = (e, t) => ze(to(e), t), ro = (e, t, n, r) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/jobs/${encodeURIComponent(String(t))}/results/${encodeURIComponent(String(n))}/download`,
	method: "GET",
	responseType: "blob",
	signal: r
}), io = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/jobs/${encodeURIComponent(String(t))}`,
	method: "DELETE",
	signal: n
}), ao = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/jobs/${encodeURIComponent(String(t))}/cancel`,
	method: "POST",
	signal: n
}), oo = (e, t, n, r) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/jobs/${encodeURIComponent(String(t))}/results`,
	method: "GET",
	params: n,
	signal: r
}), so = (e, t, n) => [`/apis/agent-hardener/v2/workspaces/${e}/jobs/${t}/results`, ...n ? [n] : []], co = (e, t, n, r) => {
	let { query: i } = r ?? {};
	return {
		queryKey: i?.queryKey ?? so(e, t, n),
		queryFn: ({ signal: r }) => oo(e, t, n, r),
		enabled: e != null && t != null,
		...i
	};
};
function lo(e, t, n, r, i) {
	let a = co(e, t, n, r);
	return Ya(Be(a, i), a.queryKey);
}
var uo = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/manifests`,
	method: "GET",
	params: t,
	signal: n
}), fo = (e, t) => [`/apis/agent-hardener/v2/workspaces/${e}/manifests`, ...t ? [t] : []], po = (e, t, n) => {
	let { query: r } = n ?? {};
	return {
		queryKey: r?.queryKey ?? fo(e, t),
		queryFn: ({ signal: n }) => uo(e, t, n),
		enabled: e != null,
		...r
	};
};
function mo(e, t, n, r) {
	let i = po(e, t, n);
	return Ya(Be(i, r), i.queryKey);
}
var ho = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/manifests`,
	method: "POST",
	headers: { "Content-Type": "application/json" },
	data: t,
	signal: n
}), go = (e) => {
	let t = ["agentHardenerCreateManifest"], { mutation: n } = e ? e.mutation && "mutationKey" in e.mutation && e.mutation.mutationKey ? e : {
		...e,
		mutation: {
			...e.mutation,
			mutationKey: t
		}
	} : { mutation: { mutationKey: t } };
	return {
		mutationFn: (e) => {
			let { workspace: t, data: n } = e ?? {};
			return ho(t, n);
		},
		...n
	};
}, _o = (e, t) => ze(go(e), t), vo = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/manifests/${encodeURIComponent(String(t))}`,
	method: "GET",
	signal: n
}), yo = (e, t) => [`/apis/agent-hardener/v2/workspaces/${e}/manifests/${t}`], bo = (e, t, n) => {
	let { query: r } = n ?? {};
	return {
		queryKey: r?.queryKey ?? yo(e, t),
		queryFn: ({ signal: n }) => vo(e, t, n),
		enabled: e != null && t != null,
		...r
	};
};
function xo(e, t, n, r) {
	let i = bo(e, t, n);
	return Ya(Be(i, r), i.queryKey);
}
var So = (e, t, n, r) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/manifests/${encodeURIComponent(String(t))}`,
	method: "PATCH",
	headers: { "Content-Type": "application/json" },
	data: n,
	signal: r
}), Co = (e) => {
	let t = ["agentHardenerUpdateManifest"], { mutation: n } = e ? e.mutation && "mutationKey" in e.mutation && e.mutation.mutationKey ? e : {
		...e,
		mutation: {
			...e.mutation,
			mutationKey: t
		}
	} : { mutation: { mutationKey: t } };
	return {
		mutationFn: (e) => {
			let { workspace: t, name: n, data: r } = e ?? {};
			return So(t, n, r);
		},
		...n
	};
}, wo = (e, t) => ze(Co(e), t), To = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/manifests/${encodeURIComponent(String(t))}`,
	method: "DELETE",
	signal: n
}), Eo = (e) => {
	let t = ["agentHardenerDeleteManifest"], { mutation: n } = e ? e.mutation && "mutationKey" in e.mutation && e.mutation.mutationKey ? e : {
		...e,
		mutation: {
			...e.mutation,
			mutationKey: t
		}
	} : { mutation: { mutationKey: t } };
	return {
		mutationFn: (e) => {
			let { workspace: t, name: n } = e ?? {};
			return To(t, n);
		},
		...n
	};
}, Do = (e, t) => ze(Eo(e), t), Oo = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/manifests/${encodeURIComponent(String(t))}/refresh`,
	method: "POST",
	signal: n
}), ko = (e) => {
	let t = ["agentHardenerRefreshManifest"], { mutation: n } = e ? e.mutation && "mutationKey" in e.mutation && e.mutation.mutationKey ? e : {
		...e,
		mutation: {
			...e.mutation,
			mutationKey: t
		}
	} : { mutation: { mutationKey: t } };
	return {
		mutationFn: (e) => {
			let { workspace: t, name: n } = e ?? {};
			return Oo(t, n);
		},
		...n
	};
}, Ao = (e, t) => ze(ko(e), t), jo = (e, t) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/model-config-defaults`,
	method: "GET",
	signal: t
}), Mo = (e) => [`/apis/agent-hardener/v2/workspaces/${e}/model-config-defaults`], No = (e, t) => {
	let { query: n } = t ?? {};
	return {
		queryKey: n?.queryKey ?? Mo(e),
		queryFn: ({ signal: t }) => jo(e, t),
		enabled: e != null,
		...n
	};
};
function Po(e, t, n) {
	let r = No(e, t);
	return Ya(Be(r, n), r.queryKey);
}
var Fo = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/model-config/validate`,
	method: "POST",
	headers: { "Content-Type": "application/json" },
	data: t,
	signal: n
}), Io = (e) => {
	let t = ["agentHardenerValidateModelConfig"], { mutation: n } = e ? e.mutation && "mutationKey" in e.mutation && e.mutation.mutationKey ? e : {
		...e,
		mutation: {
			...e.mutation,
			mutationKey: t
		}
	} : { mutation: { mutationKey: t } };
	return {
		mutationFn: (e) => {
			let { workspace: t, data: n } = e ?? {};
			return Fo(t, n);
		},
		...n
	};
}, Lo = (e, t) => ze(Io(e), t), Ro = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/runs`,
	method: "GET",
	params: t,
	signal: n
}), zo = (e, t) => [`/apis/agent-hardener/v2/workspaces/${e}/runs`, ...t ? [t] : []], Bo = (e, t, n) => {
	let { query: r } = n ?? {};
	return {
		queryKey: r?.queryKey ?? zo(e, t),
		queryFn: ({ signal: n }) => Ro(e, t, n),
		enabled: e != null,
		...r
	};
};
function Vo(e, t, n, r) {
	let i = Bo(e, t, n);
	return Ya(Be(i, r), i.queryKey);
}
var Ho = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/runs/${encodeURIComponent(String(t))}`,
	method: "GET",
	signal: n
}), Uo = (e, t) => [`/apis/agent-hardener/v2/workspaces/${e}/runs/${t}`], Wo = (e, t, n) => {
	let { query: r } = n ?? {};
	return {
		queryKey: r?.queryKey ?? Uo(e, t),
		queryFn: ({ signal: n }) => Ho(e, t, n),
		enabled: e != null && t != null,
		...r
	};
};
function Go(e, t, n, r) {
	let i = Wo(e, t, n);
	return Ya(Be(i, r), i.queryKey);
}
var Ko = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/runs/${encodeURIComponent(String(t))}`,
	method: "DELETE",
	signal: n
}), qo = (e, t, n, r) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/runs/${encodeURIComponent(String(t))}/apply-mitigation`,
	method: "POST",
	headers: { "Content-Type": "application/json" },
	data: n,
	signal: r
}), Jo = (e) => {
	let t = ["agentHardenerApplyMitigation"], { mutation: n } = e ? e.mutation && "mutationKey" in e.mutation && e.mutation.mutationKey ? e : {
		...e,
		mutation: {
			...e.mutation,
			mutationKey: t
		}
	} : { mutation: { mutationKey: t } };
	return {
		mutationFn: (e) => {
			let { workspace: t, name: n, data: r } = e ?? {};
			return qo(t, n, r);
		},
		...n
	};
}, Yo = (e, t) => ze(Jo(e), t), Xo = (e, t, n, r) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/runs/${encodeURIComponent(String(t))}/events`,
	method: "GET",
	params: n,
	signal: r
}), Zo = (e, t, n) => [`/apis/agent-hardener/v2/workspaces/${e}/runs/${t}/events`, ...n ? [n] : []], Qo = (e, t, n, r) => {
	let { query: i } = r ?? {};
	return {
		queryKey: i?.queryKey ?? Zo(e, t, n),
		queryFn: ({ signal: r }) => Xo(e, t, n, r),
		enabled: e != null && t != null,
		...i
	};
};
function $o(e, t, n, r, i) {
	let a = Qo(e, t, n, r);
	return Ya(Be(a, i), a.queryKey);
}
var es = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/synth-benign/jobs`,
	method: "POST",
	headers: { "Content-Type": "application/json" },
	data: t,
	signal: n
}), ts = (e) => {
	let t = ["agentHardenerCreateSynthBenignJob"], { mutation: n } = e ? e.mutation && "mutationKey" in e.mutation && e.mutation.mutationKey ? e : {
		...e,
		mutation: {
			...e.mutation,
			mutationKey: t
		}
	} : { mutation: { mutationKey: t } };
	return {
		mutationFn: (e) => {
			let { workspace: t, data: n } = e ?? {};
			return es(t, n);
		},
		...n
	};
}, ns = (e, t) => ze(ts(e), t), rs = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/synth-benign/jobs/${encodeURIComponent(String(t))}`,
	method: "DELETE",
	signal: n
}), is = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/synth-benign/jobs/${encodeURIComponent(String(t))}/cancel`,
	method: "POST",
	signal: n
}), as = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/victim-context/jobs`,
	method: "POST",
	headers: { "Content-Type": "application/json" },
	data: t,
	signal: n
}), os = (e) => {
	let t = ["agentHardenerCreateVictimContextJob"], { mutation: n } = e ? e.mutation && "mutationKey" in e.mutation && e.mutation.mutationKey ? e : {
		...e,
		mutation: {
			...e.mutation,
			mutationKey: t
		}
	} : { mutation: { mutationKey: t } };
	return {
		mutationFn: (e) => {
			let { workspace: t, data: n } = e ?? {};
			return as(t, n);
		},
		...n
	};
}, ss = (e, t) => ze(os(e), t), cs = (e, t, n, r) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/victim-context/jobs/${encodeURIComponent(String(t))}/results/${encodeURIComponent(String(n))}/download`,
	method: "GET",
	responseType: "blob",
	signal: r
}), ls = (e, t, n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(String(e))}/victim-context/jobs/${encodeURIComponent(String(t))}`,
	method: "GET",
	signal: n
}), us = (e, t) => [`/apis/agent-hardener/v2/workspaces/${e}/victim-context/jobs/${t}`], ds = (e, t, n) => {
	let { query: r } = n ?? {};
	return {
		queryKey: r?.queryKey ?? us(e, t),
		queryFn: ({ signal: n }) => ls(e, t, n),
		enabled: e != null && t != null,
		...r
	};
};
function fs(e, t, n, r) {
	let i = ds(e, t, n);
	return Ya(Be(i, r), i.queryKey);
}
//#endregion
//#region src/components/ModelGroupFields.tsx
var ps = {
	attack: "Attack model",
	analysis: "Analysis model",
	safety: "Safety model"
}, ms = {
	attack: "The garak red-team + detector models that probe the agent.",
	analysis: "The defenders and the benign validator — both its suite generation (synth) and judging — one shared model.",
	safety: "The LLM the generated guardrail uses to screen the agent's traffic. A blank endpoint or key falls back to the analysis one."
}, hs = "";
function gs(e, t, n) {
	let r = {
		...e[t] ?? {},
		...n
	}, i = !r.model && !r.base_url && !r.api_key_secret;
	return {
		...e,
		[t]: i ? void 0 : r
	};
}
var _s = ({ label: e, help: t, divider: n, children: r }) => /* @__PURE__ */ m(I, {
	gap: "density-sm",
	className: n ? "border-t border-base pt-4" : void 0,
	children: [/* @__PURE__ */ m("div", { children: [/* @__PURE__ */ p(L, {
		kind: "body/semibold/sm",
		children: e
	}), /* @__PURE__ */ p(L, {
		kind: "body/regular/sm",
		className: "text-subtle",
		children: t
	})] }), r]
}), vs = ({ value: e, onChange: t, workspace: n, defaults: r }) => /* @__PURE__ */ m(I, {
	gap: "density-md",
	children: [
		/* @__PURE__ */ p(ys, {
			group: "attack",
			value: e,
			onChange: t,
			workspace: n,
			defaultModel: r?.attack.model,
			defaultBaseUrl: r?.attack.base_url
		}),
		/* @__PURE__ */ p(ys, {
			group: "analysis",
			value: e,
			onChange: t,
			workspace: n,
			defaultModel: r?.analysis.model,
			defaultBaseUrl: r?.analysis.base_url,
			divider: !0
		}),
		/* @__PURE__ */ p(ys, {
			group: "safety",
			value: e,
			onChange: t,
			workspace: n,
			defaultModel: r?.safety.model,
			defaultBaseUrl: e.analysis?.base_url || r?.safety.base_url,
			fallbackKeySecret: e.analysis?.api_key_secret,
			divider: !0
		})
	]
}), ys = ({ group: e, value: t, onChange: n, workspace: r, defaultModel: i, defaultBaseUrl: a, fallbackKeySecret: o, divider: s }) => {
	let c = t[e] ?? {}, [l, u] = d(!1), [f, h] = d(null), g = Lo(), { secretsListSecrets: _, useSecretsCreateSecret: v } = Va(), y = Yi(), b = (e, t) => y[t ?? "info"](e), x = v(), C = Be({
		queryKey: ["agent-hardener-model-secrets", r],
		queryFn: ({ signal: e }) => _(r, {
			page: 1,
			page_size: 100
		}, e),
		enabled: !!r
	}), w = C.data?.data.map((e) => e.name) ?? [], T = c.api_key_secret && !w.includes(c.api_key_secret) ? [c.api_key_secret, ...w] : w, E = async () => {
		h("Testing…");
		try {
			let e = await g.mutateAsync({
				workspace: r,
				data: {
					model: c.model || i || void 0,
					base_url: c.base_url || a || "",
					api_key_secret: c.api_key_secret || o || void 0
				}
			});
			h(bs(e.ok, e.reason, e.available ?? [], e.detail));
		} catch (e) {
			h(aa("Validation failed", e));
		}
	};
	return /* @__PURE__ */ m(_s, {
		label: ps[e],
		help: ms[e],
		divider: s,
		children: [
			/* @__PURE__ */ p(pe, {
				name: `${e}-model`,
				slotLabel: "Model",
				children: /* @__PURE__ */ p(je, {
					value: c.model ?? "",
					placeholder: i ?? "Default model",
					onChange: (r) => n(gs(t, e, { model: r.target.value || void 0 }))
				})
			}),
			/* @__PURE__ */ p(pe, {
				name: `${e}-base-url`,
				slotLabel: "Custom endpoint (optional)",
				slotHelp: "OpenAI-compatible base URL; leave blank to use the default NVIDIA endpoint.",
				children: /* @__PURE__ */ p(je, {
					value: c.base_url ?? "",
					placeholder: a ?? "https://…/v1",
					onChange: (r) => n(gs(t, e, { base_url: r.target.value || void 0 }))
				})
			}),
			/* @__PURE__ */ p(pe, {
				name: `${e}-secret`,
				slotLabel: "API key secret (optional)",
				slotHelp: "A Secret holding the provider key for a custom endpoint.",
				children: /* @__PURE__ */ m(Se, {
					value: c.api_key_secret ?? hs,
					onValueChange: (r) => r === "__create__" ? u(!0) : n(gs(t, e, { api_key_secret: r || void 0 })),
					children: [/* @__PURE__ */ p(Ce, {
						className: "w-full",
						placeholder: "Select a secret (optional)"
					}), /* @__PURE__ */ p(ye, {
						className: "w-(--radix-popper-anchor-width)",
						children: /* @__PURE__ */ m(xe, { children: [
							/* @__PURE__ */ p(be, {
								value: hs,
								children: "None"
							}),
							T.map((e) => /* @__PURE__ */ p(be, {
								value: e,
								children: e
							}, e)),
							/* @__PURE__ */ p(be, {
								value: "__create__",
								children: "+ Create new secret…"
							})
						] })
					})]
				})
			}),
			/* @__PURE__ */ m(F, {
				align: "center",
				gap: "density-sm",
				children: [/* @__PURE__ */ p(P, {
					kind: "secondary",
					size: "small",
					disabled: g.isPending || !c.model && !c.base_url && !c.api_key_secret,
					onClick: () => void E(),
					children: "Test connection"
				}), f && /* @__PURE__ */ p(L, {
					kind: "body/regular/sm",
					className: "text-subtle",
					children: f
				})]
			}),
			He(/* @__PURE__ */ p(S, {
				open: l,
				onClose: () => u(!1),
				pending: x.isPending,
				errorText: x.error ? ne(x.error) : void 0,
				onNotify: b,
				onCreate: async (i) => {
					n(gs(t, e, { api_key_secret: (await x.mutateAsync({
						workspace: r,
						data: i
					})).name })), C.refetch(), u(!1);
				}
			}), document.body)
		]
	});
};
function bs(e, t, n = [], r) {
	return e ? "Connection OK." : t === "auth" ? `Credentials rejected${r ? ` (${r})` : ""}.` : t === "unreachable" ? `Endpoint unreachable${r ? ` (${r})` : ""}.` : t === "unknown_model" ? `Model not found. Reachable: ${n.slice(0, 8).join(", ") || "none"}${n.length > 8 ? ", …" : ""}.` : r || "Validation failed.";
}
//#endregion
//#region src/routes/AgentHardenerManifestDetailRoute/constants.ts
var xs = {
	light: "Light",
	standard: "Standard",
	thorough: "Thorough"
}, Ss = {
	last: "Last run",
	upload: "Upload hitlog"
}, Cs = {
	manifest: "Manifest default",
	upload: "Upload CSV"
}, ws = [
	"tool",
	"payload",
	"label",
	"rationale",
	"persona"
], Ts = ({ open: e, onClose: t, workspace: n, manifestName: r, manifest: i, starting: a, savingDefault: o, onStart: c, onSaveDefault: l }) => {
	let h = Yi(), { data: _ } = Po(n, { query: {} }), [v, y] = d(""), [b, x] = d({
		guardrails: !0,
		openshell: !0
	}), [S, C] = d("standard"), [w, D] = d("1"), [O, ee] = d({}), [k, A] = d("live"), [te, j] = d("last"), [ne, re] = d(), [M, ie] = d(), [N, ae] = d("manifest"), [oe, se] = d(), [le, ue] = d(), de = u(i);
	de.current = i;
	let me = u(!1);
	s(() => {
		if (!e) {
			me.current = !1;
			return;
		}
		if (me.current) return;
		me.current = !0;
		let t = de.current;
		y(t?.port ? String(t.port) : "");
		let n = t?.defenders ?? [];
		x(n.length ? {
			guardrails: n.includes("guardrails"),
			openshell: n.includes("openshell")
		} : {
			guardrails: !0,
			openshell: !0
		}), C(t?.attack_intensity ?? "standard"), D(t?.rounds ? String(t.rounds) : "1"), ee(t?.models ?? {}), A("live"), j("last"), re(void 0), ie(void 0), ae("manifest"), se(void 0), ue(void 0);
	}, [e]);
	let { data: he } = Vo(n, {
		sort: "-created_at",
		page_size: 20,
		filter: { manifest_id: r }
	}, { query: { enabled: e && k === "replay" && te === "last" } }), ge = (he?.data ?? []).find((e) => e.hitlog_fileset)?.hitlog_fileset, _e = Wa(), we = async (e) => {
		let t = e[0];
		if (t) {
			re(t), ie(void 0);
			try {
				ie(await _e.mutateAsync({
					workspace: n,
					manifestName: r,
					file: t
				}));
			} catch (e) {
				oa(h, "Failed to upload the hitlog file")(e);
			}
		}
	}, Te = Ga(), Ee = async (e) => {
		let t = e[0];
		if (t) {
			se(t), ue(void 0);
			try {
				ue(await Te.mutateAsync({
					workspace: n,
					manifestName: r,
					file: t
				}));
			} catch (e) {
				oa(h, "Failed to upload the benign suite file")(e);
			}
		}
	}, De = () => {
		let e = ["guardrails", "openshell"].filter((e) => b[e]);
		return e.length === 0 ? (h.error("Select at least one defender."), null) : {
			defenders: e,
			attack_intensity: S,
			rounds: Number(w) || 1,
			models: O,
			...v ? { port: Number(v) } : {}
		};
	}, Oe = () => {
		let e = De();
		if (!e) return;
		let t;
		if (k === "replay" && (t = te === "upload" ? M : ge, !t)) {
			h.error(te === "upload" ? "Upload a hitlog file to replay first." : "No previous run of this manifest has recorded hits — run a live attack once, or upload a hitlog.");
			return;
		}
		if (N === "upload" && !le) {
			h.error("Upload a benign suite CSV first.");
			return;
		}
		c({
			...e,
			...t ? { replay_hitlog_fileset: t } : {},
			...N === "upload" && le ? { benign_suite_fileset: le } : {}
		});
	};
	return /* @__PURE__ */ p(E, {
		open: e,
		title: "Start war-game",
		submitButtonText: "Start",
		loading: a,
		submitDisabled: k === "replay" && (te === "upload" ? !M || _e.isPending : !ge) || N === "upload" && (!le || Te.isPending),
		onSubmit: (e) => {
			e.preventDefault(), Oe();
		},
		onClose: t,
		children: /* @__PURE__ */ m(I, {
			gap: "density-md",
			children: [
				/* @__PURE__ */ p(L, {
					kind: "body/regular/sm",
					className: "text-subtle",
					children: "Config applies to this run only. Use “Save as default” to make it the manifest baseline."
				}),
				/* @__PURE__ */ p(pe, {
					name: "port",
					slotLabel: "Victim Port",
					slotHelp: "Port the war-game targets on the victim agent.",
					children: /* @__PURE__ */ p(je, {
						value: v,
						onChange: (e) => y(e.target.value.replace(/[^0-9]/g, ""))
					})
				}),
				/* @__PURE__ */ m(I, {
					gap: "density-sm",
					children: [
						/* @__PURE__ */ p(L, {
							kind: "body/semibold/sm",
							className: "uppercase tracking-wide text-subtle",
							children: "Defenders"
						}),
						/* @__PURE__ */ m(F, {
							align: "center",
							gap: "density-sm",
							children: [/* @__PURE__ */ p(fe, {
								checked: b.guardrails,
								onCheckedChange: (e) => x((t) => ({
									...t,
									guardrails: e === !0
								})),
								attributes: { CheckboxInput: { "aria-label": "Guardrails defender" } }
							}), /* @__PURE__ */ p(L, {
								kind: "body/regular/sm",
								children: "Guardrails defender"
							})]
						}),
						/* @__PURE__ */ m(F, {
							align: "center",
							gap: "density-sm",
							children: [/* @__PURE__ */ p(fe, {
								checked: b.openshell,
								onCheckedChange: (e) => x((t) => ({
									...t,
									openshell: e === !0
								})),
								attributes: { CheckboxInput: { "aria-label": "OpenShell policy defender" } }
							}), /* @__PURE__ */ p(L, {
								kind: "body/regular/sm",
								children: "OpenShell policy defender"
							})]
						})
					]
				}),
				/* @__PURE__ */ p(pe, {
					name: "intensity",
					slotLabel: "Attack Intensity",
					slotHelp: "How hard the garak attacker probes the agent — more probes and generations at higher levels.",
					children: /* @__PURE__ */ m(Se, {
						value: xs[S],
						onValueChange: (e) => C(e.toLowerCase()),
						children: [/* @__PURE__ */ p(Ce, {
							className: "w-full",
							placeholder: "Select intensity"
						}), /* @__PURE__ */ p(ye, {
							className: "w-(--radix-popper-anchor-width)",
							children: /* @__PURE__ */ m(xe, { children: [
								/* @__PURE__ */ p(be, {
									value: "Light",
									children: "Light"
								}),
								/* @__PURE__ */ p(be, {
									value: "Standard",
									children: "Standard"
								}),
								/* @__PURE__ */ p(be, {
									value: "Thorough",
									children: "Thorough"
								})
							] })
						})]
					})
				}),
				/* @__PURE__ */ p(pe, {
					name: "rounds",
					slotLabel: "Rounds",
					slotHelp: "Iterative attack → defend → validate → redeploy cycles. More rounds go deeper but take longer.",
					children: /* @__PURE__ */ p(je, {
						value: w,
						onChange: (e) => D(e.target.value.replace(/[^0-9]/g, ""))
					})
				}),
				/* @__PURE__ */ p(ce, { children: /* @__PURE__ */ p(g, {
					value: "models",
					title: "Models (optional)",
					children: /* @__PURE__ */ p(vs, {
						value: O,
						onChange: ee,
						workspace: n,
						defaults: _
					})
				}) }),
				/* @__PURE__ */ p(pe, {
					name: "benignSource",
					slotLabel: "Benign suite",
					slotHelp: "The benign requests replayed after hardening to confirm the agent still works. Defaults to the manifest's suite; upload a requests.csv to override it for this run.",
					children: /* @__PURE__ */ m(Se, {
						value: Cs[N],
						onValueChange: (e) => ae(e === Cs.upload ? "upload" : "manifest"),
						children: [/* @__PURE__ */ p(Ce, {
							className: "w-full",
							placeholder: "Select benign suite"
						}), /* @__PURE__ */ p(ye, {
							className: "w-(--radix-popper-anchor-width)",
							children: /* @__PURE__ */ m(xe, { children: [/* @__PURE__ */ p(be, {
								value: Cs.manifest,
								children: Cs.manifest
							}), /* @__PURE__ */ p(be, {
								value: Cs.upload,
								children: Cs.upload
							})] })
						})]
					})
				}),
				N === "upload" ? /* @__PURE__ */ p(T, {
					label: "Benign suite",
					accept: { "text/csv": [".csv"] },
					multiple: !1,
					files: oe ? [oe] : [],
					onDropAccepted: (e) => void Ee(e),
					onRemoveFile: () => {
						se(void 0), ue(void 0);
					},
					helperText: Te.isPending ? "Uploading…" : le ? "Uploaded — will override the manifest suite for this run." : "A benign requests.csv (tool,payload,label,rationale,persona)."
				}) : null,
				/* @__PURE__ */ p(pe, {
					name: "mode",
					slotLabel: "Attack mode",
					slotHelp: "Live runs a fresh garak attack; Replay skips it and replays recorded hits against the defended agent.",
					children: /* @__PURE__ */ p(ve, {
						className: "w-full",
						value: k,
						onValueChange: (e) => A(e),
						items: [{
							value: "live",
							children: "Live attack"
						}, {
							value: "replay",
							children: "Replay recorded hits"
						}]
					})
				}),
				k === "replay" ? /* @__PURE__ */ m(f, { children: [/* @__PURE__ */ p(pe, {
					name: "replaySource",
					slotLabel: "Hits to replay",
					children: /* @__PURE__ */ m(Se, {
						value: Ss[te],
						onValueChange: (e) => j(e === Ss.upload ? "upload" : "last"),
						children: [/* @__PURE__ */ p(Ce, {
							className: "w-full",
							placeholder: "Select hits to replay"
						}), /* @__PURE__ */ p(ye, {
							className: "w-(--radix-popper-anchor-width)",
							children: /* @__PURE__ */ m(xe, { children: [/* @__PURE__ */ p(be, {
								value: Ss.last,
								children: Ss.last
							}), /* @__PURE__ */ p(be, {
								value: Ss.upload,
								children: Ss.upload
							})] })
						})]
					})
				}), te === "last" ? /* @__PURE__ */ p(L, {
					kind: "body/regular/sm",
					className: ge ? "text-subtle" : void 0,
					style: ge ? void 0 : { color: Fa.warning },
					children: ge ? "Replays this manifest's most recent recorded hits." : "No previous run of this manifest has recorded hits — run a live attack once, or upload a hitlog."
				}) : /* @__PURE__ */ p(T, {
					label: "Hitlog",
					accept: { "application/jsonl": [".jsonl", ".json"] },
					multiple: !1,
					files: ne ? [ne] : [],
					onDropAccepted: (e) => void we(e),
					onRemoveFile: () => {
						re(void 0), ie(void 0);
					},
					helperText: _e.isPending ? "Uploading…" : M ? "Uploaded — ready to replay." : "A garak hitlog (.jsonl) recording the attack hits to replay."
				})] }) : null,
				/* @__PURE__ */ p(F, { children: /* @__PURE__ */ p(P, {
					kind: "tertiary",
					type: "button",
					disabled: o,
					onClick: () => {
						let e = De();
						e && l(e);
					},
					children: "Save as default"
				}) })
			]
		})
	});
}, Es = ({ label: e, children: t }) => /* @__PURE__ */ m(F, {
	gap: "density-md",
	className: "items-baseline",
	children: [/* @__PURE__ */ p(L, {
		kind: "body/regular/sm",
		className: "shrink-0 text-fg-secondary",
		style: { width: "8rem" },
		children: e
	}), /* @__PURE__ */ p(L, {
		kind: "body/regular/sm",
		className: "break-all",
		children: t
	})]
}), Ds = ({ children: e }) => /* @__PURE__ */ p("span", {
	className: "text-fg-secondary",
	children: e
}), Os = ({ manifest: e, onRefresh: t, refreshing: n, onEditEnv: r }) => {
	let i = e.egress ?? [], a = e.secrets ?? [], o = Object.entries(e.env ?? {});
	return /* @__PURE__ */ p(ge, { children: /* @__PURE__ */ m(I, {
		gap: "density-lg",
		padding: "density-lg",
		children: [
			/* @__PURE__ */ m(F, {
				className: "items-center justify-between",
				children: [/* @__PURE__ */ p(L, {
					kind: "body/semibold/md",
					children: "Target"
				}), t ? /* @__PURE__ */ p(P, {
					kind: "secondary",
					size: "small",
					disabled: n,
					onClick: t,
					children: n ? "Refreshing" : "Refresh Target"
				}) : null]
			}),
			/* @__PURE__ */ p(L, {
				kind: "body/regular/sm",
				className: "text-fg-secondary",
				children: "Resolved from the agent when this manifest was created. Editing the agent afterwards does not change it — refresh to take those changes."
			}),
			/* @__PURE__ */ m(I, {
				gap: "density-sm",
				children: [
					/* @__PURE__ */ p(Es, {
						label: "Agent",
						children: e.agent || /* @__PURE__ */ p(Ds, { children: "unknown" })
					}),
					/* @__PURE__ */ p(Es, {
						label: "Victim Port",
						children: e.port || /* @__PURE__ */ p(Ds, { children: "not set" })
					}),
					/* @__PURE__ */ p(Es, {
						label: "Egress",
						children: i.length ? i.join(", ") : /* @__PURE__ */ p(Ds, { children: "none — the victim's outbound calls are blocked" })
					}),
					/* @__PURE__ */ p(Es, {
						label: "Secrets",
						children: a.length ? a.join(", ") : /* @__PURE__ */ p(Ds, { children: "none" })
					}),
					/* @__PURE__ */ p(Es, {
						label: "Environment",
						children: /* @__PURE__ */ m(F, {
							gap: "density-md",
							className: "items-baseline",
							children: [/* @__PURE__ */ p("span", { children: o.length ? o.map(([e, t]) => `${e}=${t}`).join(", ") : /* @__PURE__ */ p(Ds, { children: "none" }) }), r ? /* @__PURE__ */ p(P, {
								kind: "tertiary",
								size: "small",
								onClick: r,
								children: "Edit"
							}) : null]
						})
					})
				]
			}),
			e.manifest_yaml ? /* @__PURE__ */ p(ce, { children: /* @__PURE__ */ p(g, {
				value: "manifest-yaml",
				title: "agent-hardener.yaml",
				children: /* @__PURE__ */ p("pre", {
					className: "overflow-auto px-density-xs text-sm text-fg-secondary whitespace-pre-wrap",
					style: { maxHeight: "24rem" },
					children: e.manifest_yaml
				})
			}) }) : null
		]
	}) });
}, ks = async ({ workspace: e, name: t }, n, r) => {
	try {
		await n(e, t);
	} catch (n) {
		if (!Ai(n) || n.response?.status !== 404) throw n;
		await r(e, t);
	}
}, As = (e) => ks(e, ao, is), js = async (e, t) => {
	if (t.job_id) try {
		await ks({
			workspace: e,
			name: t.job_id
		}, io, rs);
	} catch (e) {
		if (!Ai(e) || e.response?.status !== 404) throw e;
	}
	await Ko(e, t.name);
}, Ms = async (e, t, { attempts: n, intervalMs: r, stop: i }) => {
	for (let a = 0; a < n && !i?.(); a++) {
		let { data: n } = await Ro(e, {
			sort: "-created_at",
			page_size: 20
		}), i = n?.find((e) => e.job_id === t);
		if (i?.name) return i.name;
		await new Promise((e) => setTimeout(e, r));
	}
}, Ns = (e) => {
	let t = e?.interview, n = e?.interview_response;
	return !t || typeof t.round != "number" || n?.round === t.round ? null : t;
}, Ps = (e) => {
	let t = e?.review, n = e?.review_response;
	return !t || typeof t.round != "number" || n?.round === t.round ? null : t;
}, Fs = {
	analyzer: "#c855fa",
	attacker: "#ff3855",
	defender: "#00e676",
	victim: "#448aff",
	validator: "#ffab40",
	update: "#29b6f6",
	summary: "#cfd8dc"
}, Is = [
	{
		id: "benign_analyzer",
		title: "Benign Analyzer",
		group: "analyzer",
		x: 170,
		y: 430
	},
	{
		id: "attacker_manager",
		title: "Attacker Manager",
		group: "attacker",
		x: 190,
		y: 150,
		isManager: !0
	},
	{
		id: "attacker",
		title: "Attacker",
		group: "attacker",
		x: 190,
		y: 290
	},
	{
		id: "victim_agent",
		title: "Victim Agent",
		group: "victim",
		x: 500,
		y: 300
	},
	{
		id: "defender_manager",
		title: "Defender Manager",
		group: "defender",
		x: 810,
		y: 150,
		isManager: !0
	},
	{
		id: "guardrails_defender",
		title: "Guardrails Defender",
		group: "defender",
		x: 730,
		y: 290
	},
	{
		id: "openshell_defender",
		title: "OpenShell Defender",
		group: "defender",
		x: 890,
		y: 290
	},
	{
		id: "update_victim_agent_policy",
		title: "Deploy Agent",
		group: "update",
		x: 810,
		y: 430
	},
	{
		id: "validator_manager",
		title: "Validator Manager",
		group: "validator",
		x: 440,
		y: 600,
		isManager: !0
	},
	{
		id: "attacker_validator",
		title: "Attacker Validator",
		group: "validator",
		x: 330,
		y: 660
	},
	{
		id: "benign_validator",
		title: "Benign Validator",
		group: "validator",
		x: 550,
		y: 660
	},
	{
		id: "summary",
		title: "Summary",
		group: "summary",
		x: 810,
		y: 600
	}
], Ls = [
	{
		from: "attacker_manager",
		to: "attacker"
	},
	{
		from: "attacker",
		to: "victim_agent"
	},
	{
		from: "benign_analyzer",
		to: "victim_agent"
	},
	{
		from: "defender_manager",
		to: "guardrails_defender"
	},
	{
		from: "defender_manager",
		to: "openshell_defender"
	},
	{
		from: "defender_manager",
		to: "update_victim_agent_policy"
	},
	{
		from: "guardrails_defender",
		to: "victim_agent"
	},
	{
		from: "openshell_defender",
		to: "victim_agent"
	},
	{
		from: "update_victim_agent_policy",
		to: "victim_agent"
	},
	{
		from: "victim_agent",
		to: "validator_manager"
	},
	{
		from: "validator_manager",
		to: "attacker_validator"
	},
	{
		from: "validator_manager",
		to: "benign_validator"
	},
	{
		from: "validator_manager",
		to: "summary"
	}
], Rs = {
	attackers: ["attacker_manager", "attacker"],
	defenders: [
		"defender_manager",
		"guardrails_defender",
		"openshell_defender"
	],
	victim: ["victim_agent"],
	validators: [
		"validator_manager",
		"attacker_validator",
		"benign_validator"
	]
}, zs = (e) => e.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, ""), Bs = Object.fromEntries(Is.map((e) => [zs(e.title), e.id])), Vs = {
	attacker: "attacker_manager",
	defender: "defender_manager",
	validator: "validator_manager"
}, Hs = (e) => typeof e == "string" ? e : void 0, Us = (e) => {
	let t = Hs(e.agent_name);
	if (t && Bs[zs(t)]) return Bs[zs(t)];
	let n = t ? zs(t) : "", r = Hs(e.agent_role), i = Hs(e.validator_kind);
	switch (r) {
		case "attacker": return "attacker";
		case "victim": return "victim_agent";
		case "defender": return n.includes("guardrail") ? "guardrails_defender" : n.includes("openshell") || n.includes("policy") ? "openshell_defender" : "defender_manager";
		case "validator": return i === "attack" ? "attacker_validator" : i === "benign" ? "benign_validator" : "validator_manager";
		default: return r ? Vs[r] : void 0;
	}
}, Ws = (e) => {
	let t = Object.fromEntries(Is.map((e) => [e.id, "pending"])), n = {}, r = {}, i = {}, a = "", o = 0, s = !1, c = (e, n) => e?.forEach((e) => t[e] = n), l = (e, t) => (n[e] ??= []).push(t), u = (e, t) => (r[e] ??= []).push(t), d = (e, t) => (i[e] ??= []).push(t);
	for (let n of e) {
		let { event: e, payload: r } = n, i = Hs(r.phase);
		switch (e) {
			case "phase_started":
				c(i ? Rs[i] : void 0, "running"), i && (a = i);
				break;
			case "phase_completed":
				c(i ? Rs[i] : void 0, "success");
				break;
			case "agent_started": {
				let e = Us(r);
				e && (t[e] = "running", l(e, {
					ts: n.ts,
					label: "started",
					text: Hs(r.agent_name) ?? "",
					level: "info"
				}));
				break;
			}
			case "agent_progress": {
				let e = Us(r), t = Hs(r.message);
				e && t && l(e, {
					ts: n.ts,
					label: "progress",
					text: t,
					level: "info"
				});
				break;
			}
			case "agent_completed": {
				let e = Us(r);
				if (e) {
					let i = Hs(r.not_applicable);
					t[e] = i ? "skipped" : r.ok === !1 ? "failed" : "success";
					let a = typeof r.duration_seconds == "number" ? ` (${r.duration_seconds.toFixed(1)}s)` : "";
					l(e, {
						ts: n.ts,
						label: `completed${a}`,
						text: i ? `did not apply: ${i}` : Hs(r.summary) ?? "",
						level: "info"
					});
				}
				break;
			}
			case "agent_failed": {
				let e = Us(r);
				e && (t[e] = "failed", l(e, {
					ts: n.ts,
					label: "failed",
					text: Hs(r.error) ?? "",
					level: "error"
				}));
				break;
			}
			case "agent_exchange": {
				let e = Us(r), t = {
					ts: n.ts,
					request: Hs(r.request) ?? "",
					response: Hs(r.response) ?? "",
					label: Hs(r.label),
					ok: r.ok !== !1,
					blocked: typeof r.blocked == "boolean" ? r.blocked : void 0
				};
				e && u(e, t);
				let i = Hs(r.agent_role);
				e !== "victim_agent" && (i === "attacker" || i === "benign" || i === "validator") && u("victim_agent", t);
				break;
			}
			case "llm_call": {
				let e = Us(r);
				e && d(e, {
					ts: n.ts,
					request: Hs(r.request) ?? "",
					response: Hs(r.response) ?? "",
					label: Hs(r.label),
					ok: r.ok !== !1
				});
				break;
			}
			case "synth_phase":
			case "interview_started":
				t.benign_analyzer = "running";
				break;
			case "interview_completed":
				t.benign_analyzer = "success";
				break;
			case "victim_control_started":
			case "openshell_upload":
			case "relay_policy_upload":
				t.update_victim_agent_policy = "running", t.victim_agent = "running";
				break;
			case "victim_control_completed":
				t.update_victim_agent_policy = "success", t.victim_agent = "success";
				break;
			case "round_started":
				o = typeof r.round == "number" ? r.round : o + 1;
				break;
			case "report_written":
				t.summary = "success";
				break;
			case "round_completed": r.success === !0 && (s = !0, a = "FINAL PASS", t.summary = "success");
		}
	}
	return {
		statuses: t,
		phase: a,
		round: o,
		finalPass: s,
		nodeLogs: n,
		nodeExchanges: r,
		nodeLlmCalls: i
	};
}, Gs = (e) => {
	let t = /* @__PURE__ */ new Map();
	for (let n of e) {
		if (n.event !== "synth_phase") continue;
		let e = Hs(n.payload.phase);
		e && t.set(e, {
			phase: e,
			label: Ks(n.payload, e)
		});
	}
	return [...t.values()];
}, Ks = (e, t) => {
	let n = Hs(e.label) ?? t, r = Hs(e.detail);
	return r ? `${n} — ${r}` : n;
}, qs = (e) => {
	let t;
	for (let n of e) n.event === "status_started" ? t = Hs(n.payload.label) ?? t : n.event === "victim_control_started" ? t = "Deploying victim…" : n.event === "victim_control_completed" && (t = "Victim deployed");
	return t;
}, Js = 1e3, Ys = (e, t, n = !1) => {
	let [r, i] = d(0), [a, o] = d([]), { data: c } = $o(e, t, { after: r }, { query: {
		enabled: !!t,
		refetchInterval: !n && Js
	} });
	return s(() => {
		if (!c?.events?.length) return;
		let e = c.events.filter((e) => typeof e.id == "number").map((e) => ({
			id: e.id,
			event: typeof e.event == "string" ? e.event : "",
			payload: e.payload ?? {},
			ts: Date.now()
		}));
		e.length && (o((t) => [...t, ...e]), i((t) => Math.max(t, ...e.map((e) => e.id))));
	}, [c]), s(() => {
		o([]), i(0);
	}, [e, t]), a;
}, Xs = (e) => typeof e == "string" && e.trim() ? e.trim() : void 0;
function Zs(e) {
	let t = Xs(e?.error_details?.message);
	if (t) return {
		message: t,
		remediation: Xs(e?.error_details?.remediation)
	};
}
function Qs(e) {
	let t = e?.status;
	if (!(!t || t === "completed" || !v.includes(t))) return t === "cancelled" ? { message: "The job was cancelled." } : Zs(e) ?? { message: "The job ended without completing and left no error message." };
}
//#endregion
//#region src/components/useJobFailure.ts
var $s = (e, t, n, { announce: r = !0 } = {}) => {
	let i = Yi(), a = Qs(n), o = u("");
	return s(() => {
		!a || !r || o.current === t || (o.current = t, i.error(`${e} failed: ${a.message}`));
	}, [
		a,
		r,
		t,
		e,
		i
	]), a;
}, ec = (e, t, n) => {
	let r = Yi(), [i, a] = d(""), [o, c] = d(""), f = u(""), p = async (t) => {
		try {
			let n = await Ms(e, t, {
				attempts: 15,
				intervalMs: 1e3
			});
			n && c(n);
		} catch (e) {
			oa(r, "Could not follow the generation run")(e);
		}
	}, m = u(""), [h, g] = d(!1);
	s(() => {
		i || m.current === t || (m.current = t, (async () => {
			try {
				let { data: n } = await Ro(e, {
					sort: "-created_at",
					page_size: 1,
					filter: {
						manifest_id: t,
						status: "running"
					}
				}), r = n?.[0];
				r?.job_id && (c(r.name ?? ""), a(r.job_id), g(!0));
			} catch (e) {
				oa(r, "Could not check for a generation in progress")(e);
			}
		})());
	}, [
		e,
		t,
		i,
		r
	]);
	let _ = ns({ mutation: {
		onSuccess: (e) => {
			c(""), f.current = "", m.current = t, g(!1), a(e.name), p(e.name);
		},
		onError: oa(r, "Failed to start benign-suite generation")
	} }), { useJobsUpdateJobAttemptStatusDetails: y } = Va(), { data: b } = Ha(e, i), x = b?.status_details, S = Ns(x), C = Ps(x), w = y(), T = (t) => b && w.mutate({
		workspace: e,
		job: i,
		attempt: b.attempt_id,
		data: t
	}), E = b?.status, D = !!(E && v.includes(E)), O = $s("Benign-suite generation", i, b, { announce: !h }), ee = Ys(e, o, D), k = l(() => Gs(ee), [ee]), A = l(() => qs(ee), [ee]);
	return s(() => {
		i && D && f.current !== i && (f.current = i, E === "completed" && n?.());
	}, [
		i,
		D,
		E,
		n
	]), {
		start: () => _.mutate({
			workspace: e,
			data: { spec: {
				manifest_id: t,
				driver: "service"
			} }
		}),
		active: _.isPending || !!i && !D,
		starting: _.isPending || !!i && !o,
		status: E,
		jobName: i,
		failure: O,
		activity: A,
		recon: k,
		interview: S,
		review: C,
		submitInterview: (e) => S && T({ interview_response: {
			round: S.round,
			answers: e
		} }),
		submitReview: (e) => C && T({ review_response: {
			round: C.round,
			suite: e
		} }),
		isResponding: w.isPending
	};
}, tc = (e) => {
	let t = Ie(), n = Yi(), [r, i] = d(""), { data: a } = Ha(e, r), o = $s("War-game", r, a), c = u("");
	s(() => {
		o && (c.current = r);
	}, [o, r]);
	let l = async (r) => {
		let i = () => c.current === r, a;
		try {
			a = await Ms(e, r, {
				attempts: 60,
				intervalMs: 500,
				stop: i
			});
		} catch (e) {
			oa(n, "Could not open the war-game run")(e);
			return;
		}
		a ? t(ea(e, a)) : i() || (n.info("The run has not appeared yet; it will show in this list once its job starts."), t($i(e)));
	};
	return {
		...no({ mutation: {
			onSuccess: (e) => {
				n.success("War-game started — opening the run…"), i(e.name), l(e.name);
			},
			onError: oa(n, "Failed to start the war-game")
		} }),
		jobName: r,
		failure: o
	};
}, nc = "victim-context", rc = (e, t) => {
	let n = Yi(), [r, i] = d(""), [a, o] = d(!1), s = `${t}-victim-context.tar.gz`, c = ss({ mutation: {
		onSuccess: (e) => i(e.name),
		onError: oa(n, "Failed to start staging the victim build context")
	} }), { data: l } = fs(e, r, { query: {
		enabled: !!r,
		refetchInterval: (e) => re(e.state.data?.status)
	} }), u = l?.status, f = !!(u && v.includes(u)), p = $s("Staging the victim build context", r, l);
	return {
		start: () => c.mutate({
			workspace: e,
			data: { spec: { manifest_id: t } }
		}),
		staging: c.isPending || !!r && !f,
		ready: u === "completed",
		jobName: r,
		failure: p,
		download: async () => {
			o(!0);
			try {
				let t = await cs(e, r, nc);
				ae(t, s);
			} catch (e) {
				oa(n, "Failed to download the victim build context")(e);
			} finally {
				o(!1);
			}
		},
		downloading: a,
		archiveName: s
	};
}, ic = "<registry>/<image>:<tag>", ac = (e) => !/\s/.test(e), oc = ({ workspace: e, manifestName: t, victimImage: n, saving: r, onSave: i }) => {
	let a = rc(e, t), [o, c] = d(n);
	s(() => c(n), [n]);
	let l = ac(o.trim()), u = o.trim() || ic, f = [`docker build -t ${u} - < ${a.archiveName}`, `docker push ${u}`].join("\n");
	return /* @__PURE__ */ p(ge, { children: /* @__PURE__ */ m(I, {
		gap: "density-lg",
		padding: "density-lg",
		children: [
			/* @__PURE__ */ p(L, {
				kind: "body/semibold/md",
				children: "Victim Image"
			}),
			/* @__PURE__ */ p(L, {
				kind: "body/regular/sm",
				className: "text-fg-secondary",
				children: "War-games here run on Kubernetes, which cannot build the victim. Stage its build context, build and push the image to a registry the OpenShell gateway can pull from, then record it below. Re-stage after refreshing the target or upgrading Agent Hardener: a run rejects an image built from a different context."
			}),
			n ? null : /* @__PURE__ */ p(ue, {
				kind: "inline",
				status: "warning",
				children: /* @__PURE__ */ p(L, {
					kind: "body/regular/sm",
					children: "No victim image yet. War-games and benign-suite generation stay disabled until one is set."
				})
			}),
			/* @__PURE__ */ m(F, {
				gap: "density-sm",
				className: "items-center",
				children: [/* @__PURE__ */ p(P, {
					kind: "secondary",
					size: "small",
					disabled: a.staging,
					onClick: a.start,
					children: a.staging ? "Staging…" : "Stage Build Context"
				}), a.ready ? /* @__PURE__ */ p(P, {
					kind: "secondary",
					size: "small",
					disabled: a.downloading,
					onClick: () => void a.download(),
					children: a.downloading ? "Downloading…" : `Download ${a.archiveName}`
				}) : null]
			}),
			a.failure && !a.staging ? /* @__PURE__ */ p(Na, {
				title: "Staging failed",
				failure: a.failure,
				workspace: e,
				jobName: a.jobName
			}) : null,
			/* @__PURE__ */ p("pre", {
				className: "overflow-auto rounded bg-surface-sunken p-2 text-xs text-primary",
				children: f
			}),
			/* @__PURE__ */ m(F, {
				gap: "density-sm",
				className: "items-end",
				children: [/* @__PURE__ */ p(pe, {
					className: "grow",
					name: "victim_image",
					slotLabel: "Image Reference",
					slotError: l ? void 0 : "An image reference cannot contain whitespace.",
					children: /* @__PURE__ */ p(je, {
						value: o,
						placeholder: ic,
						onChange: (e) => c(e.target.value)
					})
				}), /* @__PURE__ */ p(P, {
					kind: "secondary",
					disabled: r || !l || o.trim() === n,
					onClick: () => i(o.trim()),
					children: r ? "Saving…" : "Save"
				})]
			})
		]
	}) });
}, sc = (e) => ie(e, {
	columns: ws,
	formulaSafe: !1
}), cc = () => {
	let e = qi(), { agentHardenerManifestName: t = "" } = Le(), n = Yi(), r = Ji(), i = Ve();
	Xi({ items: [
		{
			href: $i(e),
			slotLabel: "Agent Hardening"
		},
		{
			href: ta(e),
			slotLabel: "Manifests"
		},
		{ slotLabel: t }
	] });
	let { data: o, isLoading: c } = xo(e, t, { query: { enabled: !!t } }), { data: l } = $a(e), f = !!l?.victim_image_required, _ = f && !o?.victim_image, [v, b] = d([]), [x, S] = d(!1), C = u(!1);
	s(() => {
		!o || C.current || (C.current = !0, b((o.benign_suite ?? []).map((e) => ({
			tool: e.tool ?? "",
			payload: e.payload ?? "",
			label: e.label,
			persona: e.persona,
			rationale: e.rationale
		}))));
	}, [o]);
	let w = wo({ mutation: {
		onSuccess: () => {
			n.success("Manifest saved."), i.invalidateQueries({ queryKey: yo(e, t) });
		},
		onError: oa(n, "Failed to save the manifest")
	} }), T = tc(e), D = wo(), [O, ee] = d(!1), [k, A] = d(!1), [te, j] = d(!1), [ne, re] = d(""), M = ec(e, t, a(() => {
		n.success("Benign suite generated."), i.invalidateQueries({ queryKey: yo(e, t) }), C.current = !1;
	}, [
		n,
		i,
		e,
		t
	])), ie = async () => (await D.mutateAsync({
		workspace: e,
		name: t,
		data: { benign_suite: [] }
	}), b([]), i.invalidateQueries({ queryKey: yo(e, t) }), !0), N = a(() => {
		re(Object.entries(o?.env ?? {}).map(([e, t]) => `${e}=${t}`).join(", ")), j(!0);
	}, [o?.env]), oe = async () => {
		let r = {};
		for (let e of ne.split(",").map((e) => e.trim())) {
			let t = e.indexOf("=");
			t > 0 && (r[e.slice(0, t).trim()] = e.slice(t + 1).trim());
		}
		try {
			await D.mutateAsync({
				workspace: e,
				name: t,
				data: { env: r }
			}), i.invalidateQueries({ queryKey: yo(e, t) }), n.success("Environment variables saved."), j(!1);
		} catch (e) {
			oa(n, "Failed to save the environment variables")(e);
		}
	}, se = Ao();
	return /* @__PURE__ */ m(h, {
		title: `Agent Hardener manifest — ${t}`,
		children: [
			/* @__PURE__ */ m(I, {
				className: "h-full overflow-auto",
				gap: "density-2xl",
				padding: "density-2xl",
				children: [
					/* @__PURE__ */ p(he, {
						className: "p-0",
						slotHeading: t,
						slotDescription: o?.agent ? `Hardens ${o.agent}` : void 0,
						slotActions: /* @__PURE__ */ m(F, {
							gap: "density-sm",
							children: [/* @__PURE__ */ p(P, {
								kind: "secondary",
								disabled: M.active || _,
								onClick: () => M.start(),
								children: M.active ? "Generating…" : v.length ? "Regenerate benign suite" : "Generate benign suite"
							}), /* @__PURE__ */ p(P, {
								color: "brand",
								disabled: T.isPending || _,
								onClick: () => {
									if (v.length === 0) {
										n.error("No benign suite yet — generate it first, then run the war-game.");
										return;
									}
									S(!0);
								},
								children: "Run war-game"
							})]
						})
					}),
					T.failure ? /* @__PURE__ */ p(Na, {
						title: "War-game failed",
						failure: T.failure,
						workspace: e,
						jobName: T.jobName
					}) : null,
					M.failure && !M.active ? /* @__PURE__ */ p(Na, {
						title: "Benign-suite generation failed",
						failure: M.failure,
						workspace: e,
						jobName: M.jobName
					}) : null,
					M.active ? /* @__PURE__ */ p(ge, { children: /* @__PURE__ */ m(I, {
						gap: "density-lg",
						padding: "density-lg",
						children: [
							/* @__PURE__ */ p(L, {
								kind: "body/semibold/md",
								children: "Generating benign suite"
							}),
							/* @__PURE__ */ p(L, {
								kind: "body/regular/sm",
								className: "text-subtle",
								children: M.interview ? "Answer the interview to shape the benign test suite." : M.review ? "Review and approve the generated requests." : M.starting ? "Starting the sandbox and probing the agent…" : "Synthesizing the benign suite…"
							}),
							/* @__PURE__ */ p(La, {
								steps: M.recon,
								busy: !M.interview && !M.review,
								activity: M.activity
							}),
							M.interview ? /* @__PURE__ */ p(Ma, {
								prompt: M.interview,
								loading: M.isResponding,
								onSubmit: M.submitInterview
							}, M.interview.round) : M.review ? /* @__PURE__ */ p(Ba, {
								suite: M.review.suite,
								loading: M.isResponding,
								onSubmit: M.submitReview
							}, M.review.round) : null
						]
					}) }) : null,
					o && f ? /* @__PURE__ */ p(oc, {
						workspace: e,
						manifestName: t,
						victimImage: o.victim_image ?? "",
						saving: w.isPending,
						onSave: (n) => w.mutate({
							workspace: e,
							name: t,
							data: { victim_image: n }
						})
					}) : null,
					o ? /* @__PURE__ */ p(Os, {
						manifest: o,
						onRefresh: () => A(!0),
						refreshing: se.isPending,
						onEditEnv: N
					}) : null,
					/* @__PURE__ */ p(ge, { children: /* @__PURE__ */ m(I, {
						gap: "density-lg",
						padding: "density-lg",
						children: [
							/* @__PURE__ */ m(F, {
								className: "items-center justify-between",
								children: [/* @__PURE__ */ p(L, {
									kind: "body/semibold/md",
									children: "Benign suite (requests.csv)"
								}), /* @__PURE__ */ p(P, {
									kind: "secondary",
									size: "small",
									disabled: v.length === 0,
									onClick: () => {
										ae(new Blob([sc(v)], { type: "text/csv" }), `${t}-requests.csv`);
									},
									children: "Download"
								})]
							}),
							/* @__PURE__ */ p(L, {
								kind: "body/regular/sm",
								className: "text-subtle",
								children: "The benign requests replayed after hardening to confirm the agent still works. Edit a row inline or generate the suite — changes save automatically."
							}),
							c && !o ? /* @__PURE__ */ p(L, {
								kind: "body/regular/md",
								className: "text-subtle",
								children: "Loading…"
							}) : /* @__PURE__ */ p(ka, {
								value: v,
								onChange: (n) => {
									b(n), w.mutate({
										workspace: e,
										name: t,
										data: { benign_suite: n.map((e) => ({
											tool: e.tool,
											payload: e.payload,
											label: e.label ?? "",
											persona: e.persona ?? "",
											rationale: e.rationale ?? ""
										})) }
									});
								},
								disabled: w.isPending
							}),
							/* @__PURE__ */ p(F, { children: /* @__PURE__ */ p(P, {
								kind: "tertiary",
								color: "danger",
								disabled: v.length === 0 || D.isPending,
								onClick: () => ee(!0),
								children: "Clear benign requests"
							}) })
						]
					}) }),
					o?.benign_interview && o.benign_interview.length > 0 ? /* @__PURE__ */ p(ce, {
						multiple: !0,
						children: /* @__PURE__ */ p(g, {
							value: "interview",
							title: `Interview Q&A (${o.benign_interview.length})`,
							children: /* @__PURE__ */ m(I, {
								gap: "density-md",
								children: [/* @__PURE__ */ p(L, {
									kind: "body/regular/sm",
									className: "text-subtle",
									children: "Your answers from the last benign-suite generation — the context that shaped this suite."
								}), /* @__PURE__ */ p(sa, { interview: o.benign_interview })]
							})
						})
					}) : null
				]
			}),
			/* @__PURE__ */ p(y, {
				onNotify: r,
				open: O,
				onClose: () => ee(!1),
				title: "Clear benign requests?",
				description: "This removes every benign request from this manifest. You'll need to regenerate the suite before the next run.",
				submitButtonText: "Clear requests",
				submitButtonColor: "danger",
				successText: "Benign requests cleared.",
				errorText: "Failed to clear the benign requests.",
				onConfirm: ie
			}),
			/* @__PURE__ */ p(E, {
				open: te,
				title: "Edit Environment Variables",
				submitButtonText: "Save",
				loading: D.isPending,
				onSubmit: (e) => {
					e.preventDefault(), oe();
				},
				onClose: () => j(!1),
				children: /* @__PURE__ */ m(I, {
					gap: "density-md",
					children: [/* @__PURE__ */ p(L, {
						kind: "body/regular/sm",
						className: "text-fg-secondary",
						children: "Non-secret settings the agent reads, as comma-separated KEY=VALUE pairs. Credentials belong in the manifest's secrets — values here are stored in plain text."
					}), /* @__PURE__ */ p(pe, {
						name: "env",
						slotLabel: "Environment Variables",
						children: /* @__PURE__ */ p(je, {
							value: ne,
							onChange: (e) => re(e.target.value)
						})
					})]
				})
			}),
			/* @__PURE__ */ p(y, {
				onNotify: r,
				open: k,
				onClose: () => A(!1),
				title: `Refresh ${t}?`,
				description: "Re-resolves this manifest against the agent as it is now, so the next run attacks the current agent instead of the one saved here. Your egress, secrets, models, defenders and benign suite are kept.",
				submitButtonText: "Refresh Target",
				successText: "Target refreshed from the agent.",
				errorText: "Failed to refresh the target.",
				onConfirm: async () => (await se.mutateAsync({
					workspace: e,
					name: t
				}), i.invalidateQueries({ queryKey: yo(e, t) }), !0)
			}),
			/* @__PURE__ */ p(Ts, {
				open: x,
				onClose: () => S(!1),
				workspace: e,
				manifestName: t,
				manifest: o,
				starting: T.isPending,
				savingDefault: w.isPending,
				onStart: (n) => {
					T.mutate({
						workspace: e,
						data: { spec: {
							manifest_id: t,
							driver: "service",
							stop_after_synth: !1,
							...n
						} }
					}), S(!1);
				},
				onSaveDefault: (n) => {
					w.mutate({
						workspace: e,
						name: t,
						data: n
					});
				}
			})
		]
	});
}, lc = () => {
	let e = Ie(), t = qi(), n = Yi(), r = Ji(), i = Ve(), a = oe({ defaultSort: [{
		id: "created_at",
		desc: !0
	}] }), [o, s] = d(null), { data: c, isLoading: u } = mo(t, {
		sort: M(a.sorting.state),
		page: a.pagination.state.pageIndex + 1,
		page_size: a.pagination.state.pageSize
	}, { query: {
		placeholderData: Re,
		refetchOnMount: "always",
		retry: !1
	} }), h = () => i.invalidateQueries({ queryKey: fo(t) }), g = tc(t), _ = Do(), v = (r) => {
		if (r.name) {
			if (!r.benign_suite?.length) {
				n.error("No benign suite for this manifest yet — generate it first, then run the war-game."), e(ra(t, r.name));
				return;
			}
			g.mutate({
				workspace: t,
				data: { spec: {
					manifest_id: r.name,
					driver: "service"
				} }
			});
		}
	}, y = l(() => (c?.data ?? []).map((e) => ({
		...e,
		id: e.id || `${e.workspace ?? ""}/${e.name ?? ""}`
	})), [c]), b = c?.pagination?.total_results ?? y.length;
	return /* @__PURE__ */ m(f, { children: [
		g.failure ? /* @__PURE__ */ p(Na, {
			title: "War-game failed",
			failure: g.failure,
			workspace: t,
			jobName: g.jobName
		}) : null,
		/* @__PURE__ */ p(te, {
			dataViewState: a,
			makeColumns: ({ accessor: n }, { rowActionsColumn: r }) => [
				n("name", {
					header: "Manifest",
					cell: ({ row: e }) => e.original.name ?? "-"
				}),
				n("agent", {
					header: "Agent",
					cell: ({ row: e }) => /* @__PURE__ */ p(L, {
						className: "truncate",
						style: { maxWidth: 240 },
						kind: "body/regular/md",
						children: e.original.agent || "-"
					})
				}),
				n("created_at", {
					id: "created_at",
					header: "Created",
					enableSorting: !0,
					size: 160,
					cell: ({ row: e }) => e.original.created_at ? /* @__PURE__ */ p(k, { datetime: e.original.created_at }) : null
				}),
				r({
					size: 70,
					cell: ({ row: n }) => /* @__PURE__ */ p(O, { actions: [
						{
							label: "Run war-game",
							onSelect: () => v(n.original)
						},
						{
							label: "Edit",
							onSelect: () => n.original.name && e(ra(t, n.original.name))
						},
						{
							label: "Delete",
							onSelect: () => s(n.original)
						}
					] })
				})
			],
			onRowClick: (n) => n.name && e(ra(t, n.name)),
			attributes: {
				DataViewRoot: {
					data: y,
					totalCount: b,
					requestStatus: u && !c ? "loading" : void 0
				},
				DataViewTableContent: { renderEmptyState: () => /* @__PURE__ */ p(j, {
					header: "No manifests yet",
					emptyMessage: "Create a manifest from a registered agent, then run the war-game against it. An agent that is not on the platform — bespoke orchestration, its own image — is war-gamed with the agent-hardener CLI instead.",
					actions: /* @__PURE__ */ p(P, {
						asChild: !0,
						color: "brand",
						children: /* @__PURE__ */ p(Me, {
							to: na(t),
							children: "New Manifest"
						})
					})
				}) }
			}
		}),
		/* @__PURE__ */ p(C, {
			onNotify: r,
			open: !!o,
			onClose: () => s(null),
			title: `Delete ${o?.name ?? "manifest"}?`,
			description: "This permanently deletes the manifest and its cached benign suite.",
			successText: "Manifest deleted.",
			errorText: "Failed to delete the manifest.",
			onDelete: async () => o?.name ? (await _.mutateAsync({
				workspace: t,
				name: o.name
			}), h(), !0) : !1
		})
	] });
}, uc = () => {
	let e = qi();
	return Xi({ items: [{
		href: $i(e),
		slotLabel: "Agent Hardening"
	}, { slotLabel: "Manifests" }] }), /* @__PURE__ */ m(h, {
		title: "Agent Hardener Manifests",
		children: [/* @__PURE__ */ m(I, {
			className: "h-full overflow-auto",
			gap: "density-2xl",
			padding: "density-2xl",
			children: [/* @__PURE__ */ p(he, {
				className: "p-0",
				slotHeading: "Manifests",
				slotDescription: "Reusable war-game targets scaffolded from a deployed agent. Create one, then run the war-game against it.",
				slotActions: /* @__PURE__ */ p(P, {
					asChild: !0,
					color: "brand",
					children: /* @__PURE__ */ p(Me, {
						to: na(e),
						children: "New Manifest"
					})
				})
			}), /* @__PURE__ */ p(lc, {})]
		}), /* @__PURE__ */ p(Ne, {})]
	});
}, dc = "cancelling", fc = ({ workspace: e, jobName: t, jobStatus: n, compact: r }) => {
	let [i, a] = d(!1), o = Yi(), s = Ve(), { useJobsCancelJob: c, getJobsGetJobQueryKey: l, getJobsListJobsQueryKey: u } = Va(), { mutateAsync: h, isPending: g } = c({ mutation: { onSuccess: () => {
		o.success("Job cancelled successfully."), s.invalidateQueries({ queryKey: l(e, t) }), s.invalidateQueries({ queryKey: u(e) });
	} } }), v = async () => {
		try {
			await h({
				workspace: e,
				name: t
			}), a(!1);
		} catch (e) {
			oa(o, "Failed to cancel job")(e);
		}
	}, y = !!(n && _.includes(n)), b = n === dc;
	if (!y && !b) return null;
	let x = (e) => {
		e.stopPropagation(), a(!0);
	};
	return /* @__PURE__ */ m(f, { children: [r ? /* @__PURE__ */ m(P, {
		kind: "secondary",
		color: "danger",
		size: "small",
		onClick: x,
		disabled: g || b,
		children: [/* @__PURE__ */ p(Ta, { className: "w-3 h-3" }), "Cancel"]
	}) : /* @__PURE__ */ p(P, {
		kind: "secondary",
		onClick: x,
		disabled: g || b,
		children: g || b ? "Cancelling..." : "Cancel Job"
	}), /* @__PURE__ */ p(E, {
		open: i,
		title: `Cancel ${t}`,
		submitButtonText: "Cancel Job",
		onSubmit: (e) => {
			e.preventDefault(), v();
		},
		onClose: () => a(!1),
		disabled: g,
		loading: g,
		attributes: { SubmitButton: { color: "danger" } },
		children: /* @__PURE__ */ p(F, { children: /* @__PURE__ */ p(L, {
			className: "leading-relaxed",
			children: "Canceling this job will permanently stop it. This action cannot be undone, and the job cannot be relaunched or deleted. Are you sure you want to proceed?"
		}) })
	})] });
}, pc = /* @__PURE__ */ Ze((/* @__PURE__ */ Ye(((e, t) => {
	(function() {
		var e = {}.hasOwnProperty;
		function n() {
			for (var e = "", t = 0; t < arguments.length; t++) {
				var n = arguments[t];
				n && (e = i(e, r(n)));
			}
			return e;
		}
		function r(t) {
			if (typeof t == "string" || typeof t == "number") return t;
			if (typeof t != "object") return "";
			if (Array.isArray(t)) return n.apply(null, t);
			if (t.toString !== Object.prototype.toString && !t.toString.toString().includes("[native code]")) return t.toString();
			var r = "";
			for (var a in t) e.call(t, a) && t[a] && (r = i(r, a));
			return r;
		}
		function i(e, t) {
			return t ? e ? e + " " + t : e + t : e;
		}
		t !== void 0 && t.exports ? (n.default = n, t.exports = n) : typeof define == "function" && typeof define.amd == "object" && define.amd ? define("classnames", [], function() {
			return n;
		}) : window.classNames = n;
	})();
})))(), 1), mc = Number.isNaN || function(e) {
	return typeof e == "number" && e !== e;
};
function hc(e, t) {
	return !!(e === t || mc(e) && mc(t));
}
function gc(e, t) {
	if (e.length !== t.length) return !1;
	for (var n = 0; n < e.length; n++) if (!hc(e[n], t[n])) return !1;
	return !0;
}
function _c(e, t) {
	t === void 0 && (t = gc);
	var n = null;
	function r() {
		var r = [...arguments];
		if (n && n.lastThis === this && t(r, n.lastArgs)) return n.lastResult;
		var i = e.apply(this, r);
		return n = {
			lastResult: i,
			lastArgs: r,
			lastThis: this
		}, i;
	}
	return r.clear = function() {
		n = null;
	}, r;
}
//#endregion
//#region node_modules/.pnpm/react-diff-viewer-continued@4.4.0_@types+react@19.2.18_react-dom@19.2.8_react@19.2.8__react@19.2.8/node_modules/react-diff-viewer-continued/lib/esm/src/compute-hidden-blocks.js
function vc(e, t, n) {
	let r = 0, i, a = {}, o = [];
	return e.forEach((e, s) => {
		let c = t.some((e) => e >= s - n && e <= s + n);
		!c && i === void 0 ? (i = {
			index: r,
			startLine: s,
			endLine: s,
			lines: 1
		}, o.push(i), a[s] = i.index, r++) : !c && i ? (i.endLine = s, i.lines++, a[s] = i.index) : i = void 0;
	}), {
		lineBlocks: a,
		blocks: o
	};
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/diff/base.js
var yc = class {
	diff(e, t, n = {}) {
		let r;
		typeof n == "function" ? (r = n, n = {}) : "callback" in n && (r = n.callback);
		let i = this.castInput(e, n), a = this.castInput(t, n), o = this.removeEmpty(this.tokenize(i, n)), s = this.removeEmpty(this.tokenize(a, n));
		return this.diffWithOptionsObj(o, s, n, r);
	}
	diffWithOptionsObj(e, t, n, r) {
		let i = (e) => {
			if (e = this.postProcess(e, n), r) {
				setTimeout(function() {
					r(e);
				}, 0);
				return;
			}
			return e;
		}, a = t.length, o = e.length, s = 1, c = a + o;
		n.maxEditLength != null && (c = Math.min(c, n.maxEditLength));
		let l = n.timeout ?? Infinity, u = Date.now() + l, d = [{
			oldPos: -1,
			lastComponent: void 0
		}], f = this.extractCommon(d[0], t, e, 0, n);
		if (d[0].oldPos + 1 >= o && f + 1 >= a) return i(this.buildValues(d[0].lastComponent, t, e));
		let p = -Infinity, m = Infinity, h = () => {
			for (let r = Math.max(p, -s); r <= Math.min(m, s); r += 2) {
				let s, c = d[r - 1], l = d[r + 1];
				c && (d[r - 1] = void 0);
				let u = !1;
				if (l) {
					let e = l.oldPos - r;
					u = l && 0 <= e && e < a;
				}
				let h = c && c.oldPos + 1 < o;
				if (!u && !h) {
					d[r] = void 0;
					continue;
				}
				if (s = !h || u && c.oldPos < l.oldPos ? this.addToPath(l, !0, !1, 0, n) : this.addToPath(c, !1, !0, 1, n), f = this.extractCommon(s, t, e, r, n), s.oldPos + 1 >= o && f + 1 >= a) return i(this.buildValues(s.lastComponent, t, e)) || !0;
				d[r] = s, s.oldPos + 1 >= o && (m = Math.min(m, r - 1)), f + 1 >= a && (p = Math.max(p, r + 1));
			}
			s++;
		};
		if (r) (function e() {
			setTimeout(function() {
				if (s > c || Date.now() > u) return r(void 0);
				h() || e();
			}, 0);
		})();
		else for (; s <= c && Date.now() <= u;) {
			let e = h();
			if (e) return e;
		}
	}
	addToPath(e, t, n, r, i) {
		let a = e.lastComponent;
		return a && !i.oneChangePerToken && a.added === t && a.removed === n ? {
			oldPos: e.oldPos + r,
			lastComponent: {
				count: a.count + 1,
				added: t,
				removed: n,
				previousComponent: a.previousComponent
			}
		} : {
			oldPos: e.oldPos + r,
			lastComponent: {
				count: 1,
				added: t,
				removed: n,
				previousComponent: a
			}
		};
	}
	extractCommon(e, t, n, r, i) {
		let a = t.length, o = n.length, s = e.oldPos, c = s - r, l = 0;
		for (; c + 1 < a && s + 1 < o && this.equals(n[s + 1], t[c + 1], i);) c++, s++, l++, i.oneChangePerToken && (e.lastComponent = {
			count: 1,
			previousComponent: e.lastComponent,
			added: !1,
			removed: !1
		});
		return l && !i.oneChangePerToken && (e.lastComponent = {
			count: l,
			previousComponent: e.lastComponent,
			added: !1,
			removed: !1
		}), e.oldPos = s, c;
	}
	equals(e, t, n) {
		return n.comparator ? n.comparator(e, t) : e === t || !!n.ignoreCase && e.toLowerCase() === t.toLowerCase();
	}
	removeEmpty(e) {
		let t = [];
		for (let n = 0; n < e.length; n++) e[n] && t.push(e[n]);
		return t;
	}
	castInput(e, t) {
		return e;
	}
	tokenize(e, t) {
		return Array.from(e);
	}
	join(e) {
		return e.join("");
	}
	postProcess(e, t) {
		return e;
	}
	get useLongestToken() {
		return !1;
	}
	buildValues(e, t, n) {
		let r = [], i;
		for (; e;) r.push(e), i = e.previousComponent, delete e.previousComponent, e = i;
		r.reverse();
		let a = r.length, o = 0, s = 0, c = 0;
		for (; o < a; o++) {
			let e = r[o];
			if (e.removed) e.value = this.join(n.slice(c, c + e.count)), c += e.count;
			else {
				if (!e.added && this.useLongestToken) {
					let r = t.slice(s, s + e.count);
					r = r.map(function(e, t) {
						let r = n[c + t];
						return r.length > e.length ? r : e;
					}), e.value = this.join(r);
				} else e.value = this.join(t.slice(s, s + e.count));
				s += e.count, e.added || (c += e.count);
			}
		}
		return r;
	}
}, bc = new class extends yc {}();
function xc(e, t, n) {
	return bc.diff(e, t, n);
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/util/string.js
function Sc(e, t) {
	let n;
	for (n = 0; n < e.length && n < t.length; n++) if (e[n] != t[n]) return e.slice(0, n);
	return e.slice(0, n);
}
function Cc(e, t) {
	let n;
	if (!e || !t || e[e.length - 1] != t[t.length - 1]) return "";
	for (n = 0; n < e.length && n < t.length; n++) if (e[e.length - (n + 1)] != t[t.length - (n + 1)]) return e.slice(-n);
	return e.slice(-n);
}
function wc(e, t, n) {
	if (e.slice(0, t.length) != t) throw Error(`string ${JSON.stringify(e)} doesn't start with prefix ${JSON.stringify(t)}; this is a bug`);
	return n + e.slice(t.length);
}
function Tc(e, t, n) {
	if (!t) return e + n;
	if (e.slice(-t.length) != t) throw Error(`string ${JSON.stringify(e)} doesn't end with suffix ${JSON.stringify(t)}; this is a bug`);
	return e.slice(0, -t.length) + n;
}
function Ec(e, t) {
	return wc(e, t, "");
}
function Dc(e, t) {
	return Tc(e, t, "");
}
function Oc(e, t) {
	return t.slice(0, kc(e, t));
}
function kc(e, t) {
	let n = 0;
	e.length > t.length && (n = e.length - t.length);
	let r = t.length;
	e.length < t.length && (r = e.length);
	let i = Array(r), a = 0;
	i[0] = 0;
	for (let e = 1; e < r; e++) {
		for (t[e] == t[a] ? i[e] = i[a] : i[e] = a; a > 0 && t[e] != t[a];) a = i[a];
		t[e] == t[a] && a++;
	}
	a = 0;
	for (let r = n; r < e.length; r++) {
		for (; a > 0 && e[r] != t[a];) a = i[a];
		e[r] == t[a] && a++;
	}
	return a;
}
function Ac(e) {
	return e.includes("\r\n") && !e.startsWith("\n") && !e.match(/[^\r]\n/);
}
function jc(e) {
	return !e.includes("\r\n") && e.includes("\n");
}
function Mc(e, t) {
	let n = [];
	for (let r of Array.from(t.segment(e))) {
		let e = r.segment;
		n.length && /\s/.test(n[n.length - 1]) && /\s/.test(e) ? n[n.length - 1] += e : n.push(e);
	}
	return n;
}
function Nc(e, t) {
	if (t) return Fc(e, t)[1];
	let n;
	for (n = e.length - 1; n >= 0 && e[n].match(/\s/); n--);
	return e.substring(n + 1);
}
function Pc(e, t) {
	if (t) return Fc(e, t)[0];
	let n = e.match(/^\s*/);
	return n ? n[0] : "";
}
function Fc(e, t) {
	if (!t) return [Pc(e), Nc(e)];
	if (t.resolvedOptions().granularity != "word") throw Error("The segmenter passed must have a granularity of \"word\"");
	let n = Mc(e, t), r = n[0], i = n[n.length - 1];
	return [/\s/.test(r) ? r : "", /\s/.test(i) ? i : ""];
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/diff/word.js
var Ic = "a-zA-Z0-9_\\u{AD}\\u{C0}-\\u{D6}\\u{D8}-\\u{F6}\\u{F8}-\\u{2C6}\\u{2C8}-\\u{2D7}\\u{2DE}-\\u{2FF}\\u{1E00}-\\u{1EFF}", Lc = RegExp(`[${Ic}]+|\\s+|[^${Ic}]`, "ug"), Rc = new class extends yc {
	equals(e, t, n) {
		return n.ignoreCase && (e = e.toLowerCase(), t = t.toLowerCase()), e.trim() === t.trim();
	}
	tokenize(e, t = {}) {
		let n;
		if (t.intlSegmenter) {
			let r = t.intlSegmenter;
			if (r.resolvedOptions().granularity != "word") throw Error("The segmenter passed must have a granularity of \"word\"");
			n = Mc(e, r);
		} else n = e.match(Lc) || [];
		let r = [], i = null;
		return n.forEach((e) => {
			/\s/.test(e) ? i == null ? r.push(e) : r.push(r.pop() + e) : i != null && /\s/.test(i) ? r[r.length - 1] == i ? r.push(r.pop() + e) : r.push(i + e) : r.push(e), i = e;
		}), r;
	}
	join(e) {
		return e.map((e, t) => t == 0 ? e : e.replace(/^\s+/, "")).join("");
	}
	postProcess(e, t) {
		if (!e || t.oneChangePerToken) return e;
		let n = null, r = null, i = null;
		return e.forEach((e) => {
			e.added ? r = e : e.removed ? i = e : ((r || i) && Bc(n, i, r, e, t.intlSegmenter), n = e, r = null, i = null);
		}), (r || i) && Bc(n, i, r, null, t.intlSegmenter), e;
	}
}();
function zc(e, t, n) {
	return n?.ignoreWhitespace != null && !n.ignoreWhitespace ? Hc(e, t, n) : Rc.diff(e, t, n);
}
function Bc(e, t, n, r, i) {
	if (t && n) {
		let [a, o] = Fc(t.value, i), [s, c] = Fc(n.value, i);
		if (e) {
			let r = Sc(a, s);
			e.value = Tc(e.value, s, r), t.value = Ec(t.value, r), n.value = Ec(n.value, r);
		}
		if (r) {
			let e = Cc(o, c);
			r.value = wc(r.value, c, e), t.value = Dc(t.value, e), n.value = Dc(n.value, e);
		}
	} else if (n) {
		if (e) {
			let e = Pc(n.value, i);
			n.value = n.value.substring(e.length);
		}
		if (r) {
			let e = Pc(r.value, i);
			r.value = r.value.substring(e.length);
		}
	} else if (e && r) {
		let n = Pc(r.value, i), [a, o] = Fc(t.value, i), s = Sc(n, a);
		t.value = Ec(t.value, s);
		let c = Cc(Ec(n, s), o);
		t.value = Dc(t.value, c), r.value = wc(r.value, n, c), e.value = Tc(e.value, n, n.slice(0, n.length - c.length));
	} else if (r) {
		let e = Pc(r.value, i), n = Oc(Nc(t.value, i), e);
		t.value = Dc(t.value, n);
	} else if (e) {
		let n = Oc(Nc(e.value, i), Pc(t.value, i));
		t.value = Ec(t.value, n);
	}
}
var Vc = new class extends yc {
	tokenize(e) {
		let t = RegExp(`(\\r?\\n)|[${Ic}]+|[^\\S\\n\\r]+|[^${Ic}]`, "ug");
		return e.match(t) || [];
	}
}();
function Hc(e, t, n) {
	return Vc.diff(e, t, n);
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/util/params.js
function Uc(e, t) {
	if (typeof e == "function") t.callback = e;
	else if (e) for (let n in e)
 /* istanbul ignore else */
	Object.prototype.hasOwnProperty.call(e, n) && (t[n] = e[n]);
	return t;
}
var Wc = new class extends yc {
	constructor() {
		super(...arguments), this.tokenize = qc;
	}
	equals(e, t, n) {
		return n.ignoreWhitespace ? ((!n.newlineIsToken || !e.includes("\n")) && (e = e.trim()), (!n.newlineIsToken || !t.includes("\n")) && (t = t.trim())) : n.ignoreNewlineAtEof && !n.newlineIsToken && (e.endsWith("\n") && (e = e.slice(0, -1)), t.endsWith("\n") && (t = t.slice(0, -1))), super.equals(e, t, n);
	}
}();
function Gc(e, t, n) {
	return Wc.diff(e, t, n);
}
function Kc(e, t, n) {
	return n = Uc(n, { ignoreWhitespace: !0 }), Wc.diff(e, t, n);
}
function qc(e, t) {
	t.stripTrailingCr && (e = e.replace(/\r\n/g, "\n"));
	let n = [], r = e.split(/(\n|\r\n)/);
	r[r.length - 1] || r.pop();
	for (let e = 0; e < r.length; e++) {
		let i = r[e];
		e % 2 && !t.newlineIsToken ? n[n.length - 1] += i : n.push(i);
	}
	return n;
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/diff/sentence.js
function Jc(e) {
	return e == "." || e == "!" || e == "?";
}
var Yc = new class extends yc {
	tokenize(e) {
		let t = [], n = 0;
		for (let r = 0; r < e.length; r++) {
			if (r == e.length - 1) {
				t.push(e.slice(n));
				break;
			}
			if (Jc(e[r]) && e[r + 1].match(/\s/)) {
				for (t.push(e.slice(n, r + 1)), r = n = r + 1; e[r + 1]?.match(/\s/);) r++;
				t.push(e.slice(n, r + 1)), n = r + 1;
			}
		}
		return t;
	}
}();
function Xc(e, t, n) {
	return Yc.diff(e, t, n);
}
var Zc = new class extends yc {
	tokenize(e) {
		return e.split(/([{}:;,]|\s+)/);
	}
}();
function Qc(e, t, n) {
	return Zc.diff(e, t, n);
}
var $c = new class extends yc {
	constructor() {
		super(...arguments), this.tokenize = qc;
	}
	get useLongestToken() {
		return !0;
	}
	castInput(e, t) {
		let { undefinedReplacement: n, stringifyReplacer: r = (e, t) => t === void 0 ? n : t } = t;
		return typeof e == "string" ? e : JSON.stringify(tl(e, null, null, r), null, "  ");
	}
	equals(e, t, n) {
		return super.equals(e.replace(/,([\r\n])/g, "$1"), t.replace(/,([\r\n])/g, "$1"), n);
	}
}();
function el(e, t, n) {
	return $c.diff(e, t, n);
}
function tl(e, t, n, r, i) {
	t ||= [], n ||= [], r && (e = r(i === void 0 ? "" : i, e));
	let a;
	for (a = 0; a < t.length; a += 1) if (t[a] === e) return n[a];
	let o;
	if (Object.prototype.toString.call(e) === "[object Array]") {
		for (t.push(e), o = Array(e.length), n.push(o), a = 0; a < e.length; a += 1) o[a] = tl(e[a], t, n, r, String(a));
		return t.pop(), n.pop(), o;
	}
	if (e && e.toJSON && (e = e.toJSON()), typeof e == "object" && e) {
		t.push(e), o = {}, n.push(o);
		let i = [], s;
		for (s in e)
 /* istanbul ignore else */
		Object.prototype.hasOwnProperty.call(e, s) && i.push(s);
		for (i.sort(), a = 0; a < i.length; a += 1) s = i[a], o[s] = tl(e[s], t, n, r, s);
		t.pop(), n.pop();
	} else o = e;
	return o;
}
var nl = new class extends yc {
	tokenize(e) {
		return e.slice();
	}
	join(e) {
		return e;
	}
	removeEmpty(e) {
		return e;
	}
}();
function rl(e, t, n) {
	return nl.diff(e, t, n);
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/patch/line-endings.js
function il(e) {
	return Array.isArray(e) ? e.map((e) => il(e)) : Object.assign(Object.assign({}, e), { hunks: e.hunks.map((e) => Object.assign(Object.assign({}, e), { lines: e.lines.map((t, n) => t.startsWith("\\") || t.endsWith("\r") || e.lines[n + 1]?.startsWith("\\") ? t : t + "\r") })) });
}
function al(e) {
	return Array.isArray(e) ? e.map((e) => al(e)) : Object.assign(Object.assign({}, e), { hunks: e.hunks.map((e) => Object.assign(Object.assign({}, e), { lines: e.lines.map((e) => e.endsWith("\r") ? e.substring(0, e.length - 1) : e) })) });
}
function ol(e) {
	return Array.isArray(e) || (e = [e]), !e.some((e) => e.hunks.some((e) => e.lines.some((e) => !e.startsWith("\\") && e.endsWith("\r"))));
}
function sl(e) {
	return Array.isArray(e) || (e = [e]), e.some((e) => e.hunks.some((e) => e.lines.some((e) => e.endsWith("\r")))) && e.every((e) => e.hunks.every((e) => e.lines.every((t, n) => t.startsWith("\\") || t.endsWith("\r") || e.lines[n + 1]?.startsWith("\\"))));
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/patch/parse.js
function cl(e) {
	let t = e.split(/\n/), n = [], r = 0;
	function i(e) {
		return /^diff --git /.test(e);
	}
	function a(e) {
		return i(e) || /^Index:\s/.test(e) || /^diff(?: -r \w+)+\s/.test(e);
	}
	function o(e) {
		return /^(---|\+\+\+)\s/.test(e);
	}
	function s(e) {
		return /^@@\s/.test(e);
	}
	function c() {
		let e = {};
		e.hunks = [], n.push(e);
		let c = !1;
		for (; r < t.length;) {
			let n = t[r];
			if (o(n) || s(n)) break;
			if (i(n)) {
				if (c) return;
				c = !0, e.isGit = !0;
				let i = l(n);
				for (i && (e.oldFileName = i.oldFileName, e.newFileName = i.newFileName), r++; r < t.length;) {
					let n = t[r];
					if (o(n) || s(n) || a(n)) break;
					let i = /^rename from (.*)/.exec(n);
					i && (e.oldFileName = "a/" + u(i[1]), e.isRename = !0);
					let c = /^rename to (.*)/.exec(n);
					c && (e.newFileName = "b/" + u(c[1]), e.isRename = !0);
					let l = /^copy from (.*)/.exec(n);
					l && (e.oldFileName = "a/" + u(l[1]), e.isCopy = !0);
					let d = /^copy to (.*)/.exec(n);
					d && (e.newFileName = "b/" + u(d[1]), e.isCopy = !0);
					let f = /^new file mode (\d+)/.exec(n);
					f && (e.isCreate = !0, e.newMode = f[1]);
					let p = /^deleted file mode (\d+)/.exec(n);
					p && (e.isDelete = !0, e.oldMode = p[1]);
					let m = /^old mode (\d+)/.exec(n);
					m && (e.oldMode = m[1]);
					let h = /^new mode (\d+)/.exec(n);
					h && (e.newMode = h[1]), /^Binary files /.test(n) && (e.isBinary = !0), r++;
				}
				continue;
			}
			if (a(n)) {
				if (c) return;
				c = !0;
				let t = /^(?:Index:|diff(?: -r \w+)+)\s+/.exec(n);
				t && (e.index = n.substring(t[0].length).trim());
			}
			r++;
		}
		if (f(e), f(e), e.oldFileName === void 0 != (e.newFileName === void 0)) throw Error("Missing " + (e.oldFileName === void 0 ? "\"--- ...\"" : "\"+++ ...\"") + " file header for " + (e.oldFileName ?? e.newFileName));
		for (; r < t.length;) {
			let n = t[r];
			if (a(n) || o(n) || /^===================================================================/.test(n)) break;
			s(n) ? e.hunks.push(p()) : r++;
		}
	}
	function l(e) {
		let t = e.substring(11);
		if (t.startsWith("\"")) {
			let e = d(t);
			if (e === null) return null;
			let n = t.substring(e.rawLength + 1), r;
			if (n.startsWith("\"")) {
				let e = d(n);
				if (e === null) return null;
				r = e.fileName;
			} else r = n;
			return {
				oldFileName: e.fileName,
				newFileName: r
			};
		}
		let n = t.indexOf("\"");
		if (n > 0) {
			let e = t.substring(0, n - 1), r = d(t.substring(n));
			return r === null ? null : {
				oldFileName: e,
				newFileName: r.fileName
			};
		}
		if (t.startsWith("a/")) {
			let e = [], n = 0;
			for (; n = t.indexOf(" b/", n + 1), n !== -1;) e.push(n);
			if (e.length > 0) {
				let n = e[Math.floor(e.length / 2)];
				return {
					oldFileName: t.substring(0, n),
					newFileName: t.substring(n + 1)
				};
			}
		}
		return null;
	}
	function u(e) {
		if (e.startsWith("\"")) {
			let t = d(e);
			if (t) return t.fileName;
		}
		return e;
	}
	function d(e) {
		if (!e.startsWith("\"")) return null;
		let t = "", n = 1;
		for (; n < e.length;) {
			if (e[n] === "\"") return {
				fileName: t,
				rawLength: n + 1
			};
			if (e[n] === "\\" && n + 1 < e.length) switch (n++, e[n]) {
				case "a":
					t += "\x07";
					break;
				case "b":
					t += "\b";
					break;
				case "f":
					t += "\f";
					break;
				case "n":
					t += "\n";
					break;
				case "r":
					t += "\r";
					break;
				case "t":
					t += "	";
					break;
				case "v":
					t += "\v";
					break;
				case "\\":
					t += "\\";
					break;
				case "\"":
					t += "\"";
					break;
				case "0":
				case "1":
				case "2":
				case "3":
				case "4":
				case "5":
				case "6":
				case "7": {
					if (n + 2 >= e.length || e[n + 1] < "0" || e[n + 1] > "7" || e[n + 2] < "0" || e[n + 2] > "7") return null;
					let r = [parseInt(e.substring(n, n + 3), 8)];
					for (n += 3; e[n] === "\\" && e[n + 1] >= "0" && e[n + 1] <= "7";) {
						if (n + 3 >= e.length || e[n + 2] < "0" || e[n + 2] > "7" || e[n + 3] < "0" || e[n + 3] > "7") return null;
						r.push(parseInt(e.substring(n + 1, n + 4), 8)), n += 4;
					}
					t += new TextDecoder("utf-8").decode(new Uint8Array(r));
					continue;
				}
				default: return null;
			}
			else t += e[n];
			n++;
		}
		return null;
	}
	function f(e) {
		let n = /^(---|\+\+\+)\s+/.exec(t[r]);
		if (n) {
			let i = n[1], a = t[r].substring(3).trim().split("	", 2), o = (a[1] || "").trim(), s = a[0];
			s = s.startsWith("\"") ? u(s) : s.replace(/\\\\/g, "\\"), i === "---" ? (e.oldFileName = s, e.oldHeader = o) : (e.newFileName = s, e.newHeader = o), r++;
		}
	}
	function p() {
		let e = r, n = t[r++].split(/@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/), i = {
			oldStart: +n[1],
			oldLines: n[2] === void 0 ? 1 : +n[2],
			newStart: +n[3],
			newLines: n[4] === void 0 ? 1 : +n[4],
			lines: []
		};
		i.oldLines === 0 && (i.oldStart += 1), i.newLines === 0 && (i.newStart += 1);
		let a = 0, s = 0;
		for (; r < t.length && (s < i.oldLines || a < i.newLines || t[r]?.startsWith("\\")); r++) {
			let n = t[r].length == 0 && r != t.length - 1 ? " " : t[r][0];
			if (n === "+" || n === "-" || n === " " || n === "\\") i.lines.push(t[r]), n === "+" ? a++ : n === "-" ? s++ : n === " " && (a++, s++);
			else throw Error(`Hunk at line ${e + 1} contained invalid line ${t[r]}`);
		}
		if (!a && i.newLines === 1 && (i.newLines = 0), !s && i.oldLines === 1 && (i.oldLines = 0), a !== i.newLines) throw Error("Added line count did not match for hunk at line " + (e + 1));
		if (s !== i.oldLines) throw Error("Removed line count did not match for hunk at line " + (e + 1));
		if (r < t.length && t[r] && /^[+ -]/.test(t[r]) && !o(t[r])) throw Error("Hunk at line " + (e + 1) + " has more lines than expected (expected " + i.oldLines + " old lines and " + i.newLines + " new lines)");
		return i;
	}
	for (; r < t.length;) c();
	return n;
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/util/distance-iterator.js
function ll(e, t, n) {
	let r = !0, i = !1, a = !1, o = 1;
	return function s() {
		if (r && !a) {
			if (i ? o++ : r = !1, e + o <= n) return e + o;
			a = !0;
		}
		if (!i) return a || (r = !0), t <= e - o ? e - o++ : (i = !0, s());
	};
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/patch/apply.js
function ul(e, t, n = {}) {
	let r;
	if (r = typeof t == "string" ? cl(t) : Array.isArray(t) ? t : [t], r.length > 1) throw Error("applyPatch only works with a single input.");
	return dl(e, r[0], n);
}
function dl(e, t, n = {}) {
	(n.autoConvertLineEndings || n.autoConvertLineEndings == null) && (Ac(e) && ol(t) ? t = il(t) : jc(e) && sl(t) && (t = al(t)));
	let r = e.split("\n"), i = t.hunks, a = n.compareLine || ((e, t, n, r) => t === r), o = n.fuzzFactor || 0, s = 0;
	if (o < 0 || !Number.isInteger(o)) throw Error("fuzzFactor must be a non-negative integer");
	if (!i.length) return e;
	let c = "", l = !1, u = !1;
	for (let e = 0; e < i[i.length - 1].lines.length; e++) {
		let t = i[i.length - 1].lines[e];
		t[0] == "\\" && (c[0] == "+" ? l = !0 : c[0] == "-" && (u = !0)), c = t;
	}
	if (l) {
		if (u) {
			if (!o && r[r.length - 1] == "") return !1;
		} else if (r[r.length - 1] == "") r.pop();
		else if (!o) return !1;
	} else if (u) {
		if (r[r.length - 1] != "") r.push("");
		else if (!o) return !1;
	}
	function d(e, t, n, i = 0, o = !0, s = [], c = 0) {
		let l = 0, u = !1;
		for (; i < e.length; i++) {
			let f = e[i], p = f.length > 0 ? f[0] : " ", m = f.length > 0 ? f.substr(1) : f;
			if (p === "-") {
				if (a(t + 1, r[t], p, m)) t++, l = 0;
				else return !n || r[t] == null ? null : (s[c] = r[t], d(e, t + 1, n - 1, i, !1, s, c + 1));
			}
			if (p === "+") {
				if (!o) return null;
				s[c] = m, c++, l = 0, u = !0;
			}
			if (p === " ") {
				if (l++, s[c] = r[t], a(t + 1, r[t], p, m)) c++, o = !0, u = !1, t++;
				else return u || !n ? null : r[t] && (d(e, t + 1, n - 1, i + 1, !1, s, c + 1) || d(e, t + 1, n - 1, i, !1, s, c + 1)) || d(e, t, n - 1, i + 1, !1, s, c);
			}
		}
		return c -= l, t -= l, s.length = c, {
			patchedLines: s,
			oldLineLastI: t - 1
		};
	}
	let f = [], p = 0;
	for (let e = 0; e < i.length; e++) {
		let t = i[e], n, a = r.length - t.oldLines + o, c;
		for (let e = 0; e <= o; e++) {
			c = t.oldStart + p - 1;
			let r = ll(c, s, a);
			for (; c !== void 0 && (n = d(t.lines, c, e), !n); c = r());
			if (n) break;
		}
		if (!n) return !1;
		for (let e = s; e < c; e++) f.push(r[e]);
		for (let e = 0; e < n.patchedLines.length; e++) {
			let t = n.patchedLines[e];
			f.push(t);
		}
		s = n.oldLineLastI + 1, p = c + 1 - t.oldStart;
	}
	for (let e = s; e < r.length; e++) f.push(r[e]);
	return f.join("\n");
}
function fl(e, t) {
	let n = typeof e == "string" ? cl(e) : e, r = 0;
	function i() {
		let e = n[r++];
		if (!e) return t.complete();
		t.loadFile(e, function(n, r) {
			if (n) return t.complete(n);
			let a = ul(r, e, t);
			t.patched(e, a, function(e) {
				if (e) return t.complete(e);
				i();
			});
		});
	}
	i();
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/patch/reverse.js
function pl(e) {
	return e === void 0 || e === "/dev/null" ? e : e.startsWith("a/") ? "b/" + e.slice(2) : e.startsWith("b/") ? "a/" + e.slice(2) : e;
}
function ml(e) {
	if (Array.isArray(e)) return e.map((e) => ml(e)).reverse();
	let t = Object.assign(Object.assign({}, e), {
		oldFileName: e.isGit ? pl(e.newFileName) : e.newFileName,
		oldHeader: e.newHeader,
		newFileName: e.isGit ? pl(e.oldFileName) : e.oldFileName,
		newHeader: e.oldHeader,
		oldMode: e.newMode,
		newMode: e.oldMode,
		isCreate: e.isDelete,
		isDelete: e.isCreate,
		hunks: e.hunks.map((e) => ({
			oldLines: e.newLines,
			oldStart: e.newStart,
			newLines: e.oldLines,
			newStart: e.oldStart,
			lines: e.lines.map((e) => e.startsWith("-") ? `+${e.slice(1)}` : e.startsWith("+") ? `-${e.slice(1)}` : e)
		}))
	});
	return e.isCopy && (t.newFileName = "/dev/null", t.newHeader = void 0, t.isDelete = !0, delete t.isCreate, delete t.isCopy, delete t.isRename, t.hunks = []), t;
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/patch/create.js
function hl(e) {
	for (let t = 0; t < e.length; t++) if (e[t] < " " || e[t] > "~" || e[t] === "\"" || e[t] === "\\") return !0;
	return !1;
}
function gl(e) {
	if (!hl(e)) return e;
	let t = "\"", n = new TextEncoder().encode(e), r = 0;
	for (; r < n.length;) {
		let e = n[r];
		t += e === 7 ? "\\a" : e === 8 ? "\\b" : e === 9 ? "\\t" : e === 10 ? "\\n" : e === 11 ? "\\v" : e === 12 ? "\\f" : e === 13 ? "\\r" : e === 34 ? "\\\"" : e === 92 ? "\\\\" : e >= 32 && e <= 126 ? String.fromCharCode(e) : "\\" + e.toString(8).padStart(3, "0"), r++;
	}
	return t += "\"", t;
}
var _l = {
	includeIndex: !0,
	includeUnderline: !0,
	includeFileHeaders: !0
}, vl = {
	includeIndex: !1,
	includeUnderline: !1,
	includeFileHeaders: !0
}, yl = {
	includeIndex: !1,
	includeUnderline: !1,
	includeFileHeaders: !1
};
function bl(e, t, n, r, i, a, o) {
	let s;
	s = o ? typeof o == "function" ? { callback: o } : o : {}, s.context === void 0 && (s.context = 4);
	let c = s.context;
	if (s.newlineIsToken) throw Error("newlineIsToken may not be used with patch-generation functions, only with diffing functions");
	if (s.callback) {
		let { callback: e } = s;
		Gc(n, r, Object.assign(Object.assign({}, s), { callback: (t) => {
			let n = l(t);
			e(n);
		} }));
	} else return l(Gc(n, r, s));
	function l(n) {
		if (!n) return;
		n.push({
			value: "",
			lines: []
		});
		function r(e) {
			return e.map(function(e) {
				return " " + e;
			});
		}
		let o = [], s = 0, l = 0, u = [], d = 1, f = 1;
		for (let e = 0; e < n.length; e++) {
			let t = n[e], i = t.lines || wl(t.value);
			if (t.lines = i, t.added || t.removed) {
				if (!s) {
					let t = n[e - 1];
					s = d, l = f, t && (u = c > 0 ? r(t.lines.slice(-c)) : [], s -= u.length, l -= u.length);
				}
				for (let e of i) u.push((t.added ? "+" : "-") + e);
				t.added ? f += i.length : d += i.length;
			} else {
				if (s) {
					if (i.length <= c * 2 && e < n.length - 2) for (let e of r(i)) u.push(e);
					else {
						let e = Math.min(i.length, c);
						for (let t of r(i.slice(0, e))) u.push(t);
						let t = {
							oldStart: s,
							oldLines: d - s + e,
							newStart: l,
							newLines: f - l + e,
							lines: u
						};
						o.push(t), s = 0, l = 0, u = [];
					}
				}
				d += i.length, f += i.length;
			}
		}
		for (let e of o) for (let t = 0; t < e.lines.length; t++) e.lines[t].endsWith("\n") ? e.lines[t] = e.lines[t].slice(0, -1) : (e.lines.splice(t + 1, 0, "\\ No newline at end of file"), t++);
		return {
			oldFileName: e,
			newFileName: t,
			oldHeader: i,
			newHeader: a,
			hunks: o
		};
	}
}
function xl(e, t) {
	if (t ||= _l, Array.isArray(e)) {
		if (e.length > 1 && !t.includeFileHeaders && !e.every((e) => e.isGit)) throw Error("Cannot omit file headers on a multi-file patch. (The result would be unparseable; how would a tool trying to apply the patch know which changes are to which file?)");
		return e.map((e) => xl(e, t)).join("\n");
	}
	let n = [];
	if (e.isGit) {
		if (t = _l, !e.oldFileName) throw Error("oldFileName must be specified for Git patches");
		if (!e.newFileName) throw Error("newFileName must be specified for Git patches");
		let r = e.oldFileName, i = e.newFileName;
		e.isCreate && r === "/dev/null" ? r = i.replace(/^b\//, "a/") : e.isDelete && i === "/dev/null" && (i = r.replace(/^a\//, "b/")), n.push("diff --git " + gl(r) + " " + gl(i)), e.isDelete && n.push("deleted file mode " + (e.oldMode ?? "100644")), e.isCreate && n.push("new file mode " + (e.newMode ?? "100644")), e.oldMode && e.newMode && !e.isDelete && !e.isCreate && (n.push("old mode " + e.oldMode), n.push("new mode " + e.newMode)), e.isRename && (n.push("rename from " + gl((e.oldFileName ?? "").replace(/^a\//, ""))), n.push("rename to " + gl((e.newFileName ?? "").replace(/^b\//, "")))), e.isCopy && (n.push("copy from " + gl((e.oldFileName ?? "").replace(/^a\//, ""))), n.push("copy to " + gl((e.newFileName ?? "").replace(/^b\//, ""))));
	} else t.includeIndex && e.oldFileName == e.newFileName && e.oldFileName !== void 0 && n.push("Index: " + e.oldFileName), t.includeUnderline && n.push("===================================================================");
	let r = e.hunks.length > 0;
	t.includeFileHeaders && e.oldFileName !== void 0 && e.newFileName !== void 0 && (!e.isGit || r) && (n.push("--- " + gl(e.oldFileName) + (e.oldHeader ? "	" + e.oldHeader : "")), n.push("+++ " + gl(e.newFileName) + (e.newHeader ? "	" + e.newHeader : "")));
	for (let t = 0; t < e.hunks.length; t++) {
		let r = e.hunks[t], i = r.oldLines === 0 ? r.oldStart - 1 : r.oldStart, a = r.newLines === 0 ? r.newStart - 1 : r.newStart;
		n.push("@@ -" + i + "," + r.oldLines + " +" + a + "," + r.newLines + " @@");
		for (let e of r.lines) n.push(e);
	}
	return n.join("\n") + "\n";
}
function Sl(e, t, n, r, i, a, o) {
	if (typeof o == "function" && (o = { callback: o }), o?.callback) {
		let { callback: s } = o;
		bl(e, t, n, r, i, a, Object.assign(Object.assign({}, o), { callback: (e) => {
			s(e ? xl(e, o.headerOptions) : void 0);
		} }));
	} else {
		let s = bl(e, t, n, r, i, a, o);
		return s ? xl(s, o?.headerOptions) : void 0;
	}
}
function Cl(e, t, n, r, i, a) {
	return Sl(e, e, t, n, r, i, a);
}
function wl(e) {
	let t = e.endsWith("\n"), n = e.split("\n").map((e) => e + "\n");
	return t ? n.pop() : n.push(n.pop().slice(0, -1)), n;
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/convert/dmp.js
function Tl(e) {
	let t = [], n, r;
	for (let i = 0; i < e.length; i++) n = e[i], r = n.added ? 1 : n.removed ? -1 : 0, t.push([r, n.value]);
	return t;
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/convert/xml.js
function El(e) {
	let t = [];
	for (let n = 0; n < e.length; n++) {
		let r = e[n];
		r.added ? t.push("<ins>") : r.removed && t.push("<del>"), t.push(Dl(r.value)), r.added ? t.push("</ins>") : r.removed && t.push("</del>");
	}
	return t.join("");
}
function Dl(e) {
	let t = e;
	return t = t.replace(/&/g, "&amp;"), t = t.replace(/</g, "&lt;"), t = t.replace(/>/g, "&gt;"), t = t.replace(/"/g, "&quot;"), t;
}
//#endregion
//#region node_modules/.pnpm/diff@9.0.0/node_modules/diff/libesm/index.js
var Ol = /* @__PURE__ */ z({
	Diff: () => yc,
	FILE_HEADERS_ONLY: () => vl,
	INCLUDE_HEADERS: () => _l,
	OMIT_HEADERS: () => yl,
	applyPatch: () => ul,
	applyPatches: () => fl,
	arrayDiff: () => nl,
	canonicalize: () => tl,
	characterDiff: () => bc,
	convertChangesToDMP: () => Tl,
	convertChangesToXML: () => El,
	createPatch: () => Cl,
	createTwoFilesPatch: () => Sl,
	cssDiff: () => Zc,
	diffArrays: () => rl,
	diffChars: () => xc,
	diffCss: () => Qc,
	diffJson: () => el,
	diffLines: () => Gc,
	diffSentences: () => Xc,
	diffTrimmedLines: () => Kc,
	diffWords: () => zc,
	diffWordsWithSpace: () => Hc,
	formatPatch: () => xl,
	jsonDiff: () => $c,
	lineDiff: () => Wc,
	parsePatch: () => cl,
	reversePatch: () => ml,
	sentenceDiff: () => Yc,
	structuredPatch: () => bl,
	wordDiff: () => Rc,
	wordsWithSpaceDiff: () => Vc
});
//#endregion
//#region node_modules/.pnpm/js-yaml@4.3.1/node_modules/js-yaml/dist/js-yaml.mjs
function kl(e) {
	return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, "default") ? e.default : e;
}
var Al = {}, jl = {}, Ml = {}, Nl;
function Pl() {
	if (Nl) return Ml;
	Nl = 1;
	function e(e) {
		return e == null;
	}
	function t(e) {
		return typeof e == "object" && !!e;
	}
	function n(t) {
		return Array.isArray(t) ? t : e(t) ? [] : [t];
	}
	function r(e, t) {
		if (t) {
			let n = Object.keys(t);
			for (let r = 0, i = n.length; r < i; r += 1) {
				let i = n[r];
				e[i] = t[i];
			}
		}
		return e;
	}
	function i(e, t) {
		let n = "";
		for (let r = 0; r < t; r += 1) n += e;
		return n;
	}
	function a(e) {
		return e === 0 && 1 / e == -Infinity;
	}
	return Ml.isNothing = e, Ml.isObject = t, Ml.toArray = n, Ml.repeat = i, Ml.isNegativeZero = a, Ml.extend = r, Ml;
}
var Fl, Il;
function Ll() {
	if (Il) return Fl;
	Il = 1;
	function e(e, t) {
		let n = "", r = e.reason || "(unknown reason)";
		return e.mark ? (e.mark.name && (n += "in \"" + e.mark.name + "\" "), n += "(" + (e.mark.line + 1) + ":" + (e.mark.column + 1) + ")", !t && e.mark.snippet && (n += "\n\n" + e.mark.snippet), r + " " + n) : r;
	}
	function t(t, n) {
		Error.call(this), this.name = "YAMLException", this.reason = t, this.mark = n, this.message = e(this, !1), Error.captureStackTrace ? Error.captureStackTrace(this, this.constructor) : this.stack = (/* @__PURE__ */ Error()).stack || "";
	}
	return t.prototype = Object.create(Error.prototype), t.prototype.constructor = t, t.prototype.toString = function(t) {
		return this.name + ": " + e(this, t);
	}, Fl = t, Fl;
}
var Rl, zl;
function Bl() {
	if (zl) return Rl;
	zl = 1;
	let e = Pl();
	function t(e, t, n, r, i) {
		let a = "", o = "", s = Math.floor(i / 2) - 1;
		return r - t > s && (a = " ... ", t = r - s + a.length), n - r > s && (o = " ...", n = r + s - o.length), {
			str: a + e.slice(t, n).replace(/\t/g, "→") + o,
			pos: r - t + a.length
		};
	}
	function n(t, n) {
		return e.repeat(" ", n - t.length) + t;
	}
	function r(r, i) {
		if (i = Object.create(i || null), !r.buffer) return null;
		i.maxLength || (i.maxLength = 79), typeof i.indent != "number" && (i.indent = 1), typeof i.linesBefore != "number" && (i.linesBefore = 3), typeof i.linesAfter != "number" && (i.linesAfter = 2);
		let a = /\r?\n|\r|\0/g, o = [0], s = [], c, l = -1;
		for (; c = a.exec(r.buffer);) s.push(c.index), o.push(c.index + c[0].length), r.position <= c.index && l < 0 && (l = o.length - 2);
		l < 0 && (l = o.length - 1);
		let u = "", d = Math.min(r.line + i.linesAfter, s.length).toString().length, f = i.maxLength - (i.indent + d + 3);
		for (let a = 1; a <= i.linesBefore && !(l - a < 0); a++) {
			let c = t(r.buffer, o[l - a], s[l - a], r.position - (o[l] - o[l - a]), f);
			u = e.repeat(" ", i.indent) + n((r.line - a + 1).toString(), d) + " | " + c.str + "\n" + u;
		}
		let p = t(r.buffer, o[l], s[l], r.position, f);
		u += e.repeat(" ", i.indent) + n((r.line + 1).toString(), d) + " | " + p.str + "\n", u += e.repeat("-", i.indent + d + 3 + p.pos) + "^\n";
		for (let a = 1; a <= i.linesAfter && !(l + a >= s.length); a++) {
			let c = t(r.buffer, o[l + a], s[l + a], r.position - (o[l] - o[l + a]), f);
			u += e.repeat(" ", i.indent) + n((r.line + a + 1).toString(), d) + " | " + c.str + "\n";
		}
		return u.replace(/\n$/, "");
	}
	return Rl = r, Rl;
}
var Vl, Hl;
function Ul() {
	if (Hl) return Vl;
	Hl = 1;
	let e = Ll(), t = [
		"kind",
		"multi",
		"resolve",
		"construct",
		"instanceOf",
		"predicate",
		"represent",
		"representName",
		"defaultStyle",
		"styleAliases"
	], n = [
		"scalar",
		"sequence",
		"mapping"
	];
	function r(e) {
		let t = {};
		return e !== null && Object.keys(e).forEach(function(n) {
			e[n].forEach(function(e) {
				t[String(e)] = n;
			});
		}), t;
	}
	function i(i, a) {
		if (a ||= {}, Object.keys(a).forEach(function(n) {
			if (t.indexOf(n) === -1) throw new e("Unknown option \"" + n + "\" is met in definition of \"" + i + "\" YAML type.");
		}), this.options = a, this.tag = i, this.kind = a.kind || null, this.resolve = a.resolve || function() {
			return !0;
		}, this.construct = a.construct || function(e) {
			return e;
		}, this.instanceOf = a.instanceOf || null, this.predicate = a.predicate || null, this.represent = a.represent || null, this.representName = a.representName || null, this.defaultStyle = a.defaultStyle || null, this.multi = a.multi || !1, this.styleAliases = r(a.styleAliases || null), n.indexOf(this.kind) === -1) throw new e("Unknown kind \"" + this.kind + "\" is specified for \"" + i + "\" YAML type.");
	}
	return Vl = i, Vl;
}
var Wl, Gl;
function Kl() {
	if (Gl) return Wl;
	Gl = 1;
	let e = Ll(), t = Ul();
	function n(e, t) {
		let n = [];
		return e[t].forEach(function(e) {
			let t = n.length;
			n.forEach(function(n, r) {
				n.tag === e.tag && n.kind === e.kind && n.multi === e.multi && (t = r);
			}), n[t] = e;
		}), n;
	}
	function r() {
		let e = {
			scalar: {},
			sequence: {},
			mapping: {},
			fallback: {},
			multi: {
				scalar: [],
				sequence: [],
				mapping: [],
				fallback: []
			}
		};
		function t(t) {
			t.multi ? (e.multi[t.kind].push(t), e.multi.fallback.push(t)) : e[t.kind][t.tag] = e.fallback[t.tag] = t;
		}
		for (let e = 0, n = arguments.length; e < n; e += 1) arguments[e].forEach(t);
		return e;
	}
	function i(e) {
		return this.extend(e);
	}
	return i.prototype.extend = function(a) {
		let o = [], s = [];
		if (a instanceof t) s.push(a);
		else if (Array.isArray(a)) s = s.concat(a);
		else if (a && (Array.isArray(a.implicit) || Array.isArray(a.explicit))) a.implicit && (o = o.concat(a.implicit)), a.explicit && (s = s.concat(a.explicit));
		else throw new e("Schema.extend argument should be a Type, [ Type ], or a schema definition ({ implicit: [...], explicit: [...] })");
		o.forEach(function(n) {
			if (!(n instanceof t)) throw new e("Specified list of YAML types (or a single Type object) contains a non-Type object.");
			if (n.loadKind && n.loadKind !== "scalar") throw new e("There is a non-scalar type in the implicit list of a schema. Implicit resolving of such types is not supported.");
			if (n.multi) throw new e("There is a multi type in the implicit list of a schema. Multi tags can only be listed as explicit.");
		}), s.forEach(function(n) {
			if (!(n instanceof t)) throw new e("Specified list of YAML types (or a single Type object) contains a non-Type object.");
		});
		let c = Object.create(i.prototype);
		return c.implicit = (this.implicit || []).concat(o), c.explicit = (this.explicit || []).concat(s), c.compiledImplicit = n(c, "implicit"), c.compiledExplicit = n(c, "explicit"), c.compiledTypeMap = r(c.compiledImplicit, c.compiledExplicit), c;
	}, Wl = i, Wl;
}
var ql, Jl;
function Yl() {
	return Jl ? ql : (Jl = 1, ql = new (Ul())("tag:yaml.org,2002:str", {
		kind: "scalar",
		construct: function(e) {
			return e === null ? "" : e;
		}
	}), ql);
}
var Xl, Zl;
function Ql() {
	return Zl ? Xl : (Zl = 1, Xl = new (Ul())("tag:yaml.org,2002:seq", {
		kind: "sequence",
		construct: function(e) {
			return e === null ? [] : e;
		}
	}), Xl);
}
var $l, eu;
function tu() {
	return eu ? $l : (eu = 1, $l = new (Ul())("tag:yaml.org,2002:map", {
		kind: "mapping",
		construct: function(e) {
			return e === null ? {} : e;
		}
	}), $l);
}
var nu, ru;
function iu() {
	return ru ? nu : (ru = 1, nu = new (Kl())({ explicit: [
		Yl(),
		Ql(),
		tu()
	] }), nu);
}
var au, ou;
function su() {
	if (ou) return au;
	ou = 1;
	let e = Ul();
	function t(e) {
		if (e === null) return !0;
		let t = e.length;
		return t === 1 && e === "~" || t === 4 && (e === "null" || e === "Null" || e === "NULL");
	}
	function n() {
		return null;
	}
	function r(e) {
		return e === null;
	}
	return au = new e("tag:yaml.org,2002:null", {
		kind: "scalar",
		resolve: t,
		construct: n,
		predicate: r,
		represent: {
			canonical: function() {
				return "~";
			},
			lowercase: function() {
				return "null";
			},
			uppercase: function() {
				return "NULL";
			},
			camelcase: function() {
				return "Null";
			},
			empty: function() {
				return "";
			}
		},
		defaultStyle: "lowercase"
	}), au;
}
var cu, lu;
function uu() {
	if (lu) return cu;
	lu = 1;
	let e = Ul();
	function t(e) {
		if (e === null) return !1;
		let t = e.length;
		return t === 4 && (e === "true" || e === "True" || e === "TRUE") || t === 5 && (e === "false" || e === "False" || e === "FALSE");
	}
	function n(e) {
		return e === "true" || e === "True" || e === "TRUE";
	}
	function r(e) {
		return Object.prototype.toString.call(e) === "[object Boolean]";
	}
	return cu = new e("tag:yaml.org,2002:bool", {
		kind: "scalar",
		resolve: t,
		construct: n,
		predicate: r,
		represent: {
			lowercase: function(e) {
				return e ? "true" : "false";
			},
			uppercase: function(e) {
				return e ? "TRUE" : "FALSE";
			},
			camelcase: function(e) {
				return e ? "True" : "False";
			}
		},
		defaultStyle: "lowercase"
	}), cu;
}
var du, fu;
function pu() {
	if (fu) return du;
	fu = 1;
	let e = Pl(), t = Ul();
	function n(e) {
		return e >= 48 && e <= 57 || e >= 65 && e <= 70 || e >= 97 && e <= 102;
	}
	function r(e) {
		return e >= 48 && e <= 55;
	}
	function i(e) {
		return e >= 48 && e <= 57;
	}
	function a(e) {
		if (e === null) return !1;
		let t = e.length, a = 0, s = !1;
		if (!t) return !1;
		let c = e[a];
		if ((c === "-" || c === "+") && (c = e[++a]), c === "0") {
			if (a + 1 === t) return !0;
			if (c = e[++a], c === "b") {
				for (a++; a < t; a++) {
					if (c = e[a], c !== "0" && c !== "1") return !1;
					s = !0;
				}
				return s && isFinite(o(e));
			}
			if (c === "x") {
				for (a++; a < t; a++) {
					if (!n(e.charCodeAt(a))) return !1;
					s = !0;
				}
				return s && isFinite(o(e));
			}
			if (c === "o") {
				for (a++; a < t; a++) {
					if (!r(e.charCodeAt(a))) return !1;
					s = !0;
				}
				return s && isFinite(o(e));
			}
		}
		for (; a < t; a++) {
			if (!i(e.charCodeAt(a))) return !1;
			s = !0;
		}
		return s ? isFinite(o(e)) : !1;
	}
	function o(e) {
		let t = e, n = 1, r = t[0];
		if ((r === "-" || r === "+") && (r === "-" && (n = -1), t = t.slice(1), r = t[0]), t === "0") return 0;
		if (r === "0") {
			if (t[1] === "b") return n * parseInt(t.slice(2), 2);
			if (t[1] === "x") return n * parseInt(t.slice(2), 16);
			if (t[1] === "o") return n * parseInt(t.slice(2), 8);
		}
		return n * parseInt(t, 10);
	}
	function s(e) {
		return o(e);
	}
	function c(t) {
		return Object.prototype.toString.call(t) === "[object Number]" && t % 1 == 0 && !e.isNegativeZero(t);
	}
	return du = new t("tag:yaml.org,2002:int", {
		kind: "scalar",
		resolve: a,
		construct: s,
		predicate: c,
		represent: {
			binary: function(e) {
				return e >= 0 ? "0b" + e.toString(2) : "-0b" + e.toString(2).slice(1);
			},
			octal: function(e) {
				return e >= 0 ? "0o" + e.toString(8) : "-0o" + e.toString(8).slice(1);
			},
			decimal: function(e) {
				return e.toString(10);
			},
			hexadecimal: function(e) {
				return e >= 0 ? "0x" + e.toString(16).toUpperCase() : "-0x" + e.toString(16).toUpperCase().slice(1);
			}
		},
		defaultStyle: "decimal",
		styleAliases: {
			binary: [2, "bin"],
			octal: [8, "oct"],
			decimal: [10, "dec"],
			hexadecimal: [16, "hex"]
		}
	}), du;
}
var mu, hu;
function gu() {
	if (hu) return mu;
	hu = 1;
	let e = Pl(), t = Ul(), n = /* @__PURE__ */ RegExp("^(?:[-+]?(?:[0-9]+)(?:\\.[0-9]*)?(?:[eE][-+]?[0-9]+)?|\\.[0-9]+(?:[eE][-+]?[0-9]+)?|[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN))$"), r = /* @__PURE__ */ RegExp("^(?:[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN))$");
	function i(e) {
		return e === null || !n.test(e) ? !1 : isFinite(parseFloat(e, 10)) ? !0 : r.test(e);
	}
	function a(e) {
		let t = e.toLowerCase(), n = t[0] === "-" ? -1 : 1;
		return "+-".indexOf(t[0]) >= 0 && (t = t.slice(1)), t === ".inf" ? n === 1 ? Infinity : -Infinity : t === ".nan" ? NaN : n * parseFloat(t, 10);
	}
	let o = /^[-+]?[0-9]+e/;
	function s(t, n) {
		if (isNaN(t)) switch (n) {
			case "lowercase": return ".nan";
			case "uppercase": return ".NAN";
			case "camelcase": return ".NaN";
		}
		else if (t === Infinity) switch (n) {
			case "lowercase": return ".inf";
			case "uppercase": return ".INF";
			case "camelcase": return ".Inf";
		}
		else if (t === -Infinity) switch (n) {
			case "lowercase": return "-.inf";
			case "uppercase": return "-.INF";
			case "camelcase": return "-.Inf";
		}
		else if (e.isNegativeZero(t)) return "-0.0";
		let r = t.toString(10);
		return o.test(r) ? r.replace("e", ".e") : r;
	}
	function c(t) {
		return Object.prototype.toString.call(t) === "[object Number]" && (t % 1 != 0 || e.isNegativeZero(t));
	}
	return mu = new t("tag:yaml.org,2002:float", {
		kind: "scalar",
		resolve: i,
		construct: a,
		predicate: c,
		represent: s,
		defaultStyle: "lowercase"
	}), mu;
}
var _u, vu;
function yu() {
	return vu ? _u : (vu = 1, _u = iu().extend({ implicit: [
		su(),
		uu(),
		pu(),
		gu()
	] }), _u);
}
var bu, xu;
function Su() {
	return xu ? bu : (xu = 1, bu = yu(), bu);
}
var Cu, wu;
function Tu() {
	if (wu) return Cu;
	wu = 1;
	let e = Ul(), t = /* @__PURE__ */ RegExp("^([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])$"), n = /* @__PURE__ */ RegExp("^([0-9][0-9][0-9][0-9])-([0-9][0-9]?)-([0-9][0-9]?)(?:[Tt]|[ \\t]+)([0-9][0-9]?):([0-9][0-9]):([0-9][0-9])(?:\\.([0-9]*))?(?:[ \\t]*(Z|([-+])([0-9][0-9]?)(?::([0-9][0-9]))?))?$");
	function r(e) {
		return e === null ? !1 : t.exec(e) !== null || n.exec(e) !== null;
	}
	function i(e) {
		let r = 0, i = null, a = t.exec(e);
		if (a === null && (a = n.exec(e)), a === null) throw Error("Date resolve error");
		let o = +a[1], s = a[2] - 1, c = +a[3];
		if (!a[4]) return new Date(Date.UTC(o, s, c));
		let l = +a[4], u = +a[5], d = +a[6];
		if (a[7]) {
			for (r = a[7].slice(0, 3); r.length < 3;) r += "0";
			r = +r;
		}
		if (a[9]) {
			let e = +a[10], t = +(a[11] || 0);
			i = (e * 60 + t) * 6e4, a[9] === "-" && (i = -i);
		}
		let f = new Date(Date.UTC(o, s, c, l, u, d, r));
		return i && f.setTime(f.getTime() - i), f;
	}
	function a(e) {
		return e.toISOString();
	}
	return Cu = new e("tag:yaml.org,2002:timestamp", {
		kind: "scalar",
		resolve: r,
		construct: i,
		instanceOf: Date,
		represent: a
	}), Cu;
}
var Eu, Du;
function Ou() {
	if (Du) return Eu;
	Du = 1;
	let e = Ul();
	function t(e) {
		return e === "<<" || e === null;
	}
	return Eu = new e("tag:yaml.org,2002:merge", {
		kind: "scalar",
		resolve: t
	}), Eu;
}
var ku, Au;
function ju() {
	if (Au) return ku;
	Au = 1;
	let e = Ul();
	function t(e) {
		if (e === null) return !1;
		let t = 0, n = e.length;
		for (let r = 0; r < n; r++) {
			let n = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r".indexOf(e.charAt(r));
			if (!(n > 64)) {
				if (n < 0) return !1;
				t += 6;
			}
		}
		return t % 8 == 0;
	}
	function n(e) {
		let t = e.replace(/[\r\n=]/g, ""), n = t.length, r = 0, i = [];
		for (let e = 0; e < n; e++) e % 4 == 0 && e && (i.push(r >> 16 & 255), i.push(r >> 8 & 255), i.push(r & 255)), r = r << 6 | "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r".indexOf(t.charAt(e));
		let a = n % 4 * 6;
		return a === 0 ? (i.push(r >> 16 & 255), i.push(r >> 8 & 255), i.push(r & 255)) : a === 18 ? (i.push(r >> 10 & 255), i.push(r >> 2 & 255)) : a === 12 && i.push(r >> 4 & 255), new Uint8Array(i);
	}
	function r(e) {
		let t = "", n = 0, r = e.length, i = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r";
		for (let a = 0; a < r; a++) a % 3 == 0 && a && (t += i[n >> 18 & 63], t += i[n >> 12 & 63], t += i[n >> 6 & 63], t += i[n & 63]), n = (n << 8) + e[a];
		let a = r % 3;
		return a === 0 ? (t += i[n >> 18 & 63], t += i[n >> 12 & 63], t += i[n >> 6 & 63], t += i[n & 63]) : a === 2 ? (t += i[n >> 10 & 63], t += i[n >> 4 & 63], t += i[n << 2 & 63], t += i[64]) : a === 1 && (t += i[n >> 2 & 63], t += i[n << 4 & 63], t += i[64], t += i[64]), t;
	}
	function i(e) {
		return Object.prototype.toString.call(e) === "[object Uint8Array]";
	}
	return ku = new e("tag:yaml.org,2002:binary", {
		kind: "scalar",
		resolve: t,
		construct: n,
		predicate: i,
		represent: r
	}), ku;
}
var Mu, Nu;
function Pu() {
	if (Nu) return Mu;
	Nu = 1;
	let e = Ul(), t = Object.prototype.hasOwnProperty, n = Object.prototype.toString;
	function r(e) {
		if (e === null) return !0;
		let r = {}, i = e;
		for (let e = 0, a = i.length; e < a; e += 1) {
			let a = i[e], o = !1;
			if (n.call(a) !== "[object Object]") return !1;
			let s;
			for (s in a) if (t.call(a, s)) {
				if (!o) o = !0;
				else return !1;
			}
			if (!o || t.call(r, s)) return !1;
			Object.defineProperty(r, s, { value: !0 });
		}
		return !0;
	}
	function i(e) {
		return e === null ? [] : e;
	}
	return Mu = new e("tag:yaml.org,2002:omap", {
		kind: "sequence",
		resolve: r,
		construct: i
	}), Mu;
}
var Fu, Iu;
function Lu() {
	if (Iu) return Fu;
	Iu = 1;
	let e = Ul(), t = Object.prototype.toString;
	function n(e) {
		if (e === null) return !0;
		let n = e, r = Array(n.length);
		for (let e = 0, i = n.length; e < i; e += 1) {
			let i = n[e];
			if (t.call(i) !== "[object Object]") return !1;
			let a = Object.keys(i);
			if (a.length !== 1) return !1;
			r[e] = [a[0], i[a[0]]];
		}
		return !0;
	}
	function r(e) {
		if (e === null) return [];
		let t = e, n = Array(t.length);
		for (let e = 0, r = t.length; e < r; e += 1) {
			let r = t[e], i = Object.keys(r);
			n[e] = [i[0], r[i[0]]];
		}
		return n;
	}
	return Fu = new e("tag:yaml.org,2002:pairs", {
		kind: "sequence",
		resolve: n,
		construct: r
	}), Fu;
}
var Ru, zu;
function Bu() {
	if (zu) return Ru;
	zu = 1;
	let e = Ul(), t = Object.prototype.hasOwnProperty;
	function n(e) {
		if (e === null) return !0;
		let n = e;
		for (let e in n) if (t.call(n, e) && n[e] !== null) return !1;
		return !0;
	}
	function r(e) {
		return e === null ? {} : e;
	}
	return Ru = new e("tag:yaml.org,2002:set", {
		kind: "mapping",
		resolve: n,
		construct: r
	}), Ru;
}
var Vu, Hu;
function Uu() {
	return Hu ? Vu : (Hu = 1, Vu = Su().extend({
		implicit: [Tu(), Ou()],
		explicit: [
			ju(),
			Pu(),
			Lu(),
			Bu()
		]
	}), Vu);
}
var Wu;
function Gu() {
	if (Wu) return jl;
	Wu = 1;
	let e = Pl(), t = Ll(), n = Bl(), r = Uu(), i = Object.prototype.hasOwnProperty, a = /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F\uFFFE\uFFFF]|[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?:[^\uD800-\uDBFF]|^)[\uDC00-\uDFFF]/, o = /[\x85\u2028\u2029]/, s = /[,\[\]{}]/, c = /^(?:!|!!|![0-9A-Za-z-]+!)$/, l = /^(?:!|[^,\[\]{}])(?:%[0-9a-f]{2}|[0-9a-z\-#;/?:@&=+$,_.!~*'()\[\]])*$/i;
	function u(e) {
		return Object.prototype.toString.call(e);
	}
	function d(e) {
		return e === 10 || e === 13;
	}
	function f(e) {
		return e === 9 || e === 32;
	}
	function p(e) {
		return e === 9 || e === 32 || e === 10 || e === 13;
	}
	function m(e) {
		return e === 44 || e === 91 || e === 93 || e === 123 || e === 125;
	}
	function h(e) {
		if (e >= 48 && e <= 57) return e - 48;
		let t = e | 32;
		return t >= 97 && t <= 102 ? t - 97 + 10 : -1;
	}
	function g(e) {
		return e === 120 ? 2 : e === 117 ? 4 : e === 85 ? 8 : 0;
	}
	function _(e) {
		return e >= 48 && e <= 57 ? e - 48 : -1;
	}
	function v(e) {
		switch (e) {
			case 48: return "\0";
			case 97: return "\x07";
			case 98: return "\b";
			case 116: return "	";
			case 9: return "	";
			case 110: return "\n";
			case 118: return "\v";
			case 102: return "\f";
			case 114: return "\r";
			case 101: return "\x1B";
			case 32: return " ";
			case 34: return "\"";
			case 47: return "/";
			case 92: return "\\";
			case 78: return "";
			case 95: return "\xA0";
			case 76: return "\u2028";
			case 80: return "\u2029";
			default: return "";
		}
	}
	function y(e) {
		return e <= 65535 ? String.fromCharCode(e) : String.fromCharCode((e - 65536 >> 10) + 55296, (e - 65536 & 1023) + 56320);
	}
	function b(e, t, n) {
		t === "__proto__" ? Object.defineProperty(e, t, {
			configurable: !0,
			enumerable: !0,
			writable: !0,
			value: n
		}) : e[t] = n;
	}
	let x = Array(256), S = Array(256);
	for (let e = 0; e < 256; e++) x[e] = +!!v(e), S[e] = v(e);
	function C(e, t) {
		this.input = e, this.filename = t.filename || null, this.schema = t.schema || r, this.onWarning = t.onWarning || null, this.legacy = t.legacy || !1, this.json = t.json || !1, this.listener = t.listener || null, this.maxDepth = typeof t.maxDepth == "number" ? t.maxDepth : 100, this.maxTotalMergeKeys = typeof t.maxTotalMergeKeys == "number" ? t.maxTotalMergeKeys : 1e4, this.implicitTypes = this.schema.compiledImplicit, this.typeMap = this.schema.compiledTypeMap, this.length = e.length, this.position = 0, this.line = 0, this.lineStart = 0, this.lineIndent = 0, this.depth = 0, this.totalMergeKeys = 0, this.firstTabInLine = -1, this.documents = [], this.anchorMapTransactions = [];
	}
	function w(e, r) {
		let i = {
			name: e.filename,
			buffer: e.input.slice(0, -1),
			position: e.position,
			line: e.line,
			column: e.position - e.lineStart
		};
		return i.snippet = n(i), new t(r, i);
	}
	function T(e, t) {
		throw w(e, t);
	}
	function E(e, t) {
		e.onWarning && e.onWarning.call(null, w(e, t));
	}
	function D(e, t, n) {
		let r = e.anchorMapTransactions;
		if (r.length !== 0) {
			let n = r[r.length - 1];
			i.call(n, t) || (n[t] = {
				existed: i.call(e.anchorMap, t),
				value: e.anchorMap[t]
			});
		}
		e.anchorMap[t] = n;
	}
	function O(e) {
		e.anchorMapTransactions.push(/* @__PURE__ */ Object.create(null));
	}
	function ee(e) {
		let t = e.anchorMapTransactions.pop(), n = e.anchorMapTransactions;
		if (n.length === 0) return;
		let r = n[n.length - 1], a = Object.keys(t);
		for (let e = 0, n = a.length; e < n; e += 1) {
			let n = a[e];
			i.call(r, n) || (r[n] = t[n]);
		}
	}
	function k(e) {
		let t = e.anchorMapTransactions.pop(), n = Object.keys(t);
		for (let r = n.length - 1; r >= 0; --r) {
			let i = t[n[r]];
			i.existed ? e.anchorMap[n[r]] = i.value : delete e.anchorMap[n[r]];
		}
	}
	function A(e) {
		return {
			position: e.position,
			line: e.line,
			lineStart: e.lineStart,
			lineIndent: e.lineIndent,
			firstTabInLine: e.firstTabInLine,
			tag: e.tag,
			anchor: e.anchor,
			kind: e.kind,
			result: e.result
		};
	}
	function te(e, t) {
		e.position = t.position, e.line = t.line, e.lineStart = t.lineStart, e.lineIndent = t.lineIndent, e.firstTabInLine = t.firstTabInLine, e.tag = t.tag, e.anchor = t.anchor, e.kind = t.kind, e.result = t.result;
	}
	let j = {
		YAML: function(e, t, n) {
			e.version !== null && T(e, "duplication of %YAML directive"), n.length !== 1 && T(e, "YAML directive accepts exactly one argument");
			let r = /^([0-9]+)\.([0-9]+)$/.exec(n[0]);
			r === null && T(e, "ill-formed argument of the YAML directive");
			let i = parseInt(r[1], 10), a = parseInt(r[2], 10);
			i !== 1 && T(e, "unacceptable YAML version of the document"), e.version = n[0], e.checkLineBreaks = a < 2, a !== 1 && a !== 2 && E(e, "unsupported YAML version of the document");
		},
		TAG: function(e, t, n) {
			let r;
			n.length !== 2 && T(e, "TAG directive accepts exactly two arguments");
			let a = n[0];
			r = n[1], c.test(a) || T(e, "ill-formed tag handle (first argument) of the TAG directive"), i.call(e.tagMap, a) && T(e, "there is a previously declared suffix for \"" + a + "\" tag handle"), l.test(r) || T(e, "ill-formed tag prefix (second argument) of the TAG directive");
			try {
				r = decodeURIComponent(r);
			} catch {
				T(e, "tag prefix is malformed: " + r);
			}
			e.tagMap[a] = r;
		}
	};
	function ne(e, t, n, r) {
		if (t < n) {
			let i = e.input.slice(t, n);
			if (r) for (let t = 0, n = i.length; t < n; t += 1) {
				let n = i.charCodeAt(t);
				n === 9 || n >= 32 && n <= 1114111 || T(e, "expected valid JSON character");
			}
			else a.test(i) && T(e, "the stream contains non-printable characters");
			e.result += i;
		}
	}
	function re(t, n, r, a) {
		e.isObject(r) || T(t, "cannot merge mappings; the provided source object is unacceptable");
		let o = Object.keys(r);
		for (let e = 0, s = o.length; e < s; e += 1) {
			let s = o[e];
			t.maxTotalMergeKeys !== -1 && ++t.totalMergeKeys > t.maxTotalMergeKeys && T(t, "merge keys exceeded maxTotalMergeKeys (" + t.maxTotalMergeKeys + ")"), i.call(n, s) || (b(n, s, r[s]), a[s] = !0);
		}
	}
	function M(e, t, n, r, a, o, s, c, l) {
		if (Array.isArray(a)) {
			a = Array.prototype.slice.call(a);
			for (let t = 0, n = a.length; t < n; t += 1) Array.isArray(a[t]) && T(e, "nested arrays are not supported inside keys"), typeof a == "object" && u(a[t]) === "[object Object]" && (a[t] = "[object Object]");
		}
		if (typeof a == "object" && u(a) === "[object Object]" && (a = "[object Object]"), a = String(a), t === null && (t = {}), r === "tag:yaml.org,2002:merge") {
			if (Array.isArray(o)) for (let r = 0, i = o.length; r < i; r += 1) re(e, t, o[r], n);
			else re(e, t, o, n);
		} else !e.json && !i.call(n, a) && i.call(t, a) && (e.line = s || e.line, e.lineStart = c || e.lineStart, e.position = l || e.position, T(e, "duplicated mapping key")), b(t, a, o), delete n[a];
		return t;
	}
	function ie(e) {
		let t = e.input.charCodeAt(e.position);
		t === 10 ? e.position++ : t === 13 ? (e.position++, e.input.charCodeAt(e.position) === 10 && e.position++) : T(e, "a line break is expected"), e.line += 1, e.lineStart = e.position, e.firstTabInLine = -1;
	}
	function N(e, t, n) {
		let r = 0, i = e.input.charCodeAt(e.position);
		for (; i !== 0;) {
			for (; f(i);) i === 9 && e.firstTabInLine === -1 && (e.firstTabInLine = e.position), i = e.input.charCodeAt(++e.position);
			if (t && i === 35) do
				i = e.input.charCodeAt(++e.position);
			while (i !== 10 && i !== 13 && i !== 0);
			if (d(i)) for (ie(e), i = e.input.charCodeAt(e.position), r++, e.lineIndent = 0; i === 32;) e.lineIndent++, i = e.input.charCodeAt(++e.position);
			else break;
		}
		return n !== -1 && r !== 0 && e.lineIndent < n && E(e, "deficient indentation"), r;
	}
	function ae(e) {
		let t = e.position, n = e.input.charCodeAt(t);
		return !!((n === 45 || n === 46) && n === e.input.charCodeAt(t + 1) && n === e.input.charCodeAt(t + 2) && (t += 3, n = e.input.charCodeAt(t), n === 0 || p(n)));
	}
	function oe(t, n) {
		n === 1 ? t.result += " " : n > 1 && (t.result += e.repeat("\n", n - 1));
	}
	function se(e, t, n) {
		let r, i, a, o, s, c, l = e.kind, u = e.result, h = e.input.charCodeAt(e.position);
		if (p(h) || m(h) || h === 35 || h === 38 || h === 42 || h === 33 || h === 124 || h === 62 || h === 39 || h === 34 || h === 37 || h === 64 || h === 96) return !1;
		if (h === 63 || h === 45) {
			let t = e.input.charCodeAt(e.position + 1);
			if (p(t) || n && m(t)) return !1;
		}
		for (e.kind = "scalar", e.result = "", r = i = e.position, a = !1; h !== 0;) {
			if (h === 58) {
				let t = e.input.charCodeAt(e.position + 1);
				if (p(t) || n && m(t)) break;
			} else if (h === 35) {
				if (p(e.input.charCodeAt(e.position - 1))) break;
			} else if (e.position === e.lineStart && ae(e) || n && m(h)) break;
			else if (d(h)) {
				if (o = e.line, s = e.lineStart, c = e.lineIndent, N(e, !1, -1), e.lineIndent >= t) {
					a = !0, h = e.input.charCodeAt(e.position);
					continue;
				}
				e.position = i, e.line = o, e.lineStart = s, e.lineIndent = c;
				break;
			}
			a &&= (ne(e, r, i, !1), oe(e, e.line - o), r = i = e.position, !1), f(h) || (i = e.position + 1), h = e.input.charCodeAt(++e.position);
		}
		return ne(e, r, i, !1), e.result ? !0 : (e.kind = l, e.result = u, !1);
	}
	function ce(e, t) {
		let n, r, i = e.input.charCodeAt(e.position);
		if (i !== 39) return !1;
		for (e.kind = "scalar", e.result = "", e.position++, n = r = e.position; (i = e.input.charCodeAt(e.position)) !== 0;) if (i === 39) {
			if (ne(e, n, e.position, !0), i = e.input.charCodeAt(++e.position), i === 39) n = e.position, e.position++, r = e.position;
			else return !0;
		} else d(i) ? (ne(e, n, r, !0), oe(e, N(e, !1, t)), n = r = e.position) : e.position === e.lineStart && ae(e) ? T(e, "unexpected end of the document within a single quoted scalar") : (e.position++, f(i) || (r = e.position));
		T(e, "unexpected end of the stream within a single quoted scalar");
	}
	function le(e, t) {
		let n, r, i, a = e.input.charCodeAt(e.position);
		if (a !== 34) return !1;
		for (e.kind = "scalar", e.result = "", e.position++, n = r = e.position; (a = e.input.charCodeAt(e.position)) !== 0;) if (a === 34) return ne(e, n, e.position, !0), e.position++, !0;
		else if (a === 92) {
			if (ne(e, n, e.position, !0), a = e.input.charCodeAt(++e.position), d(a)) N(e, !1, t);
			else if (a < 256 && x[a]) e.result += S[a], e.position++;
			else if ((i = g(a)) > 0) {
				let t = i, n = 0;
				for (; t > 0; t--) a = e.input.charCodeAt(++e.position), (i = h(a)) >= 0 ? n = (n << 4) + i : T(e, "expected hexadecimal character");
				e.result += y(n), e.position++;
			} else T(e, "unknown escape sequence");
			n = r = e.position;
		} else d(a) ? (ne(e, n, r, !0), oe(e, N(e, !1, t)), n = r = e.position) : e.position === e.lineStart && ae(e) ? T(e, "unexpected end of the document within a double quoted scalar") : (e.position++, f(a) || (r = e.position));
		T(e, "unexpected end of the stream within a double quoted scalar");
	}
	function ue(e, t) {
		let n = !0, r, i, a, o = e.tag, s, c = e.anchor, l, u, d, f, m = /* @__PURE__ */ Object.create(null), h, g, _, v = e.input.charCodeAt(e.position);
		if (v === 91) l = 93, f = !1, s = [];
		else if (v === 123) l = 125, f = !0, s = {};
		else return !1;
		for (e.anchor !== null && D(e, e.anchor, s), v = e.input.charCodeAt(++e.position); v !== 0;) {
			if (N(e, !0, t), v = e.input.charCodeAt(e.position), v === l) return e.position++, e.tag = o, e.anchor = c, e.kind = f ? "mapping" : "sequence", e.result = s, !0;
			n ? v === 44 && T(e, "expected the node content, but found ','") : T(e, "missed comma between flow collection entries"), g = h = _ = null, u = d = !1, v === 63 && p(e.input.charCodeAt(e.position + 1)) && (u = d = !0, e.position++, N(e, !0, t)), r = e.line, i = e.lineStart, a = e.position, ge(e, t, 1, !1, !0), g = e.tag, h = e.result, N(e, !0, t), v = e.input.charCodeAt(e.position), (d || e.line === r) && v === 58 && (u = !0, v = e.input.charCodeAt(++e.position), N(e, !0, t), ge(e, t, 1, !1, !0), _ = e.result), f ? M(e, s, m, g, h, _, r, i, a) : u ? s.push(M(e, null, m, g, h, _, r, i, a)) : s.push(h), N(e, !0, t), v = e.input.charCodeAt(e.position), v === 44 ? (n = !0, v = e.input.charCodeAt(++e.position)) : n = !1;
		}
		T(e, "unexpected end of the stream within a flow collection");
	}
	function P(t, n) {
		let r, i = 1, a = !1, o = !1, s = n, c = 0, l = !1, u, p = t.input.charCodeAt(t.position);
		if (p === 124) r = !1;
		else if (p === 62) r = !0;
		else return !1;
		for (t.kind = "scalar", t.result = ""; p !== 0;) if (p = t.input.charCodeAt(++t.position), p === 43 || p === 45) i === 1 ? i = p === 43 ? 3 : 2 : T(t, "repeat of a chomping mode identifier");
		else if ((u = _(p)) >= 0) u === 0 ? T(t, "bad explicit indentation width of a block scalar; it cannot be less than one") : o ? T(t, "repeat of an indentation width identifier") : (s = n + u - 1, o = !0);
		else break;
		if (f(p)) {
			do
				p = t.input.charCodeAt(++t.position);
			while (f(p));
			if (p === 35) do
				p = t.input.charCodeAt(++t.position);
			while (!d(p) && p !== 0);
		}
		for (; p !== 0;) {
			for (ie(t), t.lineIndent = 0, p = t.input.charCodeAt(t.position); (!o || t.lineIndent < s) && p === 32;) t.lineIndent++, p = t.input.charCodeAt(++t.position);
			if (!o && t.lineIndent > s && (s = t.lineIndent), d(p)) {
				c++;
				continue;
			}
			if (!o && s === 0 && T(t, "missing indentation for block scalar"), t.lineIndent < s) {
				i === 3 ? t.result += e.repeat("\n", a ? 1 + c : c) : i === 1 && a && (t.result += "\n");
				break;
			}
			r ? f(p) ? (l = !0, t.result += e.repeat("\n", a ? 1 + c : c)) : l ? (l = !1, t.result += e.repeat("\n", c + 1)) : c === 0 ? a && (t.result += " ") : t.result += e.repeat("\n", c) : t.result += e.repeat("\n", a ? 1 + c : c), a = !0, o = !0, c = 0;
			let n = t.position;
			for (; !d(p) && p !== 0;) p = t.input.charCodeAt(++t.position);
			ne(t, n, t.position, !1);
		}
		return !0;
	}
	function de(e, t) {
		let n = e.tag, r = e.anchor, i = [], a = !1;
		if (e.firstTabInLine !== -1) return !1;
		e.anchor !== null && D(e, e.anchor, i);
		let o = e.input.charCodeAt(e.position);
		for (; o !== 0 && (e.firstTabInLine !== -1 && (e.position = e.firstTabInLine, T(e, "tab characters must not be used in indentation")), !(o !== 45 || !p(e.input.charCodeAt(e.position + 1))));) {
			if (a = !0, e.position++, N(e, !0, -1) && e.lineIndent <= t) {
				i.push(null), o = e.input.charCodeAt(e.position);
				continue;
			}
			let n = e.line;
			if (ge(e, t, 3, !1, !0), i.push(e.result), N(e, !0, -1), o = e.input.charCodeAt(e.position), (e.line === n || e.lineIndent > t) && o !== 0) T(e, "bad indentation of a sequence entry");
			else if (e.lineIndent < t) break;
		}
		return a ? (e.tag = n, e.anchor = r, e.kind = "sequence", e.result = i, !0) : !1;
	}
	function fe(e, t, n) {
		let r, i, a, o, s = e.tag, c = e.anchor, l = {}, u = /* @__PURE__ */ Object.create(null), d = null, m = null, h = null, g = !1, _ = !1;
		if (e.firstTabInLine !== -1) return !1;
		e.anchor !== null && D(e, e.anchor, l);
		let v = e.input.charCodeAt(e.position);
		for (; v !== 0;) {
			!g && e.firstTabInLine !== -1 && (e.position = e.firstTabInLine, T(e, "tab characters must not be used in indentation"));
			let y = e.input.charCodeAt(e.position + 1), b = e.line;
			if ((v === 63 || v === 58) && p(y)) v === 63 ? (g && (M(e, l, u, d, m, null, i, a, o), d = m = h = null), _ = !0, g = !0, r = !0) : g ? (g = !1, r = !0) : T(e, "incomplete explicit mapping pair; a key node is missed; or followed by a non-tabulated empty line"), e.position += 1, v = y;
			else {
				if (i = e.line, a = e.lineStart, o = e.position, !ge(e, n, 2, !1, !0)) break;
				if (e.line === b) {
					for (v = e.input.charCodeAt(e.position); f(v);) v = e.input.charCodeAt(++e.position);
					if (v === 58) v = e.input.charCodeAt(++e.position), p(v) || T(e, "a whitespace character is expected after the key-value separator within a block mapping"), g && (M(e, l, u, d, m, null, i, a, o), d = m = h = null), _ = !0, g = !1, r = !1, d = e.tag, m = e.result;
					else if (_) T(e, "can not read an implicit mapping pair; a colon is missed");
					else return e.tag = s, e.anchor = c, !0;
				} else if (_) T(e, "can not read a block mapping entry; a multiline key may not be an implicit key");
				else return e.tag = s, e.anchor = c, !0;
			}
			if ((e.line === b || e.lineIndent > t) && (g && (i = e.line, a = e.lineStart, o = e.position), ge(e, t, 4, !0, r) && (g ? m = e.result : h = e.result), g || (M(e, l, u, d, m, h, i, a, o), d = m = h = null), N(e, !0, -1), v = e.input.charCodeAt(e.position)), (e.line === b || e.lineIndent > t) && v !== 0) T(e, "bad indentation of a mapping entry");
			else if (e.lineIndent < t) break;
		}
		return g && M(e, l, u, d, m, null, i, a, o), _ && (e.tag = s, e.anchor = c, e.kind = "mapping", e.result = l), _;
	}
	function F(e) {
		let t = !1, n = !1, r, a, o = e.input.charCodeAt(e.position);
		if (o !== 33) return !1;
		e.tag !== null && T(e, "duplication of a tag property"), o = e.input.charCodeAt(++e.position), o === 60 ? (t = !0, o = e.input.charCodeAt(++e.position)) : o === 33 ? (n = !0, r = "!!", o = e.input.charCodeAt(++e.position)) : r = "!";
		let u = e.position;
		if (t) {
			do
				o = e.input.charCodeAt(++e.position);
			while (o !== 0 && o !== 62);
			e.position < e.length ? (a = e.input.slice(u, e.position), o = e.input.charCodeAt(++e.position)) : T(e, "unexpected end of the stream within a verbatim tag");
		} else {
			for (; o !== 0 && !p(o);) o === 33 && (n ? T(e, "tag suffix cannot contain exclamation marks") : (r = e.input.slice(u - 1, e.position + 1), c.test(r) || T(e, "named tag handle cannot contain such characters"), n = !0, u = e.position + 1)), o = e.input.charCodeAt(++e.position);
			a = e.input.slice(u, e.position), s.test(a) && T(e, "tag suffix cannot contain flow indicator characters");
		}
		a && !l.test(a) && T(e, "tag name cannot contain such characters: " + a);
		try {
			a = decodeURIComponent(a);
		} catch {
			T(e, "tag name is malformed: " + a);
		}
		return t ? e.tag = a : i.call(e.tagMap, r) ? e.tag = e.tagMap[r] + a : r === "!" ? e.tag = "!" + a : r === "!!" ? e.tag = "tag:yaml.org,2002:" + a : T(e, "undeclared tag handle \"" + r + "\""), !0;
	}
	function pe(e) {
		let t = e.input.charCodeAt(e.position);
		if (t !== 38) return !1;
		e.anchor !== null && T(e, "duplication of an anchor property"), t = e.input.charCodeAt(++e.position);
		let n = e.position;
		for (; t !== 0 && !p(t) && !m(t);) t = e.input.charCodeAt(++e.position);
		return e.position === n && T(e, "name of an anchor node must contain at least one character"), e.anchor = e.input.slice(n, e.position), !0;
	}
	function me(e) {
		let t = e.input.charCodeAt(e.position);
		if (t !== 42) return !1;
		t = e.input.charCodeAt(++e.position);
		let n = e.position;
		for (; t !== 0 && !p(t) && !m(t);) t = e.input.charCodeAt(++e.position);
		e.position === n && T(e, "name of an alias node must contain at least one character");
		let r = e.input.slice(n, e.position);
		return i.call(e.anchorMap, r) || T(e, "unidentified alias \"" + r + "\""), e.result = e.anchorMap[r], N(e, !0, -1), !0;
	}
	function he(e, t, n, r) {
		let i = A(e);
		return O(e), te(e, t), e.tag = null, e.anchor = null, e.kind = null, e.result = null, fe(e, n, r) && e.kind === "mapping" ? (ee(e), !0) : (k(e), te(e, i), !1);
	}
	function ge(e, t, n, r, a) {
		let o, s, c = 1, l = !1, u = !1, d = null, f, p, m;
		e.depth >= e.maxDepth && T(e, "nesting exceeded maxDepth (" + e.maxDepth + ")"), e.depth += 1, e.listener !== null && e.listener("open", e), e.tag = null, e.anchor = null, e.kind = null, e.result = null;
		let h = o = s = n === 4 || n === 3;
		if (r && N(e, !0, -1) && (l = !0, e.lineIndent > t ? c = 1 : e.lineIndent === t ? c = 0 : e.lineIndent < t && (c = -1)), c === 1) for (;;) {
			let n = e.input.charCodeAt(e.position), r = A(e);
			if (l && (n === 33 && e.tag !== null || n === 38 && e.anchor !== null) || !F(e) && !pe(e)) break;
			d === null && (d = r), N(e, !0, -1) ? (l = !0, s = h, e.lineIndent > t ? c = 1 : e.lineIndent === t ? c = 0 : e.lineIndent < t && (c = -1)) : s = !1;
		}
		if (s &&= l || a, c === 1 || n === 4) {
			if (p = n === 1 || n === 2 ? t : t + 1, m = e.position - e.lineStart, c === 1) {
				if (s && (de(e, m) || fe(e, m, p)) || ue(e, p)) u = !0;
				else {
					let t = e.input.charCodeAt(e.position);
					d !== null && h && !s && t !== 124 && t !== 62 && he(e, d, d.position - d.lineStart, p) || o && P(e, p) || ce(e, p) || le(e, p) ? u = !0 : me(e) ? (u = !0, (e.tag !== null || e.anchor !== null) && T(e, "alias node should not have any properties")) : se(e, p, n === 1) && (u = !0, e.tag === null && (e.tag = "?")), e.anchor !== null && D(e, e.anchor, e.result);
				}
			} else c === 0 && (u = s && de(e, m));
		}
		if (e.tag === null) e.anchor !== null && D(e, e.anchor, e.result);
		else if (e.tag === "?") {
			e.result !== null && e.kind !== "scalar" && T(e, "unacceptable node kind for !<?> tag; it should be \"scalar\", not \"" + e.kind + "\"");
			for (let t = 0, n = e.implicitTypes.length; t < n; t += 1) if (f = e.implicitTypes[t], f.resolve(e.result)) {
				e.result = f.construct(e.result), e.tag = f.tag, e.anchor !== null && D(e, e.anchor, e.result);
				break;
			}
		} else if (e.tag !== "!") {
			if (i.call(e.typeMap[e.kind || "fallback"], e.tag)) f = e.typeMap[e.kind || "fallback"][e.tag];
			else {
				f = null;
				let t = e.typeMap.multi[e.kind || "fallback"];
				for (let n = 0, r = t.length; n < r; n += 1) if (e.tag.slice(0, t[n].tag.length) === t[n].tag) {
					f = t[n];
					break;
				}
			}
			f || T(e, "unknown tag !<" + e.tag + ">"), e.result !== null && f.kind !== e.kind && T(e, "unacceptable node kind for !<" + e.tag + "> tag; it should be \"" + f.kind + "\", not \"" + e.kind + "\""), f.resolve(e.result, e.tag) ? (e.result = f.construct(e.result, e.tag), e.anchor !== null && D(e, e.anchor, e.result)) : T(e, "cannot resolve a node with !<" + e.tag + "> explicit tag");
		}
		return e.listener !== null && e.listener("close", e), --e.depth, e.tag !== null || e.anchor !== null || u;
	}
	function _e(e) {
		let t = e.position, n = !1, r;
		for (e.version = null, e.checkLineBreaks = e.legacy, e.tagMap = /* @__PURE__ */ Object.create(null), e.anchorMap = /* @__PURE__ */ Object.create(null); (r = e.input.charCodeAt(e.position)) !== 0 && (N(e, !0, -1), r = e.input.charCodeAt(e.position), !(e.lineIndent > 0 || r !== 37));) {
			n = !0, r = e.input.charCodeAt(++e.position);
			let t = e.position;
			for (; r !== 0 && !p(r);) r = e.input.charCodeAt(++e.position);
			let a = e.input.slice(t, e.position), o = [];
			for (a.length < 1 && T(e, "directive name must not be less than one character in length"); r !== 0;) {
				for (; f(r);) r = e.input.charCodeAt(++e.position);
				if (r === 35) {
					do
						r = e.input.charCodeAt(++e.position);
					while (r !== 0 && !d(r));
					break;
				}
				if (d(r)) break;
				for (t = e.position; r !== 0 && !p(r);) r = e.input.charCodeAt(++e.position);
				o.push(e.input.slice(t, e.position));
			}
			r !== 0 && ie(e), i.call(j, a) ? j[a](e, a, o) : E(e, "unknown document directive \"" + a + "\"");
		}
		if (N(e, !0, -1), e.lineIndent === 0 && e.input.charCodeAt(e.position) === 45 && e.input.charCodeAt(e.position + 1) === 45 && e.input.charCodeAt(e.position + 2) === 45 ? (e.position += 3, N(e, !0, -1)) : n && T(e, "directives end mark is expected"), ge(e, e.lineIndent - 1, 4, !1, !0), N(e, !0, -1), e.checkLineBreaks && o.test(e.input.slice(t, e.position)) && E(e, "non-ASCII line breaks are interpreted as content"), e.documents.push(e.result), e.position === e.lineStart && ae(e)) {
			e.input.charCodeAt(e.position) === 46 && (e.position += 3, N(e, !0, -1));
			return;
		}
		e.position < e.length - 1 && T(e, "end of the stream or a document separator is expected");
	}
	function ve(e, t) {
		e = String(e), t ||= {}, e.length !== 0 && (e.charCodeAt(e.length - 1) !== 10 && e.charCodeAt(e.length - 1) !== 13 && (e += "\n"), e.charCodeAt(0) === 65279 && (e = e.slice(1)));
		let n = new C(e, t), r = e.indexOf("\0");
		for (r !== -1 && (n.position = r, T(n, "null byte is not allowed in input")), n.input += "\0"; n.input.charCodeAt(n.position) === 32;) n.lineIndent += 1, n.position += 1;
		for (; n.position < n.length - 1;) _e(n);
		return n.documents;
	}
	function ye(e, t, n) {
		typeof t == "object" && t && n === void 0 && (n = t, t = null);
		let r = ve(e, n);
		if (typeof t != "function") return r;
		for (let e = 0, n = r.length; e < n; e += 1) t(r[e]);
	}
	function be(e, n) {
		let r = ve(e, n);
		if (r.length !== 0) {
			if (r.length === 1) return r[0];
			throw new t("expected a single document in the stream, but found more");
		}
	}
	return jl.loadAll = ye, jl.load = be, jl;
}
var Ku = {}, qu;
function Ju() {
	if (qu) return Ku;
	qu = 1;
	let e = Pl(), t = Ll(), n = Uu(), r = Object.prototype.toString, i = Object.prototype.hasOwnProperty, a = 65279, o = {};
	o[0] = "\\0", o[7] = "\\a", o[8] = "\\b", o[9] = "\\t", o[10] = "\\n", o[11] = "\\v", o[12] = "\\f", o[13] = "\\r", o[27] = "\\e", o[34] = "\\\"", o[92] = "\\\\", o[133] = "\\N", o[160] = "\\_", o[8232] = "\\L", o[8233] = "\\P";
	let s = [
		"y",
		"Y",
		"yes",
		"Yes",
		"YES",
		"on",
		"On",
		"ON",
		"n",
		"N",
		"no",
		"No",
		"NO",
		"off",
		"Off",
		"OFF"
	], c = /^[-+]?[0-9_]+(?::[0-9_]+)+(?:\.[0-9_]*)?$/;
	function l(e, t) {
		if (t === null) return {};
		let n = {}, r = Object.keys(t);
		for (let a = 0, o = r.length; a < o; a += 1) {
			let o = r[a], s = String(t[o]);
			o.slice(0, 2) === "!!" && (o = "tag:yaml.org,2002:" + o.slice(2));
			let c = e.compiledTypeMap.fallback[o];
			c && i.call(c.styleAliases, s) && (s = c.styleAliases[s]), n[o] = s;
		}
		return n;
	}
	function u(n) {
		let r, i, a = n.toString(16).toUpperCase();
		if (n <= 255) r = "x", i = 2;
		else if (n <= 65535) r = "u", i = 4;
		else if (n <= 4294967295) r = "U", i = 8;
		else throw new t("code point within a string may not be greater than 0xFFFFFFFF");
		return "\\" + r + e.repeat("0", i - a.length) + a;
	}
	function d(t) {
		this.schema = t.schema || n, this.indent = Math.max(1, t.indent || 2), this.noArrayIndent = t.noArrayIndent || !1, this.skipInvalid = t.skipInvalid || !1, this.flowLevel = e.isNothing(t.flowLevel) ? -1 : t.flowLevel, this.styleMap = l(this.schema, t.styles || null), this.sortKeys = t.sortKeys || !1, this.lineWidth = t.lineWidth || 80, this.noRefs = t.noRefs || !1, this.noCompatMode = t.noCompatMode || !1, this.condenseFlow = t.condenseFlow || !1, this.quotingType = t.quotingType === "\"" ? 2 : 1, this.forceQuotes = t.forceQuotes || !1, this.replacer = typeof t.replacer == "function" ? t.replacer : null, this.implicitTypes = this.schema.compiledImplicit, this.explicitTypes = this.schema.compiledExplicit, this.tag = null, this.result = "", this.duplicates = [], this.usedDuplicates = null;
	}
	function f(t, n) {
		let r = e.repeat(" ", n), i = 0, a = "", o = t.length;
		for (; i < o;) {
			let e, n = t.indexOf("\n", i);
			n === -1 ? (e = t.slice(i), i = o) : (e = t.slice(i, n + 1), i = n + 1), e.length && e !== "\n" && (a += r), a += e;
		}
		return a;
	}
	function p(t, n) {
		return "\n" + e.repeat(" ", t.indent * n);
	}
	function m(e, t) {
		for (let n = 0, r = e.implicitTypes.length; n < r; n += 1) if (e.implicitTypes[n].resolve(t)) return !0;
		return !1;
	}
	function h(e) {
		return e === 32 || e === 9;
	}
	function g(e) {
		return e >= 32 && e <= 126 || e >= 161 && e <= 55295 && e !== 8232 && e !== 8233 || e >= 57344 && e <= 65533 && e !== a || e >= 65536 && e <= 1114111;
	}
	function _(e) {
		return g(e) && e !== a && e !== 13 && e !== 10;
	}
	function v(e, t, n) {
		let r = _(e), i = r && !h(e);
		return (n ? r : r && e !== 44 && e !== 91 && e !== 93 && e !== 123 && e !== 125) && e !== 35 && !(t === 58 && !i) || _(t) && !h(t) && e === 35 || t === 58 && i;
	}
	function y(e) {
		return g(e) && e !== a && !h(e) && e !== 45 && e !== 63 && e !== 58 && e !== 44 && e !== 91 && e !== 93 && e !== 123 && e !== 125 && e !== 35 && e !== 38 && e !== 42 && e !== 33 && e !== 124 && e !== 61 && e !== 62 && e !== 39 && e !== 34 && e !== 37 && e !== 64 && e !== 96;
	}
	function b(e) {
		return !h(e) && e !== 58;
	}
	function x(e, t) {
		let n = e.charCodeAt(t), r;
		return n >= 55296 && n <= 56319 && t + 1 < e.length && (r = e.charCodeAt(t + 1), r >= 56320 && r <= 57343) ? (n - 55296) * 1024 + r - 56320 + 65536 : n;
	}
	function S(e) {
		return /^\n* /.test(e);
	}
	function C(e, t, n, r, i, a, o, s) {
		let c, l = 0, u = null, d = !1, f = !1, p = r !== -1, m = -1, h = y(x(e, 0)) && b(x(e, e.length - 1));
		if (t || o) for (c = 0; c < e.length; l >= 65536 ? c += 2 : c++) {
			if (l = x(e, c), !g(l)) return 5;
			h &&= v(l, u, s), u = l;
		}
		else {
			for (c = 0; c < e.length; l >= 65536 ? c += 2 : c++) {
				if (l = x(e, c), l === 10) d = !0, p && (f ||= c - m - 1 > r && e[m + 1] !== " ", m = c);
				else if (!g(l)) return 5;
				h &&= v(l, u, s), u = l;
			}
			f ||= p && c - m - 1 > r && e[m + 1] !== " ";
		}
		return !d && !f ? h && !o && !i(e) ? 1 : a === 2 ? 5 : 2 : n > 9 && S(e) ? 5 : o ? a === 2 ? 5 : 2 : f ? 4 : 3;
	}
	function w(e, n, r, i, a) {
		e.dump = (function() {
			if (n.length === 0) return e.quotingType === 2 ? "\"\"" : "''";
			if (!e.noCompatMode && (s.indexOf(n) !== -1 || c.test(n))) return e.quotingType === 2 ? "\"" + n + "\"" : "'" + n + "'";
			let o = e.indent * Math.max(1, r), l = e.lineWidth === -1 ? -1 : Math.max(Math.min(e.lineWidth, 40), e.lineWidth - o), u = i || e.flowLevel > -1 && r >= e.flowLevel;
			function d(t) {
				return m(e, t);
			}
			switch (C(n, u, e.indent, l, d, e.quotingType, e.forceQuotes && !i, a)) {
				case 1: return n;
				case 2: return "'" + n.replace(/'/g, "''") + "'";
				case 3: return "|" + T(n, e.indent) + E(f(n, o));
				case 4: return ">" + T(n, e.indent) + E(f(D(n, l), o));
				case 5: return "\"" + ee(n) + "\"";
				default: throw new t("impossible error: invalid scalar style");
			}
		})();
	}
	function T(e, t) {
		let n = S(e) ? String(t) : "", r = e[e.length - 1] === "\n";
		return n + (r && (e[e.length - 2] === "\n" || e === "\n") ? "+" : r ? "" : "-") + "\n";
	}
	function E(e) {
		return e[e.length - 1] === "\n" ? e.slice(0, -1) : e;
	}
	function D(e, t) {
		let n = /(\n+)([^\n]*)/g, r = (function() {
			let r = e.indexOf("\n");
			return r = r === -1 ? e.length : r, n.lastIndex = r, O(e.slice(0, r), t);
		})(), i = e[0] === "\n" || e[0] === " ", a, o;
		for (; o = n.exec(e);) {
			let e = o[1], n = o[2];
			a = n[0] === " ", r += e + (!i && !a && n !== "" ? "\n" : "") + O(n, t), i = a;
		}
		return r;
	}
	function O(e, t) {
		if (e === "" || e[0] === " ") return e;
		let n = / [^ ]/g, r, i = 0, a, o = 0, s = 0, c = "";
		for (; r = n.exec(e);) s = r.index, s - i > t && (a = o > i ? o : s, c += "\n" + e.slice(i, a), i = a + 1), o = s;
		return c += "\n", e.length - i > t && o > i ? c += e.slice(i, o) + "\n" + e.slice(o + 1) : c += e.slice(i), c.slice(1);
	}
	function ee(e) {
		let t = "", n = 0;
		for (let r = 0; r < e.length; n >= 65536 ? r += 2 : r++) {
			n = x(e, r);
			let i = o[n];
			!i && g(n) ? (t += e[r], n >= 65536 && (t += e[r + 1])) : t += i || u(n);
		}
		return t;
	}
	function k(e, t, n) {
		let r = "", i = e.tag;
		for (let i = 0, a = n.length; i < a; i += 1) {
			let a = n[i];
			e.replacer && (a = e.replacer.call(n, String(i), a)), (re(e, t, a, !1, !1) || a === void 0 && re(e, t, null, !1, !1)) && (r !== "" && (r += "," + (e.condenseFlow ? "" : " ")), r += e.dump);
		}
		e.tag = i, e.dump = "[" + r + "]";
	}
	function A(e, t, n, r) {
		let i = "", a = e.tag;
		for (let a = 0, o = n.length; a < o; a += 1) {
			let o = n[a];
			e.replacer && (o = e.replacer.call(n, String(a), o)), (re(e, t + 1, o, !0, !0, !1, !0) || o === void 0 && re(e, t + 1, null, !0, !0, !1, !0)) && ((!r || i !== "") && (i += p(e, t)), e.dump && e.dump.charCodeAt(0) === 10 ? i += "-" : i += "- ", i += e.dump);
		}
		e.tag = a, e.dump = i || "[]";
	}
	function te(e, t, n) {
		let r = "", i = e.tag, a = Object.keys(n);
		for (let i = 0, o = a.length; i < o; i += 1) {
			let o = "";
			r !== "" && (o += ", "), e.condenseFlow && (o += "\"");
			let s = a[i], c = n[s];
			e.replacer && (c = e.replacer.call(n, s, c)), re(e, t, s, !1, !1) && (e.dump.length > 1024 && (o += "? "), o += e.dump + (e.condenseFlow ? "\"" : "") + ":" + (e.condenseFlow ? "" : " "), re(e, t, c, !1, !1) && (o += e.dump, r += o));
		}
		e.tag = i, e.dump = "{" + r + "}";
	}
	function j(e, n, r, i) {
		let a = "", o = e.tag, s = Object.keys(r);
		if (e.sortKeys === !0) s.sort();
		else if (typeof e.sortKeys == "function") s.sort(e.sortKeys);
		else if (e.sortKeys) throw new t("sortKeys must be a boolean or a function");
		for (let t = 0, o = s.length; t < o; t += 1) {
			let o = "";
			(!i || a !== "") && (o += p(e, n));
			let c = s[t], l = r[c];
			if (e.replacer && (l = e.replacer.call(r, c, l)), !re(e, n + 1, c, !0, !0, !0)) continue;
			let u = e.tag !== null && e.tag !== "?" || e.dump && e.dump.length > 1024;
			u && (e.dump && e.dump.charCodeAt(0) === 10 ? o += "?" : o += "? "), o += e.dump, u && (o += p(e, n)), re(e, n + 1, l, !0, u) && (e.dump && e.dump.charCodeAt(0) === 10 ? o += ":" : o += ": ", o += e.dump, a += o);
		}
		e.tag = o, e.dump = a || "{}";
	}
	function ne(e, n, a) {
		let o = a ? e.explicitTypes : e.implicitTypes;
		for (let s = 0, c = o.length; s < c; s += 1) {
			let c = o[s];
			if ((c.instanceOf || c.predicate) && (!c.instanceOf || typeof n == "object" && n instanceof c.instanceOf) && (!c.predicate || c.predicate(n))) {
				if (e.tag = a ? c.multi && c.representName ? c.representName(n) : c.tag : "?", c.represent) {
					let a = e.styleMap[c.tag] || c.defaultStyle, o;
					if (r.call(c.represent) === "[object Function]") o = c.represent(n, a);
					else if (i.call(c.represent, a)) o = c.represent[a](n, a);
					else throw new t("!<" + c.tag + "> tag resolver accepts not \"" + a + "\" style");
					e.dump = o;
				}
				return !0;
			}
		}
		return !1;
	}
	function re(e, n, i, a, o, s, c) {
		e.tag = null, e.dump = i, ne(e, i, !1) || ne(e, i, !0);
		let l = r.call(e.dump), u = a;
		a &&= e.flowLevel < 0 || e.flowLevel > n;
		let d = l === "[object Object]" || l === "[object Array]", f, p;
		if (d && (f = e.duplicates.indexOf(i), p = f !== -1), (e.tag !== null && e.tag !== "?" || p || e.indent !== 2 && n > 0) && (o = !1), p && e.usedDuplicates[f]) e.dump = "*ref_" + f;
		else {
			if (d && p && !e.usedDuplicates[f] && (e.usedDuplicates[f] = !0), l === "[object Object]") a && Object.keys(e.dump).length !== 0 ? (j(e, n, e.dump, o), p && (e.dump = "&ref_" + f + e.dump)) : (te(e, n, e.dump), p && (e.dump = "&ref_" + f + " " + e.dump));
			else if (l === "[object Array]") a && e.dump.length !== 0 ? (e.noArrayIndent && !c && n > 0 ? A(e, n - 1, e.dump, o) : A(e, n, e.dump, o), p && (e.dump = "&ref_" + f + e.dump)) : (k(e, n, e.dump), p && (e.dump = "&ref_" + f + " " + e.dump));
			else if (l === "[object String]") e.tag !== "?" && w(e, e.dump, n, s, u);
			else if (l === "[object Undefined]") return !1;
			else {
				if (e.skipInvalid) return !1;
				throw new t("unacceptable kind of an object to dump " + l);
			}
			if (e.tag !== null && e.tag !== "?") {
				let t = encodeURI(e.tag[0] === "!" ? e.tag.slice(1) : e.tag).replace(/!/g, "%21");
				t = e.tag[0] === "!" ? "!" + t : t.slice(0, 18) === "tag:yaml.org,2002:" ? "!!" + t.slice(18) : "!<" + t + ">", e.dump = t + " " + e.dump;
			}
		}
		return !0;
	}
	function M(e, t) {
		let n = [], r = [];
		ie(e, n, r);
		let i = r.length;
		for (let e = 0; e < i; e += 1) t.duplicates.push(n[r[e]]);
		t.usedDuplicates = Array(i);
	}
	function ie(e, t, n) {
		if (typeof e == "object" && e) {
			let r = t.indexOf(e);
			if (r !== -1) n.indexOf(r) === -1 && n.push(r);
			else if (t.push(e), Array.isArray(e)) for (let r = 0, i = e.length; r < i; r += 1) ie(e[r], t, n);
			else {
				let r = Object.keys(e);
				for (let i = 0, a = r.length; i < a; i += 1) ie(e[r[i]], t, n);
			}
		}
	}
	function N(e, t) {
		t ||= {};
		let n = new d(t);
		n.noRefs || M(e, n);
		let r = e;
		return n.replacer && (r = n.replacer.call({ "": r }, "", r)), re(n, 0, r, !0, !0) ? n.dump + "\n" : "";
	}
	return Ku.dump = N, Ku;
}
var Yu;
function Xu() {
	if (Yu) return Al;
	Yu = 1;
	let e = Gu(), t = Ju();
	function n(e, t) {
		return function() {
			throw Error("Function yaml." + e + " is removed in js-yaml 4. Use yaml." + t + " instead, which is now safe by default.");
		};
	}
	return Al.Type = Ul(), Al.Schema = Kl(), Al.FAILSAFE_SCHEMA = iu(), Al.JSON_SCHEMA = yu(), Al.CORE_SCHEMA = Su(), Al.DEFAULT_SCHEMA = Uu(), Al.load = e.load, Al.loadAll = e.loadAll, Al.dump = t.dump, Al.YAMLException = Ll(), Al.types = {
		binary: ju(),
		float: gu(),
		map: tu(),
		null: su(),
		pairs: Lu(),
		set: Bu(),
		timestamp: Tu(),
		bool: uu(),
		int: pu(),
		merge: Ou(),
		omap: Pu(),
		seq: Ql(),
		str: Yl()
	}, Al.safeLoad = n("safeLoad", "load"), Al.safeLoadAll = n("safeLoadAll", "loadAll"), Al.safeDump = n("safeDump", "dump"), Al;
}
var { Type: Zu, Schema: Qu, FAILSAFE_SCHEMA: $u, JSON_SCHEMA: ed, CORE_SCHEMA: td, DEFAULT_SCHEMA: nd, load: rd, loadAll: id, dump: ad, YAMLException: od, types: sd, safeLoad: cd, safeLoadAll: ld, safeDump: ud } = /* @__PURE__ */ kl(Xu()), dd = /* @__PURE__ */ z({ WORKER_CODE: () => fd }), fd, pd = R((() => {
	fd = "\"use strict\";(()=>{var hr=Object.defineProperty;var mr=(i,n)=>{for(var t in n)hr(i,t,{get:n[t],enumerable:!0})};var In={};mr(In,{Diff:()=>$,FILE_HEADERS_ONLY:()=>Ai,INCLUDE_HEADERS:()=>$e,OMIT_HEADERS:()=>xi,applyPatch:()=>Ln,applyPatches:()=>yi,arrayDiff:()=>bn,canonicalize:()=>Le,characterDiff:()=>ln,convertChangesToDMP:()=>vi,convertChangesToXML:()=>Si,createPatch:()=>Ci,createTwoFilesPatch:()=>Nn,cssDiff:()=>xn,diffArrays:()=>pi,diffChars:()=>ri,diffCss:()=>ai,diffJson:()=>di,diffLines:()=>ee,diffSentences:()=>si,diffTrimmedLines:()=>ci,diffWords:()=>ui,diffWordsWithSpace:()=>hn,formatPatch:()=>Ne,jsonDiff:()=>vn,lineDiff:()=>ze,parsePatch:()=>ke,reversePatch:()=>kn,sentenceDiff:()=>wn,structuredPatch:()=>Ke,wordDiff:()=>dn,wordsWithSpaceDiff:()=>pn});var $=class{diff(n,t,l={}){let o;typeof l==\"function\"?(o=l,l={}):\"callback\"in l&&(o=l.callback);let u=this.castInput(n,l),f=this.castInput(t,l),c=this.removeEmpty(this.tokenize(u,l)),d=this.removeEmpty(this.tokenize(f,l));return this.diffWithOptionsObj(c,d,l,o)}diffWithOptionsObj(n,t,l,o){var u;let f=S=>{if(S=this.postProcess(S,l),o){setTimeout(function(){o(S)},0);return}else return S},c=t.length,d=n.length,s=1,a=c+d;l.maxEditLength!=null&&(a=Math.min(a,l.maxEditLength));let m=(u=l.timeout)!==null&&u!==void 0?u:1/0,I=Date.now()+m,R=[{oldPos:-1,lastComponent:void 0}],v=this.extractCommon(R[0],t,n,0,l);if(R[0].oldPos+1>=d&&v+1>=c)return f(this.buildValues(R[0].lastComponent,t,n));let g=-1/0,N=1/0,O=()=>{for(let S=Math.max(g,-s);S<=Math.min(N,s);S+=2){let b,_=R[S-1],P=R[S+1];_&&(R[S-1]=void 0);let U=!1;if(P){let Y=P.oldPos-S;U=P&&0<=Y&&Y<c}let q=_&&_.oldPos+1<d;if(!U&&!q){R[S]=void 0;continue}if(!q||U&&_.oldPos<P.oldPos?b=this.addToPath(P,!0,!1,0,l):b=this.addToPath(_,!1,!0,1,l),v=this.extractCommon(b,t,n,S,l),b.oldPos+1>=d&&v+1>=c)return f(this.buildValues(b.lastComponent,t,n))||!0;R[S]=b,b.oldPos+1>=d&&(N=Math.min(N,S-1)),v+1>=c&&(g=Math.max(g,S+1))}s++};if(o)(function S(){setTimeout(function(){if(s>a||Date.now()>I)return o(void 0);O()||S()},0)})();else for(;s<=a&&Date.now()<=I;){let S=O();if(S)return S}}addToPath(n,t,l,o,u){let f=n.lastComponent;return f&&!u.oneChangePerToken&&f.added===t&&f.removed===l?{oldPos:n.oldPos+o,lastComponent:{count:f.count+1,added:t,removed:l,previousComponent:f.previousComponent}}:{oldPos:n.oldPos+o,lastComponent:{count:1,added:t,removed:l,previousComponent:f}}}extractCommon(n,t,l,o,u){let f=t.length,c=l.length,d=n.oldPos,s=d-o,a=0;for(;s+1<f&&d+1<c&&this.equals(l[d+1],t[s+1],u);)s++,d++,a++,u.oneChangePerToken&&(n.lastComponent={count:1,previousComponent:n.lastComponent,added:!1,removed:!1});return a&&!u.oneChangePerToken&&(n.lastComponent={count:a,previousComponent:n.lastComponent,added:!1,removed:!1}),n.oldPos=d,s}equals(n,t,l){return l.comparator?l.comparator(n,t):n===t||!!l.ignoreCase&&n.toLowerCase()===t.toLowerCase()}removeEmpty(n){let t=[];for(let l=0;l<n.length;l++)n[l]&&t.push(n[l]);return t}castInput(n,t){return n}tokenize(n,t){return Array.from(n)}join(n){return n.join(\"\")}postProcess(n,t){return n}get useLongestToken(){return!1}buildValues(n,t,l){let o=[],u;for(;n;)o.push(n),u=n.previousComponent,delete n.previousComponent,n=u;o.reverse();let f=o.length,c=0,d=0,s=0;for(;c<f;c++){let a=o[c];if(a.removed)a.value=this.join(l.slice(s,s+a.count)),s+=a.count;else{if(!a.added&&this.useLongestToken){let m=t.slice(d,d+a.count);m=m.map(function(I,R){let v=l[s+R];return v.length>I.length?v:I}),a.value=this.join(m)}else a.value=this.join(t.slice(d,d+a.count));d+=a.count,a.added||(s+=a.count)}}return o}};var tn=class extends ${},ln=new tn;function ri(i,n,t){return ln.diff(i,n,t)}function on(i,n){let t;for(t=0;t<i.length&&t<n.length;t++)if(i[t]!=n[t])return i.slice(0,t);return i.slice(0,t)}function un(i,n){let t;if(!i||!n||i[i.length-1]!=n[n.length-1])return\"\";for(t=0;t<i.length&&t<n.length;t++)if(i[i.length-(t+1)]!=n[n.length-(t+1)])return i.slice(-t);return i.slice(-t)}function Ye(i,n,t){if(i.slice(0,n.length)!=n)throw Error(`string ${JSON.stringify(i)} doesn't start with prefix ${JSON.stringify(n)}; this is a bug`);return t+i.slice(n.length)}function Ue(i,n,t){if(!n)return i+t;if(i.slice(-n.length)!=n)throw Error(`string ${JSON.stringify(i)} doesn't end with suffix ${JSON.stringify(n)}; this is a bug`);return i.slice(0,-n.length)+t}function xe(i,n){return Ye(i,n,\"\")}function Te(i,n){return Ue(i,n,\"\")}function fn(i,n){return n.slice(0,gr(i,n))}function gr(i,n){let t=0;i.length>n.length&&(t=i.length-n.length);let l=n.length;i.length<n.length&&(l=i.length);let o=Array(l),u=0;o[0]=0;for(let f=1;f<l;f++){for(n[f]==n[u]?o[f]=o[u]:o[f]=u;u>0&&n[f]!=n[u];)u=o[u];n[f]==n[u]&&u++}u=0;for(let f=t;f<i.length;f++){for(;u>0&&i[f]!=n[u];)u=o[u];i[f]==n[u]&&u++}return u}function ti(i){return i.includes(`\\r\n`)&&!i.startsWith(`\n`)&&!i.match(/[^\\r]\\n/)}function li(i){return!i.includes(`\\r\n`)&&i.includes(`\n`)}function cn(i,n){let t=[];for(let l of Array.from(n.segment(i))){let o=l.segment;t.length&&/\\s/.test(t[t.length-1])&&/\\s/.test(o)?t[t.length-1]+=o:t.push(o)}return t}function Be(i,n){if(n)return Ce(i,n)[1];let t;for(t=i.length-1;t>=0&&i[t].match(/\\s/);t--);return i.substring(t+1)}function he(i,n){if(n)return Ce(i,n)[0];let t=i.match(/^\\s*/);return t?t[0]:\"\"}function Ce(i,n){if(!n)return[he(i),Be(i)];if(n.resolvedOptions().granularity!=\"word\")throw new Error('The segmenter passed must have a granularity of \"word\"');let t=cn(i,n),l=t[0],o=t[t.length-1],u=/\\s/.test(l)?l:\"\",f=/\\s/.test(o)?o:\"\";return[u,f]}var Ge=\"a-zA-Z0-9_\\\\u{AD}\\\\u{C0}-\\\\u{D6}\\\\u{D8}-\\\\u{F6}\\\\u{F8}-\\\\u{2C6}\\\\u{2C8}-\\\\u{2D7}\\\\u{2DE}-\\\\u{2FF}\\\\u{1E00}-\\\\u{1EFF}\",yr=new RegExp(`[${Ge}]+|\\\\s+|[^${Ge}]`,\"ug\"),sn=class extends ${equals(n,t,l){return l.ignoreCase&&(n=n.toLowerCase(),t=t.toLowerCase()),n.trim()===t.trim()}tokenize(n,t={}){let l;if(t.intlSegmenter){let f=t.intlSegmenter;if(f.resolvedOptions().granularity!=\"word\")throw new Error('The segmenter passed must have a granularity of \"word\"');l=cn(n,f)}else l=n.match(yr)||[];let o=[],u=null;return l.forEach(f=>{/\\s/.test(f)?u==null?o.push(f):o.push(o.pop()+f):u!=null&&/\\s/.test(u)?o[o.length-1]==u?o.push(o.pop()+f):o.push(u+f):o.push(f),u=f}),o}join(n){return n.map((t,l)=>l==0?t:t.replace(/^\\s+/,\"\")).join(\"\")}postProcess(n,t){if(!n||t.oneChangePerToken)return n;let l=null,o=null,u=null;return n.forEach(f=>{f.added?o=f:f.removed?u=f:((o||u)&&oi(l,u,o,f,t.intlSegmenter),l=f,o=null,u=null)}),(o||u)&&oi(l,u,o,null,t.intlSegmenter),n}},dn=new sn;function ui(i,n,t){return t?.ignoreWhitespace!=null&&!t.ignoreWhitespace?hn(i,n,t):dn.diff(i,n,t)}function oi(i,n,t,l,o){if(n&&t){let[u,f]=Ce(n.value,o),[c,d]=Ce(t.value,o);if(i){let s=on(u,c);i.value=Ue(i.value,c,s),n.value=xe(n.value,s),t.value=xe(t.value,s)}if(l){let s=un(f,d);l.value=Ye(l.value,d,s),n.value=Te(n.value,s),t.value=Te(t.value,s)}}else if(t){if(i){let u=he(t.value,o);t.value=t.value.substring(u.length)}if(l){let u=he(l.value,o);l.value=l.value.substring(u.length)}}else if(i&&l){let u=he(l.value,o),[f,c]=Ce(n.value,o),d=on(u,f);n.value=xe(n.value,d);let s=un(xe(u,d),c);n.value=Te(n.value,s),l.value=Ye(l.value,u,s),i.value=Ue(i.value,u,u.slice(0,u.length-s.length))}else if(l){let u=he(l.value,o),f=Be(n.value,o),c=fn(f,u);n.value=Te(n.value,c)}else if(i){let u=Be(i.value,o),f=he(n.value,o),c=fn(u,f);n.value=xe(n.value,c)}}var an=class extends ${tokenize(n){let t=new RegExp(`(\\\\r?\\\\n)|[${Ge}]+|[^\\\\S\\\\n\\\\r]+|[^${Ge}]`,\"ug\");return n.match(t)||[]}},pn=new an;function hn(i,n,t){return pn.diff(i,n,t)}function fi(i,n){if(typeof i==\"function\")n.callback=i;else if(i)for(let t in i)Object.prototype.hasOwnProperty.call(i,t)&&(n[t]=i[t]);return n}var mn=class extends ${constructor(){super(...arguments),this.tokenize=gn}equals(n,t,l){return l.ignoreWhitespace?((!l.newlineIsToken||!n.includes(`\n`))&&(n=n.trim()),(!l.newlineIsToken||!t.includes(`\n`))&&(t=t.trim())):l.ignoreNewlineAtEof&&!l.newlineIsToken&&(n.endsWith(`\n`)&&(n=n.slice(0,-1)),t.endsWith(`\n`)&&(t=t.slice(0,-1))),super.equals(n,t,l)}},ze=new mn;function ee(i,n,t){return ze.diff(i,n,t)}function ci(i,n,t){return t=fi(t,{ignoreWhitespace:!0}),ze.diff(i,n,t)}function gn(i,n){n.stripTrailingCr&&(i=i.replace(/\\r\\n/g,`\n`));let t=[],l=i.split(/(\\n|\\r\\n)/);l[l.length-1]||l.pop();for(let o=0;o<l.length;o++){let u=l[o];o%2&&!n.newlineIsToken?t[t.length-1]+=u:t.push(u)}return t}function wr(i){return i==\".\"||i==\"!\"||i==\"?\"}var yn=class extends ${tokenize(n){var t;let l=[],o=0;for(let u=0;u<n.length;u++){if(u==n.length-1){l.push(n.slice(o));break}if(wr(n[u])&&n[u+1].match(/\\s/)){for(l.push(n.slice(o,u+1)),u=o=u+1;!((t=n[u+1])===null||t===void 0)&&t.match(/\\s/);)u++;l.push(n.slice(o,u+1)),o=u+1}}return l}},wn=new yn;function si(i,n,t){return wn.diff(i,n,t)}var An=class extends ${tokenize(n){return n.split(/([{}:;,]|\\s+)/)}},xn=new An;function ai(i,n,t){return xn.diff(i,n,t)}var Cn=class extends ${constructor(){super(...arguments),this.tokenize=gn}get useLongestToken(){return!0}castInput(n,t){let{undefinedReplacement:l,stringifyReplacer:o=(u,f)=>typeof f>\"u\"?l:f}=t;return typeof n==\"string\"?n:JSON.stringify(Le(n,null,null,o),null,\"  \")}equals(n,t,l){return super.equals(n.replace(/,([\\r\\n])/g,\"$1\"),t.replace(/,([\\r\\n])/g,\"$1\"),l)}},vn=new Cn;function di(i,n,t){return vn.diff(i,n,t)}function Le(i,n,t,l,o){n=n||[],t=t||[],l&&(i=l(o===void 0?\"\":o,i));let u;for(u=0;u<n.length;u+=1)if(n[u]===i)return t[u];let f;if(Object.prototype.toString.call(i)===\"[object Array]\"){for(n.push(i),f=new Array(i.length),t.push(f),u=0;u<i.length;u+=1)f[u]=Le(i[u],n,t,l,String(u));return n.pop(),t.pop(),f}if(i&&i.toJSON&&(i=i.toJSON()),typeof i==\"object\"&&i!==null){n.push(i),f={},t.push(f);let c=[],d;for(d in i)Object.prototype.hasOwnProperty.call(i,d)&&c.push(d);for(c.sort(),u=0;u<c.length;u+=1)d=c[u],f[d]=Le(i[d],n,t,l,d);n.pop(),t.pop()}else f=i;return f}var Sn=class extends ${tokenize(n){return n.slice()}join(n){return n}removeEmpty(n){return n}},bn=new Sn;function pi(i,n,t){return bn.diff(i,n,t)}function En(i){return Array.isArray(i)?i.map(n=>En(n)):Object.assign(Object.assign({},i),{hunks:i.hunks.map(n=>Object.assign(Object.assign({},n),{lines:n.lines.map((t,l)=>{var o;return t.startsWith(\"\\\\\")||t.endsWith(\"\\r\")||!((o=n.lines[l+1])===null||o===void 0)&&o.startsWith(\"\\\\\")?t:t+\"\\r\"})}))})}function Tn(i){return Array.isArray(i)?i.map(n=>Tn(n)):Object.assign(Object.assign({},i),{hunks:i.hunks.map(n=>Object.assign(Object.assign({},n),{lines:n.lines.map(t=>t.endsWith(\"\\r\")?t.substring(0,t.length-1):t)}))})}function hi(i){return Array.isArray(i)||(i=[i]),!i.some(n=>n.hunks.some(t=>t.lines.some(l=>!l.startsWith(\"\\\\\")&&l.endsWith(\"\\r\"))))}function mi(i){return Array.isArray(i)||(i=[i]),i.some(n=>n.hunks.some(t=>t.lines.some(l=>l.endsWith(\"\\r\"))))&&i.every(n=>n.hunks.every(t=>t.lines.every((l,o)=>{var u;return l.startsWith(\"\\\\\")||l.endsWith(\"\\r\")||((u=t.lines[o+1])===null||u===void 0?void 0:u.startsWith(\"\\\\\"))})))}function ke(i){let n=i.split(/\\n/),t=[],l=0;function o(v){return/^diff --git /.test(v)}function u(v){return o(v)||/^Index:\\s/.test(v)||/^diff(?: -r \\w+)+\\s/.test(v)}function f(v){return/^(---|\\+\\+\\+)\\s/.test(v)}function c(v){return/^@@\\s/.test(v)}function d(){var v;let g={};g.hunks=[],t.push(g);let N=!1;for(;l<n.length;){let O=n[l];if(f(O)||c(O))break;if(o(O)){if(N)return;N=!0,g.isGit=!0;let S=s(O);for(S&&(g.oldFileName=S.oldFileName,g.newFileName=S.newFileName),l++;l<n.length;){let b=n[l];if(f(b)||c(b)||u(b))break;let _=/^rename from (.*)/.exec(b);_&&(g.oldFileName=\"a/\"+a(_[1]),g.isRename=!0);let P=/^rename to (.*)/.exec(b);P&&(g.newFileName=\"b/\"+a(P[1]),g.isRename=!0);let U=/^copy from (.*)/.exec(b);U&&(g.oldFileName=\"a/\"+a(U[1]),g.isCopy=!0);let q=/^copy to (.*)/.exec(b);q&&(g.newFileName=\"b/\"+a(q[1]),g.isCopy=!0);let Y=/^new file mode (\\d+)/.exec(b);Y&&(g.isCreate=!0,g.newMode=Y[1]);let J=/^deleted file mode (\\d+)/.exec(b);J&&(g.isDelete=!0,g.oldMode=J[1]);let re=/^old mode (\\d+)/.exec(b);re&&(g.oldMode=re[1]);let ie=/^new mode (\\d+)/.exec(b);ie&&(g.newMode=ie[1]),/^Binary files /.test(b)&&(g.isBinary=!0),l++}continue}else if(u(O)){if(N)return;N=!0;let S=/^(?:Index:|diff(?: -r \\w+)+)\\s+/.exec(O);S&&(g.index=O.substring(S[0].length).trim())}l++}if(I(g),I(g),g.oldFileName===void 0!=(g.newFileName===void 0))throw new Error(\"Missing \"+(g.oldFileName!==void 0?'\"+++ ...\"':'\"--- ...\"')+\" file header for \"+((v=g.oldFileName)!==null&&v!==void 0?v:g.newFileName));for(;l<n.length;){let O=n[l];if(u(O)||f(O)||/^===================================================================/.test(O))break;c(O)?g.hunks.push(R()):l++}}function s(v){let g=v.substring(11);if(g.startsWith('\"')){let O=m(g);if(O===null)return null;let S=g.substring(O.rawLength+1),b;if(S.startsWith('\"')){let _=m(S);if(_===null)return null;b=_.fileName}else b=S;return{oldFileName:O.fileName,newFileName:b}}let N=g.indexOf('\"');if(N>0){let O=g.substring(0,N-1),S=m(g.substring(N));return S===null?null:{oldFileName:O,newFileName:S.fileName}}if(g.startsWith(\"a/\")){let O=[],S=0;for(;S=g.indexOf(\" b/\",S+1),S!==-1;)O.push(S);if(O.length>0){let b=O[Math.floor(O.length/2)];return{oldFileName:g.substring(0,b),newFileName:g.substring(b+1)}}}return null}function a(v){if(v.startsWith('\"')){let g=m(v);if(g)return g.fileName}return v}function m(v){if(!v.startsWith('\"'))return null;let g=\"\",N=1;for(;N<v.length;){if(v[N]==='\"')return{fileName:g,rawLength:N+1};if(v[N]===\"\\\\\"&&N+1<v.length)switch(N++,v[N]){case\"a\":g+=\"\\x07\";break;case\"b\":g+=\"\\b\";break;case\"f\":g+=\"\\f\";break;case\"n\":g+=`\n`;break;case\"r\":g+=\"\\r\";break;case\"t\":g+=\"	\";break;case\"v\":g+=\"\\v\";break;case\"\\\\\":g+=\"\\\\\";break;case'\"':g+='\"';break;case\"0\":case\"1\":case\"2\":case\"3\":case\"4\":case\"5\":case\"6\":case\"7\":{if(N+2>=v.length||v[N+1]<\"0\"||v[N+1]>\"7\"||v[N+2]<\"0\"||v[N+2]>\"7\")return null;let O=[parseInt(v.substring(N,N+3),8)];for(N+=3;v[N]===\"\\\\\"&&v[N+1]>=\"0\"&&v[N+1]<=\"7\";){if(N+3>=v.length||v[N+2]<\"0\"||v[N+2]>\"7\"||v[N+3]<\"0\"||v[N+3]>\"7\")return null;O.push(parseInt(v.substring(N+1,N+4),8)),N+=4}g+=new TextDecoder(\"utf-8\").decode(new Uint8Array(O));continue}default:return null}else g+=v[N];N++}return null}function I(v){let g=/^(---|\\+\\+\\+)\\s+/.exec(n[l]);if(g){let N=g[1],O=n[l].substring(3).trim().split(\"	\",2),S=(O[1]||\"\").trim(),b=O[0];b.startsWith('\"')?b=a(b):b=b.replace(/\\\\\\\\/g,\"\\\\\"),N===\"---\"?(v.oldFileName=b,v.oldHeader=S):(v.newFileName=b,v.newHeader=S),l++}}function R(){var v;let g=l,N=n[l++],O=N.split(/@@ -(\\d+)(?:,(\\d+))? \\+(\\d+)(?:,(\\d+))? @@/),S={oldStart:+O[1],oldLines:typeof O[2]>\"u\"?1:+O[2],newStart:+O[3],newLines:typeof O[4]>\"u\"?1:+O[4],lines:[]};S.oldLines===0&&(S.oldStart+=1),S.newLines===0&&(S.newStart+=1);let b=0,_=0;for(;l<n.length&&(_<S.oldLines||b<S.newLines||!((v=n[l])===null||v===void 0)&&v.startsWith(\"\\\\\"));l++){let P=n[l].length==0&&l!=n.length-1?\" \":n[l][0];if(P===\"+\"||P===\"-\"||P===\" \"||P===\"\\\\\")S.lines.push(n[l]),P===\"+\"?b++:P===\"-\"?_++:P===\" \"&&(b++,_++);else throw new Error(`Hunk at line ${g+1} contained invalid line ${n[l]}`)}if(!b&&S.newLines===1&&(S.newLines=0),!_&&S.oldLines===1&&(S.oldLines=0),b!==S.newLines)throw new Error(\"Added line count did not match for hunk at line \"+(g+1));if(_!==S.oldLines)throw new Error(\"Removed line count did not match for hunk at line \"+(g+1));if(l<n.length&&n[l]&&/^[+ -]/.test(n[l])&&!f(n[l]))throw new Error(\"Hunk at line \"+(g+1)+\" has more lines than expected (expected \"+S.oldLines+\" old lines and \"+S.newLines+\" new lines)\");return S}for(;l<n.length;)d();return t}function gi(i,n,t){let l=!0,o=!1,u=!1,f=1;return function c(){if(l&&!u){if(o?f++:l=!1,i+f<=t)return i+f;u=!0}if(!o)return u||(l=!0),n<=i-f?i-f++:(o=!0,c())}}function Ln(i,n,t={}){let l;if(typeof n==\"string\"?l=ke(n):Array.isArray(n)?l=n:l=[n],l.length>1)throw new Error(\"applyPatch only works with a single input.\");return Ar(i,l[0],t)}function Ar(i,n,t={}){(t.autoConvertLineEndings||t.autoConvertLineEndings==null)&&(ti(i)&&hi(n)?n=En(n):li(i)&&mi(n)&&(n=Tn(n)));let l=i.split(`\n`),o=n.hunks,u=t.compareLine||((v,g,N,O)=>g===O),f=t.fuzzFactor||0,c=0;if(f<0||!Number.isInteger(f))throw new Error(\"fuzzFactor must be a non-negative integer\");if(!o.length)return i;let d=\"\",s=!1,a=!1;for(let v=0;v<o[o.length-1].lines.length;v++){let g=o[o.length-1].lines[v];g[0]==\"\\\\\"&&(d[0]==\"+\"?s=!0:d[0]==\"-\"&&(a=!0)),d=g}if(s){if(a){if(!f&&l[l.length-1]==\"\")return!1}else if(l[l.length-1]==\"\")l.pop();else if(!f)return!1}else if(a){if(l[l.length-1]!=\"\")l.push(\"\");else if(!f)return!1}function m(v,g,N,O=0,S=!0,b=[],_=0){let P=0,U=!1;for(;O<v.length;O++){let q=v[O],Y=q.length>0?q[0]:\" \",J=q.length>0?q.substr(1):q;if(Y===\"-\")if(u(g+1,l[g],Y,J))g++,P=0;else return!N||l[g]==null?null:(b[_]=l[g],m(v,g+1,N-1,O,!1,b,_+1));if(Y===\"+\"){if(!S)return null;b[_]=J,_++,P=0,U=!0}if(Y===\" \")if(P++,b[_]=l[g],u(g+1,l[g],Y,J))_++,S=!0,U=!1,g++;else return U||!N?null:l[g]&&(m(v,g+1,N-1,O+1,!1,b,_+1)||m(v,g+1,N-1,O,!1,b,_+1))||m(v,g,N-1,O+1,!1,b,_)}return _-=P,g-=P,b.length=_,{patchedLines:b,oldLineLastI:g-1}}let I=[],R=0;for(let v=0;v<o.length;v++){let g=o[v],N,O=l.length-g.oldLines+f,S;for(let b=0;b<=f;b++){S=g.oldStart+R-1;let _=gi(S,c,O);for(;S!==void 0&&(N=m(g.lines,S,b),!N);S=_());if(N)break}if(!N)return!1;for(let b=c;b<S;b++)I.push(l[b]);for(let b=0;b<N.patchedLines.length;b++){let _=N.patchedLines[b];I.push(_)}c=N.oldLineLastI+1,R=S+1-g.oldStart}for(let v=c;v<l.length;v++)I.push(l[v]);return I.join(`\n`)}function yi(i,n){let t=typeof i==\"string\"?ke(i):i,l=0;function o(){let u=t[l++];if(!u)return n.complete();n.loadFile(u,function(f,c){if(f)return n.complete(f);let d=Ln(c,u,n);n.patched(u,d,function(s){if(s)return n.complete(s);o()})})}o()}function wi(i){return i===void 0||i===\"/dev/null\"?i:i.startsWith(\"a/\")?\"b/\"+i.slice(2):i.startsWith(\"b/\")?\"a/\"+i.slice(2):i}function kn(i){if(Array.isArray(i))return i.map(t=>kn(t)).reverse();let n=Object.assign(Object.assign({},i),{oldFileName:i.isGit?wi(i.newFileName):i.newFileName,oldHeader:i.newHeader,newFileName:i.isGit?wi(i.oldFileName):i.oldFileName,newHeader:i.oldHeader,oldMode:i.newMode,newMode:i.oldMode,isCreate:i.isDelete,isDelete:i.isCreate,hunks:i.hunks.map(t=>({oldLines:t.newLines,oldStart:t.newStart,newLines:t.oldLines,newStart:t.oldStart,lines:t.lines.map(l=>l.startsWith(\"-\")?`+${l.slice(1)}`:l.startsWith(\"+\")?`-${l.slice(1)}`:l)}))});return i.isCopy&&(n.newFileName=\"/dev/null\",n.newHeader=void 0,n.isDelete=!0,delete n.isCreate,delete n.isCopy,delete n.isRename,n.hunks=[]),n}function xr(i){for(let n=0;n<i.length;n++)if(i[n]<\" \"||i[n]>\"~\"||i[n]==='\"'||i[n]===\"\\\\\")return!0;return!1}function ae(i){if(!xr(i))return i;let n='\"',t=new TextEncoder().encode(i),l=0;for(;l<t.length;){let o=t[l];o===7?n+=\"\\\\a\":o===8?n+=\"\\\\b\":o===9?n+=\"\\\\t\":o===10?n+=\"\\\\n\":o===11?n+=\"\\\\v\":o===12?n+=\"\\\\f\":o===13?n+=\"\\\\r\":o===34?n+='\\\\\"':o===92?n+=\"\\\\\\\\\":o>=32&&o<=126?n+=String.fromCharCode(o):n+=\"\\\\\"+o.toString(8).padStart(3,\"0\"),l++}return n+='\"',n}var $e={includeIndex:!0,includeUnderline:!0,includeFileHeaders:!0},Ai={includeIndex:!1,includeUnderline:!1,includeFileHeaders:!0},xi={includeIndex:!1,includeUnderline:!1,includeFileHeaders:!1};function Ke(i,n,t,l,o,u,f){let c;f?typeof f==\"function\"?c={callback:f}:c=f:c={},typeof c.context>\"u\"&&(c.context=4);let d=c.context;if(c.newlineIsToken)throw new Error(\"newlineIsToken may not be used with patch-generation functions, only with diffing functions\");if(c.callback){let{callback:a}=c;ee(t,l,Object.assign(Object.assign({},c),{callback:m=>{let I=s(m);a(I)}}))}else return s(ee(t,l,c));function s(a){if(!a)return;a.push({value:\"\",lines:[]});function m(S){return S.map(function(b){return\" \"+b})}let I=[],R=0,v=0,g=[],N=1,O=1;for(let S=0;S<a.length;S++){let b=a[S],_=b.lines||Cr(b.value);if(b.lines=_,b.added||b.removed){if(!R){let P=a[S-1];R=N,v=O,P&&(g=d>0?m(P.lines.slice(-d)):[],R-=g.length,v-=g.length)}for(let P of _)g.push((b.added?\"+\":\"-\")+P);b.added?O+=_.length:N+=_.length}else{if(R)if(_.length<=d*2&&S<a.length-2)for(let P of m(_))g.push(P);else{let P=Math.min(_.length,d);for(let q of m(_.slice(0,P)))g.push(q);let U={oldStart:R,oldLines:N-R+P,newStart:v,newLines:O-v+P,lines:g};I.push(U),R=0,v=0,g=[]}N+=_.length,O+=_.length}}for(let S of I)for(let b=0;b<S.lines.length;b++)S.lines[b].endsWith(`\n`)?S.lines[b]=S.lines[b].slice(0,-1):(S.lines.splice(b+1,0,\"\\\\ No newline at end of file\"),b++);return{oldFileName:i,newFileName:n,oldHeader:o,newHeader:u,hunks:I}}}function Ne(i,n){var t,l,o,u,f,c;if(n||(n=$e),Array.isArray(i)){if(i.length>1&&!n.includeFileHeaders&&!i.every(a=>a.isGit))throw new Error(\"Cannot omit file headers on a multi-file patch. (The result would be unparseable; how would a tool trying to apply the patch know which changes are to which file?)\");return i.map(a=>Ne(a,n)).join(`\n`)}let d=[];if(i.isGit){if(n=$e,!i.oldFileName)throw new Error(\"oldFileName must be specified for Git patches\");if(!i.newFileName)throw new Error(\"newFileName must be specified for Git patches\");let a=i.oldFileName,m=i.newFileName;i.isCreate&&a===\"/dev/null\"?a=m.replace(/^b\\//,\"a/\"):i.isDelete&&m===\"/dev/null\"&&(m=a.replace(/^a\\//,\"b/\")),d.push(\"diff --git \"+ae(a)+\" \"+ae(m)),i.isDelete&&d.push(\"deleted file mode \"+((t=i.oldMode)!==null&&t!==void 0?t:\"100644\")),i.isCreate&&d.push(\"new file mode \"+((l=i.newMode)!==null&&l!==void 0?l:\"100644\")),i.oldMode&&i.newMode&&!i.isDelete&&!i.isCreate&&(d.push(\"old mode \"+i.oldMode),d.push(\"new mode \"+i.newMode)),i.isRename&&(d.push(\"rename from \"+ae(((o=i.oldFileName)!==null&&o!==void 0?o:\"\").replace(/^a\\//,\"\"))),d.push(\"rename to \"+ae(((u=i.newFileName)!==null&&u!==void 0?u:\"\").replace(/^b\\//,\"\")))),i.isCopy&&(d.push(\"copy from \"+ae(((f=i.oldFileName)!==null&&f!==void 0?f:\"\").replace(/^a\\//,\"\"))),d.push(\"copy to \"+ae(((c=i.newFileName)!==null&&c!==void 0?c:\"\").replace(/^b\\//,\"\"))))}else n.includeIndex&&i.oldFileName==i.newFileName&&i.oldFileName!==void 0&&d.push(\"Index: \"+i.oldFileName),n.includeUnderline&&d.push(\"===================================================================\");let s=i.hunks.length>0;n.includeFileHeaders&&i.oldFileName!==void 0&&i.newFileName!==void 0&&(!i.isGit||s)&&(d.push(\"--- \"+ae(i.oldFileName)+(i.oldHeader?\"	\"+i.oldHeader:\"\")),d.push(\"+++ \"+ae(i.newFileName)+(i.newHeader?\"	\"+i.newHeader:\"\")));for(let a=0;a<i.hunks.length;a++){let m=i.hunks[a],I=m.oldLines===0?m.oldStart-1:m.oldStart,R=m.newLines===0?m.newStart-1:m.newStart;d.push(\"@@ -\"+I+\",\"+m.oldLines+\" +\"+R+\",\"+m.newLines+\" @@\");for(let v of m.lines)d.push(v)}return d.join(`\n`)+`\n`}function Nn(i,n,t,l,o,u,f){if(typeof f==\"function\"&&(f={callback:f}),f?.callback){let{callback:c}=f;Ke(i,n,t,l,o,u,Object.assign(Object.assign({},f),{callback:d=>{c(d?Ne(d,f.headerOptions):void 0)}}))}else{let c=Ke(i,n,t,l,o,u,f);return c?Ne(c,f?.headerOptions):void 0}}function Ci(i,n,t,l,o,u){return Nn(i,i,n,t,l,o,u)}function Cr(i){let n=i.endsWith(`\n`),t=i.split(`\n`).map(l=>l+`\n`);return n?t.pop():t.push(t.pop().slice(0,-1)),t}function vi(i){let n=[],t,l;for(let o=0;o<i.length;o++)t=i[o],t.added?l=1:t.removed?l=-1:l=0,n.push([l,t.value]);return n}function Si(i){let n=[];for(let t=0;t<i.length;t++){let l=i[t];l.added?n.push(\"<ins>\"):l.removed&&n.push(\"<del>\"),n.push(vr(l.value)),l.added?n.push(\"</ins>\"):l.removed&&n.push(\"</del>\")}return n.join(\"\")}function vr(i){let n=i;return n=n.replace(/&/g,\"&amp;\"),n=n.replace(/</g,\"&lt;\"),n=n.replace(/>/g,\"&gt;\"),n=n.replace(/\"/g,\"&quot;\"),n}function Sr(i){return i&&i.__esModule&&Object.prototype.hasOwnProperty.call(i,\"default\")?i.default:i}var K={},Je={},de={},bi;function Ie(){if(bi)return de;bi=1;function i(f){return typeof f>\"u\"||f===null}function n(f){return typeof f==\"object\"&&f!==null}function t(f){return Array.isArray(f)?f:i(f)?[]:[f]}function l(f,c){if(c){let d=Object.keys(c);for(let s=0,a=d.length;s<a;s+=1){let m=d[s];f[m]=c[m]}}return f}function o(f,c){let d=\"\";for(let s=0;s<c;s+=1)d+=f;return d}function u(f){return f===0&&Number.NEGATIVE_INFINITY===1/f}return de.isNothing=i,de.isObject=n,de.toArray=t,de.repeat=o,de.isNegativeZero=u,de.extend=l,de}var On,Ei;function Oe(){if(Ei)return On;Ei=1;function i(t,l){let o=\"\",u=t.reason||\"(unknown reason)\";return t.mark?(t.mark.name&&(o+='in \"'+t.mark.name+'\" '),o+=\"(\"+(t.mark.line+1)+\":\"+(t.mark.column+1)+\")\",!l&&t.mark.snippet&&(o+=`\n\n`+t.mark.snippet),u+\" \"+o):u}function n(t,l){Error.call(this),this.name=\"YAMLException\",this.reason=t,this.mark=l,this.message=i(this,!1),Error.captureStackTrace?Error.captureStackTrace(this,this.constructor):this.stack=new Error().stack||\"\"}return n.prototype=Object.create(Error.prototype),n.prototype.constructor=n,n.prototype.toString=function(l){return this.name+\": \"+i(this,l)},On=n,On}var _n,Ti;function br(){if(Ti)return _n;Ti=1;let i=Ie();function n(o,u,f,c,d){let s=\"\",a=\"\",m=Math.floor(d/2)-1;return c-u>m&&(s=\" ... \",u=c-m+s.length),f-c>m&&(a=\" ...\",f=c+m-a.length),{str:s+o.slice(u,f).replace(/\\t/g,\"\\u2192\")+a,pos:c-u+s.length}}function t(o,u){return i.repeat(\" \",u-o.length)+o}function l(o,u){if(u=Object.create(u||null),!o.buffer)return null;u.maxLength||(u.maxLength=79),typeof u.indent!=\"number\"&&(u.indent=1),typeof u.linesBefore!=\"number\"&&(u.linesBefore=3),typeof u.linesAfter!=\"number\"&&(u.linesAfter=2);let f=/\\r?\\n|\\r|\\0/g,c=[0],d=[],s,a=-1;for(;s=f.exec(o.buffer);)d.push(s.index),c.push(s.index+s[0].length),o.position<=s.index&&a<0&&(a=c.length-2);a<0&&(a=c.length-1);let m=\"\",I=Math.min(o.line+u.linesAfter,d.length).toString().length,R=u.maxLength-(u.indent+I+3);for(let g=1;g<=u.linesBefore&&!(a-g<0);g++){let N=n(o.buffer,c[a-g],d[a-g],o.position-(c[a]-c[a-g]),R);m=i.repeat(\" \",u.indent)+t((o.line-g+1).toString(),I)+\" | \"+N.str+`\n`+m}let v=n(o.buffer,c[a],d[a],o.position,R);m+=i.repeat(\" \",u.indent)+t((o.line+1).toString(),I)+\" | \"+v.str+`\n`,m+=i.repeat(\"-\",u.indent+I+3+v.pos)+`^\n`;for(let g=1;g<=u.linesAfter&&!(a+g>=d.length);g++){let N=n(o.buffer,c[a+g],d[a+g],o.position-(c[a]-c[a+g]),R);m+=i.repeat(\" \",u.indent)+t((o.line+g+1).toString(),I)+\" | \"+N.str+`\n`}return m.replace(/\\n$/,\"\")}return _n=l,_n}var Fn,Li;function Q(){if(Li)return Fn;Li=1;let i=Oe(),n=[\"kind\",\"multi\",\"resolve\",\"construct\",\"instanceOf\",\"predicate\",\"represent\",\"representName\",\"defaultStyle\",\"styleAliases\"],t=[\"scalar\",\"sequence\",\"mapping\"];function l(u){let f={};return u!==null&&Object.keys(u).forEach(function(c){u[c].forEach(function(d){f[String(d)]=c})}),f}function o(u,f){if(f=f||{},Object.keys(f).forEach(function(c){if(n.indexOf(c)===-1)throw new i('Unknown option \"'+c+'\" is met in definition of \"'+u+'\" YAML type.')}),this.options=f,this.tag=u,this.kind=f.kind||null,this.resolve=f.resolve||function(){return!0},this.construct=f.construct||function(c){return c},this.instanceOf=f.instanceOf||null,this.predicate=f.predicate||null,this.represent=f.represent||null,this.representName=f.representName||null,this.defaultStyle=f.defaultStyle||null,this.multi=f.multi||!1,this.styleAliases=l(f.styleAliases||null),t.indexOf(this.kind)===-1)throw new i('Unknown kind \"'+this.kind+'\" is specified for \"'+u+'\" YAML type.')}return Fn=o,Fn}var Rn,ki;function Ji(){if(ki)return Rn;ki=1;let i=Oe(),n=Q();function t(u,f){let c=[];return u[f].forEach(function(d){let s=c.length;c.forEach(function(a,m){a.tag===d.tag&&a.kind===d.kind&&a.multi===d.multi&&(s=m)}),c[s]=d}),c}function l(){let u={scalar:{},sequence:{},mapping:{},fallback:{},multi:{scalar:[],sequence:[],mapping:[],fallback:[]}};function f(c){c.multi?(u.multi[c.kind].push(c),u.multi.fallback.push(c)):u[c.kind][c.tag]=u.fallback[c.tag]=c}for(let c=0,d=arguments.length;c<d;c+=1)arguments[c].forEach(f);return u}function o(u){return this.extend(u)}return o.prototype.extend=function(f){let c=[],d=[];if(f instanceof n)d.push(f);else if(Array.isArray(f))d=d.concat(f);else if(f&&(Array.isArray(f.implicit)||Array.isArray(f.explicit)))f.implicit&&(c=c.concat(f.implicit)),f.explicit&&(d=d.concat(f.explicit));else throw new i(\"Schema.extend argument should be a Type, [ Type ], or a schema definition ({ implicit: [...], explicit: [...] })\");c.forEach(function(a){if(!(a instanceof n))throw new i(\"Specified list of YAML types (or a single Type object) contains a non-Type object.\");if(a.loadKind&&a.loadKind!==\"scalar\")throw new i(\"There is a non-scalar type in the implicit list of a schema. Implicit resolving of such types is not supported.\");if(a.multi)throw new i(\"There is a multi type in the implicit list of a schema. Multi tags can only be listed as explicit.\")}),d.forEach(function(a){if(!(a instanceof n))throw new i(\"Specified list of YAML types (or a single Type object) contains a non-Type object.\")});let s=Object.create(o.prototype);return s.implicit=(this.implicit||[]).concat(c),s.explicit=(this.explicit||[]).concat(d),s.compiledImplicit=t(s,\"implicit\"),s.compiledExplicit=t(s,\"explicit\"),s.compiledTypeMap=l(s.compiledImplicit,s.compiledExplicit),s},Rn=o,Rn}var Mn,Ni;function Qi(){if(Ni)return Mn;Ni=1;let i=Q();return Mn=new i(\"tag:yaml.org,2002:str\",{kind:\"scalar\",construct:function(n){return n!==null?n:\"\"}}),Mn}var Dn,Ii;function Xi(){if(Ii)return Dn;Ii=1;let i=Q();return Dn=new i(\"tag:yaml.org,2002:seq\",{kind:\"sequence\",construct:function(n){return n!==null?n:[]}}),Dn}var Pn,Oi;function Vi(){if(Oi)return Pn;Oi=1;let i=Q();return Pn=new i(\"tag:yaml.org,2002:map\",{kind:\"mapping\",construct:function(n){return n!==null?n:{}}}),Pn}var Wn,_i;function Zi(){if(_i)return Wn;_i=1;let i=Ji();return Wn=new i({explicit:[Qi(),Xi(),Vi()]}),Wn}var Hn,Fi;function er(){if(Fi)return Hn;Fi=1;let i=Q();function n(o){if(o===null)return!0;let u=o.length;return u===1&&o===\"~\"||u===4&&(o===\"null\"||o===\"Null\"||o===\"NULL\")}function t(){return null}function l(o){return o===null}return Hn=new i(\"tag:yaml.org,2002:null\",{kind:\"scalar\",resolve:n,construct:t,predicate:l,represent:{canonical:function(){return\"~\"},lowercase:function(){return\"null\"},uppercase:function(){return\"NULL\"},camelcase:function(){return\"Null\"},empty:function(){return\"\"}},defaultStyle:\"lowercase\"}),Hn}var qn,Ri;function nr(){if(Ri)return qn;Ri=1;let i=Q();function n(o){if(o===null)return!1;let u=o.length;return u===4&&(o===\"true\"||o===\"True\"||o===\"TRUE\")||u===5&&(o===\"false\"||o===\"False\"||o===\"FALSE\")}function t(o){return o===\"true\"||o===\"True\"||o===\"TRUE\"}function l(o){return Object.prototype.toString.call(o)===\"[object Boolean]\"}return qn=new i(\"tag:yaml.org,2002:bool\",{kind:\"scalar\",resolve:n,construct:t,predicate:l,represent:{lowercase:function(o){return o?\"true\":\"false\"},uppercase:function(o){return o?\"TRUE\":\"FALSE\"},camelcase:function(o){return o?\"True\":\"False\"}},defaultStyle:\"lowercase\"}),qn}var jn,Mi;function ir(){if(Mi)return jn;Mi=1;let i=Ie(),n=Q();function t(s){return s>=48&&s<=57||s>=65&&s<=70||s>=97&&s<=102}function l(s){return s>=48&&s<=55}function o(s){return s>=48&&s<=57}function u(s){if(s===null)return!1;let a=s.length,m=0,I=!1;if(!a)return!1;let R=s[m];if((R===\"-\"||R===\"+\")&&(R=s[++m]),R===\"0\"){if(m+1===a)return!0;if(R=s[++m],R===\"b\"){for(m++;m<a;m++){if(R=s[m],R!==\"0\"&&R!==\"1\")return!1;I=!0}return I&&isFinite(f(s))}if(R===\"x\"){for(m++;m<a;m++){if(!t(s.charCodeAt(m)))return!1;I=!0}return I&&isFinite(f(s))}if(R===\"o\"){for(m++;m<a;m++){if(!l(s.charCodeAt(m)))return!1;I=!0}return I&&isFinite(f(s))}}for(;m<a;m++){if(!o(s.charCodeAt(m)))return!1;I=!0}return I?isFinite(f(s)):!1}function f(s){let a=s,m=1,I=a[0];if((I===\"-\"||I===\"+\")&&(I===\"-\"&&(m=-1),a=a.slice(1),I=a[0]),a===\"0\")return 0;if(I===\"0\"){if(a[1]===\"b\")return m*parseInt(a.slice(2),2);if(a[1]===\"x\")return m*parseInt(a.slice(2),16);if(a[1]===\"o\")return m*parseInt(a.slice(2),8)}return m*parseInt(a,10)}function c(s){return f(s)}function d(s){return Object.prototype.toString.call(s)===\"[object Number]\"&&s%1===0&&!i.isNegativeZero(s)}return jn=new n(\"tag:yaml.org,2002:int\",{kind:\"scalar\",resolve:u,construct:c,predicate:d,represent:{binary:function(s){return s>=0?\"0b\"+s.toString(2):\"-0b\"+s.toString(2).slice(1)},octal:function(s){return s>=0?\"0o\"+s.toString(8):\"-0o\"+s.toString(8).slice(1)},decimal:function(s){return s.toString(10)},hexadecimal:function(s){return s>=0?\"0x\"+s.toString(16).toUpperCase():\"-0x\"+s.toString(16).toUpperCase().slice(1)}},defaultStyle:\"decimal\",styleAliases:{binary:[2,\"bin\"],octal:[8,\"oct\"],decimal:[10,\"dec\"],hexadecimal:[16,\"hex\"]}}),jn}var Yn,Di;function rr(){if(Di)return Yn;Di=1;let i=Ie(),n=Q(),t=new RegExp(\"^(?:[-+]?(?:[0-9]+)(?:\\\\.[0-9]*)?(?:[eE][-+]?[0-9]+)?|\\\\.[0-9]+(?:[eE][-+]?[0-9]+)?|[-+]?\\\\.(?:inf|Inf|INF)|\\\\.(?:nan|NaN|NAN))$\"),l=new RegExp(\"^(?:[-+]?\\\\.(?:inf|Inf|INF)|\\\\.(?:nan|NaN|NAN))$\");function o(s){return s===null||!t.test(s)?!1:isFinite(parseFloat(s,10))?!0:l.test(s)}function u(s){let a=s.toLowerCase(),m=a[0]===\"-\"?-1:1;return\"+-\".indexOf(a[0])>=0&&(a=a.slice(1)),a===\".inf\"?m===1?Number.POSITIVE_INFINITY:Number.NEGATIVE_INFINITY:a===\".nan\"?NaN:m*parseFloat(a,10)}let f=/^[-+]?[0-9]+e/;function c(s,a){if(isNaN(s))switch(a){case\"lowercase\":return\".nan\";case\"uppercase\":return\".NAN\";case\"camelcase\":return\".NaN\"}else if(Number.POSITIVE_INFINITY===s)switch(a){case\"lowercase\":return\".inf\";case\"uppercase\":return\".INF\";case\"camelcase\":return\".Inf\"}else if(Number.NEGATIVE_INFINITY===s)switch(a){case\"lowercase\":return\"-.inf\";case\"uppercase\":return\"-.INF\";case\"camelcase\":return\"-.Inf\"}else if(i.isNegativeZero(s))return\"-0.0\";let m=s.toString(10);return f.test(m)?m.replace(\"e\",\".e\"):m}function d(s){return Object.prototype.toString.call(s)===\"[object Number]\"&&(s%1!==0||i.isNegativeZero(s))}return Yn=new n(\"tag:yaml.org,2002:float\",{kind:\"scalar\",resolve:o,construct:u,predicate:d,represent:c,defaultStyle:\"lowercase\"}),Yn}var Un,Pi;function tr(){return Pi||(Pi=1,Un=Zi().extend({implicit:[er(),nr(),ir(),rr()]})),Un}var Bn,Wi;function lr(){return Wi||(Wi=1,Bn=tr()),Bn}var Gn,Hi;function or(){if(Hi)return Gn;Hi=1;let i=Q(),n=new RegExp(\"^([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])$\"),t=new RegExp(\"^([0-9][0-9][0-9][0-9])-([0-9][0-9]?)-([0-9][0-9]?)(?:[Tt]|[ \\\\t]+)([0-9][0-9]?):([0-9][0-9]):([0-9][0-9])(?:\\\\.([0-9]*))?(?:[ \\\\t]*(Z|([-+])([0-9][0-9]?)(?::([0-9][0-9]))?))?$\");function l(f){return f===null?!1:n.exec(f)!==null||t.exec(f)!==null}function o(f){let c=0,d=null,s=n.exec(f);if(s===null&&(s=t.exec(f)),s===null)throw new Error(\"Date resolve error\");let a=+s[1],m=+s[2]-1,I=+s[3];if(!s[4])return new Date(Date.UTC(a,m,I));let R=+s[4],v=+s[5],g=+s[6];if(s[7]){for(c=s[7].slice(0,3);c.length<3;)c+=\"0\";c=+c}if(s[9]){let O=+s[10],S=+(s[11]||0);d=(O*60+S)*6e4,s[9]===\"-\"&&(d=-d)}let N=new Date(Date.UTC(a,m,I,R,v,g,c));return d&&N.setTime(N.getTime()-d),N}function u(f){return f.toISOString()}return Gn=new i(\"tag:yaml.org,2002:timestamp\",{kind:\"scalar\",resolve:l,construct:o,instanceOf:Date,represent:u}),Gn}var zn,qi;function ur(){if(qi)return zn;qi=1;let i=Q();function n(t){return t===\"<<\"||t===null}return zn=new i(\"tag:yaml.org,2002:merge\",{kind:\"scalar\",resolve:n}),zn}var $n,ji;function fr(){if(ji)return $n;ji=1;let i=Q(),n=`ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\\r`;function t(f){if(f===null)return!1;let c=0,d=f.length,s=n;for(let a=0;a<d;a++){let m=s.indexOf(f.charAt(a));if(!(m>64)){if(m<0)return!1;c+=6}}return c%8===0}function l(f){let c=f.replace(/[\\r\\n=]/g,\"\"),d=c.length,s=n,a=0,m=[];for(let R=0;R<d;R++)R%4===0&&R&&(m.push(a>>16&255),m.push(a>>8&255),m.push(a&255)),a=a<<6|s.indexOf(c.charAt(R));let I=d%4*6;return I===0?(m.push(a>>16&255),m.push(a>>8&255),m.push(a&255)):I===18?(m.push(a>>10&255),m.push(a>>2&255)):I===12&&m.push(a>>4&255),new Uint8Array(m)}function o(f){let c=\"\",d=0,s=f.length,a=n;for(let I=0;I<s;I++)I%3===0&&I&&(c+=a[d>>18&63],c+=a[d>>12&63],c+=a[d>>6&63],c+=a[d&63]),d=(d<<8)+f[I];let m=s%3;return m===0?(c+=a[d>>18&63],c+=a[d>>12&63],c+=a[d>>6&63],c+=a[d&63]):m===2?(c+=a[d>>10&63],c+=a[d>>4&63],c+=a[d<<2&63],c+=a[64]):m===1&&(c+=a[d>>2&63],c+=a[d<<4&63],c+=a[64],c+=a[64]),c}function u(f){return Object.prototype.toString.call(f)===\"[object Uint8Array]\"}return $n=new i(\"tag:yaml.org,2002:binary\",{kind:\"scalar\",resolve:t,construct:l,predicate:u,represent:o}),$n}var Kn,Yi;function cr(){if(Yi)return Kn;Yi=1;let i=Q(),n=Object.prototype.hasOwnProperty,t=Object.prototype.toString;function l(u){if(u===null)return!0;let f=[],c=u;for(let d=0,s=c.length;d<s;d+=1){let a=c[d],m=!1;if(t.call(a)!==\"[object Object]\")return!1;let I;for(I in a)if(n.call(a,I))if(!m)m=!0;else return!1;if(!m)return!1;if(f.indexOf(I)===-1)f.push(I);else return!1}return!0}function o(u){return u!==null?u:[]}return Kn=new i(\"tag:yaml.org,2002:omap\",{kind:\"sequence\",resolve:l,construct:o}),Kn}var Jn,Ui;function sr(){if(Ui)return Jn;Ui=1;let i=Q(),n=Object.prototype.toString;function t(o){if(o===null)return!0;let u=o,f=new Array(u.length);for(let c=0,d=u.length;c<d;c+=1){let s=u[c];if(n.call(s)!==\"[object Object]\")return!1;let a=Object.keys(s);if(a.length!==1)return!1;f[c]=[a[0],s[a[0]]]}return!0}function l(o){if(o===null)return[];let u=o,f=new Array(u.length);for(let c=0,d=u.length;c<d;c+=1){let s=u[c],a=Object.keys(s);f[c]=[a[0],s[a[0]]]}return f}return Jn=new i(\"tag:yaml.org,2002:pairs\",{kind:\"sequence\",resolve:t,construct:l}),Jn}var Qn,Bi;function ar(){if(Bi)return Qn;Bi=1;let i=Q(),n=Object.prototype.hasOwnProperty;function t(o){if(o===null)return!0;let u=o;for(let f in u)if(n.call(u,f)&&u[f]!==null)return!1;return!0}function l(o){return o!==null?o:{}}return Qn=new i(\"tag:yaml.org,2002:set\",{kind:\"mapping\",resolve:t,construct:l}),Qn}var Xn,Gi;function Zn(){return Gi||(Gi=1,Xn=lr().extend({implicit:[or(),ur()],explicit:[fr(),cr(),sr(),ar()]})),Xn}var zi;function Er(){if(zi)return Je;zi=1;let i=Ie(),n=Oe(),t=br(),l=Zn(),o=Object.prototype.hasOwnProperty,u=1,f=2,c=3,d=4,s=1,a=2,m=3,I=/[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F-\\x84\\x86-\\x9F\\uFFFE\\uFFFF]|[\\uD800-\\uDBFF](?![\\uDC00-\\uDFFF])|(?:[^\\uD800-\\uDBFF]|^)[\\uDC00-\\uDFFF]/,R=/[\\x85\\u2028\\u2029]/,v=/[,\\[\\]{}]/,g=/^(?:!|!!|![0-9A-Za-z-]+!)$/,N=/^(?:!|[^,\\[\\]{}])(?:%[0-9a-f]{2}|[0-9a-z\\-#;/?:@&=+$,_.!~*'()\\[\\]])*$/i;function O(e){return Object.prototype.toString.call(e)}function S(e){return e===10||e===13}function b(e){return e===9||e===32}function _(e){return e===9||e===32||e===10||e===13}function P(e){return e===44||e===91||e===93||e===123||e===125}function U(e){if(e>=48&&e<=57)return e-48;let p=e|32;return p>=97&&p<=102?p-97+10:-1}function q(e){return e===120?2:e===117?4:e===85?8:0}function Y(e){return e>=48&&e<=57?e-48:-1}function J(e){switch(e){case 48:return\"\\0\";case 97:return\"\\x07\";case 98:return\"\\b\";case 116:return\"	\";case 9:return\"	\";case 110:return`\n`;case 118:return\"\\v\";case 102:return\"\\f\";case 114:return\"\\r\";case 101:return\"\\x1B\";case 32:return\" \";case 34:return'\"';case 47:return\"/\";case 92:return\"\\\\\";case 78:return\"\\x85\";case 95:return\"\\xA0\";case 76:return\"\\u2028\";case 80:return\"\\u2029\";default:return\"\"}}function re(e){return e<=65535?String.fromCharCode(e):String.fromCharCode((e-65536>>10)+55296,(e-65536&1023)+56320)}function ie(e,p,w){p===\"__proto__\"?Object.defineProperty(e,p,{configurable:!0,enumerable:!0,writable:!0,value:w}):e[p]=w}let me=new Array(256),X=new Array(256);for(let e=0;e<256;e++)me[e]=J(e)?1:0,X[e]=J(e);function z(e,p){this.input=e,this.filename=p.filename||null,this.schema=p.schema||l,this.onWarning=p.onWarning||null,this.legacy=p.legacy||!1,this.json=p.json||!1,this.listener=p.listener||null,this.maxDepth=typeof p.maxDepth==\"number\"?p.maxDepth:100,this.maxTotalMergeKeys=typeof p.maxTotalMergeKeys==\"number\"?p.maxTotalMergeKeys:1e4,this.implicitTypes=this.schema.compiledImplicit,this.typeMap=this.schema.compiledTypeMap,this.length=e.length,this.position=0,this.line=0,this.lineStart=0,this.lineIndent=0,this.depth=0,this.totalMergeKeys=0,this.firstTabInLine=-1,this.documents=[],this.anchorMapTransactions=[]}function ge(e,p){let w={name:e.filename,buffer:e.input.slice(0,-1),position:e.position,line:e.line,column:e.position-e.lineStart};return w.snippet=t(w),new n(p,w)}function D(e,p){throw ge(e,p)}function ce(e,p){e.onWarning&&e.onWarning.call(null,ge(e,p))}function Z(e,p,w){let C=e.anchorMapTransactions;if(C.length!==0){let y=C[C.length-1];o.call(y,p)||(y[p]={existed:o.call(e.anchorMap,p),value:e.anchorMap[p]})}e.anchorMap[p]=w}function Xe(e){e.anchorMapTransactions.push(Object.create(null))}function pe(e){let p=e.anchorMapTransactions.pop(),w=e.anchorMapTransactions;if(w.length===0)return;let C=w[w.length-1],y=Object.keys(p);for(let k=0,r=y.length;k<r;k+=1){let h=y[k];o.call(C,h)||(C[h]=p[h])}}function Ve(e){let p=e.anchorMapTransactions.pop(),w=Object.keys(p);for(let C=w.length-1;C>=0;C-=1){let y=p[w[C]];y.existed?e.anchorMap[w[C]]=y.value:delete e.anchorMap[w[C]]}}function ve(e){return{position:e.position,line:e.line,lineStart:e.lineStart,lineIndent:e.lineIndent,firstTabInLine:e.firstTabInLine,tag:e.tag,anchor:e.anchor,kind:e.kind,result:e.result}}function ye(e,p){e.position=p.position,e.line=p.line,e.lineStart=p.lineStart,e.lineIndent=p.lineIndent,e.firstTabInLine=p.firstTabInLine,e.tag=p.tag,e.anchor=p.anchor,e.kind=p.kind,e.result=p.result}let _e={YAML:function(p,w,C){p.version!==null&&D(p,\"duplication of %YAML directive\"),C.length!==1&&D(p,\"YAML directive accepts exactly one argument\");let y=/^([0-9]+)\\.([0-9]+)$/.exec(C[0]);y===null&&D(p,\"ill-formed argument of the YAML directive\");let k=parseInt(y[1],10),r=parseInt(y[2],10);k!==1&&D(p,\"unacceptable YAML version of the document\"),p.version=C[0],p.checkLineBreaks=r<2,r!==1&&r!==2&&ce(p,\"unsupported YAML version of the document\")},TAG:function(p,w,C){let y;C.length!==2&&D(p,\"TAG directive accepts exactly two arguments\");let k=C[0];y=C[1],g.test(k)||D(p,\"ill-formed tag handle (first argument) of the TAG directive\"),o.call(p.tagMap,k)&&D(p,'there is a previously declared suffix for \"'+k+'\" tag handle'),N.test(y)||D(p,\"ill-formed tag prefix (second argument) of the TAG directive\");try{y=decodeURIComponent(y)}catch{D(p,\"tag prefix is malformed: \"+y)}p.tagMap[k]=y}};function V(e,p,w,C){if(p<w){let y=e.input.slice(p,w);if(C)for(let k=0,r=y.length;k<r;k+=1){let h=y.charCodeAt(k);h===9||h>=32&&h<=1114111||D(e,\"expected valid JSON character\")}else I.test(y)&&D(e,\"the stream contains non-printable characters\");e.result+=y}}function se(e,p,w,C){i.isObject(w)||D(e,\"cannot merge mappings; the provided source object is unacceptable\");let y=Object.keys(w);for(let k=0,r=y.length;k<r;k+=1){let h=y[k];e.maxTotalMergeKeys!==-1&&++e.totalMergeKeys>e.maxTotalMergeKeys&&D(e,\"merge keys exceeded maxTotalMergeKeys (\"+e.maxTotalMergeKeys+\")\"),o.call(p,h)||(ie(p,h,w[h]),C[h]=!0)}}function te(e,p,w,C,y,k,r,h,T){if(Array.isArray(y)){y=Array.prototype.slice.call(y);for(let A=0,x=y.length;A<x;A+=1)Array.isArray(y[A])&&D(e,\"nested arrays are not supported inside keys\"),typeof y==\"object\"&&O(y[A])===\"[object Object]\"&&(y[A]=\"[object Object]\")}if(typeof y==\"object\"&&O(y)===\"[object Object]\"&&(y=\"[object Object]\"),y=String(y),p===null&&(p={}),C===\"tag:yaml.org,2002:merge\")if(Array.isArray(k))for(let A=0,x=k.length;A<x;A+=1)se(e,p,k[A],w);else se(e,p,k,w);else!e.json&&!o.call(w,y)&&o.call(p,y)&&(e.line=r||e.line,e.lineStart=h||e.lineStart,e.position=T||e.position,D(e,\"duplicated mapping key\")),ie(p,y,k),delete w[y];return p}function we(e){let p=e.input.charCodeAt(e.position);p===10?e.position++:p===13?(e.position++,e.input.charCodeAt(e.position)===10&&e.position++):D(e,\"a line break is expected\"),e.line+=1,e.lineStart=e.position,e.firstTabInLine=-1}function G(e,p,w){let C=0,y=e.input.charCodeAt(e.position);for(;y!==0;){for(;b(y);)y===9&&e.firstTabInLine===-1&&(e.firstTabInLine=e.position),y=e.input.charCodeAt(++e.position);if(p&&y===35)do y=e.input.charCodeAt(++e.position);while(y!==10&&y!==13&&y!==0);if(S(y))for(we(e),y=e.input.charCodeAt(e.position),C++,e.lineIndent=0;y===32;)e.lineIndent++,y=e.input.charCodeAt(++e.position);else break}return w!==-1&&C!==0&&e.lineIndent<w&&ce(e,\"deficient indentation\"),C}function Ae(e){let p=e.position,w=e.input.charCodeAt(p);return!!((w===45||w===46)&&w===e.input.charCodeAt(p+1)&&w===e.input.charCodeAt(p+2)&&(p+=3,w=e.input.charCodeAt(p),w===0||_(w)))}function le(e,p){p===1?e.result+=\" \":p>1&&(e.result+=i.repeat(`\n`,p-1))}function Fe(e,p,w){let C,y,k,r,h,T,A=e.kind,x=e.result,L=e.input.charCodeAt(e.position);if(_(L)||P(L)||L===35||L===38||L===42||L===33||L===124||L===62||L===39||L===34||L===37||L===64||L===96)return!1;if(L===63||L===45){let E=e.input.charCodeAt(e.position+1);if(_(E)||w&&P(E))return!1}for(e.kind=\"scalar\",e.result=\"\",C=y=e.position,k=!1;L!==0;){if(L===58){let E=e.input.charCodeAt(e.position+1);if(_(E)||w&&P(E))break}else if(L===35){let E=e.input.charCodeAt(e.position-1);if(_(E))break}else{if(e.position===e.lineStart&&Ae(e)||w&&P(L))break;if(S(L))if(r=e.line,h=e.lineStart,T=e.lineIndent,G(e,!1,-1),e.lineIndent>=p){k=!0,L=e.input.charCodeAt(e.position);continue}else{e.position=y,e.line=r,e.lineStart=h,e.lineIndent=T;break}}k&&(V(e,C,y,!1),le(e,e.line-r),C=y=e.position,k=!1),b(L)||(y=e.position+1),L=e.input.charCodeAt(++e.position)}return V(e,C,y,!1),e.result?!0:(e.kind=A,e.result=x,!1)}function Re(e,p){let w,C,y=e.input.charCodeAt(e.position);if(y!==39)return!1;for(e.kind=\"scalar\",e.result=\"\",e.position++,w=C=e.position;(y=e.input.charCodeAt(e.position))!==0;)if(y===39)if(V(e,w,e.position,!0),y=e.input.charCodeAt(++e.position),y===39)w=e.position,e.position++,C=e.position;else return!0;else S(y)?(V(e,w,C,!0),le(e,G(e,!1,p)),w=C=e.position):e.position===e.lineStart&&Ae(e)?D(e,\"unexpected end of the document within a single quoted scalar\"):(e.position++,b(y)||(C=e.position));D(e,\"unexpected end of the stream within a single quoted scalar\")}function Se(e,p){let w,C,y,k=e.input.charCodeAt(e.position);if(k!==34)return!1;for(e.kind=\"scalar\",e.result=\"\",e.position++,w=C=e.position;(k=e.input.charCodeAt(e.position))!==0;){if(k===34)return V(e,w,e.position,!0),e.position++,!0;if(k===92){if(V(e,w,e.position,!0),k=e.input.charCodeAt(++e.position),S(k))G(e,!1,p);else if(k<256&&me[k])e.result+=X[k],e.position++;else if((y=q(k))>0){let r=y,h=0;for(;r>0;r--)k=e.input.charCodeAt(++e.position),(y=U(k))>=0?h=(h<<4)+y:D(e,\"expected hexadecimal character\");e.result+=re(h),e.position++}else D(e,\"unknown escape sequence\");w=C=e.position}else S(k)?(V(e,w,C,!0),le(e,G(e,!1,p)),w=C=e.position):e.position===e.lineStart&&Ae(e)?D(e,\"unexpected end of the document within a double quoted scalar\"):(e.position++,b(k)||(C=e.position))}D(e,\"unexpected end of the stream within a double quoted scalar\")}function Me(e,p){let w=!0,C,y,k,r=e.tag,h,T=e.anchor,A,x,L,E,M=Object.create(null),F,W,H,j=e.input.charCodeAt(e.position);if(j===91)A=93,E=!1,h=[];else if(j===123)A=125,E=!0,h={};else return!1;for(e.anchor!==null&&Z(e,e.anchor,h),j=e.input.charCodeAt(++e.position);j!==0;){if(G(e,!0,p),j=e.input.charCodeAt(e.position),j===A)return e.position++,e.tag=r,e.anchor=T,e.kind=E?\"mapping\":\"sequence\",e.result=h,!0;if(w?j===44&&D(e,\"expected the node content, but found ','\"):D(e,\"missed comma between flow collection entries\"),W=F=H=null,x=L=!1,j===63){let B=e.input.charCodeAt(e.position+1);_(B)&&(x=L=!0,e.position++,G(e,!0,p))}C=e.line,y=e.lineStart,k=e.position,ue(e,p,u,!1,!0),W=e.tag,F=e.result,G(e,!0,p),j=e.input.charCodeAt(e.position),(L||e.line===C)&&j===58&&(x=!0,j=e.input.charCodeAt(++e.position),G(e,!0,p),ue(e,p,u,!1,!0),H=e.result),E?te(e,h,M,W,F,H,C,y,k):x?h.push(te(e,null,M,W,F,H,C,y,k)):h.push(F),G(e,!0,p),j=e.input.charCodeAt(e.position),j===44?(w=!0,j=e.input.charCodeAt(++e.position)):w=!1}D(e,\"unexpected end of the stream within a flow collection\")}function De(e,p){let w,C=s,y=!1,k=!1,r=p,h=0,T=!1,A,x=e.input.charCodeAt(e.position);if(x===124)w=!1;else if(x===62)w=!0;else return!1;for(e.kind=\"scalar\",e.result=\"\";x!==0;)if(x=e.input.charCodeAt(++e.position),x===43||x===45)s===C?C=x===43?m:a:D(e,\"repeat of a chomping mode identifier\");else if((A=Y(x))>=0)A===0?D(e,\"bad explicit indentation width of a block scalar; it cannot be less than one\"):k?D(e,\"repeat of an indentation width identifier\"):(r=p+A-1,k=!0);else break;if(b(x)){do x=e.input.charCodeAt(++e.position);while(b(x));if(x===35)do x=e.input.charCodeAt(++e.position);while(!S(x)&&x!==0)}for(;x!==0;){for(we(e),e.lineIndent=0,x=e.input.charCodeAt(e.position);(!k||e.lineIndent<r)&&x===32;)e.lineIndent++,x=e.input.charCodeAt(++e.position);if(!k&&e.lineIndent>r&&(r=e.lineIndent),S(x)){h++;continue}if(!k&&r===0&&D(e,\"missing indentation for block scalar\"),e.lineIndent<r){C===m?e.result+=i.repeat(`\n`,y?1+h:h):C===s&&y&&(e.result+=`\n`);break}w?b(x)?(T=!0,e.result+=i.repeat(`\n`,y?1+h:h)):T?(T=!1,e.result+=i.repeat(`\n`,h+1)):h===0?y&&(e.result+=\" \"):e.result+=i.repeat(`\n`,h):e.result+=i.repeat(`\n`,y?1+h:h),y=!0,k=!0,h=0;let L=e.position;for(;!S(x)&&x!==0;)x=e.input.charCodeAt(++e.position);V(e,L,e.position,!1)}return!0}function oe(e,p){let w=e.tag,C=e.anchor,y=[],k=!1;if(e.firstTabInLine!==-1)return!1;e.anchor!==null&&Z(e,e.anchor,y);let r=e.input.charCodeAt(e.position);for(;r!==0&&(e.firstTabInLine!==-1&&(e.position=e.firstTabInLine,D(e,\"tab characters must not be used in indentation\")),r===45);){let h=e.input.charCodeAt(e.position+1);if(!_(h))break;if(k=!0,e.position++,G(e,!0,-1)&&e.lineIndent<=p){y.push(null),r=e.input.charCodeAt(e.position);continue}let T=e.line;if(ue(e,p,c,!1,!0),y.push(e.result),G(e,!0,-1),r=e.input.charCodeAt(e.position),(e.line===T||e.lineIndent>p)&&r!==0)D(e,\"bad indentation of a sequence entry\");else if(e.lineIndent<p)break}return k?(e.tag=w,e.anchor=C,e.kind=\"sequence\",e.result=y,!0):!1}function Pe(e,p,w){let C,y,k,r,h=e.tag,T=e.anchor,A={},x=Object.create(null),L=null,E=null,M=null,F=!1,W=!1;if(e.firstTabInLine!==-1)return!1;e.anchor!==null&&Z(e,e.anchor,A);let H=e.input.charCodeAt(e.position);for(;H!==0;){!F&&e.firstTabInLine!==-1&&(e.position=e.firstTabInLine,D(e,\"tab characters must not be used in indentation\"));let j=e.input.charCodeAt(e.position+1),B=e.line;if((H===63||H===58)&&_(j))H===63?(F&&(te(e,A,x,L,E,null,y,k,r),L=E=M=null),W=!0,F=!0,C=!0):F?(F=!1,C=!0):D(e,\"incomplete explicit mapping pair; a key node is missed; or followed by a non-tabulated empty line\"),e.position+=1,H=j;else{if(y=e.line,k=e.lineStart,r=e.position,!ue(e,w,f,!1,!0))break;if(e.line===B){for(H=e.input.charCodeAt(e.position);b(H);)H=e.input.charCodeAt(++e.position);if(H===58)H=e.input.charCodeAt(++e.position),_(H)||D(e,\"a whitespace character is expected after the key-value separator within a block mapping\"),F&&(te(e,A,x,L,E,null,y,k,r),L=E=M=null),W=!0,F=!1,C=!1,L=e.tag,E=e.result;else if(W)D(e,\"can not read an implicit mapping pair; a colon is missed\");else return e.tag=h,e.anchor=T,!0}else if(W)D(e,\"can not read a block mapping entry; a multiline key may not be an implicit key\");else return e.tag=h,e.anchor=T,!0}if((e.line===B||e.lineIndent>p)&&(F&&(y=e.line,k=e.lineStart,r=e.position),ue(e,p,d,!0,C)&&(F?E=e.result:M=e.result),F||(te(e,A,x,L,E,M,y,k,r),L=E=M=null),G(e,!0,-1),H=e.input.charCodeAt(e.position)),(e.line===B||e.lineIndent>p)&&H!==0)D(e,\"bad indentation of a mapping entry\");else if(e.lineIndent<p)break}return F&&te(e,A,x,L,E,null,y,k,r),W&&(e.tag=h,e.anchor=T,e.kind=\"mapping\",e.result=A),W}function Ze(e){let p=!1,w=!1,C,y,k=e.input.charCodeAt(e.position);if(k!==33)return!1;e.tag!==null&&D(e,\"duplication of a tag property\"),k=e.input.charCodeAt(++e.position),k===60?(p=!0,k=e.input.charCodeAt(++e.position)):k===33?(w=!0,C=\"!!\",k=e.input.charCodeAt(++e.position)):C=\"!\";let r=e.position;if(p){do k=e.input.charCodeAt(++e.position);while(k!==0&&k!==62);e.position<e.length?(y=e.input.slice(r,e.position),k=e.input.charCodeAt(++e.position)):D(e,\"unexpected end of the stream within a verbatim tag\")}else{for(;k!==0&&!_(k);)k===33&&(w?D(e,\"tag suffix cannot contain exclamation marks\"):(C=e.input.slice(r-1,e.position+1),g.test(C)||D(e,\"named tag handle cannot contain such characters\"),w=!0,r=e.position+1)),k=e.input.charCodeAt(++e.position);y=e.input.slice(r,e.position),v.test(y)&&D(e,\"tag suffix cannot contain flow indicator characters\")}y&&!N.test(y)&&D(e,\"tag name cannot contain such characters: \"+y);try{y=decodeURIComponent(y)}catch{D(e,\"tag name is malformed: \"+y)}return p?e.tag=y:o.call(e.tagMap,C)?e.tag=e.tagMap[C]+y:C===\"!\"?e.tag=\"!\"+y:C===\"!!\"?e.tag=\"tag:yaml.org,2002:\"+y:D(e,'undeclared tag handle \"'+C+'\"'),!0}function We(e){let p=e.input.charCodeAt(e.position);if(p!==38)return!1;e.anchor!==null&&D(e,\"duplication of an anchor property\"),p=e.input.charCodeAt(++e.position);let w=e.position;for(;p!==0&&!_(p)&&!P(p);)p=e.input.charCodeAt(++e.position);return e.position===w&&D(e,\"name of an anchor node must contain at least one character\"),e.anchor=e.input.slice(w,e.position),!0}function He(e){let p=e.input.charCodeAt(e.position);if(p!==42)return!1;p=e.input.charCodeAt(++e.position);let w=e.position;for(;p!==0&&!_(p)&&!P(p);)p=e.input.charCodeAt(++e.position);e.position===w&&D(e,\"name of an alias node must contain at least one character\");let C=e.input.slice(w,e.position);return o.call(e.anchorMap,C)||D(e,'unidentified alias \"'+C+'\"'),e.result=e.anchorMap[C],G(e,!0,-1),!0}function en(e,p,w,C){let y=ve(e);return Xe(e),ye(e,p),e.tag=null,e.anchor=null,e.kind=null,e.result=null,Pe(e,w,C)&&e.kind===\"mapping\"?(pe(e),!0):(Ve(e),ye(e,y),!1)}function ue(e,p,w,C,y){let k,r,h=1,T=!1,A=!1,x=null,L,E,M;e.depth>=e.maxDepth&&D(e,\"nesting exceeded maxDepth (\"+e.maxDepth+\")\"),e.depth+=1,e.listener!==null&&e.listener(\"open\",e),e.tag=null,e.anchor=null,e.kind=null,e.result=null;let F=k=r=d===w||c===w;if(C&&G(e,!0,-1)&&(T=!0,e.lineIndent>p?h=1:e.lineIndent===p?h=0:e.lineIndent<p&&(h=-1)),h===1)for(;;){let W=e.input.charCodeAt(e.position),H=ve(e);if(T&&(W===33&&e.tag!==null||W===38&&e.anchor!==null)||!Ze(e)&&!We(e))break;x===null&&(x=H),G(e,!0,-1)?(T=!0,r=F,e.lineIndent>p?h=1:e.lineIndent===p?h=0:e.lineIndent<p&&(h=-1)):r=!1}if(r&&(r=T||y),h===1||d===w)if(u===w||f===w?E=p:E=p+1,M=e.position-e.lineStart,h===1)if(r&&(oe(e,M)||Pe(e,M,E))||Me(e,E))A=!0;else{let W=e.input.charCodeAt(e.position);x!==null&&F&&!r&&W!==124&&W!==62&&en(e,x,x.position-x.lineStart,E)||k&&De(e,E)||Re(e,E)||Se(e,E)?A=!0:He(e)?(A=!0,(e.tag!==null||e.anchor!==null)&&D(e,\"alias node should not have any properties\")):Fe(e,E,u===w)&&(A=!0,e.tag===null&&(e.tag=\"?\")),e.anchor!==null&&Z(e,e.anchor,e.result)}else h===0&&(A=r&&oe(e,M));if(e.tag===null)e.anchor!==null&&Z(e,e.anchor,e.result);else if(e.tag===\"?\"){e.result!==null&&e.kind!==\"scalar\"&&D(e,'unacceptable node kind for !<?> tag; it should be \"scalar\", not \"'+e.kind+'\"');for(let W=0,H=e.implicitTypes.length;W<H;W+=1)if(L=e.implicitTypes[W],L.resolve(e.result)){e.result=L.construct(e.result),e.tag=L.tag,e.anchor!==null&&Z(e,e.anchor,e.result);break}}else if(e.tag!==\"!\"){if(o.call(e.typeMap[e.kind||\"fallback\"],e.tag))L=e.typeMap[e.kind||\"fallback\"][e.tag];else{L=null;let W=e.typeMap.multi[e.kind||\"fallback\"];for(let H=0,j=W.length;H<j;H+=1)if(e.tag.slice(0,W[H].tag.length)===W[H].tag){L=W[H];break}}L||D(e,\"unknown tag !<\"+e.tag+\">\"),e.result!==null&&L.kind!==e.kind&&D(e,\"unacceptable node kind for !<\"+e.tag+'> tag; it should be \"'+L.kind+'\", not \"'+e.kind+'\"'),L.resolve(e.result,e.tag)?(e.result=L.construct(e.result,e.tag),e.anchor!==null&&Z(e,e.anchor,e.result)):D(e,\"cannot resolve a node with !<\"+e.tag+\"> explicit tag\")}return e.listener!==null&&e.listener(\"close\",e),e.depth-=1,e.tag!==null||e.anchor!==null||A}function nn(e){let p=e.position,w=!1,C;for(e.version=null,e.checkLineBreaks=e.legacy,e.tagMap=Object.create(null),e.anchorMap=Object.create(null);(C=e.input.charCodeAt(e.position))!==0&&(G(e,!0,-1),C=e.input.charCodeAt(e.position),!(e.lineIndent>0||C!==37));){w=!0,C=e.input.charCodeAt(++e.position);let y=e.position;for(;C!==0&&!_(C);)C=e.input.charCodeAt(++e.position);let k=e.input.slice(y,e.position),r=[];for(k.length<1&&D(e,\"directive name must not be less than one character in length\");C!==0;){for(;b(C);)C=e.input.charCodeAt(++e.position);if(C===35){do C=e.input.charCodeAt(++e.position);while(C!==0&&!S(C));break}if(S(C))break;for(y=e.position;C!==0&&!_(C);)C=e.input.charCodeAt(++e.position);r.push(e.input.slice(y,e.position))}C!==0&&we(e),o.call(_e,k)?_e[k](e,k,r):ce(e,'unknown document directive \"'+k+'\"')}if(G(e,!0,-1),e.lineIndent===0&&e.input.charCodeAt(e.position)===45&&e.input.charCodeAt(e.position+1)===45&&e.input.charCodeAt(e.position+2)===45?(e.position+=3,G(e,!0,-1)):w&&D(e,\"directives end mark is expected\"),ue(e,e.lineIndent-1,d,!1,!0),G(e,!0,-1),e.checkLineBreaks&&R.test(e.input.slice(p,e.position))&&ce(e,\"non-ASCII line breaks are interpreted as content\"),e.documents.push(e.result),e.position===e.lineStart&&Ae(e)){e.input.charCodeAt(e.position)===46&&(e.position+=3,G(e,!0,-1));return}e.position<e.length-1&&D(e,\"end of the stream or a document separator is expected\")}function qe(e,p){e=String(e),p=p||{},e.length!==0&&(e.charCodeAt(e.length-1)!==10&&e.charCodeAt(e.length-1)!==13&&(e+=`\n`),e.charCodeAt(0)===65279&&(e=e.slice(1)));let w=new z(e,p),C=e.indexOf(\"\\0\");for(C!==-1&&(w.position=C,D(w,\"null byte is not allowed in input\")),w.input+=\"\\0\";w.input.charCodeAt(w.position)===32;)w.lineIndent+=1,w.position+=1;for(;w.position<w.length-1;)nn(w);return w.documents}function je(e,p,w){p!==null&&typeof p==\"object\"&&typeof w>\"u\"&&(w=p,p=null);let C=qe(e,w);if(typeof p!=\"function\")return C;for(let y=0,k=C.length;y<k;y+=1)p(C[y])}function rn(e,p){let w=qe(e,p);if(w.length!==0){if(w.length===1)return w[0];throw new n(\"expected a single document in the stream, but found more\")}}return Je.loadAll=je,Je.load=rn,Je}var Vn={},$i;function Tr(){if($i)return Vn;$i=1;let i=Ie(),n=Oe(),t=Zn(),l=Object.prototype.toString,o=Object.prototype.hasOwnProperty,u=65279,f=9,c=10,d=13,s=32,a=33,m=34,I=35,R=37,v=38,g=39,N=42,O=44,S=45,b=58,_=61,P=62,U=63,q=64,Y=91,J=93,re=96,ie=123,me=124,X=125,z={};z[0]=\"\\\\0\",z[7]=\"\\\\a\",z[8]=\"\\\\b\",z[9]=\"\\\\t\",z[10]=\"\\\\n\",z[11]=\"\\\\v\",z[12]=\"\\\\f\",z[13]=\"\\\\r\",z[27]=\"\\\\e\",z[34]='\\\\\"',z[92]=\"\\\\\\\\\",z[133]=\"\\\\N\",z[160]=\"\\\\_\",z[8232]=\"\\\\L\",z[8233]=\"\\\\P\";let ge=[\"y\",\"Y\",\"yes\",\"Yes\",\"YES\",\"on\",\"On\",\"ON\",\"n\",\"N\",\"no\",\"No\",\"NO\",\"off\",\"Off\",\"OFF\"],D=/^[-+]?[0-9_]+(?::[0-9_]+)+(?:\\.[0-9_]*)?$/;function ce(r,h){if(h===null)return{};let T={},A=Object.keys(h);for(let x=0,L=A.length;x<L;x+=1){let E=A[x],M=String(h[E]);E.slice(0,2)===\"!!\"&&(E=\"tag:yaml.org,2002:\"+E.slice(2));let F=r.compiledTypeMap.fallback[E];F&&o.call(F.styleAliases,M)&&(M=F.styleAliases[M]),T[E]=M}return T}function Z(r){let h,T,A=r.toString(16).toUpperCase();if(r<=255)h=\"x\",T=2;else if(r<=65535)h=\"u\",T=4;else if(r<=4294967295)h=\"U\",T=8;else throw new n(\"code point within a string may not be greater than 0xFFFFFFFF\");return\"\\\\\"+h+i.repeat(\"0\",T-A.length)+A}let Xe=1,pe=2;function Ve(r){this.schema=r.schema||t,this.indent=Math.max(1,r.indent||2),this.noArrayIndent=r.noArrayIndent||!1,this.skipInvalid=r.skipInvalid||!1,this.flowLevel=i.isNothing(r.flowLevel)?-1:r.flowLevel,this.styleMap=ce(this.schema,r.styles||null),this.sortKeys=r.sortKeys||!1,this.lineWidth=r.lineWidth||80,this.noRefs=r.noRefs||!1,this.noCompatMode=r.noCompatMode||!1,this.condenseFlow=r.condenseFlow||!1,this.quotingType=r.quotingType==='\"'?pe:Xe,this.forceQuotes=r.forceQuotes||!1,this.replacer=typeof r.replacer==\"function\"?r.replacer:null,this.implicitTypes=this.schema.compiledImplicit,this.explicitTypes=this.schema.compiledExplicit,this.tag=null,this.result=\"\",this.duplicates=[],this.usedDuplicates=null}function ve(r,h){let T=i.repeat(\" \",h),A=0,x=\"\",L=r.length;for(;A<L;){let E,M=r.indexOf(`\n`,A);M===-1?(E=r.slice(A),A=L):(E=r.slice(A,M+1),A=M+1),E.length&&E!==`\n`&&(x+=T),x+=E}return x}function ye(r,h){return`\n`+i.repeat(\" \",r.indent*h)}function _e(r,h){for(let T=0,A=r.implicitTypes.length;T<A;T+=1)if(r.implicitTypes[T].resolve(h))return!0;return!1}function V(r){return r===s||r===f}function se(r){return r>=32&&r<=126||r>=161&&r<=55295&&r!==8232&&r!==8233||r>=57344&&r<=65533&&r!==u||r>=65536&&r<=1114111}function te(r){return se(r)&&r!==u&&r!==d&&r!==c}function we(r,h,T){let A=te(r),x=A&&!V(r);return(T?A:A&&r!==O&&r!==Y&&r!==J&&r!==ie&&r!==X)&&r!==I&&!(h===b&&!x)||te(h)&&!V(h)&&r===I||h===b&&x}function G(r){return se(r)&&r!==u&&!V(r)&&r!==S&&r!==U&&r!==b&&r!==O&&r!==Y&&r!==J&&r!==ie&&r!==X&&r!==I&&r!==v&&r!==N&&r!==a&&r!==me&&r!==_&&r!==P&&r!==g&&r!==m&&r!==R&&r!==q&&r!==re}function Ae(r){return!V(r)&&r!==b}function le(r,h){let T=r.charCodeAt(h),A;return T>=55296&&T<=56319&&h+1<r.length&&(A=r.charCodeAt(h+1),A>=56320&&A<=57343)?(T-55296)*1024+A-56320+65536:T}function Fe(r){return/^\\n* /.test(r)}let Re=1,Se=2,Me=3,De=4,oe=5;function Pe(r,h,T,A,x,L,E,M){let F,W=0,H=null,j=!1,B=!1,ii=A!==-1,be=-1,Ee=G(le(r,0))&&Ae(le(r,r.length-1));if(h||E)for(F=0;F<r.length;W>=65536?F+=2:F++){if(W=le(r,F),!se(W))return oe;Ee=Ee&&we(W,H,M),H=W}else{for(F=0;F<r.length;W>=65536?F+=2:F++){if(W=le(r,F),W===c)j=!0,ii&&(B=B||F-be-1>A&&r[be+1]!==\" \",be=F);else if(!se(W))return oe;Ee=Ee&&we(W,H,M),H=W}B=B||ii&&F-be-1>A&&r[be+1]!==\" \"}return!j&&!B?Ee&&!E&&!x(r)?Re:L===pe?oe:Se:T>9&&Fe(r)?oe:E?L===pe?oe:Se:B?De:Me}function Ze(r,h,T,A,x){r.dump=(function(){if(h.length===0)return r.quotingType===pe?'\"\"':\"''\";if(!r.noCompatMode&&(ge.indexOf(h)!==-1||D.test(h)))return r.quotingType===pe?'\"'+h+'\"':\"'\"+h+\"'\";let L=r.indent*Math.max(1,T),E=r.lineWidth===-1?-1:Math.max(Math.min(r.lineWidth,40),r.lineWidth-L),M=A||r.flowLevel>-1&&T>=r.flowLevel;function F(W){return _e(r,W)}switch(Pe(h,M,r.indent,E,F,r.quotingType,r.forceQuotes&&!A,x)){case Re:return h;case Se:return\"'\"+h.replace(/'/g,\"''\")+\"'\";case Me:return\"|\"+We(h,r.indent)+He(ve(h,L));case De:return\">\"+We(h,r.indent)+He(ve(en(h,E),L));case oe:return'\"'+nn(h)+'\"';default:throw new n(\"impossible error: invalid scalar style\")}})()}function We(r,h){let T=Fe(r)?String(h):\"\",A=r[r.length-1]===`\n`,L=A&&(r[r.length-2]===`\n`||r===`\n`)?\"+\":A?\"\":\"-\";return T+L+`\n`}function He(r){return r[r.length-1]===`\n`?r.slice(0,-1):r}function en(r,h){let T=/(\\n+)([^\\n]*)/g,A=(function(){let M=r.indexOf(`\n`);return M=M!==-1?M:r.length,T.lastIndex=M,ue(r.slice(0,M),h)})(),x=r[0]===`\n`||r[0]===\" \",L,E;for(;E=T.exec(r);){let M=E[1],F=E[2];L=F[0]===\" \",A+=M+(!x&&!L&&F!==\"\"?`\n`:\"\")+ue(F,h),x=L}return A}function ue(r,h){if(r===\"\"||r[0]===\" \")return r;let T=/ [^ ]/g,A,x=0,L,E=0,M=0,F=\"\";for(;A=T.exec(r);)M=A.index,M-x>h&&(L=E>x?E:M,F+=`\n`+r.slice(x,L),x=L+1),E=M;return F+=`\n`,r.length-x>h&&E>x?F+=r.slice(x,E)+`\n`+r.slice(E+1):F+=r.slice(x),F.slice(1)}function nn(r){let h=\"\",T=0;for(let A=0;A<r.length;T>=65536?A+=2:A++){T=le(r,A);let x=z[T];!x&&se(T)?(h+=r[A],T>=65536&&(h+=r[A+1])):h+=x||Z(T)}return h}function qe(r,h,T){let A=\"\",x=r.tag;for(let L=0,E=T.length;L<E;L+=1){let M=T[L];r.replacer&&(M=r.replacer.call(T,String(L),M)),(w(r,h,M,!1,!1)||typeof M>\"u\"&&w(r,h,null,!1,!1))&&(A!==\"\"&&(A+=\",\"+(r.condenseFlow?\"\":\" \")),A+=r.dump)}r.tag=x,r.dump=\"[\"+A+\"]\"}function je(r,h,T,A){let x=\"\",L=r.tag;for(let E=0,M=T.length;E<M;E+=1){let F=T[E];r.replacer&&(F=r.replacer.call(T,String(E),F)),(w(r,h+1,F,!0,!0,!1,!0)||typeof F>\"u\"&&w(r,h+1,null,!0,!0,!1,!0))&&((!A||x!==\"\")&&(x+=ye(r,h)),r.dump&&c===r.dump.charCodeAt(0)?x+=\"-\":x+=\"- \",x+=r.dump)}r.tag=L,r.dump=x||\"[]\"}function rn(r,h,T){let A=\"\",x=r.tag,L=Object.keys(T);for(let E=0,M=L.length;E<M;E+=1){let F=\"\";A!==\"\"&&(F+=\", \"),r.condenseFlow&&(F+='\"');let W=L[E],H=T[W];r.replacer&&(H=r.replacer.call(T,W,H)),w(r,h,W,!1,!1)&&(r.dump.length>1024&&(F+=\"? \"),F+=r.dump+(r.condenseFlow?'\"':\"\")+\":\"+(r.condenseFlow?\"\":\" \"),w(r,h,H,!1,!1)&&(F+=r.dump,A+=F))}r.tag=x,r.dump=\"{\"+A+\"}\"}function e(r,h,T,A){let x=\"\",L=r.tag,E=Object.keys(T);if(r.sortKeys===!0)E.sort();else if(typeof r.sortKeys==\"function\")E.sort(r.sortKeys);else if(r.sortKeys)throw new n(\"sortKeys must be a boolean or a function\");for(let M=0,F=E.length;M<F;M+=1){let W=\"\";(!A||x!==\"\")&&(W+=ye(r,h));let H=E[M],j=T[H];if(r.replacer&&(j=r.replacer.call(T,H,j)),!w(r,h+1,H,!0,!0,!0))continue;let B=r.tag!==null&&r.tag!==\"?\"||r.dump&&r.dump.length>1024;B&&(r.dump&&c===r.dump.charCodeAt(0)?W+=\"?\":W+=\"? \"),W+=r.dump,B&&(W+=ye(r,h)),w(r,h+1,j,!0,B)&&(r.dump&&c===r.dump.charCodeAt(0)?W+=\":\":W+=\": \",W+=r.dump,x+=W)}r.tag=L,r.dump=x||\"{}\"}function p(r,h,T){let A=T?r.explicitTypes:r.implicitTypes;for(let x=0,L=A.length;x<L;x+=1){let E=A[x];if((E.instanceOf||E.predicate)&&(!E.instanceOf||typeof h==\"object\"&&h instanceof E.instanceOf)&&(!E.predicate||E.predicate(h))){if(T?E.multi&&E.representName?r.tag=E.representName(h):r.tag=E.tag:r.tag=\"?\",E.represent){let M=r.styleMap[E.tag]||E.defaultStyle,F;if(l.call(E.represent)===\"[object Function]\")F=E.represent(h,M);else if(o.call(E.represent,M))F=E.represent[M](h,M);else throw new n(\"!<\"+E.tag+'> tag resolver accepts not \"'+M+'\" style');r.dump=F}return!0}}return!1}function w(r,h,T,A,x,L,E){r.tag=null,r.dump=T,p(r,T,!1)||p(r,T,!0);let M=l.call(r.dump),F=A;A&&(A=r.flowLevel<0||r.flowLevel>h);let W=M===\"[object Object]\"||M===\"[object Array]\",H,j;if(W&&(H=r.duplicates.indexOf(T),j=H!==-1),(r.tag!==null&&r.tag!==\"?\"||j||r.indent!==2&&h>0)&&(x=!1),j&&r.usedDuplicates[H])r.dump=\"*ref_\"+H;else{if(W&&j&&!r.usedDuplicates[H]&&(r.usedDuplicates[H]=!0),M===\"[object Object]\")A&&Object.keys(r.dump).length!==0?(e(r,h,r.dump,x),j&&(r.dump=\"&ref_\"+H+r.dump)):(rn(r,h,r.dump),j&&(r.dump=\"&ref_\"+H+\" \"+r.dump));else if(M===\"[object Array]\")A&&r.dump.length!==0?(r.noArrayIndent&&!E&&h>0?je(r,h-1,r.dump,x):je(r,h,r.dump,x),j&&(r.dump=\"&ref_\"+H+r.dump)):(qe(r,h,r.dump),j&&(r.dump=\"&ref_\"+H+\" \"+r.dump));else if(M===\"[object String]\")r.tag!==\"?\"&&Ze(r,r.dump,h,L,F);else{if(M===\"[object Undefined]\")return!1;if(r.skipInvalid)return!1;throw new n(\"unacceptable kind of an object to dump \"+M)}if(r.tag!==null&&r.tag!==\"?\"){let B=encodeURI(r.tag[0]===\"!\"?r.tag.slice(1):r.tag).replace(/!/g,\"%21\");r.tag[0]===\"!\"?B=\"!\"+B:B.slice(0,18)===\"tag:yaml.org,2002:\"?B=\"!!\"+B.slice(18):B=\"!<\"+B+\">\",r.dump=B+\" \"+r.dump}}return!0}function C(r,h){let T=[],A=[];y(r,T,A);let x=A.length;for(let L=0;L<x;L+=1)h.duplicates.push(T[A[L]]);h.usedDuplicates=new Array(x)}function y(r,h,T){if(r!==null&&typeof r==\"object\"){let A=h.indexOf(r);if(A!==-1)T.indexOf(A)===-1&&T.push(A);else if(h.push(r),Array.isArray(r))for(let x=0,L=r.length;x<L;x+=1)y(r[x],h,T);else{let x=Object.keys(r);for(let L=0,E=x.length;L<E;L+=1)y(r[x[L]],h,T)}}}function k(r,h){h=h||{};let T=new Ve(h);T.noRefs||C(r,T);let A=r;return T.replacer&&(A=T.replacer.call({\"\":A},\"\",A)),w(T,0,A,!0,!0)?T.dump+`\n`:\"\"}return Vn.dump=k,Vn}var Ki;function Lr(){if(Ki)return K;Ki=1;let i=Er(),n=Tr();function t(l,o){return function(){throw new Error(\"Function yaml.\"+l+\" is removed in js-yaml 4. Use yaml.\"+o+\" instead, which is now safe by default.\")}}return K.Type=Q(),K.Schema=Ji(),K.FAILSAFE_SCHEMA=Zi(),K.JSON_SCHEMA=tr(),K.CORE_SCHEMA=lr(),K.DEFAULT_SCHEMA=Zn(),K.load=i.load,K.loadAll=i.loadAll,K.dump=n.dump,K.YAMLException=Oe(),K.types={binary:fr(),float:rr(),map:Vi(),null:er(),pairs:sr(),set:ar(),timestamp:or(),bool:nr(),int:ir(),merge:ur(),omap:cr(),seq:Xi(),str:Qi()},K.safeLoad=t(\"safeLoad\",\"load\"),K.safeLoadAll=t(\"safeLoadAll\",\"loadAll\"),K.safeDump=t(\"safeDump\",\"dump\"),K}var kr=Lr(),Nr=Sr(kr),{Type:Rt,Schema:Mt,FAILSAFE_SCHEMA:Dt,JSON_SCHEMA:Pt,CORE_SCHEMA:Wt,DEFAULT_SCHEMA:Ht,load:ei,loadAll:qt,dump:Qe,YAMLException:jt,types:Yt,safeLoad:Ut,safeLoadAll:Bt,safeDump:Gt}=Nr;var Or=In;function ne(i,n){return n===\"yaml\"?Qe(i,{indent:2,lineWidth:-1,noRefs:!0}).trimEnd():JSON.stringify(i,null,2)}function _r(i,n,t=\"json\"){let l=ne(i,t),o=ne(n,t);return l===o?[{value:l}]:ni(i,n,0,t)}function Fr(i,n){return _r(i,n,\"json\")}function Rr(i,n){try{let t=JSON.parse(i),l=JSON.parse(n),o=JSON.stringify(t),u=JSON.stringify(l);return o===u?[{value:i}]:ee(i,n,{newlineIsToken:!1})}catch{return ee(i,n,{newlineIsToken:!1})}}function Mr(i,n){let t=ei(i),l=ei(n),o=Qe(t,{indent:2,lineWidth:-1,noRefs:!0}),u=Qe(l,{indent:2,lineWidth:-1,noRefs:!0});return o===u?[{value:i}]:ee(i,n,{newlineIsToken:!1})}function ni(i,n,t,l=\"json\"){let o=ne(i,l),u=ne(n,l);return o===u?[{value:fe(o,t,l)}]:typeof i==\"object\"&&i!==null&&typeof n==\"object\"&&n!==null&&!Array.isArray(i)&&!Array.isArray(n)?Dr(i,n,t,l):Array.isArray(i)&&Array.isArray(n)?Pr(i,n,t,l):Wr(o,u,t,l)}function Dr(i,n,t,l=\"json\"){let o=[],u=\"  \".repeat(t),f=\"  \".repeat(t+1),c=new Set(Object.keys(i)),d=Object.keys(n),s=[...c].filter(m=>!(m in n)),a=[...d,...s];o.push({value:`{\n`});for(let m=0;m<a.length;m++){let I=a[m],v=m===a.length-1?\"\":\",\",g=I in i,N=I in n;if(g&&N){let O=ne(i[I],l),S=ne(n[I],l);if(O===S){let b=fe(O,t+1);o.push({value:f+JSON.stringify(I)+\": \"+b+v+`\n`})}else{let b=f+JSON.stringify(I)+\": \",_=ni(i[I],n[I],t+1,l);if(_.length>0)if(!_[0].removed&&!_[0].added)_[0].value=b+_[0].value;else{let P=_.find(q=>q.removed),U=_.find(q=>q.added);P&&(P.value=b+P.value),U&&(U.value=b+U.value)}if(v&&_.length>0){let P=_[_.length-1];P.value=P.value.replace(/\\n$/,v+`\n`)}o.push(..._)}}else if(g){let O=fe(ne(i[I],l),t+1);o.push({removed:!0,value:f+JSON.stringify(I)+\": \"+O+v+`\n`})}else{let O=fe(ne(n[I],l),t+1);o.push({added:!0,value:f+JSON.stringify(I)+\": \"+O+v+`\n`})}}return o.push({value:u+`}\n`}),o}function Pr(i,n,t,l=\"json\"){let o=[],u=\"  \".repeat(t),f=\"  \".repeat(t+1);o.push({value:`[\n`});let c=Math.max(i.length,n.length);for(let d=0;d<c;d++){let a=d===c-1?\"\":\",\";if(d>=i.length){let m=fe(ne(n[d],l),t+1);o.push({added:!0,value:f+m+a+`\n`})}else if(d>=n.length){let m=fe(ne(i[d],l),t+1);o.push({removed:!0,value:f+m+a+`\n`})}else{let m=ne(i[d],l),I=ne(n[d],l);if(m===I){let R=fe(m,t+1);o.push({value:f+R+a+`\n`})}else{let R=ni(i[d],n[d],t+1,l);if(R.length>0&&(R[0].value=f+R[0].value),a&&R.length>0){let v=R[R.length-1];v.value=v.value.replace(/\\n$/,a+`\n`)}o.push(...R)}}}return o.push({value:u+`]\n`}),o}function Wr(i,n,t,l=\"json\"){let o=fe(i,t),u=fe(n,t);return ee(o,u).map(c=>({value:c.value,added:c.added,removed:c.removed}))}function fe(i,n,t=\"json\"){if(n===0)return i;let l=\"  \".repeat(n);return i.split(`\n`).map((o,u)=>u===0?o:l+o).join(`\n`)}var dr=i=>i===\"\"?[]:i.replace(/\\n$/,\"\").split(`\n`),Hr=(i,n,t=\"diffChars\")=>{let o=(typeof t==\"string\"?Or[t]:t)(i,n),u={left:[],right:[]};return o.forEach(({added:f,removed:c,value:d})=>{if(f)u.right.push({type:1,value:d});else if(c)u.left.push({type:2,value:d});else{let s={type:0,value:d};u.right.push(s),u.left.push(s)}}),u},pr=(i,n,t=!1,l=\"diffChars\",o=0,u=[],f=!1)=>{let c=[];if(typeof i==\"string\"&&typeof n==\"string\")if(l===\"diffJson\")c=Rr(i,n);else if(l===\"diffYaml\")try{c=Mr(i,n)}catch{c=ee(i,n,{newlineIsToken:!1})}else c=ee(i,n,{newlineIsToken:!1});else c=Fr(i,n);let d=o,s=o,a=[],m=0,I=[],R=[],v=(g,N,O,S,b)=>dr(g).map((P,U)=>{let q={},Y={};if(!(R.includes(`${N}-${U}`)||b&&U!==0)){if(O||S){let J=!0;if(S){s+=1,q.lineNumber=s,q.type=2,q.value=P||\" \";let re=c[N+1];if(re?.added){let ie=dr(re.value)[U];if(ie){let me=v(ie,N,!0,!1,!0),{value:X,lineNumber:z,type:ge}=me[0].right;if(R.push(`${N+1}-${U}`),Y.lineNumber=z,q.value===X)J=!1,Y.type=0,q.type=0,Y.value=X;else{Y.type=ge;let D=500,ce=P.length>D||X.length>D;if(t||ce)Y.value=X;else if(f)q.rawValue=P,q.value=P,Y.rawValue=X,Y.value=X;else{let Z=Hr(P,X,l);Y.value=Z.right,q.value=Z.left}}}}}else d+=1,Y.lineNumber=d,Y.type=1,Y.value=P;J&&!b&&(I.includes(m)||I.push(m))}else s+=1,d+=1,q.lineNumber=s,q.type=0,q.value=P,Y.lineNumber=d,Y.type=0,Y.value=P;return(u?.includes(`L-${q.lineNumber}`)||u?.includes(`R-${Y.lineNumber}`)&&!I.includes(m))&&I.push(m),b||(m+=1),{right:Y,left:q}}}).filter(P=>P!=null);return c.forEach(({added:g,removed:N,value:O},S)=>{a=[...a,...v(O,S,g,N)]}),{lineInformation:a,diffLines:I}};self.onmessage=i=>{let{oldString:n,newString:t,disableWordDiff:l,lineCompareMethod:o,linesOffset:u,showLines:f,deferWordDiff:c}=i.data,d=pr(n,t,l,o,u,f,c);self.postMessage(d)};})();\n";
})), md = Ol, hd;
(function(e) {
	e[e.DEFAULT = 0] = "DEFAULT", e[e.ADDED = 1] = "ADDED", e[e.REMOVED = 2] = "REMOVED", e[e.CHANGED = 3] = "CHANGED";
})(hd ||= {});
function gd(e, t) {
	return t === "yaml" ? ad(e, {
		indent: 2,
		lineWidth: -1,
		noRefs: !0
	}).trimEnd() : JSON.stringify(e, null, 2);
}
function _d(e, t, n = "json") {
	let r = gd(e, n);
	return r === gd(t, n) ? [{ value: r }] : xd(e, t, 0, n);
}
function vd(e, t) {
	return _d(e, t, "json");
}
function yd(e, t) {
	try {
		let n = JSON.parse(e), r = JSON.parse(t);
		return JSON.stringify(n) === JSON.stringify(r) ? [{ value: e }] : Gc(e, t, { newlineIsToken: !1 });
	} catch {
		return Gc(e, t, { newlineIsToken: !1 });
	}
}
function bd(e, t) {
	let n = rd(e), r = rd(t);
	return ad(n, {
		indent: 2,
		lineWidth: -1,
		noRefs: !0
	}) === ad(r, {
		indent: 2,
		lineWidth: -1,
		noRefs: !0
	}) ? [{ value: e }] : Gc(e, t, { newlineIsToken: !1 });
}
function xd(e, t, n, r = "json") {
	let i = gd(e, r), a = gd(t, r);
	return i === a ? [{ value: Td(i, n, r) }] : typeof e == "object" && e && typeof t == "object" && t && !Array.isArray(e) && !Array.isArray(t) ? Sd(e, t, n, r) : Array.isArray(e) && Array.isArray(t) ? Cd(e, t, n, r) : wd(i, a, n, r);
}
function Sd(e, t, n, r = "json") {
	let i = [], a = "  ".repeat(n), o = "  ".repeat(n + 1), s = new Set(Object.keys(e)), c = Object.keys(t), l = [...s].filter((e) => !(e in t)), u = [...c, ...l];
	i.push({ value: "{\n" });
	for (let a = 0; a < u.length; a++) {
		let s = u[a], c = a === u.length - 1 ? "" : ",", l = s in e, d = s in t;
		if (l && d) {
			let a = gd(e[s], r);
			if (a === gd(t[s], r)) {
				let e = Td(a, n + 1);
				i.push({ value: o + JSON.stringify(s) + ": " + e + c + "\n" });
			} else {
				let a = o + JSON.stringify(s) + ": ", l = xd(e[s], t[s], n + 1, r);
				if (l.length > 0) {
					if (!l[0].removed && !l[0].added) l[0].value = a + l[0].value;
					else {
						let e = l.find((e) => e.removed), t = l.find((e) => e.added);
						e && (e.value = a + e.value), t && (t.value = a + t.value);
					}
				}
				if (c && l.length > 0) {
					let e = l[l.length - 1];
					e.value = e.value.replace(/\n$/, c + "\n");
				}
				i.push(...l);
			}
		} else if (l) {
			let t = Td(gd(e[s], r), n + 1);
			i.push({
				removed: !0,
				value: o + JSON.stringify(s) + ": " + t + c + "\n"
			});
		} else {
			let e = Td(gd(t[s], r), n + 1);
			i.push({
				added: !0,
				value: o + JSON.stringify(s) + ": " + e + c + "\n"
			});
		}
	}
	return i.push({ value: a + "}\n" }), i;
}
function Cd(e, t, n, r = "json") {
	let i = [], a = "  ".repeat(n), o = "  ".repeat(n + 1);
	i.push({ value: "[\n" });
	let s = Math.max(e.length, t.length);
	for (let a = 0; a < s; a++) {
		let c = a === s - 1 ? "" : ",";
		if (a >= e.length) {
			let e = Td(gd(t[a], r), n + 1);
			i.push({
				added: !0,
				value: o + e + c + "\n"
			});
		} else if (a >= t.length) {
			let t = Td(gd(e[a], r), n + 1);
			i.push({
				removed: !0,
				value: o + t + c + "\n"
			});
		} else {
			let s = gd(e[a], r);
			if (s === gd(t[a], r)) {
				let e = Td(s, n + 1);
				i.push({ value: o + e + c + "\n" });
			} else {
				let s = xd(e[a], t[a], n + 1, r);
				if (s.length > 0 && (s[0].value = o + s[0].value), c && s.length > 0) {
					let e = s[s.length - 1];
					e.value = e.value.replace(/\n$/, c + "\n");
				}
				i.push(...s);
			}
		}
	}
	return i.push({ value: a + "]\n" }), i;
}
function wd(e, t, n, r = "json") {
	return Gc(Td(e, n), Td(t, n)).map((e) => ({
		value: e.value,
		added: e.added,
		removed: e.removed
	}));
}
function Td(e, t, n = "json") {
	if (t === 0) return e;
	let r = "  ".repeat(t);
	return e.split("\n").map((e, t) => t === 0 ? e : r + e).join("\n");
}
var Ed;
(function(e) {
	e.CHARS = "diffChars", e.WORDS = "diffWords", e.WORDS_WITH_SPACE = "diffWordsWithSpace", e.LINES = "diffLines", e.TRIMMED_LINES = "diffTrimmedLines", e.SENTENCES = "diffSentences", e.CSS = "diffCss", e.JSON = "diffJson", e.YAML = "diffYaml";
})(Ed ||= {});
var Dd = (e) => e === "" ? [] : e.replace(/\n$/, "").split("\n"), Od = (e, t, n = Ed.CHARS) => {
	let r = (typeof n == "string" ? md[n] : n)(e, t), i = {
		left: [],
		right: []
	};
	return r.forEach(({ added: e, removed: t, value: n }) => {
		if (e) i.right.push({
			type: hd.ADDED,
			value: n
		});
		else if (t) i.left.push({
			type: hd.REMOVED,
			value: n
		});
		else {
			let e = {
				type: hd.DEFAULT,
				value: n
			};
			i.right.push(e), i.left.push(e);
		}
	}), i;
}, kd = (e, t, n = !1, r = Ed.CHARS, i = 0, a = [], o = !1) => {
	let s = [];
	if (typeof e == "string" && typeof t == "string") {
		if (r === Ed.JSON) s = yd(e, t);
		else if (r === Ed.YAML) try {
			s = bd(e, t);
		} catch {
			s = Gc(e, t, { newlineIsToken: !1 });
		}
		else s = Gc(e, t, { newlineIsToken: !1 });
	} else s = vd(e, t);
	let c = i, l = i, u = [], d = 0, f = [], p = [], m = (e, t, i, u, h) => Dd(e).map((e, g) => {
		let _ = {}, v = {};
		if (!(p.includes(`${t}-${g}`) || h && g !== 0)) {
			if (i || u) {
				let i = !0;
				if (u) {
					l += 1, _.lineNumber = l, _.type = hd.REMOVED, _.value = e || " ";
					let a = s[t + 1];
					if (a?.added) {
						let s = Dd(a.value)[g];
						if (s) {
							let { value: a, lineNumber: c, type: l } = m(s, t, !0, !1, !0)[0].right;
							if (p.push(`${t + 1}-${g}`), v.lineNumber = c, _.value === a) i = !1, v.type = 0, _.type = 0, v.value = a;
							else {
								v.type = l;
								let t = e.length > 500 || a.length > 500;
								if (n || t) v.value = a;
								else if (o) _.rawValue = e, _.value = e, v.rawValue = a, v.value = a;
								else {
									let t = Od(e, a, r);
									v.value = t.right, _.value = t.left;
								}
							}
						}
					}
				} else c += 1, v.lineNumber = c, v.type = hd.ADDED, v.value = e;
				i && !h && (f.includes(d) || f.push(d));
			} else l += 1, c += 1, _.lineNumber = l, _.type = hd.DEFAULT, _.value = e, v.lineNumber = c, v.type = hd.DEFAULT, v.value = e;
			return (a?.includes(`L-${_.lineNumber}`) || a?.includes(`R-${v.lineNumber}`) && !f.includes(d)) && f.push(d), h || (d += 1), {
				right: v,
				left: _
			};
		}
	}).filter((e) => e != null);
	return s.forEach(({ added: e, removed: t, value: n }, r) => {
		u = [...u, ...m(n, r, e, t)];
	}), {
		lineInformation: u,
		diffLines: f
	};
}, Ad = null, jd = null, Md = async () => {
	if (Ad !== null) return Ad;
	if (jd === !1) return null;
	if (typeof Worker > "u" || typeof Blob > "u" || typeof URL > "u") return jd = !1, null;
	try {
		let { WORKER_CODE: e } = await Promise.resolve().then(() => (pd(), dd)), t = new Blob([e], { type: "application/javascript" });
		Ad = URL.createObjectURL(t), jd = !0;
	} catch {
		jd = !1, Ad = null;
	}
	return Ad;
}, Nd = async (e, t, n = !1, r = Ed.CHARS, i = 0, a = [], o = !1, s = !1) => {
	let c = () => kd(e, t, n, r, i, a, o);
	if (s) return Promise.resolve(c());
	let l = await Md();
	return l ? new Promise((s) => {
		let u;
		try {
			u = new Worker(l);
		} catch {
			jd = !1, s(c());
			return;
		}
		u.onmessage = (e) => {
			s(e.data), u.terminate();
		}, u.onerror = () => {
			jd = !1, u.terminate(), s(c());
		}, u.postMessage({
			oldString: e,
			newString: t,
			disableWordDiff: n,
			lineCompareMethod: r,
			linesOffset: i,
			showLines: a,
			deferWordDiff: o
		});
	}) : Promise.resolve(c());
};
//#endregion
//#region node_modules/.pnpm/react-diff-viewer-continued@4.4.0_@types+react@19.2.18_react-dom@19.2.8_react@19.2.8__react@19.2.8/node_modules/react-diff-viewer-continued/lib/esm/src/expand.js
function Pd() {
	return m("svg", {
		xmlns: "http://www.w3.org/2000/svg",
		viewBox: "0 0 16 16",
		width: "16",
		height: "16",
		children: [p("title", { children: "expand" }), p("path", { d: "m8.177.677 2.896 2.896a.25.25 0 0 1-.177.427H8.75v1.25a.75.75 0 0 1-1.5 0V4H5.104a.25.25 0 0 1-.177-.427L7.823.677a.25.25 0 0 1 .354 0ZM7.25 10.75a.75.75 0 0 1 1.5 0V12h2.146a.25.25 0 0 1 .177.427l-2.896 2.896a.25.25 0 0 1-.354 0l-2.896-2.896A.25.25 0 0 1 5.104 12H7.25v-1.25Zm-5-2a.75.75 0 0 0 0-1.5h-.5a.75.75 0 0 0 0 1.5h.5ZM6 8a.75.75 0 0 1-.75.75h-.5a.75.75 0 0 1 0-1.5h.5A.75.75 0 0 1 6 8Zm2.25.75a.75.75 0 0 0 0-1.5h-.5a.75.75 0 0 0 0 1.5h.5ZM12 8a.75.75 0 0 1-.75.75h-.5a.75.75 0 0 1 0-1.5h.5A.75.75 0 0 1 12 8Zm2.25.75a.75.75 0 0 0 0-1.5h-.5a.75.75 0 0 0 0 1.5h.5Z" })]
	});
}
//#endregion
//#region node_modules/.pnpm/@emotion+sheet@1.4.0/node_modules/@emotion/sheet/dist/emotion-sheet.esm.js
function Fd(e) {
	if (e.sheet) return e.sheet;
	/* istanbul ignore next */
	for (var t = 0; t < document.styleSheets.length; t++) if (document.styleSheets[t].ownerNode === e) return document.styleSheets[t];
}
function Id(e) {
	var t = document.createElement("style");
	return t.setAttribute("data-emotion", e.key), e.nonce !== void 0 && t.setAttribute("nonce", e.nonce), t.appendChild(document.createTextNode("")), t.setAttribute("data-s", ""), t;
}
var Ld = /*#__PURE__*/ function() {
	function e(e) {
		var t = this;
		this._insertTag = function(e) {
			var n = t.tags.length === 0 ? t.insertionPoint ? t.insertionPoint.nextSibling : t.prepend ? t.container.firstChild : t.before : t.tags[t.tags.length - 1].nextSibling;
			t.container.insertBefore(e, n), t.tags.push(e);
		}, this.isSpeedy = e.speedy === void 0 || e.speedy, this.tags = [], this.ctr = 0, this.nonce = e.nonce, this.key = e.key, this.container = e.container, this.prepend = e.prepend, this.insertionPoint = e.insertionPoint, this.before = null;
	}
	var t = e.prototype;
	return t.hydrate = function(e) {
		e.forEach(this._insertTag);
	}, t.insert = function(e) {
		this.ctr % (this.isSpeedy ? 65e3 : 1) == 0 && this._insertTag(Id(this));
		var t = this.tags[this.tags.length - 1];
		if (this.isSpeedy) {
			var n = Fd(t);
			try {
				n.insertRule(e, n.cssRules.length);
			} catch {}
		} else t.appendChild(document.createTextNode(e));
		this.ctr++;
	}, t.flush = function() {
		this.tags.forEach(function(e) {
			return e.parentNode?.removeChild(e);
		}), this.tags = [], this.ctr = 0;
	}, e;
}(), Rd = "-ms-", zd = "-moz-", Bd = "-webkit-", Vd = "comm", Hd = "rule", Ud = "decl", Wd = "@import", Gd = "@keyframes", Kd = "@layer", qd = Math.abs, Jd = String.fromCharCode, Yd = Object.assign;
function Xd(e, t) {
	return tf(e, 0) ^ 45 ? (((t << 2 ^ tf(e, 0)) << 2 ^ tf(e, 1)) << 2 ^ tf(e, 2)) << 2 ^ tf(e, 3) : 0;
}
function Zd(e) {
	return e.trim();
}
function Qd(e, t) {
	return (e = t.exec(e)) ? e[0] : e;
}
function $d(e, t, n) {
	return e.replace(t, n);
}
function ef(e, t) {
	return e.indexOf(t);
}
function tf(e, t) {
	return e.charCodeAt(t) | 0;
}
function nf(e, t, n) {
	return e.slice(t, n);
}
function rf(e) {
	return e.length;
}
function af(e) {
	return e.length;
}
function of(e, t) {
	return t.push(e), e;
}
function sf(e, t) {
	return e.map(t).join("");
}
//#endregion
//#region node_modules/.pnpm/stylis@4.2.0/node_modules/stylis/src/Tokenizer.js
var cf = 1, lf = 1, uf = 0, df = 0, ff = 0, pf = "";
function mf(e, t, n, r, i, a, o) {
	return {
		value: e,
		root: t,
		parent: n,
		type: r,
		props: i,
		children: a,
		line: cf,
		column: lf,
		length: o,
		return: ""
	};
}
function hf(e, t) {
	return Yd(mf("", null, null, "", null, null, 0), e, { length: -e.length }, t);
}
function gf() {
	return ff;
}
function _f() {
	return ff = df > 0 ? tf(pf, --df) : 0, lf--, ff === 10 && (lf = 1, cf--), ff;
}
function vf() {
	return ff = df < uf ? tf(pf, df++) : 0, lf++, ff === 10 && (lf = 1, cf++), ff;
}
function yf() {
	return tf(pf, df);
}
function bf() {
	return df;
}
function xf(e, t) {
	return nf(pf, e, t);
}
function Sf(e) {
	switch (e) {
		case 0:
		case 9:
		case 10:
		case 13:
		case 32: return 5;
		case 33:
		case 43:
		case 44:
		case 47:
		case 62:
		case 64:
		case 126:
		case 59:
		case 123:
		case 125: return 4;
		case 58: return 3;
		case 34:
		case 39:
		case 40:
		case 91: return 2;
		case 41:
		case 93: return 1;
	}
	return 0;
}
function Cf(e) {
	return cf = lf = 1, uf = rf(pf = e), df = 0, [];
}
function wf(e) {
	return pf = "", e;
}
function Tf(e) {
	return Zd(xf(df - 1, Of(e === 91 ? e + 2 : e === 40 ? e + 1 : e)));
}
function Ef(e) {
	for (; (ff = yf()) && ff < 33;) vf();
	return Sf(e) > 2 || Sf(ff) > 3 ? "" : " ";
}
function Df(e, t) {
	for (; --t && vf() && !(ff < 48 || ff > 102 || ff > 57 && ff < 65 || ff > 70 && ff < 97););
	return xf(e, bf() + (t < 6 && yf() == 32 && vf() == 32));
}
function Of(e) {
	for (; vf();) switch (ff) {
		case e: return df;
		case 34:
		case 39:
			e !== 34 && e !== 39 && Of(ff);
			break;
		case 40:
			e === 41 && Of(e);
			break;
		case 92:
			vf();
			break;
	}
	return df;
}
function kf(e, t) {
	for (; vf() && e + ff !== 57 && (e + ff !== 84 || yf() !== 47););
	return "/*" + xf(t, df - 1) + "*" + Jd(e === 47 ? e : vf());
}
function Af(e) {
	for (; !Sf(yf());) vf();
	return xf(e, df);
}
//#endregion
//#region node_modules/.pnpm/stylis@4.2.0/node_modules/stylis/src/Parser.js
function jf(e) {
	return wf(Mf("", null, null, null, [""], e = Cf(e), 0, [0], e));
}
function Mf(e, t, n, r, i, a, o, s, c) {
	for (var l = 0, u = 0, d = o, f = 0, p = 0, m = 0, h = 1, g = 1, _ = 1, v = 0, y = "", b = i, x = a, S = r, C = y; g;) switch (m = v, v = vf()) {
		case 40: if (m != 108 && tf(C, d - 1) == 58) {
			ef(C += $d(Tf(v), "&", "&\f"), "&\f") != -1 && (_ = -1);
			break;
		}
		case 34:
		case 39:
		case 91:
			C += Tf(v);
			break;
		case 9:
		case 10:
		case 13:
		case 32:
			C += Ef(m);
			break;
		case 92:
			C += Df(bf() - 1, 7);
			continue;
		case 47:
			switch (yf()) {
				case 42:
				case 47:
					of(Pf(kf(vf(), bf()), t, n), c);
					break;
				default: C += "/";
			}
			break;
		case 123 * h: s[l++] = rf(C) * _;
		case 125 * h:
		case 59:
		case 0:
			switch (v) {
				case 0:
				case 125: g = 0;
				case 59 + u:
					_ == -1 && (C = $d(C, /\f/g, "")), p > 0 && rf(C) - d && of(p > 32 ? Ff(C + ";", r, n, d - 1) : Ff($d(C, " ", "") + ";", r, n, d - 2), c);
					break;
				case 59: C += ";";
				default: if (of(S = Nf(C, t, n, l, u, i, s, y, b = [], x = [], d), a), v === 123) {
					if (u === 0) Mf(C, t, S, S, b, a, d, s, x);
					else switch (f === 99 && tf(C, 3) === 110 ? 100 : f) {
						case 100:
						case 108:
						case 109:
						case 115:
							Mf(e, S, S, r && of(Nf(e, S, S, 0, 0, i, s, y, i, b = [], d), x), i, x, d, s, r ? b : x);
							break;
						default: Mf(C, S, S, S, [""], x, 0, s, x);
					}
				}
			}
			l = u = p = 0, h = _ = 1, y = C = "", d = o;
			break;
		case 58: d = 1 + rf(C), p = m;
		default:
			if (h < 1) {
				if (v == 123) --h;
				else if (v == 125 && h++ == 0 && _f() == 125) continue;
			}
			switch (C += Jd(v), v * h) {
				case 38:
					_ = u > 0 ? 1 : (C += "\f", -1);
					break;
				case 44:
					s[l++] = (rf(C) - 1) * _, _ = 1;
					break;
				case 64:
					yf() === 45 && (C += Tf(vf())), f = yf(), u = d = rf(y = C += Af(bf())), v++;
					break;
				case 45: m === 45 && rf(C) == 2 && (h = 0);
			}
	}
	return a;
}
function Nf(e, t, n, r, i, a, o, s, c, l, u) {
	for (var d = i - 1, f = i === 0 ? a : [""], p = af(f), m = 0, h = 0, g = 0; m < r; ++m) for (var _ = 0, v = nf(e, d + 1, d = qd(h = o[m])), y = e; _ < p; ++_) (y = Zd(h > 0 ? f[_] + " " + v : $d(v, /&\f/g, f[_]))) && (c[g++] = y);
	return mf(e, t, n, i === 0 ? Hd : s, c, l, u);
}
function Pf(e, t, n) {
	return mf(e, t, n, Vd, Jd(gf()), nf(e, 2, -2), 0);
}
function Ff(e, t, n, r) {
	return mf(e, t, n, Ud, nf(e, 0, r), nf(e, r + 1, -1), r);
}
//#endregion
//#region node_modules/.pnpm/stylis@4.2.0/node_modules/stylis/src/Serializer.js
function If(e, t) {
	for (var n = "", r = af(e), i = 0; i < r; i++) n += t(e[i], i, e, t) || "";
	return n;
}
function Lf(e, t, n, r) {
	switch (e.type) {
		case Kd: if (e.children.length) break;
		case Wd:
		case Ud: return e.return = e.return || e.value;
		case Vd: return "";
		case Gd: return e.return = e.value + "{" + If(e.children, r) + "}";
		case Hd: e.value = e.props.join(",");
	}
	return rf(n = If(e.children, r)) ? e.return = e.value + "{" + n + "}" : "";
}
//#endregion
//#region node_modules/.pnpm/stylis@4.2.0/node_modules/stylis/src/Middleware.js
function Rf(e) {
	var t = af(e);
	return function(n, r, i, a) {
		for (var o = "", s = 0; s < t; s++) o += e[s](n, r, i, a) || "";
		return o;
	};
}
function zf(e) {
	return function(t) {
		t.root || (t = t.return) && e(t);
	};
}
//#endregion
//#region node_modules/.pnpm/@emotion+memoize@0.9.0/node_modules/@emotion/memoize/dist/emotion-memoize.esm.js
function Bf(e) {
	var t = Object.create(null);
	return function(n) {
		return t[n] === void 0 && (t[n] = e(n)), t[n];
	};
}
//#endregion
//#region node_modules/.pnpm/@emotion+cache@11.14.0/node_modules/@emotion/cache/dist/emotion-cache.browser.esm.js
var Vf = function(e, t, n) {
	for (var r = 0, i = 0; r = i, i = yf(), r === 38 && i === 12 && (t[n] = 1), !Sf(i);) vf();
	return xf(e, df);
}, Hf = function(e, t) {
	var n = -1, r = 44;
	do
		switch (Sf(r)) {
			case 0:
				r === 38 && yf() === 12 && (t[n] = 1), e[n] += Vf(df - 1, t, n);
				break;
			case 2:
				e[n] += Tf(r);
				break;
			case 4: if (r === 44) {
				e[++n] = yf() === 58 ? "&\f" : "", t[n] = e[n].length;
				break;
			}
			default: e[n] += Jd(r);
		}
	while (r = vf());
	return e;
}, Uf = function(e, t) {
	return wf(Hf(Cf(e), t));
}, Wf = /* #__PURE__ */ new WeakMap(), Gf = function(e) {
	if (!(e.type !== "rule" || !e.parent || e.length < 1)) {
		for (var t = e.value, n = e.parent, r = e.column === n.column && e.line === n.line; n.type !== "rule";) if (n = n.parent, !n) return;
		if (!(e.props.length === 1 && t.charCodeAt(0) !== 58 && !Wf.get(n)) && !r) {
			Wf.set(e, !0);
			for (var i = [], a = Uf(t, i), o = n.props, s = 0, c = 0; s < a.length; s++) for (var l = 0; l < o.length; l++, c++) e.props[c] = i[s] ? a[s].replace(/&\f/g, o[l]) : o[l] + " " + a[s];
		}
	}
}, Kf = function(e) {
	if (e.type === "decl") {
		var t = e.value;
		t.charCodeAt(0) === 108 && t.charCodeAt(2) === 98 && (e.return = "", e.value = "");
	}
};
function qf(e, t) {
	switch (Xd(e, t)) {
		case 5103: return Bd + "print-" + e + e;
		case 5737:
		case 4201:
		case 3177:
		case 3433:
		case 1641:
		case 4457:
		case 2921:
		case 5572:
		case 6356:
		case 5844:
		case 3191:
		case 6645:
		case 3005:
		case 6391:
		case 5879:
		case 5623:
		case 6135:
		case 4599:
		case 4855:
		case 4215:
		case 6389:
		case 5109:
		case 5365:
		case 5621:
		case 3829: return Bd + e + e;
		case 5349:
		case 4246:
		case 4810:
		case 6968:
		case 2756: return Bd + e + zd + e + Rd + e + e;
		case 6828:
		case 4268: return Bd + e + Rd + e + e;
		case 6165: return Bd + e + Rd + "flex-" + e + e;
		case 5187: return Bd + e + $d(e, /(\w+).+(:[^]+)/, Bd + "box-$1$2" + Rd + "flex-$1$2") + e;
		case 5443: return Bd + e + Rd + "flex-item-" + $d(e, /flex-|-self/, "") + e;
		case 4675: return Bd + e + Rd + "flex-line-pack" + $d(e, /align-content|flex-|-self/, "") + e;
		case 5548: return Bd + e + Rd + $d(e, "shrink", "negative") + e;
		case 5292: return Bd + e + Rd + $d(e, "basis", "preferred-size") + e;
		case 6060: return Bd + "box-" + $d(e, "-grow", "") + Bd + e + Rd + $d(e, "grow", "positive") + e;
		case 4554: return Bd + $d(e, /([^-])(transform)/g, "$1" + Bd + "$2") + e;
		case 6187: return $d($d($d(e, /(zoom-|grab)/, Bd + "$1"), /(image-set)/, Bd + "$1"), e, "") + e;
		case 5495:
		case 3959: return $d(e, /(image-set\([^]*)/, Bd + "$1$`$1");
		case 4968: return $d($d(e, /(.+:)(flex-)?(.*)/, Bd + "box-pack:$3" + Rd + "flex-pack:$3"), /s.+-b[^;]+/, "justify") + Bd + e + e;
		case 4095:
		case 3583:
		case 4068:
		case 2532: return $d(e, /(.+)-inline(.+)/, Bd + "$1$2") + e;
		case 8116:
		case 7059:
		case 5753:
		case 5535:
		case 5445:
		case 5701:
		case 4933:
		case 4677:
		case 5533:
		case 5789:
		case 5021:
		case 4765:
			if (rf(e) - 1 - t > 6) switch (tf(e, t + 1)) {
				case 109: if (tf(e, t + 4) !== 45) break;
				case 102: return $d(e, /(.+:)(.+)-([^]+)/, "$1" + Bd + "$2-$3$1" + zd + (tf(e, t + 3) == 108 ? "$3" : "$2-$3")) + e;
				case 115: return ~ef(e, "stretch") ? qf($d(e, "stretch", "fill-available"), t) + e : e;
			}
			break;
		case 4949: if (tf(e, t + 1) !== 115) break;
		case 6444:
			switch (tf(e, rf(e) - 3 - (~ef(e, "!important") && 10))) {
				case 107: return $d(e, ":", ":" + Bd) + e;
				case 101: return $d(e, /(.+:)([^;!]+)(;|!.+)?/, "$1" + Bd + (tf(e, 14) === 45 ? "inline-" : "") + "box$3$1" + Bd + "$2$3$1" + Rd + "$2box$3") + e;
			}
			break;
		case 5936:
			switch (tf(e, t + 11)) {
				case 114: return Bd + e + Rd + $d(e, /[svh]\w+-[tblr]{2}/, "tb") + e;
				case 108: return Bd + e + Rd + $d(e, /[svh]\w+-[tblr]{2}/, "tb-rl") + e;
				case 45: return Bd + e + Rd + $d(e, /[svh]\w+-[tblr]{2}/, "lr") + e;
			}
			return Bd + e + Rd + e + e;
	}
	return e;
}
var Jf = [function(e, t, n, r) {
	if (e.length > -1 && !e.return) switch (e.type) {
		case Ud:
			e.return = qf(e.value, e.length);
			break;
		case Gd: return If([hf(e, { value: $d(e.value, "@", "@" + Bd) })], r);
		case Hd: if (e.length) return sf(e.props, function(t) {
			switch (Qd(t, /(::plac\w+|:read-\w+)/)) {
				case ":read-only":
				case ":read-write": return If([hf(e, { props: [$d(t, /:(read-\w+)/, ":" + zd + "$1")] })], r);
				case "::placeholder": return If([
					hf(e, { props: [$d(t, /:(plac\w+)/, ":" + Bd + "input-$1")] }),
					hf(e, { props: [$d(t, /:(plac\w+)/, ":" + zd + "$1")] }),
					hf(e, { props: [$d(t, /:(plac\w+)/, Rd + "input-$1")] })
				], r);
			}
			return "";
		});
	}
}], Yf = function(e) {
	var t = e.key;
	if (t === "css") {
		var n = document.querySelectorAll("style[data-emotion]:not([data-s])");
		Array.prototype.forEach.call(n, function(e) {
			e.getAttribute("data-emotion").indexOf(" ") !== -1 && (document.head.appendChild(e), e.setAttribute("data-s", ""));
		});
	}
	var r = e.stylisPlugins || Jf, i = {}, a, o = [];
	a = e.container || document.head, Array.prototype.forEach.call(document.querySelectorAll("style[data-emotion^=\"" + t + " \"]"), function(e) {
		for (var t = e.getAttribute("data-emotion").split(" "), n = 1; n < t.length; n++) i[t[n]] = !0;
		o.push(e);
	});
	var s, c = [Gf, Kf], l, u = [Lf, zf(function(e) {
		l.insert(e);
	})], d = Rf(c.concat(r, u)), f = function(e) {
		return If(jf(e), d);
	};
	s = function(e, t, n, r) {
		l = n, f(e ? e + "{" + t.styles + "}" : t.styles), r && (p.inserted[t.name] = !0);
	};
	var p = {
		key: t,
		sheet: new Ld({
			key: t,
			container: a,
			nonce: e.nonce,
			speedy: e.speedy,
			prepend: e.prepend,
			insertionPoint: e.insertionPoint
		}),
		nonce: e.nonce,
		inserted: i,
		registered: {},
		insert: s
	};
	return p.sheet.hydrate(o), p;
};
//#endregion
//#region node_modules/.pnpm/@emotion+hash@0.9.2/node_modules/@emotion/hash/dist/emotion-hash.esm.js
function Xf(e) {
	for (var t = 0, n, r = 0, i = e.length; i >= 4; ++r, i -= 4) n = e.charCodeAt(r) & 255 | (e.charCodeAt(++r) & 255) << 8 | (e.charCodeAt(++r) & 255) << 16 | (e.charCodeAt(++r) & 255) << 24, n = (n & 65535) * 1540483477 + ((n >>> 16) * 59797 << 16), n ^= n >>> 24, t = (n & 65535) * 1540483477 + ((n >>> 16) * 59797 << 16) ^ (t & 65535) * 1540483477 + ((t >>> 16) * 59797 << 16);
	switch (i) {
		case 3: t ^= (e.charCodeAt(r + 2) & 255) << 16;
		case 2: t ^= (e.charCodeAt(r + 1) & 255) << 8;
		case 1: t ^= e.charCodeAt(r) & 255, t = (t & 65535) * 1540483477 + ((t >>> 16) * 59797 << 16);
	}
	return t ^= t >>> 13, t = (t & 65535) * 1540483477 + ((t >>> 16) * 59797 << 16), ((t ^ t >>> 15) >>> 0).toString(36);
}
//#endregion
//#region node_modules/.pnpm/@emotion+unitless@0.10.0/node_modules/@emotion/unitless/dist/emotion-unitless.esm.js
var Zf = {
	animationIterationCount: 1,
	aspectRatio: 1,
	borderImageOutset: 1,
	borderImageSlice: 1,
	borderImageWidth: 1,
	boxFlex: 1,
	boxFlexGroup: 1,
	boxOrdinalGroup: 1,
	columnCount: 1,
	columns: 1,
	flex: 1,
	flexGrow: 1,
	flexPositive: 1,
	flexShrink: 1,
	flexNegative: 1,
	flexOrder: 1,
	gridRow: 1,
	gridRowEnd: 1,
	gridRowSpan: 1,
	gridRowStart: 1,
	gridColumn: 1,
	gridColumnEnd: 1,
	gridColumnSpan: 1,
	gridColumnStart: 1,
	msGridRow: 1,
	msGridRowSpan: 1,
	msGridColumn: 1,
	msGridColumnSpan: 1,
	fontWeight: 1,
	lineHeight: 1,
	opacity: 1,
	order: 1,
	orphans: 1,
	scale: 1,
	tabSize: 1,
	widows: 1,
	zIndex: 1,
	zoom: 1,
	WebkitLineClamp: 1,
	fillOpacity: 1,
	floodOpacity: 1,
	stopOpacity: 1,
	strokeDasharray: 1,
	strokeDashoffset: 1,
	strokeMiterlimit: 1,
	strokeOpacity: 1,
	strokeWidth: 1
}, Qf = /[A-Z]|^ms/g, $f = /_EMO_([^_]+?)_([^]*?)_EMO_/g, ep = function(e) {
	return e.charCodeAt(1) === 45;
}, tp = function(e) {
	return e != null && typeof e != "boolean";
}, np = /* #__PURE__ */ Bf(function(e) {
	return ep(e) ? e : e.replace(Qf, "-$&").toLowerCase();
}), rp = function(e, t) {
	switch (e) {
		case "animation":
		case "animationName": if (typeof t == "string") return t.replace($f, function(e, t, n) {
			return sp = {
				name: t,
				styles: n,
				next: sp
			}, t;
		});
	}
	return Zf[e] !== 1 && !ep(e) && typeof t == "number" && t !== 0 ? t + "px" : t;
};
function ip(e, t, n) {
	if (n == null) return "";
	var r = n;
	if (r.__emotion_styles !== void 0) return r;
	switch (typeof n) {
		case "boolean": return "";
		case "object":
			var i = n;
			if (i.anim === 1) return sp = {
				name: i.name,
				styles: i.styles,
				next: sp
			}, i.name;
			var a = n;
			if (a.styles !== void 0) {
				var o = a.next;
				if (o !== void 0) for (; o !== void 0;) sp = {
					name: o.name,
					styles: o.styles,
					next: sp
				}, o = o.next;
				return a.styles + ";";
			}
			return ap(e, t, n);
		case "function": if (e !== void 0) {
			var s = sp, c = n(e);
			return sp = s, ip(e, t, c);
		}
	}
	var l = n;
	if (t == null) return l;
	var u = t[l];
	return u === void 0 ? l : u;
}
function ap(e, t, n) {
	var r = "";
	if (Array.isArray(n)) for (var i = 0; i < n.length; i++) r += ip(e, t, n[i]) + ";";
	else for (var a in n) {
		var o = n[a];
		if (typeof o != "object") {
			var s = o;
			t != null && t[s] !== void 0 ? r += a + "{" + t[s] + "}" : tp(s) && (r += np(a) + ":" + rp(a, s) + ";");
		} else if (Array.isArray(o) && typeof o[0] == "string" && (t == null || t[o[0]] === void 0)) for (var c = 0; c < o.length; c++) tp(o[c]) && (r += np(a) + ":" + rp(a, o[c]) + ";");
		else {
			var l = ip(e, t, o);
			switch (a) {
				case "animation":
				case "animationName":
					r += np(a) + ":" + l + ";";
					break;
				default: r += a + "{" + l + "}";
			}
		}
	}
	return r;
}
var op = /label:\s*([^\s;{]+)\s*(;|$)/g, sp;
function cp(e, t, n) {
	if (e.length === 1 && typeof e[0] == "object" && e[0] !== null && e[0].styles !== void 0) return e[0];
	var r = !0, i = "";
	sp = void 0;
	var a = e[0];
	a == null || a.raw === void 0 ? (r = !1, i += ip(n, t, a)) : i += a[0];
	for (var o = 1; o < e.length; o++) i += ip(n, t, e[o]), r && (i += a[o]);
	op.lastIndex = 0;
	for (var s = "", c; (c = op.exec(i)) !== null;) s += "-" + c[1];
	return {
		name: Xf(i) + s,
		styles: i,
		next: sp
	};
}
//#endregion
//#region node_modules/.pnpm/@emotion+utils@1.4.2/node_modules/@emotion/utils/dist/emotion-utils.browser.esm.js
function lp(e, t, n) {
	var r = "";
	return n.split(" ").forEach(function(n) {
		e[n] === void 0 ? n && (r += n + " ") : t.push(e[n] + ";");
	}), r;
}
var up = function(e, t, n) {
	var r = e.key + "-" + t.name;
	n === !1 && e.registered[r] === void 0 && (e.registered[r] = t.styles);
}, dp = function(e, t, n) {
	up(e, t, n);
	var r = e.key + "-" + t.name;
	if (e.inserted[t.name] === void 0) {
		var i = t;
		do
			e.insert(t === i ? "." + r : "", i, e.sheet, !0), i = i.next;
		while (i !== void 0);
	}
};
//#endregion
//#region node_modules/.pnpm/@emotion+css@11.13.5/node_modules/@emotion/css/create-instance/dist/emotion-css-create-instance.esm.js
function fp(e, t) {
	if (e.inserted[t.name] === void 0) return e.insert("", t, e.sheet, !0);
}
function pp(e, t, n) {
	var r = [], i = lp(e, r, n);
	return r.length < 2 ? n : i + t(r);
}
var mp = function(e) {
	var t = Yf(e);
	t.sheet.speedy = function(e) {
		this.isSpeedy = e;
	}, t.compat = !0;
	var n = function() {
		var e = cp([...arguments], t.registered, void 0);
		return dp(t, e, !1), t.key + "-" + e.name;
	};
	return {
		css: n,
		cx: function() {
			var e = [...arguments];
			return pp(t.registered, n, hp(e));
		},
		injectGlobal: function() {
			fp(t, cp([...arguments], t.registered));
		},
		keyframes: function() {
			var e = cp([...arguments], t.registered), n = "animation-" + e.name;
			return fp(t, {
				name: e.name,
				styles: "@keyframes " + n + "{" + e.styles + "}"
			}), n;
		},
		hydrate: function(e) {
			e.forEach(function(e) {
				t.inserted[e] = !0;
			});
		},
		flush: function() {
			t.registered = {}, t.inserted = {}, t.sheet.flush();
		},
		sheet: t.sheet,
		cache: t,
		getRegisteredStyles: lp.bind(null, t.registered),
		merge: pp.bind(null, t.registered, n)
	};
}, hp = function e(t) {
	for (var n = "", r = 0; r < t.length; r++) {
		var i = t[r];
		if (i != null) {
			var a = void 0;
			switch (typeof i) {
				case "boolean": break;
				case "object":
					if (Array.isArray(i)) a = e(i);
					else for (var o in a = "", i) i[o] && o && (a && (a += " "), a += o);
					break;
				default: a = i;
			}
			a && (n && (n += " "), n += a);
		}
	}
	return n;
}, gp = Object.freeze({
	diffViewerBackground: "#fff",
	diffViewerColor: "#212529",
	addedBackground: "#e6ffed",
	addedColor: "#24292e",
	removedBackground: "#ffeef0",
	removedColor: "#24292e",
	changedBackground: "#fffbdd",
	wordAddedBackground: "#acf2bd",
	wordRemovedBackground: "#fdb8c0",
	addedGutterBackground: "#cdffd8",
	removedGutterBackground: "#ffdce0",
	gutterBackground: "#f7f7f7",
	gutterBackgroundDark: "#f3f1f1",
	highlightBackground: "#fffbdd",
	highlightGutterBackground: "#fff5b1",
	codeFoldGutterBackground: "#dbedff",
	codeFoldBackground: "#f1f8ff",
	emptyLineBackground: "#fafbfc",
	gutterColor: "#212529",
	addedGutterColor: "#212529",
	removedGutterColor: "#212529",
	codeFoldContentColor: "#212529",
	diffViewerTitleBackground: "#fafbfc",
	diffViewerTitleColor: "#212529",
	diffViewerTitleBorderColor: "#eee"
}), _p = Object.freeze({
	diffViewerBackground: "#2e303c",
	diffViewerColor: "#FFF",
	addedBackground: "#2ea04326",
	addedColor: "white",
	removedBackground: "#f851491a",
	removedColor: "white",
	changedBackground: "#3e302c",
	wordAddedBackground: "#2ea04366",
	wordRemovedBackground: "#f8514966",
	addedGutterBackground: "#3fb9504d",
	removedGutterBackground: "#f851494d",
	gutterBackground: "#2c2f3a",
	gutterBackgroundDark: "#262933",
	highlightBackground: "#2a3967",
	highlightGutterBackground: "#2d4077",
	codeFoldGutterBackground: "#262831",
	codeFoldBackground: "#262831",
	emptyLineBackground: "#363946",
	gutterColor: "#f0f6fc",
	addedGutterColor: "#f0f6fc",
	removedGutterColor: "#f0f6fc",
	codeFoldContentColor: "#9198a1",
	diffViewerTitleBackground: "#2f323e",
	diffViewerTitleColor: "#f0f6fc",
	diffViewerTitleBorderColor: "#353846"
}), vp = (e, t = !1, n = "") => {
	let { variables: r = {}, ...i } = e, a = {
		light: {
			...gp,
			...r.light || {}
		},
		dark: {
			..._p,
			...r.dark || {}
		}
	}, o = t ? a.dark : a.light, { css: s, cx: c } = mp({
		key: "react-diff",
		nonce: n
	}), l = s({
		width: "auto",
		overflow: "hidden",
		padding: 0,
		label: "content"
	}), u = s({ label: "split-view" }), d = s({
		position: "sticky",
		top: 0,
		zIndex: 2,
		label: "sticky-header"
	}), f = s({
		background: o.diffViewerTitleBackground,
		color: o.diffViewerTitleColor,
		padding: "0.5em 1em",
		display: "flex",
		alignItems: "center",
		gap: "0.5em",
		fontFamily: "monospace",
		fontSize: 12,
		fill: o.diffViewerTitleColor
	}), p = s({
		display: "flex",
		label: "column-headers"
	}), m = s({
		width: "100%",
		minWidth: "1000px",
		overflowX: "auto",
		tableLayout: "fixed",
		fontSize: 12,
		background: o.diffViewerBackground,
		pre: {
			margin: 0,
			whiteSpace: "pre-wrap",
			lineHeight: "1.6em",
			width: "fit-content"
		},
		label: "diff-container",
		borderCollapse: "collapse",
		borderRadius: 0,
		"& td, & th": {
			border: 0,
			verticalAlign: "baseline"
		},
		"@media (max-width: 768px)": { minWidth: "unset" }
	}), h = s({
		overflow: "hidden",
		width: "100%"
	}), g = s({
		color: o.diffViewerColor,
		whiteSpace: "pre-wrap",
		fontFamily: "monospace",
		lineBreak: "anywhere",
		textDecoration: "none",
		label: "content-text"
	}), _ = s({
		display: "flex",
		alignItems: "baseline",
		label: "content-flex"
	}), v = s({
		whiteSpace: "pre",
		wordBreak: "keep-all",
		flex: "0 0 auto",
		label: "line-indent"
	}), y = s({
		flex: "1 1 auto",
		minWidth: 0,
		label: "line-body"
	}), b = s({
		userSelect: "none",
		label: "unselectable"
	}), x = s({
		label: "no-wrap",
		pre: { whiteSpace: "pre" },
		[`.${g}`]: {
			whiteSpace: "pre",
			lineBreak: "auto"
		}
	}), S = s({
		background: "transparent",
		border: "none",
		cursor: "pointer",
		display: "flex",
		alignItems: "center",
		justifyContent: "center",
		margin: 0,
		label: "all-expand-button",
		":hover": { fill: o.addedGutterColor },
		":focus": { outline: `1px ${o.addedGutterColor} solid` }
	}), C = s({
		background: o.diffViewerTitleBackground,
		padding: "0.5em",
		lineHeight: "1.4em",
		height: "2.4em",
		overflow: "hidden",
		width: "50%",
		borderBottom: `1px solid ${o.diffViewerTitleBorderColor}`,
		boxSizing: "border-box",
		fontSize: 12,
		label: "title-block",
		":only-child": { width: "100%" },
		":last-child:not(:only-child)": { borderLeft: `1px solid ${o.diffViewerTitleBorderColor}` },
		[`.${g}`]: { color: o.diffViewerTitleColor }
	}), w = s({
		color: o.gutterColor,
		label: "line-number"
	}), T = s({
		background: o.removedBackground,
		color: o.removedColor,
		pre: { color: o.removedColor },
		[`.${w}`]: { color: o.removedGutterColor },
		label: "diff-removed"
	}), E = s({
		background: o.addedBackground,
		color: o.addedColor,
		pre: { color: o.addedColor },
		[`.${w}`]: { color: o.addedGutterColor },
		label: "diff-added"
	}), D = s({
		background: o.changedBackground,
		[`.${w}`]: { color: o.gutterColor },
		label: "diff-changed"
	}), O = s({
		display: "inline",
		textDecoration: "none",
		label: "word-diff"
	}), ee = s({
		background: o.wordAddedBackground,
		label: "word-added"
	}), k = s({
		background: o.wordRemovedBackground,
		label: "word-removed"
	}), A = s({
		backgroundColor: o.codeFoldGutterBackground,
		label: "code-fold-gutter",
		minWidth: "50px",
		width: "50px",
		padding: 0,
		textAlign: "center",
		fill: o.codeFoldContentColor
	}), te = s({
		padding: 0,
		overflow: "hidden",
		"&&&": { verticalAlign: "middle" },
		label: "code-fold-content-container"
	}), j = s({
		background: o.codeFoldBackground,
		cursor: "pointer",
		display: "block",
		margin: 0,
		padding: 0,
		border: "none",
		font: "inherit",
		lineHeight: "inherit",
		textAlign: "left",
		fill: o.codeFoldContentColor,
		label: "code-fold-expand-button"
	}), ne = s({
		color: o.codeFoldContentColor,
		fontFamily: "monospace",
		label: "code-fold-content"
	}), re = s({
		display: "block",
		width: "10px",
		height: "10px",
		backgroundColor: "#ddd",
		borderWidth: "1px",
		borderStyle: "solid",
		borderColor: o.diffViewerTitleBorderColor
	}), M = s({ backgroundColor: o.wordAddedBackground }), ie = s({ backgroundColor: o.wordRemovedBackground }), N = s({
		backgroundColor: o.codeFoldBackground,
		fontSize: 12,
		alignItems: "center",
		userSelect: "none",
		fontWeight: 700,
		cursor: "pointer",
		label: "code-fold",
		"& td": {
			paddingTop: 0,
			paddingBottom: 0
		},
		"&:hover": {
			color: o.diffViewerColor,
			fill: o.diffViewerColor,
			"& *": {
				color: o.diffViewerColor,
				fill: o.diffViewerColor
			}
		},
		a: {
			textDecoration: "underline !important",
			cursor: "pointer",
			pre: { display: "inline" }
		}
	}), ae = s({
		backgroundColor: o.emptyLineBackground,
		label: "empty-line"
	}), oe = s({
		width: 28,
		paddingLeft: 10,
		paddingRight: 10,
		paddingTop: 0,
		paddingBottom: 0,
		userSelect: "none",
		label: "marker",
		[`&.${E}`]: { pre: { color: o.addedColor } },
		[`&.${T}`]: { pre: { color: o.removedColor } }
	}), se = s({
		background: o.highlightBackground,
		label: "highlighted-line",
		[`.${ee}, .${k}`]: { backgroundColor: "initial" }
	}), ce = s({ label: "highlighted-gutter" }), le = s({
		userSelect: "none",
		minWidth: 50,
		width: "50px",
		padding: "0 10px",
		whiteSpace: "nowrap",
		label: "gutter",
		textAlign: "center",
		color: o.gutterColor,
		background: o.gutterBackground,
		"&:hover": {
			cursor: "pointer",
			background: o.gutterBackgroundDark,
			pre: { opacity: 1 }
		},
		pre: {
			opacity: .5,
			textAlign: "center",
			width: "100%"
		},
		[`&.${E}`]: { background: o.addedGutterBackground },
		[`&.${T}`]: { background: o.removedGutterBackground },
		[`&.${ce}`]: {
			background: o.highlightGutterBackground,
			"&:hover": { background: o.highlightGutterBackground }
		}
	}), ue = s({
		"&:hover": {
			background: o.gutterBackground,
			cursor: "initial"
		},
		label: "empty-gutter"
	}), P = {
		diffContainer: m,
		diffRemoved: T,
		diffAdded: E,
		diffChanged: D,
		splitView: u,
		marker: oe,
		highlightedGutter: ce,
		highlightedLine: se,
		gutter: le,
		line: s({
			verticalAlign: "baseline",
			label: "line",
			textDecoration: "none",
			backgroundColor: "transparent",
			fontSize: 12
		}),
		lineContent: h,
		wordDiff: O,
		wordAdded: ee,
		summary: f,
		block: re,
		blockAddition: M,
		blockDeletion: ie,
		wordRemoved: k,
		noSelect: b,
		noWrap: x,
		codeFoldGutter: A,
		codeFoldExpandButton: j,
		codeFoldContentContainer: te,
		codeFold: N,
		emptyGutter: ue,
		emptyLine: ae,
		lineNumber: w,
		contentText: g,
		contentFlex: _,
		lineIndent: v,
		lineBody: y,
		content: l,
		column: s({}),
		codeFoldContent: ne,
		stickyHeader: d,
		columnHeaders: p,
		titleBlock: C,
		allExpandButton: S
	}, de = Object.keys(i).reduce((e, t) => ({
		...e,
		[t]: s(i[t])
	}), {});
	return Object.keys(P).reduce((e, t) => ({
		...e,
		[t]: de[t] ? c(P[t], de[t]) : P[t]
	}), {});
};
//#endregion
//#region node_modules/.pnpm/react-diff-viewer-continued@4.4.0_@types+react@19.2.18_react-dom@19.2.8_react@19.2.8__react@19.2.8/node_modules/react-diff-viewer-continued/lib/esm/src/fold.js
function yp() {
	return m("svg", {
		xmlns: "http://www.w3.org/2000/svg",
		viewBox: "0 0 16 16",
		width: "16",
		height: "16",
		children: [p("title", { children: "fold" }), p("path", { d: "M10.896 2H8.75V.75a.75.75 0 0 0-1.5 0V2H5.104a.25.25 0 0 0-.177.427l2.896 2.896a.25.25 0 0 0 .354 0l2.896-2.896A.25.25 0 0 0 10.896 2ZM8.75 15.25a.75.75 0 0 1-1.5 0V14H5.104a.25.25 0 0 1-.177-.427l2.896-2.896a.25.25 0 0 1 .354 0l2.896 2.896a.25.25 0 0 1-.177.427H8.75v1.25Zm-6.5-6.5a.75.75 0 0 0 0-1.5h-.5a.75.75 0 0 0 0 1.5h.5ZM6 8a.75.75 0 0 1-.75.75h-.5a.75.75 0 0 1 0-1.5h.5A.75.75 0 0 1 6 8Zm2.25.75a.75.75 0 0 0 0-1.5h-.5a.75.75 0 0 0 0 1.5h.5ZM12 8a.75.75 0 0 1-.75.75h-.5a.75.75 0 0 1 0-1.5h.5A.75.75 0 0 1 12 8Zm2.25.75a.75.75 0 0 0 0-1.5h-.5a.75.75 0 0 0 0 1.5h.5Z" })]
	});
}
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/lib/util/schema.js
var bp = class {
	constructor(e, t, n) {
		this.normal = t, this.property = e, n && (this.space = n);
	}
};
bp.prototype.normal = {}, bp.prototype.property = {}, bp.prototype.space = void 0;
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/lib/util/merge.js
function xp(e, t) {
	let n = {}, r = {};
	for (let t of e) Object.assign(n, t.property), Object.assign(r, t.normal);
	return new bp(n, r, t);
}
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/lib/normalize.js
function Sp(e) {
	return e.toLowerCase();
}
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/lib/util/info.js
var Cp = class {
	constructor(e, t) {
		this.attribute = t, this.property = e;
	}
};
Cp.prototype.attribute = "", Cp.prototype.booleanish = !1, Cp.prototype.boolean = !1, Cp.prototype.commaOrSpaceSeparated = !1, Cp.prototype.commaSeparated = !1, Cp.prototype.defined = !1, Cp.prototype.mustUseProperty = !1, Cp.prototype.number = !1, Cp.prototype.overloadedBoolean = !1, Cp.prototype.property = "", Cp.prototype.spaceSeparated = !1, Cp.prototype.space = void 0;
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/lib/util/types.js
var wp = /* @__PURE__ */ z({
	boolean: () => H,
	booleanish: () => Ep,
	commaOrSpaceSeparated: () => Ap,
	commaSeparated: () => kp,
	number: () => U,
	overloadedBoolean: () => Dp,
	spaceSeparated: () => Op
}), Tp = 0, H = jp(), Ep = jp(), Dp = jp(), U = jp(), Op = jp(), kp = jp(), Ap = jp();
function jp() {
	return 2 ** ++Tp;
}
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/lib/util/defined-info.js
var Mp = Object.keys(wp), Np = class extends Cp {
	constructor(e, t, n, r) {
		let i = -1;
		if (super(e, t), Pp(this, "space", r), typeof n == "number") for (; ++i < Mp.length;) {
			let e = Mp[i];
			Pp(this, Mp[i], (n & wp[e]) === wp[e]);
		}
	}
};
Np.prototype.defined = !0;
function Pp(e, t, n) {
	n && (e[t] = n);
}
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/lib/util/create.js
function Fp(e) {
	let t = {}, n = {};
	for (let [r, i] of Object.entries(e.properties)) {
		let a = new Np(r, e.transform(e.attributes || {}, r), i, e.space);
		e.mustUseProperty && e.mustUseProperty.includes(r) && (a.mustUseProperty = !0), t[r] = a, n[Sp(r)] = r, n[Sp(a.attribute)] = r;
	}
	return new bp(t, n, e.space);
}
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/lib/aria.js
var Ip = Fp({
	properties: {
		ariaActiveDescendant: null,
		ariaAtomic: Ep,
		ariaAutoComplete: null,
		ariaBusy: Ep,
		ariaChecked: Ep,
		ariaColCount: U,
		ariaColIndex: U,
		ariaColSpan: U,
		ariaControls: Op,
		ariaCurrent: null,
		ariaDescribedBy: Op,
		ariaDetails: null,
		ariaDisabled: Ep,
		ariaDropEffect: Op,
		ariaErrorMessage: null,
		ariaExpanded: Ep,
		ariaFlowTo: Op,
		ariaGrabbed: Ep,
		ariaHasPopup: null,
		ariaHidden: Ep,
		ariaInvalid: null,
		ariaKeyShortcuts: null,
		ariaLabel: null,
		ariaLabelledBy: Op,
		ariaLevel: U,
		ariaLive: null,
		ariaModal: Ep,
		ariaMultiLine: Ep,
		ariaMultiSelectable: Ep,
		ariaOrientation: null,
		ariaOwns: Op,
		ariaPlaceholder: null,
		ariaPosInSet: U,
		ariaPressed: Ep,
		ariaReadOnly: Ep,
		ariaRelevant: null,
		ariaRequired: Ep,
		ariaRoleDescription: Op,
		ariaRowCount: U,
		ariaRowIndex: U,
		ariaRowSpan: U,
		ariaSelected: Ep,
		ariaSetSize: U,
		ariaSort: null,
		ariaValueMax: U,
		ariaValueMin: U,
		ariaValueNow: U,
		ariaValueText: null,
		role: null
	},
	transform(e, t) {
		return t === "role" ? t : "aria-" + t.slice(4).toLowerCase();
	}
});
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/lib/util/case-sensitive-transform.js
function Lp(e, t) {
	return t in e ? e[t] : t;
}
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/lib/util/case-insensitive-transform.js
function Rp(e, t) {
	return Lp(e, t.toLowerCase());
}
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/lib/html.js
var zp = Fp({
	attributes: {
		acceptcharset: "accept-charset",
		classname: "class",
		htmlfor: "for",
		httpequiv: "http-equiv"
	},
	mustUseProperty: [
		"checked",
		"multiple",
		"muted",
		"selected"
	],
	properties: {
		abbr: null,
		accept: kp,
		acceptCharset: Op,
		accessKey: Op,
		action: null,
		allow: null,
		allowFullScreen: H,
		allowPaymentRequest: H,
		allowUserMedia: H,
		alpha: H,
		alt: null,
		as: null,
		async: H,
		autoCapitalize: null,
		autoComplete: Op,
		autoFocus: H,
		autoPlay: H,
		blocking: Op,
		capture: null,
		charSet: null,
		checked: H,
		cite: null,
		className: Op,
		closedBy: null,
		colorSpace: null,
		cols: U,
		colSpan: U,
		command: null,
		commandFor: null,
		content: null,
		contentEditable: Ep,
		controls: H,
		controlsList: Op,
		coords: U | kp,
		crossOrigin: null,
		data: null,
		dateTime: null,
		decoding: null,
		default: H,
		defer: H,
		dir: null,
		dirName: null,
		disabled: H,
		download: Dp,
		draggable: Ep,
		encType: null,
		enterKeyHint: null,
		fetchPriority: null,
		form: null,
		formAction: null,
		formEncType: null,
		formMethod: null,
		formNoValidate: H,
		formTarget: null,
		headers: Op,
		height: U,
		hidden: Dp,
		high: U,
		href: null,
		hrefLang: null,
		htmlFor: Op,
		httpEquiv: Op,
		id: null,
		imageSizes: null,
		imageSrcSet: null,
		inert: H,
		inputMode: null,
		integrity: null,
		is: null,
		isMap: H,
		itemId: null,
		itemProp: Op,
		itemRef: Op,
		itemScope: H,
		itemType: Op,
		kind: null,
		label: null,
		lang: null,
		language: null,
		list: null,
		loading: null,
		loop: H,
		low: U,
		manifest: null,
		max: null,
		maxLength: U,
		media: null,
		method: null,
		min: null,
		minLength: U,
		multiple: H,
		muted: H,
		name: null,
		nonce: null,
		noModule: H,
		noValidate: H,
		onAbort: null,
		onAfterPrint: null,
		onAuxClick: null,
		onBeforeMatch: null,
		onBeforePrint: null,
		onBeforeToggle: null,
		onBeforeUnload: null,
		onBlur: null,
		onCancel: null,
		onCanPlay: null,
		onCanPlayThrough: null,
		onChange: null,
		onClick: null,
		onClose: null,
		onContextLost: null,
		onContextMenu: null,
		onContextRestored: null,
		onCopy: null,
		onCueChange: null,
		onCut: null,
		onDblClick: null,
		onDrag: null,
		onDragEnd: null,
		onDragEnter: null,
		onDragExit: null,
		onDragLeave: null,
		onDragOver: null,
		onDragStart: null,
		onDrop: null,
		onDurationChange: null,
		onEmptied: null,
		onEnded: null,
		onError: null,
		onFocus: null,
		onFormData: null,
		onHashChange: null,
		onInput: null,
		onInvalid: null,
		onKeyDown: null,
		onKeyPress: null,
		onKeyUp: null,
		onLanguageChange: null,
		onLoad: null,
		onLoadedData: null,
		onLoadedMetadata: null,
		onLoadEnd: null,
		onLoadStart: null,
		onMessage: null,
		onMessageError: null,
		onMouseDown: null,
		onMouseEnter: null,
		onMouseLeave: null,
		onMouseMove: null,
		onMouseOut: null,
		onMouseOver: null,
		onMouseUp: null,
		onOffline: null,
		onOnline: null,
		onPageHide: null,
		onPageShow: null,
		onPaste: null,
		onPause: null,
		onPlay: null,
		onPlaying: null,
		onPopState: null,
		onProgress: null,
		onRateChange: null,
		onRejectionHandled: null,
		onReset: null,
		onResize: null,
		onScroll: null,
		onScrollEnd: null,
		onSecurityPolicyViolation: null,
		onSeeked: null,
		onSeeking: null,
		onSelect: null,
		onSlotChange: null,
		onStalled: null,
		onStorage: null,
		onSubmit: null,
		onSuspend: null,
		onTimeUpdate: null,
		onToggle: null,
		onUnhandledRejection: null,
		onUnload: null,
		onVolumeChange: null,
		onWaiting: null,
		onWheel: null,
		open: H,
		optimum: U,
		pattern: null,
		ping: Op,
		placeholder: null,
		playsInline: H,
		popover: null,
		popoverTarget: null,
		popoverTargetAction: null,
		poster: null,
		preload: null,
		readOnly: H,
		referrerPolicy: null,
		rel: Op,
		required: H,
		reversed: H,
		rows: U,
		rowSpan: U,
		sandbox: Op,
		scope: null,
		scoped: H,
		seamless: H,
		selected: H,
		shadowRootClonable: H,
		shadowRootCustomElementRegistry: H,
		shadowRootDelegatesFocus: H,
		shadowRootMode: null,
		shadowRootSerializable: H,
		shape: null,
		size: U,
		sizes: null,
		slot: null,
		span: U,
		spellCheck: Ep,
		src: null,
		srcDoc: null,
		srcLang: null,
		srcSet: null,
		start: U,
		step: null,
		style: null,
		tabIndex: U,
		target: null,
		title: null,
		translate: null,
		type: null,
		typeMustMatch: H,
		useMap: null,
		value: Ep,
		width: U,
		wrap: null,
		writingSuggestions: null,
		align: null,
		aLink: null,
		archive: Op,
		axis: null,
		background: null,
		bgColor: null,
		border: U,
		borderColor: null,
		bottomMargin: U,
		cellPadding: null,
		cellSpacing: null,
		char: null,
		charOff: null,
		classId: null,
		clear: null,
		code: null,
		codeBase: null,
		codeType: null,
		color: null,
		compact: H,
		declare: H,
		event: null,
		face: null,
		frame: null,
		frameBorder: null,
		hSpace: U,
		leftMargin: U,
		link: null,
		longDesc: null,
		lowSrc: null,
		marginHeight: U,
		marginWidth: U,
		noResize: H,
		noHref: H,
		noShade: H,
		noWrap: H,
		object: null,
		profile: null,
		prompt: null,
		rev: null,
		rightMargin: U,
		rules: null,
		scheme: null,
		scrolling: Ep,
		standby: null,
		summary: null,
		text: null,
		topMargin: U,
		valueType: null,
		version: null,
		vAlign: null,
		vLink: null,
		vSpace: U,
		allowTransparency: null,
		autoCorrect: null,
		autoSave: null,
		credentialless: H,
		disablePictureInPicture: H,
		disableRemotePlayback: H,
		exportParts: kp,
		part: Op,
		prefix: null,
		property: null,
		results: U,
		security: null,
		unselectable: null
	},
	space: "html",
	transform: Rp
}), Bp = Fp({
	attributes: {
		accentHeight: "accent-height",
		alignmentBaseline: "alignment-baseline",
		arabicForm: "arabic-form",
		baselineShift: "baseline-shift",
		capHeight: "cap-height",
		className: "class",
		clipPath: "clip-path",
		clipRule: "clip-rule",
		colorInterpolation: "color-interpolation",
		colorInterpolationFilters: "color-interpolation-filters",
		colorProfile: "color-profile",
		colorRendering: "color-rendering",
		crossOrigin: "crossorigin",
		dataType: "datatype",
		dominantBaseline: "dominant-baseline",
		enableBackground: "enable-background",
		fillOpacity: "fill-opacity",
		fillRule: "fill-rule",
		floodColor: "flood-color",
		floodOpacity: "flood-opacity",
		fontFamily: "font-family",
		fontSize: "font-size",
		fontSizeAdjust: "font-size-adjust",
		fontStretch: "font-stretch",
		fontStyle: "font-style",
		fontVariant: "font-variant",
		fontWeight: "font-weight",
		glyphName: "glyph-name",
		glyphOrientationHorizontal: "glyph-orientation-horizontal",
		glyphOrientationVertical: "glyph-orientation-vertical",
		hrefLang: "hreflang",
		horizAdvX: "horiz-adv-x",
		horizOriginX: "horiz-origin-x",
		horizOriginY: "horiz-origin-y",
		imageRendering: "image-rendering",
		letterSpacing: "letter-spacing",
		lightingColor: "lighting-color",
		markerEnd: "marker-end",
		markerMid: "marker-mid",
		markerStart: "marker-start",
		maskType: "mask-type",
		navDown: "nav-down",
		navDownLeft: "nav-down-left",
		navDownRight: "nav-down-right",
		navLeft: "nav-left",
		navNext: "nav-next",
		navPrev: "nav-prev",
		navRight: "nav-right",
		navUp: "nav-up",
		navUpLeft: "nav-up-left",
		navUpRight: "nav-up-right",
		onAbort: "onabort",
		onActivate: "onactivate",
		onAfterPrint: "onafterprint",
		onBeforePrint: "onbeforeprint",
		onBegin: "onbegin",
		onCancel: "oncancel",
		onCanPlay: "oncanplay",
		onCanPlayThrough: "oncanplaythrough",
		onChange: "onchange",
		onClick: "onclick",
		onClose: "onclose",
		onCopy: "oncopy",
		onCueChange: "oncuechange",
		onCut: "oncut",
		onDblClick: "ondblclick",
		onDrag: "ondrag",
		onDragEnd: "ondragend",
		onDragEnter: "ondragenter",
		onDragExit: "ondragexit",
		onDragLeave: "ondragleave",
		onDragOver: "ondragover",
		onDragStart: "ondragstart",
		onDrop: "ondrop",
		onDurationChange: "ondurationchange",
		onEmptied: "onemptied",
		onEnd: "onend",
		onEnded: "onended",
		onError: "onerror",
		onFocus: "onfocus",
		onFocusIn: "onfocusin",
		onFocusOut: "onfocusout",
		onHashChange: "onhashchange",
		onInput: "oninput",
		onInvalid: "oninvalid",
		onKeyDown: "onkeydown",
		onKeyPress: "onkeypress",
		onKeyUp: "onkeyup",
		onLoad: "onload",
		onLoadedData: "onloadeddata",
		onLoadedMetadata: "onloadedmetadata",
		onLoadStart: "onloadstart",
		onMessage: "onmessage",
		onMouseDown: "onmousedown",
		onMouseEnter: "onmouseenter",
		onMouseLeave: "onmouseleave",
		onMouseMove: "onmousemove",
		onMouseOut: "onmouseout",
		onMouseOver: "onmouseover",
		onMouseUp: "onmouseup",
		onMouseWheel: "onmousewheel",
		onOffline: "onoffline",
		onOnline: "ononline",
		onPageHide: "onpagehide",
		onPageShow: "onpageshow",
		onPaste: "onpaste",
		onPause: "onpause",
		onPlay: "onplay",
		onPlaying: "onplaying",
		onPopState: "onpopstate",
		onProgress: "onprogress",
		onRateChange: "onratechange",
		onRepeat: "onrepeat",
		onReset: "onreset",
		onResize: "onresize",
		onScroll: "onscroll",
		onSeeked: "onseeked",
		onSeeking: "onseeking",
		onSelect: "onselect",
		onShow: "onshow",
		onStalled: "onstalled",
		onStorage: "onstorage",
		onSubmit: "onsubmit",
		onSuspend: "onsuspend",
		onTimeUpdate: "ontimeupdate",
		onToggle: "ontoggle",
		onUnload: "onunload",
		onVolumeChange: "onvolumechange",
		onWaiting: "onwaiting",
		onZoom: "onzoom",
		overlinePosition: "overline-position",
		overlineThickness: "overline-thickness",
		paintOrder: "paint-order",
		panose1: "panose-1",
		pointerEvents: "pointer-events",
		referrerPolicy: "referrerpolicy",
		renderingIntent: "rendering-intent",
		shapeRendering: "shape-rendering",
		stopColor: "stop-color",
		stopOpacity: "stop-opacity",
		strikethroughPosition: "strikethrough-position",
		strikethroughThickness: "strikethrough-thickness",
		strokeDashArray: "stroke-dasharray",
		strokeDashOffset: "stroke-dashoffset",
		strokeLineCap: "stroke-linecap",
		strokeLineJoin: "stroke-linejoin",
		strokeMiterLimit: "stroke-miterlimit",
		strokeOpacity: "stroke-opacity",
		strokeWidth: "stroke-width",
		tabIndex: "tabindex",
		textAnchor: "text-anchor",
		textDecoration: "text-decoration",
		textRendering: "text-rendering",
		transformOrigin: "transform-origin",
		typeOf: "typeof",
		underlinePosition: "underline-position",
		underlineThickness: "underline-thickness",
		unicodeBidi: "unicode-bidi",
		unicodeRange: "unicode-range",
		unitsPerEm: "units-per-em",
		vAlphabetic: "v-alphabetic",
		vHanging: "v-hanging",
		vIdeographic: "v-ideographic",
		vMathematical: "v-mathematical",
		vectorEffect: "vector-effect",
		vertAdvY: "vert-adv-y",
		vertOriginX: "vert-origin-x",
		vertOriginY: "vert-origin-y",
		wordSpacing: "word-spacing",
		writingMode: "writing-mode",
		xHeight: "x-height",
		playbackOrder: "playbackorder",
		timelineBegin: "timelinebegin"
	},
	properties: {
		about: Ap,
		accentHeight: U,
		accumulate: null,
		additive: null,
		alignmentBaseline: null,
		alphabetic: U,
		amplitude: U,
		arabicForm: null,
		ascent: U,
		attributeName: null,
		attributeType: null,
		azimuth: U,
		bandwidth: null,
		baselineShift: null,
		baseFrequency: null,
		baseProfile: null,
		bbox: null,
		begin: null,
		bias: U,
		by: null,
		calcMode: null,
		capHeight: U,
		className: Op,
		clip: null,
		clipPath: null,
		clipPathUnits: null,
		clipRule: null,
		color: null,
		colorInterpolation: null,
		colorInterpolationFilters: null,
		colorProfile: null,
		colorRendering: null,
		content: null,
		contentScriptType: null,
		contentStyleType: null,
		crossOrigin: null,
		cursor: null,
		cx: null,
		cy: null,
		d: null,
		dataType: null,
		defaultAction: null,
		descent: U,
		diffuseConstant: U,
		direction: null,
		display: null,
		dur: null,
		divisor: U,
		dominantBaseline: null,
		download: H,
		dx: null,
		dy: null,
		edgeMode: null,
		editable: null,
		elevation: U,
		enableBackground: null,
		end: null,
		event: null,
		exponent: U,
		externalResourcesRequired: null,
		fill: null,
		fillOpacity: U,
		fillRule: null,
		filter: null,
		filterRes: null,
		filterUnits: null,
		floodColor: null,
		floodOpacity: null,
		focusable: null,
		focusHighlight: null,
		fontFamily: null,
		fontSize: null,
		fontSizeAdjust: null,
		fontStretch: null,
		fontStyle: null,
		fontVariant: null,
		fontWeight: null,
		format: null,
		fr: null,
		from: null,
		fx: null,
		fy: null,
		g1: kp,
		g2: kp,
		glyphName: kp,
		glyphOrientationHorizontal: null,
		glyphOrientationVertical: null,
		glyphRef: null,
		gradientTransform: null,
		gradientUnits: null,
		handler: null,
		hanging: U,
		hatchContentUnits: null,
		hatchUnits: null,
		height: null,
		href: null,
		hrefLang: null,
		horizAdvX: U,
		horizOriginX: U,
		horizOriginY: U,
		id: null,
		ideographic: U,
		imageRendering: null,
		initialVisibility: null,
		in: null,
		in2: null,
		intercept: U,
		k: U,
		k1: U,
		k2: U,
		k3: U,
		k4: U,
		kernelMatrix: Ap,
		kernelUnitLength: null,
		keyPoints: null,
		keySplines: null,
		keyTimes: null,
		kerning: null,
		lang: null,
		lengthAdjust: null,
		letterSpacing: null,
		lightingColor: null,
		limitingConeAngle: U,
		local: null,
		markerEnd: null,
		markerMid: null,
		markerStart: null,
		markerHeight: null,
		markerUnits: null,
		markerWidth: null,
		mask: null,
		maskContentUnits: null,
		maskType: null,
		maskUnits: null,
		mathematical: null,
		max: null,
		media: null,
		mediaCharacterEncoding: null,
		mediaContentEncodings: null,
		mediaSize: U,
		mediaTime: null,
		method: null,
		min: null,
		mode: null,
		name: null,
		navDown: null,
		navDownLeft: null,
		navDownRight: null,
		navLeft: null,
		navNext: null,
		navPrev: null,
		navRight: null,
		navUp: null,
		navUpLeft: null,
		navUpRight: null,
		numOctaves: null,
		observer: null,
		offset: null,
		onAbort: null,
		onActivate: null,
		onAfterPrint: null,
		onBeforePrint: null,
		onBegin: null,
		onCancel: null,
		onCanPlay: null,
		onCanPlayThrough: null,
		onChange: null,
		onClick: null,
		onClose: null,
		onCopy: null,
		onCueChange: null,
		onCut: null,
		onDblClick: null,
		onDrag: null,
		onDragEnd: null,
		onDragEnter: null,
		onDragExit: null,
		onDragLeave: null,
		onDragOver: null,
		onDragStart: null,
		onDrop: null,
		onDurationChange: null,
		onEmptied: null,
		onEnd: null,
		onEnded: null,
		onError: null,
		onFocus: null,
		onFocusIn: null,
		onFocusOut: null,
		onHashChange: null,
		onInput: null,
		onInvalid: null,
		onKeyDown: null,
		onKeyPress: null,
		onKeyUp: null,
		onLoad: null,
		onLoadedData: null,
		onLoadedMetadata: null,
		onLoadStart: null,
		onMessage: null,
		onMouseDown: null,
		onMouseEnter: null,
		onMouseLeave: null,
		onMouseMove: null,
		onMouseOut: null,
		onMouseOver: null,
		onMouseUp: null,
		onMouseWheel: null,
		onOffline: null,
		onOnline: null,
		onPageHide: null,
		onPageShow: null,
		onPaste: null,
		onPause: null,
		onPlay: null,
		onPlaying: null,
		onPopState: null,
		onProgress: null,
		onRateChange: null,
		onRepeat: null,
		onReset: null,
		onResize: null,
		onScroll: null,
		onSeeked: null,
		onSeeking: null,
		onSelect: null,
		onShow: null,
		onStalled: null,
		onStorage: null,
		onSubmit: null,
		onSuspend: null,
		onTimeUpdate: null,
		onToggle: null,
		onUnload: null,
		onVolumeChange: null,
		onWaiting: null,
		onZoom: null,
		opacity: null,
		operator: null,
		order: null,
		orient: null,
		orientation: null,
		origin: null,
		overflow: null,
		overlay: null,
		overlinePosition: U,
		overlineThickness: U,
		paintOrder: null,
		panose1: null,
		path: null,
		pathLength: U,
		patternContentUnits: null,
		patternTransform: null,
		patternUnits: null,
		phase: null,
		ping: Op,
		pitch: null,
		playbackOrder: null,
		pointerEvents: null,
		points: null,
		pointsAtX: U,
		pointsAtY: U,
		pointsAtZ: U,
		preserveAlpha: null,
		preserveAspectRatio: null,
		primitiveUnits: null,
		propagate: null,
		property: Ap,
		r: null,
		radius: null,
		referrerPolicy: null,
		refX: null,
		refY: null,
		rel: Ap,
		rev: Ap,
		renderingIntent: null,
		repeatCount: null,
		repeatDur: null,
		requiredExtensions: Ap,
		requiredFeatures: Ap,
		requiredFonts: Ap,
		requiredFormats: Ap,
		resource: null,
		restart: null,
		result: null,
		rotate: null,
		rx: null,
		ry: null,
		scale: null,
		seed: null,
		shapeRendering: null,
		side: null,
		slope: null,
		snapshotTime: null,
		specularConstant: U,
		specularExponent: U,
		spreadMethod: null,
		spacing: null,
		startOffset: null,
		stdDeviation: null,
		stemh: null,
		stemv: null,
		stitchTiles: null,
		stopColor: null,
		stopOpacity: null,
		strikethroughPosition: U,
		strikethroughThickness: U,
		string: null,
		stroke: null,
		strokeDashArray: Ap,
		strokeDashOffset: null,
		strokeLineCap: null,
		strokeLineJoin: null,
		strokeMiterLimit: U,
		strokeOpacity: U,
		strokeWidth: null,
		style: null,
		surfaceScale: U,
		syncBehavior: null,
		syncBehaviorDefault: null,
		syncMaster: null,
		syncTolerance: null,
		syncToleranceDefault: null,
		systemLanguage: Ap,
		tabIndex: U,
		tableValues: null,
		target: null,
		targetX: U,
		targetY: U,
		textAnchor: null,
		textDecoration: null,
		textRendering: null,
		textLength: null,
		timelineBegin: null,
		title: null,
		transformBehavior: null,
		type: null,
		typeOf: Ap,
		to: null,
		transform: null,
		transformOrigin: null,
		u1: null,
		u2: null,
		underlinePosition: U,
		underlineThickness: U,
		unicode: null,
		unicodeBidi: null,
		unicodeRange: null,
		unitsPerEm: U,
		values: null,
		vAlphabetic: U,
		vMathematical: U,
		vectorEffect: null,
		vHanging: U,
		vIdeographic: U,
		version: null,
		vertAdvY: U,
		vertOriginX: U,
		vertOriginY: U,
		viewBox: null,
		viewTarget: null,
		visibility: null,
		width: null,
		widths: null,
		wordSpacing: null,
		writingMode: null,
		x: null,
		x1: null,
		x2: null,
		xChannelSelector: null,
		xHeight: U,
		y: null,
		y1: null,
		y2: null,
		yChannelSelector: null,
		z: null,
		zoomAndPan: null
	},
	space: "svg",
	transform: Lp
}), Vp = Fp({
	properties: {
		xLinkActuate: null,
		xLinkArcRole: null,
		xLinkHref: null,
		xLinkRole: null,
		xLinkShow: null,
		xLinkTitle: null,
		xLinkType: null
	},
	space: "xlink",
	transform(e, t) {
		return "xlink:" + t.slice(5).toLowerCase();
	}
}), Hp = Fp({
	attributes: { xmlnsxlink: "xmlns:xlink" },
	properties: {
		xmlnsXLink: null,
		xmlns: null
	},
	space: "xmlns",
	transform: Rp
}), Up = Fp({
	properties: {
		xmlBase: null,
		xmlLang: null,
		xmlSpace: null
	},
	space: "xml",
	transform(e, t) {
		return "xml:" + t.slice(3).toLowerCase();
	}
}), Wp = /[A-Z]/g, Gp = /-[a-z]/g, Kp = /^data[-\w.:]+$/i;
function qp(e, t) {
	let n = Sp(t), r = t, i = Cp;
	if (n in e.normal) return e.property[e.normal[n]];
	if (n.length > 4 && n.slice(0, 4) === "data" && Kp.test(t)) {
		if (t.charAt(4) === "-") {
			let e = t.slice(5).replace(Gp, Yp);
			r = "data" + e.charAt(0).toUpperCase() + e.slice(1);
		} else {
			let e = t.slice(4);
			if (!Gp.test(e)) {
				let n = e.replace(Wp, Jp);
				n.charAt(0) !== "-" && (n = "-" + n), t = "data" + n;
			}
		}
		i = Np;
	}
	return new i(r, t);
}
function Jp(e) {
	return "-" + e.toLowerCase();
}
function Yp(e) {
	return e.charAt(1).toUpperCase();
}
//#endregion
//#region node_modules/.pnpm/property-information@7.2.0/node_modules/property-information/index.js
var Xp = xp([
	Ip,
	zp,
	Vp,
	Hp,
	Up
], "html"), Zp = xp([
	Ip,
	Bp,
	Vp,
	Hp,
	Up
], "svg");
//#endregion
//#region node_modules/.pnpm/comma-separated-tokens@2.0.3/node_modules/comma-separated-tokens/index.js
function Qp(e) {
	let t = [], n = String(e || ""), r = n.indexOf(","), i = 0, a = !1;
	for (; !a;) {
		r === -1 && (r = n.length, a = !0);
		let e = n.slice(i, r).trim();
		(e || !a) && t.push(e), i = r + 1, r = n.indexOf(",", i);
	}
	return t;
}
//#endregion
//#region node_modules/.pnpm/hast-util-parse-selector@4.0.0/node_modules/hast-util-parse-selector/lib/index.js
var $p = /[#.]/g;
function em(e, t) {
	let n = e || "", r = {}, i = 0, a, o;
	for (; i < n.length;) {
		$p.lastIndex = i;
		let e = $p.exec(n), t = n.slice(i, e ? e.index : n.length);
		t && (a ? a === "#" ? r.id = t : Array.isArray(r.className) ? r.className.push(t) : r.className = [t] : o = t, i += t.length), e && (a = e[0], i++);
	}
	return {
		type: "element",
		tagName: o || t || "div",
		properties: r,
		children: []
	};
}
//#endregion
//#region node_modules/.pnpm/space-separated-tokens@2.0.2/node_modules/space-separated-tokens/index.js
function tm(e) {
	let t = String(e || "").trim();
	return t ? t.split(/[ \t\n\r\f]+/g) : [];
}
//#endregion
//#region node_modules/.pnpm/hastscript@9.0.1/node_modules/hastscript/lib/create-h.js
function nm(e, t, n) {
	let r = n ? cm(n) : void 0;
	function i(n, i, ...a) {
		let o;
		if (n == null) {
			o = {
				type: "root",
				children: []
			};
			let e = i;
			a.unshift(e);
		} else {
			o = em(n, t);
			let s = o.tagName.toLowerCase(), c = r ? r.get(s) : void 0;
			if (o.tagName = c || s, rm(i)) a.unshift(i);
			else for (let [t, n] of Object.entries(i)) im(e, o.properties, t, n);
		}
		for (let e of a) am(o.children, e);
		return o.type === "element" && o.tagName === "template" && (o.content = {
			type: "root",
			children: o.children
		}, o.children = []), o;
	}
	return i;
}
function rm(e) {
	if (typeof e != "object" || !e || Array.isArray(e)) return !0;
	if (typeof e.type != "string") return !1;
	let t = e, n = Object.keys(e);
	for (let e of n) {
		let n = t[e];
		if (n && typeof n == "object") {
			if (!Array.isArray(n)) return !0;
			let e = n;
			for (let t of e) if (typeof t != "number" && typeof t != "string") return !0;
		}
	}
	return !!("children" in e && Array.isArray(e.children));
}
function im(e, t, n, r) {
	let i = qp(e, n), a;
	if (r != null) {
		if (typeof r == "number") {
			if (Number.isNaN(r)) return;
			a = r;
		} else a = typeof r == "boolean" ? r : typeof r == "string" ? i.spaceSeparated ? tm(r) : i.commaSeparated ? Qp(r) : i.commaOrSpaceSeparated ? tm(Qp(r).join(" ")) : om(i, i.property, r) : Array.isArray(r) ? [...r] : i.property === "style" ? sm(r) : String(r);
		if (Array.isArray(a)) {
			let e = [];
			for (let t of a) e.push(om(i, i.property, t));
			a = e;
		}
		i.property === "className" && Array.isArray(t.className) && (a = t.className.concat(a)), t[i.property] = a;
	}
}
function am(e, t) {
	if (t != null) {
		if (typeof t == "number" || typeof t == "string") e.push({
			type: "text",
			value: String(t)
		});
		else if (Array.isArray(t)) for (let n of t) am(e, n);
		else if (typeof t == "object" && "type" in t) t.type === "root" ? am(e, t.children) : e.push(t);
		else throw Error("Expected node, nodes, or string, got `" + t + "`");
	}
}
function om(e, t, n) {
	if (typeof n == "string") {
		if (e.number && n && !Number.isNaN(Number(n))) return Number(n);
		if ((e.boolean || e.overloadedBoolean) && (n === "" || Sp(n) === Sp(t))) return !0;
	}
	return n;
}
function sm(e) {
	let t = [];
	for (let [n, r] of Object.entries(e)) t.push([n, r].join(": "));
	return t.join("; ");
}
function cm(e) {
	let t = /* @__PURE__ */ new Map();
	for (let n of e) t.set(n.toLowerCase(), n);
	return t;
}
//#endregion
//#region node_modules/.pnpm/hastscript@9.0.1/node_modules/hastscript/lib/svg-case-sensitive-tag-names.js
var lm = /* @__PURE__ */ "altGlyph.altGlyphDef.altGlyphItem.animateColor.animateMotion.animateTransform.clipPath.feBlend.feColorMatrix.feComponentTransfer.feComposite.feConvolveMatrix.feDiffuseLighting.feDisplacementMap.feDistantLight.feDropShadow.feFlood.feFuncA.feFuncB.feFuncG.feFuncR.feGaussianBlur.feImage.feMerge.feMergeNode.feMorphology.feOffset.fePointLight.feSpecularLighting.feSpotLight.feTile.feTurbulence.foreignObject.glyphRef.linearGradient.radialGradient.solidColor.textArea.textPath".split("."), um = nm(Xp, "div");
nm(Zp, "g", lm);
//#endregion
//#region node_modules/.pnpm/character-entities-legacy@3.0.0/node_modules/character-entities-legacy/index.js
var dm = /* @__PURE__ */ "AElig.AMP.Aacute.Acirc.Agrave.Aring.Atilde.Auml.COPY.Ccedil.ETH.Eacute.Ecirc.Egrave.Euml.GT.Iacute.Icirc.Igrave.Iuml.LT.Ntilde.Oacute.Ocirc.Ograve.Oslash.Otilde.Ouml.QUOT.REG.THORN.Uacute.Ucirc.Ugrave.Uuml.Yacute.aacute.acirc.acute.aelig.agrave.amp.aring.atilde.auml.brvbar.ccedil.cedil.cent.copy.curren.deg.divide.eacute.ecirc.egrave.eth.euml.frac12.frac14.frac34.gt.iacute.icirc.iexcl.igrave.iquest.iuml.laquo.lt.macr.micro.middot.nbsp.not.ntilde.oacute.ocirc.ograve.ordf.ordm.oslash.otilde.ouml.para.plusmn.pound.quot.raquo.reg.sect.shy.sup1.sup2.sup3.szlig.thorn.times.uacute.ucirc.ugrave.uml.uuml.yacute.yen.yuml".split("."), fm = {
	0: "�",
	128: "€",
	130: "‚",
	131: "ƒ",
	132: "„",
	133: "…",
	134: "†",
	135: "‡",
	136: "ˆ",
	137: "‰",
	138: "Š",
	139: "‹",
	140: "Œ",
	142: "Ž",
	145: "‘",
	146: "’",
	147: "“",
	148: "”",
	149: "•",
	150: "–",
	151: "—",
	152: "˜",
	153: "™",
	154: "š",
	155: "›",
	156: "œ",
	158: "ž",
	159: "Ÿ"
};
//#endregion
//#region node_modules/.pnpm/is-decimal@2.0.1/node_modules/is-decimal/index.js
function pm(e) {
	let t = typeof e == "string" ? e.charCodeAt(0) : e;
	return t >= 48 && t <= 57;
}
//#endregion
//#region node_modules/.pnpm/is-hexadecimal@2.0.1/node_modules/is-hexadecimal/index.js
function mm(e) {
	let t = typeof e == "string" ? e.charCodeAt(0) : e;
	return t >= 97 && t <= 102 || t >= 65 && t <= 70 || t >= 48 && t <= 57;
}
//#endregion
//#region node_modules/.pnpm/is-alphabetical@2.0.1/node_modules/is-alphabetical/index.js
function hm(e) {
	let t = typeof e == "string" ? e.charCodeAt(0) : e;
	return t >= 97 && t <= 122 || t >= 65 && t <= 90;
}
//#endregion
//#region node_modules/.pnpm/is-alphanumerical@2.0.1/node_modules/is-alphanumerical/index.js
function gm(e) {
	return hm(e) || pm(e);
}
//#endregion
//#region node_modules/.pnpm/decode-named-character-reference@1.3.0/node_modules/decode-named-character-reference/index.dom.js
var _m = document.createElement("i");
function vm(e) {
	let t = "&" + e + ";";
	_m.innerHTML = t;
	let n = _m.textContent;
	return n.charCodeAt(n.length - 1) === 59 && e !== "semi" ? !1 : n !== t && n;
}
//#endregion
//#region node_modules/.pnpm/parse-entities@4.0.2/node_modules/parse-entities/lib/index.js
var ym = [
	"",
	"Named character references must be terminated by a semicolon",
	"Numeric character references must be terminated by a semicolon",
	"Named character references cannot be empty",
	"Numeric character references cannot be empty",
	"Named character references must be known",
	"Numeric character references cannot be disallowed",
	"Numeric character references cannot be outside the permissible Unicode range"
];
function bm(e, t) {
	let n = t || {}, r = typeof n.additional == "string" ? n.additional.charCodeAt(0) : n.additional, i = [], a = 0, o = -1, s = "", c, l;
	n.position && ("start" in n.position || "indent" in n.position ? (l = n.position.indent, c = n.position.start) : c = n.position);
	let u = (c ? c.line : 0) || 1, d = (c ? c.column : 0) || 1, f = m(), p;
	for (a--; ++a <= e.length;) if (p === 10 && (d = (l ? l[o] : 0) || 1), p = e.charCodeAt(a), p === 38) {
		let t = e.charCodeAt(a + 1);
		if (t === 9 || t === 10 || t === 12 || t === 32 || t === 38 || t === 60 || Number.isNaN(t) || r && t === r) {
			s += String.fromCharCode(p), d++;
			continue;
		}
		let o = a + 1, c = o, l = o, u;
		if (t === 35) {
			l = ++c;
			let t = e.charCodeAt(l);
			t === 88 || t === 120 ? (u = "hexadecimal", l = ++c) : u = "decimal";
		} else u = "named";
		let _ = "", v = "", y = "", b = u === "named" ? gm : u === "decimal" ? pm : mm;
		for (l--; ++l <= e.length;) {
			let t = e.charCodeAt(l);
			if (!b(t)) break;
			y += String.fromCharCode(t), u === "named" && dm.includes(y) && (_ = y, v = vm(y));
		}
		let x = e.charCodeAt(l) === 59;
		if (x) {
			l++;
			let e = u === "named" && vm(y);
			e && (_ = y, v = e);
		}
		let S = 1 + l - o, C = "";
		if (!(!x && n.nonTerminated === !1)) {
			if (!y) u !== "named" && h(4, S);
			else if (u === "named") {
				if (x && !v) h(5, 1);
				else if (_ !== y && (l = c + _.length, S = 1 + l - c, x = !1), !x) {
					let t = _ ? 1 : 3;
					if (n.attribute) {
						let n = e.charCodeAt(l);
						n === 61 ? (h(t, S), v = "") : gm(n) ? v = "" : h(t, S);
					} else h(t, S);
				}
				C = v;
			} else {
				x || h(2, S);
				let e = Number.parseInt(y, u === "hexadecimal" ? 16 : 10);
				if (xm(e)) h(7, S), C = "�";
				else if (e in fm) h(6, S), C = fm[e];
				else {
					let t = "";
					Sm(e) && h(6, S), e > 65535 && (e -= 65536, t += String.fromCharCode(e >>> 10 | 55296), e = 56320 | e & 1023), C = t + String.fromCharCode(e);
				}
			}
		}
		if (C) {
			g(), f = m(), a = l - 1, d += l - o + 1, i.push(C);
			let t = m();
			t.offset++, n.reference && n.reference.call(n.referenceContext || void 0, C, {
				start: f,
				end: t
			}, e.slice(o - 1, l)), f = t;
		} else y = e.slice(o - 1, l), s += y, d += y.length, a = l - 1;
	} else p === 10 && (u++, o++, d = 0), Number.isNaN(p) ? g() : (s += String.fromCharCode(p), d++);
	return i.join("");
	function m() {
		return {
			line: u,
			column: d,
			offset: a + ((c ? c.offset : 0) || 0)
		};
	}
	function h(e, t) {
		let r;
		n.warning && (r = m(), r.column += t, r.offset += t, n.warning.call(n.warningContext || void 0, ym[e], r, e));
	}
	function g() {
		s &&= (i.push(s), n.text && n.text.call(n.textContext || void 0, s, {
			start: f,
			end: m()
		}), "");
	}
}
function xm(e) {
	return e >= 55296 && e <= 57343 || e > 1114111;
}
function Sm(e) {
	return e >= 1 && e <= 8 || e === 11 || e >= 13 && e <= 31 || e >= 127 && e <= 159 || e >= 64976 && e <= 65007 || (e & 65535) == 65535 || (e & 65535) == 65534;
}
//#endregion
//#region node_modules/.pnpm/refractor@5.0.0/node_modules/refractor/lib/prism-core.js
var Cm = 0, wm = {}, Tm = {
	util: {
		type: function(e) {
			return Object.prototype.toString.call(e).slice(8, -1);
		},
		objId: function(e) {
			return e.__id || Object.defineProperty(e, "__id", { value: ++Cm }), e.__id;
		},
		clone: function e(t, n) {
			n ||= {};
			var r, i;
			switch (Tm.util.type(t)) {
				case "Object":
					if (i = Tm.util.objId(t), n[i]) return n[i];
					for (var a in r = {}, n[i] = r, t) t.hasOwnProperty(a) && (r[a] = e(t[a], n));
					return r;
				case "Array": return i = Tm.util.objId(t), n[i] ? n[i] : (r = [], n[i] = r, t.forEach(function(t, i) {
					r[i] = e(t, n);
				}), r);
				default: return t;
			}
		}
	},
	languages: {
		plain: wm,
		plaintext: wm,
		text: wm,
		txt: wm,
		extend: function(e, t) {
			var n = Tm.util.clone(Tm.languages[e]);
			for (var r in t) n[r] = t[r];
			return n;
		},
		insertBefore: function(e, t, n, r) {
			r ||= Tm.languages;
			var i = r[e], a = {};
			for (var o in i) if (i.hasOwnProperty(o)) {
				if (o == t) for (var s in n) n.hasOwnProperty(s) && (a[s] = n[s]);
				n.hasOwnProperty(o) || (a[o] = i[o]);
			}
			var c = r[e];
			return r[e] = a, Tm.languages.DFS(Tm.languages, function(t, n) {
				n === c && t != e && (this[t] = a);
			}), a;
		},
		DFS: function e(t, n, r, i) {
			i ||= {};
			var a = Tm.util.objId;
			for (var o in t) if (t.hasOwnProperty(o)) {
				n.call(t, o, t[o], r || o);
				var s = t[o], c = Tm.util.type(s);
				c === "Object" && !i[a(s)] ? (i[a(s)] = !0, e(s, n, null, i)) : c === "Array" && !i[a(s)] && (i[a(s)] = !0, e(s, n, o, i));
			}
		}
	},
	plugins: {},
	highlight: function(e, t, n) {
		var r = {
			code: e,
			grammar: t,
			language: n
		};
		if (Tm.hooks.run("before-tokenize", r), !r.grammar) throw Error("The language \"" + r.language + "\" has no grammar.");
		return r.tokens = Tm.tokenize(r.code, r.grammar), Tm.hooks.run("after-tokenize", r), Em.stringify(Tm.util.encode(r.tokens), r.language);
	},
	tokenize: function(e, t) {
		var n = t.rest;
		if (n) {
			for (var r in n) t[r] = n[r];
			delete t.rest;
		}
		var i = new km();
		return Am(i, i.head, e), Om(e, i, t, i.head, 0), Mm(i);
	},
	hooks: {
		all: {},
		add: function(e, t) {
			var n = Tm.hooks.all;
			n[e] = n[e] || [], n[e].push(t);
		},
		run: function(e, t) {
			var n = Tm.hooks.all[e];
			if (!(!n || !n.length)) for (var r = 0, i; i = n[r++];) i(t);
		}
	},
	Token: Em
};
function Em(e, t, n, r) {
	this.type = e, this.content = t, this.alias = n, this.length = (r || "").length | 0;
}
function Dm(e, t, n, r) {
	e.lastIndex = t;
	var i = e.exec(n);
	if (i && r && i[1]) {
		var a = i[1].length;
		i.index += a, i[0] = i[0].slice(a);
	}
	return i;
}
function Om(e, t, n, r, i, a) {
	for (var o in n) if (!(!n.hasOwnProperty(o) || !n[o])) {
		var s = n[o];
		s = Array.isArray(s) ? s : [s];
		for (var c = 0; c < s.length; ++c) {
			if (a && a.cause == o + "," + c) return;
			var l = s[c], u = l.inside, d = !!l.lookbehind, f = !!l.greedy, p = l.alias;
			if (f && !l.pattern.global) {
				var m = l.pattern.toString().match(/[imsuy]*$/)[0];
				l.pattern = RegExp(l.pattern.source, m + "g");
			}
			for (var h = l.pattern || l, g = r.next, _ = i; g !== t.tail && !(a && _ >= a.reach); _ += g.value.length, g = g.next) {
				var v = g.value;
				if (t.length > e.length) return;
				if (!(v instanceof Em)) {
					var y = 1, b;
					if (f) {
						if (b = Dm(h, _, e, d), !b || b.index >= e.length) break;
						var x = b.index, S = b.index + b[0].length, C = _;
						for (C += g.value.length; x >= C;) g = g.next, C += g.value.length;
						if (C -= g.value.length, _ = C, g.value instanceof Em) continue;
						for (var w = g; w !== t.tail && (C < S || typeof w.value == "string"); w = w.next) y++, C += w.value.length;
						y--, v = e.slice(_, C), b.index -= _;
					} else if (b = Dm(h, 0, v, d), !b) continue;
					var x = b.index, T = b[0], E = v.slice(0, x), D = v.slice(x + T.length), O = _ + v.length;
					a && O > a.reach && (a.reach = O);
					var ee = g.prev;
					E && (ee = Am(t, ee, E), _ += E.length), jm(t, ee, y);
					var k = new Em(o, u ? Tm.tokenize(T, u) : T, p, T);
					if (g = Am(t, ee, k), D && Am(t, g, D), y > 1) {
						var A = {
							cause: o + "," + c,
							reach: O
						};
						Om(e, t, n, g.prev, _, A), a && A.reach > a.reach && (a.reach = A.reach);
					}
				}
			}
		}
	}
}
function km() {
	var e = {
		value: null,
		prev: null,
		next: null
	}, t = {
		value: null,
		prev: e,
		next: null
	};
	e.next = t, this.head = e, this.tail = t, this.length = 0;
}
function Am(e, t, n) {
	var r = t.next, i = {
		value: n,
		prev: t,
		next: r
	};
	return t.next = i, r.prev = i, e.length++, i;
}
function jm(e, t, n) {
	for (var r = t.next, i = 0; i < n && r !== e.tail; i++) r = r.next;
	t.next = r, r.prev = t, e.length -= i;
}
function Mm(e) {
	for (var t = [], n = e.head.next; n !== e.tail;) t.push(n.value), n = n.next;
	return t;
}
var Nm = Tm;
//#endregion
//#region node_modules/.pnpm/refractor@5.0.0/node_modules/refractor/lib/core.js
function Pm() {}
Pm.prototype = Nm;
var Fm = new Pm();
Fm.highlight = Im, Fm.register = Lm, Fm.alias = Rm, Fm.registered = zm, Fm.listLanguages = Bm, Fm.util.encode = Hm, Fm.Token.stringify = Vm;
function Im(e, t) {
	if (typeof e != "string") throw TypeError("Expected `string` for `value`, got `" + e + "`");
	let n, r;
	/* c8 ignore next 2 */
	if (t && typeof t == "object") n = t;
	else {
		if (r = t, typeof r != "string") throw TypeError("Expected `string` for `name`, got `" + r + "`");
		if (Object.hasOwn(Fm.languages, r)) n = Fm.languages[r];
		else throw Error("Unknown language: `" + r + "` is not registered");
	}
	return {
		type: "root",
		children: Nm.highlight.call(Fm, e, n, r)
	};
}
function Lm(e) {
	if (typeof e != "function" || !e.displayName) throw Error("Expected `function` for `syntax`, got `" + e + "`");
	Object.hasOwn(Fm.languages, e.displayName) || e(Fm);
}
function Rm(e, t) {
	let n = Fm.languages, r = {};
	typeof e == "string" ? t && (r[e] = t) : r = e;
	let i;
	for (i in r) if (Object.hasOwn(r, i)) {
		let e = r[i], t = typeof e == "string" ? [e] : e, a = -1;
		for (; ++a < t.length;) n[t[a]] = n[i];
	}
}
function zm(e) {
	if (typeof e != "string") throw TypeError("Expected `string` for `aliasOrLanguage`, got `" + e + "`");
	return Object.hasOwn(Fm.languages, e);
}
function Bm() {
	let e = Fm.languages, t = [], n;
	for (n in e) Object.hasOwn(e, n) && typeof e[n] == "object" && t.push(n);
	return t;
}
function Vm(e, t) {
	if (typeof e == "string") return {
		type: "text",
		value: e
	};
	if (Array.isArray(e)) {
		let n = [], r = -1;
		for (; ++r < e.length;) e[r] !== null && e[r] !== void 0 && e[r] !== "" && n.push(Vm(e[r], t));
		return n;
	}
	let n = {
		attributes: {},
		classes: ["token", e.type],
		content: Vm(e.content, t),
		language: t,
		tag: "span",
		type: e.type
	};
	return e.alias && n.classes.push(...typeof e.alias == "string" ? [e.alias] : e.alias), Fm.hooks.run("wrap", n), um(n.tag + "." + n.classes.join("."), Um(n.attributes), n.content);
}
function Hm(e) {
	return e;
}
function Um(e) {
	let t;
	for (t in e) Object.hasOwn(e, t) && (e[t] = bm(e[t]));
	return e;
}
//#endregion
//#region node_modules/.pnpm/react-diff-viewer-continued@4.4.0_@types+react@19.2.18_react-dom@19.2.8_react@19.2.8__react@19.2.8/node_modules/react-diff-viewer-continued/lib/esm/src/highlight-theme.js
var Wm = {
	default: "#24292e",
	comment: "#6a737d",
	prolog: "#6a737d",
	doctype: "#6a737d",
	cdata: "#6a737d",
	punctuation: "#24292e",
	property: "#005cc5",
	tag: "#22863a",
	boolean: "#005cc5",
	number: "#005cc5",
	constant: "#005cc5",
	symbol: "#005cc5",
	deleted: "#b31d28",
	selector: "#6f42c1",
	"attr-name": "#6f42c1",
	string: "#032f62",
	char: "#032f62",
	builtin: "#005cc5",
	inserted: "#22863a",
	operator: "#d73a49",
	entity: "#22863a",
	url: "#032f62",
	"attr-value": "#032f62",
	keyword: "#d73a49",
	atrule: "#d73a49",
	"class-name": "#6f42c1",
	function: "#6f42c1",
	regex: "#032f62",
	important: "#e36209",
	variable: "#e36209"
}, Gm = {
	default: "#f8f8f2",
	comment: "#6272a4",
	prolog: "#6272a4",
	doctype: "#6272a4",
	cdata: "#6272a4",
	punctuation: "#f8f8f2",
	property: "#8be9fd",
	tag: "#ff79c6",
	boolean: "#bd93f9",
	number: "#bd93f9",
	constant: "#bd93f9",
	symbol: "#bd93f9",
	deleted: "#ff5555",
	selector: "#50fa7b",
	"attr-name": "#50fa7b",
	string: "#f1fa8c",
	char: "#f1fa8c",
	builtin: "#8be9fd",
	inserted: "#50fa7b",
	operator: "#f8f8f2",
	entity: "#ff79c6",
	url: "#f1fa8c",
	"attr-value": "#f1fa8c",
	keyword: "#ff79c6",
	atrule: "#ff79c6",
	"class-name": "#8be9fd",
	function: "#50fa7b",
	regex: "#ffb86c",
	important: "#ffb86c",
	variable: "#f8f8f2"
}, Km = (e, t) => {
	for (let n = t.length - 1; n >= 0; n--) {
		let r = t[n];
		if (r !== "token" && e[r]) return e[r];
	}
	return e.default;
}, qm = /* @__PURE__ */ z({ default: () => Jm });
function Jm(e) {
	e.languages.clike = {
		comment: [{
			pattern: /(^|[^\\])\/\*[\s\S]*?(?:\*\/|$)/,
			lookbehind: !0,
			greedy: !0
		}, {
			pattern: /(^|[^\\:])\/\/.*/,
			lookbehind: !0,
			greedy: !0
		}],
		string: {
			pattern: /(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,
			greedy: !0
		},
		"class-name": {
			pattern: /(\b(?:class|extends|implements|instanceof|interface|new|trait)\s+|\bcatch\s+\()[\w.\\]+/i,
			lookbehind: !0,
			inside: { punctuation: /[.\\]/ }
		},
		keyword: /\b(?:break|catch|continue|do|else|finally|for|function|if|in|instanceof|new|null|return|throw|try|while)\b/,
		boolean: /\b(?:false|true)\b/,
		function: /\b\w+(?=\()/,
		number: /\b0x[\da-f]+\b|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:e[+-]?\d+)?/i,
		operator: /[<>]=?|[!=]=?=?|--?|\+\+?|&&?|\|\|?|[?*/~^%]/,
		punctuation: /[{}[\];(),.:]/
	};
}
var Ym = R((() => {
	Jm.displayName = "clike", Jm.aliases = [];
})), Xm = /* @__PURE__ */ z({ default: () => Zm });
function Zm(e) {
	e.register(Jm), e.languages.c = e.languages.extend("clike", {
		comment: {
			pattern: /\/\/(?:[^\r\n\\]|\\(?:\r\n?|\n|(?![\r\n])))*|\/\*[\s\S]*?(?:\*\/|$)/,
			greedy: !0
		},
		string: {
			pattern: /"(?:\\(?:\r\n|[\s\S])|[^"\\\r\n])*"/,
			greedy: !0
		},
		"class-name": {
			pattern: /(\b(?:enum|struct)\s+(?:__attribute__\s*\(\([\s\S]*?\)\)\s*)?)\w+|\b[a-z]\w*_t\b/,
			lookbehind: !0
		},
		keyword: /\b(?:_Alignas|_Alignof|_Atomic|_Bool|_Complex|_Generic|_Imaginary|_Noreturn|_Static_assert|_Thread_local|__attribute__|asm|auto|break|case|char|const|continue|default|do|double|else|enum|extern|float|for|goto|if|inline|int|long|register|return|short|signed|sizeof|static|struct|switch|typedef|typeof|union|unsigned|void|volatile|while)\b/,
		function: /\b[a-z_]\w*(?=\s*\()/i,
		number: /(?:\b0x(?:[\da-f]+(?:\.[\da-f]*)?|\.[\da-f]+)(?:p[+-]?\d+)?|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:e[+-]?\d+)?)[ful]{0,4}/i,
		operator: />>=?|<<=?|->|([-+&|:])\1|[?:~]|[-+*/%&|^!=<>]=?/
	}), e.languages.insertBefore("c", "string", { char: {
		pattern: /'(?:\\(?:\r\n|[\s\S])|[^'\\\r\n]){0,32}'/,
		greedy: !0
	} }), e.languages.insertBefore("c", "string", { macro: {
		pattern: /(^[\t ]*)#\s*[a-z](?:[^\r\n\\/]|\/(?!\*)|\/\*(?:[^*]|\*(?!\/))*\*\/|\\(?:\r\n|[\s\S]))*/im,
		lookbehind: !0,
		greedy: !0,
		alias: "property",
		inside: {
			string: [{
				pattern: /^(#\s*include\s*)<[^>]+>/,
				lookbehind: !0
			}, e.languages.c.string],
			char: e.languages.c.char,
			comment: e.languages.c.comment,
			"macro-name": [{
				pattern: /(^#\s*define\s+)\w+\b(?!\()/i,
				lookbehind: !0
			}, {
				pattern: /(^#\s*define\s+)\w+\b(?=\()/i,
				lookbehind: !0,
				alias: "function"
			}],
			directive: {
				pattern: /^(#\s*)[a-z]+/,
				lookbehind: !0,
				alias: "keyword"
			},
			"directive-hash": /^#/,
			punctuation: /##|\\(?=[\r\n])/,
			expression: {
				pattern: /\S[\s\S]*/,
				inside: e.languages.c
			}
		}
	} }), e.languages.insertBefore("c", "function", { constant: /\b(?:EOF|NULL|SEEK_CUR|SEEK_END|SEEK_SET|__DATE__|__FILE__|__LINE__|__TIMESTAMP__|__TIME__|__func__|stderr|stdin|stdout)\b/ }), delete e.languages.c.boolean;
}
var Qm = R((() => {
	Ym(), Zm.displayName = "c", Zm.aliases = [];
})), $m = /* @__PURE__ */ z({ default: () => eh });
function eh(e) {
	e.register(Zm), (function(e) {
		var t = /\b(?:alignas|alignof|asm|auto|bool|break|case|catch|char|char16_t|char32_t|char8_t|class|co_await|co_return|co_yield|compl|concept|const|const_cast|consteval|constexpr|constinit|continue|decltype|default|delete|do|double|dynamic_cast|else|enum|explicit|export|extern|final|float|for|friend|goto|if|import|inline|int|int16_t|int32_t|int64_t|int8_t|long|module|mutable|namespace|new|noexcept|nullptr|operator|override|private|protected|public|register|reinterpret_cast|requires|return|short|signed|sizeof|static|static_assert|static_cast|struct|switch|template|this|thread_local|throw|try|typedef|typeid|typename|uint16_t|uint32_t|uint64_t|uint8_t|union|unsigned|using|virtual|void|volatile|wchar_t|while)\b/, n = "\\b(?!<keyword>)\\w+(?:\\s*\\.\\s*\\w+)*\\b".replace(/<keyword>/g, function() {
			return t.source;
		});
		e.languages.cpp = e.languages.extend("c", {
			"class-name": [
				{
					pattern: RegExp("(\\b(?:class|concept|enum|struct|typename)\\s+)(?!<keyword>)\\w+".replace(/<keyword>/g, function() {
						return t.source;
					})),
					lookbehind: !0
				},
				/\b[A-Z]\w*(?=\s*::\s*\w+\s*\()/,
				/\b[A-Z_]\w*(?=\s*::\s*~\w+\s*\()/i,
				/\b\w+(?=\s*<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>\s*::\s*\w+\s*\()/
			],
			keyword: t,
			number: {
				pattern: /(?:\b0b[01']+|\b0x(?:[\da-f']+(?:\.[\da-f']*)?|\.[\da-f']+)(?:p[+-]?[\d']+)?|(?:\b[\d']+(?:\.[\d']*)?|\B\.[\d']+)(?:e[+-]?[\d']+)?)[ful]{0,4}/i,
				greedy: !0
			},
			operator: />>=?|<<=?|->|--|\+\+|&&|\|\||[?:~]|<=>|[-+*/%&|^!=<>]=?|\b(?:and|and_eq|bitand|bitor|not|not_eq|or|or_eq|xor|xor_eq)\b/,
			boolean: /\b(?:false|true)\b/
		}), e.languages.insertBefore("cpp", "string", {
			module: {
				pattern: RegExp("(\\b(?:import|module)\\s+)(?:\"(?:\\\\(?:\\r\\n|[\\s\\S])|[^\"\\\\\\r\\n])*\"|<[^<>\\r\\n]*>|" + "<mod-name>(?:\\s*:\\s*<mod-name>)?|:\\s*<mod-name>".replace(/<mod-name>/g, function() {
					return n;
				}) + ")"),
				lookbehind: !0,
				greedy: !0,
				inside: {
					string: /^[<"][\s\S]+/,
					operator: /:/,
					punctuation: /\./
				}
			},
			"raw-string": {
				pattern: /R"([^()\\ ]{0,16})\([\s\S]*?\)\1"/,
				alias: "string",
				greedy: !0
			}
		}), e.languages.insertBefore("cpp", "keyword", { "generic-function": {
			pattern: /\b(?!operator\b)[a-z_]\w*\s*<(?:[^<>]|<[^<>]*>)*>(?=\s*\()/i,
			inside: {
				function: /^\w+/,
				generic: {
					pattern: /<[\s\S]+/,
					alias: "class-name",
					inside: e.languages.cpp
				}
			}
		} }), e.languages.insertBefore("cpp", "operator", { "double-colon": {
			pattern: /::/,
			alias: "punctuation"
		} }), e.languages.insertBefore("cpp", "class-name", { "base-clause": {
			pattern: /(\b(?:class|struct)\s+\w+\s*:\s*)[^;{}"'\s]+(?:\s+[^;{}"'\s]+)*(?=\s*[;{])/,
			lookbehind: !0,
			greedy: !0,
			inside: e.languages.extend("cpp", {})
		} }), e.languages.insertBefore("inside", "double-colon", { "class-name": /\b[a-z_]\w*\b(?!\s*::)/i }, e.languages.cpp["base-clause"]);
	})(e);
}
var th = R((() => {
	Qm(), eh.displayName = "cpp", eh.aliases = [];
})), nh = /* @__PURE__ */ z({ default: () => rh });
function rh(e) {
	e.register(eh), e.languages.arduino = e.languages.extend("cpp", {
		keyword: /\b(?:String|array|bool|boolean|break|byte|case|catch|continue|default|do|double|else|finally|for|function|goto|if|in|instanceof|int|integer|long|loop|new|null|return|setup|string|switch|throw|try|void|while|word)\b/,
		constant: /\b(?:ANALOG_MESSAGE|DEFAULT|DIGITAL_MESSAGE|EXTERNAL|FIRMATA_STRING|HIGH|INPUT|INPUT_PULLUP|INTERNAL|INTERNAL1V1|INTERNAL2V56|LED_BUILTIN|LOW|OUTPUT|REPORT_ANALOG|REPORT_DIGITAL|SET_PIN_MODE|SYSEX_START|SYSTEM_RESET)\b/,
		builtin: /\b(?:Audio|BSSID|Bridge|Client|Console|EEPROM|Esplora|EsploraTFT|Ethernet|EthernetClient|EthernetServer|EthernetUDP|File|FileIO|FileSystem|Firmata|GPRS|GSM|GSMBand|GSMClient|GSMModem|GSMPIN|GSMScanner|GSMServer|GSMVoiceCall|GSM_SMS|HttpClient|IPAddress|IRread|Keyboard|KeyboardController|LiquidCrystal|LiquidCrystal_I2C|Mailbox|Mouse|MouseController|PImage|Process|RSSI|RobotControl|RobotMotor|SD|SPI|SSID|Scheduler|Serial|Server|Servo|SoftwareSerial|Stepper|Stream|TFT|Task|USBHost|WiFi|WiFiClient|WiFiServer|WiFiUDP|Wire|YunClient|YunServer|abs|addParameter|analogRead|analogReadResolution|analogReference|analogWrite|analogWriteResolution|answerCall|attach|attachGPRS|attachInterrupt|attached|autoscroll|available|background|beep|begin|beginPacket|beginSD|beginSMS|beginSpeaker|beginTFT|beginTransmission|beginWrite|bit|bitClear|bitRead|bitSet|bitWrite|blink|blinkVersion|buffer|changePIN|checkPIN|checkPUK|checkReg|circle|cityNameRead|cityNameWrite|clear|clearScreen|click|close|compassRead|config|connect|connected|constrain|cos|countryNameRead|countryNameWrite|createChar|cursor|debugPrint|delay|delayMicroseconds|detach|detachInterrupt|digitalRead|digitalWrite|disconnect|display|displayLogos|drawBMP|drawCompass|encryptionType|end|endPacket|endSMS|endTransmission|endWrite|exists|exitValue|fill|find|findUntil|flush|gatewayIP|get|getAsynchronously|getBand|getButton|getCurrentCarrier|getIMEI|getKey|getModifiers|getOemKey|getPINUsed|getResult|getSignalStrength|getSocket|getVoiceCallStatus|getXChange|getYChange|hangCall|height|highByte|home|image|interrupts|isActionDone|isDirectory|isListening|isPIN|isPressed|isValid|keyPressed|keyReleased|keyboardRead|knobRead|leftToRight|line|lineFollowConfig|listen|listenOnLocalhost|loadImage|localIP|lowByte|macAddress|maintain|map|max|messageAvailable|micros|millis|min|mkdir|motorsStop|motorsWrite|mouseDragged|mouseMoved|mousePressed|mouseReleased|move|noAutoscroll|noBlink|noBuffer|noCursor|noDisplay|noFill|noInterrupts|noListenOnLocalhost|noStroke|noTone|onReceive|onRequest|open|openNextFile|overflow|parseCommand|parseFloat|parseInt|parsePacket|pauseMode|peek|pinMode|playFile|playMelody|point|pointTo|position|pow|prepare|press|print|printFirmwareVersion|printVersion|println|process|processInput|pulseIn|put|random|randomSeed|read|readAccelerometer|readBlue|readButton|readBytes|readBytesUntil|readGreen|readJoystickButton|readJoystickSwitch|readJoystickX|readJoystickY|readLightSensor|readMessage|readMicrophone|readNetworks|readRed|readSlider|readString|readStringUntil|readTemperature|ready|rect|release|releaseAll|remoteIP|remoteNumber|remotePort|remove|requestFrom|retrieveCallingNumber|rewindDirectory|rightToLeft|rmdir|robotNameRead|robotNameWrite|run|runAsynchronously|runShellCommand|runShellCommandAsynchronously|running|scanNetworks|scrollDisplayLeft|scrollDisplayRight|seek|sendAnalog|sendDigitalPortPair|sendDigitalPorts|sendString|sendSysex|serialEvent|setBand|setBitOrder|setClockDivider|setCursor|setDNS|setDataMode|setFirmwareVersion|setMode|setPINUsed|setSpeed|setTextSize|setTimeout|shiftIn|shiftOut|shutdown|sin|size|sqrt|startLoop|step|stop|stroke|subnetMask|switchPIN|tan|tempoWrite|text|tone|transfer|tuneWrite|turn|updateIR|userNameRead|userNameWrite|voiceCall|waitContinue|width|write|writeBlue|writeGreen|writeJSON|writeMessage|writeMicroseconds|writeRGB|writeRed|yield)\b/
	}), e.languages.ino = e.languages.arduino;
}
var ih = R((() => {
	th(), rh.displayName = "arduino", rh.aliases = ["ino"];
})), ah = /* @__PURE__ */ z({ default: () => oh });
function oh(e) {
	(function(e) {
		var t = "\\b(?:BASH|BASHOPTS|BASH_ALIASES|BASH_ARGC|BASH_ARGV|BASH_CMDS|BASH_COMPLETION_COMPAT_DIR|BASH_LINENO|BASH_REMATCH|BASH_SOURCE|BASH_VERSINFO|BASH_VERSION|COLORTERM|COLUMNS|COMP_WORDBREAKS|DBUS_SESSION_BUS_ADDRESS|DEFAULTS_PATH|DESKTOP_SESSION|DIRSTACK|DISPLAY|EUID|GDMSESSION|GDM_LANG|GNOME_KEYRING_CONTROL|GNOME_KEYRING_PID|GPG_AGENT_INFO|GROUPS|HISTCONTROL|HISTFILE|HISTFILESIZE|HISTSIZE|HOME|HOSTNAME|HOSTTYPE|IFS|INSTANCE|JOB|LANG|LANGUAGE|LC_ADDRESS|LC_ALL|LC_IDENTIFICATION|LC_MEASUREMENT|LC_MONETARY|LC_NAME|LC_NUMERIC|LC_PAPER|LC_TELEPHONE|LC_TIME|LESSCLOSE|LESSOPEN|LINES|LOGNAME|LS_COLORS|MACHTYPE|MAILCHECK|MANDATORY_PATH|NO_AT_BRIDGE|OLDPWD|OPTERR|OPTIND|ORBIT_SOCKETDIR|OSTYPE|PAPERSIZE|PATH|PIPESTATUS|PPID|PS1|PS2|PS3|PS4|PWD|RANDOM|REPLY|SECONDS|SELINUX_INIT|SESSION|SESSIONTYPE|SESSION_MANAGER|SHELL|SHELLOPTS|SHLVL|SSH_AUTH_SOCK|TERM|UID|UPSTART_EVENTS|UPSTART_INSTANCE|UPSTART_JOB|UPSTART_SESSION|USER|WINDOWID|XAUTHORITY|XDG_CONFIG_DIRS|XDG_CURRENT_DESKTOP|XDG_DATA_DIRS|XDG_GREETER_DATA_DIR|XDG_MENU_PREFIX|XDG_RUNTIME_DIR|XDG_SEAT|XDG_SEAT_PATH|XDG_SESSION_DESKTOP|XDG_SESSION_ID|XDG_SESSION_PATH|XDG_SESSION_TYPE|XDG_VTNR|XMODIFIERS)\\b", n = {
			pattern: /(^(["']?)\w+\2)[ \t]+\S.*/,
			lookbehind: !0,
			alias: "punctuation",
			inside: null
		}, r = {
			bash: n,
			environment: {
				pattern: RegExp("\\$" + t),
				alias: "constant"
			},
			variable: [
				{
					pattern: /\$?\(\([\s\S]+?\)\)/,
					greedy: !0,
					inside: {
						variable: [{
							pattern: /(^\$\(\([\s\S]+)\)\)/,
							lookbehind: !0
						}, /^\$\(\(/],
						number: /\b0x[\dA-Fa-f]+\b|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:[Ee]-?\d+)?/,
						operator: /--|\+\+|\*\*=?|<<=?|>>=?|&&|\|\||[=!+\-*/%<>^&|]=?|[?~:]/,
						punctuation: /\(\(?|\)\)?|,|;/
					}
				},
				{
					pattern: /\$\((?:\([^)]+\)|[^()])+\)|`[^`]+`/,
					greedy: !0,
					inside: { variable: /^\$\(|^`|\)$|`$/ }
				},
				{
					pattern: /\$\{[^}]+\}/,
					greedy: !0,
					inside: {
						operator: /:[-=?+]?|[!\/]|##?|%%?|\^\^?|,,?/,
						punctuation: /[\[\]]/,
						environment: {
							pattern: RegExp("(\\{)" + t),
							lookbehind: !0,
							alias: "constant"
						}
					}
				},
				/\$(?:\w+|[#?*!@$])/
			],
			entity: /\\(?:[abceEfnrtv\\"]|O?[0-7]{1,3}|U[0-9a-fA-F]{8}|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{1,2})/
		};
		e.languages.bash = {
			shebang: {
				pattern: /^#!\s*\/.*/,
				alias: "important"
			},
			comment: {
				pattern: /(^|[^"{\\$])#.*/,
				lookbehind: !0
			},
			"function-name": [{
				pattern: /(\bfunction\s+)[\w-]+(?=(?:\s*\(?:\s*\))?\s*\{)/,
				lookbehind: !0,
				alias: "function"
			}, {
				pattern: /\b[\w-]+(?=\s*\(\s*\)\s*\{)/,
				alias: "function"
			}],
			"for-or-select": {
				pattern: /(\b(?:for|select)\s+)\w+(?=\s+in\s)/,
				alias: "variable",
				lookbehind: !0
			},
			"assign-left": {
				pattern: /(^|[\s;|&]|[<>]\()\w+(?:\.\w+)*(?=\+?=)/,
				inside: { environment: {
					pattern: RegExp("(^|[\\s;|&]|[<>]\\()" + t),
					lookbehind: !0,
					alias: "constant"
				} },
				alias: "variable",
				lookbehind: !0
			},
			parameter: {
				pattern: /(^|\s)-{1,2}(?:\w+:[+-]?)?\w+(?:\.\w+)*(?=[=\s]|$)/,
				alias: "variable",
				lookbehind: !0
			},
			string: [
				{
					pattern: /((?:^|[^<])<<-?\s*)(\w+)\s[\s\S]*?(?:\r?\n|\r)\2/,
					lookbehind: !0,
					greedy: !0,
					inside: r
				},
				{
					pattern: /((?:^|[^<])<<-?\s*)(["'])(\w+)\2\s[\s\S]*?(?:\r?\n|\r)\3/,
					lookbehind: !0,
					greedy: !0,
					inside: { bash: n }
				},
				{
					pattern: /(^|[^\\](?:\\\\)*)"(?:\\[\s\S]|\$\([^)]+\)|\$(?!\()|`[^`]+`|[^"\\`$])*"/,
					lookbehind: !0,
					greedy: !0,
					inside: r
				},
				{
					pattern: /(^|[^$\\])'[^']*'/,
					lookbehind: !0,
					greedy: !0
				},
				{
					pattern: /\$'(?:[^'\\]|\\[\s\S])*'/,
					greedy: !0,
					inside: { entity: r.entity }
				}
			],
			environment: {
				pattern: RegExp("\\$?" + t),
				alias: "constant"
			},
			variable: r.variable,
			function: {
				pattern: /(^|[\s;|&]|[<>]\()(?:add|apropos|apt|apt-cache|apt-get|aptitude|aspell|automysqlbackup|awk|basename|bash|bc|bconsole|bg|bzip2|cal|cargo|cat|cfdisk|chgrp|chkconfig|chmod|chown|chroot|cksum|clear|cmp|column|comm|composer|cp|cron|crontab|csplit|curl|cut|date|dc|dd|ddrescue|debootstrap|df|diff|diff3|dig|dir|dircolors|dirname|dirs|dmesg|docker|docker-compose|du|egrep|eject|env|ethtool|expand|expect|expr|fdformat|fdisk|fg|fgrep|file|find|fmt|fold|format|free|fsck|ftp|fuser|gawk|git|gparted|grep|groupadd|groupdel|groupmod|groups|grub-mkconfig|gzip|halt|head|hg|history|host|hostname|htop|iconv|id|ifconfig|ifdown|ifup|import|install|ip|java|jobs|join|kill|killall|less|link|ln|locate|logname|logrotate|look|lpc|lpr|lprint|lprintd|lprintq|lprm|ls|lsof|lynx|make|man|mc|mdadm|mkconfig|mkdir|mke2fs|mkfifo|mkfs|mkisofs|mknod|mkswap|mmv|more|most|mount|mtools|mtr|mutt|mv|nano|nc|netstat|nice|nl|node|nohup|notify-send|npm|nslookup|op|open|parted|passwd|paste|pathchk|ping|pkill|pnpm|podman|podman-compose|popd|pr|printcap|printenv|ps|pushd|pv|quota|quotacheck|quotactl|ram|rar|rcp|reboot|remsync|rename|renice|rev|rm|rmdir|rpm|rsync|scp|screen|sdiff|sed|sendmail|seq|service|sftp|sh|shellcheck|shuf|shutdown|sleep|slocate|sort|split|ssh|stat|strace|su|sudo|sum|suspend|swapon|sync|sysctl|tac|tail|tar|tee|time|timeout|top|touch|tr|traceroute|tsort|tty|umount|uname|unexpand|uniq|units|unrar|unshar|unzip|update-grub|uptime|useradd|userdel|usermod|users|uudecode|uuencode|v|vcpkg|vdir|vi|vim|virsh|vmstat|wait|watch|wc|wget|whereis|which|who|whoami|write|xargs|xdg-open|yarn|yes|zenity|zip|zsh|zypper)(?=$|[)\s;|&])/,
				lookbehind: !0
			},
			keyword: {
				pattern: /(^|[\s;|&]|[<>]\()(?:case|do|done|elif|else|esac|fi|for|function|if|in|select|then|until|while)(?=$|[)\s;|&])/,
				lookbehind: !0
			},
			builtin: {
				pattern: /(^|[\s;|&]|[<>]\()(?:\.|:|alias|bind|break|builtin|caller|cd|command|continue|declare|echo|enable|eval|exec|exit|export|getopts|hash|help|let|local|logout|mapfile|printf|pwd|read|readarray|readonly|return|set|shift|shopt|source|test|times|trap|type|typeset|ulimit|umask|unalias|unset)(?=$|[)\s;|&])/,
				lookbehind: !0,
				alias: "class-name"
			},
			boolean: {
				pattern: /(^|[\s;|&]|[<>]\()(?:false|true)(?=$|[)\s;|&])/,
				lookbehind: !0
			},
			"file-descriptor": {
				pattern: /\B&\d\b/,
				alias: "important"
			},
			operator: {
				pattern: /\d?<>|>\||\+=|=[=~]?|!=?|<<[<-]?|[&\d]?>>|\d[<>]&?|[<>][&=]?|&[>&]?|\|[&|]?/,
				inside: { "file-descriptor": {
					pattern: /^\d/,
					alias: "important"
				} }
			},
			punctuation: /\$?\(\(?|\)\)?|\.\.|[{}[\];\\]/,
			number: {
				pattern: /(^|\s)(?:[1-9]\d*|0)(?:[.,]\d+)?\b/,
				lookbehind: !0
			}
		}, n.inside = e.languages.bash;
		for (var i = [
			"comment",
			"function-name",
			"for-or-select",
			"assign-left",
			"parameter",
			"string",
			"environment",
			"function",
			"keyword",
			"builtin",
			"boolean",
			"file-descriptor",
			"operator",
			"punctuation",
			"number"
		], a = r.variable[1].inside, o = 0; o < i.length; o++) a[i[o]] = e.languages.bash[i[o]];
		e.languages.sh = e.languages.bash, e.languages.shell = e.languages.bash;
	})(e);
}
var sh = R((() => {
	oh.displayName = "bash", oh.aliases = ["sh", "shell"];
})), ch = /* @__PURE__ */ z({ default: () => lh });
function lh(e) {
	e.languages.basic = {
		comment: {
			pattern: /(?:!|REM\b).+/i,
			inside: { keyword: /^REM/i }
		},
		string: {
			pattern: /"(?:""|[!#$%&'()*,\/:;<=>?^\w +\-.])*"/,
			greedy: !0
		},
		number: /(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:E[+-]?\d+)?/i,
		keyword: /\b(?:AS|BEEP|BLOAD|BSAVE|CALL(?: ABSOLUTE)?|CASE|CHAIN|CHDIR|CLEAR|CLOSE|CLS|COM|COMMON|CONST|DATA|DECLARE|DEF(?: FN| SEG|DBL|INT|LNG|SNG|STR)|DIM|DO|DOUBLE|ELSE|ELSEIF|END|ENVIRON|ERASE|ERROR|EXIT|FIELD|FILES|FOR|FUNCTION|GET|GOSUB|GOTO|IF|INPUT|INTEGER|IOCTL|KEY|KILL|LINE INPUT|LOCATE|LOCK|LONG|LOOP|LSET|MKDIR|NAME|NEXT|OFF|ON(?: COM| ERROR| KEY| TIMER)?|OPEN|OPTION BASE|OUT|POKE|PUT|READ|REDIM|REM|RESTORE|RESUME|RETURN|RMDIR|RSET|RUN|SELECT CASE|SHARED|SHELL|SINGLE|SLEEP|STATIC|STEP|STOP|STRING|SUB|SWAP|SYSTEM|THEN|TIMER|TO|TROFF|TRON|TYPE|UNLOCK|UNTIL|USING|VIEW PRINT|WAIT|WEND|WHILE|WRITE)(?:\$|\b)/i,
		function: /\b(?:ABS|ACCESS|ACOS|ANGLE|AREA|ARITHMETIC|ARRAY|ASIN|ASK|AT|ATN|BASE|BEGIN|BREAK|CAUSE|CEIL|CHR|CLIP|COLLATE|COLOR|CON|COS|COSH|COT|CSC|DATE|DATUM|DEBUG|DECIMAL|DEF|DEG|DEGREES|DELETE|DET|DEVICE|DISPLAY|DOT|ELAPSED|EPS|ERASABLE|EXLINE|EXP|EXTERNAL|EXTYPE|FILETYPE|FIXED|FP|GO|GRAPH|HANDLER|IDN|IMAGE|IN|INT|INTERNAL|IP|IS|KEYED|LBOUND|LCASE|LEFT|LEN|LENGTH|LET|LINE|LINES|LOG|LOG10|LOG2|LTRIM|MARGIN|MAT|MAX|MAXNUM|MID|MIN|MISSING|MOD|NATIVE|NUL|NUMERIC|OF|OPTION|ORD|ORGANIZATION|OUTIN|OUTPUT|PI|POINT|POINTER|POINTS|POS|PRINT|PROGRAM|PROMPT|RAD|RADIANS|RANDOMIZE|RECORD|RECSIZE|RECTYPE|RELATIVE|REMAINDER|REPEAT|REST|RETRY|REWRITE|RIGHT|RND|ROUND|RTRIM|SAME|SEC|SELECT|SEQUENTIAL|SET|SETTER|SGN|SIN|SINH|SIZE|SKIP|SQR|STANDARD|STATUS|STR|STREAM|STYLE|TAB|TAN|TANH|TEMPLATE|TEXT|THERE|TIME|TIMEOUT|TRACE|TRANSFORM|TRUNCATE|UBOUND|UCASE|USE|VAL|VARIABLE|VIEWPORT|WHEN|WINDOW|WITH|ZER|ZONEWIDTH)(?:\$|\b)/i,
		operator: /<[=>]?|>=?|[+\-*\/^=&]|\b(?:AND|EQV|IMP|NOT|OR|XOR)\b/i,
		punctuation: /[,;:()]/
	};
}
var uh = R((() => {
	lh.displayName = "basic", lh.aliases = [];
})), dh = /* @__PURE__ */ z({ default: () => fh });
function fh(e) {
	e.register(Jm), (function(e) {
		function t(e, t) {
			return e.replace(/<<(\d+)>>/g, function(e, n) {
				return "(?:" + t[+n] + ")";
			});
		}
		function n(e, n, r) {
			return RegExp(t(e, n), r || "");
		}
		function r(e, t) {
			for (var n = 0; n < t; n++) e = e.replace(/<<self>>/g, function() {
				return "(?:" + e + ")";
			});
			return e.replace(/<<self>>/g, "[^\\s\\S]");
		}
		var i = {
			type: "bool byte char decimal double dynamic float int long object sbyte short string uint ulong ushort var void",
			typeDeclaration: "class enum interface record struct",
			contextual: "add alias and ascending async await by descending from(?=\\s*(?:\\w|$)) get global group into init(?=\\s*;) join let nameof not notnull on or orderby partial remove select set unmanaged value when where with(?=\\s*{)",
			other: "abstract as base break case catch checked const continue default delegate do else event explicit extern finally fixed for foreach goto if implicit in internal is lock namespace new null operator out override params private protected public readonly ref return sealed sizeof stackalloc static switch this throw try typeof unchecked unsafe using virtual volatile while yield"
		};
		function a(e) {
			return "\\b(?:" + e.trim().replace(/ /g, "|") + ")\\b";
		}
		var o = a(i.typeDeclaration), s = RegExp(a(i.type + " " + i.typeDeclaration + " " + i.contextual + " " + i.other)), c = a(i.typeDeclaration + " " + i.contextual + " " + i.other), l = a(i.type + " " + i.typeDeclaration + " " + i.other), u = r("<(?:[^<>;=+\\-*/%&|^]|<<self>>)*>", 2), d = r("\\((?:[^()]|<<self>>)*\\)", 2), f = "@?\\b[A-Za-z_]\\w*\\b", p = t("<<0>>(?:\\s*<<1>>)?", [f, u]), m = t("(?!<<0>>)<<1>>(?:\\s*\\.\\s*<<1>>)*", [c, p]), h = "\\[\\s*(?:,\\s*)*\\]", g = t("<<0>>(?:\\s*(?:\\?\\s*)?<<1>>)*(?:\\s*\\?)?", [m, h]), _ = t("(?:<<0>>|<<1>>)(?:\\s*(?:\\?\\s*)?<<2>>)*(?:\\s*\\?)?", [
			t("\\(<<0>>+(?:,<<0>>+)+\\)", [t("[^,()<>[\\];=+\\-*/%&|^]|<<0>>|<<1>>|<<2>>", [
				u,
				d,
				h
			])]),
			m,
			h
		]), v = {
			keyword: s,
			punctuation: /[<>()?,.:[\]]/
		}, y = "'(?:[^\\r\\n'\\\\]|\\\\.|\\\\[Uux][\\da-fA-F]{1,8})'", b = "\"(?:\\\\.|[^\\\\\"\\r\\n])*\"", x = "@\"(?:\"\"|\\\\[\\s\\S]|[^\\\\\"])*\"(?!\")";
		e.languages.csharp = e.languages.extend("clike", {
			string: [{
				pattern: n("(^|[^$\\\\])<<0>>", [x]),
				lookbehind: !0,
				greedy: !0
			}, {
				pattern: n("(^|[^@$\\\\])<<0>>", [b]),
				lookbehind: !0,
				greedy: !0
			}],
			"class-name": [
				{
					pattern: n("(\\busing\\s+static\\s+)<<0>>(?=\\s*;)", [m]),
					lookbehind: !0,
					inside: v
				},
				{
					pattern: n("(\\busing\\s+<<0>>\\s*=\\s*)<<1>>(?=\\s*;)", [f, _]),
					lookbehind: !0,
					inside: v
				},
				{
					pattern: n("(\\busing\\s+)<<0>>(?=\\s*=)", [f]),
					lookbehind: !0
				},
				{
					pattern: n("(\\b<<0>>\\s+)<<1>>", [o, p]),
					lookbehind: !0,
					inside: v
				},
				{
					pattern: n("(\\bcatch\\s*\\(\\s*)<<0>>", [m]),
					lookbehind: !0,
					inside: v
				},
				{
					pattern: n("(\\bwhere\\s+)<<0>>", [f]),
					lookbehind: !0
				},
				{
					pattern: n("(\\b(?:is(?:\\s+not)?|as)\\s+)<<0>>", [g]),
					lookbehind: !0,
					inside: v
				},
				{
					pattern: n("\\b<<0>>(?=\\s+(?!<<1>>|with\\s*\\{)<<2>>(?:\\s*[=,;:{)\\]]|\\s+(?:in|when)\\b))", [
						_,
						l,
						f
					]),
					inside: v
				}
			],
			keyword: s,
			number: /(?:\b0(?:x[\da-f_]*[\da-f]|b[01_]*[01])|(?:\B\.\d+(?:_+\d+)*|\b\d+(?:_+\d+)*(?:\.\d+(?:_+\d+)*)?)(?:e[-+]?\d+(?:_+\d+)*)?)(?:[dflmu]|lu|ul)?\b/i,
			operator: />>=?|<<=?|[-=]>|([-+&|])\1|~|\?\?=?|[-+*/%&|^!=<>]=?/,
			punctuation: /\?\.?|::|[{}[\];(),.:]/
		}), e.languages.insertBefore("csharp", "number", { range: {
			pattern: /\.\./,
			alias: "operator"
		} }), e.languages.insertBefore("csharp", "punctuation", { "named-parameter": {
			pattern: n("([(,]\\s*)<<0>>(?=\\s*:)", [f]),
			lookbehind: !0,
			alias: "punctuation"
		} }), e.languages.insertBefore("csharp", "class-name", {
			namespace: {
				pattern: n("(\\b(?:namespace|using)\\s+)<<0>>(?:\\s*\\.\\s*<<0>>)*(?=\\s*[;{])", [f]),
				lookbehind: !0,
				inside: { punctuation: /\./ }
			},
			"type-expression": {
				pattern: n("(\\b(?:default|sizeof|typeof)\\s*\\(\\s*(?!\\s))(?:[^()\\s]|\\s(?!\\s)|<<0>>)*(?=\\s*\\))", [d]),
				lookbehind: !0,
				alias: "class-name",
				inside: v
			},
			"return-type": {
				pattern: n("<<0>>(?=\\s+(?:<<1>>\\s*(?:=>|[({]|\\.\\s*this\\s*\\[)|this\\s*\\[))", [_, m]),
				inside: v,
				alias: "class-name"
			},
			"constructor-invocation": {
				pattern: n("(\\bnew\\s+)<<0>>(?=\\s*[[({])", [_]),
				lookbehind: !0,
				inside: v,
				alias: "class-name"
			},
			"generic-method": {
				pattern: n("<<0>>\\s*<<1>>(?=\\s*\\()", [f, u]),
				inside: {
					function: n("^<<0>>", [f]),
					generic: {
						pattern: RegExp(u),
						alias: "class-name",
						inside: v
					}
				}
			},
			"type-list": {
				pattern: n("\\b((?:<<0>>\\s+<<1>>|record\\s+<<1>>\\s*<<5>>|where\\s+<<2>>)\\s*:\\s*)(?:<<3>>|<<4>>|<<1>>\\s*<<5>>|<<6>>)(?:\\s*,\\s*(?:<<3>>|<<4>>|<<6>>))*(?=\\s*(?:where|[{;]|=>|$))", [
					o,
					p,
					f,
					_,
					s.source,
					d,
					"\\bnew\\s*\\(\\s*\\)"
				]),
				lookbehind: !0,
				inside: {
					"record-arguments": {
						pattern: n("(^(?!new\\s*\\()<<0>>\\s*)<<1>>", [p, d]),
						lookbehind: !0,
						greedy: !0,
						inside: e.languages.csharp
					},
					keyword: s,
					"class-name": {
						pattern: RegExp(_),
						greedy: !0,
						inside: v
					},
					punctuation: /[,()]/
				}
			},
			preprocessor: {
				pattern: /(^[\t ]*)#.*/m,
				lookbehind: !0,
				alias: "property",
				inside: { directive: {
					pattern: /(#)\b(?:define|elif|else|endif|endregion|error|if|line|nullable|pragma|region|undef|warning)\b/,
					lookbehind: !0,
					alias: "keyword"
				} }
			}
		});
		var S = b + "|" + y, C = t("\\/(?![*/])|\\/\\/[^\\r\\n]*[\\r\\n]|\\/\\*(?:[^*]|\\*(?!\\/))*\\*\\/|<<0>>", [S]), w = r(t("[^\"'/()]|<<0>>|\\(<<self>>*\\)", [C]), 2), T = "\\b(?:assembly|event|field|method|module|param|property|return|type)\\b", E = t("<<0>>(?:\\s*\\(<<1>>*\\))?", [m, w]);
		e.languages.insertBefore("csharp", "class-name", { attribute: {
			pattern: n("((?:^|[^\\s\\w>)?])\\s*\\[\\s*)(?:<<0>>\\s*:\\s*)?<<1>>(?:\\s*,\\s*<<1>>)*(?=\\s*\\])", [T, E]),
			lookbehind: !0,
			greedy: !0,
			inside: {
				target: {
					pattern: n("^<<0>>(?=\\s*:)", [T]),
					alias: "keyword"
				},
				"attribute-arguments": {
					pattern: n("\\(<<0>>*\\)", [w]),
					inside: e.languages.csharp
				},
				"class-name": {
					pattern: RegExp(m),
					inside: { punctuation: /\./ }
				},
				punctuation: /[:,]/
			}
		} });
		var D = ":[^}\\r\\n]+", O = r(t("[^\"'/()]|<<0>>|\\(<<self>>*\\)", [C]), 2), ee = t("\\{(?!\\{)(?:(?![}:])<<0>>)*<<1>>?\\}", [O, D]), k = r(t("[^\"'/()]|\\/(?!\\*)|\\/\\*(?:[^*]|\\*(?!\\/))*\\*\\/|<<0>>|\\(<<self>>*\\)", [S]), 2), A = t("\\{(?!\\{)(?:(?![}:])<<0>>)*<<1>>?\\}", [k, D]);
		function te(t, r) {
			return {
				interpolation: {
					pattern: n("((?:^|[^{])(?:\\{\\{)*)<<0>>", [t]),
					lookbehind: !0,
					inside: {
						"format-string": {
							pattern: n("(^\\{(?:(?![}:])<<0>>)*)<<1>>(?=\\}$)", [r, D]),
							lookbehind: !0,
							inside: { punctuation: /^:/ }
						},
						punctuation: /^\{|\}$/,
						expression: {
							pattern: /[\s\S]+/,
							alias: "language-csharp",
							inside: e.languages.csharp
						}
					}
				},
				string: /[\s\S]+/
			};
		}
		e.languages.insertBefore("csharp", "string", {
			"interpolation-string": [{
				pattern: n("(^|[^\\\\])(?:\\$@|@\\$)\"(?:\"\"|\\\\[\\s\\S]|\\{\\{|<<0>>|[^\\\\{\"])*\"", [ee]),
				lookbehind: !0,
				greedy: !0,
				inside: te(ee, O)
			}, {
				pattern: n("(^|[^@\\\\])\\$\"(?:\\\\.|\\{\\{|<<0>>|[^\\\\\"{])*\"", [A]),
				lookbehind: !0,
				greedy: !0,
				inside: te(A, k)
			}],
			char: {
				pattern: RegExp(y),
				greedy: !0
			}
		}), e.languages.dotnet = e.languages.cs = e.languages.csharp;
	})(e);
}
var ph = R((() => {
	Ym(), fh.displayName = "csharp", fh.aliases = ["cs", "dotnet"];
})), mh = /* @__PURE__ */ z({ default: () => hh });
function hh(e) {
	(function(e) {
		var t = /(?:"(?:\\(?:\r\n|[\s\S])|[^"\\\r\n])*"|'(?:\\(?:\r\n|[\s\S])|[^'\\\r\n])*')/;
		e.languages.css = {
			comment: /\/\*[\s\S]*?\*\//,
			atrule: {
				pattern: RegExp("@[\\w-](?:[^;{\\s\"']|\\s+(?!\\s)|" + t.source + ")*?(?:;|(?=\\s*\\{))"),
				inside: {
					rule: /^@[\w-]+/,
					"selector-function-argument": {
						pattern: /(\bselector\s*\(\s*(?![\s)]))(?:[^()\s]|\s+(?![\s)])|\((?:[^()]|\([^()]*\))*\))+(?=\s*\))/,
						lookbehind: !0,
						alias: "selector"
					},
					keyword: {
						pattern: /(^|[^\w-])(?:and|not|only|or)(?![\w-])/,
						lookbehind: !0
					}
				}
			},
			url: {
				pattern: RegExp("\\burl\\((?:" + t.source + "|(?:[^\\\\\\r\\n()\"']|\\\\[\\s\\S])*)\\)", "i"),
				greedy: !0,
				inside: {
					function: /^url/i,
					punctuation: /^\(|\)$/,
					string: {
						pattern: RegExp("^" + t.source + "$"),
						alias: "url"
					}
				}
			},
			selector: {
				pattern: RegExp("(^|[{}\\s])[^{}\\s](?:[^{};\"'\\s]|\\s+(?![\\s{])|" + t.source + ")*(?=\\s*\\{)"),
				lookbehind: !0
			},
			string: {
				pattern: t,
				greedy: !0
			},
			property: {
				pattern: /(^|[^-\w\xA0-\uFFFF])(?!\s)[-_a-z\xA0-\uFFFF](?:(?!\s)[-\w\xA0-\uFFFF])*(?=\s*:)/i,
				lookbehind: !0
			},
			important: /!important\b/i,
			function: {
				pattern: /(^|[^-a-z0-9])[-a-z0-9]+(?=\()/i,
				lookbehind: !0
			},
			punctuation: /[(){};:,]/
		}, e.languages.css.atrule.inside.rest = e.languages.css;
		var n = e.languages.markup;
		n && (n.tag.addInlined("style", "css"), n.tag.addAttribute("style", "css"));
	})(e);
}
var gh = R((() => {
	hh.displayName = "css", hh.aliases = [];
})), _h = /* @__PURE__ */ z({ default: () => vh });
function vh(e) {
	(function(e) {
		e.languages.diff = { coord: [
			/^(?:\*{3}|-{3}|\+{3}).*$/m,
			/^@@.*@@$/m,
			/^\d.*$/m
		] };
		var t = {
			"deleted-sign": "-",
			"deleted-arrow": "<",
			"inserted-sign": "+",
			"inserted-arrow": ">",
			unchanged: " ",
			diff: "!"
		};
		Object.keys(t).forEach(function(n) {
			var r = t[n], i = [];
			/^\w+$/.test(n) || i.push(/\w+/.exec(n)[0]), n === "diff" && i.push("bold"), e.languages.diff[n] = {
				pattern: RegExp("^(?:[" + r + "].*(?:\r\n?|\n|(?![\\s\\S])))+", "m"),
				alias: i,
				inside: {
					line: {
						pattern: /(.)(?=[\s\S]).*(?:\r\n?|\n)?/,
						lookbehind: !0
					},
					prefix: {
						pattern: /[\s\S]/,
						alias: /\w+/.exec(n)[0]
					}
				}
			};
		}), Object.defineProperty(e.languages.diff, "PREFIXES", { value: t });
	})(e);
}
var yh = R((() => {
	vh.displayName = "diff", vh.aliases = [];
})), bh = /* @__PURE__ */ z({ default: () => xh });
function xh(e) {
	e.register(Jm), e.languages.go = e.languages.extend("clike", {
		string: {
			pattern: /(^|[^\\])"(?:\\.|[^"\\\r\n])*"|`[^`]*`/,
			lookbehind: !0,
			greedy: !0
		},
		keyword: /\b(?:break|case|chan|const|continue|default|defer|else|fallthrough|for|func|go(?:to)?|if|import|interface|map|package|range|return|select|struct|switch|type|var)\b/,
		boolean: /\b(?:_|false|iota|nil|true)\b/,
		number: [
			/\b0(?:b[01_]+|o[0-7_]+)i?\b/i,
			/\b0x(?:[a-f\d_]+(?:\.[a-f\d_]*)?|\.[a-f\d_]+)(?:p[+-]?\d+(?:_\d+)*)?i?(?!\w)/i,
			/(?:\b\d[\d_]*(?:\.[\d_]*)?|\B\.\d[\d_]*)(?:e[+-]?[\d_]+)?i?(?!\w)/i
		],
		operator: /[*\/%^!=]=?|\+[=+]?|-[=-]?|\|[=|]?|&(?:=|&|\^=?)?|>(?:>=?|=)?|<(?:<=?|=|-)?|:=|\.\.\./,
		builtin: /\b(?:append|bool|byte|cap|close|complex|complex(?:64|128)|copy|delete|error|float(?:32|64)|u?int(?:8|16|32|64)?|imag|len|make|new|panic|print(?:ln)?|real|recover|rune|string|uintptr)\b/
	}), e.languages.insertBefore("go", "string", { char: {
		pattern: /'(?:\\.|[^'\\\r\n]){0,10}'/,
		greedy: !0
	} }), delete e.languages.go["class-name"];
}
var Sh = R((() => {
	Ym(), xh.displayName = "go", xh.aliases = [];
})), Ch = /* @__PURE__ */ z({ default: () => wh });
function wh(e) {
	e.languages.ini = {
		comment: {
			pattern: /(^[ \f\t\v]*)[#;][^\n\r]*/m,
			lookbehind: !0
		},
		section: {
			pattern: /(^[ \f\t\v]*)\[[^\n\r\]]*\]?/m,
			lookbehind: !0,
			inside: {
				"section-name": {
					pattern: /(^\[[ \f\t\v]*)[^ \f\t\v\]]+(?:[ \f\t\v]+[^ \f\t\v\]]+)*/,
					lookbehind: !0,
					alias: "selector"
				},
				punctuation: /\[|\]/
			}
		},
		key: {
			pattern: /(^[ \f\t\v]*)[^ \f\n\r\t\v=]+(?:[ \f\t\v]+[^ \f\n\r\t\v=]+)*(?=[ \f\t\v]*=)/m,
			lookbehind: !0,
			alias: "attr-name"
		},
		value: {
			pattern: /(=[ \f\t\v]*)[^ \f\n\r\t\v]+(?:[ \f\t\v]+[^ \f\n\r\t\v]+)*/,
			lookbehind: !0,
			alias: "attr-value",
			inside: { "inner-value": {
				pattern: /^("|').+(?=\1$)/,
				lookbehind: !0
			} }
		},
		punctuation: /=/
	};
}
var Th = R((() => {
	wh.displayName = "ini", wh.aliases = [];
})), Eh = /* @__PURE__ */ z({ default: () => Dh });
function Dh(e) {
	e.register(Jm), (function(e) {
		var t = /\b(?:abstract|assert|boolean|break|byte|case|catch|char|class|const|continue|default|do|double|else|enum|exports|extends|final|finally|float|for|goto|if|implements|import|instanceof|int|interface|long|module|native|new|non-sealed|null|open|opens|package|permits|private|protected|provides|public|record(?!\s*[(){}[\]<>=%~.:,;?+\-*/&|^])|requires|return|sealed|short|static|strictfp|super|switch|synchronized|this|throw|throws|to|transient|transitive|try|uses|var|void|volatile|while|with|yield)\b/, n = "(?:[a-z]\\w*\\s*\\.\\s*)*(?:[A-Z]\\w*\\s*\\.\\s*)*", r = {
			pattern: RegExp("(^|[^\\w.])" + n + "[A-Z](?:[\\d_A-Z]*[a-z]\\w*)?\\b"),
			lookbehind: !0,
			inside: {
				namespace: {
					pattern: /^[a-z]\w*(?:\s*\.\s*[a-z]\w*)*(?:\s*\.)?/,
					inside: { punctuation: /\./ }
				},
				punctuation: /\./
			}
		};
		e.languages.java = e.languages.extend("clike", {
			string: {
				pattern: /(^|[^\\])"(?:\\.|[^"\\\r\n])*"/,
				lookbehind: !0,
				greedy: !0
			},
			"class-name": [
				r,
				{
					pattern: RegExp("(^|[^\\w.])" + n + "[A-Z]\\w*(?=\\s+\\w+\\s*[;,=()]|\\s*(?:\\[[\\s,]*\\]\\s*)?::\\s*new\\b)"),
					lookbehind: !0,
					inside: r.inside
				},
				{
					pattern: RegExp("(\\b(?:class|enum|extends|implements|instanceof|interface|new|record|throws)\\s+)" + n + "[A-Z]\\w*\\b"),
					lookbehind: !0,
					inside: r.inside
				}
			],
			keyword: t,
			function: [e.languages.clike.function, {
				pattern: /(::\s*)[a-z_]\w*/,
				lookbehind: !0
			}],
			number: /\b0b[01][01_]*L?\b|\b0x(?:\.[\da-f_p+-]+|[\da-f_]+(?:\.[\da-f_p+-]+)?)\b|(?:\b\d[\d_]*(?:\.[\d_]*)?|\B\.\d[\d_]*)(?:e[+-]?\d[\d_]*)?[dfl]?/i,
			operator: {
				pattern: /(^|[^.])(?:<<=?|>>>?=?|->|--|\+\+|&&|\|\||::|[?:~]|[-+*/%&|^!=<>]=?)/m,
				lookbehind: !0
			},
			constant: /\b[A-Z][A-Z_\d]+\b/
		}), e.languages.insertBefore("java", "string", {
			"triple-quoted-string": {
				pattern: /"""[ \t]*[\r\n](?:(?:"|"")?(?:\\.|[^"\\]))*"""/,
				greedy: !0,
				alias: "string"
			},
			char: {
				pattern: /'(?:\\.|[^'\\\r\n]){1,6}'/,
				greedy: !0
			}
		}), e.languages.insertBefore("java", "class-name", {
			annotation: {
				pattern: /(^|[^.])@\w+(?:\s*\.\s*\w+)*/,
				lookbehind: !0,
				alias: "punctuation"
			},
			generics: {
				pattern: /<(?:[\w\s,.?]|&(?!&)|<(?:[\w\s,.?]|&(?!&)|<(?:[\w\s,.?]|&(?!&)|<(?:[\w\s,.?]|&(?!&))*>)*>)*>)*>/,
				inside: {
					"class-name": r,
					keyword: t,
					punctuation: /[<>(),.:]/,
					operator: /[?&|]/
				}
			},
			import: [{
				pattern: RegExp("(\\bimport\\s+)" + n + "(?:[A-Z]\\w*|\\*)(?=\\s*;)"),
				lookbehind: !0,
				inside: {
					namespace: r.inside.namespace,
					punctuation: /\./,
					operator: /\*/,
					"class-name": /\w+/
				}
			}, {
				pattern: RegExp("(\\bimport\\s+static\\s+)" + n + "(?:\\w+|\\*)(?=\\s*;)"),
				lookbehind: !0,
				alias: "static",
				inside: {
					namespace: r.inside.namespace,
					static: /\b\w+$/,
					punctuation: /\./,
					operator: /\*/,
					"class-name": /\w+/
				}
			}],
			namespace: {
				pattern: RegExp("(\\b(?:exports|import(?:\\s+static)?|module|open|opens|package|provides|requires|to|transitive|uses|with)\\s+)(?!<keyword>)[a-z]\\w*(?:\\.[a-z]\\w*)*\\.?".replace(/<keyword>/g, function() {
					return t.source;
				})),
				lookbehind: !0,
				inside: { punctuation: /\./ }
			}
		});
	})(e);
}
var Oh = R((() => {
	Ym(), Dh.displayName = "java", Dh.aliases = [];
})), kh = /* @__PURE__ */ z({ default: () => Ah });
function Ah(e) {
	e.register(Jm), e.languages.javascript = e.languages.extend("clike", {
		"class-name": [e.languages.clike["class-name"], {
			pattern: /(^|[^$\w\xA0-\uFFFF])(?!\s)[_$A-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\.(?:constructor|prototype))/,
			lookbehind: !0
		}],
		keyword: [{
			pattern: /((?:^|\})\s*)catch\b/,
			lookbehind: !0
		}, {
			pattern: /(^|[^.]|\.\.\.\s*)\b(?:as|assert(?=\s*\{)|async(?=\s*(?:function\b|\(|[$\w\xA0-\uFFFF]|$))|await|break|case|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally(?=\s*(?:\{|$))|for|from(?=\s*(?:['"]|$))|function|(?:get|set)(?=\s*(?:[#\[$\w\xA0-\uFFFF]|$))|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)\b/,
			lookbehind: !0
		}],
		function: /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*(?:\.\s*(?:apply|bind|call)\s*)?\()/,
		number: {
			pattern: RegExp("(^|[^\\w$])(?:NaN|Infinity|0[bB][01]+(?:_[01]+)*n?|0[oO][0-7]+(?:_[0-7]+)*n?|0[xX][\\dA-Fa-f]+(?:_[\\dA-Fa-f]+)*n?|\\d+(?:_\\d+)*n|(?:\\d+(?:_\\d+)*(?:\\.(?:\\d+(?:_\\d+)*)?)?|\\.\\d+(?:_\\d+)*)(?:[Ee][+-]?\\d+(?:_\\d+)*)?)(?![\\w$])"),
			lookbehind: !0
		},
		operator: /--|\+\+|\*\*=?|=>|&&=?|\|\|=?|[!=]==|<<=?|>>>?=?|[-+*/%&|^!=<>]=?|\.{3}|\?\?=?|\?\.?|[~:]/
	}), e.languages.javascript["class-name"][0].pattern = /(\b(?:class|extends|implements|instanceof|interface|new)\s+)[\w.\\]+/, e.languages.insertBefore("javascript", "keyword", {
		regex: {
			pattern: RegExp("((?:^|[^$\\w\\xA0-\\uFFFF.\"'\\])\\s]|\\b(?:return|yield))\\s*)\\/(?:(?:\\[(?:[^\\]\\\\\\r\\n]|\\\\.)*\\]|\\\\.|[^/\\\\\\[\\r\\n])+\\/[dgimyus]{0,7}|(?:\\[(?:[^[\\]\\\\\\r\\n]|\\\\.|\\[(?:[^[\\]\\\\\\r\\n]|\\\\.|\\[(?:[^[\\]\\\\\\r\\n]|\\\\.)*\\])*\\])*\\]|\\\\.|[^/\\\\\\[\\r\\n])+\\/[dgimyus]{0,7}v[dgimyus]{0,7})(?=(?:\\s|\\/\\*(?:[^*]|\\*(?!\\/))*\\*\\/)*(?:$|[\\r\\n,.;:})\\]]|\\/\\/))"),
			lookbehind: !0,
			greedy: !0,
			inside: {
				"regex-source": {
					pattern: /^(\/)[\s\S]+(?=\/[a-z]*$)/,
					lookbehind: !0,
					alias: "language-regex",
					inside: e.languages.regex
				},
				"regex-delimiter": /^\/|\/$/,
				"regex-flags": /^[a-z]+$/
			}
		},
		"function-variable": {
			pattern: /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*[=:]\s*(?:async\s*)?(?:\bfunction\b|(?:\((?:[^()]|\([^()]*\))*\)|(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*)\s*=>))/,
			alias: "function"
		},
		parameter: [
			{
				pattern: /(function(?:\s+(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*)?\s*\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\))/,
				lookbehind: !0,
				inside: e.languages.javascript
			},
			{
				pattern: /(^|[^$\w\xA0-\uFFFF])(?!\s)[_$a-z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*=>)/i,
				lookbehind: !0,
				inside: e.languages.javascript
			},
			{
				pattern: /(\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\)\s*=>)/,
				lookbehind: !0,
				inside: e.languages.javascript
			},
			{
				pattern: /((?:\b|\s|^)(?!(?:as|async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally|for|from|function|get|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|set|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)(?![$\w\xA0-\uFFFF]))(?:(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*\s*)\(\s*|\]\s*\(\s*)(?!\s)(?:[^()\s]|\s+(?![\s)])|\([^()]*\))+(?=\s*\)\s*\{)/,
				lookbehind: !0,
				inside: e.languages.javascript
			}
		],
		constant: /\b[A-Z](?:[A-Z_]|\dx?)*\b/
	}), e.languages.insertBefore("javascript", "string", {
		hashbang: {
			pattern: /^#!.*/,
			greedy: !0,
			alias: "comment"
		},
		"template-string": {
			pattern: /`(?:\\[\s\S]|\$\{(?:[^{}]|\{(?:[^{}]|\{[^}]*\})*\})+\}|(?!\$\{)[^\\`])*`/,
			greedy: !0,
			inside: {
				"template-punctuation": {
					pattern: /^`|`$/,
					alias: "string"
				},
				interpolation: {
					pattern: /((?:^|[^\\])(?:\\{2})*)\$\{(?:[^{}]|\{(?:[^{}]|\{[^}]*\})*\})+\}/,
					lookbehind: !0,
					inside: {
						"interpolation-punctuation": {
							pattern: /^\$\{|\}$/,
							alias: "punctuation"
						},
						rest: e.languages.javascript
					}
				},
				string: /[\s\S]+/
			}
		},
		"string-property": {
			pattern: /((?:^|[,{])[ \t]*)(["'])(?:\\(?:\r\n|[\s\S])|(?!\2)[^\\\r\n])*\2(?=\s*:)/m,
			lookbehind: !0,
			greedy: !0,
			alias: "property"
		}
	}), e.languages.insertBefore("javascript", "operator", { "literal-property": {
		pattern: /((?:^|[,{])[ \t]*)(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?=\s*:)/m,
		lookbehind: !0,
		alias: "property"
	} }), e.languages.markup && (e.languages.markup.tag.addInlined("script", "javascript"), e.languages.markup.tag.addAttribute("on(?:abort|blur|change|click|composition(?:end|start|update)|dblclick|error|focus(?:in|out)?|key(?:down|up)|load|mouse(?:down|enter|leave|move|out|over|up)|reset|resize|scroll|select|slotchange|submit|unload|wheel)", "javascript")), e.languages.js = e.languages.javascript;
}
var jh = R((() => {
	Ym(), Ah.displayName = "javascript", Ah.aliases = ["js"];
})), Mh = /* @__PURE__ */ z({ default: () => Nh });
function Nh(e) {
	e.languages.json = {
		property: {
			pattern: /(^|[^\\])"(?:\\.|[^\\"\r\n])*"(?=\s*:)/,
			lookbehind: !0,
			greedy: !0
		},
		string: {
			pattern: /(^|[^\\])"(?:\\.|[^\\"\r\n])*"(?!\s*:)/,
			lookbehind: !0,
			greedy: !0
		},
		comment: {
			pattern: /\/\/.*|\/\*[\s\S]*?(?:\*\/|$)/,
			greedy: !0
		},
		number: /-?\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b/i,
		punctuation: /[{}[\],]/,
		operator: /:/,
		boolean: /\b(?:false|true)\b/,
		null: {
			pattern: /\bnull\b/,
			alias: "keyword"
		}
	}, e.languages.webmanifest = e.languages.json;
}
var Ph = R((() => {
	Nh.displayName = "json", Nh.aliases = ["webmanifest"];
})), Fh = /* @__PURE__ */ z({ default: () => Ih });
function Ih(e) {
	e.register(Jm), (function(e) {
		e.languages.kotlin = e.languages.extend("clike", {
			keyword: {
				pattern: /(^|[^.])\b(?:abstract|actual|annotation|as|break|by|catch|class|companion|const|constructor|continue|crossinline|data|do|dynamic|else|enum|expect|external|final|finally|for|fun|get|if|import|in|infix|init|inline|inner|interface|internal|is|lateinit|noinline|null|object|open|operator|out|override|package|private|protected|public|reified|return|sealed|set|super|suspend|tailrec|this|throw|to|try|typealias|val|var|vararg|when|where|while)\b/,
				lookbehind: !0
			},
			function: [{
				pattern: /(?:`[^\r\n`]+`|\b\w+)(?=\s*\()/,
				greedy: !0
			}, {
				pattern: /(\.)(?:`[^\r\n`]+`|\w+)(?=\s*\{)/,
				lookbehind: !0,
				greedy: !0
			}],
			number: /\b(?:0[xX][\da-fA-F]+(?:_[\da-fA-F]+)*|0[bB][01]+(?:_[01]+)*|\d+(?:_\d+)*(?:\.\d+(?:_\d+)*)?(?:[eE][+-]?\d+(?:_\d+)*)?[fFL]?)\b/,
			operator: /\+[+=]?|-[-=>]?|==?=?|!(?:!|==?)?|[\/*%<>]=?|[?:]:?|\.\.|&&|\|\||\b(?:and|inv|or|shl|shr|ushr|xor)\b/
		}), delete e.languages.kotlin["class-name"];
		var t = {
			"interpolation-punctuation": {
				pattern: /^\$\{?|\}$/,
				alias: "punctuation"
			},
			expression: {
				pattern: /[\s\S]+/,
				inside: e.languages.kotlin
			}
		};
		e.languages.insertBefore("kotlin", "string", {
			"string-literal": [{
				pattern: /"""(?:[^$]|\$(?:(?!\{)|\{[^{}]*\}))*?"""/,
				alias: "multiline",
				inside: {
					interpolation: {
						pattern: /\$(?:[a-z_]\w*|\{[^{}]*\})/i,
						inside: t
					},
					string: /[\s\S]+/
				}
			}, {
				pattern: /"(?:[^"\\\r\n$]|\\.|\$(?:(?!\{)|\{[^{}]*\}))*"/,
				alias: "singleline",
				inside: {
					interpolation: {
						pattern: /((?:^|[^\\])(?:\\{2})*)\$(?:[a-z_]\w*|\{[^{}]*\})/i,
						lookbehind: !0,
						inside: t
					},
					string: /[\s\S]+/
				}
			}],
			char: {
				pattern: /'(?:[^'\\\r\n]|\\(?:.|u[a-fA-F0-9]{0,4}))'/,
				greedy: !0
			}
		}), delete e.languages.kotlin.string, e.languages.insertBefore("kotlin", "keyword", { annotation: {
			pattern: /\B@(?:\w+:)?(?:[A-Z]\w*|\[[^\]]+\])/,
			alias: "builtin"
		} }), e.languages.insertBefore("kotlin", "function", { label: {
			pattern: /\b\w+@|@\w+\b/,
			alias: "symbol"
		} }), e.languages.kt = e.languages.kotlin, e.languages.kts = e.languages.kotlin;
	})(e);
}
var Lh = R((() => {
	Ym(), Ih.displayName = "kotlin", Ih.aliases = ["kt", "kts"];
})), Rh = /* @__PURE__ */ z({ default: () => zh });
function zh(e) {
	e.register(hh), e.languages.less = e.languages.extend("css", {
		comment: [/\/\*[\s\S]*?\*\//, {
			pattern: /(^|[^\\])\/\/.*/,
			lookbehind: !0
		}],
		atrule: {
			pattern: /@[\w-](?:\((?:[^(){}]|\([^(){}]*\))*\)|[^(){};\s]|\s+(?!\s))*?(?=\s*\{)/,
			inside: { punctuation: /[:()]/ }
		},
		selector: {
			pattern: /(?:@\{[\w-]+\}|[^{};\s@])(?:@\{[\w-]+\}|\((?:[^(){}]|\([^(){}]*\))*\)|[^(){};@\s]|\s+(?!\s))*?(?=\s*\{)/,
			inside: { variable: /@+[\w-]+/ }
		},
		property: /(?:@\{[\w-]+\}|[\w-])+(?:\+_?)?(?=\s*:)/,
		operator: /[+\-*\/]/
	}), e.languages.insertBefore("less", "property", {
		variable: [{
			pattern: /@[\w-]+\s*:/,
			inside: { punctuation: /:/ }
		}, /@@?[\w-]+/],
		"mixin-usage": {
			pattern: /([{;]\s*)[.#](?!\d)[\w-].*?(?=[(;])/,
			lookbehind: !0,
			alias: "function"
		}
	});
}
var Bh = R((() => {
	gh(), zh.displayName = "less", zh.aliases = [];
})), Vh = /* @__PURE__ */ z({ default: () => Hh });
function Hh(e) {
	e.languages.lua = {
		comment: /^#!.+|--(?:\[(=*)\[[\s\S]*?\]\1\]|.*)/m,
		string: {
			pattern: /(["'])(?:(?!\1)[^\\\r\n]|\\z(?:\r\n|\s)|\\(?:\r\n|[^z]))*\1|\[(=*)\[[\s\S]*?\]\2\]/,
			greedy: !0
		},
		number: /\b0x[a-f\d]+(?:\.[a-f\d]*)?(?:p[+-]?\d+)?\b|\b\d+(?:\.\B|(?:\.\d*)?(?:e[+-]?\d+)?\b)|\B\.\d+(?:e[+-]?\d+)?\b/i,
		keyword: /\b(?:and|break|do|else|elseif|end|false|for|function|goto|if|in|local|nil|not|or|repeat|return|then|true|until|while)\b/,
		function: /(?!\d)\w+(?=\s*(?:[({]))/,
		operator: [/[-+*%^&|#]|\/\/?|<[<=]?|>[>=]?|[=~]=?/, {
			pattern: /(^|[^.])\.\.(?!\.)/,
			lookbehind: !0
		}],
		punctuation: /[\[\](){},;]|\.+|:+/
	};
}
var Uh = R((() => {
	Hh.displayName = "lua", Hh.aliases = [];
})), Wh = /* @__PURE__ */ z({ default: () => Gh });
function Gh(e) {
	e.languages.makefile = {
		comment: {
			pattern: /(^|[^\\])#(?:\\(?:\r\n|[\s\S])|[^\\\r\n])*/,
			lookbehind: !0
		},
		string: {
			pattern: /(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,
			greedy: !0
		},
		"builtin-target": {
			pattern: /\.[A-Z][^:#=\s]+(?=\s*:(?!=))/,
			alias: "builtin"
		},
		target: {
			pattern: /^(?:[^:=\s]|[ \t]+(?![\s:]))+(?=\s*:(?!=))/m,
			alias: "symbol",
			inside: { variable: /\$+(?:(?!\$)[^(){}:#=\s]+|(?=[({]))/ }
		},
		variable: /\$+(?:(?!\$)[^(){}:#=\s]+|\([@*%<^+?][DF]\)|(?=[({]))/,
		keyword: /-include\b|\b(?:define|else|endef|endif|export|ifn?def|ifn?eq|include|override|private|sinclude|undefine|unexport|vpath)\b/,
		function: {
			pattern: /(\()(?:abspath|addsuffix|and|basename|call|dir|error|eval|file|filter(?:-out)?|findstring|firstword|flavor|foreach|guile|if|info|join|lastword|load|notdir|or|origin|patsubst|realpath|shell|sort|strip|subst|suffix|value|warning|wildcard|word(?:list|s)?)(?=[ \t])/,
			lookbehind: !0
		},
		operator: /(?:::|[?:+!])?=|[|@]/,
		punctuation: /[:;(){}]/
	};
}
var Kh = R((() => {
	Gh.displayName = "makefile", Gh.aliases = [];
})), qh = /* @__PURE__ */ z({ default: () => Jh });
function Jh(e) {
	e.languages.markup = {
		comment: {
			pattern: /<!--(?:(?!<!--)[\s\S])*?-->/,
			greedy: !0
		},
		prolog: {
			pattern: /<\?[\s\S]+?\?>/,
			greedy: !0
		},
		doctype: {
			pattern: /<!DOCTYPE(?:[^>"'[\]]|"[^"]*"|'[^']*')+(?:\[(?:[^<"'\]]|"[^"]*"|'[^']*'|<(?!!--)|<!--(?:[^-]|-(?!->))*-->)*\]\s*)?>/i,
			greedy: !0,
			inside: {
				"internal-subset": {
					pattern: /(^[^\[]*\[)[\s\S]+(?=\]>$)/,
					lookbehind: !0,
					greedy: !0,
					inside: null
				},
				string: {
					pattern: /"[^"]*"|'[^']*'/,
					greedy: !0
				},
				punctuation: /^<!|>$|[[\]]/,
				"doctype-tag": /^DOCTYPE/i,
				name: /[^\s<>'"]+/
			}
		},
		cdata: {
			pattern: /<!\[CDATA\[[\s\S]*?\]\]>/i,
			greedy: !0
		},
		tag: {
			pattern: /<\/?(?!\d)[^\s>\/=$<%]+(?:\s(?:\s*[^\s>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+(?=[\s>]))|(?=[\s/>])))+)?\s*\/?>/,
			greedy: !0,
			inside: {
				tag: {
					pattern: /^<\/?[^\s>\/]+/,
					inside: {
						punctuation: /^<\/?/,
						namespace: /^[^\s>\/:]+:/
					}
				},
				"special-attr": [],
				"attr-value": {
					pattern: /=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+)/,
					inside: { punctuation: [{
						pattern: /^=/,
						alias: "attr-equals"
					}, {
						pattern: /^(\s*)["']|["']$/,
						lookbehind: !0
					}] }
				},
				punctuation: /\/?>/,
				"attr-name": {
					pattern: /[^\s>\/]+/,
					inside: { namespace: /^[^\s>\/:]+:/ }
				}
			}
		},
		entity: [{
			pattern: /&[\da-z]{1,8};/i,
			alias: "named-entity"
		}, /&#x?[\da-f]{1,8};/i]
	}, e.languages.markup.tag.inside["attr-value"].inside.entity = e.languages.markup.entity, e.languages.markup.doctype.inside["internal-subset"].inside = e.languages.markup, e.hooks.add("wrap", function(e) {
		e.type === "entity" && (e.attributes.title = e.content.value.replace(/&amp;/, "&"));
	}), Object.defineProperty(e.languages.markup.tag, "addInlined", { value: function(t, n) {
		var r = {};
		r["language-" + n] = {
			pattern: /(^<!\[CDATA\[)[\s\S]+?(?=\]\]>$)/i,
			lookbehind: !0,
			inside: e.languages[n]
		}, r.cdata = /^<!\[CDATA\[|\]\]>$/i;
		var i = { "included-cdata": {
			pattern: /<!\[CDATA\[[\s\S]*?\]\]>/i,
			inside: r
		} };
		i["language-" + n] = {
			pattern: /[\s\S]+/,
			inside: e.languages[n]
		};
		var a = {};
		a[t] = {
			pattern: RegExp("(<__[^>]*>)(?:<!\\[CDATA\\[(?:[^\\]]|\\](?!\\]>))*\\]\\]>|(?!<!\\[CDATA\\[)[\\s\\S])*?(?=<\\/__>)".replace(/__/g, function() {
				return t;
			}), "i"),
			lookbehind: !0,
			greedy: !0,
			inside: i
		}, e.languages.insertBefore("markup", "cdata", a);
	} }), Object.defineProperty(e.languages.markup.tag, "addAttribute", { value: function(t, n) {
		e.languages.markup.tag.inside["special-attr"].push({
			pattern: RegExp("(^|[\"'\\s])(?:" + t + ")\\s*=\\s*(?:\"[^\"]*\"|'[^']*'|[^\\s'\">=]+(?=[\\s>]))", "i"),
			lookbehind: !0,
			inside: {
				"attr-name": /^[^\s=]+/,
				"attr-value": {
					pattern: /=[\s\S]+/,
					inside: {
						value: {
							pattern: /(^=\s*(["']|(?!["'])))\S[\s\S]*(?=\2$)/,
							lookbehind: !0,
							alias: [n, "language-" + n],
							inside: e.languages[n]
						},
						punctuation: [{
							pattern: /^=/,
							alias: "attr-equals"
						}, /"|'/]
					}
				}
			}
		});
	} }), e.languages.html = e.languages.markup, e.languages.mathml = e.languages.markup, e.languages.svg = e.languages.markup, e.languages.xml = e.languages.extend("markup", {}), e.languages.ssml = e.languages.xml, e.languages.atom = e.languages.xml, e.languages.rss = e.languages.xml;
}
var Yh = R((() => {
	Jh.displayName = "markup", Jh.aliases = [
		"atom",
		"html",
		"mathml",
		"rss",
		"ssml",
		"svg",
		"xml"
	];
})), Xh = /* @__PURE__ */ z({ default: () => Zh });
function Zh(e) {
	e.register(Jh), (function(e) {
		function t(e) {
			return e = e.replace(/<inner>/g, function() {
				return "(?:\\\\.|[^\\\\\\n\\r]|(?:\\n|\\r\\n?)(?![\\r\\n]))";
			}), RegExp("((?:^|[^\\\\])(?:\\\\{2})*)(?:" + e + ")");
		}
		var n = "(?:\\\\.|``(?:[^`\\r\\n]|`(?!`))+``|`[^`\\r\\n]+`|[^\\\\|\\r\\n`])+", r = "\\|?__(?:\\|__)+\\|?(?:(?:\\n|\\r\\n?)|(?![\\s\\S]))".replace(/__/g, function() {
			return n;
		}), i = "\\|?[ \\t]*:?-{3,}:?[ \\t]*(?:\\|[ \\t]*:?-{3,}:?[ \\t]*)+\\|?(?:\\n|\\r\\n?)";
		e.languages.markdown = e.languages.extend("markup", {}), e.languages.insertBefore("markdown", "prolog", {
			"front-matter-block": {
				pattern: /(^(?:\s*[\r\n])?)---(?!.)[\s\S]*?[\r\n]---(?!.)/,
				lookbehind: !0,
				greedy: !0,
				inside: {
					punctuation: /^---|---$/,
					"front-matter": {
						pattern: /\S+(?:\s+\S+)*/,
						alias: ["yaml", "language-yaml"],
						inside: e.languages.yaml
					}
				}
			},
			blockquote: {
				pattern: /^>(?:[\t ]*>)*/m,
				alias: "punctuation"
			},
			table: {
				pattern: RegExp("^" + r + i + "(?:" + r + ")*", "m"),
				inside: {
					"table-data-rows": {
						pattern: RegExp("^(" + r + i + ")(?:" + r + ")*$"),
						lookbehind: !0,
						inside: {
							"table-data": {
								pattern: RegExp(n),
								inside: e.languages.markdown
							},
							punctuation: /\|/
						}
					},
					"table-line": {
						pattern: RegExp("^(" + r + ")" + i + "$"),
						lookbehind: !0,
						inside: { punctuation: /\||:?-{3,}:?/ }
					},
					"table-header-row": {
						pattern: RegExp("^" + r + "$"),
						inside: {
							"table-header": {
								pattern: RegExp(n),
								alias: "important",
								inside: e.languages.markdown
							},
							punctuation: /\|/
						}
					}
				}
			},
			code: [{
				pattern: /((?:^|\n)[ \t]*\n|(?:^|\r\n?)[ \t]*\r\n?)(?: {4}|\t).+(?:(?:\n|\r\n?)(?: {4}|\t).+)*/,
				lookbehind: !0,
				alias: "keyword"
			}, {
				pattern: /^```[\s\S]*?^```$/m,
				greedy: !0,
				inside: {
					"code-block": {
						pattern: /^(```.*(?:\n|\r\n?))[\s\S]+?(?=(?:\n|\r\n?)^```$)/m,
						lookbehind: !0
					},
					"code-language": {
						pattern: /^(```).+/,
						lookbehind: !0
					},
					punctuation: /```/
				}
			}],
			title: [{
				pattern: /\S.*(?:\n|\r\n?)(?:==+|--+)(?=[ \t]*$)/m,
				alias: "important",
				inside: { punctuation: /==+$|--+$/ }
			}, {
				pattern: /(^\s*)#.+/m,
				lookbehind: !0,
				alias: "important",
				inside: { punctuation: /^#+|#+$/ }
			}],
			hr: {
				pattern: /(^\s*)([*-])(?:[\t ]*\2){2,}(?=\s*$)/m,
				lookbehind: !0,
				alias: "punctuation"
			},
			list: {
				pattern: /(^\s*)(?:[*+-]|\d+\.)(?=[\t ].)/m,
				lookbehind: !0,
				alias: "punctuation"
			},
			"url-reference": {
				pattern: /!?\[[^\]]+\]:[\t ]+(?:\S+|<(?:\\.|[^>\\])+>)(?:[\t ]+(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\((?:\\.|[^)\\])*\)))?/,
				inside: {
					variable: {
						pattern: /^(!?\[)[^\]]+/,
						lookbehind: !0
					},
					string: /(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\((?:\\.|[^)\\])*\))$/,
					punctuation: /^[\[\]!:]|[<>]/
				},
				alias: "url"
			},
			bold: {
				pattern: t("\\b__(?:(?!_)<inner>|_(?:(?!_)<inner>)+_)+__\\b|\\*\\*(?:(?!\\*)<inner>|\\*(?:(?!\\*)<inner>)+\\*)+\\*\\*"),
				lookbehind: !0,
				greedy: !0,
				inside: {
					content: {
						pattern: /(^..)[\s\S]+(?=..$)/,
						lookbehind: !0,
						inside: {}
					},
					punctuation: /\*\*|__/
				}
			},
			italic: {
				pattern: t("\\b_(?:(?!_)<inner>|__(?:(?!_)<inner>)+__)+_\\b|\\*(?:(?!\\*)<inner>|\\*\\*(?:(?!\\*)<inner>)+\\*\\*)+\\*"),
				lookbehind: !0,
				greedy: !0,
				inside: {
					content: {
						pattern: /(^.)[\s\S]+(?=.$)/,
						lookbehind: !0,
						inside: {}
					},
					punctuation: /[*_]/
				}
			},
			strike: {
				pattern: t("(~~?)(?:(?!~)<inner>)+\\2"),
				lookbehind: !0,
				greedy: !0,
				inside: {
					content: {
						pattern: /(^~~?)[\s\S]+(?=\1$)/,
						lookbehind: !0,
						inside: {}
					},
					punctuation: /~~?/
				}
			},
			"code-snippet": {
				pattern: /(^|[^\\`])(?:``[^`\r\n]+(?:`[^`\r\n]+)*``(?!`)|`[^`\r\n]+`(?!`))/,
				lookbehind: !0,
				greedy: !0,
				alias: ["code", "keyword"]
			},
			url: {
				pattern: t("!?\\[(?:(?!\\])<inner>)+\\](?:\\([^\\s)]+(?:[\\t ]+\"(?:\\\\.|[^\"\\\\])*\")?\\)|[ \\t]?\\[(?:(?!\\])<inner>)+\\])"),
				lookbehind: !0,
				greedy: !0,
				inside: {
					operator: /^!/,
					content: {
						pattern: /(^\[)[^\]]+(?=\])/,
						lookbehind: !0,
						inside: {}
					},
					variable: {
						pattern: /(^\][ \t]?\[)[^\]]+(?=\]$)/,
						lookbehind: !0
					},
					url: {
						pattern: /(^\]\()[^\s)]+/,
						lookbehind: !0
					},
					string: {
						pattern: /(^[ \t]+)"(?:\\.|[^"\\])*"(?=\)$)/,
						lookbehind: !0
					}
				}
			}
		}), [
			"url",
			"bold",
			"italic",
			"strike"
		].forEach(function(t) {
			[
				"url",
				"bold",
				"italic",
				"strike",
				"code-snippet"
			].forEach(function(n) {
				t !== n && (e.languages.markdown[t].inside.content.inside[n] = e.languages.markdown[n]);
			});
		}), e.hooks.add("after-tokenize", function(e) {
			if (e.language !== "markdown" && e.language !== "md") return;
			function t(e) {
				if (!(!e || typeof e == "string")) for (var n = 0, r = e.length; n < r; n++) {
					var i = e[n];
					if (i.type !== "code") {
						t(i.content);
						continue;
					}
					var a = i.content[1], o = i.content[3];
					if (a && o && a.type === "code-language" && o.type === "code-block" && typeof a.content == "string") {
						var s = a.content.replace(/\b#/g, "sharp").replace(/\b\+\+/g, "pp");
						s = (/[a-z][\w-]*/i.exec(s) || [""])[0].toLowerCase();
						var c = "language-" + s;
						o.alias ? typeof o.alias == "string" ? o.alias = [o.alias, c] : o.alias.push(c) : o.alias = [c];
					}
				}
			}
			t(e.tokens);
		}), e.hooks.add("wrap", function(t) {
			if (t.type === "code-block") {
				for (var n = "", r = 0, i = t.classes.length; r < i; r++) {
					var a = t.classes[r], o = /language-(.+)/.exec(a);
					if (o) {
						n = o[1];
						break;
					}
				}
				var s = e.languages[n];
				if (s) t.content = e.highlight(t.content.value, s, n);
				else if (n && n !== "none" && e.plugins.autoloader) {
					var c = "md-" + (/* @__PURE__ */ new Date()).valueOf() + "-" + Math.floor(Math.random() * 0x2386f26fc10000);
					t.attributes.id = c, e.plugins.autoloader.loadLanguages(n, function() {
						var t = document.getElementById(c);
						t && (t.innerHTML = e.highlight(t.textContent, e.languages[n], n));
					});
				}
			}
		}), RegExp(e.languages.markup.tag.pattern.source, "gi"), String.fromCodePoint || String.fromCharCode, e.languages.md = e.languages.markdown;
	})(e);
}
var Qh = R((() => {
	Yh(), Zh.displayName = "markdown", Zh.aliases = ["md"];
})), $h = /* @__PURE__ */ z({ default: () => eg });
function eg(e) {
	e.register(Jh), (function(e) {
		function t(e, t) {
			return "___" + e.toUpperCase() + t + "___";
		}
		Object.defineProperties(e.languages["markup-templating"] = {}, {
			buildPlaceholders: { value: function(n, r, i, a) {
				if (n.language === r) {
					var o = n.tokenStack = [];
					n.code = n.code.replace(i, function(e) {
						if (typeof a == "function" && !a(e)) return e;
						for (var i = o.length, s; n.code.indexOf(s = t(r, i)) !== -1;) ++i;
						return o[i] = e, s;
					}), n.grammar = e.languages.markup;
				}
			} },
			tokenizePlaceholders: { value: function(n, r) {
				if (n.language !== r || !n.tokenStack) return;
				n.grammar = e.languages[r];
				var i = 0, a = Object.keys(n.tokenStack);
				function o(s) {
					for (var c = 0; c < s.length && !(i >= a.length); c++) {
						var l = s[c];
						if (typeof l == "string" || l.content && typeof l.content == "string") {
							var u = a[i], d = n.tokenStack[u], f = typeof l == "string" ? l : l.content, p = t(r, u), m = f.indexOf(p);
							if (m > -1) {
								++i;
								var h = f.substring(0, m), g = new e.Token(r, e.tokenize(d, n.grammar), "language-" + r, d), _ = f.substring(m + p.length), v = [];
								h && v.push.apply(v, o([h])), v.push(g), _ && v.push.apply(v, o([_])), typeof l == "string" ? s.splice.apply(s, [c, 1].concat(v)) : l.content = v;
							}
						} else l.content && o(l.content);
					}
					return s;
				}
				o(n.tokens);
			} }
		});
	})(e);
}
var tg = R((() => {
	Yh(), eg.displayName = "markup-templating", eg.aliases = [];
})), ng = /* @__PURE__ */ z({ default: () => rg });
function rg(e) {
	e.register(Zm), e.languages.objectivec = e.languages.extend("c", {
		string: {
			pattern: /@?"(?:\\(?:\r\n|[\s\S])|[^"\\\r\n])*"/,
			greedy: !0
		},
		keyword: /\b(?:asm|auto|break|case|char|const|continue|default|do|double|else|enum|extern|float|for|goto|if|in|inline|int|long|register|return|self|short|signed|sizeof|static|struct|super|switch|typedef|typeof|union|unsigned|void|volatile|while)\b|(?:@interface|@end|@implementation|@protocol|@class|@public|@protected|@private|@property|@try|@catch|@finally|@throw|@synthesize|@dynamic|@selector)\b/,
		operator: /-[->]?|\+\+?|!=?|<<?=?|>>?=?|==?|&&?|\|\|?|[~^%?*\/@]/
	}), delete e.languages.objectivec["class-name"], e.languages.objc = e.languages.objectivec;
}
var ig = R((() => {
	Qm(), rg.displayName = "objectivec", rg.aliases = ["objc"];
})), ag = /* @__PURE__ */ z({ default: () => og });
function og(e) {
	(function(e) {
		var t = "(?:\\((?:[^()\\\\]|\\\\[\\s\\S])*\\)|\\{(?:[^{}\\\\]|\\\\[\\s\\S])*\\}|\\[(?:[^[\\]\\\\]|\\\\[\\s\\S])*\\]|<(?:[^<>\\\\]|\\\\[\\s\\S])*>)";
		e.languages.perl = {
			comment: [{
				pattern: /(^\s*)=\w[\s\S]*?=cut.*/m,
				lookbehind: !0,
				greedy: !0
			}, {
				pattern: /(^|[^\\$])#.*/,
				lookbehind: !0,
				greedy: !0
			}],
			string: [
				{
					pattern: RegExp("\\b(?:q|qq|qw|qx)(?![a-zA-Z0-9])\\s*(?:" + [
						"([^a-zA-Z0-9\\s{(\\[<])(?:(?!\\1)[^\\\\]|\\\\[\\s\\S])*\\1",
						"([a-zA-Z0-9])(?:(?!\\2)[^\\\\]|\\\\[\\s\\S])*\\2",
						t
					].join("|") + ")"),
					greedy: !0
				},
				{
					pattern: /("|`)(?:(?!\1)[^\\]|\\[\s\S])*\1/,
					greedy: !0
				},
				{
					pattern: /'(?:[^'\\\r\n]|\\.)*'/,
					greedy: !0
				}
			],
			regex: [
				{
					pattern: RegExp("\\b(?:m|qr)(?![a-zA-Z0-9])\\s*(?:" + [
						"([^a-zA-Z0-9\\s{(\\[<])(?:(?!\\1)[^\\\\]|\\\\[\\s\\S])*\\1",
						"([a-zA-Z0-9])(?:(?!\\2)[^\\\\]|\\\\[\\s\\S])*\\2",
						t
					].join("|") + ")[msixpodualngc]*"),
					greedy: !0
				},
				{
					pattern: RegExp("(^|[^-])\\b(?:s|tr|y)(?![a-zA-Z0-9])\\s*(?:" + [
						"([^a-zA-Z0-9\\s{(\\[<])(?:(?!\\2)[^\\\\]|\\\\[\\s\\S])*\\2(?:(?!\\2)[^\\\\]|\\\\[\\s\\S])*\\2",
						"([a-zA-Z0-9])(?:(?!\\3)[^\\\\]|\\\\[\\s\\S])*\\3(?:(?!\\3)[^\\\\]|\\\\[\\s\\S])*\\3",
						t + "\\s*" + t
					].join("|") + ")[msixpodualngcer]*"),
					lookbehind: !0,
					greedy: !0
				},
				{
					pattern: /\/(?:[^\/\\\r\n]|\\.)*\/[msixpodualngc]*(?=\s*(?:$|[\r\n,.;})&|\-+*~<>!?^]|(?:and|cmp|eq|ge|gt|le|lt|ne|not|or|x|xor)\b))/,
					greedy: !0
				}
			],
			variable: [
				/[&*$@%]\{\^[A-Z]+\}/,
				/[&*$@%]\^[A-Z_]/,
				/[&*$@%]#?(?=\{)/,
				/[&*$@%]#?(?:(?:::)*'?(?!\d)[\w$]+(?![\w$]))+(?:::)*/,
				/[&*$@%]\d+/,
				/(?!%=)[$@%][!"#$%&'()*+,\-.\/:;<=>?@[\\\]^_`{|}~]/
			],
			filehandle: {
				pattern: /<(?![<=])\S*?>|\b_\b/,
				alias: "symbol"
			},
			"v-string": {
				pattern: /v\d+(?:\.\d+)*|\d+(?:\.\d+){2,}/,
				alias: "string"
			},
			function: {
				pattern: /(\bsub[ \t]+)\w+/,
				lookbehind: !0
			},
			keyword: /\b(?:any|break|continue|default|delete|die|do|else|elsif|eval|for|foreach|given|goto|if|last|local|my|next|our|package|print|redo|require|return|say|state|sub|switch|undef|unless|until|use|when|while)\b/,
			number: /\b(?:0x[\dA-Fa-f](?:_?[\dA-Fa-f])*|0b[01](?:_?[01])*|(?:(?:\d(?:_?\d)*)?\.)?\d(?:_?\d)*(?:[Ee][+-]?\d+)?)\b/,
			operator: /-[rwxoRWXOezsfdlpSbctugkTBMAC]\b|\+[+=]?|-[-=>]?|\*\*?=?|\/\/?=?|=[=~>]?|~[~=]?|\|\|?=?|&&?=?|<(?:=>?|<=?)?|>>?=?|![~=]?|[%^]=?|\.(?:=|\.\.?)?|[\\?]|\bx(?:=|\b)|\b(?:and|cmp|eq|ge|gt|le|lt|ne|not|or|xor)\b/,
			punctuation: /[{}[\];(),:]/
		};
	})(e);
}
var sg = R((() => {
	og.displayName = "perl", og.aliases = [];
})), cg = /* @__PURE__ */ z({ default: () => lg });
function lg(e) {
	e.register(eg), (function(e) {
		var t = /\/\*[\s\S]*?\*\/|\/\/.*|#(?!\[).*/, n = [
			{
				pattern: /\b(?:false|true)\b/i,
				alias: "boolean"
			},
			{
				pattern: /(::\s*)\b[a-z_]\w*\b(?!\s*\()/i,
				greedy: !0,
				lookbehind: !0
			},
			{
				pattern: /(\b(?:case|const)\s+)\b[a-z_]\w*(?=\s*[;=])/i,
				greedy: !0,
				lookbehind: !0
			},
			/\b(?:null)\b/i,
			/\b[A-Z_][A-Z0-9_]*\b(?!\s*\()/
		], r = /\b0b[01]+(?:_[01]+)*\b|\b0o[0-7]+(?:_[0-7]+)*\b|\b0x[\da-f]+(?:_[\da-f]+)*\b|(?:\b\d+(?:_\d+)*\.?(?:\d+(?:_\d+)*)?|\B\.\d+)(?:e[+-]?\d+)?/i, i = /<?=>|\?\?=?|\.{3}|\??->|[!=]=?=?|::|\*\*=?|--|\+\+|&&|\|\||<<|>>|[?~]|[/^|%*&<>.+-]=?/, a = /[{}\[\](),:;]/;
		e.languages.php = {
			delimiter: {
				pattern: /\?>$|^<\?(?:php(?=\s)|=)?/i,
				alias: "important"
			},
			comment: t,
			variable: /\$+(?:\w+\b|(?=\{))/,
			package: {
				pattern: /(namespace\s+|use\s+(?:function\s+)?)(?:\\?\b[a-z_]\w*)+\b(?!\\)/i,
				lookbehind: !0,
				inside: { punctuation: /\\/ }
			},
			"class-name-definition": {
				pattern: /(\b(?:class|enum|interface|trait)\s+)\b[a-z_]\w*(?!\\)\b/i,
				lookbehind: !0,
				alias: "class-name"
			},
			"function-definition": {
				pattern: /(\bfunction\s+)[a-z_]\w*(?=\s*\()/i,
				lookbehind: !0,
				alias: "function"
			},
			keyword: [
				{
					pattern: /(\(\s*)\b(?:array|bool|boolean|float|int|integer|object|string)\b(?=\s*\))/i,
					alias: "type-casting",
					greedy: !0,
					lookbehind: !0
				},
				{
					pattern: /([(,?]\s*)\b(?:array(?!\s*\()|bool|callable|(?:false|null)(?=\s*\|)|float|int|iterable|mixed|object|self|static|string)\b(?=\s*\$)/i,
					alias: "type-hint",
					greedy: !0,
					lookbehind: !0
				},
				{
					pattern: /(\)\s*:\s*(?:\?\s*)?)\b(?:array(?!\s*\()|bool|callable|(?:false|null)(?=\s*\|)|float|int|iterable|mixed|never|object|self|static|string|void)\b/i,
					alias: "return-type",
					greedy: !0,
					lookbehind: !0
				},
				{
					pattern: /\b(?:array(?!\s*\()|bool|float|int|iterable|mixed|object|string|void)\b/i,
					alias: "type-declaration",
					greedy: !0
				},
				{
					pattern: /(\|\s*)(?:false|null)\b|\b(?:false|null)(?=\s*\|)/i,
					alias: "type-declaration",
					greedy: !0,
					lookbehind: !0
				},
				{
					pattern: /\b(?:parent|self|static)(?=\s*::)/i,
					alias: "static-context",
					greedy: !0
				},
				{
					pattern: /(\byield\s+)from\b/i,
					lookbehind: !0
				},
				/\bclass\b/i,
				{
					pattern: /((?:^|[^\s>:]|(?:^|[^-])>|(?:^|[^:]):)\s*)\b(?:abstract|and|array|as|break|callable|case|catch|clone|const|continue|declare|default|die|do|echo|else|elseif|empty|enddeclare|endfor|endforeach|endif|endswitch|endwhile|enum|eval|exit|extends|final|finally|fn|for|foreach|function|global|goto|if|implements|include|include_once|instanceof|insteadof|interface|isset|list|match|namespace|never|new|or|parent|print|private|protected|public|readonly|require|require_once|return|self|static|switch|throw|trait|try|unset|use|var|while|xor|yield|__halt_compiler)\b/i,
					lookbehind: !0
				}
			],
			"argument-name": {
				pattern: /([(,]\s*)\b[a-z_]\w*(?=\s*:(?!:))/i,
				lookbehind: !0
			},
			"class-name": [
				{
					pattern: /(\b(?:extends|implements|instanceof|new(?!\s+self|\s+static))\s+|\bcatch\s*\()\b[a-z_]\w*(?!\\)\b/i,
					greedy: !0,
					lookbehind: !0
				},
				{
					pattern: /(\|\s*)\b[a-z_]\w*(?!\\)\b/i,
					greedy: !0,
					lookbehind: !0
				},
				{
					pattern: /\b[a-z_]\w*(?!\\)\b(?=\s*\|)/i,
					greedy: !0
				},
				{
					pattern: /(\|\s*)(?:\\?\b[a-z_]\w*)+\b/i,
					alias: "class-name-fully-qualified",
					greedy: !0,
					lookbehind: !0,
					inside: { punctuation: /\\/ }
				},
				{
					pattern: /(?:\\?\b[a-z_]\w*)+\b(?=\s*\|)/i,
					alias: "class-name-fully-qualified",
					greedy: !0,
					inside: { punctuation: /\\/ }
				},
				{
					pattern: /(\b(?:extends|implements|instanceof|new(?!\s+self\b|\s+static\b))\s+|\bcatch\s*\()(?:\\?\b[a-z_]\w*)+\b(?!\\)/i,
					alias: "class-name-fully-qualified",
					greedy: !0,
					lookbehind: !0,
					inside: { punctuation: /\\/ }
				},
				{
					pattern: /\b[a-z_]\w*(?=\s*\$)/i,
					alias: "type-declaration",
					greedy: !0
				},
				{
					pattern: /(?:\\?\b[a-z_]\w*)+(?=\s*\$)/i,
					alias: ["class-name-fully-qualified", "type-declaration"],
					greedy: !0,
					inside: { punctuation: /\\/ }
				},
				{
					pattern: /\b[a-z_]\w*(?=\s*::)/i,
					alias: "static-context",
					greedy: !0
				},
				{
					pattern: /(?:\\?\b[a-z_]\w*)+(?=\s*::)/i,
					alias: ["class-name-fully-qualified", "static-context"],
					greedy: !0,
					inside: { punctuation: /\\/ }
				},
				{
					pattern: /([(,?]\s*)[a-z_]\w*(?=\s*\$)/i,
					alias: "type-hint",
					greedy: !0,
					lookbehind: !0
				},
				{
					pattern: /([(,?]\s*)(?:\\?\b[a-z_]\w*)+(?=\s*\$)/i,
					alias: ["class-name-fully-qualified", "type-hint"],
					greedy: !0,
					lookbehind: !0,
					inside: { punctuation: /\\/ }
				},
				{
					pattern: /(\)\s*:\s*(?:\?\s*)?)\b[a-z_]\w*(?!\\)\b/i,
					alias: "return-type",
					greedy: !0,
					lookbehind: !0
				},
				{
					pattern: /(\)\s*:\s*(?:\?\s*)?)(?:\\?\b[a-z_]\w*)+\b(?!\\)/i,
					alias: ["class-name-fully-qualified", "return-type"],
					greedy: !0,
					lookbehind: !0,
					inside: { punctuation: /\\/ }
				}
			],
			constant: n,
			function: {
				pattern: /(^|[^\\\w])\\?[a-z_](?:[\w\\]*\w)?(?=\s*\()/i,
				lookbehind: !0,
				inside: { punctuation: /\\/ }
			},
			property: {
				pattern: /(->\s*)\w+/,
				lookbehind: !0
			},
			number: r,
			operator: i,
			punctuation: a
		};
		var o = {
			pattern: /\{\$(?:\{(?:\{[^{}]+\}|[^{}]+)\}|[^{}])+\}|(^|[^\\{])\$+(?:\w+(?:\[[^\r\n\[\]]+\]|->\w+)?)/,
			lookbehind: !0,
			inside: e.languages.php
		}, s = [
			{
				pattern: /<<<'([^']+)'[\r\n](?:.*[\r\n])*?\1;/,
				alias: "nowdoc-string",
				greedy: !0,
				inside: { delimiter: {
					pattern: /^<<<'[^']+'|[a-z_]\w*;$/i,
					alias: "symbol",
					inside: { punctuation: /^<<<'?|[';]$/ }
				} }
			},
			{
				pattern: /<<<(?:"([^"]+)"[\r\n](?:.*[\r\n])*?\1;|([a-z_]\w*)[\r\n](?:.*[\r\n])*?\2;)/i,
				alias: "heredoc-string",
				greedy: !0,
				inside: {
					delimiter: {
						pattern: /^<<<(?:"[^"]+"|[a-z_]\w*)|[a-z_]\w*;$/i,
						alias: "symbol",
						inside: { punctuation: /^<<<"?|[";]$/ }
					},
					interpolation: o
				}
			},
			{
				pattern: /`(?:\\[\s\S]|[^\\`])*`/,
				alias: "backtick-quoted-string",
				greedy: !0
			},
			{
				pattern: /'(?:\\[\s\S]|[^\\'])*'/,
				alias: "single-quoted-string",
				greedy: !0
			},
			{
				pattern: /"(?:\\[\s\S]|[^\\"])*"/,
				alias: "double-quoted-string",
				greedy: !0,
				inside: { interpolation: o }
			}
		];
		e.languages.insertBefore("php", "variable", {
			string: s,
			attribute: {
				pattern: /#\[(?:[^"'\/#]|\/(?![*/])|\/\/.*$|#(?!\[).*$|\/\*(?:[^*]|\*(?!\/))*\*\/|"(?:\\[\s\S]|[^\\"])*"|'(?:\\[\s\S]|[^\\'])*')+\](?=\s*[a-z$#])/im,
				greedy: !0,
				inside: {
					"attribute-content": {
						pattern: /^(#\[)[\s\S]+(?=\]$)/,
						lookbehind: !0,
						inside: {
							comment: t,
							string: s,
							"attribute-class-name": [{
								pattern: /([^:]|^)\b[a-z_]\w*(?!\\)\b/i,
								alias: "class-name",
								greedy: !0,
								lookbehind: !0
							}, {
								pattern: /([^:]|^)(?:\\?\b[a-z_]\w*)+/i,
								alias: ["class-name", "class-name-fully-qualified"],
								greedy: !0,
								lookbehind: !0,
								inside: { punctuation: /\\/ }
							}],
							constant: n,
							number: r,
							operator: i,
							punctuation: a
						}
					},
					delimiter: {
						pattern: /^#\[|\]$/,
						alias: "punctuation"
					}
				}
			}
		}), e.hooks.add("before-tokenize", function(t) {
			/<\?/.test(t.code) && e.languages["markup-templating"].buildPlaceholders(t, "php", /<\?(?:[^"'/#]|\/(?![*/])|("|')(?:\\[\s\S]|(?!\1)[^\\])*\1|(?:\/\/|#(?!\[))(?:[^?\n\r]|\?(?!>))*(?=$|\?>|[\r\n])|#\[|\/\*(?:[^*]|\*(?!\/))*(?:\*\/|$))*?(?:\?>|$)/g);
		}), e.hooks.add("after-tokenize", function(t) {
			e.languages["markup-templating"].tokenizePlaceholders(t, "php");
		});
	})(e);
}
var ug = R((() => {
	tg(), lg.displayName = "php", lg.aliases = [];
})), dg = /* @__PURE__ */ z({ default: () => fg });
function fg(e) {
	e.languages.python = {
		comment: {
			pattern: /(^|[^\\])#.*/,
			lookbehind: !0,
			greedy: !0
		},
		"string-interpolation": {
			pattern: /(?:f|fr|rf)(?:("""|''')[\s\S]*?\1|("|')(?:\\.|(?!\2)[^\\\r\n])*\2)/i,
			greedy: !0,
			inside: {
				interpolation: {
					pattern: /((?:^|[^{])(?:\{\{)*)\{(?!\{)(?:[^{}]|\{(?!\{)(?:[^{}]|\{(?!\{)(?:[^{}])+\})+\})+\}/,
					lookbehind: !0,
					inside: {
						"format-spec": {
							pattern: /(:)[^:(){}]+(?=\}$)/,
							lookbehind: !0
						},
						"conversion-option": {
							pattern: /![sra](?=[:}]$)/,
							alias: "punctuation"
						},
						rest: null
					}
				},
				string: /[\s\S]+/
			}
		},
		"triple-quoted-string": {
			pattern: /(?:[rub]|br|rb)?("""|''')[\s\S]*?\1/i,
			greedy: !0,
			alias: "string"
		},
		string: {
			pattern: /(?:[rub]|br|rb)?("|')(?:\\.|(?!\1)[^\\\r\n])*\1/i,
			greedy: !0
		},
		function: {
			pattern: /((?:^|\s)def[ \t]+)[a-zA-Z_]\w*(?=\s*\()/g,
			lookbehind: !0
		},
		"class-name": {
			pattern: /(\bclass\s+)\w+/i,
			lookbehind: !0
		},
		decorator: {
			pattern: /(^[\t ]*)@\w+(?:\.\w+)*/m,
			lookbehind: !0,
			alias: ["annotation", "punctuation"],
			inside: { punctuation: /\./ }
		},
		keyword: /\b(?:_(?=\s*:)|and|as|assert|async|await|break|case|class|continue|def|del|elif|else|except|exec|finally|for|from|global|if|import|in|is|lambda|match|nonlocal|not|or|pass|print|raise|return|try|while|with|yield)\b/,
		builtin: /\b(?:__import__|abs|all|any|apply|ascii|basestring|bin|bool|buffer|bytearray|bytes|callable|chr|classmethod|cmp|coerce|compile|complex|delattr|dict|dir|divmod|enumerate|eval|execfile|file|filter|float|format|frozenset|getattr|globals|hasattr|hash|help|hex|id|input|int|intern|isinstance|issubclass|iter|len|list|locals|long|map|max|memoryview|min|next|object|oct|open|ord|pow|property|range|raw_input|reduce|reload|repr|reversed|round|set|setattr|slice|sorted|staticmethod|str|sum|super|tuple|type|unichr|unicode|vars|xrange|zip)\b/,
		boolean: /\b(?:False|None|True)\b/,
		number: /\b0(?:b(?:_?[01])+|o(?:_?[0-7])+|x(?:_?[a-f0-9])+)\b|(?:\b\d+(?:_\d+)*(?:\.(?:\d+(?:_\d+)*)?)?|\B\.\d+(?:_\d+)*)(?:e[+-]?\d+(?:_\d+)*)?j?(?!\w)/i,
		operator: /[-+%=]=?|!=|:=|\*\*?=?|\/\/?=?|<[<=>]?|>[=>]?|[&|^~]/,
		punctuation: /[{}[\];(),.:]/
	}, e.languages.python["string-interpolation"].inside.interpolation.inside.rest = e.languages.python, e.languages.py = e.languages.python;
}
var pg = R((() => {
	fg.displayName = "python", fg.aliases = ["py"];
})), mg = /* @__PURE__ */ z({ default: () => hg });
function hg(e) {
	e.languages.r = {
		comment: /#.*/,
		string: {
			pattern: /(['"])(?:\\.|(?!\1)[^\\\r\n])*\1/,
			greedy: !0
		},
		"percent-operator": {
			pattern: /%[^%\s]*%/,
			alias: "operator"
		},
		boolean: /\b(?:FALSE|TRUE)\b/,
		ellipsis: /\.\.(?:\.|\d+)/,
		number: [/\b(?:Inf|NaN)\b/, /(?:\b0x[\dA-Fa-f]+(?:\.\d*)?|\b\d+(?:\.\d*)?|\B\.\d+)(?:[EePp][+-]?\d+)?[iL]?/],
		keyword: /\b(?:NA|NA_character_|NA_complex_|NA_integer_|NA_real_|NULL|break|else|for|function|if|in|next|repeat|while)\b/,
		operator: /->?>?|<(?:=|<?-)?|[>=!]=?|::?|&&?|\|\|?|[+*\/^$@~]/,
		punctuation: /[(){}\[\],;]/
	};
}
var gg = R((() => {
	hg.displayName = "r", hg.aliases = [];
})), _g = /* @__PURE__ */ z({ default: () => vg });
function vg(e) {
	(function(e) {
		var t = {
			pattern: /\\[\\(){}[\]^$+*?|.]/,
			alias: "escape"
		}, n = /\\(?:x[\da-fA-F]{2}|u[\da-fA-F]{4}|u\{[\da-fA-F]+\}|0[0-7]{0,2}|[123][0-7]{2}|c[a-zA-Z]|.)/, r = {
			pattern: /\.|\\[wsd]|\\p\{[^{}]+\}/i,
			alias: "class-name"
		}, i = {
			pattern: /\\[wsd]|\\p\{[^{}]+\}/i,
			alias: "class-name"
		}, a = "(?:[^\\\\-]|" + n.source + ")", o = RegExp(a + "-" + a), s = {
			pattern: /(<|')[^<>']+(?=[>']$)/,
			lookbehind: !0,
			alias: "variable"
		};
		e.languages.regex = {
			"char-class": {
				pattern: /((?:^|[^\\])(?:\\\\)*)\[(?:[^\\\]]|\\[\s\S])*\]/,
				lookbehind: !0,
				inside: {
					"char-class-negation": {
						pattern: /(^\[)\^/,
						lookbehind: !0,
						alias: "operator"
					},
					"char-class-punctuation": {
						pattern: /^\[|\]$/,
						alias: "punctuation"
					},
					range: {
						pattern: o,
						inside: {
							escape: n,
							"range-punctuation": {
								pattern: /-/,
								alias: "operator"
							}
						}
					},
					"special-escape": t,
					"char-set": i,
					escape: n
				}
			},
			"special-escape": t,
			"char-set": r,
			backreference: [{
				pattern: /\\(?![123][0-7]{2})[1-9]/,
				alias: "keyword"
			}, {
				pattern: /\\k<[^<>']+>/,
				alias: "keyword",
				inside: { "group-name": s }
			}],
			anchor: {
				pattern: /[$^]|\\[ABbGZz]/,
				alias: "function"
			},
			escape: n,
			group: [{
				pattern: /\((?:\?(?:<[^<>']+>|'[^<>']+'|[>:]|<?[=!]|[idmnsuxU]+(?:-[idmnsuxU]+)?:?))?/,
				alias: "punctuation",
				inside: { "group-name": s }
			}, {
				pattern: /\)/,
				alias: "punctuation"
			}],
			quantifier: {
				pattern: /(?:[+*?]|\{\d+(?:,\d*)?\})[?+]?/,
				alias: "number"
			},
			alternation: {
				pattern: /\|/,
				alias: "keyword"
			}
		};
	})(e);
}
var yg = R((() => {
	vg.displayName = "regex", vg.aliases = [];
})), bg = /* @__PURE__ */ z({ default: () => xg });
function xg(e) {
	e.register(Jm), (function(e) {
		e.languages.ruby = e.languages.extend("clike", {
			comment: {
				pattern: /#.*|^=begin\s[\s\S]*?^=end/m,
				greedy: !0
			},
			"class-name": {
				pattern: /(\b(?:class|module)\s+|\bcatch\s+\()[\w.\\]+|\b[A-Z_]\w*(?=\s*\.\s*new\b)/,
				lookbehind: !0,
				inside: { punctuation: /[.\\]/ }
			},
			keyword: /\b(?:BEGIN|END|alias|and|begin|break|case|class|def|define_method|defined|do|each|else|elsif|end|ensure|extend|for|if|in|include|module|new|next|nil|not|or|prepend|private|protected|public|raise|redo|require|rescue|retry|return|self|super|then|throw|undef|unless|until|when|while|yield)\b/,
			operator: /\.{2,3}|&\.|===|<?=>|[!=]?~|(?:&&|\|\||<<|>>|\*\*|[+\-*/%<>!^&|=])=?|[?:]/,
			punctuation: /[(){}[\].,;]/
		}), e.languages.insertBefore("ruby", "operator", { "double-colon": {
			pattern: /::/,
			alias: "punctuation"
		} });
		var t = {
			pattern: /((?:^|[^\\])(?:\\{2})*)#\{(?:[^{}]|\{[^{}]*\})*\}/,
			lookbehind: !0,
			inside: {
				content: {
					pattern: /^(#\{)[\s\S]+(?=\}$)/,
					lookbehind: !0,
					inside: e.languages.ruby
				},
				delimiter: {
					pattern: /^#\{|\}$/,
					alias: "punctuation"
				}
			}
		};
		delete e.languages.ruby.function;
		var n = "(?:" + [
			"([^a-zA-Z0-9\\s{(\\[<=])(?:(?!\\1)[^\\\\]|\\\\[\\s\\S])*\\1",
			"\\((?:[^()\\\\]|\\\\[\\s\\S]|\\((?:[^()\\\\]|\\\\[\\s\\S])*\\))*\\)",
			"\\{(?:[^{}\\\\]|\\\\[\\s\\S]|\\{(?:[^{}\\\\]|\\\\[\\s\\S])*\\})*\\}",
			"\\[(?:[^\\[\\]\\\\]|\\\\[\\s\\S]|\\[(?:[^\\[\\]\\\\]|\\\\[\\s\\S])*\\])*\\]",
			"<(?:[^<>\\\\]|\\\\[\\s\\S]|<(?:[^<>\\\\]|\\\\[\\s\\S])*>)*>"
		].join("|") + ")", r = "(?:\"(?:\\\\.|[^\"\\\\\\r\\n])*\"|(?:\\b[a-zA-Z_]\\w*|[^\\s\\0-\\x7F]+)[?!]?|\\$.)";
		e.languages.insertBefore("ruby", "keyword", {
			"regex-literal": [{
				pattern: RegExp("%r" + n + "[egimnosux]{0,6}"),
				greedy: !0,
				inside: {
					interpolation: t,
					regex: /[\s\S]+/
				}
			}, {
				pattern: /(^|[^/])\/(?!\/)(?:\[[^\r\n\]]+\]|\\.|[^[/\\\r\n])+\/[egimnosux]{0,6}(?=\s*(?:$|[\r\n,.;})#]))/,
				lookbehind: !0,
				greedy: !0,
				inside: {
					interpolation: t,
					regex: /[\s\S]+/
				}
			}],
			variable: /[@$]+[a-zA-Z_]\w*(?:[?!]|\b)/,
			symbol: [{
				pattern: RegExp("(^|[^:]):" + r),
				lookbehind: !0,
				greedy: !0
			}, {
				pattern: RegExp("([\\r\\n{(,][ \\t]*)" + r + "(?=:(?!:))"),
				lookbehind: !0,
				greedy: !0
			}],
			"method-definition": {
				pattern: /(\bdef\s+)\w+(?:\s*\.\s*\w+)?/,
				lookbehind: !0,
				inside: {
					function: /\b\w+$/,
					keyword: /^self\b/,
					"class-name": /^\w+/,
					punctuation: /\./
				}
			}
		}), e.languages.insertBefore("ruby", "string", {
			"string-literal": [
				{
					pattern: RegExp("%[qQiIwWs]?" + n),
					greedy: !0,
					inside: {
						interpolation: t,
						string: /[\s\S]+/
					}
				},
				{
					pattern: /("|')(?:#\{[^}]+\}|#(?!\{)|\\(?:\r\n|[\s\S])|(?!\1)[^\\#\r\n])*\1/,
					greedy: !0,
					inside: {
						interpolation: t,
						string: /[\s\S]+/
					}
				},
				{
					pattern: /<<[-~]?([a-z_]\w*)[\r\n](?:.*[\r\n])*?[\t ]*\1/i,
					alias: "heredoc-string",
					greedy: !0,
					inside: {
						delimiter: {
							pattern: /^<<[-~]?[a-z_]\w*|\b[a-z_]\w*$/i,
							inside: {
								symbol: /\b\w+/,
								punctuation: /^<<[-~]?/
							}
						},
						interpolation: t,
						string: /[\s\S]+/
					}
				},
				{
					pattern: /<<[-~]?'([a-z_]\w*)'[\r\n](?:.*[\r\n])*?[\t ]*\1/i,
					alias: "heredoc-string",
					greedy: !0,
					inside: {
						delimiter: {
							pattern: /^<<[-~]?'[a-z_]\w*'|\b[a-z_]\w*$/i,
							inside: {
								symbol: /\b\w+/,
								punctuation: /^<<[-~]?'|'$/
							}
						},
						string: /[\s\S]+/
					}
				}
			],
			"command-literal": [{
				pattern: RegExp("%x" + n),
				greedy: !0,
				inside: {
					interpolation: t,
					command: {
						pattern: /[\s\S]+/,
						alias: "string"
					}
				}
			}, {
				pattern: /`(?:#\{[^}]+\}|#(?!\{)|\\(?:\r\n|[\s\S])|[^\\`#\r\n])*`/,
				greedy: !0,
				inside: {
					interpolation: t,
					command: {
						pattern: /[\s\S]+/,
						alias: "string"
					}
				}
			}]
		}), delete e.languages.ruby.string, e.languages.insertBefore("ruby", "number", {
			builtin: /\b(?:Array|Bignum|Binding|Class|Continuation|Dir|Exception|FalseClass|File|Fixnum|Float|Hash|IO|Integer|MatchData|Method|Module|NilClass|Numeric|Object|Proc|Range|Regexp|Stat|String|Struct|Symbol|TMS|Thread|ThreadGroup|Time|TrueClass)\b/,
			constant: /\b[A-Z][A-Z0-9_]*(?:[?!]|\b)/
		}), e.languages.rb = e.languages.ruby;
	})(e);
}
var Sg = R((() => {
	Ym(), xg.displayName = "ruby", xg.aliases = ["rb"];
})), Cg = /* @__PURE__ */ z({ default: () => wg });
function wg(e) {
	(function(e) {
		for (var t = "\\/\\*(?:[^*/]|\\*(?!\\/)|\\/(?!\\*)|<self>)*\\*\\/", n = 0; n < 2; n++) t = t.replace(/<self>/g, function() {
			return t;
		});
		t = t.replace(/<self>/g, function() {
			return "[^\\s\\S]";
		}), e.languages.rust = {
			comment: [{
				pattern: RegExp("(^|[^\\\\])" + t),
				lookbehind: !0,
				greedy: !0
			}, {
				pattern: /(^|[^\\:])\/\/.*/,
				lookbehind: !0,
				greedy: !0
			}],
			string: {
				pattern: /b?"(?:\\[\s\S]|[^\\"])*"|b?r(#*)"(?:[^"]|"(?!\1))*"\1/,
				greedy: !0
			},
			char: {
				pattern: /b?'(?:\\(?:x[0-7][\da-fA-F]|u\{(?:[\da-fA-F]_*){1,6}\}|.)|[^\\\r\n\t'])'/,
				greedy: !0
			},
			attribute: {
				pattern: /#!?\[(?:[^\[\]"]|"(?:\\[\s\S]|[^\\"])*")*\]/,
				greedy: !0,
				alias: "attr-name",
				inside: { string: null }
			},
			"closure-params": {
				pattern: /([=(,:]\s*|\bmove\s*)\|[^|]*\||\|[^|]*\|(?=\s*(?:\{|->))/,
				lookbehind: !0,
				greedy: !0,
				inside: {
					"closure-punctuation": {
						pattern: /^\||\|$/,
						alias: "punctuation"
					},
					rest: null
				}
			},
			"lifetime-annotation": {
				pattern: /'\w+/,
				alias: "symbol"
			},
			"fragment-specifier": {
				pattern: /(\$\w+:)[a-z]+/,
				lookbehind: !0,
				alias: "punctuation"
			},
			variable: /\$\w+/,
			"function-definition": {
				pattern: /(\bfn\s+)\w+/,
				lookbehind: !0,
				alias: "function"
			},
			"type-definition": {
				pattern: /(\b(?:enum|struct|trait|type|union)\s+)\w+/,
				lookbehind: !0,
				alias: "class-name"
			},
			"module-declaration": [{
				pattern: /(\b(?:crate|mod)\s+)[a-z][a-z_\d]*/,
				lookbehind: !0,
				alias: "namespace"
			}, {
				pattern: /(\b(?:crate|self|super)\s*)::\s*[a-z][a-z_\d]*\b(?:\s*::(?:\s*[a-z][a-z_\d]*\s*::)*)?/,
				lookbehind: !0,
				alias: "namespace",
				inside: { punctuation: /::/ }
			}],
			keyword: [/\b(?:Self|abstract|as|async|await|become|box|break|const|continue|crate|do|dyn|else|enum|extern|final|fn|for|if|impl|in|let|loop|macro|match|mod|move|mut|override|priv|pub|ref|return|self|static|struct|super|trait|try|type|typeof|union|unsafe|unsized|use|virtual|where|while|yield)\b/, /\b(?:bool|char|f(?:32|64)|[ui](?:8|16|32|64|128|size)|str)\b/],
			function: /\b[a-z_]\w*(?=\s*(?:::\s*<|\())/,
			macro: {
				pattern: /\b\w+!/,
				alias: "property"
			},
			constant: /\b[A-Z_][A-Z_\d]+\b/,
			"class-name": /\b[A-Z]\w*\b/,
			namespace: {
				pattern: /(?:\b[a-z][a-z_\d]*\s*::\s*)*\b[a-z][a-z_\d]*\s*::(?!\s*<)/,
				inside: { punctuation: /::/ }
			},
			number: /\b(?:0x[\dA-Fa-f](?:_?[\dA-Fa-f])*|0o[0-7](?:_?[0-7])*|0b[01](?:_?[01])*|(?:(?:\d(?:_?\d)*)?\.)?\d(?:_?\d)*(?:[Ee][+-]?\d+)?)(?:_?(?:f32|f64|[iu](?:8|16|32|64|size)?))?\b/,
			boolean: /\b(?:false|true)\b/,
			punctuation: /->|\.\.=|\.{1,3}|::|[{}[\];(),:]/,
			operator: /[-+*\/%!^]=?|=[=>]?|&[&=]?|\|[|=]?|<<?=?|>>?=?|[@?]/
		}, e.languages.rust["closure-params"].inside.rest = e.languages.rust, e.languages.rust.attribute.inside.string = e.languages.rust.string;
	})(e);
}
var Tg = R((() => {
	wg.displayName = "rust", wg.aliases = [];
})), Eg = /* @__PURE__ */ z({ default: () => Dg });
function Dg(e) {
	e.register(hh), (function(e) {
		e.languages.sass = e.languages.extend("css", { comment: {
			pattern: /^([ \t]*)\/[\/*].*(?:(?:\r?\n|\r)\1[ \t].+)*/m,
			lookbehind: !0,
			greedy: !0
		} }), e.languages.insertBefore("sass", "atrule", { "atrule-line": {
			pattern: /^(?:[ \t]*)[@+=].+/m,
			greedy: !0,
			inside: { atrule: /(?:@[\w-]+|[+=])/ }
		} }), delete e.languages.sass.atrule;
		var t = /\$[-\w]+|#\{\$[-\w]+\}/, n = [/[+*\/%]|[=!]=|<=?|>=?|\b(?:and|not|or)\b/, {
			pattern: /(\s)-(?=\s)/,
			lookbehind: !0
		}];
		e.languages.insertBefore("sass", "property", {
			"variable-line": {
				pattern: /^[ \t]*\$.+/m,
				greedy: !0,
				inside: {
					punctuation: /:/,
					variable: t,
					operator: n
				}
			},
			"property-line": {
				pattern: /^[ \t]*(?:[^:\s]+ *:.*|:[^:\s].*)/m,
				greedy: !0,
				inside: {
					property: [/[^:\s]+(?=\s*:)/, {
						pattern: /(:)[^:\s]+/,
						lookbehind: !0
					}],
					punctuation: /:/,
					variable: t,
					operator: n,
					important: e.languages.sass.important
				}
			}
		}), delete e.languages.sass.property, delete e.languages.sass.important, e.languages.insertBefore("sass", "punctuation", { selector: {
			pattern: /^([ \t]*)\S(?:,[^,\r\n]+|[^,\r\n]*)(?:,[^,\r\n]+)*(?:,(?:\r?\n|\r)\1[ \t]+\S(?:,[^,\r\n]+|[^,\r\n]*)(?:,[^,\r\n]+)*)*/m,
			lookbehind: !0,
			greedy: !0
		} });
	})(e);
}
var Og = R((() => {
	gh(), Dg.displayName = "sass", Dg.aliases = [];
})), kg = /* @__PURE__ */ z({ default: () => Ag });
function Ag(e) {
	e.register(hh), e.languages.scss = e.languages.extend("css", {
		comment: {
			pattern: /(^|[^\\])(?:\/\*[\s\S]*?\*\/|\/\/.*)/,
			lookbehind: !0
		},
		atrule: {
			pattern: /@[\w-](?:\([^()]+\)|[^()\s]|\s+(?!\s))*?(?=\s+[{;])/,
			inside: { rule: /@[\w-]+/ }
		},
		url: /(?:[-a-z]+-)?url(?=\()/i,
		selector: {
			pattern: /(?=\S)[^@;{}()]?(?:[^@;{}()\s]|\s+(?!\s)|#\{\$[-\w]+\})+(?=\s*\{(?:\}|\s|[^}][^:{}]*[:{][^}]))/,
			inside: {
				parent: {
					pattern: /&/,
					alias: "important"
				},
				placeholder: /%[-\w]+/,
				variable: /\$[-\w]+|#\{\$[-\w]+\}/
			}
		},
		property: {
			pattern: /(?:[-\w]|\$[-\w]|#\{\$[-\w]+\})+(?=\s*:)/,
			inside: { variable: /\$[-\w]+|#\{\$[-\w]+\}/ }
		}
	}), e.languages.insertBefore("scss", "atrule", { keyword: [/@(?:content|debug|each|else(?: if)?|extend|for|forward|function|if|import|include|mixin|return|use|warn|while)\b/i, {
		pattern: /( )(?:from|through)(?= )/,
		lookbehind: !0
	}] }), e.languages.insertBefore("scss", "important", { variable: /\$[-\w]+|#\{\$[-\w]+\}/ }), e.languages.insertBefore("scss", "function", {
		"module-modifier": {
			pattern: /\b(?:as|hide|show|with)\b/i,
			alias: "keyword"
		},
		placeholder: {
			pattern: /%[-\w]+/,
			alias: "selector"
		},
		statement: {
			pattern: /\B!(?:default|optional)\b/i,
			alias: "keyword"
		},
		boolean: /\b(?:false|true)\b/,
		null: {
			pattern: /\bnull\b/,
			alias: "keyword"
		},
		operator: {
			pattern: /(\s)(?:[-+*\/%]|[=!]=|<=?|>=?|and|not|or)(?=\s)/,
			lookbehind: !0
		}
	}), e.languages.scss.atrule.inside.rest = e.languages.scss;
}
var jg = R((() => {
	gh(), Ag.displayName = "scss", Ag.aliases = [];
})), Mg = /* @__PURE__ */ z({ default: () => Ng });
function Ng(e) {
	e.languages.sql = {
		comment: {
			pattern: /(^|[^\\])(?:\/\*[\s\S]*?\*\/|(?:--|\/\/|#).*)/,
			lookbehind: !0
		},
		variable: [{
			pattern: /@(["'`])(?:\\[\s\S]|(?!\1)[^\\])+\1/,
			greedy: !0
		}, /@[\w.$]+/],
		string: {
			pattern: /(^|[^@\\])("|')(?:\\[\s\S]|(?!\2)[^\\]|\2\2)*\2/,
			greedy: !0,
			lookbehind: !0
		},
		identifier: {
			pattern: /(^|[^@\\])`(?:\\[\s\S]|[^`\\]|``)*`/,
			greedy: !0,
			lookbehind: !0,
			inside: { punctuation: /^`|`$/ }
		},
		function: /\b(?:AVG|COUNT|FIRST|FORMAT|LAST|LCASE|LEN|MAX|MID|MIN|MOD|NOW|ROUND|SUM|UCASE)(?=\s*\()/i,
		keyword: /\b(?:ACTION|ADD|AFTER|ALGORITHM|ALL|ALTER|ANALYZE|ANY|APPLY|AS|ASC|AUTHORIZATION|AUTO_INCREMENT|BACKUP|BDB|BEGIN|BERKELEYDB|BIGINT|BINARY|BIT|BLOB|BOOL|BOOLEAN|BREAK|BROWSE|BTREE|BULK|BY|CALL|CASCADED?|CASE|CHAIN|CHAR(?:ACTER|SET)?|CHECK(?:POINT)?|CLOSE|CLUSTERED|COALESCE|COLLATE|COLUMNS?|COMMENT|COMMIT(?:TED)?|COMPUTE|CONNECT|CONSISTENT|CONSTRAINT|CONTAINS(?:TABLE)?|CONTINUE|CONVERT|CREATE|CROSS|CURRENT(?:_DATE|_TIME|_TIMESTAMP|_USER)?|CURSOR|CYCLE|DATA(?:BASES?)?|DATE(?:TIME)?|DAY|DBCC|DEALLOCATE|DEC|DECIMAL|DECLARE|DEFAULT|DEFINER|DELAYED|DELETE|DELIMITERS?|DENY|DESC|DESCRIBE|DETERMINISTIC|DISABLE|DISCARD|DISK|DISTINCT|DISTINCTROW|DISTRIBUTED|DO|DOUBLE|DROP|DUMMY|DUMP(?:FILE)?|DUPLICATE|ELSE(?:IF)?|ENABLE|ENCLOSED|END|ENGINE|ENUM|ERRLVL|ERRORS|ESCAPED?|EXCEPT|EXEC(?:UTE)?|EXISTS|EXIT|EXPLAIN|EXTENDED|FETCH|FIELDS|FILE|FILLFACTOR|FIRST|FIXED|FLOAT|FOLLOWING|FOR(?: EACH ROW)?|FORCE|FOREIGN|FREETEXT(?:TABLE)?|FROM|FULL|FUNCTION|GEOMETRY(?:COLLECTION)?|GLOBAL|GOTO|GRANT|GROUP|HANDLER|HASH|HAVING|HOLDLOCK|HOUR|IDENTITY(?:COL|_INSERT)?|IF|IGNORE|IMPORT|INDEX|INFILE|INNER|INNODB|INOUT|INSERT|INT|INTEGER|INTERSECT|INTERVAL|INTO|INVOKER|ISOLATION|ITERATE|JOIN|KEYS?|KILL|LANGUAGE|LAST|LEAVE|LEFT|LEVEL|LIMIT|LINENO|LINES|LINESTRING|LOAD|LOCAL|LOCK|LONG(?:BLOB|TEXT)|LOOP|MATCH(?:ED)?|MEDIUM(?:BLOB|INT|TEXT)|MERGE|MIDDLEINT|MINUTE|MODE|MODIFIES|MODIFY|MONTH|MULTI(?:LINESTRING|POINT|POLYGON)|NATIONAL|NATURAL|NCHAR|NEXT|NO|NONCLUSTERED|NULLIF|NUMERIC|OFF?|OFFSETS?|ON|OPEN(?:DATASOURCE|QUERY|ROWSET)?|OPTIMIZE|OPTION(?:ALLY)?|ORDER|OUT(?:ER|FILE)?|OVER|PARTIAL|PARTITION|PERCENT|PIVOT|PLAN|POINT|POLYGON|PRECEDING|PRECISION|PREPARE|PREV|PRIMARY|PRINT|PRIVILEGES|PROC(?:EDURE)?|PUBLIC|PURGE|QUICK|RAISERROR|READS?|REAL|RECONFIGURE|REFERENCES|RELEASE|RENAME|REPEAT(?:ABLE)?|REPLACE|REPLICATION|REQUIRE|RESIGNAL|RESTORE|RESTRICT|RETURN(?:ING|S)?|REVOKE|RIGHT|ROLLBACK|ROUTINE|ROW(?:COUNT|GUIDCOL|S)?|RTREE|RULE|SAVE(?:POINT)?|SCHEMA|SECOND|SELECT|SERIAL(?:IZABLE)?|SESSION(?:_USER)?|SET(?:USER)?|SHARE|SHOW|SHUTDOWN|SIMPLE|SMALLINT|SNAPSHOT|SOME|SONAME|SQL|START(?:ING)?|STATISTICS|STATUS|STRIPED|SYSTEM_USER|TABLES?|TABLESPACE|TEMP(?:ORARY|TABLE)?|TERMINATED|TEXT(?:SIZE)?|THEN|TIME(?:STAMP)?|TINY(?:BLOB|INT|TEXT)|TOP?|TRAN(?:SACTIONS?)?|TRIGGER|TRUNCATE|TSEQUAL|TYPES?|UNBOUNDED|UNCOMMITTED|UNDEFINED|UNION|UNIQUE|UNLOCK|UNPIVOT|UNSIGNED|UPDATE(?:TEXT)?|USAGE|USE|USER|USING|VALUES?|VAR(?:BINARY|CHAR|CHARACTER|YING)|VIEW|WAITFOR|WARNINGS|WHEN|WHERE|WHILE|WITH(?: ROLLUP|IN)?|WORK|WRITE(?:TEXT)?|YEAR)\b/i,
		boolean: /\b(?:FALSE|NULL|TRUE)\b/i,
		number: /\b0x[\da-f]+\b|\b\d+(?:\.\d*)?|\B\.\d+\b/i,
		operator: /[-+*\/=%^~]|&&?|\|\|?|!=?|<(?:=>?|<|>)?|>[>=]?|\b(?:AND|BETWEEN|DIV|ILIKE|IN|IS|LIKE|NOT|OR|REGEXP|RLIKE|SOUNDS LIKE|XOR)\b/i,
		punctuation: /[;[\]()`,.]/
	};
}
var Pg = R((() => {
	Ng.displayName = "sql", Ng.aliases = [];
})), Fg = /* @__PURE__ */ z({ default: () => Ig });
function Ig(e) {
	e.languages.swift = {
		comment: {
			pattern: /(^|[^\\:])(?:\/\/.*|\/\*(?:[^/*]|\/(?!\*)|\*(?!\/)|\/\*(?:[^*]|\*(?!\/))*\*\/)*\*\/)/,
			lookbehind: !0,
			greedy: !0
		},
		"string-literal": [{
			pattern: RegExp("(^|[^\"#])(?:\"(?:\\\\(?:\\((?:[^()]|\\([^()]*\\))*\\)|\\r\\n|[^(])|[^\\\\\\r\\n\"])*\"|\"\"\"(?:\\\\(?:\\((?:[^()]|\\([^()]*\\))*\\)|[^(])|[^\\\\\"]|\"(?!\"\"))*\"\"\")(?![\"#])"),
			lookbehind: !0,
			greedy: !0,
			inside: {
				interpolation: {
					pattern: /(\\\()(?:[^()]|\([^()]*\))*(?=\))/,
					lookbehind: !0,
					inside: null
				},
				"interpolation-punctuation": {
					pattern: /^\)|\\\($/,
					alias: "punctuation"
				},
				punctuation: /\\(?=[\r\n])/,
				string: /[\s\S]+/
			}
		}, {
			pattern: RegExp("(^|[^\"#])(#+)(?:\"(?:\\\\(?:#+\\((?:[^()]|\\([^()]*\\))*\\)|\\r\\n|[^#])|[^\\\\\\r\\n])*?\"|\"\"\"(?:\\\\(?:#+\\((?:[^()]|\\([^()]*\\))*\\)|[^#])|[^\\\\])*?\"\"\")\\2"),
			lookbehind: !0,
			greedy: !0,
			inside: {
				interpolation: {
					pattern: /(\\#+\()(?:[^()]|\([^()]*\))*(?=\))/,
					lookbehind: !0,
					inside: null
				},
				"interpolation-punctuation": {
					pattern: /^\)|\\#+\($/,
					alias: "punctuation"
				},
				string: /[\s\S]+/
			}
		}],
		directive: {
			pattern: RegExp("#(?:(?:elseif|if)\\b(?:[ 	]*(?:![ \\t]*)?(?:\\b\\w+\\b(?:[ \\t]*\\((?:[^()]|\\([^()]*\\))*\\))?|\\((?:[^()]|\\([^()]*\\))*\\))(?:[ \\t]*(?:&&|\\|\\|))?)+|(?:else|endif)\\b)"),
			alias: "property",
			inside: {
				"directive-name": /^#\w+/,
				boolean: /\b(?:false|true)\b/,
				number: /\b\d+(?:\.\d+)*\b/,
				operator: /!|&&|\|\||[<>]=?/,
				punctuation: /[(),]/
			}
		},
		literal: {
			pattern: /#(?:colorLiteral|column|dsohandle|file(?:ID|Literal|Path)?|function|imageLiteral|line)\b/,
			alias: "constant"
		},
		"other-directive": {
			pattern: /#\w+\b/,
			alias: "property"
		},
		attribute: {
			pattern: /@\w+/,
			alias: "atrule"
		},
		"function-definition": {
			pattern: /(\bfunc\s+)\w+/,
			lookbehind: !0,
			alias: "function"
		},
		label: {
			pattern: /\b(break|continue)\s+\w+|\b[a-zA-Z_]\w*(?=\s*:\s*(?:for|repeat|while)\b)/,
			lookbehind: !0,
			alias: "important"
		},
		keyword: /\b(?:Any|Protocol|Self|Type|actor|as|assignment|associatedtype|associativity|async|await|break|case|catch|class|continue|convenience|default|defer|deinit|didSet|do|dynamic|else|enum|extension|fallthrough|fileprivate|final|for|func|get|guard|higherThan|if|import|in|indirect|infix|init|inout|internal|is|isolated|lazy|left|let|lowerThan|mutating|none|nonisolated|nonmutating|open|operator|optional|override|postfix|precedencegroup|prefix|private|protocol|public|repeat|required|rethrows|return|right|safe|self|set|some|static|struct|subscript|super|switch|throw|throws|try|typealias|unowned|unsafe|var|weak|where|while|willSet)\b/,
		boolean: /\b(?:false|true)\b/,
		nil: {
			pattern: /\bnil\b/,
			alias: "constant"
		},
		"short-argument": /\$\d+\b/,
		omit: {
			pattern: /\b_\b/,
			alias: "keyword"
		},
		number: /\b(?:[\d_]+(?:\.[\de_]+)?|0x[a-f0-9_]+(?:\.[a-f0-9p_]+)?|0b[01_]+|0o[0-7_]+)\b/i,
		"class-name": /\b[A-Z](?:[A-Z_\d]*[a-z]\w*)?\b/,
		function: /\b[a-z_]\w*(?=\s*\()/i,
		constant: /\b(?:[A-Z_]{2,}|k[A-Z][A-Za-z_]+)\b/,
		operator: /[-+*/%=!<>&|^~?]+|\.[.\-+*/%=!<>&|^~?]+/,
		punctuation: /[{}[\]();,.:\\]/
	}, e.languages.swift["string-literal"].forEach(function(t) {
		t.inside.interpolation.inside = e.languages.swift;
	});
}
var Lg = R((() => {
	Ig.displayName = "swift", Ig.aliases = [];
})), Rg = /* @__PURE__ */ z({ default: () => zg });
function zg(e) {
	e.register(Ah), (function(e) {
		e.languages.typescript = e.languages.extend("javascript", {
			"class-name": {
				pattern: /(\b(?:class|extends|implements|instanceof|interface|new|type)\s+)(?!keyof\b)(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?:\s*<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>)?/,
				lookbehind: !0,
				greedy: !0,
				inside: null
			},
			builtin: /\b(?:Array|Function|Promise|any|boolean|console|never|number|string|symbol|unknown)\b/
		}), e.languages.typescript.keyword.push(/\b(?:abstract|declare|is|keyof|readonly|require)\b/, /\b(?:asserts|infer|interface|module|namespace|type)\b(?=\s*(?:[{_$a-zA-Z\xA0-\uFFFF]|$))/, /\btype\b(?=\s*(?:[\{*]|$))/), delete e.languages.typescript.parameter, delete e.languages.typescript["literal-property"];
		var t = e.languages.extend("typescript", {});
		delete t["class-name"], e.languages.typescript["class-name"].inside = t, e.languages.insertBefore("typescript", "function", {
			decorator: {
				pattern: /@[$\w\xA0-\uFFFF]+/,
				inside: {
					at: {
						pattern: /^@/,
						alias: "operator"
					},
					function: /^[\s\S]+/
				}
			},
			"generic-function": {
				pattern: /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*\s*<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>(?=\s*\()/,
				greedy: !0,
				inside: {
					function: /^#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*/,
					generic: {
						pattern: /<[\s\S]+/,
						alias: "class-name",
						inside: t
					}
				}
			}
		}), e.languages.ts = e.languages.typescript;
	})(e);
}
var Bg = R((() => {
	jh(), zg.displayName = "typescript", zg.aliases = ["ts"];
})), Vg = /* @__PURE__ */ z({ default: () => Hg });
function Hg(e) {
	e.register(lh), e.languages.vbnet = e.languages.extend("basic", {
		comment: [{
			pattern: /(?:!|REM\b).+/i,
			inside: { keyword: /^REM/i }
		}, {
			pattern: /(^|[^\\:])'.*/,
			lookbehind: !0,
			greedy: !0
		}],
		string: {
			pattern: /(^|[^"])"(?:""|[^"])*"(?!")/,
			lookbehind: !0,
			greedy: !0
		},
		keyword: /(?:\b(?:ADDHANDLER|ADDRESSOF|ALIAS|AND|ANDALSO|AS|BEEP|BLOAD|BOOLEAN|BSAVE|BYREF|BYTE|BYVAL|CALL(?: ABSOLUTE)?|CASE|CATCH|CBOOL|CBYTE|CCHAR|CDATE|CDBL|CDEC|CHAIN|CHAR|CHDIR|CINT|CLASS|CLEAR|CLNG|CLOSE|CLS|COBJ|COM|COMMON|CONST|CONTINUE|CSBYTE|CSHORT|CSNG|CSTR|CTYPE|CUINT|CULNG|CUSHORT|DATA|DATE|DECIMAL|DECLARE|DEF(?: FN| SEG|DBL|INT|LNG|SNG|STR)|DEFAULT|DELEGATE|DIM|DIRECTCAST|DO|DOUBLE|ELSE|ELSEIF|END|ENUM|ENVIRON|ERASE|ERROR|EVENT|EXIT|FALSE|FIELD|FILES|FINALLY|FOR(?: EACH)?|FRIEND|FUNCTION|GET|GETTYPE|GETXMLNAMESPACE|GLOBAL|GOSUB|GOTO|HANDLES|IF|IMPLEMENTS|IMPORTS|IN|INHERITS|INPUT|INTEGER|INTERFACE|IOCTL|IS|ISNOT|KEY|KILL|LET|LIB|LIKE|LINE INPUT|LOCATE|LOCK|LONG|LOOP|LSET|ME|MKDIR|MOD|MODULE|MUSTINHERIT|MUSTOVERRIDE|MYBASE|MYCLASS|NAME|NAMESPACE|NARROWING|NEW|NEXT|NOT|NOTHING|NOTINHERITABLE|NOTOVERRIDABLE|OBJECT|OF|OFF|ON(?: COM| ERROR| KEY| TIMER)?|OPEN|OPERATOR|OPTION(?: BASE)?|OPTIONAL|OR|ORELSE|OUT|OVERLOADS|OVERRIDABLE|OVERRIDES|PARAMARRAY|PARTIAL|POKE|PRIVATE|PROPERTY|PROTECTED|PUBLIC|PUT|RAISEEVENT|READ|READONLY|REDIM|REM|REMOVEHANDLER|RESTORE|RESUME|RETURN|RMDIR|RSET|RUN|SBYTE|SELECT(?: CASE)?|SET|SHADOWS|SHARED|SHELL|SHORT|SINGLE|SLEEP|STATIC|STEP|STOP|STRING|STRUCTURE|SUB|SWAP|SYNCLOCK|SYSTEM|THEN|THROW|TIMER|TO|TROFF|TRON|TRUE|TRY|TRYCAST|TYPE|TYPEOF|UINTEGER|ULONG|UNLOCK|UNTIL|USHORT|USING|VIEW PRINT|WAIT|WEND|WHEN|WHILE|WIDENING|WITH|WITHEVENTS|WRITE|WRITEONLY|XOR)|\B(?:#CONST|#ELSE|#ELSEIF|#END|#IF))(?:\$|\b)/i,
		punctuation: /[,;:(){}]/
	});
}
var Ug = R((() => {
	uh(), Hg.displayName = "vbnet", Hg.aliases = [];
})), Wg = /* @__PURE__ */ z({ default: () => Gg });
function Gg(e) {
	(function(e) {
		var t = /[*&][^\s[\]{},]+/, n = /!(?:<[\w\-%#;/?:@&=+$,.!~*'()[\]]+>|(?:[a-zA-Z\d-]*!)?[\w\-%#;/?:@&=+$.~*'()]+)?/, r = "(?:" + n.source + "(?:[ 	]+" + t.source + ")?|" + t.source + "(?:[ 	]+" + n.source + ")?)", i = "(?:[^\\s\\x00-\\x08\\x0e-\\x1f!\"#%&'*,\\-:>?@[\\]`{|}\\x7f-\\x84\\x86-\\x9f\\ud800-\\udfff\\ufffe\\uffff]|[?:-]<PLAIN>)(?:[ \\t]*(?:(?![#:])<PLAIN>|:<PLAIN>))*".replace(/<PLAIN>/g, function() {
			return "[^\\s\\x00-\\x08\\x0e-\\x1f,[\\]{}\\x7f-\\x84\\x86-\\x9f\\ud800-\\udfff\\ufffe\\uffff]";
		}), a = "\"(?:[^\"\\\\\\r\\n]|\\\\.)*\"|'(?:[^'\\\\\\r\\n]|\\\\.)*'";
		function o(e, t) {
			t = (t || "").replace(/m/g, "") + "m";
			var n = "([:\\-,[{]\\s*(?:\\s<<prop>>[ \\t]+)?)(?:<<value>>)(?=[ \\t]*(?:$|,|\\]|\\}|(?:[\\r\\n]\\s*)?#))".replace(/<<prop>>/g, function() {
				return r;
			}).replace(/<<value>>/g, function() {
				return e;
			});
			return RegExp(n, t);
		}
		e.languages.yaml = {
			scalar: {
				pattern: RegExp("([\\-:]\\s*(?:\\s<<prop>>[ \\t]+)?[|>])[ \\t]*(?:((?:\\r?\\n|\\r)[ \\t]+)\\S[^\\r\\n]*(?:\\2[^\\r\\n]+)*)".replace(/<<prop>>/g, function() {
					return r;
				})),
				lookbehind: !0,
				alias: "string"
			},
			comment: /#.*/,
			key: {
				pattern: RegExp("((?:^|[:\\-,[{\\r\\n?])[ \\t]*(?:<<prop>>[ \\t]+)?)<<key>>(?=\\s*:\\s)".replace(/<<prop>>/g, function() {
					return r;
				}).replace(/<<key>>/g, function() {
					return "(?:" + i + "|" + a + ")";
				})),
				lookbehind: !0,
				greedy: !0,
				alias: "atrule"
			},
			directive: {
				pattern: /(^[ \t]*)%.+/m,
				lookbehind: !0,
				alias: "important"
			},
			datetime: {
				pattern: o("\\d{4}-\\d\\d?-\\d\\d?(?:[tT]|[ \\t]+)\\d\\d?:\\d{2}:\\d{2}(?:\\.\\d*)?(?:[ \\t]*(?:Z|[-+]\\d\\d?(?::\\d{2})?))?|\\d{4}-\\d{2}-\\d{2}|\\d\\d?:\\d{2}(?::\\d{2}(?:\\.\\d*)?)?"),
				lookbehind: !0,
				alias: "number"
			},
			boolean: {
				pattern: o("false|true", "i"),
				lookbehind: !0,
				alias: "important"
			},
			null: {
				pattern: o("null|~", "i"),
				lookbehind: !0,
				alias: "important"
			},
			string: {
				pattern: o(a),
				lookbehind: !0,
				greedy: !0
			},
			number: {
				pattern: o("[+-]?(?:0x[\\da-f]+|0o[0-7]+|(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:e[+-]?\\d+)?|\\.inf|\\.nan)", "i"),
				lookbehind: !0
			},
			tag: n,
			important: t,
			punctuation: /---|[:[\]{}\-,|>?]|\.\.\./
		}, e.languages.yml = e.languages.yaml;
	})(e);
}
var Kg = R((() => {
	Gg.displayName = "yaml", Gg.aliases = ["yml"];
})), qg = {
	arduino: () => Promise.resolve().then(() => (ih(), nh)),
	bash: () => Promise.resolve().then(() => (sh(), ah)),
	basic: () => Promise.resolve().then(() => (uh(), ch)),
	c: () => Promise.resolve().then(() => (Qm(), Xm)),
	clike: () => Promise.resolve().then(() => (Ym(), qm)),
	cpp: () => Promise.resolve().then(() => (th(), $m)),
	csharp: () => Promise.resolve().then(() => (ph(), dh)),
	css: () => Promise.resolve().then(() => (gh(), mh)),
	diff: () => Promise.resolve().then(() => (yh(), _h)),
	go: () => Promise.resolve().then(() => (Sh(), bh)),
	ini: () => Promise.resolve().then(() => (Th(), Ch)),
	java: () => Promise.resolve().then(() => (Oh(), Eh)),
	javascript: () => Promise.resolve().then(() => (jh(), kh)),
	json: () => Promise.resolve().then(() => (Ph(), Mh)),
	kotlin: () => Promise.resolve().then(() => (Lh(), Fh)),
	less: () => Promise.resolve().then(() => (Bh(), Rh)),
	lua: () => Promise.resolve().then(() => (Uh(), Vh)),
	makefile: () => Promise.resolve().then(() => (Kh(), Wh)),
	markdown: () => Promise.resolve().then(() => (Qh(), Xh)),
	markup: () => Promise.resolve().then(() => (Yh(), qh)),
	"markup-templating": () => Promise.resolve().then(() => (tg(), $h)),
	objectivec: () => Promise.resolve().then(() => (ig(), ng)),
	perl: () => Promise.resolve().then(() => (sg(), ag)),
	php: () => Promise.resolve().then(() => (ug(), cg)),
	python: () => Promise.resolve().then(() => (pg(), dg)),
	r: () => Promise.resolve().then(() => (gg(), mg)),
	regex: () => Promise.resolve().then(() => (yg(), _g)),
	ruby: () => Promise.resolve().then(() => (Sg(), bg)),
	rust: () => Promise.resolve().then(() => (Tg(), Cg)),
	sass: () => Promise.resolve().then(() => (Og(), Eg)),
	scss: () => Promise.resolve().then(() => (jg(), kg)),
	sql: () => Promise.resolve().then(() => (Pg(), Mg)),
	swift: () => Promise.resolve().then(() => (Lg(), Fg)),
	typescript: () => Promise.resolve().then(() => (Bg(), Rg)),
	vbnet: () => Promise.resolve().then(() => (Ug(), Vg)),
	yaml: () => Promise.resolve().then(() => (Kg(), Wg))
}, Jg = {
	html: "markup",
	xml: "markup",
	svg: "markup",
	mathml: "markup",
	ssml: "markup",
	atom: "markup",
	rss: "markup",
	js: "javascript",
	jsx: "javascript",
	node: "javascript",
	ts: "typescript",
	tsx: "typescript",
	py: "python",
	rb: "ruby",
	yml: "yaml",
	md: "markdown",
	sh: "bash",
	shell: "bash",
	zsh: "bash",
	"c++": "cpp",
	objc: "objectivec",
	cs: "csharp",
	dotnet: "csharp",
	kt: "kotlin",
	rs: "rust",
	golang: "go"
}, Yg = /* @__PURE__ */ new Set(), Xg = /* @__PURE__ */ new Map(), Zg = (e) => {
	let t = e.toLowerCase();
	return Jg[t] ?? t;
}, Qg = async (e) => {
	let t = Zg(e);
	if (Fm.registered(t)) return t;
	if (Yg.has(t)) return null;
	let n = qg[t];
	if (!n) return Yg.add(t), null;
	let r = Xg.get(t);
	return r || (r = n().then((e) => {
		Fm.register(e.default);
	}).catch(() => {
		Yg.add(t);
	}).finally(() => {
		Xg.delete(t);
	}), Xg.set(t, r)), await r, Fm.registered(t) ? t : null;
}, $g = (e, t, n, r, i) => {
	let a = r;
	for (let r of e) if (r.type === "text") {
		let e = r.value ?? "";
		e.length > 0 && (i.push({
			start: a,
			end: a + e.length,
			color: n,
			text: e
		}), a += e.length);
	} else if (r.children) {
		let e = r.properties?.className, o = Array.isArray(e) ? Km(t, e) : n;
		a = $g(r.children, t, o, a, i);
	}
	return a;
}, e_ = (e, t) => {
	let n = Array.from({ length: t }, () => []), r = 0, i = 0;
	for (let t of e) {
		let e = t.start;
		for (let a = 0; a < t.text.length; a++) if (t.text[a] === "\n") {
			let o = t.start + a;
			o > e && n[r] && n[r].push({
				start: e - i,
				end: o - i,
				color: t.color
			}), r += 1, i = t.start + a + 1, e = i;
		}
		t.end > e && n[r] && n[r].push({
			start: e - i,
			end: t.end - i,
			color: t.color
		});
	}
	return n;
}, t_ = (e, t, n) => {
	if (e.length === 0) return [];
	let r;
	try {
		r = Fm.highlight(e, t);
	} catch {
		return null;
	}
	let i = [];
	$g(r.children ?? [], n, n.default, 0, i);
	let a = e.split("\n").length;
	return e_(i, a);
}, n_ = (e, t) => {
	if (t <= 0) return e;
	let n = [];
	for (let r of e) r.end <= t || n.push({
		start: Math.max(r.start, t) - t,
		end: r.end - t,
		color: r.color
	});
	return n;
}, r_ = (e, t, n, { styles: r, showHighlight: i }, a) => {
	let o = t ? { color: t } : void 0;
	return n === hd.ADDED ? p("ins", {
		className: (0, pc.default)(r.wordDiff, { [r.wordAdded]: i }),
		style: o,
		children: e
	}, a) : n === hd.REMOVED ? p("del", {
		className: (0, pc.default)(r.wordDiff, { [r.wordRemoved]: i }),
		style: o,
		children: e
	}, a) : p("span", {
		className: r.wordDiff,
		style: o,
		children: e
	}, a);
}, i_ = (e, t, n, r) => {
	let i = [], a = 0;
	for (let e of n) {
		let t = typeof e.value == "string" ? e.value : "";
		t.length > 0 && (i.push({
			start: a,
			end: a + t.length,
			type: e.type ?? hd.DEFAULT
		}), a += t.length);
	}
	let o = e.length, s = [], c = 0, l = 0, u = 0, d = 0;
	for (; u < o;) {
		let n = t[c], a = i[l], f = n ? n.end : o, p = a ? a.end : o, m = Math.min(f, p, o);
		if (m <= u) {
			if (f <= u) c += 1;
			else if (p <= u) l += 1;
			else break;
			continue;
		}
		s.push(r_(e.slice(u, m), n?.color, a ? a.type : hd.DEFAULT, r, d++)), u = m, f <= u && (c += 1), p <= u && (l += 1);
	}
	return s;
}, a_ = (e, t) => t.length === 0 ? [p("span", { children: e }, 0)] : t.map((t, n) => p("span", {
	style: { color: t.color },
	children: e.slice(t.start, t.end)
}, n)), o_ = /^[ \t]+/;
function s_(e) {
	if (typeof e == "string") {
		let t = e.match(o_);
		return t ? {
			indent: t[0],
			rest: e.slice(t[0].length)
		} : {
			indent: "",
			rest: e
		};
	}
	if (Array.isArray(e)) {
		let t = e.map((e) => ({ ...e })), n = "", r = 0;
		for (; r < t.length; r++) {
			let e = typeof t[r].value == "string" ? t[r].value : null;
			if (e === null) break;
			let i = e.match(o_);
			if (!i) break;
			if (i[0].length === e.length) {
				n += e;
				continue;
			}
			n += i[0], t[r] = {
				...t[r],
				value: e.slice(i[0].length)
			};
			break;
		}
		return n ? {
			indent: n,
			rest: t.slice(r)
		} : {
			indent: "",
			rest: e
		};
	}
	return {
		indent: "",
		rest: e
	};
}
function c_(e, t, n) {
	let r = [], i = 0;
	for (let e of t) {
		let t = typeof e.value == "string" ? e.value : "";
		t.length > 0 && (r.push({
			start: i,
			end: i + t.length,
			type: e.type ?? hd.DEFAULT
		}), i += t.length);
	}
	let a = [], o = 0;
	for (; o < e.length;) if (e[o] === "<") {
		let t = e.indexOf(">", o);
		if (t === -1) {
			a.push({
				type: "text",
				content: e.slice(o)
			});
			break;
		}
		a.push({
			type: "tag",
			content: e.slice(o, t + 1)
		}), o = t + 1;
	} else {
		let t = e.indexOf("<", o);
		t === -1 && (t = e.length), a.push({
			type: "text",
			content: e.slice(o, t)
		}), o = t;
	}
	function s(e) {
		return e.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").replace(/&quot;/g, "\"").replace(/&#39;/g, "'").replace(/&#x27;/g, "'").replace(/&nbsp;/g, "\xA0");
	}
	function c(e) {
		return e === hd.ADDED ? {
			open: `<ins class="${n.wordDiff} ${n.wordAdded}">`,
			close: "</ins>"
		} : e === hd.REMOVED ? {
			open: `<del class="${n.wordDiff} ${n.wordRemoved}">`,
			close: "</del>"
		} : {
			open: `<span class="${n.wordDiff}">`,
			close: "</span>"
		};
	}
	let l = 0, u = "";
	for (let e of a) if (e.type === "tag") u += e.content;
	else {
		let t = e.content, n = s(t), i = 0, a = 0;
		for (; i < n.length;) {
			let e = l + i, o = r.find((t) => e >= t.start && e < t.end);
			if (!o) {
				let e = t[a];
				u += e, a++, i++;
				continue;
			}
			let s = o.end - e, d = n.length - i, f = Math.min(s, d), p = a, m = 0;
			for (; m < f && p < t.length;) {
				if (t[p] === "&") {
					let e = t.indexOf(";", p);
					e !== -1 && e - p < 10 ? p = e + 1 : p++;
				} else p++;
				m++;
			}
			let h = t.slice(a, p), g = c(o.type);
			u += g ? g.open + h + g.close : h, a = p, i += f;
		}
		l += n.length;
	}
	return u;
}
var l_;
(function(e) {
	e.LEFT = "L", e.RIGHT = "R";
})(l_ ||= {});
var u_ = class t extends e.Component {
	styles;
	wordDiffCache = /* @__PURE__ */ new Map();
	highlightPrecedenceWarned = !1;
	contentColumnRef = e.createRef();
	charMeasureRef = e.createRef();
	stickyHeaderRef = e.createRef();
	resizeObserver = null;
	scrollDebounceTimer = null;
	lastRenderedRange = {
		start: 0,
		end: Infinity
	};
	static defaultProps = {
		oldValue: "",
		newValue: "",
		splitView: !0,
		highlightLines: [],
		disableWordDiff: !1,
		compareMethod: Ed.CHARS,
		styles: {},
		hideLineNumbers: !1,
		extraLinesSurroundingDiff: 3,
		showDiffOnly: !0,
		useDarkTheme: !1,
		linesOffset: 0,
		nonce: ""
	};
	constructor(t) {
		super(t), this.state = {
			expandedBlocks: [],
			noSelect: void 0,
			scrollableContainerRef: e.createRef(),
			computedDiffResult: {},
			isLoading: !1,
			visibleStartRow: 0,
			contentColumnWidth: null,
			charWidth: null,
			cumulativeOffsets: null,
			isScrolling: !1,
			highlightResult: null
		};
	}
	getWordDiffValues = (e, t, n) => {
		if (!e || !t) return {
			leftValue: e?.value,
			rightValue: t?.value
		};
		if (e.rawValue === void 0 || t.rawValue === void 0) return {
			leftValue: e.value,
			rightValue: t.value
		};
		let r = `${n}-${e.rawValue}-${t.rawValue}`, i = this.wordDiffCache.get(r);
		if (!i) {
			let n = this.props.compareMethod === Ed.JSON || this.props.compareMethod === Ed.YAML ? Ed.CHARS : this.props.compareMethod, a = Od(e.rawValue, t.rawValue, n);
			i = {
				left: a.left ?? [],
				right: a.right ?? []
			}, this.wordDiffCache.set(r, i);
		}
		return {
			leftValue: i.left,
			rightValue: i.right
		};
	};
	resetCodeBlocks = () => this.state.expandedBlocks.length > 0 && (this.setState({ expandedBlocks: [] }), !0);
	onBlockExpand = (e) => {
		let t = this.state.expandedBlocks.slice();
		t.push(e), this.setState({ expandedBlocks: t }, () => this.recalculateOffsets());
	};
	getStickyHeaderHeight() {
		return this.stickyHeaderRef.current?.offsetHeight || 0;
	}
	measureCharWidth() {
		let e = this.charMeasureRef.current;
		return e && e.getBoundingClientRect().width || 7.2;
	}
	measureContentColumnWidth() {
		let e = this.contentColumnRef.current;
		if (e && e.clientWidth > 0) {
			let t = window.getComputedStyle(e), n = parseFloat(t.paddingLeft) + parseFloat(t.paddingRight), r = e.clientWidth - n;
			if (r > 0) return r;
		}
		let t = this.state.scrollableContainerRef.current;
		if (!t || t.clientWidth <= 0) return null;
		let n = t.clientWidth, r = this.props.hideLineNumbers ? 0 : 50;
		this.props.splitView;
		let i = this.props.splitView ? 2 : 1, a = this.props.splitView ? 2 : 1, o = n - (2 * r + i * 28);
		return Math.max(100, o / a);
	}
	getTextLength(e) {
		return e ? typeof e == "string" ? e.length : e.reduce((e, t) => e + (typeof t.value == "string" ? t.value.length : 0), 0) : 0;
	}
	buildCumulativeOffsets(e, n, r, i, a, o, s, c) {
		let l = [0], u = /* @__PURE__ */ new Set();
		for (let d = 0; d < e.length; d++) {
			let f = e[d];
			if (a) {
				let e = n[d];
				if (e !== void 0 && !i.includes(e)) {
					let n = r[e].endLine === d;
					!u.has(e) && n && (u.add(e), l.push(l[l.length - 1] + t.ESTIMATED_ROW_HEIGHT));
					continue;
				}
			}
			let p = f.left?.value ? this.getTextLength(f.left.value) : 0, m = f.right?.value ? this.getTextLength(f.right.value) : 0, h = c ? Math.max(p, m) : p || m, g = Math.floor(s / o), _ = g > 0 ? Math.max(1, Math.ceil(h / g)) : 1;
			l.push(l[l.length - 1] + _ * t.ESTIMATED_ROW_HEIGHT);
		}
		return l;
	}
	findLineAtOffset(e, t) {
		let n = 0, r = t.length - 2;
		for (; n < r;) {
			let i = Math.floor((n + r + 1) / 2);
			t[i] <= e ? n = i : r = i - 1;
		}
		return n;
	}
	recalculateOffsets = () => {
		if (!this.props.infiniteLoading) return;
		let e = this.measureContentColumnWidth(), t = this.measureCharWidth();
		if (!e) return;
		let n = this.getMemoisedKey(), { lineInformation: r, lineBlocks: i, blocks: a } = this.state.computedDiffResult[n] ?? {};
		if (!r) return;
		let o = this.buildCumulativeOffsets(r, i, a, this.state.expandedBlocks, this.props.showDiffOnly ?? !0, t, e, this.props.splitView ?? !0);
		this.setState({
			cumulativeOffsets: o,
			contentColumnWidth: e,
			charWidth: t
		}, () => {
			this.onScroll();
		});
	};
	computeStyles = _c(vp);
	onLineNumberClickProxy = (e) => this.props.onLineNumberClick ? (t) => this.props.onLineNumberClick(e, t) : () => {};
	shouldHighlightWordDiff = () => {
		let { compareMethod: e } = this.props;
		return e === Ed.CHARS || e === Ed.WORDS || e === Ed.WORDS_WITH_SPACE || e === Ed.JSON || e === Ed.YAML;
	};
	renderWordDiff = (t, n) => {
		let r = this.shouldHighlightWordDiff(), i = t.map((e) => typeof e.value == "string" ? e.value : "").join("");
		if (i.length > 500) return [p("span", { children: i }, "long-line")];
		if (n) {
			let a = n(i), o = a?.props?.dangerouslySetInnerHTML?.__html;
			if (typeof o == "string") {
				let n = c_(o, t, {
					wordDiff: this.styles.wordDiff,
					wordAdded: r ? this.styles.wordAdded : "",
					wordRemoved: r ? this.styles.wordRemoved : ""
				});
				return [e.cloneElement(a, {
					key: "highlighted-diff",
					dangerouslySetInnerHTML: { __html: n }
				})];
			}
		}
		return t.map((e, t) => {
			let n;
			return n = typeof e.value == "string" ? e.value : void 0, e.type === hd.ADDED ? p("ins", {
				className: (0, pc.default)(this.styles.wordDiff, { [this.styles.wordAdded]: r }),
				children: n
			}, t) : e.type === hd.REMOVED ? p("del", {
				className: (0, pc.default)(this.styles.wordDiff, { [this.styles.wordRemoved]: r }),
				children: n
			}, t) : p("span", {
				className: (0, pc.default)(this.styles.wordDiff),
				children: n
			}, t);
		});
	};
	renderLine = (e, t, n, r, i, a) => {
		let o = `${n}-${e}`, s = `${a}-${i}`, c = this.props.highlightLines ?? [], l = c.includes(o) || c.includes(s), u = t === hd.ADDED, d = t === hd.REMOVED, h = t === hd.CHANGED, { indent: g, rest: _ } = s_(r), v = Array.isArray(_), y = this.state.highlightResult ? n === l_.LEFT ? this.state.highlightResult.left : this.state.highlightResult.right : null, b = e ?? i ?? void 0, x = y && b != null ? y.get(b) : void 0, S = x ? n_(x, g.length) : void 0, C;
		if (S) {
			if (v) {
				let e = _.map((e) => typeof e.value == "string" ? e.value : "").join("");
				C = e.length > 500 ? a_(e, S) : i_(e, S, _, {
					styles: {
						wordDiff: this.styles.wordDiff,
						wordAdded: this.styles.wordAdded,
						wordRemoved: this.styles.wordRemoved
					},
					showHighlight: this.shouldHighlightWordDiff()
				});
			} else C = typeof _ == "string" ? a_(_, S) : _;
		} else C = v ? this.renderWordDiff(_, this.props.renderContent) : this.props.renderContent && typeof _ == "string" ? this.props.renderContent(_) : _;
		let w = "div";
		u && !v ? w = "ins" : d && !v && (w = "del");
		let T = !C && !g;
		return m(f, { children: [
			!this.props.hideLineNumbers && p("td", {
				onClick: e && this.onLineNumberClickProxy(o),
				className: (0, pc.default)(this.styles.gutter, {
					[this.styles.emptyGutter]: !e,
					[this.styles.diffAdded]: u,
					[this.styles.diffRemoved]: d,
					[this.styles.diffChanged]: h,
					[this.styles.highlightedGutter]: l
				}),
				children: p("pre", {
					className: this.styles.lineNumber,
					children: e
				})
			}),
			!this.props.splitView && !this.props.hideLineNumbers && p("td", {
				onClick: i && this.onLineNumberClickProxy(s),
				className: (0, pc.default)(this.styles.gutter, {
					[this.styles.emptyGutter]: !i,
					[this.styles.diffAdded]: u,
					[this.styles.diffRemoved]: d,
					[this.styles.diffChanged]: h,
					[this.styles.highlightedGutter]: l
				}),
				children: p("pre", {
					className: this.styles.lineNumber,
					children: i
				})
			}),
			this.props.renderGutter ? this.props.renderGutter({
				lineNumber: e ?? 0,
				type: t ?? hd.DEFAULT,
				prefix: n,
				value: r ?? "",
				additionalLineNumber: i ?? void 0,
				additionalPrefix: a,
				styles: this.styles
			}) : null,
			p("td", {
				className: (0, pc.default)(this.styles.marker, {
					[this.styles.emptyLine]: T,
					[this.styles.diffAdded]: u,
					[this.styles.diffRemoved]: d,
					[this.styles.diffChanged]: h,
					[this.styles.highlightedLine]: l
				}),
				children: m("pre", { children: [u && "+", d && "-"] })
			}),
			p("td", {
				ref: n === l_.LEFT && !this.state.cumulativeOffsets ? this.contentColumnRef : void 0,
				className: (0, pc.default)(this.styles.content, {
					[this.styles.emptyLine]: T,
					[this.styles.diffAdded]: u,
					[this.styles.diffRemoved]: d,
					[this.styles.diffChanged]: h,
					[this.styles.highlightedLine]: l,
					left: n === l_.LEFT,
					right: n === l_.RIGHT
				}),
				onMouseDown: () => {
					let e = document.getElementsByClassName("right");
					for (let t = 0; t < e.length; t++) e[t]?.classList.remove(this.styles.noSelect);
					let t = document.getElementsByClassName("left");
					for (let e = 0; e < t.length; e++) t[e]?.classList.remove(this.styles.noSelect);
					let r = document.getElementsByClassName(n === l_.LEFT ? "right" : "left");
					for (let e = 0; e < r.length; e++) r[e]?.classList.add(this.styles.noSelect);
				},
				title: u && !v ? "Added line" : d && !v ? "Removed line" : void 0,
				children: m(w, {
					className: (0, pc.default)(this.styles.contentText, this.styles.contentFlex),
					children: [g ? p("span", {
						className: this.styles.lineIndent,
						children: g
					}) : null, p("span", {
						className: this.styles.lineBody,
						children: C
					})]
				})
			})
		] });
	};
	renderSplitView = ({ left: e, right: t }, n) => {
		let { leftValue: r, rightValue: i } = this.getWordDiffValues(e, t, n);
		return m("tr", {
			className: this.styles.line,
			children: [this.renderLine(e.lineNumber, e.type, l_.LEFT, r), this.renderLine(t.lineNumber, t.type, l_.RIGHT, i)]
		}, n);
	};
	renderInlineView = ({ left: t, right: n }, r) => {
		let { leftValue: i, rightValue: a } = this.getWordDiffValues(t, n, r), o;
		return t.type === hd.REMOVED && n.type === hd.ADDED ? m(e.Fragment, { children: [p("tr", {
			className: this.styles.line,
			children: this.renderLine(t.lineNumber, t.type, l_.LEFT, i, null)
		}), p("tr", {
			className: this.styles.line,
			children: this.renderLine(null, n.type, l_.RIGHT, a, n.lineNumber, l_.RIGHT)
		})] }, r) : (t.type === hd.REMOVED && (o = this.renderLine(t.lineNumber, t.type, l_.LEFT, i, null)), t.type === hd.DEFAULT && (o = this.renderLine(t.lineNumber, t.type, l_.LEFT, i, n.lineNumber, l_.RIGHT)), n.type === hd.ADDED && (o = this.renderLine(null, n.type, l_.RIGHT, a, n.lineNumber)), p("tr", {
			className: this.styles.line,
			children: o
		}, r));
	};
	onBlockClickProxy = (e) => () => this.onBlockExpand(e);
	renderSkippedLineIndicator = (t, n, r, i) => {
		let { hideLineNumbers: a, splitView: o } = this.props, s = this.props.codeFoldMessageRenderer ? this.props.codeFoldMessageRenderer(t, r, i) : m("span", {
			className: this.styles.codeFoldContent,
			children: [
				"@@ -",
				r - t,
				",",
				t,
				" +",
				i - t,
				",",
				t,
				" @@"
			]
		}), c = p("td", {
			className: this.styles.codeFoldContentContainer,
			children: p("button", {
				type: "button",
				className: this.styles.codeFoldExpandButton,
				onClick: this.onBlockClickProxy(n),
				tabIndex: 0,
				children: s
			})
		}), l = !o && !a, u = p("td", {
			className: this.styles.codeFoldGutter,
			children: p(Pd, {})
		});
		return m("tr", {
			className: this.styles.codeFold,
			onClick: this.onBlockClickProxy(n),
			role: "button",
			tabIndex: 0,
			children: [
				!a && u,
				this.props.renderGutter ? p("td", { className: this.styles.codeFoldGutter }) : null,
				p("td", { className: (0, pc.default)({ [this.styles.codeFoldGutter]: l }) }),
				l ? m(e.Fragment, { children: [p("td", {}), c] }) : m(e.Fragment, { children: [
					c,
					this.props.renderGutter ? p("td", {}) : null,
					p("td", {}),
					p("td", {}),
					a ? null : p("td", {})
				] })
			]
		}, `${r}-${i}`);
	};
	getMemoisedKey = () => {
		let { oldValue: e, newValue: t, disableWordDiff: n, compareMethod: r, linesOffset: i, alwaysShowLines: a, extraLinesSurroundingDiff: o } = this.props;
		return JSON.stringify({
			oldValue: e,
			newValue: t,
			disableWordDiff: n,
			compareMethod: r,
			linesOffset: i,
			alwaysShowLines: a,
			extraLinesSurroundingDiff: o
		});
	};
	memoisedCompute = async () => {
		let { oldValue: e, newValue: t, disableWordDiff: n, compareMethod: r, linesOffset: i } = this.props, a = this.getMemoisedKey();
		if (this.state.computedDiffResult[a]) {
			this.setState((e) => ({
				...e,
				isLoading: !1
			})), this.updateHighlight();
			return;
		}
		let o = this.props.infiniteLoading?.containerHeight, s = o ? typeof o == "number" ? o : parseInt(o, 10) || 0 : 0, c = !n && !!this.props.infiniteLoading && s > 0 && s < 2e3, { lineInformation: l, diffLines: u } = await Nd(e, t, n, r, i, this.props.alwaysShowLines, c, this.props.disableWorker), d = this.props.extraLinesSurroundingDiff ?? 3, { lineBlocks: f, blocks: p } = vc(l, u, d < 0 ? 0 : Math.round(d));
		this.state.computedDiffResult[a] = {
			lineInformation: l,
			lineBlocks: f,
			blocks: p
		}, this.setState((e) => ({
			...e,
			computedDiffResult: this.state.computedDiffResult,
			isLoading: !1
		}), () => {
			this.updateHighlight(), this.props.infiniteLoading && requestAnimationFrame(() => this.recalculateOffsets());
		});
	};
	lineToText = (e) => typeof e.rawValue == "string" ? e.rawValue : typeof e.value == "string" ? e.value : Array.isArray(e.value) ? e.value.map((e) => typeof e.value == "string" ? e.value : "").join("") : "";
	buildSideTokens = (e, t, n, r) => {
		let i = [];
		for (let n of e) {
			let e = n[t];
			!e || e.lineNumber == null || i.push({
				lineNumber: e.lineNumber,
				text: this.lineToText(e)
			});
		}
		i.sort((e, t) => e.lineNumber - t.lineNumber);
		let a = /* @__PURE__ */ new Map();
		if (i.length === 0) return a;
		let o = t_(i.map((e) => e.text).join("\n"), n, r);
		if (!o) return a;
		for (let e = 0; e < i.length; e++) a.set(i[e].lineNumber, o[e] ?? []);
		return a;
	};
	resolveHighlightTheme = () => {
		let e = this.props.useDarkTheme ? Gm : Wm;
		return this.props.highlightTheme ? {
			...e,
			...this.props.highlightTheme
		} : e;
	};
	updateHighlight = async () => {
		let { highlightLanguage: e } = this.props;
		if (!e) {
			this.state.highlightResult && this.setState({ highlightResult: null });
			return;
		}
		let t = this.getMemoisedKey(), n = this.state.computedDiffResult[t];
		if (!n) return;
		let r = `${t}::${e}::${this.props.useDarkTheme ?? !1}::${JSON.stringify(this.props.highlightTheme ?? null)}`;
		if (this.state.highlightResult?.key === r) return;
		let i = await Qg(e);
		if (this.props.highlightLanguage !== e || this.getMemoisedKey() !== t) return;
		if (!i) {
			this.state.highlightResult && this.setState({ highlightResult: null });
			return;
		}
		!this.highlightPrecedenceWarned && this.props.renderContent && typeof process < "u" && process.env.NODE_ENV !== "production" && (this.highlightPrecedenceWarned = !0, console.warn("[react-diff-viewer] `highlightLanguage` takes precedence over `renderContent`; `renderContent` is ignored for line content while highlighting is active."));
		let a = this.resolveHighlightTheme(), o = this.buildSideTokens(n.lineInformation, "left", i, a), s = this.buildSideTokens(n.lineInformation, "right", i, a);
		this.setState({ highlightResult: {
			key: r,
			left: o,
			right: s
		} });
	};
	static ESTIMATED_ROW_HEIGHT = 19;
	onScroll = () => {
		let e = this.state.scrollableContainerRef.current;
		if (!e || !this.props.infiniteLoading) return;
		let n = this.getStickyHeaderHeight(), r = Math.max(0, e.scrollTop - n), { cumulativeOffsets: i } = this.state, a = i ? this.findLineAtOffset(r, i) : Math.floor(r / t.ESTIMATED_ROW_HEIGHT), o = a + Math.ceil(e.clientHeight / t.ESTIMATED_ROW_HEIGHT), s = a < this.lastRenderedRange.start || o > this.lastRenderedRange.end;
		this.scrollDebounceTimer && clearTimeout(this.scrollDebounceTimer);
		let c = {};
		a !== this.state.visibleStartRow && (c.visibleStartRow = a), s && !this.state.isScrolling && (c.isScrolling = !0), Object.keys(c).length > 0 && this.setState(c), (this.state.isScrolling || s) && (this.scrollDebounceTimer = setTimeout(() => {
			this.setState({ isScrolling: !1 });
		}, 150));
	};
	renderDiff = () => {
		let { splitView: n, infiniteLoading: r, showDiffOnly: i } = this.props, { computedDiffResult: a, expandedBlocks: o, visibleStartRow: s, scrollableContainerRef: c, cumulativeOffsets: l } = this.state, { lineInformation: u = [], lineBlocks: d = [], blocks: f = [] } = a[this.getMemoisedKey()] ?? {}, m = 0, h = Infinity, g = r?.overscan ?? 20;
		if (r && c.current) {
			let e = c.current, n = this.getStickyHeaderHeight(), r = Math.max(0, e.scrollTop - n);
			if (l) {
				let n = l[l.length - 1] || 0, i = l.length - 2;
				m = Math.max(0, this.findLineAtOffset(r, l) - g), h = this.findLineAtOffset(r + e.clientHeight, l) + g;
				let a = Math.ceil(e.clientHeight / t.ESTIMATED_ROW_HEIGHT);
				h = Math.max(h, m + a + g), r + e.clientHeight >= n - g * t.ESTIMATED_ROW_HEIGHT && (h = i + g);
			} else {
				let n = Math.ceil(e.clientHeight / t.ESTIMATED_ROW_HEIGHT);
				m = Math.max(0, s - g), h = s + n + g;
			}
		}
		let _ = /* @__PURE__ */ new Map(), v = /* @__PURE__ */ new Set(), y = 0;
		for (let e = 0; e < u.length; e++) {
			let t = d[e];
			if (i && t !== void 0) {
				if (o.includes(t)) _.set(e, y), y++;
				else {
					let n = f[t].endLine === e;
					!v.has(t) && n && (v.add(t), _.set(e, y), y++);
				}
			} else _.set(e, y), y++;
		}
		let b = y, x = [], S = 0, C = !1, w = -1;
		v.clear();
		for (let r = 0; r < u.length; r++) {
			let a = u[r], s = _.get(r);
			if (s !== void 0 && !(s < m)) {
				if (s > h) break;
				if (C ||= (S = l ? l[s] || 0 : s * t.ESTIMATED_ROW_HEIGHT, !0), w = s, i) {
					let t = d[r];
					if (t !== void 0) {
						let n = f[t].endLine === r;
						if (!o.includes(t) && n) {
							x.push(p(e.Fragment, { children: this.renderSkippedLineIndicator(f[t].lines, t, a.left.lineNumber ?? 0, a.right.lineNumber ?? 0) }, r));
							continue;
						}
						if (!o.includes(t)) continue;
					}
				}
				x.push(n ? this.renderSplitView(a, r) : this.renderInlineView(a, r));
			}
		}
		let T = l ? l[l.length - 1] || 0 : b * t.ESTIMATED_ROW_HEIGHT, E = l && w >= 0 ? T - (l[w + 1] || T) : 0;
		return this.lastRenderedRange = {
			start: m,
			end: h
		}, {
			diffNodes: x,
			blocks: f,
			lineInformation: u,
			totalRenderedRows: b,
			topPadding: S,
			bottomPadding: E,
			totalContentHeight: T,
			renderedCount: x.length,
			debug: {
				visibleRowStart: m,
				visibleRowEnd: h,
				totalRows: b,
				offsetsLength: l?.length ?? 0,
				renderedCount: x.length,
				scrollTop: c.current?.scrollTop ?? 0,
				headerHeight: this.getStickyHeaderHeight(),
				contentScrollTop: c.current ? Math.max(0, c.current.scrollTop - this.getStickyHeaderHeight()) : 0,
				clientHeight: c.current?.clientHeight ?? 0
			}
		};
	};
	componentDidUpdate(e) {
		if (e.oldValue !== this.props.oldValue || e.newValue !== this.props.newValue || e.compareMethod !== this.props.compareMethod || e.disableWordDiff !== this.props.disableWordDiff || e.linesOffset !== this.props.linesOffset) {
			this.wordDiffCache.clear();
			let e = this.state.scrollableContainerRef.current;
			e && (e.scrollTop = 0), this.setState((e) => ({
				...e,
				isLoading: !0,
				visibleStartRow: 0,
				cumulativeOffsets: null
			})), this.memoisedCompute();
		} else (e.highlightLanguage !== this.props.highlightLanguage || e.highlightTheme !== this.props.highlightTheme || e.useDarkTheme !== this.props.useDarkTheme) && this.updateHighlight();
	}
	componentDidMount() {
		if (this.setState((e) => ({
			...e,
			isLoading: !0
		})), this.memoisedCompute(), typeof ResizeObserver < "u" && this.props.infiniteLoading) {
			this.resizeObserver = new ResizeObserver(() => {
				requestAnimationFrame(() => this.recalculateOffsets());
			});
			let e = this.state.scrollableContainerRef.current;
			e && this.resizeObserver.observe(e);
		}
	}
	componentWillUnmount() {
		this.resizeObserver?.disconnect(), this.scrollDebounceTimer && clearTimeout(this.scrollDebounceTimer);
	}
	render = () => {
		let { oldValue: e, newValue: t, useDarkTheme: n, leftTitle: r, rightTitle: i, splitView: a, compareMethod: o, hideLineNumbers: s, nonce: c } = this.props;
		if (typeof o == "string" && o !== Ed.JSON && (typeof e != "string" || typeof t != "string")) throw Error("\"oldValue\" and \"newValue\" should be strings");
		this.styles = this.computeStyles(this.props.styles ?? {}, n ?? !1, c ?? "");
		let l = this.renderDiff(), u = 3, d = 4;
		s && (--u, --d), this.props.renderGutter && (u += 1, d += 1);
		let h = 0, g = 0;
		for (let e of l.lineInformation) e.left.type === hd.ADDED && g++, e.right.type === hd.ADDED && g++, e.left.type === hd.REMOVED && h++, e.right.type === hd.REMOVED && h++;
		let _ = h + g, v = Math.round(g / _ * 100), y = [];
		for (let e = 0; e < 5; e++) v > e * 20 ? y.push(p("span", { className: (0, pc.default)(this.styles.block, this.styles.blockAddition) }, e)) : y.push(p("span", { className: (0, pc.default)(this.styles.block, this.styles.blockDeletion) }, e));
		let b = this.state.expandedBlocks.length === l.blocks.length, x = this.props.loadingElement, S = this.props.infiniteLoading ? {
			overflowY: "scroll",
			overflowX: "hidden",
			height: this.props.infiniteLoading.containerHeight
		} : {}, C = !!this.props.infiniteLoading && !this.state.cumulativeOffsets, w = m("table", {
			className: (0, pc.default)(this.styles.diffContainer, {
				[this.styles.splitView]: a,
				[this.styles.noWrap]: C
			}),
			onMouseUp: () => {},
			children: [m("colgroup", { children: [
				!this.props.hideLineNumbers && p("col", { width: "50px" }),
				!a && !this.props.hideLineNumbers && p("col", { width: "50px" }),
				this.props.renderGutter && p("col", { width: "50px" }),
				p("col", { width: "28px" }),
				p("col", { width: "auto" }),
				a && m(f, { children: [
					!this.props.hideLineNumbers && p("col", { width: "50px" }),
					this.props.renderGutter && p("col", { width: "50px" }),
					p("col", { width: "28px" }),
					p("col", { width: "auto" })
				] })
			] }), p("tbody", { children: l.diffNodes })]
		}), T = (this.state.isLoading || this.state.isScrolling) && !!x;
		return m("div", {
			style: { position: "relative" },
			children: [T && p("div", {
				style: {
					position: "absolute",
					top: 0,
					left: 0,
					right: 0,
					bottom: 0,
					zIndex: 3
				},
				children: p(x, {})
			}), m("div", {
				style: {
					...S,
					position: "relative"
				},
				onScroll: this.onScroll,
				ref: this.state.scrollableContainerRef,
				children: [
					(!this.props.hideSummary || r || i) && m("div", {
						ref: this.stickyHeaderRef,
						className: this.styles.stickyHeader,
						children: [!this.props.hideSummary && m("div", {
							className: this.styles.summary,
							role: "banner",
							children: [
								p("button", {
									type: "button",
									className: this.styles.allExpandButton,
									onClick: () => {
										this.setState({ expandedBlocks: b ? [] : l.blocks.map((e) => e.index) }, () => this.recalculateOffsets());
									},
									children: p(b ? yp : Pd, {})
								}),
								" ",
								_,
								p("div", {
									style: {
										display: "flex",
										gap: "1px"
									},
									children: y
								}),
								this.props.summary ? p("span", { children: this.props.summary }) : null
							]
						}), (r || i) && m("div", {
							className: this.styles.columnHeaders,
							children: [p("div", {
								className: this.styles.titleBlock,
								children: r ? p("pre", {
									className: this.styles.contentText,
									children: r
								}) : null
							}), a && p("div", {
								className: this.styles.titleBlock,
								children: i ? p("pre", {
									className: this.styles.contentText,
									children: i
								}) : null
							})]
						})]
					}),
					this.props.infiniteLoading ? p("div", {
						style: {
							height: l.totalContentHeight,
							position: "relative"
						},
						children: p("div", {
							style: {
								position: "absolute",
								top: l.topPadding,
								left: 0,
								right: 0,
								visibility: this.state.isScrolling ? "hidden" : "visible"
							},
							children: w
						})
					}) : w,
					p("span", {
						ref: this.charMeasureRef,
						style: {
							position: "absolute",
							top: 0,
							left: "-9999px",
							visibility: "hidden",
							whiteSpace: "pre",
							fontFamily: "monospace",
							fontSize: 12
						},
						"aria-hidden": "true",
						children: "M"
					}),
					this.props.infiniteLoading && this.props.showDebugInfo && m("div", {
						style: {
							position: "fixed",
							top: 10,
							right: 10,
							background: "rgba(0,0,0,0.85)",
							color: "#0f0",
							padding: "10px",
							fontFamily: "monospace",
							fontSize: "11px",
							zIndex: 9999,
							borderRadius: "4px",
							maxWidth: "300px",
							lineHeight: 1.4
						},
						children: [
							p("div", {
								style: {
									fontWeight: "bold",
									marginBottom: "5px",
									color: "#fff"
								},
								children: "Debug Info"
							}),
							m("div", { children: ["scrollTop: ", l.debug.scrollTop] }),
							m("div", { children: ["headerHeight: ", l.debug.headerHeight] }),
							m("div", { children: ["contentScrollTop: ", l.debug.contentScrollTop] }),
							m("div", { children: ["clientHeight: ", l.debug.clientHeight] }),
							m("div", {
								style: {
									marginTop: "5px",
									borderTop: "1px solid #444",
									paddingTop: "5px"
								},
								children: [m("div", { children: ["visibleRowStart: ", l.debug.visibleRowStart] }), m("div", { children: ["visibleRowEnd: ", l.debug.visibleRowEnd] })]
							}),
							m("div", {
								style: {
									marginTop: "5px",
									borderTop: "1px solid #444",
									paddingTop: "5px"
								},
								children: [
									m("div", { children: ["totalRows: ", l.debug.totalRows] }),
									m("div", { children: ["offsetsLength: ", l.debug.offsetsLength] }),
									m("div", { children: ["renderedCount: ", l.debug.renderedCount] })
								]
							}),
							m("div", {
								style: {
									marginTop: "5px",
									borderTop: "1px solid #444",
									paddingTop: "5px"
								},
								children: [
									m("div", { children: ["topPadding: ", l.topPadding.toFixed(0)] }),
									m("div", { children: ["bottomPadding: ", l.bottomPadding.toFixed(0)] }),
									m("div", { children: ["totalContentHeight: ", l.totalContentHeight.toFixed(0)] })
								]
							}),
							m("div", {
								style: {
									marginTop: "5px",
									borderTop: "1px solid #444",
									paddingTop: "5px",
									color: "#ff0"
								},
								children: [
									m("div", { children: ["cumulativeOffsets: ", this.state.cumulativeOffsets ? "SET" : "NULL"] }),
									m("div", { children: [
										"columnWidth: ",
										this.state.contentColumnWidth?.toFixed(0) ?? "N/A",
										"px"
									] }),
									m("div", { children: [
										"charWidth: ",
										this.state.charWidth?.toFixed(2) ?? "N/A",
										"px"
									] }),
									m("div", { children: ["charsPerRow: ", this.state.contentColumnWidth && this.state.charWidth ? Math.floor(this.state.contentColumnWidth / this.state.charWidth) : "N/A"] })
								]
							}),
							this.state.cumulativeOffsets && m("div", {
								style: {
									marginTop: "5px",
									borderTop: "1px solid #444",
									paddingTop: "5px",
									color: "#0ff",
									fontSize: "10px"
								},
								children: [
									m("div", { children: [
										"offsets[",
										l.debug.visibleRowEnd,
										"]: ",
										this.state.cumulativeOffsets[l.debug.visibleRowEnd]?.toFixed(0) ?? "N/A"
									] }),
									m("div", { children: [
										"offsets[",
										l.debug.totalRows - 1,
										"]: ",
										this.state.cumulativeOffsets[l.debug.totalRows - 1]?.toFixed(0) ?? "N/A"
									] }),
									m("div", { children: [
										"offsets[",
										l.debug.totalRows,
										"]: ",
										this.state.cumulativeOffsets[l.debug.totalRows]?.toFixed(0) ?? "N/A"
									] }),
									m("div", {
										style: { marginTop: "3px" },
										children: ["viewportEnd: ", (l.debug.contentScrollTop + l.debug.clientHeight).toFixed(0)]
									}),
									m("div", {
										style: {
											marginTop: "3px",
											color: "#f0f"
										},
										children: ["scrollHeight: ", this.state.scrollableContainerRef.current?.scrollHeight ?? "N/A"]
									}),
									m("div", { children: ["maxScrollTop: ", (this.state.scrollableContainerRef.current?.scrollHeight ?? 0) - l.debug.clientHeight] })
								]
							})
						]
					})
				]
			})]
		});
	};
}, d_ = {
	variables: { dark: {
		diffViewerBackground: "#111827",
		diffViewerColor: "#e5e7eb",
		addedBackground: "#052e1690",
		addedColor: "#bbf7d0",
		removedBackground: "#450a0a90",
		removedColor: "#fecaca",
		wordAddedBackground: "#166534",
		wordRemovedBackground: "#991b1b",
		addedGutterBackground: "#065f4640",
		removedGutterBackground: "#7f1d1d40",
		gutterBackground: "#111827",
		gutterColor: "#6b7280",
		codeFoldBackground: "#1f2937",
		codeFoldGutterBackground: "#1f2937",
		emptyLineBackground: "#111827"
	} },
	contentText: {
		fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
		fontSize: "12px"
	},
	gutter: {
		fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
		fontSize: "11px"
	}
}, f_ = ({ before: e, after: t, language: n }) => e === t ? /* @__PURE__ */ p(L, {
	kind: "body/regular/sm",
	className: "text-subtle",
	children: "No changes."
}) : /* @__PURE__ */ p("div", {
	className: "max-h-[480px] overflow-auto rounded-md border border-base",
	children: /* @__PURE__ */ p(u_, {
		oldValue: e,
		newValue: t,
		splitView: !0,
		useDarkTheme: !0,
		compareMethod: Ed.WORDS,
		leftTitle: `Before (${n})`,
		rightTitle: `After, hardened (${n})`,
		styles: d_
	})
}), p_ = (e) => e.outcome ?? (e.status === "blocked" ? "blocked" : e.status === "not_blocked" ? "landed" : "error"), m_ = (e) => e.outcome ?? (e.status === "passed" ? "served" : e.status === "refused" ? "refused" : "error"), h_ = (e, t) => e.filter((e) => e === t).length, g_ = (e, t) => {
	if (e.length === 0) {
		let e = t.benign_not_counted ?? 0, n = Math.max((t.benign_total ?? 0) - e, 0);
		if (t.benign_served !== void 0) {
			let r = t.benign_served, i = t.benign_refused ?? 0;
			return {
				ok: r,
				bad: i,
				errored: Math.max(n - r - i, 0),
				notCounted: e,
				total: n
			};
		}
		let r = t.benign_false_positives ?? 0;
		return {
			ok: Math.max(n - r, 0),
			bad: r,
			errored: 0,
			notCounted: e,
			total: n
		};
	}
	let n = e.map(m_), r = h_(n, "not_counted"), i = h_(n, "served"), a = h_(n, "refused"), o = e.length - r;
	return {
		ok: i,
		bad: a,
		errored: o - i - a,
		notCounted: r,
		total: o
	};
}, __ = (e, t) => {
	if (e.length === 0) {
		let e = t.attacks_total ?? 0, n = t.attacks_blocked ?? 0, r = t.attacks_landed ?? Math.max(e - n, 0);
		return {
			ok: n,
			bad: r,
			errored: Math.max(e - n - r, 0),
			notCounted: 0,
			total: e
		};
	}
	let n = e.map(p_), r = h_(n, "blocked"), i = h_(n, "landed");
	return {
		ok: r,
		bad: i,
		errored: e.length - r - i,
		notCounted: 0,
		total: e.length
	};
}, v_ = (e, t) => `${e} ${t}${e === 1 ? "" : "s"}`, y_ = (e, t) => [
	`${e.ok} / ${e.total}`,
	e.bad > 0 ? v_(e.bad, t) : "",
	e.errored > 0 ? v_(e.errored, "error") : "",
	e.notCounted > 0 ? `${e.notCounted} not counted` : ""
].filter(Boolean).join(" · "), b_ = {
	blocked_guardrail: "By guardrail",
	blocked_network: "By network",
	agent_refused: "Agent refused",
	tool_not_called: "Tool not called",
	tool_failed: "Tool failed"
}, x_ = {
	blocked_guardrail: "By guardrail",
	blocked_network: "By network",
	agent_refused: "Agent declined",
	tool_failed: "Tool failed"
}, S_ = (e) => {
	let t = p_(e);
	if (t === "error") return {
		text: "Error",
		color: "gray"
	};
	let n = b_[e.status ?? ""];
	return t === "blocked" ? {
		text: "Blocked",
		color: "green",
		reason: n
	} : {
		text: "Not blocked",
		color: "yellow",
		reason: n
	};
}, C_ = (e) => {
	let t = m_(e);
	if (t === "error") return {
		text: "Error",
		color: "gray"
	};
	let n = x_[e.status ?? ""];
	return t === "served" ? {
		text: "Passed",
		color: "green",
		reason: n
	} : t === "refused" ? {
		text: "Wrongly blocked",
		color: "yellow",
		reason: n
	} : {
		text: "Not counted",
		color: "gray",
		reason: n
	};
}, w_ = ({ verdict: e }) => /* @__PURE__ */ m(F, {
	align: "center",
	gap: "density-xs",
	className: "shrink-0",
	children: [e.reason ? /* @__PURE__ */ p(L, {
		kind: "body/regular/xs",
		className: "text-subtle",
		children: e.reason
	}) : null, /* @__PURE__ */ p(le, {
		color: e.color,
		children: e.text
	})]
}), T_ = ({ label: e, value: t, good: n }) => /* @__PURE__ */ p(de, {
	className: "flex-1 p-4",
	children: /* @__PURE__ */ m(I, {
		gap: "density-xs",
		children: [/* @__PURE__ */ p(L, {
			kind: "body/regular/sm",
			className: "text-subtle",
			children: e
		}), /* @__PURE__ */ p(L, {
			kind: "title/lg",
			style: { color: n ? Fa.success : Fa.warning },
			children: t
		})]
	})
}), E_ = ({ row: e }) => /* @__PURE__ */ m(F, {
	justify: "between",
	align: "start",
	gap: "density-md",
	className: "min-w-0 py-2",
	children: [/* @__PURE__ */ m(I, {
		gap: "density-xxs",
		className: "min-w-0",
		children: [
			/* @__PURE__ */ p(L, {
				kind: "body/semibold/sm",
				children: e.probe ?? e.attack_id ?? "attack"
			}),
			e.goal ? /* @__PURE__ */ p(L, {
				kind: "body/regular/sm",
				className: "truncate text-subtle",
				children: e.goal
			}) : null,
			e.prompt_excerpt ? /* @__PURE__ */ p(L, {
				kind: "body/regular/xs",
				className: "truncate text-subtle",
				children: e.prompt_excerpt
			}) : null
		]
	}), /* @__PURE__ */ p(w_, { verdict: S_(e) })]
}), D_ = ({ row: e }) => /* @__PURE__ */ m(F, {
	justify: "between",
	align: "start",
	gap: "density-md",
	className: "min-w-0 py-2",
	children: [/* @__PURE__ */ m(I, {
		gap: "density-xxs",
		className: "min-w-0",
		children: [/* @__PURE__ */ p(L, {
			kind: "body/semibold/sm",
			children: e.tool ?? e.label ?? `request ${e.index ?? ""}`
		}), e.payload_excerpt ? /* @__PURE__ */ p(L, {
			kind: "body/regular/xs",
			className: "truncate text-subtle",
			children: e.payload_excerpt
		}) : null]
	}), /* @__PURE__ */ p(w_, { verdict: C_(e) })]
}), O_ = ({ report: e }) => {
	let { summary: t, attacks: n, benign: r } = e, i = __(n, t), a = g_(r, t), o = i.ok === i.total && i.errored === 0, s = a.bad === 0 && a.errored === 0;
	return /* @__PURE__ */ m(I, {
		gap: "density-xl",
		children: [
			/* @__PURE__ */ m(F, {
				gap: "density-md",
				children: [/* @__PURE__ */ p(T_, {
					label: "Attacks blocked",
					value: y_(i, "not blocked"),
					good: o
				}), /* @__PURE__ */ p(T_, {
					label: "Benign preserved",
					value: y_(a, "false positive"),
					good: s
				})]
			}),
			n.length > 0 ? /* @__PURE__ */ m(I, {
				gap: "density-xs",
				children: [/* @__PURE__ */ m(L, {
					kind: "body/semibold/md",
					children: [
						"Attacks (",
						n.length,
						")"
					]
				}), /* @__PURE__ */ p(de, {
					className: "p-3 [&>*+*]:border-t [&>*+*]:border-base",
					children: n.map((e, t) => /* @__PURE__ */ p(E_, { row: e }, e.attack_id ?? t))
				})]
			}) : null,
			r.length > 0 ? /* @__PURE__ */ m(I, {
				gap: "density-xs",
				children: [/* @__PURE__ */ m(L, {
					kind: "body/semibold/md",
					children: [
						"Benign requests (",
						r.length,
						")"
					]
				}), /* @__PURE__ */ p(de, {
					className: "p-3 [&>*+*]:border-t [&>*+*]:border-base",
					children: r.map((e, t) => /* @__PURE__ */ p(D_, { row: e }, e.index ?? t))
				})]
			}) : null
		]
	});
}, k_ = 2e4, A_ = (e, t, n, r, i) => {
	let a = !!(i && v.includes(i)), [o, c] = d(!1), { data: l } = lo(e, t ?? "", void 0, { query: {
		enabled: !!t,
		refetchInterval: (e) => e.state.data?.data?.some((e) => e.name === n) || a && o ? !1 : D
	} }), u = !!l?.data?.some((e) => e.name === n);
	return s(() => {
		if (!a || u) {
			c(!1);
			return;
		}
		let e = setTimeout(() => c(!0), k_);
		return () => clearTimeout(e);
	}, [
		a,
		u,
		t,
		n
	]), {
		data: Be({
			queryKey: [
				"agent-hardener-job-artifact",
				e,
				t,
				n
			],
			enabled: u && !!t,
			queryFn: () => ro(e, t ?? "", n).then(r)
		}).data,
		present: u,
		isLoading: !!t && !u && (!a || !o),
		missing: !!t && a && !u && o
	};
}, j_ = (e) => e.text().then((e) => JSON.parse(e)), M_ = (e) => e.text(), N_ = Symbol.for("yaml.alias"), P_ = Symbol.for("yaml.document"), F_ = Symbol.for("yaml.map"), I_ = Symbol.for("yaml.pair"), L_ = Symbol.for("yaml.scalar"), R_ = Symbol.for("yaml.seq"), z_ = Symbol.for("yaml.node.type"), B_ = (e) => !!e && typeof e == "object" && e[z_] === N_, V_ = (e) => !!e && typeof e == "object" && e[z_] === P_, H_ = (e) => !!e && typeof e == "object" && e[z_] === F_, U_ = (e) => !!e && typeof e == "object" && e[z_] === I_, W_ = (e) => !!e && typeof e == "object" && e[z_] === L_, G_ = (e) => !!e && typeof e == "object" && e[z_] === R_;
function K_(e) {
	if (e && typeof e == "object") switch (e[z_]) {
		case F_:
		case R_: return !0;
	}
	return !1;
}
function q_(e) {
	if (e && typeof e == "object") switch (e[z_]) {
		case N_:
		case F_:
		case L_:
		case R_: return !0;
	}
	return !1;
}
var J_ = (e) => (W_(e) || K_(e)) && !!e.anchor, Y_ = Symbol("break visit"), X_ = Symbol("skip children"), Z_ = Symbol("remove node");
function Q_(e, t) {
	let n = ev(t);
	V_(e) ? $_(null, e.contents, n, Object.freeze([e])) === Z_ && (e.contents = null) : $_(null, e, n, Object.freeze([]));
}
Q_.BREAK = Y_, Q_.SKIP = X_, Q_.REMOVE = Z_;
function $_(e, t, n, r) {
	let i = tv(e, t, n, r);
	if (q_(i) || U_(i)) return nv(e, r, i), $_(e, i, n, r);
	if (typeof i != "symbol") {
		if (K_(t)) {
			r = Object.freeze(r.concat(t));
			for (let e = 0; e < t.items.length; ++e) {
				let i = $_(e, t.items[e], n, r);
				if (typeof i == "number") e = i - 1;
				else if (i === Y_) return Y_;
				else i === Z_ && (t.items.splice(e, 1), --e);
			}
		} else if (U_(t)) {
			r = Object.freeze(r.concat(t));
			let e = $_("key", t.key, n, r);
			if (e === Y_) return Y_;
			e === Z_ && (t.key = null);
			let i = $_("value", t.value, n, r);
			if (i === Y_) return Y_;
			i === Z_ && (t.value = null);
		}
	}
	return i;
}
function ev(e) {
	return typeof e == "object" && (e.Collection || e.Node || e.Value) ? Object.assign({
		Alias: e.Node,
		Map: e.Node,
		Scalar: e.Node,
		Seq: e.Node
	}, e.Value && {
		Map: e.Value,
		Scalar: e.Value,
		Seq: e.Value
	}, e.Collection && {
		Map: e.Collection,
		Seq: e.Collection
	}, e) : e;
}
function tv(e, t, n, r) {
	if (typeof n == "function") return n(e, t, r);
	if (H_(t)) return n.Map?.(e, t, r);
	if (G_(t)) return n.Seq?.(e, t, r);
	if (U_(t)) return n.Pair?.(e, t, r);
	if (W_(t)) return n.Scalar?.(e, t, r);
	if (B_(t)) return n.Alias?.(e, t, r);
}
function nv(e, t, n) {
	let r = t[t.length - 1];
	if (K_(r)) r.items[e] = n;
	else if (U_(r)) e === "key" ? r.key = n : r.value = n;
	else if (V_(r)) r.contents = n;
	else {
		let e = B_(r) ? "alias" : "scalar";
		throw Error(`Cannot replace node with ${e} parent`);
	}
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/doc/directives.js
var rv = {
	"!": "%21",
	",": "%2C",
	"[": "%5B",
	"]": "%5D",
	"{": "%7B",
	"}": "%7D"
}, iv = (e) => e.replace(/[!,[\]{}]/g, (e) => rv[e]), av = class e {
	constructor(t, n) {
		this.docStart = null, this.docEnd = !1, this.yaml = Object.assign({}, e.defaultYaml, t), this.tags = Object.assign({}, e.defaultTags, n);
	}
	clone() {
		let t = new e(this.yaml, this.tags);
		return t.docStart = this.docStart, t;
	}
	atDocument() {
		let t = new e(this.yaml, this.tags);
		switch (this.yaml.version) {
			case "1.1":
				this.atNextDocument = !0;
				break;
			case "1.2": this.atNextDocument = !1, this.yaml = {
				explicit: e.defaultYaml.explicit,
				version: "1.2"
			}, this.tags = Object.assign({}, e.defaultTags);
		}
		return t;
	}
	add(t, n) {
		this.atNextDocument &&= (this.yaml = {
			explicit: e.defaultYaml.explicit,
			version: "1.1"
		}, this.tags = Object.assign({}, e.defaultTags), !1);
		let r = t.trim().split(/[ \t]+/), i = r.shift();
		switch (i) {
			case "%TAG": {
				if (r.length !== 2 && (n(0, "%TAG directive should contain exactly two parts"), r.length < 2)) return !1;
				let [e, t] = r;
				return this.tags[e] = t, !0;
			}
			case "%YAML": {
				if (this.yaml.explicit = !0, r.length !== 1) return n(0, "%YAML directive should contain exactly one part"), !1;
				let [e] = r;
				if (e === "1.1" || e === "1.2") return this.yaml.version = e, !0;
				{
					let t = /^\d+\.\d+$/.test(e);
					return n(6, `Unsupported YAML version ${e}`, t), !1;
				}
			}
			default: return n(0, `Unknown directive ${i}`, !0), !1;
		}
	}
	tagName(e, t) {
		if (e === "!") return "!";
		if (e[0] !== "!") return t(`Not a valid tag: ${e}`), null;
		if (e[1] === "<") {
			let n = e.slice(2, -1);
			return n === "!" || n === "!!" ? (t(`Verbatim tags aren't resolved, so ${e} is invalid.`), null) : (e[e.length - 1] !== ">" && t("Verbatim tags must end with a >"), n);
		}
		let [, n, r] = e.match(/^(.*!)([^!]*)$/s);
		r || t(`The ${e} tag has no suffix`);
		let i = this.tags[n];
		if (i) try {
			return i + decodeURIComponent(r);
		} catch (e) {
			return t(String(e)), null;
		}
		return n === "!" ? e : (t(`Could not resolve tag: ${e}`), null);
	}
	tagString(e) {
		for (let [t, n] of Object.entries(this.tags)) if (e.startsWith(n)) return t + iv(e.substring(n.length));
		return e[0] === "!" ? e : `!<${e}>`;
	}
	toString(e) {
		let t = this.yaml.explicit ? [`%YAML ${this.yaml.version || "1.2"}`] : [], n = Object.entries(this.tags), r;
		if (e && n.length > 0 && q_(e.contents)) {
			let t = {};
			Q_(e.contents, (e, n) => {
				q_(n) && n.tag && (t[n.tag] = !0);
			}), r = Object.keys(t);
		} else r = [];
		for (let [i, a] of n) (i !== "!!" || a !== "tag:yaml.org,2002:") && (!e || r.some((e) => e.startsWith(a))) && t.push(`%TAG ${i} ${a}`);
		return t.join("\n");
	}
};
av.defaultYaml = {
	explicit: !1,
	version: "1.2"
}, av.defaultTags = { "!!": "tag:yaml.org,2002:" };
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/doc/anchors.js
function ov(e) {
	if (/[\x00-\x19\s,[\]{}]/.test(e)) {
		let t = `Anchor must not contain whitespace or control characters: ${JSON.stringify(e)}`;
		throw Error(t);
	}
	return !0;
}
function sv(e) {
	let t = /* @__PURE__ */ new Set();
	return Q_(e, { Value(e, n) {
		n.anchor && t.add(n.anchor);
	} }), t;
}
function cv(e, t) {
	for (let n = 1;; ++n) {
		let r = `${e}${n}`;
		if (!t.has(r)) return r;
	}
}
function lv(e, t) {
	let n = [], r = /* @__PURE__ */ new Map(), i = null;
	return {
		onAnchor: (r) => {
			n.push(r), i ??= sv(e);
			let a = cv(t, i);
			return i.add(a), a;
		},
		setAnchors: () => {
			for (let e of n) {
				let t = r.get(e);
				if (typeof t == "object" && t.anchor && (W_(t.node) || K_(t.node))) t.node.anchor = t.anchor;
				else {
					let t = /* @__PURE__ */ Error("Failed to resolve repeated object (this should not happen)");
					throw t.source = e, t;
				}
			}
		},
		sourceObjects: r
	};
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/doc/applyReviver.js
function uv(e, t, n, r) {
	if (r && typeof r == "object") {
		if (Array.isArray(r)) for (let t = 0, n = r.length; t < n; ++t) {
			let n = r[t], i = uv(e, r, String(t), n);
			i === void 0 ? delete r[t] : i !== n && (r[t] = i);
		}
		else if (r instanceof Map) for (let t of Array.from(r.keys())) {
			let n = r.get(t), i = uv(e, r, t, n);
			i === void 0 ? r.delete(t) : i !== n && r.set(t, i);
		}
		else if (r instanceof Set) for (let t of Array.from(r)) {
			let n = uv(e, r, t, t);
			n === void 0 ? r.delete(t) : n !== t && (r.delete(t), r.add(n));
		}
		else for (let [t, n] of Object.entries(r)) {
			let i = uv(e, r, t, n);
			i === void 0 ? delete r[t] : i !== n && (r[t] = i);
		}
	}
	return e.call(t, n, r);
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/nodes/toJS.js
function dv(e, t, n) {
	if (Array.isArray(e)) return e.map((e, t) => dv(e, String(t), n));
	if (e && typeof e.toJSON == "function") {
		if (!n || !J_(e)) return e.toJSON(t, n);
		let r = {
			aliasCount: 0,
			count: 1,
			res: void 0
		};
		n.anchors.set(e, r), n.onCreate = (e) => {
			r.res = e, delete n.onCreate;
		};
		let i = e.toJSON(t, n);
		return n.onCreate && n.onCreate(i), i;
	}
	return typeof e == "bigint" && !n?.keep ? Number(e) : e;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/nodes/Node.js
var fv = class {
	constructor(e) {
		Object.defineProperty(this, z_, { value: e });
	}
	clone() {
		let e = Object.create(Object.getPrototypeOf(this), Object.getOwnPropertyDescriptors(this));
		return this.range && (e.range = this.range.slice()), e;
	}
	toJS(e, { mapAsMap: t, maxAliasCount: n, onAnchor: r, reviver: i } = {}) {
		if (!V_(e)) throw TypeError("A document argument is required");
		let a = {
			anchors: /* @__PURE__ */ new Map(),
			doc: e,
			keep: !0,
			mapAsMap: t === !0,
			mapKeyWarned: !1,
			maxAliasCount: typeof n == "number" ? n : 100
		}, o = dv(this, "", a);
		if (typeof r == "function") for (let { count: e, res: t } of a.anchors.values()) r(t, e);
		return typeof i == "function" ? uv(i, { "": o }, "", o) : o;
	}
}, pv = class extends fv {
	constructor(e) {
		super(N_), this.source = e, Object.defineProperty(this, "tag", { set() {
			throw Error("Alias nodes cannot have tags");
		} });
	}
	resolve(e, t) {
		if (t?.maxAliasCount === 0) throw ReferenceError("Alias resolution is disabled");
		let n;
		t?.aliasResolveCache ? n = t.aliasResolveCache : (n = [], Q_(e, { Node: (e, t) => {
			(B_(t) || J_(t)) && n.push(t);
		} }), t && (t.aliasResolveCache = n));
		let r;
		for (let e of n) {
			if (e === this) break;
			e.anchor === this.source && (r = e);
		}
		return r;
	}
	toJSON(e, t) {
		if (!t) return { source: this.source };
		let { anchors: n, doc: r, maxAliasCount: i } = t, a = this.resolve(r, t);
		if (!a) {
			let e = `Unresolved alias (the anchor must be set before the alias): ${this.source}`;
			throw ReferenceError(e);
		}
		let o = n.get(a);
		/* istanbul ignore if */
		if (o ||= (dv(a, null, t), n.get(a)), o?.res === void 0) throw ReferenceError("This should not happen: Alias anchor was not resolved?");
		if (i >= 0 && (o.count += 1, o.aliasCount === 0 && (o.aliasCount = mv(r, a, n)), o.count * o.aliasCount > i)) throw ReferenceError("Excessive alias count indicates a resource exhaustion attack");
		return o.res;
	}
	toString(e, t, n) {
		let r = `*${this.source}`;
		if (e) {
			if (ov(this.source), e.options.verifyAliasOrder && !e.anchors.has(this.source)) {
				let e = `Unresolved alias (the anchor must be set before the alias): ${this.source}`;
				throw Error(e);
			}
			if (e.implicitKey) return `${r} `;
		}
		return r;
	}
};
function mv(e, t, n) {
	if (B_(t)) {
		let r = t.resolve(e), i = n && r && n.get(r);
		return i ? i.count * i.aliasCount : 0;
	}
	if (K_(t)) {
		let r = 0;
		for (let i of t.items) {
			let t = mv(e, i, n);
			t > r && (r = t);
		}
		return r;
	}
	if (U_(t)) {
		let r = mv(e, t.key, n), i = mv(e, t.value, n);
		return Math.max(r, i);
	}
	return 1;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/nodes/Scalar.js
var hv = (e) => !e || typeof e != "function" && typeof e != "object", W = class extends fv {
	constructor(e) {
		super(L_), this.value = e;
	}
	toJSON(e, t) {
		return t?.keep ? this.value : dv(this.value, e, t);
	}
	toString() {
		return String(this.value);
	}
};
W.BLOCK_FOLDED = "BLOCK_FOLDED", W.BLOCK_LITERAL = "BLOCK_LITERAL", W.PLAIN = "PLAIN", W.QUOTE_DOUBLE = "QUOTE_DOUBLE", W.QUOTE_SINGLE = "QUOTE_SINGLE";
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/doc/createNode.js
var gv = "tag:yaml.org,2002:";
function _v(e, t, n) {
	if (t) {
		let e = n.filter((e) => e.tag === t), r = e.find((e) => !e.format) ?? e[0];
		if (!r) throw Error(`Tag ${t} not found`);
		return r;
	}
	return n.find((t) => t.identify?.(e) && !t.format);
}
function vv(e, t, n) {
	if (V_(e) && (e = e.contents), q_(e)) return e;
	if (U_(e)) {
		let t = n.schema[F_].createNode?.(n.schema, null, n);
		return t.items.push(e), t;
	}
	(e instanceof String || e instanceof Number || e instanceof Boolean || typeof BigInt < "u" && e instanceof BigInt) && (e = e.valueOf());
	let { aliasDuplicateObjects: r, onAnchor: i, onTagObj: a, schema: o, sourceObjects: s } = n, c;
	if (r && e && typeof e == "object") {
		if (c = s.get(e), c) return c.anchor ?? (c.anchor = i(e)), new pv(c.anchor);
		c = {
			anchor: null,
			node: null
		}, s.set(e, c);
	}
	t?.startsWith("!!") && (t = gv + t.slice(2));
	let l = _v(e, t, o.tags);
	if (!l) {
		if (e && typeof e.toJSON == "function" && (e = e.toJSON()), !e || typeof e != "object") {
			let t = new W(e);
			return c && (c.node = t), t;
		}
		l = e instanceof Map ? o[F_] : Symbol.iterator in Object(e) ? o[R_] : o[F_];
	}
	a && (a(l), delete n.onTagObj);
	let u = l?.createNode ? l.createNode(n.schema, e, n) : typeof l?.nodeClass?.from == "function" ? l.nodeClass.from(n.schema, e, n) : new W(e);
	return t ? u.tag = t : l.default || (u.tag = l.tag), c && (c.node = u), u;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/nodes/Collection.js
function yv(e, t, n) {
	let r = n;
	for (let e = t.length - 1; e >= 0; --e) {
		let n = t[e];
		if (typeof n == "number" && Number.isInteger(n) && n >= 0) {
			let e = [];
			e[n] = r, r = e;
		} else r = /* @__PURE__ */ new Map([[n, r]]);
	}
	return vv(r, void 0, {
		aliasDuplicateObjects: !1,
		keepUndefined: !1,
		onAnchor: () => {
			throw Error("This should not happen, please report a bug.");
		},
		schema: e,
		sourceObjects: /* @__PURE__ */ new Map()
	});
}
var bv = (e) => e == null || typeof e == "object" && !!e[Symbol.iterator]().next().done, xv = class extends fv {
	constructor(e, t) {
		super(e), Object.defineProperty(this, "schema", {
			value: t,
			configurable: !0,
			enumerable: !1,
			writable: !0
		});
	}
	clone(e) {
		let t = Object.create(Object.getPrototypeOf(this), Object.getOwnPropertyDescriptors(this));
		return e && (t.schema = e), t.items = t.items.map((t) => q_(t) || U_(t) ? t.clone(e) : t), this.range && (t.range = this.range.slice()), t;
	}
	addIn(e, t) {
		if (bv(e)) this.add(t);
		else {
			let [n, ...r] = e, i = this.get(n, !0);
			if (K_(i)) i.addIn(r, t);
			else if (i === void 0 && this.schema) this.set(n, yv(this.schema, r, t));
			else throw Error(`Expected YAML collection at ${n}. Remaining path: ${r}`);
		}
	}
	deleteIn(e) {
		let [t, ...n] = e;
		if (n.length === 0) return this.delete(t);
		let r = this.get(t, !0);
		if (K_(r)) return r.deleteIn(n);
		throw Error(`Expected YAML collection at ${t}. Remaining path: ${n}`);
	}
	getIn(e, t) {
		let [n, ...r] = e, i = this.get(n, !0);
		return r.length === 0 ? !t && W_(i) ? i.value : i : K_(i) ? i.getIn(r, t) : void 0;
	}
	hasAllNullValues(e) {
		return this.items.every((t) => {
			if (!U_(t)) return !1;
			let n = t.value;
			return n == null || e && W_(n) && n.value == null && !n.commentBefore && !n.comment && !n.tag;
		});
	}
	hasIn(e) {
		let [t, ...n] = e;
		if (n.length === 0) return this.has(t);
		let r = this.get(t, !0);
		return K_(r) ? r.hasIn(n) : !1;
	}
	setIn(e, t) {
		let [n, ...r] = e;
		if (r.length === 0) this.set(n, t);
		else {
			let e = this.get(n, !0);
			if (K_(e)) e.setIn(r, t);
			else if (e === void 0 && this.schema) this.set(n, yv(this.schema, r, t));
			else throw Error(`Expected YAML collection at ${n}. Remaining path: ${r}`);
		}
	}
}, Sv = (e) => e.replace(/^(?!$)(?: $)?/gm, "#");
function Cv(e, t) {
	return /^\n+$/.test(e) ? e.substring(1) : t ? e.replace(/^(?! *$)/gm, t) : e;
}
var wv = (e, t, n) => e.endsWith("\n") ? Cv(n, t) : n.includes("\n") ? "\n" + Cv(n, t) : (e.endsWith(" ") ? "" : " ") + n, Tv = "flow", Ev = "block", Dv = "quoted";
function Ov(e, t, n = "flow", { indentAtStart: r, lineWidth: i = 80, minContentWidth: a = 20, onFold: o, onOverflow: s } = {}) {
	if (!i || i < 0) return e;
	i < a && (a = 0);
	let c = Math.max(1 + a, 1 + i - t.length);
	if (e.length <= c) return e;
	let l = [], u = {}, d = i - t.length;
	typeof r == "number" && (r > i - Math.max(2, a) ? l.push(0) : d = i - r);
	let f, p, m = !1, h = -1, g = -1, _ = -1;
	n === "block" && (h = kv(e, h, t.length), h !== -1 && (d = h + c));
	for (let r; r = e[h += 1];) {
		if (n === "quoted" && r === "\\") {
			switch (g = h, e[h + 1]) {
				case "x":
					h += 3;
					break;
				case "u":
					h += 5;
					break;
				case "U":
					h += 9;
					break;
				default: h += 1;
			}
			_ = h;
		}
		if (r === "\n") n === "block" && (h = kv(e, h, t.length)), d = h + t.length + c, f = void 0;
		else {
			if (r === " " && p && p !== " " && p !== "\n" && p !== "	") {
				let t = e[h + 1];
				t && t !== " " && t !== "\n" && t !== "	" && (f = h);
			}
			if (h >= d) {
				if (f) l.push(f), d = f + c, f = void 0;
				else if (n === "quoted") {
					for (; p === " " || p === "	";) p = r, r = e[h += 1], m = !0;
					let t = h > _ + 1 ? h - 2 : g - 1;
					if (u[t]) return e;
					l.push(t), u[t] = !0, d = t + c, f = void 0;
				} else m = !0;
			}
		}
		p = r;
	}
	if (m && s && s(), l.length === 0) return e;
	o && o();
	let v = e.slice(0, l[0]);
	for (let r = 0; r < l.length; ++r) {
		let i = l[r], a = l[r + 1] || e.length;
		i === 0 ? v = `\n${t}${e.slice(0, a)}` : (n === "quoted" && u[i] && (v += `${e[i]}\\`), v += `\n${t}${e.slice(i + 1, a)}`);
	}
	return v;
}
function kv(e, t, n) {
	let r = t, i = t + 1, a = e[i];
	for (; a === " " || a === "	";) if (t < i + n) a = e[++t];
	else {
		do
			a = e[++t];
		while (a && a !== "\n");
		r = t, i = t + 1, a = e[i];
	}
	return r;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/stringify/stringifyString.js
var Av = (e, t) => ({
	indentAtStart: t ? e.indent.length : e.indentAtStart,
	lineWidth: e.options.lineWidth,
	minContentWidth: e.options.minContentWidth
}), jv = (e) => /^(%|---|\.\.\.)/m.test(e);
function Mv(e, t, n) {
	if (!t || t < 0) return !1;
	let r = t - n, i = e.length;
	if (i <= r) return !1;
	for (let t = 0, n = 0; t < i; ++t) if (e[t] === "\n") {
		if (t - n > r) return !0;
		if (n = t + 1, i - n <= r) return !1;
	}
	return !0;
}
function Nv(e, t) {
	let n = JSON.stringify(e);
	if (t.options.doubleQuotedAsJSON) return n;
	let { implicitKey: r } = t, i = t.options.doubleQuotedMinMultiLineLength, a = t.indent || (jv(e) ? "  " : ""), o = "", s = 0;
	for (let e = 0, t = n[e]; t; t = n[++e]) if (t === " " && n[e + 1] === "\\" && n[e + 2] === "n" && (o += n.slice(s, e) + "\\ ", e += 1, s = e, t = "\\"), t === "\\") switch (n[e + 1]) {
		case "u":
			{
				o += n.slice(s, e);
				let t = n.substr(e + 2, 4);
				switch (t) {
					case "0000":
						o += "\\0";
						break;
					case "0007":
						o += "\\a";
						break;
					case "000b":
						o += "\\v";
						break;
					case "001b":
						o += "\\e";
						break;
					case "0085":
						o += "\\N";
						break;
					case "00a0":
						o += "\\_";
						break;
					case "2028":
						o += "\\L";
						break;
					case "2029":
						o += "\\P";
						break;
					default: t.substr(0, 2) === "00" ? o += "\\x" + t.substr(2) : o += n.substr(e, 6);
				}
				e += 5, s = e + 1;
			}
			break;
		case "n":
			if (r || n[e + 2] === "\"" || n.length < i) e += 1;
			else {
				for (o += n.slice(s, e) + "\n\n"; n[e + 2] === "\\" && n[e + 3] === "n" && n[e + 4] !== "\"";) o += "\n", e += 2;
				o += a, n[e + 2] === " " && (o += "\\"), e += 1, s = e + 1;
			}
			break;
		default: e += 1;
	}
	return o = s ? o + n.slice(s) : n, r ? o : Ov(o, a, Dv, Av(t, !1));
}
function Pv(e, t) {
	if (t.options.singleQuote === !1 || t.implicitKey && e.includes("\n") || /[ \t]\n|\n[ \t]/.test(e)) return Nv(e, t);
	let n = t.indent || (jv(e) ? "  " : ""), r = "'" + e.replace(/'/g, "''").replace(/\n+/g, `$&\n${n}`) + "'";
	return t.implicitKey ? r : Ov(r, n, Tv, Av(t, !1));
}
function Fv(e, t) {
	let { singleQuote: n } = t.options, r;
	if (n === !1) r = Nv;
	else {
		let t = e.includes("\""), i = e.includes("'");
		r = t && !i ? Pv : i && !t ? Nv : n ? Pv : Nv;
	}
	return r(e, t);
}
var Iv;
try {
	Iv = /* @__PURE__ */ RegExp("(^|(?<!\n))\n+(?!\n|$)", "g");
} catch {
	Iv = /\n+(?!\n|$)/g;
}
function Lv({ comment: e, type: t, value: n }, r, i, a) {
	let { blockQuote: o, commentString: s, lineWidth: c } = r.options;
	if (!o || /\n[\t ]+$/.test(n)) return Fv(n, r);
	let l = r.indent || (r.forceBlockIndent || jv(n) ? "  " : ""), u = o === "literal" ? !0 : o === "folded" || t === W.BLOCK_FOLDED ? !1 : t === W.BLOCK_LITERAL || !Mv(n, c, l.length);
	if (!n) return u ? "|\n" : ">\n";
	let d, f;
	for (f = n.length; f > 0; --f) {
		let e = n[f - 1];
		if (e !== "\n" && e !== "	" && e !== " ") break;
	}
	let p = n.substring(f), m = p.indexOf("\n");
	m === -1 ? d = "-" : n === p || m !== p.length - 1 ? (d = "+", a && a()) : d = "", p &&= (n = n.slice(0, -p.length), p[p.length - 1] === "\n" && (p = p.slice(0, -1)), p.replace(Iv, `$&${l}`));
	let h = !1, g, _ = -1;
	for (g = 0; g < n.length; ++g) {
		let e = n[g];
		if (e === " ") h = !0;
		else if (e === "\n") _ = g;
		else break;
	}
	let v = n.substring(0, _ < g ? _ + 1 : g);
	v &&= (n = n.substring(v.length), v.replace(/\n+/g, `$&${l}`));
	let y = (h ? l ? "2" : "1" : "") + d;
	if (e && (y += " " + s(e.replace(/ ?[\r\n]+/g, " ")), i && i()), !u) {
		let e = n.replace(/\n+/g, "\n$&").replace(/(?:^|\n)([\t ].*)(?:([\n\t ]*)\n(?![\n\t ]))?/g, "$1$2").replace(/\n+/g, `$&${l}`), i = !1, a = Av(r, !0);
		o !== "folded" && t !== W.BLOCK_FOLDED && (a.onOverflow = () => {
			i = !0;
		});
		let s = Ov(`${v}${e}${p}`, l, Ev, a);
		if (!i) return `>${y}\n${l}${s}`;
	}
	return n = n.replace(/\n+/g, `$&${l}`), `|${y}\n${l}${v}${n}${p}`;
}
function Rv(e, t, n, r) {
	let { type: i, value: a } = e, { actualString: o, implicitKey: s, indent: c, indentStep: l, inFlow: u } = t;
	if (s && a.includes("\n") || u && /[[\]{},]/.test(a)) return Fv(a, t);
	if (/^[\n\t ,[\]{}#&*!|>'"%@`]|^[?-]$|^[?-][ \t]|[\n:][ \t]|[ \t]\n|[\n\t ]#|[\n\t :]$/.test(a)) return s || u || !a.includes("\n") ? Fv(a, t) : Lv(e, t, n, r);
	if (!s && !u && i !== W.PLAIN && a.includes("\n")) return Lv(e, t, n, r);
	if (jv(a)) {
		if (c === "") return t.forceBlockIndent = !0, Lv(e, t, n, r);
		if (s && c === l) return Fv(a, t);
	}
	let d = a.replace(/\n+/g, `$&\n${c}`);
	if (o) {
		let e = (e) => e.default && e.tag !== "tag:yaml.org,2002:str" && e.test?.test(d), { compat: n, tags: r } = t.doc.schema;
		if (r.some(e) || n?.some(e)) return Fv(a, t);
	}
	return s ? d : Ov(d, c, Tv, Av(t, !1));
}
function zv(e, t, n, r) {
	let { implicitKey: i, inFlow: a } = t, o = typeof e.value == "string" ? e : Object.assign({}, e, { value: String(e.value) }), { type: s } = e;
	s !== W.QUOTE_DOUBLE && /[\x00-\x08\x0b-\x1f\x7f-\x9f\u{D800}-\u{DFFF}]/u.test(o.value) && (s = W.QUOTE_DOUBLE);
	let c = (e) => {
		switch (e) {
			case W.BLOCK_FOLDED:
			case W.BLOCK_LITERAL: return i || a ? Fv(o.value, t) : Lv(o, t, n, r);
			case W.QUOTE_DOUBLE: return Nv(o.value, t);
			case W.QUOTE_SINGLE: return Pv(o.value, t);
			case W.PLAIN: return Rv(o, t, n, r);
			default: return null;
		}
	}, l = c(s);
	if (l === null) {
		let { defaultKeyType: e, defaultStringType: n } = t.options, r = i && e || n;
		if (l = c(r), l === null) throw Error(`Unsupported default string type ${r}`);
	}
	return l;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/stringify/stringify.js
function Bv(e, t) {
	let n = Object.assign({
		blockQuote: !0,
		commentString: Sv,
		defaultKeyType: null,
		defaultStringType: "PLAIN",
		directives: null,
		doubleQuotedAsJSON: !1,
		doubleQuotedMinMultiLineLength: 40,
		falseStr: "false",
		flowCollectionPadding: !0,
		indentSeq: !0,
		lineWidth: 80,
		minContentWidth: 20,
		nullStr: "null",
		simpleKeys: !1,
		singleQuote: null,
		trailingComma: !1,
		trueStr: "true",
		verifyAliasOrder: !0
	}, e.schema.toStringOptions, t), r;
	switch (n.collectionStyle) {
		case "block":
			r = !1;
			break;
		case "flow":
			r = !0;
			break;
		default: r = null;
	}
	return {
		anchors: /* @__PURE__ */ new Set(),
		doc: e,
		flowCollectionPadding: n.flowCollectionPadding ? " " : "",
		indent: "",
		indentStep: typeof n.indent == "number" ? " ".repeat(n.indent) : "  ",
		inFlow: r,
		options: n
	};
}
function Vv(e, t) {
	if (t.tag) {
		let n = e.filter((e) => e.tag === t.tag);
		if (n.length > 0) return n.find((e) => e.format === t.format) ?? n[0];
	}
	let n, r;
	if (W_(t)) {
		r = t.value;
		let i = e.filter((e) => e.identify?.(r));
		if (i.length > 1) {
			let e = i.filter((e) => e.test);
			e.length > 0 && (i = e);
		}
		n = i.find((e) => e.format === t.format) ?? i.find((e) => !e.format);
	} else r = t, n = e.find((e) => e.nodeClass && r instanceof e.nodeClass);
	if (!n) {
		let e = r?.constructor?.name ?? (r === null ? "null" : typeof r);
		throw Error(`Tag not resolved for ${e} value`);
	}
	return n;
}
function Hv(e, t, { anchors: n, doc: r }) {
	if (!r.directives) return "";
	let i = [], a = (W_(e) || K_(e)) && e.anchor;
	a && ov(a) && (n.add(a), i.push(`&${a}`));
	let o = e.tag ?? (t.default ? null : t.tag);
	return o && i.push(r.directives.tagString(o)), i.join(" ");
}
function Uv(e, t, n, r) {
	if (U_(e)) return e.toString(t, n, r);
	if (B_(e)) {
		if (t.doc.directives) return e.toString(t);
		if (t.resolvedAliases?.has(e)) throw TypeError("Cannot stringify circular structure without alias nodes");
		t.resolvedAliases ? t.resolvedAliases.add(e) : t.resolvedAliases = /* @__PURE__ */ new Set([e]), e = e.resolve(t.doc);
	}
	let i, a = q_(e) ? e : t.doc.createNode(e, { onTagObj: (e) => i = e });
	i ??= Vv(t.doc.schema.tags, a);
	let o = Hv(a, i, t);
	o.length > 0 && (t.indentAtStart = (t.indentAtStart ?? 0) + o.length + 1);
	let s = typeof i.stringify == "function" ? i.stringify(a, t, n, r) : W_(a) ? zv(a, t, n, r) : a.toString(t, n, r);
	return o ? W_(a) || s[0] === "{" || s[0] === "[" ? `${o} ${s}` : `${o}\n${t.indent}${s}` : s;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/stringify/stringifyPair.js
function Wv({ key: e, value: t }, n, r, i) {
	let { allNullValues: a, doc: o, indent: s, indentStep: c, options: { commentString: l, indentSeq: u, simpleKeys: d } } = n, f = q_(e) && e.comment || null;
	if (d) {
		if (f) throw Error("With simple keys, key nodes cannot have comments");
		if (K_(e) || !q_(e) && typeof e == "object") throw Error("With simple keys, collection cannot be used as a key value");
	}
	let p = !d && (!e || f && t == null && !n.inFlow || K_(e) || (W_(e) ? e.type === W.BLOCK_FOLDED || e.type === W.BLOCK_LITERAL : typeof e == "object"));
	n = Object.assign({}, n, {
		allNullValues: !1,
		implicitKey: !p && (d || !a),
		indent: s + c
	});
	let m = !1, h = !1, g = Uv(e, n, () => m = !0, () => h = !0);
	if (!p && !n.inFlow && g.length > 1024) {
		if (d) throw Error("With simple keys, single line scalar must not span more than 1024 characters");
		p = !0;
	}
	if (n.inFlow) {
		if (a || t == null) return m && r && r(), g === "" ? "?" : p ? `? ${g}` : g;
	} else if (a && !d || t == null && p) return g = `? ${g}`, f && !m ? g += wv(g, n.indent, l(f)) : h && i && i(), g;
	m && (f = null), p ? (f && (g += wv(g, n.indent, l(f))), g = `? ${g}\n${s}:`) : (g = `${g}:`, f && (g += wv(g, n.indent, l(f))));
	let _, v, y;
	q_(t) ? (_ = !!t.spaceBefore, v = t.commentBefore, y = t.comment) : (_ = !1, v = null, y = null, t && typeof t == "object" && (t = o.createNode(t))), n.implicitKey = !1, !p && !f && W_(t) && (n.indentAtStart = g.length + 1), h = !1, !u && c.length >= 2 && !n.inFlow && !p && G_(t) && !t.flow && !t.tag && !t.anchor && (n.indent = n.indent.substring(2));
	let b = !1, x = Uv(t, n, () => b = !0, () => h = !0), S = " ";
	if (f || _ || v) {
		if (S = _ ? "\n" : "", v) {
			let e = l(v);
			S += `\n${Cv(e, n.indent)}`;
		}
		x === "" && !n.inFlow ? S === "\n" && y && (S = "\n\n") : S += `\n${n.indent}`;
	} else if (!p && K_(t)) {
		let e = x[0], r = x.indexOf("\n"), i = r !== -1, a = n.inFlow ?? t.flow ?? t.items.length === 0;
		if (i || !a) {
			let t = !1;
			if (i && (e === "&" || e === "!")) {
				let n = x.indexOf(" ");
				e === "&" && n !== -1 && n < r && x[n + 1] === "!" && (n = x.indexOf(" ", n + 1)), (n === -1 || r < n) && (t = !0);
			}
			t || (S = `\n${n.indent}`);
		}
	} else (x === "" || x[0] === "\n") && (S = "");
	return g += S + x, n.inFlow ? b && r && r() : y && !b ? g += wv(g, n.indent, l(y)) : h && i && i(), g;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/log.js
function Gv(e, t) {
	(e === "debug" || e === "warn") && console.warn(t);
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/schema/yaml-1.1/merge.js
var Kv = "<<", qv = {
	identify: (e) => e === Kv || typeof e == "symbol" && e.description === Kv,
	default: "key",
	tag: "tag:yaml.org,2002:merge",
	test: /^<<$/,
	resolve: () => Object.assign(new W(Symbol(Kv)), { addToJSMap: Yv }),
	stringify: () => Kv
}, Jv = (e, t) => (qv.identify(t) || W_(t) && (!t.type || t.type === W.PLAIN) && qv.identify(t.value)) && e?.doc.schema.tags.some((e) => e.tag === qv.tag && e.default);
function Yv(e, t, n) {
	let r = Zv(e, n);
	if (G_(r)) for (let n of r.items) Xv(e, t, n);
	else if (Array.isArray(r)) for (let n of r) Xv(e, t, n);
	else Xv(e, t, r);
}
function Xv(e, t, n) {
	let r = Zv(e, n);
	if (!H_(r)) throw Error("Merge sources must be maps or map aliases");
	let i = r.toJSON(null, e, Map);
	for (let [e, n] of i) t instanceof Map ? t.has(e) || t.set(e, n) : t instanceof Set ? t.add(e) : Object.prototype.hasOwnProperty.call(t, e) || Object.defineProperty(t, e, {
		value: n,
		writable: !0,
		enumerable: !0,
		configurable: !0
	});
	return t;
}
function Zv(e, t) {
	return e && B_(t) ? t.resolve(e.doc, e) : t;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/nodes/addPairToJSMap.js
function Qv(e, t, { key: n, value: r }) {
	if (q_(n) && n.addToJSMap) n.addToJSMap(e, t, r);
	else if (Jv(e, n)) Yv(e, t, r);
	else {
		let i = dv(n, "", e);
		if (t instanceof Map) t.set(i, dv(r, i, e));
		else if (t instanceof Set) t.add(i);
		else {
			let a = $v(n, i, e), o = dv(r, a, e);
			a in t ? Object.defineProperty(t, a, {
				value: o,
				writable: !0,
				enumerable: !0,
				configurable: !0
			}) : t[a] = o;
		}
	}
	return t;
}
function $v(e, t, n) {
	if (t === null) return "";
	if (typeof t != "object") return String(t);
	if (q_(e) && n?.doc) {
		let t = Bv(n.doc, {});
		t.anchors = /* @__PURE__ */ new Set();
		for (let e of n.anchors.keys()) t.anchors.add(e.anchor);
		t.inFlow = !0, t.inStringifyKey = !0;
		let r = e.toString(t);
		if (!n.mapKeyWarned) {
			let e = JSON.stringify(r);
			e.length > 40 && (e = e.substring(0, 36) + "...\""), Gv(n.doc.options.logLevel, `Keys with collection values will be stringified due to JS Object restrictions: ${e}. Set mapAsMap: true to use object keys.`), n.mapKeyWarned = !0;
		}
		return r;
	}
	return JSON.stringify(t);
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/nodes/Pair.js
function ey(e, t, n) {
	return new ty(vv(e, void 0, n), vv(t, void 0, n));
}
var ty = class e {
	constructor(e, t = null) {
		Object.defineProperty(this, z_, { value: I_ }), this.key = e, this.value = t;
	}
	clone(t) {
		let { key: n, value: r } = this;
		return q_(n) && (n = n.clone(t)), q_(r) && (r = r.clone(t)), new e(n, r);
	}
	toJSON(e, t) {
		return Qv(t, t?.mapAsMap ? /* @__PURE__ */ new Map() : {}, this);
	}
	toString(e, t, n) {
		return e?.doc ? Wv(this, e, t, n) : JSON.stringify(this);
	}
};
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/stringify/stringifyCollection.js
function ny(e, t, n) {
	return (t.inFlow ?? e.flow ? iy : ry)(e, t, n);
}
function ry({ comment: e, items: t }, n, { blockItemPrefix: r, flowChars: i, itemIndent: a, onChompKeep: o, onComment: s }) {
	let { indent: c, options: { commentString: l } } = n, u = Object.assign({}, n, {
		indent: a,
		type: null
	}), d = !1, f = [];
	for (let e = 0; e < t.length; ++e) {
		let i = t[e], o = null;
		if (q_(i)) !d && i.spaceBefore && f.push(""), ay(n, f, i.commentBefore, d), i.comment && (o = i.comment);
		else if (U_(i)) {
			let e = q_(i.key) ? i.key : null;
			e && (!d && e.spaceBefore && f.push(""), ay(n, f, e.commentBefore, d));
		}
		d = !1;
		let s = Uv(i, u, () => o = null, () => d = !0);
		o && (s += wv(s, a, l(o))), d && o && (d = !1), f.push(r + s);
	}
	let p;
	if (f.length === 0) p = i.start + i.end;
	else {
		p = f[0];
		for (let e = 1; e < f.length; ++e) {
			let t = f[e];
			p += t ? `\n${c}${t}` : "\n";
		}
	}
	return e ? (p += "\n" + Cv(l(e), c), s && s()) : d && o && o(), p;
}
function iy({ items: e }, t, { flowChars: n, itemIndent: r }) {
	let { indent: i, indentStep: a, flowCollectionPadding: o, options: { commentString: s } } = t;
	r += a;
	let c = Object.assign({}, t, {
		indent: r,
		inFlow: !0,
		type: null
	}), l = !1, u = 0, d = [];
	for (let n = 0; n < e.length; ++n) {
		let i = e[n], a = null;
		if (q_(i)) i.spaceBefore && d.push(""), ay(t, d, i.commentBefore, !1), i.comment && (a = i.comment);
		else if (U_(i)) {
			let e = q_(i.key) ? i.key : null;
			e && (e.spaceBefore && d.push(""), ay(t, d, e.commentBefore, !1), e.comment && (l = !0));
			let n = q_(i.value) ? i.value : null;
			n ? (n.comment && (a = n.comment), n.commentBefore && (l = !0)) : i.value == null && e?.comment && (a = e.comment);
		}
		a && (l = !0);
		let o = Uv(i, c, () => a = null);
		l ||= d.length > u || o.includes("\n"), n < e.length - 1 ? o += "," : t.options.trailingComma && (t.options.lineWidth > 0 && (l ||= d.reduce((e, t) => e + t.length + 2, 2) + (o.length + 2) > t.options.lineWidth), l && (o += ",")), a && (o += wv(o, r, s(a))), d.push(o), u = d.length;
	}
	let { start: f, end: p } = n;
	if (d.length === 0) return f + p;
	if (!l) {
		let e = d.reduce((e, t) => e + t.length + 2, 2);
		l = t.options.lineWidth > 0 && e > t.options.lineWidth;
	}
	if (l) {
		let e = f;
		for (let t of d) e += t ? `\n${a}${i}${t}` : "\n";
		return `${e}\n${i}${p}`;
	}
	return `${f}${o}${d.join(" ")}${o}${p}`;
}
function ay({ indent: e, options: { commentString: t } }, n, r, i) {
	if (r && i && (r = r.replace(/^\n+/, "")), r) {
		let i = Cv(t(r), e);
		n.push(i.trimStart());
	}
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/nodes/YAMLMap.js
function oy(e, t) {
	let n = W_(t) ? t.value : t;
	for (let r of e) if (U_(r) && (r.key === t || r.key === n || W_(r.key) && r.key.value === n)) return r;
}
var sy = class extends xv {
	static get tagName() {
		return "tag:yaml.org,2002:map";
	}
	constructor(e) {
		super(F_, e), this.items = [];
	}
	static from(e, t, n) {
		let { keepUndefined: r, replacer: i } = n, a = new this(e), o = (e, o) => {
			if (typeof i == "function") o = i.call(t, e, o);
			else if (Array.isArray(i) && !i.includes(e)) return;
			(o !== void 0 || r) && a.items.push(ey(e, o, n));
		};
		if (t instanceof Map) for (let [e, n] of t) o(e, n);
		else if (t && typeof t == "object") for (let e of Object.keys(t)) o(e, t[e]);
		return typeof e.sortMapEntries == "function" && a.items.sort(e.sortMapEntries), a;
	}
	add(e, t) {
		let n;
		n = U_(e) ? e : !e || typeof e != "object" || !("key" in e) ? new ty(e, e?.value) : new ty(e.key, e.value);
		let r = oy(this.items, n.key), i = this.schema?.sortMapEntries;
		if (r) {
			if (!t) throw Error(`Key ${n.key} already set`);
			W_(r.value) && hv(n.value) ? r.value.value = n.value : r.value = n.value;
		} else if (i) {
			let e = this.items.findIndex((e) => i(n, e) < 0);
			e === -1 ? this.items.push(n) : this.items.splice(e, 0, n);
		} else this.items.push(n);
	}
	delete(e) {
		let t = oy(this.items, e);
		return t ? this.items.splice(this.items.indexOf(t), 1).length > 0 : !1;
	}
	get(e, t) {
		let n = oy(this.items, e)?.value;
		return (!t && W_(n) ? n.value : n) ?? void 0;
	}
	has(e) {
		return !!oy(this.items, e);
	}
	set(e, t) {
		this.add(new ty(e, t), !0);
	}
	toJSON(e, t, n) {
		let r = n ? new n() : t?.mapAsMap ? /* @__PURE__ */ new Map() : {};
		t?.onCreate && t.onCreate(r);
		for (let e of this.items) Qv(t, r, e);
		return r;
	}
	toString(e, t, n) {
		if (!e) return JSON.stringify(this);
		for (let e of this.items) if (!U_(e)) throw Error(`Map items must all be pairs; found ${JSON.stringify(e)} instead`);
		return !e.allNullValues && this.hasAllNullValues(!1) && (e = Object.assign({}, e, { allNullValues: !0 })), ny(this, e, {
			blockItemPrefix: "",
			flowChars: {
				start: "{",
				end: "}"
			},
			itemIndent: e.indent || "",
			onChompKeep: n,
			onComment: t
		});
	}
}, cy = {
	collection: "map",
	default: !0,
	nodeClass: sy,
	tag: "tag:yaml.org,2002:map",
	resolve(e, t) {
		return H_(e) || t("Expected a mapping for this tag"), e;
	},
	createNode: (e, t, n) => sy.from(e, t, n)
}, ly = class extends xv {
	static get tagName() {
		return "tag:yaml.org,2002:seq";
	}
	constructor(e) {
		super(R_, e), this.items = [];
	}
	add(e) {
		this.items.push(e);
	}
	delete(e) {
		let t = uy(e);
		return typeof t == "number" && this.items.splice(t, 1).length > 0;
	}
	get(e, t) {
		let n = uy(e);
		if (typeof n != "number") return;
		let r = this.items[n];
		return !t && W_(r) ? r.value : r;
	}
	has(e) {
		let t = uy(e);
		return typeof t == "number" && t < this.items.length;
	}
	set(e, t) {
		let n = uy(e);
		if (typeof n != "number") throw Error(`Expected a valid index, not ${e}.`);
		let r = this.items[n];
		W_(r) && hv(t) ? r.value = t : this.items[n] = t;
	}
	toJSON(e, t) {
		let n = [];
		t?.onCreate && t.onCreate(n);
		let r = 0;
		for (let e of this.items) n.push(dv(e, String(r++), t));
		return n;
	}
	toString(e, t, n) {
		return e ? ny(this, e, {
			blockItemPrefix: "- ",
			flowChars: {
				start: "[",
				end: "]"
			},
			itemIndent: (e.indent || "") + "  ",
			onChompKeep: n,
			onComment: t
		}) : JSON.stringify(this);
	}
	static from(e, t, n) {
		let { replacer: r } = n, i = new this(e);
		if (t && Symbol.iterator in Object(t)) {
			let e = 0;
			for (let a of t) {
				if (typeof r == "function") {
					let n = t instanceof Set ? a : String(e++);
					a = r.call(t, n, a);
				}
				i.items.push(vv(a, void 0, n));
			}
		}
		return i;
	}
};
function uy(e) {
	let t = W_(e) ? e.value : e;
	return t && typeof t == "string" && (t = Number(t)), typeof t == "number" && Number.isInteger(t) && t >= 0 ? t : null;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/schema/common/seq.js
var dy = {
	collection: "seq",
	default: !0,
	nodeClass: ly,
	tag: "tag:yaml.org,2002:seq",
	resolve(e, t) {
		return G_(e) || t("Expected a sequence for this tag"), e;
	},
	createNode: (e, t, n) => ly.from(e, t, n)
}, fy = {
	identify: (e) => typeof e == "string",
	default: !0,
	tag: "tag:yaml.org,2002:str",
	resolve: (e) => e,
	stringify(e, t, n, r) {
		return t = Object.assign({ actualString: !0 }, t), zv(e, t, n, r);
	}
}, py = {
	identify: (e) => e == null,
	createNode: () => new W(null),
	default: !0,
	tag: "tag:yaml.org,2002:null",
	test: /^(?:~|[Nn]ull|NULL)?$/,
	resolve: () => new W(null),
	stringify: ({ source: e }, t) => typeof e == "string" && py.test.test(e) ? e : t.options.nullStr
}, my = {
	identify: (e) => typeof e == "boolean",
	default: !0,
	tag: "tag:yaml.org,2002:bool",
	test: /^(?:[Tt]rue|TRUE|[Ff]alse|FALSE)$/,
	resolve: (e) => new W(e[0] === "t" || e[0] === "T"),
	stringify({ source: e, value: t }, n) {
		return e && my.test.test(e) && t === (e[0] === "t" || e[0] === "T") ? e : t ? n.options.trueStr : n.options.falseStr;
	}
};
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/stringify/stringifyNumber.js
function hy({ format: e, minFractionDigits: t, tag: n, value: r }) {
	if (typeof r == "bigint") return String(r);
	let i = typeof r == "number" ? r : Number(r);
	if (!isFinite(i)) return isNaN(i) ? ".nan" : i < 0 ? "-.inf" : ".inf";
	let a = Object.is(r, -0) ? "-0" : JSON.stringify(r);
	if (!e && t && (!n || n === "tag:yaml.org,2002:float") && /^-?\d/.test(a) && !a.includes("e")) {
		let e = a.indexOf(".");
		e < 0 && (e = a.length, a += ".");
		let n = t - (a.length - e - 1);
		for (; n-- > 0;) a += "0";
	}
	return a;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/schema/core/float.js
var gy = {
	identify: (e) => typeof e == "number",
	default: !0,
	tag: "tag:yaml.org,2002:float",
	test: /^(?:[-+]?\.(?:inf|Inf|INF)|\.nan|\.NaN|\.NAN)$/,
	resolve: (e) => e.slice(-3).toLowerCase() === "nan" ? NaN : e[0] === "-" ? -Infinity : Infinity,
	stringify: hy
}, _y = {
	identify: (e) => typeof e == "number",
	default: !0,
	tag: "tag:yaml.org,2002:float",
	format: "EXP",
	test: /^[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)[eE][-+]?[0-9]+$/,
	resolve: (e) => parseFloat(e),
	stringify(e) {
		let t = Number(e.value);
		return isFinite(t) ? t.toExponential() : hy(e);
	}
}, vy = {
	identify: (e) => typeof e == "number",
	default: !0,
	tag: "tag:yaml.org,2002:float",
	test: /^[-+]?(?:\.[0-9]+|[0-9]+\.[0-9]*)$/,
	resolve(e) {
		let t = new W(parseFloat(e)), n = e.indexOf(".");
		return n !== -1 && e[e.length - 1] === "0" && (t.minFractionDigits = e.length - n - 1), t;
	},
	stringify: hy
}, yy = (e) => typeof e == "bigint" || Number.isInteger(e), by = (e, t, n, { intAsBigInt: r }) => r ? BigInt(e) : parseInt(e.substring(t), n);
function xy(e, t, n) {
	let { value: r } = e;
	return yy(r) && r >= 0 ? n + r.toString(t) : hy(e);
}
var Sy = {
	identify: (e) => yy(e) && e >= 0,
	default: !0,
	tag: "tag:yaml.org,2002:int",
	format: "OCT",
	test: /^0o[0-7]+$/,
	resolve: (e, t, n) => by(e, 2, 8, n),
	stringify: (e) => xy(e, 8, "0o")
}, Cy = {
	identify: yy,
	default: !0,
	tag: "tag:yaml.org,2002:int",
	test: /^[-+]?[0-9]+$/,
	resolve: (e, t, n) => by(e, 0, 10, n),
	stringify: hy
}, wy = {
	identify: (e) => yy(e) && e >= 0,
	default: !0,
	tag: "tag:yaml.org,2002:int",
	format: "HEX",
	test: /^0x[0-9a-fA-F]+$/,
	resolve: (e, t, n) => by(e, 2, 16, n),
	stringify: (e) => xy(e, 16, "0x")
}, Ty = [
	cy,
	dy,
	fy,
	py,
	my,
	Sy,
	Cy,
	wy,
	gy,
	_y,
	vy
];
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/schema/json/schema.js
function Ey(e) {
	return typeof e == "bigint" || Number.isInteger(e);
}
var Dy = ({ value: e }) => JSON.stringify(e), Oy = [
	{
		identify: (e) => typeof e == "string",
		default: !0,
		tag: "tag:yaml.org,2002:str",
		resolve: (e) => e,
		stringify: Dy
	},
	{
		identify: (e) => e == null,
		createNode: () => new W(null),
		default: !0,
		tag: "tag:yaml.org,2002:null",
		test: /^null$/,
		resolve: () => null,
		stringify: Dy
	},
	{
		identify: (e) => typeof e == "boolean",
		default: !0,
		tag: "tag:yaml.org,2002:bool",
		test: /^true$|^false$/,
		resolve: (e) => e === "true",
		stringify: Dy
	},
	{
		identify: Ey,
		default: !0,
		tag: "tag:yaml.org,2002:int",
		test: /^-?(?:0|[1-9][0-9]*)$/,
		resolve: (e, t, { intAsBigInt: n }) => n ? BigInt(e) : parseInt(e, 10),
		stringify: ({ value: e }) => Ey(e) ? e.toString() : JSON.stringify(e)
	},
	{
		identify: (e) => typeof e == "number",
		default: !0,
		tag: "tag:yaml.org,2002:float",
		test: /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?$/,
		resolve: (e) => parseFloat(e),
		stringify: Dy
	}
], ky = [cy, dy].concat(Oy, {
	default: !0,
	tag: "",
	test: /^/,
	resolve(e, t) {
		return t(`Unresolved plain scalar ${JSON.stringify(e)}`), e;
	}
}), Ay = {
	identify: (e) => e instanceof Uint8Array,
	default: !1,
	tag: "tag:yaml.org,2002:binary",
	resolve(e, t) {
		if (typeof atob == "function") {
			let t = atob(e.replace(/[\n\r]/g, "")), n = new Uint8Array(t.length);
			for (let e = 0; e < t.length; ++e) n[e] = t.charCodeAt(e);
			return n;
		}
		return t("This environment does not support reading binary tags; either Buffer or atob is required"), e;
	},
	stringify({ comment: e, type: t, value: n }, r, i, a) {
		if (!n) return "";
		let o = n, s;
		if (typeof btoa == "function") {
			let e = "";
			for (let t = 0; t < o.length; ++t) e += String.fromCharCode(o[t]);
			s = btoa(e);
		} else throw Error("This environment does not support writing binary tags; either Buffer or btoa is required");
		if (t ??= W.BLOCK_LITERAL, t !== W.QUOTE_DOUBLE) {
			let e = Math.max(r.options.lineWidth - r.indent.length, r.options.minContentWidth), n = Math.ceil(s.length / e), i = Array(n);
			for (let t = 0, r = 0; t < n; ++t, r += e) i[t] = s.substr(r, e);
			s = i.join(t === W.BLOCK_LITERAL ? "\n" : " ");
		}
		return zv({
			comment: e,
			type: t,
			value: s
		}, r, i, a);
	}
};
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/schema/yaml-1.1/pairs.js
function jy(e, t) {
	if (G_(e)) for (let n = 0; n < e.items.length; ++n) {
		let r = e.items[n];
		if (!U_(r)) {
			if (H_(r)) {
				r.items.length > 1 && t("Each pair must have its own sequence indicator");
				let e = r.items[0] || new ty(new W(null));
				if (r.commentBefore && (e.key.commentBefore = e.key.commentBefore ? `${r.commentBefore}\n${e.key.commentBefore}` : r.commentBefore), r.comment) {
					let t = e.value ?? e.key;
					t.comment = t.comment ? `${r.comment}\n${t.comment}` : r.comment;
				}
				r = e;
			}
			e.items[n] = U_(r) ? r : new ty(r);
		}
	}
	else t("Expected a sequence for this tag");
	return e;
}
function My(e, t, n) {
	let { replacer: r } = n, i = new ly(e);
	i.tag = "tag:yaml.org,2002:pairs";
	let a = 0;
	if (t && Symbol.iterator in Object(t)) for (let e of t) {
		typeof r == "function" && (e = r.call(t, String(a++), e));
		let o, s;
		if (Array.isArray(e)) {
			if (e.length === 2) o = e[0], s = e[1];
			else throw TypeError(`Expected [key, value] tuple: ${e}`);
		} else if (e && e instanceof Object) {
			let t = Object.keys(e);
			if (t.length === 1) o = t[0], s = e[o];
			else throw TypeError(`Expected tuple with one key, not ${t.length} keys`);
		} else o = e;
		i.items.push(ey(o, s, n));
	}
	return i;
}
var Ny = {
	collection: "seq",
	default: !1,
	tag: "tag:yaml.org,2002:pairs",
	resolve: jy,
	createNode: My
}, Py = class e extends ly {
	constructor() {
		super(), this.add = sy.prototype.add.bind(this), this.delete = sy.prototype.delete.bind(this), this.get = sy.prototype.get.bind(this), this.has = sy.prototype.has.bind(this), this.set = sy.prototype.set.bind(this), this.tag = e.tag;
	}
	toJSON(e, t) {
		if (!t) return super.toJSON(e);
		let n = /* @__PURE__ */ new Map();
		t?.onCreate && t.onCreate(n);
		for (let e of this.items) {
			let r, i;
			if (U_(e) ? (r = dv(e.key, "", t), i = dv(e.value, r, t)) : r = dv(e, "", t), n.has(r)) throw Error("Ordered maps must not include duplicate keys");
			n.set(r, i);
		}
		return n;
	}
	static from(e, t, n) {
		let r = My(e, t, n), i = new this();
		return i.items = r.items, i;
	}
};
Py.tag = "tag:yaml.org,2002:omap";
var Fy = {
	collection: "seq",
	identify: (e) => e instanceof Map,
	nodeClass: Py,
	default: !1,
	tag: "tag:yaml.org,2002:omap",
	resolve(e, t) {
		let n = jy(e, t), r = [];
		for (let { key: e } of n.items) W_(e) && (r.includes(e.value) ? t(`Ordered maps must not include duplicate keys: ${e.value}`) : r.push(e.value));
		return Object.assign(new Py(), n);
	},
	createNode: (e, t, n) => Py.from(e, t, n)
};
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/schema/yaml-1.1/bool.js
function Iy({ value: e, source: t }, n) {
	return t && (e ? Ly : Ry).test.test(t) ? t : e ? n.options.trueStr : n.options.falseStr;
}
var Ly = {
	identify: (e) => e === !0,
	default: !0,
	tag: "tag:yaml.org,2002:bool",
	test: /^(?:Y|y|[Yy]es|YES|[Tt]rue|TRUE|[Oo]n|ON)$/,
	resolve: () => new W(!0),
	stringify: Iy
}, Ry = {
	identify: (e) => e === !1,
	default: !0,
	tag: "tag:yaml.org,2002:bool",
	test: /^(?:N|n|[Nn]o|NO|[Ff]alse|FALSE|[Oo]ff|OFF)$/,
	resolve: () => new W(!1),
	stringify: Iy
}, zy = {
	identify: (e) => typeof e == "number",
	default: !0,
	tag: "tag:yaml.org,2002:float",
	test: /^(?:[-+]?\.(?:inf|Inf|INF)|\.nan|\.NaN|\.NAN)$/,
	resolve: (e) => e.slice(-3).toLowerCase() === "nan" ? NaN : e[0] === "-" ? -Infinity : Infinity,
	stringify: hy
}, By = {
	identify: (e) => typeof e == "number",
	default: !0,
	tag: "tag:yaml.org,2002:float",
	format: "EXP",
	test: /^[-+]?(?:[0-9][0-9_]*)?(?:\.[0-9_]*)?[eE][-+]?[0-9]+$/,
	resolve: (e) => parseFloat(e.replace(/_/g, "")),
	stringify(e) {
		let t = Number(e.value);
		return isFinite(t) ? t.toExponential() : hy(e);
	}
}, Vy = {
	identify: (e) => typeof e == "number",
	default: !0,
	tag: "tag:yaml.org,2002:float",
	test: /^[-+]?(?:[0-9][0-9_]*)?\.[0-9_]*$/,
	resolve(e) {
		let t = new W(parseFloat(e.replace(/_/g, ""))), n = e.indexOf(".");
		if (n !== -1) {
			let r = e.substring(n + 1).replace(/_/g, "");
			r[r.length - 1] === "0" && (t.minFractionDigits = r.length);
		}
		return t;
	},
	stringify: hy
}, Hy = (e) => typeof e == "bigint" || Number.isInteger(e);
function Uy(e, t, n, { intAsBigInt: r }) {
	let i = e[0];
	if ((i === "-" || i === "+") && (t += 1), e = e.substring(t).replace(/_/g, ""), r) {
		switch (n) {
			case 2:
				e = `0b${e}`;
				break;
			case 8:
				e = `0o${e}`;
				break;
			case 16: e = `0x${e}`;
		}
		let t = BigInt(e);
		return i === "-" ? BigInt(-1) * t : t;
	}
	let a = parseInt(e, n);
	return i === "-" ? -1 * a : a;
}
function Wy(e, t, n) {
	let { value: r } = e;
	if (Hy(r)) {
		let e = r.toString(t);
		return r < 0 ? "-" + n + e.substr(1) : n + e;
	}
	return hy(e);
}
var Gy = {
	identify: Hy,
	default: !0,
	tag: "tag:yaml.org,2002:int",
	format: "BIN",
	test: /^[-+]?0b[0-1_]+$/,
	resolve: (e, t, n) => Uy(e, 2, 2, n),
	stringify: (e) => Wy(e, 2, "0b")
}, Ky = {
	identify: Hy,
	default: !0,
	tag: "tag:yaml.org,2002:int",
	format: "OCT",
	test: /^[-+]?0[0-7_]+$/,
	resolve: (e, t, n) => Uy(e, 1, 8, n),
	stringify: (e) => Wy(e, 8, "0")
}, qy = {
	identify: Hy,
	default: !0,
	tag: "tag:yaml.org,2002:int",
	test: /^[-+]?[0-9][0-9_]*$/,
	resolve: (e, t, n) => Uy(e, 0, 10, n),
	stringify: hy
}, Jy = {
	identify: Hy,
	default: !0,
	tag: "tag:yaml.org,2002:int",
	format: "HEX",
	test: /^[-+]?0x[0-9a-fA-F_]+$/,
	resolve: (e, t, n) => Uy(e, 2, 16, n),
	stringify: (e) => Wy(e, 16, "0x")
}, Yy = class e extends sy {
	constructor(t) {
		super(t), this.tag = e.tag;
	}
	add(e) {
		let t;
		t = U_(e) ? e : e && typeof e == "object" && "key" in e && "value" in e && e.value === null ? new ty(e.key, null) : new ty(e, null), oy(this.items, t.key) || this.items.push(t);
	}
	get(e, t) {
		let n = oy(this.items, e);
		return !t && U_(n) ? W_(n.key) ? n.key.value : n.key : n;
	}
	set(e, t) {
		if (typeof t != "boolean") throw Error(`Expected boolean value for set(key, value) in a YAML set, not ${typeof t}`);
		let n = oy(this.items, e);
		n && !t ? this.items.splice(this.items.indexOf(n), 1) : !n && t && this.items.push(new ty(e));
	}
	toJSON(e, t) {
		return super.toJSON(e, t, Set);
	}
	toString(e, t, n) {
		if (!e) return JSON.stringify(this);
		if (this.hasAllNullValues(!0)) return super.toString(Object.assign({}, e, { allNullValues: !0 }), t, n);
		throw Error("Set items must all have null values");
	}
	static from(e, t, n) {
		let { replacer: r } = n, i = new this(e);
		if (t && Symbol.iterator in Object(t)) for (let e of t) typeof r == "function" && (e = r.call(t, e, e)), i.items.push(ey(e, null, n));
		return i;
	}
};
Yy.tag = "tag:yaml.org,2002:set";
var Xy = {
	collection: "map",
	identify: (e) => e instanceof Set,
	nodeClass: Yy,
	default: !1,
	tag: "tag:yaml.org,2002:set",
	createNode: (e, t, n) => Yy.from(e, t, n),
	resolve(e, t) {
		if (H_(e)) {
			if (e.hasAllNullValues(!0)) return Object.assign(new Yy(), e);
			t("Set items must all have null values");
		} else t("Expected a mapping for this tag");
		return e;
	}
};
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/schema/yaml-1.1/timestamp.js
function Zy(e, t) {
	let n = e[0], r = n === "-" || n === "+" ? e.substring(1) : e, i = (e) => t ? BigInt(e) : Number(e), a = r.replace(/_/g, "").split(":").reduce((e, t) => e * i(60) + i(t), i(0));
	return n === "-" ? i(-1) * a : a;
}
function Qy(e) {
	let { value: t } = e, n = (e) => e;
	if (typeof t == "bigint") n = (e) => BigInt(e);
	else if (isNaN(t) || !isFinite(t)) return hy(e);
	let r = "";
	t < 0 && (r = "-", t *= n(-1));
	let i = n(60), a = [t % i];
	return t < 60 ? a.unshift(0) : (t = (t - a[0]) / i, a.unshift(t % i), t >= 60 && (t = (t - a[0]) / i, a.unshift(t))), r + a.map((e) => String(e).padStart(2, "0")).join(":").replace(/000000\d*$/, "");
}
var $y = {
	identify: (e) => typeof e == "bigint" || Number.isInteger(e),
	default: !0,
	tag: "tag:yaml.org,2002:int",
	format: "TIME",
	test: /^[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+$/,
	resolve: (e, t, { intAsBigInt: n }) => Zy(e, n),
	stringify: Qy
}, eb = {
	identify: (e) => typeof e == "number",
	default: !0,
	tag: "tag:yaml.org,2002:float",
	format: "TIME",
	test: /^[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*$/,
	resolve: (e) => Zy(e, !1),
	stringify: Qy
}, tb = {
	identify: (e) => e instanceof Date,
	default: !0,
	tag: "tag:yaml.org,2002:timestamp",
	test: RegExp("^([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})(?:(?:t|T|[ \\t]+)([0-9]{1,2}):([0-9]{1,2}):([0-9]{1,2}(\\.[0-9]+)?)(?:[ \\t]*(Z|[-+][012]?[0-9](?::[0-9]{2})?))?)?$"),
	resolve(e) {
		let t = e.match(tb.test);
		if (!t) throw Error("!!timestamp expects a date, starting with yyyy-mm-dd");
		let [, n, r, i, a, o, s] = t.map(Number), c = t[7] ? Number((t[7] + "00").substr(1, 3)) : 0, l = Date.UTC(n, r - 1, i, a || 0, o || 0, s || 0, c), u = t[8];
		if (u && u !== "Z") {
			let e = Zy(u, !1);
			Math.abs(e) < 30 && (e *= 60), l -= 6e4 * e;
		}
		return new Date(l);
	},
	stringify: ({ value: e }) => e?.toISOString().replace(/(T00:00:00)?\.000Z$/, "") ?? ""
}, nb = [
	cy,
	dy,
	fy,
	py,
	Ly,
	Ry,
	Gy,
	Ky,
	qy,
	Jy,
	zy,
	By,
	Vy,
	Ay,
	qv,
	Fy,
	Ny,
	Xy,
	$y,
	eb,
	tb
], rb = /* @__PURE__ */ new Map([
	["core", Ty],
	["failsafe", [
		cy,
		dy,
		fy
	]],
	["json", ky],
	["yaml11", nb],
	["yaml-1.1", nb]
]), ib = {
	binary: Ay,
	bool: my,
	float: vy,
	floatExp: _y,
	floatNaN: gy,
	floatTime: eb,
	int: Cy,
	intHex: wy,
	intOct: Sy,
	intTime: $y,
	map: cy,
	merge: qv,
	null: py,
	omap: Fy,
	pairs: Ny,
	seq: dy,
	set: Xy,
	timestamp: tb
}, ab = {
	"tag:yaml.org,2002:binary": Ay,
	"tag:yaml.org,2002:merge": qv,
	"tag:yaml.org,2002:omap": Fy,
	"tag:yaml.org,2002:pairs": Ny,
	"tag:yaml.org,2002:set": Xy,
	"tag:yaml.org,2002:timestamp": tb
};
function ob(e, t, n) {
	let r = rb.get(t);
	if (r && !e) return n && !r.includes(qv) ? r.concat(qv) : r.slice();
	let i = r;
	if (!i) {
		if (Array.isArray(e)) i = [];
		else {
			let e = Array.from(rb.keys()).filter((e) => e !== "yaml11").map((e) => JSON.stringify(e)).join(", ");
			throw Error(`Unknown schema "${t}"; use one of ${e} or define customTags array`);
		}
	}
	if (Array.isArray(e)) for (let t of e) i = i.concat(t);
	else typeof e == "function" && (i = e(i.slice()));
	return n && (i = i.concat(qv)), i.reduce((e, t) => {
		let n = typeof t == "string" ? ib[t] : t;
		if (!n) {
			let e = JSON.stringify(t), n = Object.keys(ib).map((e) => JSON.stringify(e)).join(", ");
			throw Error(`Unknown custom tag ${e}; use one of ${n}`);
		}
		return e.includes(n) || e.push(n), e;
	}, []);
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/schema/Schema.js
var sb = (e, t) => e.key < t.key ? -1 : +(e.key > t.key), cb = class e {
	constructor({ compat: e, customTags: t, merge: n, resolveKnownTags: r, schema: i, sortMapEntries: a, toStringDefaults: o }) {
		this.compat = Array.isArray(e) ? ob(e, "compat") : e ? ob(null, e) : null, this.name = typeof i == "string" && i || "core", this.knownTags = r ? ab : {}, this.tags = ob(t, this.name, n), this.toStringOptions = o ?? null, Object.defineProperty(this, F_, { value: cy }), Object.defineProperty(this, L_, { value: fy }), Object.defineProperty(this, R_, { value: dy }), this.sortMapEntries = typeof a == "function" ? a : a === !0 ? sb : null;
	}
	clone() {
		let t = Object.create(e.prototype, Object.getOwnPropertyDescriptors(this));
		return t.tags = this.tags.slice(), t;
	}
};
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/stringify/stringifyDocument.js
function lb(e, t) {
	let n = [], r = t.directives === !0;
	if (t.directives !== !1 && e.directives) {
		let t = e.directives.toString(e);
		t ? (n.push(t), r = !0) : e.directives.docStart && (r = !0);
	}
	r && n.push("---");
	let i = Bv(e, t), { commentString: a } = i.options;
	if (e.commentBefore) {
		n.length !== 1 && n.unshift("");
		let t = a(e.commentBefore);
		n.unshift(Cv(t, ""));
	}
	let o = !1, s = null;
	if (e.contents) {
		if (q_(e.contents)) {
			if (e.contents.spaceBefore && r && n.push(""), e.contents.commentBefore) {
				let t = a(e.contents.commentBefore);
				n.push(Cv(t, ""));
			}
			i.forceBlockIndent = !!e.comment, s = e.contents.comment;
		}
		let t = s ? void 0 : () => o = !0, c = Uv(e.contents, i, () => s = null, t);
		s && (c += wv(c, "", a(s))), (c[0] === "|" || c[0] === ">") && n[n.length - 1] === "---" ? n[n.length - 1] = `--- ${c}` : n.push(c);
	} else n.push(Uv(e.contents, i));
	if (e.directives?.docEnd) {
		if (e.comment) {
			let t = a(e.comment);
			t.includes("\n") ? (n.push("..."), n.push(Cv(t, ""))) : n.push(`... ${t}`);
		} else n.push("...");
	} else {
		let t = e.comment;
		t && o && (t = t.replace(/^\n+/, "")), t && ((!o || s) && n[n.length - 1] !== "" && n.push(""), n.push(Cv(a(t), "")));
	}
	return n.join("\n") + "\n";
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/doc/Document.js
var ub = class e {
	constructor(e, t, n) {
		this.commentBefore = null, this.comment = null, this.errors = [], this.warnings = [], Object.defineProperty(this, z_, { value: P_ });
		let r = null;
		typeof t == "function" || Array.isArray(t) ? r = t : n === void 0 && t && (n = t, t = void 0);
		let i = Object.assign({
			intAsBigInt: !1,
			keepSourceTokens: !1,
			logLevel: "warn",
			prettyErrors: !0,
			strict: !0,
			stringKeys: !1,
			uniqueKeys: !0,
			version: "1.2"
		}, n);
		this.options = i;
		let { version: a } = i;
		n?._directives ? (this.directives = n._directives.atDocument(), this.directives.yaml.explicit && (a = this.directives.yaml.version)) : this.directives = new av({ version: a }), this.setSchema(a, n), this.contents = e === void 0 ? null : this.createNode(e, r, n);
	}
	clone() {
		let t = Object.create(e.prototype, { [z_]: { value: P_ } });
		return t.commentBefore = this.commentBefore, t.comment = this.comment, t.errors = this.errors.slice(), t.warnings = this.warnings.slice(), t.options = Object.assign({}, this.options), this.directives && (t.directives = this.directives.clone()), t.schema = this.schema.clone(), t.contents = q_(this.contents) ? this.contents.clone(t.schema) : this.contents, this.range && (t.range = this.range.slice()), t;
	}
	add(e) {
		db(this.contents) && this.contents.add(e);
	}
	addIn(e, t) {
		db(this.contents) && this.contents.addIn(e, t);
	}
	createAlias(e, t) {
		if (!e.anchor) {
			let n = sv(this);
			e.anchor = !t || n.has(t) ? cv(t || "a", n) : t;
		}
		return new pv(e.anchor);
	}
	createNode(e, t, n) {
		let r;
		if (typeof t == "function") e = t.call({ "": e }, "", e), r = t;
		else if (Array.isArray(t)) {
			let e = t.filter((e) => typeof e == "number" || e instanceof String || e instanceof Number).map(String);
			e.length > 0 && (t = t.concat(e)), r = t;
		} else n === void 0 && t && (n = t, t = void 0);
		let { aliasDuplicateObjects: i, anchorPrefix: a, flow: o, keepUndefined: s, onTagObj: c, tag: l } = n ?? {}, { onAnchor: u, setAnchors: d, sourceObjects: f } = lv(this, a || "a"), p = {
			aliasDuplicateObjects: i ?? !0,
			keepUndefined: s ?? !1,
			onAnchor: u,
			onTagObj: c,
			replacer: r,
			schema: this.schema,
			sourceObjects: f
		}, m = vv(e, l, p);
		return o && K_(m) && (m.flow = !0), d(), m;
	}
	createPair(e, t, n = {}) {
		return new ty(this.createNode(e, null, n), this.createNode(t, null, n));
	}
	delete(e) {
		return db(this.contents) ? this.contents.delete(e) : !1;
	}
	deleteIn(e) {
		return bv(e) ? this.contents != null && (this.contents = null, !0) : db(this.contents) ? this.contents.deleteIn(e) : !1;
	}
	get(e, t) {
		return K_(this.contents) ? this.contents.get(e, t) : void 0;
	}
	getIn(e, t) {
		return bv(e) ? !t && W_(this.contents) ? this.contents.value : this.contents : K_(this.contents) ? this.contents.getIn(e, t) : void 0;
	}
	has(e) {
		return K_(this.contents) ? this.contents.has(e) : !1;
	}
	hasIn(e) {
		return bv(e) ? this.contents !== void 0 : K_(this.contents) ? this.contents.hasIn(e) : !1;
	}
	set(e, t) {
		this.contents == null ? this.contents = yv(this.schema, [e], t) : db(this.contents) && this.contents.set(e, t);
	}
	setIn(e, t) {
		bv(e) ? this.contents = t : this.contents == null ? this.contents = yv(this.schema, Array.from(e), t) : db(this.contents) && this.contents.setIn(e, t);
	}
	setSchema(e, t = {}) {
		typeof e == "number" && (e = String(e));
		let n;
		switch (e) {
			case "1.1":
				this.directives ? this.directives.yaml.version = "1.1" : this.directives = new av({ version: "1.1" }), n = {
					resolveKnownTags: !1,
					schema: "yaml-1.1"
				};
				break;
			case "1.2":
			case "next":
				this.directives ? this.directives.yaml.version = e : this.directives = new av({ version: e }), n = {
					resolveKnownTags: !0,
					schema: "core"
				};
				break;
			case null:
				this.directives && delete this.directives, n = null;
				break;
			default: {
				let t = JSON.stringify(e);
				throw Error(`Expected '1.1', '1.2' or null as first argument, but found: ${t}`);
			}
		}
		if (t.schema instanceof Object) this.schema = t.schema;
		else if (n) this.schema = new cb(Object.assign(n, t));
		else throw Error("With a null YAML version, the { schema: Schema } option is required");
	}
	toJS({ json: e, jsonArg: t, mapAsMap: n, maxAliasCount: r, onAnchor: i, reviver: a } = {}) {
		let o = {
			anchors: /* @__PURE__ */ new Map(),
			doc: this,
			keep: !e,
			mapAsMap: n === !0,
			mapKeyWarned: !1,
			maxAliasCount: typeof r == "number" ? r : 100
		}, s = dv(this.contents, t ?? "", o);
		if (typeof i == "function") for (let { count: e, res: t } of o.anchors.values()) i(t, e);
		return typeof a == "function" ? uv(a, { "": s }, "", s) : s;
	}
	toJSON(e, t) {
		return this.toJS({
			json: !0,
			jsonArg: e,
			mapAsMap: !1,
			onAnchor: t
		});
	}
	toString(e = {}) {
		if (this.errors.length > 0) throw Error("Document with errors cannot be stringified");
		if ("indent" in e && (!Number.isInteger(e.indent) || Number(e.indent) <= 0)) {
			let t = JSON.stringify(e.indent);
			throw Error(`"indent" option must be a positive integer, not ${t}`);
		}
		return lb(this, e);
	}
};
function db(e) {
	if (K_(e)) return !0;
	throw Error("Expected a YAML collection as document contents");
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/errors.js
var fb = class extends Error {
	constructor(e, t, n, r) {
		super(), this.name = e, this.code = n, this.message = r, this.pos = t;
	}
}, pb = class extends fb {
	constructor(e, t, n) {
		super("YAMLParseError", e, t, n);
	}
}, mb = class extends fb {
	constructor(e, t, n) {
		super("YAMLWarning", e, t, n);
	}
}, hb = (e, t) => (n) => {
	if (n.pos[0] === -1) return;
	n.linePos = n.pos.map((e) => t.linePos(e));
	let { line: r, col: i } = n.linePos[0];
	n.message += ` at line ${r}, column ${i}`;
	let a = i - 1, o = e.substring(t.lineStarts[r - 1], t.lineStarts[r]).replace(/[\n\r]+$/, "");
	if (a >= 60 && o.length > 80) {
		let e = Math.min(a - 39, o.length - 79);
		o = "…" + o.substring(e), a -= e - 1;
	}
	if (o.length > 80 && (o = o.substring(0, 79) + "…"), r > 1 && /^ *$/.test(o.substring(0, a))) {
		let n = e.substring(t.lineStarts[r - 2], t.lineStarts[r - 1]);
		n.length > 80 && (n = n.substring(0, 79) + "…\n"), o = n + o;
	}
	if (/[^ ]/.test(o)) {
		let e = 1, t = n.linePos[1];
		t?.line === r && t.col > i && (e = Math.max(1, Math.min(t.col - i, 80 - a)));
		let s = " ".repeat(a) + "^".repeat(e);
		n.message += `:\n\n${o}\n${s}\n`;
	}
};
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/resolve-props.js
function gb(e, { flow: t, indicator: n, next: r, offset: i, onError: a, parentIndent: o, startOnNewline: s }) {
	let c = !1, l = s, u = s, d = "", f = "", p = !1, m = !1, h = null, g = null, _ = null, v = null, y = null, b = null, x = null;
	for (let i of e) switch (m &&= (i.type !== "space" && i.type !== "newline" && i.type !== "comma" && a(i.offset, "MISSING_CHAR", "Tags and anchors must be separated from the next token by white space"), !1), h &&= (l && i.type !== "comment" && i.type !== "newline" && a(h, "TAB_AS_INDENT", "Tabs are not allowed as indentation"), null), i.type) {
		case "space":
			!t && (n !== "doc-start" || r?.type !== "flow-collection") && i.source.includes("	") && (h = i), u = !0;
			break;
		case "comment": {
			u || a(i, "MISSING_CHAR", "Comments must be separated from other tokens by white space characters");
			let e = i.source.substring(1) || " ";
			d ? d += f + e : d = e, f = "", l = !1;
			break;
		}
		case "newline":
			l ? d ? d += i.source : (!b || n !== "seq-item-ind") && (c = !0) : f += i.source, l = !0, p = !0, (g || _) && (v = i), u = !0;
			break;
		case "anchor":
			g && a(i, "MULTIPLE_ANCHORS", "A node can have at most one anchor"), i.source.endsWith(":") && a(i.offset + i.source.length - 1, "BAD_ALIAS", "Anchor ending in : is ambiguous", !0), g = i, x ??= i.offset, l = !1, u = !1, m = !0;
			break;
		case "tag":
			_ && a(i, "MULTIPLE_TAGS", "A node can have at most one tag"), _ = i, x ??= i.offset, l = !1, u = !1, m = !0;
			break;
		case n:
			(g || _) && a(i, "BAD_PROP_ORDER", `Anchors and tags must be after the ${i.source} indicator`), b && a(i, "UNEXPECTED_TOKEN", `Unexpected ${i.source} in ${t ?? "collection"}`), b = i, l = n === "seq-item-ind" || n === "explicit-key-ind", u = !1;
			break;
		case "comma": if (t) {
			y && a(i, "UNEXPECTED_TOKEN", `Unexpected , in ${t}`), y = i, l = !1, u = !1;
			break;
		}
		default: a(i, "UNEXPECTED_TOKEN", `Unexpected ${i.type} token`), l = !1, u = !1;
	}
	let S = e[e.length - 1], C = S ? S.offset + S.source.length : i;
	return m && r && r.type !== "space" && r.type !== "newline" && r.type !== "comma" && (r.type !== "scalar" || r.source !== "") && a(r.offset, "MISSING_CHAR", "Tags and anchors must be separated from the next token by white space"), h && (l && h.indent <= o || r?.type === "block-map" || r?.type === "block-seq") && a(h, "TAB_AS_INDENT", "Tabs are not allowed as indentation"), {
		comma: y,
		found: b,
		spaceBefore: c,
		comment: d,
		hasNewline: p,
		anchor: g,
		tag: _,
		newlineAfterProp: v,
		end: C,
		start: x ?? C
	};
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/util-contains-newline.js
function _b(e) {
	if (!e) return null;
	switch (e.type) {
		case "alias":
		case "scalar":
		case "double-quoted-scalar":
		case "single-quoted-scalar":
			if (e.source.includes("\n")) return !0;
			if (e.end) {
				for (let t of e.end) if (t.type === "newline") return !0;
			}
			return !1;
		case "flow-collection":
			for (let t of e.items) {
				for (let e of t.start) if (e.type === "newline") return !0;
				if (t.sep) {
					for (let e of t.sep) if (e.type === "newline") return !0;
				}
				if (_b(t.key) || _b(t.value)) return !0;
			}
			return !1;
		default: return !0;
	}
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/util-flow-indent-check.js
function vb(e, t, n) {
	if (t?.type === "flow-collection") {
		let r = t.end[0];
		r.indent === e && (r.source === "]" || r.source === "}") && _b(t) && n(r, "BAD_INDENT", "Flow end indicator should be more indented than parent", !0);
	}
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/util-map-includes.js
function yb(e, t, n) {
	let { uniqueKeys: r } = e.options;
	if (r === !1) return !1;
	let i = typeof r == "function" ? r : (e, t) => e === t || W_(e) && W_(t) && e.value === t.value;
	return t.some((e) => i(e.key, n));
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/resolve-block-map.js
var bb = "All mapping items must start at the same column";
function xb({ composeNode: e, composeEmptyNode: t }, n, r, i, a) {
	let o = new ((a?.nodeClass) ?? sy)(n.schema);
	n.atRoot &&= !1;
	let s = r.offset, c = null;
	for (let a of r.items) {
		let { start: l, key: u, sep: d, value: f } = a, p = gb(l, {
			indicator: "explicit-key-ind",
			next: u ?? d?.[0],
			offset: s,
			onError: i,
			parentIndent: r.indent,
			startOnNewline: !0
		}), m = !p.found;
		if (m) {
			if (u && (u.type === "block-seq" ? i(s, "BLOCK_AS_IMPLICIT_KEY", "A block sequence may not be used as an implicit map key") : "indent" in u && u.indent !== r.indent && i(s, "BAD_INDENT", bb)), !p.anchor && !p.tag && !d) {
				c = p.end, p.comment && (o.comment ? o.comment += "\n" + p.comment : o.comment = p.comment);
				continue;
			}
			(p.newlineAfterProp || _b(u)) && i(u ?? l[l.length - 1], "MULTILINE_IMPLICIT_KEY", "Implicit keys need to be on a single line");
		} else p.found?.indent !== r.indent && i(s, "BAD_INDENT", bb);
		n.atKey = !0;
		let h = p.end, g = u ? e(n, u, p, i) : t(n, h, l, null, p, i);
		n.schema.compat && vb(r.indent, u, i), n.atKey = !1, yb(n, o.items, g) && i(h, "DUPLICATE_KEY", "Map keys must be unique");
		let _ = gb(d ?? [], {
			indicator: "map-value-ind",
			next: f,
			offset: g.range[2],
			onError: i,
			parentIndent: r.indent,
			startOnNewline: !u || u.type === "block-scalar"
		});
		if (s = _.end, _.found) {
			m && (f?.type === "block-map" && !_.hasNewline && i(s, "BLOCK_AS_IMPLICIT_KEY", "Nested mappings are not allowed in compact mappings"), n.options.strict && p.start < _.found.offset - 1024 && i(g.range, "KEY_OVER_1024_CHARS", "The : indicator must be at most 1024 chars after the start of an implicit block mapping key"));
			let c = f ? e(n, f, _, i) : t(n, s, d, null, _, i);
			n.schema.compat && vb(r.indent, f, i), s = c.range[2];
			let l = new ty(g, c);
			n.options.keepSourceTokens && (l.srcToken = a), o.items.push(l);
		} else {
			m && i(g.range, "MISSING_CHAR", "Implicit map keys need to be followed by map values"), _.comment && (g.comment ? g.comment += "\n" + _.comment : g.comment = _.comment);
			let e = new ty(g);
			n.options.keepSourceTokens && (e.srcToken = a), o.items.push(e);
		}
	}
	return c && c < s && i(c, "IMPOSSIBLE", "Map comment with trailing content"), o.range = [
		r.offset,
		s,
		c ?? s
	], o;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/resolve-block-seq.js
function Sb({ composeNode: e, composeEmptyNode: t }, n, r, i, a) {
	let o = new ((a?.nodeClass) ?? ly)(n.schema);
	n.atRoot &&= !1, n.atKey &&= !1;
	let s = r.offset, c = null;
	for (let { start: a, value: l } of r.items) {
		let u = gb(a, {
			indicator: "seq-item-ind",
			next: l,
			offset: s,
			onError: i,
			parentIndent: r.indent,
			startOnNewline: !0
		});
		if (!u.found) {
			if (u.anchor || u.tag || l) l?.type === "block-seq" ? i(u.end, "BAD_INDENT", "All sequence items must start at the same column") : i(s, "MISSING_CHAR", "Sequence item without - indicator");
			else {
				c = u.end, u.comment && (o.comment = u.comment);
				continue;
			}
		}
		let d = l ? e(n, l, u, i) : t(n, u.end, a, null, u, i);
		n.schema.compat && vb(r.indent, l, i), s = d.range[2], o.items.push(d);
	}
	return o.range = [
		r.offset,
		s,
		c ?? s
	], o;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/resolve-end.js
function Cb(e, t, n, r) {
	let i = "";
	if (e) {
		let a = !1, o = "";
		for (let s of e) {
			let { source: e, type: c } = s;
			switch (c) {
				case "space":
					a = !0;
					break;
				case "comment": {
					n && !a && r(s, "MISSING_CHAR", "Comments must be separated from other tokens by white space characters");
					let t = e.substring(1) || " ";
					i ? i += o + t : i = t, o = "";
					break;
				}
				case "newline":
					i && (o += e), a = !0;
					break;
				default: r(s, "UNEXPECTED_TOKEN", `Unexpected ${c} at node end`);
			}
			t += e.length;
		}
	}
	return {
		comment: i,
		offset: t
	};
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/resolve-flow-collection.js
var wb = "Block collections are not allowed within flow collections", Tb = (e) => e && (e.type === "block-map" || e.type === "block-seq");
function Eb({ composeNode: e, composeEmptyNode: t }, n, r, i, a) {
	let o = r.start.source === "{", s = o ? "flow map" : "flow sequence", c = new ((a?.nodeClass) ?? (o ? sy : ly))(n.schema);
	c.flow = !0;
	let l = n.atRoot;
	l && (n.atRoot = !1), n.atKey &&= !1;
	let u = r.offset + r.start.source.length;
	for (let a = 0; a < r.items.length; ++a) {
		let l = r.items[a], { start: d, key: f, sep: p, value: m } = l, h = gb(d, {
			flow: s,
			indicator: "explicit-key-ind",
			next: f ?? p?.[0],
			offset: u,
			onError: i,
			parentIndent: r.indent,
			startOnNewline: !1
		});
		if (!h.found) {
			if (!h.anchor && !h.tag && !p && !m) {
				a === 0 && h.comma ? i(h.comma, "UNEXPECTED_TOKEN", `Unexpected , in ${s}`) : a < r.items.length - 1 && i(h.start, "UNEXPECTED_TOKEN", `Unexpected empty item in ${s}`), h.comment && (c.comment ? c.comment += "\n" + h.comment : c.comment = h.comment), u = h.end;
				continue;
			}
			!o && n.options.strict && _b(f) && i(f, "MULTILINE_IMPLICIT_KEY", "Implicit keys of flow sequence pairs need to be on a single line");
		}
		if (a === 0) h.comma && i(h.comma, "UNEXPECTED_TOKEN", `Unexpected , in ${s}`);
		else if (h.comma || i(h.start, "MISSING_CHAR", `Missing , between ${s} items`), h.comment) {
			let e = "";
			loop: for (let t of d) switch (t.type) {
				case "comma":
				case "space": break;
				case "comment":
					e = t.source.substring(1);
					break loop;
				default: break loop;
			}
			if (e) {
				let t = c.items[c.items.length - 1];
				U_(t) && (t = t.value ?? t.key), t.comment ? t.comment += "\n" + e : t.comment = e, h.comment = h.comment.substring(e.length + 1);
			}
		}
		if (!o && !p && !h.found) {
			let r = m ? e(n, m, h, i) : t(n, h.end, p, null, h, i);
			c.items.push(r), u = r.range[2], Tb(m) && i(r.range, "BLOCK_IN_FLOW", wb);
		} else {
			n.atKey = !0;
			let a = h.end, g = f ? e(n, f, h, i) : t(n, a, d, null, h, i);
			Tb(f) && i(g.range, "BLOCK_IN_FLOW", wb), n.atKey = !1;
			let _ = gb(p ?? [], {
				flow: s,
				indicator: "map-value-ind",
				next: m,
				offset: g.range[2],
				onError: i,
				parentIndent: r.indent,
				startOnNewline: !1
			});
			if (_.found) {
				if (!o && !h.found && n.options.strict) {
					if (p) for (let e of p) {
						if (e === _.found) break;
						if (e.type === "newline") {
							i(e, "MULTILINE_IMPLICIT_KEY", "Implicit keys of flow sequence pairs need to be on a single line");
							break;
						}
					}
					h.start < _.found.offset - 1024 && i(_.found, "KEY_OVER_1024_CHARS", "The : indicator must be at most 1024 chars after the start of an implicit flow sequence key");
				}
			} else m && ("source" in m && m.source?.[0] === ":" ? i(m, "MISSING_CHAR", `Missing space after : in ${s}`) : i(_.start, "MISSING_CHAR", `Missing , or : between ${s} items`));
			let v = m ? e(n, m, _, i) : _.found ? t(n, _.end, p, null, _, i) : null;
			v ? Tb(m) && i(v.range, "BLOCK_IN_FLOW", wb) : _.comment && (g.comment ? g.comment += "\n" + _.comment : g.comment = _.comment);
			let y = new ty(g, v);
			if (n.options.keepSourceTokens && (y.srcToken = l), o) {
				let e = c;
				yb(n, e.items, g) && i(a, "DUPLICATE_KEY", "Map keys must be unique"), e.items.push(y);
			} else {
				let e = new sy(n.schema);
				e.flow = !0, e.items.push(y);
				let t = (v ?? g).range;
				e.range = [
					g.range[0],
					t[1],
					t[2]
				], c.items.push(e);
			}
			u = v ? v.range[2] : _.end;
		}
	}
	let d = o ? "}" : "]", [f, ...p] = r.end, m = u;
	if (f?.source === d) m = f.offset + f.source.length;
	else {
		let e = s[0].toUpperCase() + s.substring(1), t = l ? `${e} must end with a ${d}` : `${e} in block collection must be sufficiently indented and end with a ${d}`;
		i(u, l ? "MISSING_CHAR" : "BAD_INDENT", t), f && f.source.length !== 1 && p.unshift(f);
	}
	if (p.length > 0) {
		let e = Cb(p, m, n.options.strict, i);
		e.comment && (c.comment ? c.comment += "\n" + e.comment : c.comment = e.comment), c.range = [
			r.offset,
			m,
			e.offset
		];
	} else c.range = [
		r.offset,
		m,
		m
	];
	return c;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/compose-collection.js
function Db(e, t, n, r, i, a) {
	let o = n.type === "block-map" ? xb(e, t, n, r, a) : n.type === "block-seq" ? Sb(e, t, n, r, a) : Eb(e, t, n, r, a), s = o.constructor;
	return i === "!" || i === s.tagName ? (o.tag = s.tagName, o) : (i && (o.tag = i), o);
}
function Ob(e, t, n, r, i) {
	let a = r.tag, o = a ? t.directives.tagName(a.source, (e) => i(a, "TAG_RESOLVE_FAILED", e)) : null;
	if (n.type === "block-seq") {
		let { anchor: e, newlineAfterProp: t } = r, n = e && a ? e.offset > a.offset ? e : a : e ?? a;
		n && (!t || t.offset < n.offset) && i(n, "MISSING_CHAR", "Missing newline after block sequence props");
	}
	let s = n.type === "block-map" ? "map" : n.type === "block-seq" ? "seq" : n.start.source === "{" ? "map" : "seq";
	if (!a || !o || o === "!" || o === sy.tagName && s === "map" || o === ly.tagName && s === "seq") return Db(e, t, n, i, o);
	let c = t.schema.tags.find((e) => e.tag === o && e.collection === s);
	if (!c) {
		let r = t.schema.knownTags[o];
		if (r?.collection === s) t.schema.tags.push(Object.assign({}, r, { default: !1 })), c = r;
		else return r ? i(a, "BAD_COLLECTION_TYPE", `${r.tag} used for ${s} collection, but expects ${r.collection ?? "scalar"}`, !0) : i(a, "TAG_RESOLVE_FAILED", `Unresolved tag: ${o}`, !0), Db(e, t, n, i, o);
	}
	let l = Db(e, t, n, i, o, c), u = c.resolve?.(l, (e) => i(a, "TAG_RESOLVE_FAILED", e), t.options) ?? l, d = q_(u) ? u : new W(u);
	return d.range = l.range, d.tag = o, c?.format && (d.format = c.format), d;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/resolve-block-scalar.js
function kb(e, t, n) {
	let r = t.offset, i = Ab(t, e.options.strict, n);
	if (!i) return {
		value: "",
		type: null,
		comment: "",
		range: [
			r,
			r,
			r
		]
	};
	let a = i.mode === ">" ? W.BLOCK_FOLDED : W.BLOCK_LITERAL, o = t.source ? jb(t.source) : [], s = o.length;
	for (let e = o.length - 1; e >= 0; --e) {
		let t = o[e][1];
		if (t === "" || t === "\r") s = e;
		else break;
	}
	if (s === 0) {
		let e = i.chomp === "+" && o.length > 0 ? "\n".repeat(Math.max(1, o.length - 1)) : "", n = r + i.length;
		return t.source && (n += t.source.length), {
			value: e,
			type: a,
			comment: i.comment,
			range: [
				r,
				n,
				n
			]
		};
	}
	let c = t.indent + i.indent, l = t.offset + i.length, u = 0;
	for (let t = 0; t < s; ++t) {
		let [r, a] = o[t];
		if (a === "" || a === "\r") i.indent === 0 && r.length > c && (c = r.length);
		else {
			r.length < c && n(l + r.length, "MISSING_CHAR", "Block scalars with more-indented leading empty lines must use an explicit indentation indicator"), i.indent === 0 && (c = r.length), u = t, c === 0 && !e.atRoot && n(l, "BAD_INDENT", "Block scalar values in collections must be indented");
			break;
		}
		l += r.length + a.length + 1;
	}
	for (let e = o.length - 1; e >= s; --e) o[e][0].length > c && (s = e + 1);
	let d = "", f = "", p = !1;
	for (let e = 0; e < u; ++e) d += o[e][0].slice(c) + "\n";
	for (let e = u; e < s; ++e) {
		let [t, r] = o[e];
		l += t.length + r.length + 1;
		let s = r[r.length - 1] === "\r";
		/* istanbul ignore if already caught in lexer */
		if (s && (r = r.slice(0, -1)), r && t.length < c) {
			let e = `Block scalar lines must not be less indented than their ${i.indent ? "explicit indentation indicator" : "first line"}`;
			n(l - r.length - (s ? 2 : 1), "BAD_INDENT", e), t = "";
		}
		a === W.BLOCK_LITERAL ? (d += f + t.slice(c) + r, f = "\n") : t.length > c || r[0] === "	" ? (f === " " ? f = "\n" : !p && f === "\n" && (f = "\n\n"), d += f + t.slice(c) + r, f = "\n", p = !0) : r === "" ? f === "\n" ? d += "\n" : f = "\n" : (d += f + r, f = " ", p = !1);
	}
	switch (i.chomp) {
		case "-": break;
		case "+":
			for (let e = s; e < o.length; ++e) d += "\n" + o[e][0].slice(c);
			d[d.length - 1] !== "\n" && (d += "\n");
			break;
		default: d += "\n";
	}
	let m = r + i.length + t.source.length;
	return {
		value: d,
		type: a,
		comment: i.comment,
		range: [
			r,
			m,
			m
		]
	};
}
function Ab({ offset: e, props: t }, n, r) {
	/* istanbul ignore if should not happen */
	if (t[0].type !== "block-scalar-header") return r(t[0], "IMPOSSIBLE", "Block scalar header not found"), null;
	let { source: i } = t[0], a = i[0], o = 0, s = "", c = -1;
	for (let t = 1; t < i.length; ++t) {
		let n = i[t];
		if (!s && (n === "-" || n === "+")) s = n;
		else {
			let r = Number(n);
			!o && r ? o = r : c === -1 && (c = e + t);
		}
	}
	c !== -1 && r(c, "UNEXPECTED_TOKEN", `Block scalar header includes extra characters: ${i}`);
	let l = !1, u = "", d = i.length;
	for (let e = 1; e < t.length; ++e) {
		let i = t[e];
		switch (i.type) {
			case "space": l = !0;
			case "newline":
				d += i.source.length;
				break;
			case "comment":
				n && !l && r(i, "MISSING_CHAR", "Comments must be separated from other tokens by white space characters"), d += i.source.length, u = i.source.substring(1);
				break;
			case "error":
				r(i, "UNEXPECTED_TOKEN", i.message), d += i.source.length;
				break;
			/* istanbul ignore next should not happen */
			default: {
				r(i, "UNEXPECTED_TOKEN", `Unexpected token in block scalar header: ${i.type}`);
				let e = i.source;
				e && typeof e == "string" && (d += e.length);
			}
		}
	}
	return {
		mode: a,
		indent: o,
		chomp: s,
		comment: u,
		length: d
	};
}
function jb(e) {
	let t = e.split(/\n( *)/), n = t[0], r = n.match(/^( *)/), i = [r?.[1] ? [r[1], n.slice(r[1].length)] : ["", n]];
	for (let e = 1; e < t.length; e += 2) i.push([t[e], t[e + 1]]);
	return i;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/resolve-flow-scalar.js
function Mb(e, t, n) {
	let { offset: r, type: i, source: a, end: o } = e, s, c, l = (e, t, i) => n(r + e, t, i);
	switch (i) {
		case "scalar":
			s = W.PLAIN, c = Nb(a, l);
			break;
		case "single-quoted-scalar":
			s = W.QUOTE_SINGLE, c = Pb(a, l);
			break;
		case "double-quoted-scalar":
			s = W.QUOTE_DOUBLE, c = Ib(a, l);
			break;
		/* istanbul ignore next should not happen */
		default: return n(e, "UNEXPECTED_TOKEN", `Expected a flow scalar value, but found: ${i}`), {
			value: "",
			type: null,
			comment: "",
			range: [
				r,
				r + a.length,
				r + a.length
			]
		};
	}
	let u = r + a.length, d = Cb(o, u, t, n);
	return {
		value: c,
		type: s,
		comment: d.comment,
		range: [
			r,
			u,
			d.offset
		]
	};
}
function Nb(e, t) {
	let n = "";
	switch (e[0]) {
		/* istanbul ignore next should not happen */
		case "	":
			n = "a tab character";
			break;
		case ",":
			n = "flow indicator character ,";
			break;
		case "%":
			n = "directive indicator character %";
			break;
		case "|":
		case ">":
			n = `block scalar indicator ${e[0]}`;
			break;
		case "@":
		case "`": n = `reserved character ${e[0]}`;
	}
	return n && t(0, "BAD_SCALAR_START", `Plain value cannot start with ${n}`), Fb(e);
}
function Pb(e, t) {
	return (e[e.length - 1] !== "'" || e.length === 1) && t(e.length, "MISSING_CHAR", "Missing closing 'quote"), Fb(e.slice(1, -1)).replace(/''/g, "'");
}
function Fb(e) {
	let t, n;
	try {
		t = /* @__PURE__ */ RegExp("(.*?)(?<![ 	])[ 	]*\r?\n", "sy"), n = /* @__PURE__ */ RegExp("[ 	]*(.*?)(?:(?<![ 	])[ 	]*)?\r?\n", "sy");
	} catch {
		t = /(.*?)[ \t]*\r?\n/sy, n = /[ \t]*(.*?)[ \t]*\r?\n/sy;
	}
	let r = t.exec(e);
	if (!r) return e;
	let i = r[1], a = " ", o = t.lastIndex;
	for (n.lastIndex = o; r = n.exec(e);) r[1] === "" ? a === "\n" ? i += a : a = "\n" : (i += a + r[1], a = " "), o = n.lastIndex;
	let s = /[ \t]*(.*)/sy;
	return s.lastIndex = o, r = s.exec(e), i + a + (r?.[1] ?? "");
}
function Ib(e, t) {
	let n = "";
	for (let r = 1; r < e.length - 1; ++r) {
		let i = e[r];
		if (i !== "\r" || e[r + 1] !== "\n") {
			if (i === "\n") {
				let { fold: t, offset: i } = Lb(e, r);
				n += t, r = i;
			} else if (i === "\\") {
				let i = e[++r], a = Rb[i];
				if (a) n += a;
				else if (i === "\n") for (i = e[r + 1]; i === " " || i === "	";) i = e[++r + 1];
				else if (i === "\r" && e[r + 1] === "\n") for (i = e[++r + 1]; i === " " || i === "	";) i = e[++r + 1];
				else if (i === "x" || i === "u" || i === "U") {
					let a = i === "x" ? 2 : i === "u" ? 4 : 8;
					n += zb(e, r + 1, a, t), r += a;
				} else {
					let i = e.substr(r - 1, 2);
					t(r - 1, "BAD_DQ_ESCAPE", `Invalid escape sequence ${i}`), n += i;
				}
			} else if (i === " " || i === "	") {
				let t = r, a = e[r + 1];
				for (; a === " " || a === "	";) a = e[++r + 1];
				a !== "\n" && (a !== "\r" || e[r + 2] !== "\n") && (n += r > t ? e.slice(t, r + 1) : i);
			} else n += i;
		}
	}
	return (e[e.length - 1] !== "\"" || e.length === 1) && t(e.length, "MISSING_CHAR", "Missing closing \"quote"), n;
}
function Lb(e, t) {
	let n = "", r = e[t + 1];
	for (; (r === " " || r === "	" || r === "\n" || r === "\r") && (r !== "\r" || e[t + 2] === "\n");) r === "\n" && (n += "\n"), t += 1, r = e[t + 1];
	return n ||= " ", {
		fold: n,
		offset: t
	};
}
var Rb = {
	0: "\0",
	a: "\x07",
	b: "\b",
	e: "\x1B",
	f: "\f",
	n: "\n",
	r: "\r",
	t: "	",
	v: "\v",
	N: "",
	_: "\xA0",
	L: "\u2028",
	P: "\u2029",
	" ": " ",
	"\"": "\"",
	"/": "/",
	"\\": "\\",
	"	": "	"
};
function zb(e, t, n, r) {
	let i = e.substr(t, n), a = i.length === n && /^[0-9a-fA-F]+$/.test(i) ? parseInt(i, 16) : NaN;
	try {
		return String.fromCodePoint(a);
	} catch {
		let i = e.substr(t - 2, n + 2);
		return r(t - 2, "BAD_DQ_ESCAPE", `Invalid escape sequence ${i}`), i;
	}
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/compose-scalar.js
function Bb(e, t, n, r) {
	let { value: i, type: a, comment: o, range: s } = t.type === "block-scalar" ? kb(e, t, r) : Mb(t, e.options.strict, r), c = n ? e.directives.tagName(n.source, (e) => r(n, "TAG_RESOLVE_FAILED", e)) : null, l;
	l = e.options.stringKeys && e.atKey ? e.schema[L_] : c ? Vb(e.schema, i, c, n, r) : t.type === "scalar" ? Hb(e, i, t, r) : e.schema[L_];
	let u;
	try {
		let a = l.resolve(i, (e) => r(n ?? t, "TAG_RESOLVE_FAILED", e), e.options);
		u = W_(a) ? a : new W(a);
	} catch (e) {
		let a = e instanceof Error ? e.message : String(e);
		r(n ?? t, "TAG_RESOLVE_FAILED", a), u = new W(i);
	}
	return u.range = s, u.source = i, a && (u.type = a), c && (u.tag = c), l.format && (u.format = l.format), o && (u.comment = o), u;
}
function Vb(e, t, n, r, i) {
	if (n === "!") return e[L_];
	let a = [];
	for (let t of e.tags) if (!t.collection && t.tag === n) {
		if (t.default && t.test) a.push(t);
		else return t;
	}
	for (let e of a) if (e.test?.test(t)) return e;
	let o = e.knownTags[n];
	return o && !o.collection ? (e.tags.push(Object.assign({}, o, {
		default: !1,
		test: void 0
	})), o) : (i(r, "TAG_RESOLVE_FAILED", `Unresolved tag: ${n}`, n !== "tag:yaml.org,2002:str"), e[L_]);
}
function Hb({ atKey: e, directives: t, schema: n }, r, i, a) {
	let o = n.tags.find((t) => (t.default === !0 || e && t.default === "key") && t.test?.test(r)) || n[L_];
	if (n.compat) {
		let e = n.compat.find((e) => e.default && e.test?.test(r)) ?? n[L_];
		o.tag !== e.tag && a(i, "TAG_RESOLVE_FAILED", `Value may be parsed as either ${t.tagString(o.tag)} or ${t.tagString(e.tag)}`, !0);
	}
	return o;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/util-empty-scalar-position.js
function Ub(e, t, n) {
	if (t) {
		n ??= t.length;
		for (let r = n - 1; r >= 0; --r) {
			let n = t[r];
			switch (n.type) {
				case "space":
				case "comment":
				case "newline":
					e -= n.source.length;
					continue;
			}
			for (n = t[++r]; n?.type === "space";) e += n.source.length, n = t[++r];
			break;
		}
	}
	return e;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/compose-node.js
var Wb = {
	composeNode: Gb,
	composeEmptyNode: Kb
};
function Gb(e, t, n, r) {
	let i = e.atKey, { spaceBefore: a, comment: o, anchor: s, tag: c } = n, l, u = !0;
	switch (t.type) {
		case "alias":
			l = qb(e, t, r), (s || c) && r(t, "ALIAS_PROPS", "An alias node must not specify any properties");
			break;
		case "scalar":
		case "single-quoted-scalar":
		case "double-quoted-scalar":
		case "block-scalar":
			l = Bb(e, t, c, r), s && (l.anchor = s.source.substring(1));
			break;
		case "block-map":
		case "block-seq":
		case "flow-collection":
			try {
				l = Ob(Wb, e, t, n, r), s && (l.anchor = s.source.substring(1));
			} catch (e) {
				r(t, "RESOURCE_EXHAUSTION", e instanceof Error ? e.message : String(e));
			}
			break;
		default: r(t, "UNEXPECTED_TOKEN", t.type === "error" ? t.message : `Unsupported token (type: ${t.type})`), u = !1;
	}
	return l ??= Kb(e, t.offset, void 0, null, n, r), s && l.anchor === "" && r(s, "BAD_ALIAS", "Anchor cannot be an empty string"), i && e.options.stringKeys && (!W_(l) || typeof l.value != "string" || l.tag && l.tag !== "tag:yaml.org,2002:str") && r(c ?? t, "NON_STRING_KEY", "With stringKeys, all keys must be strings"), a && (l.spaceBefore = !0), o && (t.type === "scalar" && t.source === "" ? l.comment = o : l.commentBefore = o), e.options.keepSourceTokens && u && (l.srcToken = t), l;
}
function Kb(e, t, n, r, { spaceBefore: i, comment: a, anchor: o, tag: s, end: c }, l) {
	let u = Bb(e, {
		type: "scalar",
		offset: Ub(t, n, r),
		indent: -1,
		source: ""
	}, s, l);
	return o && (u.anchor = o.source.substring(1), u.anchor === "" && l(o, "BAD_ALIAS", "Anchor cannot be an empty string")), i && (u.spaceBefore = !0), a && (u.comment = a, u.range[2] = c), u;
}
function qb({ options: e }, { offset: t, source: n, end: r }, i) {
	let a = new pv(n.substring(1));
	a.source === "" && i(t, "BAD_ALIAS", "Alias cannot be an empty string"), a.source.endsWith(":") && i(t + n.length - 1, "BAD_ALIAS", "Alias ending in : is ambiguous", !0);
	let o = t + n.length, s = Cb(r, o, e.strict, i);
	return a.range = [
		t,
		o,
		s.offset
	], s.comment && (a.comment = s.comment), a;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/compose-doc.js
function Jb(e, t, { offset: n, start: r, value: i, end: a }, o) {
	let s = new ub(void 0, Object.assign({ _directives: t }, e)), c = {
		atKey: !1,
		atRoot: !0,
		directives: s.directives,
		options: s.options,
		schema: s.schema
	}, l = gb(r, {
		indicator: "doc-start",
		next: i ?? a?.[0],
		offset: n,
		onError: o,
		parentIndent: 0,
		startOnNewline: !0
	});
	l.found && (s.directives.docStart = !0, i && (i.type === "block-map" || i.type === "block-seq") && !l.hasNewline && o(l.end, "MISSING_CHAR", "Block collection cannot start on same line with directives-end marker")), s.contents = i ? Gb(c, i, l, o) : Kb(c, l.end, r, null, l, o);
	let u = s.contents.range[2], d = Cb(a, u, !1, o);
	return d.comment && (s.comment = d.comment), s.range = [
		n,
		u,
		d.offset
	], s;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/compose/composer.js
function Yb(e) {
	if (typeof e == "number") return [e, e + 1];
	if (Array.isArray(e)) return e.length === 2 ? e : [e[0], e[1]];
	let { offset: t, source: n } = e;
	return [t, t + (typeof n == "string" ? n.length : 1)];
}
function Xb(e) {
	let t = "", n = !1, r = !1;
	for (let i = 0; i < e.length; ++i) {
		let a = e[i];
		switch (a[0]) {
			case "#":
				t += (t === "" ? "" : r ? "\n\n" : "\n") + (a.substring(1) || " "), n = !0, r = !1;
				break;
			case "%":
				e[i + 1]?.[0] !== "#" && (i += 1), n = !1;
				break;
			default: n || (r = !0), n = !1;
		}
	}
	return {
		comment: t,
		afterEmptyLine: r
	};
}
var Zb = class {
	constructor(e = {}) {
		this.doc = null, this.atDirectives = !1, this.prelude = [], this.errors = [], this.warnings = [], this.onError = (e, t, n, r) => {
			let i = Yb(e);
			r ? this.warnings.push(new mb(i, t, n)) : this.errors.push(new pb(i, t, n));
		}, this.directives = new av({ version: e.version || "1.2" }), this.options = e;
	}
	decorate(e, t) {
		let { comment: n, afterEmptyLine: r } = Xb(this.prelude);
		if (n) {
			let i = e.contents;
			if (t) e.comment = e.comment ? `${e.comment}\n${n}` : n;
			else if (r || e.directives.docStart || !i) e.commentBefore = n;
			else if (K_(i) && !i.flow && i.items.length > 0) {
				let e = i.items[0];
				U_(e) && (e = e.key);
				let t = e.commentBefore;
				e.commentBefore = t ? `${n}\n${t}` : n;
			} else {
				let e = i.commentBefore;
				i.commentBefore = e ? `${n}\n${e}` : n;
			}
		}
		if (t) {
			for (let t = 0; t < this.errors.length; ++t) e.errors.push(this.errors[t]);
			for (let t = 0; t < this.warnings.length; ++t) e.warnings.push(this.warnings[t]);
		} else e.errors = this.errors, e.warnings = this.warnings;
		this.prelude = [], this.errors = [], this.warnings = [];
	}
	streamInfo() {
		return {
			comment: Xb(this.prelude).comment,
			directives: this.directives,
			errors: this.errors,
			warnings: this.warnings
		};
	}
	*compose(e, t = !1, n = -1) {
		for (let t of e) yield* this.next(t);
		yield* this.end(t, n);
	}
	*next(e) {
		switch (e.type) {
			case "directive":
				this.directives.add(e.source, (t, n, r) => {
					let i = Yb(e);
					i[0] += t, this.onError(i, "BAD_DIRECTIVE", n, r);
				}), this.prelude.push(e.source), this.atDirectives = !0;
				break;
			case "document": {
				let t = Jb(this.options, this.directives, e, this.onError);
				this.atDirectives && !t.directives.docStart && this.onError(e, "MISSING_CHAR", "Missing directives-end/doc-start indicator line"), this.decorate(t, !1), this.doc && (yield this.doc), this.doc = t, this.atDirectives = !1;
				break;
			}
			case "byte-order-mark":
			case "space": break;
			case "comment":
			case "newline":
				this.prelude.push(e.source);
				break;
			case "error": {
				let t = e.source ? `${e.message}: ${JSON.stringify(e.source)}` : e.message, n = new pb(Yb(e), "UNEXPECTED_TOKEN", t);
				this.atDirectives || !this.doc ? this.errors.push(n) : this.doc.errors.push(n);
				break;
			}
			case "doc-end": {
				if (!this.doc) {
					this.errors.push(new pb(Yb(e), "UNEXPECTED_TOKEN", "Unexpected doc-end without preceding document"));
					break;
				}
				this.doc.directives.docEnd = !0;
				let t = Cb(e.end, e.offset + e.source.length, this.doc.options.strict, this.onError);
				if (this.decorate(this.doc, !0), t.comment) {
					let e = this.doc.comment;
					this.doc.comment = e ? `${e}\n${t.comment}` : t.comment;
				}
				this.doc.range[2] = t.offset;
				break;
			}
			default: this.errors.push(new pb(Yb(e), "UNEXPECTED_TOKEN", `Unsupported token ${e.type}`));
		}
	}
	*end(e = !1, t = -1) {
		if (this.doc) this.decorate(this.doc, !0), yield this.doc, this.doc = null;
		else if (e) {
			let e = new ub(void 0, Object.assign({ _directives: this.directives }, this.options));
			this.atDirectives && this.onError(t, "MISSING_CHAR", "Missing directives-end indicator line"), e.range = [
				0,
				t,
				t
			], this.decorate(e, !1), yield e;
		}
	}
}, Qb = Symbol("break visit"), $b = Symbol("skip children"), ex = Symbol("remove item");
function tx(e, t) {
	"type" in e && e.type === "document" && (e = {
		start: e.start,
		value: e.value
	}), nx(Object.freeze([]), e, t);
}
tx.BREAK = Qb, tx.SKIP = $b, tx.REMOVE = ex, tx.itemAtPath = (e, t) => {
	let n = e;
	for (let [e, r] of t) {
		let t = n?.[e];
		if (t && "items" in t) n = t.items[r];
		else return;
	}
	return n;
}, tx.parentCollection = (e, t) => {
	let n = tx.itemAtPath(e, t.slice(0, -1)), r = t[t.length - 1][0], i = n?.[r];
	if (i && "items" in i) return i;
	throw Error("Parent collection not found");
};
function nx(e, t, n) {
	let r = n(t, e);
	if (typeof r == "symbol") return r;
	for (let i of ["key", "value"]) {
		let a = t[i];
		if (a && "items" in a) {
			for (let t = 0; t < a.items.length; ++t) {
				let r = nx(Object.freeze(e.concat([[i, t]])), a.items[t], n);
				if (typeof r == "number") t = r - 1;
				else if (r === Qb) return Qb;
				else r === ex && (a.items.splice(t, 1), --t);
			}
			typeof r == "function" && i === "key" && (r = r(t, e));
		}
	}
	return typeof r == "function" ? r(t, e) : r;
}
function rx(e) {
	switch (e) {
		case "﻿": return "byte-order-mark";
		case "": return "doc-mode";
		case "": return "flow-error-end";
		case "": return "scalar";
		case "---": return "doc-start";
		case "...": return "doc-end";
		case "":
		case "\n":
		case "\r\n": return "newline";
		case "-": return "seq-item-ind";
		case "?": return "explicit-key-ind";
		case ":": return "map-value-ind";
		case "{": return "flow-map-start";
		case "}": return "flow-map-end";
		case "[": return "flow-seq-start";
		case "]": return "flow-seq-end";
		case ",": return "comma";
	}
	switch (e[0]) {
		case " ":
		case "	": return "space";
		case "#": return "comment";
		case "%": return "directive-line";
		case "*": return "alias";
		case "&": return "anchor";
		case "!": return "tag";
		case "'": return "single-quoted-scalar";
		case "\"": return "double-quoted-scalar";
		case "|":
		case ">": return "block-scalar-header";
	}
	return null;
}
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/parse/lexer.js
function ix(e) {
	switch (e) {
		case void 0:
		case " ":
		case "\n":
		case "\r":
		case "	": return !0;
		default: return !1;
	}
}
var ax = /* @__PURE__ */ new Set("0123456789ABCDEFabcdef"), ox = /* @__PURE__ */ new Set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-#;/?:@&=+$_.!~*'()"), sx = /* @__PURE__ */ new Set(",[]{}"), cx = /* @__PURE__ */ new Set(" ,[]{}\n\r	"), lx = (e) => !e || cx.has(e), ux = class {
	constructor() {
		this.atEnd = !1, this.blockScalarIndent = -1, this.blockScalarKeep = !1, this.buffer = "", this.flowKey = !1, this.flowLevel = 0, this.indentNext = 0, this.indentValue = 0, this.lineEndPos = null, this.next = null, this.pos = 0;
	}
	*lex(e, t = !1) {
		if (e) {
			if (typeof e != "string") throw TypeError("source is not a string");
			this.buffer = this.buffer ? this.buffer + e : e, this.lineEndPos = null;
		}
		this.atEnd = !t;
		let n = this.next ?? "stream";
		for (; n && (t || this.hasChars(1));) n = yield* this.parseNext(n);
	}
	atLineEnd() {
		let e = this.pos, t = this.buffer[e];
		for (; t === " " || t === "	";) t = this.buffer[++e];
		return !t || t === "#" || t === "\n" || t === "\r" && this.buffer[e + 1] === "\n";
	}
	charAt(e) {
		return this.buffer[this.pos + e];
	}
	continueScalar(e) {
		let t = this.buffer[e];
		if (this.indentNext > 0) {
			let n = 0;
			for (; t === " ";) t = this.buffer[++n + e];
			if (t === "\r") {
				let t = this.buffer[n + e + 1];
				if (t === "\n" || !t && !this.atEnd) return e + n + 1;
			}
			return t === "\n" || n >= this.indentNext || !t && !this.atEnd ? e + n : -1;
		}
		if (t === "-" || t === ".") {
			let t = this.buffer.substr(e, 3);
			if ((t === "---" || t === "...") && ix(this.buffer[e + 3])) return -1;
		}
		return e;
	}
	getLine() {
		let e = this.lineEndPos;
		return (typeof e != "number" || e !== -1 && e < this.pos) && (e = this.buffer.indexOf("\n", this.pos), this.lineEndPos = e), e === -1 ? this.atEnd ? this.buffer.substring(this.pos) : null : (this.buffer[e - 1] === "\r" && --e, this.buffer.substring(this.pos, e));
	}
	hasChars(e) {
		return this.pos + e <= this.buffer.length;
	}
	setNext(e) {
		return this.buffer = this.buffer.substring(this.pos), this.pos = 0, this.lineEndPos = null, this.next = e, null;
	}
	peek(e) {
		return this.buffer.substr(this.pos, e);
	}
	*parseNext(e) {
		switch (e) {
			case "stream": return yield* this.parseStream();
			case "line-start": return yield* this.parseLineStart();
			case "block-start": return yield* this.parseBlockStart();
			case "doc": return yield* this.parseDocument();
			case "flow": return yield* this.parseFlowCollection();
			case "quoted-scalar": return yield* this.parseQuotedScalar();
			case "block-scalar": return yield* this.parseBlockScalar();
			case "plain-scalar": return yield* this.parsePlainScalar();
		}
	}
	*parseStream() {
		let e = this.getLine();
		if (e === null) return this.setNext("stream");
		if (e[0] === "﻿" && (yield* this.pushCount(1), e = e.substring(1)), e[0] === "%") {
			let t = e.length, n = e.indexOf("#");
			for (; n !== -1;) {
				let r = e[n - 1];
				if (r === " " || r === "	") {
					t = n - 1;
					break;
				}
				n = e.indexOf("#", n + 1);
			}
			for (;;) {
				let n = e[t - 1];
				if (n === " " || n === "	") --t;
				else break;
			}
			let r = (yield* this.pushCount(t)) + (yield* this.pushSpaces(!0));
			return yield* this.pushCount(e.length - r), this.pushNewline(), "stream";
		}
		if (this.atLineEnd()) {
			let t = yield* this.pushSpaces(!0);
			return yield* this.pushCount(e.length - t), yield* this.pushNewline(), "stream";
		}
		return yield "", yield* this.parseLineStart();
	}
	*parseLineStart() {
		let e = this.charAt(0);
		if (!e && !this.atEnd) return this.setNext("line-start");
		if (e === "-" || e === ".") {
			if (!this.atEnd && !this.hasChars(4)) return this.setNext("line-start");
			let e = this.peek(3);
			if ((e === "---" || e === "...") && ix(this.charAt(3))) return yield* this.pushCount(3), this.indentValue = 0, this.indentNext = 0, e === "---" ? "doc" : "stream";
		}
		return this.indentValue = yield* this.pushSpaces(!1), this.indentNext > this.indentValue && !ix(this.charAt(1)) && (this.indentNext = this.indentValue), yield* this.parseBlockStart();
	}
	*parseBlockStart() {
		let [e, t] = this.peek(2);
		if (!t && !this.atEnd) return this.setNext("block-start");
		if ((e === "-" || e === "?" || e === ":") && ix(t)) {
			let e = (yield* this.pushCount(1)) + (yield* this.pushSpaces(!0));
			return this.indentNext = this.indentValue + 1, this.indentValue += e, "block-start";
		}
		return "doc";
	}
	*parseDocument() {
		yield* this.pushSpaces(!0);
		let e = this.getLine();
		if (e === null) return this.setNext("doc");
		let t = yield* this.pushIndicators();
		switch (e[t]) {
			case "#": yield* this.pushCount(e.length - t);
			case void 0: return yield* this.pushNewline(), yield* this.parseLineStart();
			case "{":
			case "[": return yield* this.pushCount(1), this.flowKey = !1, this.flowLevel = 1, "flow";
			case "}":
			case "]": return yield* this.pushCount(1), "doc";
			case "*": return yield* this.pushUntil(lx), "doc";
			case "\"":
			case "'": return yield* this.parseQuotedScalar();
			case "|":
			case ">": return t += yield* this.parseBlockScalarHeader(), t += yield* this.pushSpaces(!0), yield* this.pushCount(e.length - t), yield* this.pushNewline(), yield* this.parseBlockScalar();
			default: return yield* this.parsePlainScalar();
		}
	}
	*parseFlowCollection() {
		let e, t, n = -1;
		do
			e = yield* this.pushNewline(), e > 0 ? (t = yield* this.pushSpaces(!1), this.indentValue = n = t) : t = 0, t += yield* this.pushSpaces(!0);
		while (e + t > 0);
		let r = this.getLine();
		if (r === null) return this.setNext("flow");
		if ((n !== -1 && n < this.indentNext && r[0] !== "#" || n === 0 && (r.startsWith("---") || r.startsWith("...")) && ix(r[3])) && (n !== this.indentNext - 1 || this.flowLevel !== 1 || r[0] !== "]" && r[0] !== "}")) return this.flowLevel = 0, yield "", yield* this.parseLineStart();
		let i = 0;
		for (; r[i] === ",";) i += yield* this.pushCount(1), i += yield* this.pushSpaces(!0), this.flowKey = !1;
		switch (i += yield* this.pushIndicators(), r[i]) {
			case void 0: return "flow";
			case "#": return yield* this.pushCount(r.length - i), "flow";
			case "{":
			case "[": return yield* this.pushCount(1), this.flowKey = !1, this.flowLevel += 1, "flow";
			case "}":
			case "]": return yield* this.pushCount(1), this.flowKey = !0, --this.flowLevel, this.flowLevel ? "flow" : "doc";
			case "*": return yield* this.pushUntil(lx), "flow";
			case "\"":
			case "'": return this.flowKey = !0, yield* this.parseQuotedScalar();
			case ":": {
				let e = this.charAt(1);
				if (this.flowKey || ix(e) || e === ",") return this.flowKey = !1, yield* this.pushCount(1), yield* this.pushSpaces(!0), "flow";
			}
			default: return this.flowKey = !1, yield* this.parsePlainScalar();
		}
	}
	*parseQuotedScalar() {
		let e = this.charAt(0), t = this.buffer.indexOf(e, this.pos + 1);
		if (e === "'") for (; t !== -1 && this.buffer[t + 1] === "'";) t = this.buffer.indexOf("'", t + 2);
		else for (; t !== -1;) {
			let e = 0;
			for (; this.buffer[t - 1 - e] === "\\";) e += 1;
			if (e % 2 == 0) break;
			t = this.buffer.indexOf("\"", t + 1);
		}
		let n = this.buffer.substring(0, t), r = n.indexOf("\n", this.pos);
		if (r !== -1) {
			for (; r !== -1;) {
				let e = this.continueScalar(r + 1);
				if (e === -1) break;
				r = n.indexOf("\n", e);
			}
			r !== -1 && (t = r - (n[r - 1] === "\r" ? 2 : 1));
		}
		if (t === -1) {
			if (!this.atEnd) return this.setNext("quoted-scalar");
			t = this.buffer.length;
		}
		return yield* this.pushToIndex(t + 1, !1), this.flowLevel ? "flow" : "doc";
	}
	*parseBlockScalarHeader() {
		this.blockScalarIndent = -1, this.blockScalarKeep = !1;
		let e = this.pos;
		for (;;) {
			let t = this.buffer[++e];
			if (t === "+") this.blockScalarKeep = !0;
			else if (t > "0" && t <= "9") this.blockScalarIndent = Number(t) - 1;
			else if (t !== "-") break;
		}
		return yield* this.pushUntil((e) => ix(e) || e === "#");
	}
	*parseBlockScalar() {
		let e = this.pos - 1, t = 0, n;
		loop: for (let r = this.pos; n = this.buffer[r]; ++r) switch (n) {
			case " ":
				t += 1;
				break;
			case "\n":
				e = r, t = 0;
				break;
			case "\r": {
				let e = this.buffer[r + 1];
				if (!e && !this.atEnd) return this.setNext("block-scalar");
				if (e === "\n") break;
			}
			default: break loop;
		}
		if (!n && !this.atEnd) return this.setNext("block-scalar");
		if (t >= this.indentNext) {
			this.indentNext = this.blockScalarIndent === -1 ? t : this.blockScalarIndent + (this.indentNext === 0 ? 1 : this.indentNext);
			do {
				let t = this.continueScalar(e + 1);
				if (t === -1) break;
				e = this.buffer.indexOf("\n", t);
			} while (e !== -1);
			if (e === -1) {
				if (!this.atEnd) return this.setNext("block-scalar");
				e = this.buffer.length;
			}
		}
		let r = e + 1;
		for (n = this.buffer[r]; n === " ";) n = this.buffer[++r];
		if (n === "	") {
			for (; n === "	" || n === " " || n === "\r" || n === "\n";) n = this.buffer[++r];
			e = r - 1;
		} else if (!this.blockScalarKeep) do {
			let n = e - 1, r = this.buffer[n];
			r === "\r" && (r = this.buffer[--n]);
			let i = n;
			for (; r === " ";) r = this.buffer[--n];
			if (r === "\n" && n >= this.pos && n + 1 + t > i) e = n;
			else break;
		} while (!0);
		return yield "", yield* this.pushToIndex(e + 1, !0), yield* this.parseLineStart();
	}
	*parsePlainScalar() {
		let e = this.flowLevel > 0, t = this.pos - 1, n = this.pos - 1, r;
		for (; r = this.buffer[++n];) if (r === ":") {
			let r = this.buffer[n + 1];
			if (ix(r) || e && sx.has(r)) break;
			t = n;
		} else if (ix(r)) {
			let i = this.buffer[n + 1];
			if (r === "\r" && (i === "\n" ? (n += 1, r = "\n", i = this.buffer[n + 1]) : t = n), i === "#" || e && sx.has(i)) break;
			if (r === "\n") {
				let e = this.continueScalar(n + 1);
				if (e === -1) break;
				n = Math.max(n, e - 2);
			}
		} else {
			if (e && sx.has(r)) break;
			t = n;
		}
		return !r && !this.atEnd ? this.setNext("plain-scalar") : (yield "", yield* this.pushToIndex(t + 1, !0), e ? "flow" : "doc");
	}
	*pushCount(e) {
		return e > 0 ? (yield this.buffer.substr(this.pos, e), this.pos += e, e) : 0;
	}
	*pushToIndex(e, t) {
		let n = this.buffer.slice(this.pos, e);
		return n ? (yield n, this.pos += n.length, n.length) : (t && (yield ""), 0);
	}
	*pushIndicators() {
		let e = 0;
		loop: for (;;) {
			switch (this.charAt(0)) {
				case "!":
					e += yield* this.pushTag(), e += yield* this.pushSpaces(!0);
					continue loop;
				case "&":
					e += yield* this.pushUntil(lx), e += yield* this.pushSpaces(!0);
					continue loop;
				case "-":
				case "?":
				case ":": {
					let t = this.flowLevel > 0, n = this.charAt(1);
					if (ix(n) || t && sx.has(n)) {
						t ? this.flowKey &&= !1 : this.indentNext = this.indentValue + 1, e += yield* this.pushCount(1), e += yield* this.pushSpaces(!0);
						continue loop;
					}
				}
			}
			break loop;
		}
		return e;
	}
	*pushTag() {
		if (this.charAt(1) === "<") {
			let e = this.pos + 2, t = this.buffer[e];
			for (; !ix(t) && t !== ">";) t = this.buffer[++e];
			return yield* this.pushToIndex(t === ">" ? e + 1 : e, !1);
		}
		{
			let e = this.pos + 1, t = this.buffer[e];
			for (; t;) if (ox.has(t)) t = this.buffer[++e];
			else if (t === "%" && ax.has(this.buffer[e + 1]) && ax.has(this.buffer[e + 2])) t = this.buffer[e += 3];
			else break;
			return yield* this.pushToIndex(e, !1);
		}
	}
	*pushNewline() {
		let e = this.buffer[this.pos];
		return e === "\n" ? yield* this.pushCount(1) : e === "\r" && this.charAt(1) === "\n" ? yield* this.pushCount(2) : 0;
	}
	*pushSpaces(e) {
		let t = this.pos - 1, n;
		do
			n = this.buffer[++t];
		while (n === " " || e && n === "	");
		let r = t - this.pos;
		return r > 0 && (yield this.buffer.substr(this.pos, r), this.pos = t), r;
	}
	*pushUntil(e) {
		let t = this.pos, n = this.buffer[t];
		for (; !e(n);) n = this.buffer[++t];
		return yield* this.pushToIndex(t, !1);
	}
}, dx = class {
	constructor() {
		this.lineStarts = [], this.addNewLine = (e) => this.lineStarts.push(e), this.linePos = (e) => {
			let t = 0, n = this.lineStarts.length;
			for (; t < n;) {
				let r = t + n >> 1;
				this.lineStarts[r] < e ? t = r + 1 : n = r;
			}
			if (this.lineStarts[t] === e) return {
				line: t + 1,
				col: 1
			};
			if (t === 0) return {
				line: 0,
				col: e
			};
			let r = this.lineStarts[t - 1];
			return {
				line: t,
				col: e - r + 1
			};
		};
	}
};
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/parse/parser.js
function fx(e, t) {
	for (let n = 0; n < e.length; ++n) if (e[n].type === t) return !0;
	return !1;
}
function px(e) {
	for (let t = 0; t < e.length; ++t) switch (e[t].type) {
		case "space":
		case "comment":
		case "newline": break;
		default: return t;
	}
	return -1;
}
function mx(e) {
	switch (e?.type) {
		case "alias":
		case "scalar":
		case "single-quoted-scalar":
		case "double-quoted-scalar":
		case "flow-collection": return !0;
		default: return !1;
	}
}
function hx(e) {
	switch (e.type) {
		case "document": return e.start;
		case "block-map": {
			let t = e.items[e.items.length - 1];
			return t.sep ?? t.start;
		}
		case "block-seq": return e.items[e.items.length - 1].start;
		/* istanbul ignore next should not happen */
		default: return [];
	}
}
function gx(e) {
	if (e.length === 0) return [];
	let t = e.length;
	loop: for (; --t >= 0;) switch (e[t].type) {
		case "doc-start":
		case "explicit-key-ind":
		case "map-value-ind":
		case "seq-item-ind":
		case "newline": break loop;
	}
	for (; e[++t]?.type === "space";);
	return e.splice(t, e.length);
}
function _x(e, t) {
	if (t.length < 1e5) Array.prototype.push.apply(e, t);
	else for (let n = 0; n < t.length; ++n) e.push(t[n]);
}
function vx(e) {
	if (e.start.type === "flow-seq-start") for (let t of e.items) t.sep && !t.value && !fx(t.start, "explicit-key-ind") && !fx(t.sep, "map-value-ind") && (t.key && (t.value = t.key), delete t.key, mx(t.value) ? t.value.end ? _x(t.value.end, t.sep) : t.value.end = t.sep : _x(t.start, t.sep), delete t.sep);
}
var yx = class {
	constructor(e) {
		this.atNewLine = !0, this.atScalar = !1, this.indent = 0, this.offset = 0, this.onKeyLine = !1, this.stack = [], this.source = "", this.type = "", this.lexer = new ux(), this.onNewLine = e;
	}
	*parse(e, t = !1) {
		this.onNewLine && this.offset === 0 && this.onNewLine(0);
		for (let n of this.lexer.lex(e, t)) yield* this.next(n);
		t || (yield* this.end());
	}
	*next(e) {
		if (this.source = e, this.atScalar) {
			this.atScalar = !1, yield* this.step(), this.offset += e.length;
			return;
		}
		let t = rx(e);
		if (!t) {
			let t = `Not a YAML token: ${e}`;
			yield* this.pop({
				type: "error",
				offset: this.offset,
				message: t,
				source: e
			}), this.offset += e.length;
		} else if (t === "scalar") this.atNewLine = !1, this.atScalar = !0, this.type = "scalar";
		else {
			switch (this.type = t, yield* this.step(), t) {
				case "newline":
					this.atNewLine = !0, this.indent = 0, this.onNewLine && this.onNewLine(this.offset + e.length);
					break;
				case "space":
					this.atNewLine && e[0] === " " && (this.indent += e.length);
					break;
				case "explicit-key-ind":
				case "map-value-ind":
				case "seq-item-ind":
					this.atNewLine && (this.indent += e.length);
					break;
				case "doc-mode":
				case "flow-error-end": return;
				default: this.atNewLine = !1;
			}
			this.offset += e.length;
		}
	}
	*end() {
		for (; this.stack.length > 0;) yield* this.pop();
	}
	get sourceToken() {
		return {
			type: this.type,
			offset: this.offset,
			indent: this.indent,
			source: this.source
		};
	}
	*step() {
		let e = this.peek(1);
		if (this.type === "doc-end" && e?.type !== "doc-end") {
			for (; this.stack.length > 0;) yield* this.pop();
			this.stack.push({
				type: "doc-end",
				offset: this.offset,
				source: this.source
			});
			return;
		}
		if (!e) return yield* this.stream();
		switch (e.type) {
			case "document": return yield* this.document(e);
			case "alias":
			case "scalar":
			case "single-quoted-scalar":
			case "double-quoted-scalar": return yield* this.scalar(e);
			case "block-scalar": return yield* this.blockScalar(e);
			case "block-map": return yield* this.blockMap(e);
			case "block-seq": return yield* this.blockSequence(e);
			case "flow-collection": return yield* this.flowCollection(e);
			case "doc-end": return yield* this.documentEnd(e);
		}
		/* istanbul ignore next should not happen */
		yield* this.pop();
	}
	peek(e) {
		return this.stack[this.stack.length - e];
	}
	*pop(e) {
		let t = e ?? this.stack.pop();
		/* istanbul ignore if should not happen */
		if (!t) yield {
			type: "error",
			offset: this.offset,
			source: "",
			message: "Tried to pop an empty stack"
		};
		else if (this.stack.length === 0) yield t;
		else {
			let e = this.peek(1);
			switch (t.type === "block-scalar" ? t.indent = "indent" in e ? e.indent : 0 : t.type === "flow-collection" && e.type === "document" && (t.indent = 0), t.type === "flow-collection" && vx(t), e.type) {
				case "document":
					e.value = t;
					break;
				case "block-scalar":
					e.props.push(t);
					break;
				case "block-map": {
					let n = e.items[e.items.length - 1];
					if (n.value) {
						e.items.push({
							start: [],
							key: t,
							sep: []
						}), this.onKeyLine = !0;
						return;
					}
					if (n.sep) n.value = t;
					else {
						Object.assign(n, {
							key: t,
							sep: []
						}), this.onKeyLine = !n.explicitKey;
						return;
					}
					break;
				}
				case "block-seq": {
					let n = e.items[e.items.length - 1];
					n.value ? e.items.push({
						start: [],
						value: t
					}) : n.value = t;
					break;
				}
				case "flow-collection": {
					let n = e.items[e.items.length - 1];
					!n || n.value ? e.items.push({
						start: [],
						key: t,
						sep: []
					}) : n.sep ? n.value = t : Object.assign(n, {
						key: t,
						sep: []
					});
					return;
				}
				/* istanbul ignore next should not happen */
				default: yield* this.pop(), yield* this.pop(t);
			}
			if ((e.type === "document" || e.type === "block-map" || e.type === "block-seq") && (t.type === "block-map" || t.type === "block-seq")) {
				let n = t.items[t.items.length - 1];
				n && !n.sep && !n.value && n.start.length > 0 && px(n.start) === -1 && (t.indent === 0 || n.start.every((e) => e.type !== "comment" || e.indent < t.indent)) && (e.type === "document" ? e.end = n.start : e.items.push({ start: n.start }), t.items.splice(-1, 1));
			}
		}
	}
	*stream() {
		switch (this.type) {
			case "directive-line":
				yield {
					type: "directive",
					offset: this.offset,
					source: this.source
				};
				return;
			case "byte-order-mark":
			case "space":
			case "comment":
			case "newline":
				yield this.sourceToken;
				return;
			case "doc-mode":
			case "doc-start": {
				let e = {
					type: "document",
					offset: this.offset,
					start: []
				};
				this.type === "doc-start" && e.start.push(this.sourceToken), this.stack.push(e);
				return;
			}
		}
		yield {
			type: "error",
			offset: this.offset,
			message: `Unexpected ${this.type} token in YAML stream`,
			source: this.source
		};
	}
	*document(e) {
		if (e.value) return yield* this.lineEnd(e);
		switch (this.type) {
			case "doc-start":
				px(e.start) === -1 ? e.start.push(this.sourceToken) : (yield* this.pop(), yield* this.step());
				return;
			case "anchor":
			case "tag":
			case "space":
			case "comment":
			case "newline":
				e.start.push(this.sourceToken);
				return;
		}
		let t = this.startBlockValue(e);
		t ? this.stack.push(t) : yield {
			type: "error",
			offset: this.offset,
			message: `Unexpected ${this.type} token in YAML document`,
			source: this.source
		};
	}
	*scalar(e) {
		if (this.type === "map-value-ind") {
			let t = gx(hx(this.peek(2))), n;
			e.end ? (n = e.end, n.push(this.sourceToken), delete e.end) : n = [this.sourceToken];
			let r = {
				type: "block-map",
				offset: e.offset,
				indent: e.indent,
				items: [{
					start: t,
					key: e,
					sep: n
				}]
			};
			this.onKeyLine = !0, this.stack[this.stack.length - 1] = r;
		} else yield* this.lineEnd(e);
	}
	*blockScalar(e) {
		switch (this.type) {
			case "space":
			case "comment":
			case "newline":
				e.props.push(this.sourceToken);
				return;
			case "scalar":
				if (e.source = this.source, this.atNewLine = !0, this.indent = 0, this.onNewLine) {
					let e = this.source.indexOf("\n") + 1;
					for (; e !== 0;) this.onNewLine(this.offset + e), e = this.source.indexOf("\n", e) + 1;
				}
				yield* this.pop();
				break;
			/* istanbul ignore next should not happen */
			default: yield* this.pop(), yield* this.step();
		}
	}
	*blockMap(e) {
		let t = e.items[e.items.length - 1];
		switch (this.type) {
			case "newline":
				if (this.onKeyLine = !1, t.value) {
					let n = "end" in t.value ? t.value.end : void 0;
					(Array.isArray(n) ? n[n.length - 1] : void 0)?.type === "comment" ? n?.push(this.sourceToken) : e.items.push({ start: [this.sourceToken] });
				} else t.sep ? t.sep.push(this.sourceToken) : t.start.push(this.sourceToken);
				return;
			case "space":
			case "comment":
				if (t.value) e.items.push({ start: [this.sourceToken] });
				else if (t.sep) t.sep.push(this.sourceToken);
				else {
					if (this.atIndentedComment(t.start, e.indent)) {
						let n = e.items[e.items.length - 2]?.value?.end;
						if (Array.isArray(n)) {
							_x(n, t.start), n.push(this.sourceToken), e.items.pop();
							return;
						}
					}
					t.start.push(this.sourceToken);
				}
				return;
		}
		if (this.indent >= e.indent) {
			let n = !this.onKeyLine && this.indent === e.indent, r = n && (t.sep || t.explicitKey) && this.type !== "seq-item-ind", i = [];
			if (r && t.sep && !t.value) {
				let n = [];
				for (let r = 0; r < t.sep.length; ++r) {
					let i = t.sep[r];
					switch (i.type) {
						case "newline":
							n.push(r);
							break;
						case "space": break;
						case "comment":
							i.indent > e.indent && (n.length = 0);
							break;
						default: n.length = 0;
					}
				}
				n.length >= 2 && (i = t.sep.splice(n[1]));
			}
			switch (this.type) {
				case "anchor":
				case "tag":
					r || t.value ? (i.push(this.sourceToken), e.items.push({ start: i }), this.onKeyLine = !0) : t.sep ? t.sep.push(this.sourceToken) : t.start.push(this.sourceToken);
					return;
				case "explicit-key-ind":
					!t.sep && !t.explicitKey ? (t.start.push(this.sourceToken), t.explicitKey = !0) : r || t.value ? (i.push(this.sourceToken), e.items.push({
						start: i,
						explicitKey: !0
					})) : this.stack.push({
						type: "block-map",
						offset: this.offset,
						indent: this.indent,
						items: [{
							start: [this.sourceToken],
							explicitKey: !0
						}]
					}), this.onKeyLine = !0;
					return;
				case "map-value-ind":
					if (t.explicitKey) {
						if (!t.sep) {
							if (fx(t.start, "newline")) Object.assign(t, {
								key: null,
								sep: [this.sourceToken]
							});
							else {
								let e = gx(t.start);
								this.stack.push({
									type: "block-map",
									offset: this.offset,
									indent: this.indent,
									items: [{
										start: e,
										key: null,
										sep: [this.sourceToken]
									}]
								});
							}
						} else if (t.value) e.items.push({
							start: [],
							key: null,
							sep: [this.sourceToken]
						});
						else if (fx(t.sep, "map-value-ind")) this.stack.push({
							type: "block-map",
							offset: this.offset,
							indent: this.indent,
							items: [{
								start: i,
								key: null,
								sep: [this.sourceToken]
							}]
						});
						else if (mx(t.key) && !fx(t.sep, "newline")) {
							let e = gx(t.start), n = t.key, r = t.sep;
							r.push(this.sourceToken), delete t.key, delete t.sep, this.stack.push({
								type: "block-map",
								offset: this.offset,
								indent: this.indent,
								items: [{
									start: e,
									key: n,
									sep: r
								}]
							});
						} else i.length > 0 ? t.sep = t.sep.concat(i, this.sourceToken) : t.sep.push(this.sourceToken);
					} else t.sep ? t.value || r ? e.items.push({
						start: i,
						key: null,
						sep: [this.sourceToken]
					}) : fx(t.sep, "map-value-ind") ? this.stack.push({
						type: "block-map",
						offset: this.offset,
						indent: this.indent,
						items: [{
							start: [],
							key: null,
							sep: [this.sourceToken]
						}]
					}) : t.sep.push(this.sourceToken) : Object.assign(t, {
						key: null,
						sep: [this.sourceToken]
					});
					this.onKeyLine = !0;
					return;
				case "alias":
				case "scalar":
				case "single-quoted-scalar":
				case "double-quoted-scalar": {
					let n = this.flowScalar(this.type);
					r || t.value ? (e.items.push({
						start: i,
						key: n,
						sep: []
					}), this.onKeyLine = !0) : t.sep ? this.stack.push(n) : (Object.assign(t, {
						key: n,
						sep: []
					}), this.onKeyLine = !0);
					return;
				}
				default: {
					let r = this.startBlockValue(e);
					if (r) {
						if (r.type === "block-seq") {
							if (!t.explicitKey && t.sep && !fx(t.sep, "newline")) {
								yield* this.pop({
									type: "error",
									offset: this.offset,
									message: "Unexpected block-seq-ind on same line with key",
									source: this.source
								});
								return;
							}
						} else n && e.items.push({ start: i });
						this.stack.push(r);
						return;
					}
				}
			}
		}
		yield* this.pop(), yield* this.step();
	}
	*blockSequence(e) {
		let t = e.items[e.items.length - 1];
		switch (this.type) {
			case "newline":
				if (t.value) {
					let n = "end" in t.value ? t.value.end : void 0;
					(Array.isArray(n) ? n[n.length - 1] : void 0)?.type === "comment" ? n?.push(this.sourceToken) : e.items.push({ start: [this.sourceToken] });
				} else t.start.push(this.sourceToken);
				return;
			case "space":
			case "comment":
				if (t.value) e.items.push({ start: [this.sourceToken] });
				else {
					if (this.atIndentedComment(t.start, e.indent)) {
						let n = e.items[e.items.length - 2]?.value?.end;
						if (Array.isArray(n)) {
							_x(n, t.start), n.push(this.sourceToken), e.items.pop();
							return;
						}
					}
					t.start.push(this.sourceToken);
				}
				return;
			case "anchor":
			case "tag":
				if (t.value || this.indent <= e.indent) break;
				t.start.push(this.sourceToken);
				return;
			case "seq-item-ind":
				if (this.indent !== e.indent) break;
				t.value || fx(t.start, "seq-item-ind") ? e.items.push({ start: [this.sourceToken] }) : t.start.push(this.sourceToken);
				return;
		}
		if (this.indent > e.indent) {
			let t = this.startBlockValue(e);
			if (t) {
				this.stack.push(t);
				return;
			}
		}
		yield* this.pop(), yield* this.step();
	}
	*flowCollection(e) {
		let t = e.items[e.items.length - 1];
		if (this.type === "flow-error-end") {
			let e;
			do
				yield* this.pop(), e = this.peek(1);
			while (e?.type === "flow-collection");
		} else if (e.end.length === 0) {
			switch (this.type) {
				case "comma":
				case "explicit-key-ind":
					!t || t.sep ? e.items.push({ start: [this.sourceToken] }) : t.start.push(this.sourceToken);
					return;
				case "map-value-ind":
					!t || t.value ? e.items.push({
						start: [],
						key: null,
						sep: [this.sourceToken]
					}) : t.sep ? t.sep.push(this.sourceToken) : Object.assign(t, {
						key: null,
						sep: [this.sourceToken]
					});
					return;
				case "space":
				case "comment":
				case "newline":
				case "anchor":
				case "tag":
					!t || t.value ? e.items.push({ start: [this.sourceToken] }) : t.sep ? t.sep.push(this.sourceToken) : t.start.push(this.sourceToken);
					return;
				case "alias":
				case "scalar":
				case "single-quoted-scalar":
				case "double-quoted-scalar": {
					let n = this.flowScalar(this.type);
					!t || t.value ? e.items.push({
						start: [],
						key: n,
						sep: []
					}) : t.sep ? this.stack.push(n) : Object.assign(t, {
						key: n,
						sep: []
					});
					return;
				}
				case "flow-map-end":
				case "flow-seq-end":
					e.end.push(this.sourceToken);
					return;
			}
			let n = this.startBlockValue(e);
			/* istanbul ignore else should not happen */
			n ? this.stack.push(n) : (yield* this.pop(), yield* this.step());
		} else {
			let t = this.peek(2);
			if (t.type === "block-map" && (this.type === "map-value-ind" && t.indent === e.indent || this.type === "newline" && !t.items[t.items.length - 1].sep)) yield* this.pop(), yield* this.step();
			else if (this.type === "map-value-ind" && t.type !== "flow-collection") {
				let n = gx(hx(t));
				vx(e);
				let r = e.end.splice(1, e.end.length);
				r.push(this.sourceToken);
				let i = {
					type: "block-map",
					offset: e.offset,
					indent: e.indent,
					items: [{
						start: n,
						key: e,
						sep: r
					}]
				};
				this.onKeyLine = !0, this.stack[this.stack.length - 1] = i;
			} else yield* this.lineEnd(e);
		}
	}
	flowScalar(e) {
		if (this.onNewLine) {
			let e = this.source.indexOf("\n") + 1;
			for (; e !== 0;) this.onNewLine(this.offset + e), e = this.source.indexOf("\n", e) + 1;
		}
		return {
			type: e,
			offset: this.offset,
			indent: this.indent,
			source: this.source
		};
	}
	startBlockValue(e) {
		switch (this.type) {
			case "alias":
			case "scalar":
			case "single-quoted-scalar":
			case "double-quoted-scalar": return this.flowScalar(this.type);
			case "block-scalar-header": return {
				type: "block-scalar",
				offset: this.offset,
				indent: this.indent,
				props: [this.sourceToken],
				source: ""
			};
			case "flow-map-start":
			case "flow-seq-start": return {
				type: "flow-collection",
				offset: this.offset,
				indent: this.indent,
				start: this.sourceToken,
				items: [],
				end: []
			};
			case "seq-item-ind": return {
				type: "block-seq",
				offset: this.offset,
				indent: this.indent,
				items: [{ start: [this.sourceToken] }]
			};
			case "explicit-key-ind": {
				this.onKeyLine = !0;
				let t = gx(hx(e));
				return t.push(this.sourceToken), {
					type: "block-map",
					offset: this.offset,
					indent: this.indent,
					items: [{
						start: t,
						explicitKey: !0
					}]
				};
			}
			case "map-value-ind": {
				this.onKeyLine = !0;
				let t = gx(hx(e));
				return {
					type: "block-map",
					offset: this.offset,
					indent: this.indent,
					items: [{
						start: t,
						key: null,
						sep: [this.sourceToken]
					}]
				};
			}
		}
		return null;
	}
	atIndentedComment(e, t) {
		return this.type !== "comment" || this.indent <= t ? !1 : e.every((e) => e.type === "newline" || e.type === "space");
	}
	*documentEnd(e) {
		this.type !== "doc-mode" && (e.end ? e.end.push(this.sourceToken) : e.end = [this.sourceToken], this.type === "newline" && (yield* this.pop()));
	}
	*lineEnd(e) {
		switch (this.type) {
			case "comma":
			case "doc-start":
			case "doc-end":
			case "flow-seq-end":
			case "flow-map-end":
			case "map-value-ind":
				yield* this.pop(), yield* this.step();
				break;
			case "newline": this.onKeyLine = !1;
			default: e.end ? e.end.push(this.sourceToken) : e.end = [this.sourceToken], this.type === "newline" && (yield* this.pop());
		}
	}
};
//#endregion
//#region node_modules/.pnpm/yaml@2.9.0/node_modules/yaml/browser/dist/public-api.js
function bx(e) {
	let t = e.prettyErrors !== !1;
	return {
		lineCounter: e.lineCounter || t && new dx() || null,
		prettyErrors: t
	};
}
function xx(e, t = {}) {
	let { lineCounter: n, prettyErrors: r } = bx(t), i = new yx(n?.addNewLine), a = new Zb(t), o = null;
	for (let t of a.compose(i.parse(e), !0, e.length)) if (!o) o = t;
	else if (o.options.logLevel !== "silent") {
		o.errors.push(new pb(t.range.slice(0, 2), "MULTIPLE_DOCS", "Source contains multiple documents; please use YAML.parseAllDocuments()"));
		break;
	}
	return r && n && (o.errors.forEach(hb(e, n)), o.warnings.forEach(hb(e, n))), o;
}
function Sx(e, t, n) {
	let r;
	typeof t == "function" ? r = t : n === void 0 && t && typeof t == "object" && (n = t);
	let i = xx(e, n);
	if (!i) return null;
	if (i.warnings.forEach((e) => Gv(i.options.logLevel, e)), i.errors.length > 0) {
		if (i.options.logLevel !== "silent") throw i.errors[0];
		i.errors = [];
	}
	return i.toJS(Object.assign({ reviver: r }, n));
}
//#endregion
//#region src/components/useMitigations.ts
var Cx = "mitigations", wx = (e) => {
	if (!e) return "";
	let t = [...e.matchAll(/'text':\s*'((?:[^'\\]|\\.)*)'/g)].map((e) => e[1].replace(/\\n/g, "\n").replace(/\\'/g, "'").replace(/\\"/g, "\"").replace(/\\\\/g, "\\"));
	return (t.length ? t.join("\n\n") : e).trim();
}, Tx = (e) => e ? e.split(".").pop() ?? e : "", Ex = (e) => e && typeof e == "object" && !Array.isArray(e) ? e : {}, Dx = (e) => {
	let t = {};
	if (!e) return t;
	let n = e.split(/\[\[components\.config\.guardrails\]\]/).slice(1);
	for (let e of n) {
		let n = (t) => {
			let n = e.match(RegExp(`^${t}\\s*=\\s*(?:"""([\\s\\S]*?)"""|"((?:[^"\\\\]|\\\\.)*)")`, "m"));
			return n ? n[1] ?? n[2] : void 0;
		}, r = n("name");
		r && (t[r] = {
			name: r,
			target_tool: n("target_tool"),
			system_instructions: n("system_instructions")
		});
	}
	return t;
}, Ox = (e) => {
	let t = e.replace(/\s+/g, " ").trim(), n = t.search(/(?<=[.!?])\s/), r = (n > 0 ? t.slice(0, n + 1) : t).trim();
	return r.length > 220 ? `${r.slice(0, 217)}…` : r;
}, kx = (e) => {
	if (!e) return [];
	let t = [];
	if (e.guardrails) {
		let n = Dx(e.guardrails.before), r = Dx(e.guardrails.after);
		for (let [e, i] of Object.entries(r)) {
			if (e in n) continue;
			let r = i.target_tool;
			t.push({
				title: r ? `Added a guardrail on ${r}` : "Added a tool-call guardrail",
				detail: i.system_instructions ? Ox(i.system_instructions) : "Verifies tool calls before execution."
			});
		}
	}
	if (e.policy) {
		let n = Ax(e.policy.before, e.policy.after);
		n && t.push({
			title: "OpenShell policy changes",
			detail: n
		});
	}
	return t;
}, Ax = (e, t) => {
	let n, r;
	try {
		n = Ex(Sx(e)), r = Ex(Sx(t));
	} catch {
		let n = jx(e, t);
		return n > 0 ? `${n} line${n === 1 ? "" : "s"} changed.` : "";
	}
	let i = (e, t, n) => {
		let r = Ex(e[t])[n];
		return Array.isArray(r) ? r.length : 0;
	}, a = (e) => Object.values(Ex(e.network_policies)).reduce((e, t) => {
		let n = Ex(t).endpoints;
		return e + (Array.isArray(n) ? n.filter((e) => Array.isArray(Ex(e).allowed_ips)).length : 0);
	}, 0), o = [], s = i(n, "filesystem_policy", "read_only"), c = i(r, "filesystem_policy", "read_only");
	s !== c && o.push(`Filesystem read-only paths: ${s} → ${c}`);
	let l = i(n, "filesystem_policy", "read_write"), u = i(r, "filesystem_policy", "read_write");
	l !== u && o.push(`Filesystem read-write paths: ${l} → ${u}`);
	let d = a(n), f = a(r);
	if (d !== f && o.push(`Network endpoints with IP allow-lists: ${d} → ${f}`), o.length > 0) return `${o.join(" · ")}. See the diff below.`;
	let p = jx(e, t);
	return p > 0 ? `${p} line${p === 1 ? "" : "s"} changed. See the diff below.` : "";
}, jx = (e, t) => {
	let n = (e) => {
		let t = /* @__PURE__ */ new Map();
		for (let n of e.split("\n")) {
			let e = n.trim();
			e && t.set(e, (t.get(e) ?? 0) + 1);
		}
		return t;
	}, r = n(e), i = n(t), a = /* @__PURE__ */ new Set([...r.keys(), ...i.keys()]), o = 0;
	for (let e of a) o += Math.abs((r.get(e) ?? 0) - (i.get(e) ?? 0));
	return o;
}, Mx = (e, t, n) => {
	let r = A_(e, t, Cx, j_, n);
	return {
		mitigations: r.data,
		recommendations: kx(r.data),
		defenses: r.data?.defenses ?? [],
		isLoading: r.isLoading,
		hasMitigations: r.present
	};
}, Nx = "validation", Px = "composed-guardrails", Fx = (e, t) => ze({ mutationFn: (n) => Ui({
	url: `/apis/agent-hardener/v2/workspaces/${encodeURIComponent(e)}/runs/${encodeURIComponent(t)}/compose-defense`,
	method: "POST",
	headers: { "Content-Type": "application/json" },
	data: {
		mitigations: n.mitigations,
		selected_defense_ids: n.selectedDefenseIds
	}
}) }), Ix = (e) => {
	let t = no();
	return {
		submit: async (n) => (await t.mutateAsync({
			workspace: e,
			data: { spec: n }
		})).name,
		isPending: t.isPending
	};
}, Lx = (e, t, n) => {
	let r = A_(e, t, Nx, j_, n);
	return {
		report: r.data,
		isLoading: r.isLoading,
		hasReport: r.present,
		missingReport: r.missing
	};
}, Rx = (e, t, n) => A_(e, t, Px, M_, n).data, zx = (e, t) => {
	let { data: n } = Be({
		queryKey: [
			"agent-hardener-sanity-lookup",
			e,
			t
		],
		enabled: !!t,
		refetchInterval: (e) => !e.state.data && D,
		queryFn: async () => ((await Ro(e, {
			sort: "-created_at",
			page_size: 50
		})).data ?? []).find((e) => e.source_run === t)?.job_id
	});
	return n ?? void 0;
}, Bx = (e) => {
	let t = [], n = /* @__PURE__ */ new Map();
	for (let r of e) {
		let e = r.target_tool || (r.kind === "policy" ? "OpenShell sandbox policy" : "Other");
		n.has(e) || (n.set(e, []), t.push(e)), n.get(e)?.push(r);
	}
	return t.map((e) => ({
		tool: e,
		items: n.get(e) ?? []
	}));
}, Vx = (e) => e.hasComposedGuardrails ? !e.hasManifestId || e.manifestQuerySucceeded && !e.isProjectSource : !1, Hx = ({ defense: e, checked: t, onToggle: n }) => {
	let [r, i] = d(!1), a = e.attack;
	return /* @__PURE__ */ m("div", {
		className: `border-t border-base border-l-2 transition-opacity ${t ? "" : "opacity-55"}`,
		style: { borderLeftColor: t ? Ia(Pa.green, 70) : "transparent" },
		children: [/* @__PURE__ */ m(F, {
			align: "center",
			gap: "density-sm",
			className: "px-3 py-2",
			children: [
				/* @__PURE__ */ p(Te, {
					checked: t,
					onCheckedChange: n,
					"aria-label": `Include ${e.id}`
				}),
				/* @__PURE__ */ m("button", {
					type: "button",
					onClick: () => i((e) => !e),
					className: "flex min-w-0 flex-1 items-center gap-2 text-left",
					children: [
						/* @__PURE__ */ p("span", {
							className: "w-3 shrink-0 text-subtle",
							children: r ? "▾" : "▸"
						}),
						/* @__PURE__ */ p(le, {
							color: e.kind === "guardrail" ? "green" : "purple",
							children: e.kind === "guardrail" ? "Guardrail" : "Policy"
						}),
						/* @__PURE__ */ p(L, {
							kind: "body/semibold/sm",
							className: "truncate",
							children: e.summary
						})
					]
				}),
				a?.probe ? /* @__PURE__ */ p("div", {
					className: "shrink-0",
					children: /* @__PURE__ */ p(le, {
						color: "yellow",
						children: Tx(a.probe)
					})
				}) : null
			]
		}), r ? /* @__PURE__ */ m(me, {
			cols: {
				base: 1,
				lg: 2
			},
			gap: "density-sm",
			className: "px-3 pb-3 pt-1",
			children: [/* @__PURE__ */ m(I, {
				gap: "density-xxs",
				className: "rounded border p-3",
				style: {
					borderColor: Ia(Pa.red, 30),
					backgroundColor: Ia(Pa.red, 5)
				},
				children: [
					/* @__PURE__ */ m(L, {
						kind: "body/semibold/xs",
						style: { color: Pa.red },
						children: ["ATTACK", a?.probe ? ` · ${Tx(a.probe)}` : ""]
					}),
					a?.goal ? /* @__PURE__ */ p(L, {
						kind: "body/regular/sm",
						children: a.goal
					}) : null,
					a?.prompt_excerpt ? /* @__PURE__ */ p(L, {
						kind: "body/regular/xs",
						className: "whitespace-pre-wrap text-subtle",
						children: wx(a.prompt_excerpt)
					}) : /* @__PURE__ */ p(L, {
						kind: "body/regular/xs",
						className: "text-subtle",
						children: "No linked attack recorded for this defense."
					})
				]
			}), /* @__PURE__ */ m(I, {
				gap: "density-xxs",
				className: "rounded border p-3",
				style: {
					borderColor: Ia(Pa.green, 30),
					backgroundColor: Ia(Pa.green, 5)
				},
				children: [
					/* @__PURE__ */ m(L, {
						kind: "body/semibold/xs",
						style: { color: Pa.green },
						children: ["MITIGATION", e.target_tool ? ` · ${e.target_tool}` : ""]
					}),
					/* @__PURE__ */ p(L, {
						kind: "body/regular/sm",
						children: e.summary
					}),
					e.config_fragment ? /* @__PURE__ */ p("pre", {
						className: "max-h-56 overflow-auto rounded bg-surface-overlay p-2 text-xs text-primary",
						children: e.config_fragment
					}) : null
				]
			})]
		}) : null]
	});
}, Ux = ({ mitigations: e, defenses: t, isLoading: n, workspace: r, runName: i, agentName: a, manifestId: o, hitlogFileset: c, sanityJob: l, onSanityJobChange: u, composedGuardrails: h, onComposedGuardrailsChange: _ }) => {
	let v = Yi(), b = Ji(), [x, S] = d(() => new Set(t.map((e) => e.id)));
	s(() => {
		S(new Set(t.map((e) => e.id)));
	}, [t]);
	let C = Fx(r, i), { submit: w, isPending: T } = Ix(r), E = zx(r, i), D = l ?? E, { data: O } = Ha(r, D ?? ""), ee = O?.status, k = Qs(O), { report: A, isLoading: te, missingReport: j } = Lx(r, D, ee), ne = Rx(r, D, ee), re = h ?? ne, [M, ie] = d(), [N, ae] = d(!1), oe = Yo(), se = xo(r, o ?? "", { query: { enabled: !!o } }), ue = se.data?.source_type === "project";
	if (n) return /* @__PURE__ */ m(F, {
		align: "center",
		gap: "density-sm",
		className: "p-6",
		children: [/* @__PURE__ */ p(we, {
			size: "small",
			"aria-label": "Loading recommendations"
		}), /* @__PURE__ */ p(L, {
			kind: "body/regular/md",
			className: "text-subtle",
			children: "Loading recommendations…"
		})]
	});
	if (t.length === 0 && !e?.guardrails && !e?.policy) return /* @__PURE__ */ p(de, {
		className: "p-6",
		children: /* @__PURE__ */ p(L, {
			kind: "body/regular/md",
			className: "text-subtle",
			children: "No mitigations were produced for this run."
		})
	});
	let fe = [...x], pe = t.length, me = Bx(t), he = (e) => S((t) => {
		let n = new Set(t);
		return n.has(e) ? n.delete(e) : n.add(e), n;
	}), ge = (e) => S((t) => {
		let n = new Set(t), r = e.every((e) => n.has(e.id));
		for (let t of e) r ? n.delete(t.id) : n.add(t.id);
		return n;
	}), _e = async () => {
		if (!e) return null;
		try {
			return await C.mutateAsync({
				mitigations: e,
				selectedDefenseIds: fe
			});
		} catch (e) {
			return oa(v, "Failed to compose the selected defenses")(e), null;
		}
	}, ve = async () => {
		let e = await _e();
		e && ie({ guardrails: e.guardrails_toml ?? void 0 });
	}, ye = async () => {
		if (!c) return v.error("This run has no recorded attacks to replay.");
		if (!o) return v.error("This run has no manifest to validate against.");
		let e = await _e();
		if (e) {
			_(e.guardrails_toml ?? void 0);
			try {
				u(await w({
					manifest_id: o,
					driver: "service",
					validate_only: !0,
					replay_hitlog_fileset: c,
					source_run: i,
					...e.guardrails_toml ? { defense_guardrails: e.guardrails_toml } : {},
					...e.policy_yaml ? { defense_policy: e.policy_yaml } : {}
				})), v.success("Sanity check started — replaying attacks against your selection…");
			} catch (e) {
				oa(v, "Failed to start the sanity check")(e);
			}
		}
	}, be = async () => re ? (await oe.mutateAsync({
		workspace: r,
		name: i,
		data: { guardrails_toml: re }
	}), !0) : !1, xe = C.isPending || T, Se = pe ? Math.round(fe.length / pe * 100) : 0;
	return /* @__PURE__ */ m(I, {
		gap: "density-xl",
		children: [
			/* @__PURE__ */ p(de, {
				className: "p-4",
				children: /* @__PURE__ */ m(I, {
					gap: "density-sm",
					children: [
						/* @__PURE__ */ p(L, {
							kind: "body/semibold/lg",
							children: "Review & apply defenses"
						}),
						/* @__PURE__ */ m(L, {
							kind: "body/regular/sm",
							className: "text-subtle",
							children: [
								pe,
								" defense",
								pe === 1 ? "" : "s",
								" generated across ",
								me.length,
								" tool",
								me.length === 1 ? "" : "s",
								" from this run's attacks. Keep the ones you want, sanity-check the selection, then apply to the agent."
							]
						}),
						/* @__PURE__ */ p("div", {
							className: "flex h-2 w-full gap-0.5",
							"aria-label": `${Se}% of defenses selected`,
							children: t.map((e, t) => /* @__PURE__ */ p("div", {
								className: "flex-1 rounded-sm bg-surface-raised transition-colors",
								style: t < fe.length ? { backgroundColor: Pa.green } : void 0
							}, e.id))
						}),
						/* @__PURE__ */ m(F, {
							justify: "between",
							align: "center",
							children: [/* @__PURE__ */ m(L, {
								kind: "body/regular/sm",
								className: "text-primary",
								children: [
									fe.length,
									" of ",
									pe,
									" selected"
								]
							}), /* @__PURE__ */ m(F, {
								gap: "density-sm",
								children: [/* @__PURE__ */ p(P, {
									kind: "secondary",
									size: "small",
									onClick: () => S(new Set(t.map((e) => e.id))),
									children: "All"
								}), /* @__PURE__ */ p(P, {
									kind: "secondary",
									size: "small",
									onClick: () => S(/* @__PURE__ */ new Set()),
									children: "None"
								})]
							})]
						})
					]
				})
			}),
			/* @__PURE__ */ p(de, {
				className: "p-0",
				children: me.map((e, t) => {
					let n = e.items.every((e) => x.has(e.id));
					return /* @__PURE__ */ m("div", {
						className: t > 0 ? "border-t border-base" : "",
						children: [/* @__PURE__ */ m(F, {
							align: "center",
							justify: "between",
							className: "bg-surface-sunken px-3 py-2",
							children: [/* @__PURE__ */ m(F, {
								align: "center",
								gap: "density-sm",
								children: [/* @__PURE__ */ p(L, {
									kind: "body/semibold/sm",
									className: "font-mono text-primary",
									children: e.tool
								}), /* @__PURE__ */ p(le, {
									color: "gray",
									children: e.items.length
								})]
							}), /* @__PURE__ */ p(Te, {
								checked: n,
								onCheckedChange: () => ge(e.items),
								"aria-label": `Toggle all ${e.tool}`
							})]
						}), /* @__PURE__ */ p("div", { children: e.items.map((e) => /* @__PURE__ */ p(Hx, {
							defense: e,
							checked: x.has(e.id),
							onToggle: () => he(e.id)
						}, e.id)) })]
					}, e.tool);
				})
			}),
			/* @__PURE__ */ m(F, {
				justify: "between",
				align: "center",
				className: "sticky bottom-0 border-t border-base bg-surface-base py-3",
				children: [/* @__PURE__ */ m(L, {
					kind: "body/regular/sm",
					className: "text-primary",
					children: [
						fe.length,
						" of ",
						pe,
						" selected"
					]
				}), /* @__PURE__ */ m(F, {
					gap: "density-sm",
					align: "center",
					children: [
						c ? null : /* @__PURE__ */ p(L, {
							kind: "body/regular/xs",
							className: "text-subtle",
							children: "No recorded attacks to replay"
						}),
						/* @__PURE__ */ p(P, {
							kind: "secondary",
							onClick: ve,
							disabled: xe,
							children: "Preview composed"
						}),
						/* @__PURE__ */ p(P, {
							kind: "primary",
							onClick: ye,
							disabled: xe || !c,
							children: "▶ Run sanity check"
						})
					]
				})]
			}),
			M?.guardrails && e?.guardrails ? /* @__PURE__ */ p(ce, {
				multiple: !0,
				defaultValue: ["preview"],
				children: /* @__PURE__ */ p(g, {
					value: "preview",
					title: "Composed guardrails (your selection)",
					children: /* @__PURE__ */ p(f_, {
						before: e.guardrails.before,
						after: M.guardrails,
						language: "toml"
					})
				})
			}) : null,
			D ? /* @__PURE__ */ m(I, {
				gap: "density-md",
				children: [/* @__PURE__ */ p(L, {
					kind: "body/semibold/lg",
					children: "Sanity check"
				}), A ? /* @__PURE__ */ m(f, { children: [/* @__PURE__ */ p(O_, { report: A }), /* @__PURE__ */ m(F, {
					justify: "end",
					align: "center",
					gap: "density-sm",
					className: "min-w-0",
					children: [ue ? /* @__PURE__ */ p(L, {
						kind: "body/regular/sm",
						className: "text-subtle",
						children: "This manifest brings its own image, so there is no registered agent to update — copy the guardrails into your project instead."
					}) : null, /* @__PURE__ */ p(P, {
						kind: "primary",
						size: "small",
						onClick: () => ae(!0),
						disabled: !Vx({
							hasComposedGuardrails: !!re,
							hasManifestId: !!o,
							manifestQuerySucceeded: se.isSuccess,
							isProjectSource: ue
						}),
						children: "Apply to Agent"
					})]
				})] }) : j ? /* @__PURE__ */ p(I, {
					gap: "density-xs",
					className: "p-4",
					children: k && D ? /* @__PURE__ */ p(Na, {
						title: "The sanity check did not complete",
						failure: k,
						workspace: r,
						jobName: D
					}) : /* @__PURE__ */ p(L, {
						kind: "body/regular/md",
						className: "text-feedback-danger",
						children: "The sanity check finished without producing a report — it may have failed or been cancelled. Try running it again."
					})
				}) : /* @__PURE__ */ m(F, {
					align: "center",
					gap: "density-sm",
					className: "p-4",
					children: [/* @__PURE__ */ p(we, {
						size: "small",
						"aria-label": "Running sanity check"
					}), /* @__PURE__ */ p(L, {
						kind: "body/regular/md",
						className: "text-subtle",
						children: te ? "Replaying attacks + benign requests against your selection…" : "Starting sanity check…"
					})]
				})]
			}) : null,
			/* @__PURE__ */ p(y, {
				onNotify: b,
				open: N,
				onClose: () => ae(!1),
				onConfirm: be,
				title: a ? `Apply selected defenses to ${a}?` : "Apply selected defenses?",
				description: "This records your selected guardrails on the agent as Relay components. Redeploy the agent afterward to activate them.",
				submitButtonText: "Apply",
				successText: "Applied. Redeploy the agent to activate the guardrails.",
				errorText: "Could not apply the selected defenses to the agent."
			})
		]
	});
}, Wx = {
	status_started: "lifecycle",
	status_completed: "lifecycle",
	output: "lifecycle",
	round_started: "round",
	round_completed: "round",
	iteration_started: "round",
	iteration_completed: "round",
	report_written: "round",
	phase_started: "phase",
	phase_completed: "phase",
	victim_control_started: "deploy",
	victim_control_completed: "deploy",
	openshell_upload: "deploy",
	relay_policy_upload: "deploy",
	victim_warning: "deploy",
	preloaded_attacks_loaded: "attack",
	attackers_completed: "attack",
	attack_summary: "attack",
	artifact_written: "attack",
	attacker_summaries_prepared: "defense",
	defender_summary: "defense",
	policy_patches_aggregated: "defense",
	agent_started: "agent",
	agent_progress: "agent",
	agent_completed: "agent",
	agent_failed: "agent",
	agent_exchange: "agent",
	llm_call: "agent",
	synth_phase: "synth",
	interview_started: "synth",
	interview_completed: "synth"
}, Gx = (e) => Wx[e], Kx = {
	lifecycle: Pa.gray,
	round: Pa.blue,
	phase: Pa.teal,
	deploy: Pa.purple,
	attack: Pa.red,
	defense: Pa.green,
	agent: Pa.yellow,
	synth: Pa.teal
}, qx = {
	analyzer: Pa.purple,
	attacker: Pa.red,
	defender: Pa.green,
	victim: Pa.blue,
	validator: Pa.yellow,
	update: Pa.teal,
	summary: Pa.gray
}, Jx = Object.fromEntries(Is.map((e) => [e.id, e.group])), Yx = (e) => typeof e == "string" ? e : void 0, Xx = (e) => typeof e == "number" ? e : void 0, Zx = (e) => e.replace(/_/g, " ").replace(/^\w/, (e) => e.toUpperCase()), Qx = (e) => e.event === "output" ? Yx(e.payload.level) : void 0, $x = {
	ok: Pa.green,
	warn: Pa.yellow,
	fail: Pa.red
}, eS = (e) => Qx(e) !== "detail", tS = (e) => $x[Qx(e) ?? ""], nS = (e) => {
	let t = tS(e);
	if (t) return t;
	let n = Us(e.payload);
	return n && Jx[n] ? qx[Jx[n]] : Kx[Gx(e.event) ?? "lifecycle"];
}, rS = (e) => Object.entries(e).filter(([, e]) => typeof e == "string" || typeof e == "number" || typeof e == "boolean").map(([e, t]) => `${e}=${String(t)}`).join(" "), iS = (e) => {
	let t = e.payload, n = Yx(t.agent_name) ?? "Agent";
	switch (e.event) {
		case "output": return Yx(t.line) ?? "";
		case "status_started":
		case "status_completed": return Yx(t.label) ?? Zx(e.event);
		case "phase_started": return `Phase started: ${Yx(t.phase) ?? ""}`;
		case "phase_completed": return `Phase completed: ${Yx(t.phase) ?? ""}`;
		case "agent_started": return `${n} started`;
		case "agent_progress": return `${n}: ${Yx(t.message) ?? ""}`;
		case "agent_completed": {
			let e = Xx(t.duration_seconds), r = e === void 0 ? "" : ` (${e.toFixed(1)}s)`, i = Yx(t.not_applicable);
			return i ? `○ ${n} did not apply: ${i}${r}` : `${t.ok === !1 ? "✗" : "✓"} ${n}${r}`;
		}
		case "agent_failed": return `✗ ${n} failed: ${Yx(t.error) ?? ""}`;
		case "agent_exchange": return `${n} → victim${Yx(t.label) ? ` [${Yx(t.label)}]` : ""}${t.blocked === !0 ? " (blocked)" : t.blocked === !1 ? " (allowed)" : ""}`;
		case "llm_call": return `${n} · LLM call${Yx(t.label) ? ` (${Yx(t.label)})` : ""}`;
		case "round_started": return `Round ${Xx(t.round) ?? ""} started`.trim();
		case "round_completed": return t.success === !0 ? "Round passed" : "Round completed";
		case "report_written": return "Report written";
		case "attack_summary": return `Attack summary (${Array.isArray(t.attacks) ? t.attacks.length : 0} attacker(s))`;
		case "defender_summary": return `Defender summary (${Array.isArray(t.defenders) ? t.defenders.length : 0} defender(s))`;
		case "synth_phase": return Ks(t, Yx(t.phase) ?? "Recon step");
		case "interview_started": return "Interview started";
		case "interview_completed": return "Interview completed";
		case "victim_control_started": return "Deploying victim…";
		case "victim_control_completed": return "Victim deployed";
		default: {
			let n = rS(t);
			return n ? `${Zx(e.event)} — ${n}` : Zx(e.event);
		}
	}
}, aS = (e) => e ? new Date(e).toLocaleTimeString([], { hour12: !1 }) : "", oS = ({ events: e }) => {
	let t = l(() => e.filter(eS), [e]), n = u(null), r = u(!0);
	c(() => {
		let e = n.current;
		e && r.current && (e.scrollTop = e.scrollHeight);
	}, [t]);
	let i = () => {
		let e = n.current;
		e && (r.current = e.scrollHeight - e.scrollTop - e.clientHeight < 40);
	}, a = l(() => {
		let e = /* @__PURE__ */ new Map();
		for (let n of t) {
			let t = Us(n.payload);
			t && (n.event === "agent_started" ? e.set(t, Yx(n.payload.agent_name) ?? t) : (n.event === "agent_completed" || n.event === "agent_failed") && e.delete(t));
		}
		return Array.from(e.entries());
	}, [t]);
	return t.length === 0 ? /* @__PURE__ */ p(L, {
		kind: "body/regular/md",
		className: "text-subtle",
		children: "Waiting for live events…"
	}) : /* @__PURE__ */ m("div", {
		className: "flex h-full flex-col",
		children: [a.length > 0 ? /* @__PURE__ */ m(F, {
			align: "center",
			gap: "density-xs",
			className: "mb-2 shrink-0 flex-wrap",
			children: [/* @__PURE__ */ p(L, {
				kind: "body/regular/sm",
				className: "text-subtle",
				children: "Now running:"
			}), a.map(([e, t]) => /* @__PURE__ */ p("span", {
				className: "rounded-full bg-surface-sunken px-2 py-0.5 text-xs font-medium",
				style: { color: qx[Jx[e]] ?? Pa.gray },
				children: t
			}, e))]
		}) : null, /* @__PURE__ */ p("div", {
			ref: n,
			onScroll: i,
			className: "min-h-0 flex-1 overflow-auto pr-density-xs",
			children: t.map((e) => {
				let t = Us(e.payload), n = t ? a.some(([e]) => e === t) : !1;
				return /* @__PURE__ */ m("div", {
					className: `flex items-baseline gap-2.5 border-t border-base px-1 py-1.5 ${n ? "bg-surface-sunken" : ""}`,
					children: [
						/* @__PURE__ */ p("span", {
							className: "shrink-0 pt-1 text-xs tabular-nums text-subtle",
							children: aS(e.ts)
						}),
						/* @__PURE__ */ p("span", {
							className: "shrink-0 text-xs leading-normal",
							style: { color: nS(e) },
							"aria-hidden": !0,
							children: "●"
						}),
						/* @__PURE__ */ p(L, {
							kind: "body/regular/sm",
							className: `min-w-0 break-words ${n || Qx(e) === "section" ? "font-semibold" : ""}`,
							style: tS(e) ? { color: tS(e) } : void 0,
							children: iS(e)
						})
					]
				}, e.id);
			})
		})]
	});
}, sS = ({ color: e }) => /* @__PURE__ */ p("svg", {
	width: 10,
	height: 10,
	"aria-hidden": !0,
	children: /* @__PURE__ */ p("circle", {
		cx: 5,
		cy: 5,
		r: 5,
		fill: e
	})
}), cS = {
	pending: "Idle",
	running: "Running",
	success: "Succeeded",
	blocked: "Blocked",
	failed: "Failed",
	skipped: "Did not apply"
}, lS = (e, t) => e === "failed" ? "#ff3855" : e === "blocked" ? "#ffab40" : t, uS = [
	"defender",
	"validator",
	"analyzer"
], dS = ({ children: e }) => /* @__PURE__ */ p(L, {
	kind: "body/semibold/sm",
	className: "uppercase tracking-wide text-subtle",
	children: e
}), fS = () => /* @__PURE__ */ p("span", {
	className: "rounded px-2 py-1 text-xs font-semibold uppercase tracking-wide",
	style: {
		color: Pa.blue,
		backgroundColor: Ia(Pa.blue, 15)
	},
	children: "Victim"
}), pS = ({ node: e, swarm: t }) => {
	let n = Is.filter((t) => t.group === e.group && !t.isManager), r = n.reduce((e, n) => e + (t.nodeExchanges[n.id]?.length ?? 0), 0);
	return /* @__PURE__ */ m(I, {
		gap: "density-xs",
		children: [
			/* @__PURE__ */ p(dS, { children: "Swarm" }),
			n.map((e) => {
				let n = t.statuses[e.id] ?? "pending", r = t.nodeExchanges[e.id]?.length ?? 0;
				return /* @__PURE__ */ m(F, {
					className: "items-center justify-between",
					children: [/* @__PURE__ */ m(F, {
						className: "items-center gap-2",
						children: [/* @__PURE__ */ p(sS, { color: lS(n, Fs[e.group]) }), /* @__PURE__ */ p(L, {
							kind: "body/regular/sm",
							children: e.title
						})]
					}), /* @__PURE__ */ m(L, {
						kind: "body/regular/sm",
						className: "text-subtle",
						children: [cS[n], r ? ` · ${r} prompt${r === 1 ? "" : "s"}` : ""]
					})]
				}, e.id);
			}),
			/* @__PURE__ */ m(L, {
				kind: "body/regular/sm",
				className: "text-subtle",
				children: [
					n.length,
					" agent",
					n.length === 1 ? "" : "s",
					" · ",
					r,
					" exchange",
					r === 1 ? "" : "s"
				]
			})
		]
	});
}, mS = ({ node: e, swarm: t }) => {
	if (!e) return /* @__PURE__ */ p(L, {
		kind: "body/regular/md",
		className: "text-subtle",
		children: "Select an agent in the graph to inspect its activity, logs, and prompts."
	});
	let n = Fs[e.group], r = t.statuses[e.id] ?? "pending", i = t.nodeLogs[e.id] ?? [], a = t.nodeExchanges[e.id] ?? [], o = uS.includes(e.group) ? [] : t.nodeLlmCalls[e.id] ?? [], s = /* @__PURE__ */ m(I, {
		gap: "density-sm",
		children: [
			/* @__PURE__ */ p(L, {
				kind: "body/semibold/lg",
				children: e.title
			}),
			/* @__PURE__ */ m(F, {
				className: "items-center gap-2",
				children: [/* @__PURE__ */ p(sS, { color: n }), /* @__PURE__ */ m(L, {
					kind: "body/regular/sm",
					children: [e.group, e.isManager ? " · manager" : ""]
				})]
			}),
			/* @__PURE__ */ m(F, {
				className: "items-center gap-2",
				children: [/* @__PURE__ */ p(sS, { color: lS(r, n) }), /* @__PURE__ */ p(L, {
					kind: "body/regular/md",
					children: cS[r]
				})]
			})
		]
	});
	return e.isManager ? /* @__PURE__ */ m(I, {
		gap: "density-lg",
		className: "min-h-0",
		children: [s, /* @__PURE__ */ p(pS, {
			node: e,
			swarm: t
		})]
	}) : /* @__PURE__ */ m(I, {
		gap: "density-lg",
		className: "min-h-0",
		children: [
			s,
			/* @__PURE__ */ m(I, {
				gap: "density-xs",
				children: [/* @__PURE__ */ p(dS, { children: "Activity" }), i.length === 0 ? /* @__PURE__ */ p(L, {
					kind: "body/regular/sm",
					className: "text-subtle",
					children: "No activity yet."
				}) : /* @__PURE__ */ p(I, {
					gap: "density-xs",
					className: "font-mono",
					children: i.map((e, t) => /* @__PURE__ */ m(L, {
						kind: "body/regular/sm",
						style: e.level === "error" ? { color: Fa.danger } : void 0,
						children: [/* @__PURE__ */ p("span", {
							className: "text-subtle",
							children: e.label
						}), e.text ? ` ${e.text}` : ""]
					}, t))
				})]
			}),
			/* @__PURE__ */ m(I, {
				gap: "density-xs",
				children: [/* @__PURE__ */ p(dS, { children: `Prompts (${a.length})` }), a.length === 0 ? /* @__PURE__ */ p(L, {
					kind: "body/regular/sm",
					className: "text-subtle",
					children: "No prompts yet — they appear as this agent runs."
				}) : /* @__PURE__ */ p(I, {
					gap: "density-sm",
					children: a.map((e, t) => /* @__PURE__ */ m(I, {
						gap: "density-xs",
						className: "rounded-md border border-base p-2",
						children: [
							/* @__PURE__ */ m(F, {
								className: "items-center justify-between",
								children: [e.label ? /* @__PURE__ */ p(L, {
									kind: "body/semibold/sm",
									className: e.ok ? "text-subtle" : void 0,
									style: e.ok ? void 0 : { color: Fa.danger },
									children: e.label
								}) : /* @__PURE__ */ p("span", {}), e.blocked === void 0 ? null : /* @__PURE__ */ p(le, {
									color: e.blocked ? "red" : "green",
									children: e.blocked ? "Blocked" : "Allowed"
								})]
							}),
							/* @__PURE__ */ p(dS, { children: "Request" }),
							/* @__PURE__ */ p(w, {
								message: e.request || "(empty)",
								characterLimit: 220
							}),
							/* @__PURE__ */ m(F, {
								className: "items-center gap-2",
								children: [/* @__PURE__ */ p(dS, { children: "Response" }), /* @__PURE__ */ p(fS, {})]
							}),
							/* @__PURE__ */ p(w, {
								message: e.response || "(empty)",
								characterLimit: 220
							})
						]
					}, t))
				})]
			}),
			o.length > 0 ? /* @__PURE__ */ m(I, {
				gap: "density-xs",
				children: [/* @__PURE__ */ p(dS, { children: `LLM calls (${o.length})` }), /* @__PURE__ */ p(I, {
					gap: "density-sm",
					children: o.map((e, t) => /* @__PURE__ */ m(I, {
						gap: "density-xs",
						className: "rounded-md border border-base p-2",
						children: [
							e.label ? /* @__PURE__ */ p(L, {
								kind: "body/regular/sm",
								className: "text-subtle",
								children: e.label
							}) : null,
							/* @__PURE__ */ p(dS, { children: "Prompt" }),
							/* @__PURE__ */ p(w, {
								message: e.request || "(empty)",
								characterLimit: 220
							}),
							/* @__PURE__ */ p(dS, { children: "Completion" }),
							/* @__PURE__ */ p(w, {
								message: e.response || "(empty)",
								characterLimit: 220
							})
						]
					}, t))
				})]
			}) : null
		]
	});
}, hS = 1e3, gS = 720, _S = .6, vS = 3, yS = 4, bS = (e) => Math.min(vS, Math.max(_S, e)), xS = [
	{
		label: "ATTACKER SWARM",
		x: 40,
		y: 90,
		w: 320,
		h: 300,
		color: Fs.attacker
	},
	{
		label: "OPENSHELL SANDBOX",
		x: 400,
		y: 200,
		w: 200,
		h: 220,
		color: Fs.victim
	},
	{
		label: "DEFENDER SWARM",
		x: 640,
		y: 90,
		w: 320,
		h: 300,
		color: Fs.defender
	},
	{
		label: "VALIDATOR SWARM",
		x: 250,
		y: 540,
		w: 400,
		h: 160,
		color: Fs.validator
	}
], SS = (e) => e.group === "victim" ? 40 : e.isManager ? 32 : 26, CS = (e, t) => t === "failed" ? "rgba(255,56,85,0.18)" : t === "blocked" ? "rgba(255,171,64,0.18)" : t === "running" ? `${e}33` : t === "success" ? `${e}44` : "rgba(255,255,255,0.03)", wS = (e, t) => t === "failed" ? "#ff3855" : t === "blocked" ? "#ffab40" : e, TS = ({ swarm: e, selectedId: t, onSelect: n }) => {
	let [r, i] = d(1), [a, o] = d({
		x: hS / 2,
		y: gS / 2
	}), [s, c] = d({}), l = u(null), f = u(null), h = u(null), g = hS / r, _ = gS / r, v = `${a.x - g / 2} ${a.y - _ / 2} ${g} ${_}`, y = (e) => s[e.id] ?? {
		x: e.x,
		y: e.y
	}, b = (e, t, n) => ({
		x: e * g / n.width,
		y: t * _ / n.height
	}), x = (e) => i((t) => bS(t * e)), S = () => {
		i(1), o({
			x: hS / 2,
			y: gS / 2
		}), c({});
	}, C = (e, t) => {
		e.stopPropagation(), h.current = {
			id: t.id,
			base: y(t),
			startX: e.clientX,
			startY: e.clientY
		};
	}, w = (e) => {
		f.current = {
			x: e.clientX,
			y: e.clientY
		};
	}, T = (e) => {
		let t = l.current?.getBoundingClientRect();
		if (!t) return;
		let n = h.current;
		if (n) {
			let r = b(e.clientX - n.startX, e.clientY - n.startY, t);
			c((e) => ({
				...e,
				[n.id]: {
					x: n.base.x + r.x,
					y: n.base.y + r.y
				}
			}));
			return;
		}
		let r = f.current;
		if (!r) return;
		let i = b(e.clientX - r.x, e.clientY - r.y, t);
		f.current = {
			x: e.clientX,
			y: e.clientY
		}, o((e) => ({
			x: e.x - i.x,
			y: e.y - i.y
		}));
	}, E = (e) => {
		let t = h.current;
		t && (Math.hypot(e.clientX - t.startX, e.clientY - t.startY) > yS || n(t.id), h.current = null), f.current = null;
	};
	return /* @__PURE__ */ m("div", {
		className: "relative h-full w-full",
		children: [/* @__PURE__ */ m("svg", {
			ref: l,
			viewBox: v,
			width: "100%",
			height: "100%",
			preserveAspectRatio: "xMidYMid meet",
			role: "img",
			"aria-label": "Agent Hardener agents",
			className: "cursor-grab touch-none active:cursor-grabbing",
			onPointerDown: w,
			onPointerMove: T,
			onPointerUp: E,
			onPointerLeave: E,
			children: [
				xS.map((e) => /* @__PURE__ */ m("g", { children: [/* @__PURE__ */ p("rect", {
					x: e.x,
					y: e.y,
					width: e.w,
					height: e.h,
					rx: 16,
					fill: `${e.color}0d`,
					stroke: `${e.color}44`,
					strokeDasharray: "6 6"
				}), /* @__PURE__ */ p("text", {
					x: e.x + 12,
					y: e.y + 22,
					fill: e.color,
					fontSize: 13,
					fontFamily: "monospace",
					letterSpacing: 1.5,
					children: e.label
				})] }, e.label)),
				Ls.map((t, n) => {
					let r = Is.find((e) => e.id === t.from), i = Is.find((e) => e.id === t.to);
					if (!r || !i) return null;
					let a = y(r), o = y(i), s = e.statuses[t.to] === "running", c = `iron-edge-${n}`, l = `M ${a.x} ${a.y} L ${o.x} ${o.y}`;
					return /* @__PURE__ */ m("g", { children: [/* @__PURE__ */ p("path", {
						id: c,
						d: l,
						fill: "none",
						stroke: s ? Fs.victim : "rgba(255,255,255,0.12)",
						strokeWidth: s ? 2 : 1
					}), s && /* @__PURE__ */ p("circle", {
						r: 4,
						fill: Fs.victim,
						children: /* @__PURE__ */ p("animateMotion", {
							dur: "1.6s",
							repeatCount: "indefinite",
							children: /* @__PURE__ */ p("mpath", { href: `#${c}` })
						})
					})] }, c);
				}),
				Is.map((n) => {
					let r = e.statuses[n.id] ?? "pending", i = Fs[n.group], a = SS(n), o = t === n.id, s = y(n), c = n.isManager ? 0 : e.nodeExchanges[n.id]?.length ?? 0;
					return /* @__PURE__ */ m("g", {
						onPointerDown: (e) => C(e, n),
						className: "cursor-grab active:cursor-grabbing",
						children: [
							r === "running" && /* @__PURE__ */ m("circle", {
								cx: s.x,
								cy: s.y,
								r: a,
								fill: "none",
								stroke: i,
								strokeWidth: 1.5,
								opacity: .6,
								children: [/* @__PURE__ */ p("animate", {
									attributeName: "r",
									values: `${a};${a + 12}`,
									dur: "1.4s",
									repeatCount: "indefinite"
								}), /* @__PURE__ */ p("animate", {
									attributeName: "opacity",
									values: "0.6;0",
									dur: "1.4s",
									repeatCount: "indefinite"
								})]
							}),
							/* @__PURE__ */ p("circle", {
								cx: s.x,
								cy: s.y,
								r: a,
								fill: CS(i, r),
								stroke: wS(i, r),
								strokeWidth: o ? 4 : 2,
								opacity: r === "pending" ? .5 : 1
							}),
							/* @__PURE__ */ p("text", {
								x: s.x,
								y: s.y + a + 16,
								textAnchor: "middle",
								fill: "#c9d6de",
								fontSize: 12,
								children: n.title
							}),
							n.isManager && /* @__PURE__ */ p("text", {
								x: s.x,
								y: s.y - a - 8,
								textAnchor: "middle",
								fill: i,
								fontSize: 9,
								fontFamily: "monospace",
								letterSpacing: 1,
								children: "MANAGER"
							}),
							c > 0 && /* @__PURE__ */ m("g", { children: [/* @__PURE__ */ p("circle", {
								cx: s.x + a * .72,
								cy: s.y - a * .72,
								r: 9,
								fill: i
							}), /* @__PURE__ */ p("text", {
								x: s.x + a * .72,
								y: s.y - a * .72 + 3,
								textAnchor: "middle",
								fill: "#0b0f14",
								fontSize: 9,
								fontFamily: "monospace",
								children: c
							})] })
						]
					}, n.id);
				})
			]
		}), /* @__PURE__ */ m("div", {
			className: "absolute right-1 top-1 flex flex-col gap-1",
			children: [
				/* @__PURE__ */ p(P, {
					kind: "secondary",
					size: "small",
					"aria-label": "Zoom in",
					onClick: () => x(1.2),
					children: /* @__PURE__ */ p(Ca, { className: "h-4 w-4" })
				}),
				/* @__PURE__ */ p(P, {
					kind: "secondary",
					size: "small",
					"aria-label": "Zoom out",
					onClick: () => x(1 / 1.2),
					children: /* @__PURE__ */ p(xa, { className: "h-4 w-4" })
				}),
				/* @__PURE__ */ p(P, {
					kind: "secondary",
					size: "small",
					"aria-label": "Reset view",
					onClick: S,
					children: /* @__PURE__ */ p(ba, { className: "h-4 w-4" })
				})
			]
		})]
	});
}, ES = 560, DS = 420, OS = {
	good: Pa.green,
	active: Pa.teal
}, kS = ({ label: e, tone: t }) => {
	let n = t ? OS[t] : void 0;
	return /* @__PURE__ */ p("span", {
		className: `rounded-full border px-3 py-1 text-xs uppercase tracking-wide ${n ? "" : "border-base text-subtle"}`,
		style: n ? {
			color: n,
			borderColor: Ia(n, 40)
		} : void 0,
		children: e
	});
}, AS = () => {
	let e = qi(), { agentHardenerRunName: t = "" } = Le(), [n, r] = d(null), { data: i } = Go(e, t, { query: {
		enabled: !!t,
		refetchInterval: (e) => e.state.data?.status === "running" && D
	} }), a = Ys(e, t, !!i?.status && i?.status !== "running"), o = l(() => Ws(a), [a]), c = i?.job_id ?? "", { useJobsUpdateJobAttemptStatusDetails: u } = Va(), { data: f } = Ha(e, c), g = u(), _ = f?.status_details, v = Ns(_), y = Ps(_), { mitigations: b, recommendations: x, defenses: S, isLoading: C, hasMitigations: w } = Mx(e, c, f?.status);
	Xi({ items: [{
		href: $i(e),
		slotLabel: "Agent Hardening"
	}, { slotLabel: t }] });
	let T = (t) => f && g.mutate({
		workspace: e,
		job: c,
		attempt: f.attempt_id,
		data: t
	}), E = (e) => v && T({ interview_response: {
		round: v.round,
		answers: e
	} }), O = (e) => y && T({ review_response: {
		round: y.round,
		suite: e
	} }), ee = Is.find((e) => e.id === n) ?? null, k = !!(v || y), [A, te] = d("swarm"), [j, ne] = d(), [re, M] = d();
	s(() => {
		k && te("interview");
	}, [k]), s(() => {
		w && !k && te("mitigations");
	}, [w, k]);
	let ie = /* @__PURE__ */ m(me, {
		cols: {
			base: 1,
			xl: 2
		},
		gap: "density-xl",
		children: [/* @__PURE__ */ p(de, {
			className: "p-2",
			style: { height: ES },
			children: /* @__PURE__ */ p(TS, {
				swarm: o,
				selectedId: n,
				onSelect: r
			})
		}), /* @__PURE__ */ m(I, {
			gap: "density-xl",
			className: "min-h-0",
			style: { height: ES },
			children: [/* @__PURE__ */ p(de, {
				className: "min-h-0 overflow-auto p-4",
				style: { flex: 5 },
				children: /* @__PURE__ */ p(mS, {
					node: ee,
					swarm: o
				})
			}), /* @__PURE__ */ m("div", {
				className: "flex min-h-0 flex-col rounded-md border border-base bg-surface-raised p-4",
				style: { flex: 6 },
				children: [/* @__PURE__ */ p(L, {
					kind: "body/semibold/md",
					className: "mb-2 shrink-0",
					children: "Live Agent Feed"
				}), /* @__PURE__ */ p("div", {
					className: "min-h-0 flex-1",
					children: /* @__PURE__ */ p(oS, { events: a })
				})]
			})]
		})]
	}), N = f?.error_details?.message, ae = i?.error_message || (typeof N == "string" ? N : "") || i?.summary || "War-game run failed.";
	return /* @__PURE__ */ p(h, {
		title: `Agent Hardening — ${t}`,
		children: /* @__PURE__ */ m(I, {
			className: "min-h-full",
			gap: "density-xl",
			padding: "density-2xl",
			children: [
				/* @__PURE__ */ p(he, {
					className: "p-0",
					slotHeading: i?.agent ? `Hardening ${i.agent}` : "War-Game Run",
					slotDescription: i?.summary,
					slotActions: /* @__PURE__ */ m(F, {
						gap: "density-sm",
						align: "center",
						children: [
							o.round > 0 && /* @__PURE__ */ p(kS, { label: `Round ${o.round}` }),
							o.phase && /* @__PURE__ */ p(kS, {
								label: o.phase,
								tone: o.finalPass ? "good" : "active"
							}),
							i?.job_id && /* @__PURE__ */ p(fc, {
								workspace: e,
								jobName: i.job_id,
								jobStatus: f?.status,
								compact: !0
							})
						]
					})
				}),
				i?.status === "failed" && /* @__PURE__ */ p(ue, {
					kind: "inline",
					status: "error",
					children: /* @__PURE__ */ m(I, {
						gap: "density-xs",
						children: [/* @__PURE__ */ p(L, {
							kind: "body/semibold/md",
							children: ae
						}), i.error_remediation ? /* @__PURE__ */ p(L, {
							kind: "body/regular/sm",
							children: i.error_remediation
						}) : null]
					})
				}),
				k || w ? /* @__PURE__ */ m(Oe, {
					value: A,
					onValueChange: te,
					children: [
						/* @__PURE__ */ m(De, { children: [
							/* @__PURE__ */ p(ke, {
								value: "swarm",
								children: "Swarm"
							}),
							k ? /* @__PURE__ */ p(ke, {
								value: "interview",
								children: /* @__PURE__ */ m(F, {
									gap: "density-xs",
									align: "center",
									children: ["Interview", /* @__PURE__ */ p(le, {
										color: "yellow",
										children: "Action required"
									})]
								})
							}) : null,
							w ? /* @__PURE__ */ p(ke, {
								value: "mitigations",
								children: /* @__PURE__ */ m(F, {
									gap: "density-xs",
									align: "center",
									children: ["Harden", (S.length || x.length) > 0 && /* @__PURE__ */ p(le, {
										color: "green",
										children: S.length || x.length
									})]
								})
							}) : null
						] }),
						/* @__PURE__ */ p(Ee, {
							value: "swarm",
							className: "p-0 pt-4",
							children: ie
						}),
						k ? /* @__PURE__ */ p(Ee, {
							value: "interview",
							className: "p-0 pt-4",
							children: /* @__PURE__ */ p(de, {
								className: "p-6",
								style: { minHeight: DS },
								children: v ? /* @__PURE__ */ p(Ma, {
									prompt: v,
									loading: g.isPending,
									onSubmit: E
								}, v.round) : y ? /* @__PURE__ */ p(Ba, {
									suite: y.suite,
									loading: g.isPending,
									onSubmit: O
								}, y.round) : null
							})
						}) : null,
						w ? /* @__PURE__ */ p(Ee, {
							value: "mitigations",
							className: "p-0 pt-4",
							children: /* @__PURE__ */ p(Ux, {
								mitigations: b,
								defenses: S,
								isLoading: C,
								workspace: e,
								runName: t,
								agentName: i?.agent,
								manifestId: i?.manifest_id,
								hitlogFileset: i?.hitlog_fileset,
								sanityJob: j,
								onSanityJobChange: ne,
								composedGuardrails: re,
								onComposedGuardrailsChange: M
							})
						}) : null
					]
				}) : ie
			]
		})
	});
}, jS = {
	running: {
		label: "Running",
		color: "blue"
	},
	completed: {
		label: "Completed",
		color: "green"
	},
	failed: {
		label: "Failed",
		color: "red"
	}
}, MS = [
	{
		label: "Running",
		value: "running"
	},
	{
		label: "Completed",
		value: "completed"
	},
	{
		label: "Failed",
		value: "failed"
	}
], NS = () => {
	let e = Ie(), t = qi(), n = Yi(), r = Ji(), i = Ve(), [a, o] = d(null), s = oe({ defaultSort: [{
		id: "created_at",
		desc: !0
	}] }), c = () => i.invalidateQueries({ queryKey: zo(t) }), u = ze({
		mutationFn: As,
		onSuccess: () => {
			n.success("Run job cancelled."), c();
		},
		onError: oa(n, "Failed to cancel the run job")
	}), { data: h, isLoading: g } = Vo(t, {
		sort: M(s.sorting.state),
		page: s.pagination.state.pageIndex + 1,
		page_size: s.pagination.state.pageSize,
		filter: {
			...s.apiFilter.filter ?? {},
			...s.apiFilter.searchText ? se({ agent: { $like: s.apiFilter.searchText } }) : {}
		}
	}, { query: {
		placeholderData: Re,
		refetchInterval: D,
		refetchOnMount: "always",
		retry: !1
	} }), _ = l(() => (h?.data ?? []).map((e) => ({
		...e,
		id: e.id || `${e.workspace ?? ""}/${e.name ?? ""}`
	})), [h]), v = h?.pagination?.total_results ?? 0;
	return /* @__PURE__ */ m(f, { children: [/* @__PURE__ */ p(te, {
		dataViewState: s,
		searchField: "agent",
		makeColumns: ({ accessor: e }, { rowActionsColumn: n }) => [
			e("name", {
				header: "Run",
				cell: ({ row: e }) => e.original.name ?? "-"
			}),
			e("agent", {
				header: "Agent",
				cell: ({ row: e }) => /* @__PURE__ */ p(L, {
					className: "truncate",
					style: { maxWidth: 240 },
					kind: "body/regular/md",
					children: e.original.agent || "-"
				})
			}),
			e("status", {
				header: "Status",
				size: 125,
				meta: { filter: {
					type: "single-select",
					label: "Status",
					options: MS
				} },
				cell: ({ row: e }) => {
					if (!e.original.status) return null;
					let t = /* @__PURE__ */ p(A, {
						status: e.original.status,
						statusConfig: jS
					});
					return e.original.status === "failed" && e.original.error_message ? /* @__PURE__ */ p("span", {
						title: e.original.error_message,
						children: t
					}) : t;
				}
			}),
			e("created_at", {
				id: "created_at",
				header: "Started",
				enableSorting: !0,
				size: 160,
				cell: ({ row: e }) => e.original.created_at ? /* @__PURE__ */ p(k, { datetime: e.original.created_at }) : null
			}),
			n({
				size: 70,
				cell: ({ row: e }) => {
					let n = e.original.job_id;
					return /* @__PURE__ */ p(O, { actions: [...e.original.status === "running" && n ? [{
						label: "Cancel",
						onSelect: () => u.mutate({
							workspace: t,
							name: n
						})
					}] : [], {
						label: "Delete",
						onSelect: () => o(e.original)
					}] });
				}
			})
		],
		onRowClick: (n) => n.name && e(ea(t, n.name)),
		attributes: {
			DataViewSearchBar: { placeholder: "Search by agent..." },
			DataViewRoot: {
				data: _,
				totalCount: v,
				requestStatus: g && !h ? "loading" : void 0
			},
			DataViewTableContent: { renderEmptyState: () => /* @__PURE__ */ p(j, {
				header: "No war-game runs yet",
				emptyMessage: "Agent Hardener runs appear here once you harden an agent from the CLI or a submitted job."
			}) }
		}
	}), /* @__PURE__ */ p(C, {
		onNotify: r,
		open: !!a,
		onClose: () => o(null),
		title: `Delete ${a?.name ?? "run"}?`,
		description: "This permanently deletes the run record and its platform job.",
		successText: "Run deleted.",
		errorText: "Failed to delete the run.",
		onDelete: async () => a?.name ? (await js(t, {
			name: a.name,
			job_id: a.job_id
		}), c(), !0) : !1
	})] });
}, PS = () => {
	let e = qi();
	return Xi({ items: [{ slotLabel: "Agent Hardening" }] }), /* @__PURE__ */ m(h, {
		title: "Agent Hardening",
		children: [/* @__PURE__ */ m(I, {
			className: "h-full overflow-auto",
			gap: "density-2xl",
			padding: "density-2xl",
			children: [/* @__PURE__ */ p(he, {
				className: "p-0",
				slotHeading: "Agent Hardening",
				slotDescription: "Attack, defend, and validate war-game runs that harden your deployed agents.",
				slotActions: /* @__PURE__ */ p(P, {
					asChild: !0,
					color: "brand",
					children: /* @__PURE__ */ p(Me, {
						to: ta(e),
						children: "Manifests"
					})
				})
			}), /* @__PURE__ */ p(NS, {})]
		}), /* @__PURE__ */ p(Ne, {})]
	});
}, FS = 100, IS = (e) => {
	let { sdk: t } = Ki();
	return Be({
		queryKey: [
			"agent-hardener-init",
			"agents",
			e
		],
		queryFn: async ({ signal: n }) => {
			let r = [], i = 1;
			for (;;) {
				let a = await t.agents.agentsListAgents(e, {
					page: i,
					page_size: FS,
					sort: "name"
				}, n), o = a.data ?? [];
				r.push(...o);
				let s = a.pagination?.total_pages;
				if (s ? i >= s : o.length < FS) break;
				i += 1;
			}
			return r;
		},
		enabled: !!e
	});
}, LS = (e) => (e ?? "").split(",").map((e) => e.trim()).filter(Boolean), RS = (e) => Object.fromEntries(LS(e).map((e) => {
	let t = e.indexOf("=");
	return t > 0 ? [e.slice(0, t).trim(), e.slice(t + 1).trim()] : null;
}).filter((e) => e !== null)), zS;
(function(e) {
	e.assertEqual = (e) => {};
	function t(e) {}
	e.assertIs = t;
	function n(e) {
		throw Error();
	}
	e.assertNever = n, e.arrayToEnum = (e) => {
		let t = {};
		for (let n of e) t[n] = n;
		return t;
	}, e.getValidEnumValues = (t) => {
		let n = e.objectKeys(t).filter((e) => typeof t[t[e]] != "number"), r = {};
		for (let e of n) r[e] = t[e];
		return e.objectValues(r);
	}, e.objectValues = (t) => e.objectKeys(t).map(function(e) {
		return t[e];
	}), e.objectKeys = typeof Object.keys == "function" ? (e) => Object.keys(e) : (e) => {
		let t = [];
		for (let n in e) Object.prototype.hasOwnProperty.call(e, n) && t.push(n);
		return t;
	}, e.find = (e, t) => {
		for (let n of e) if (t(n)) return n;
	}, e.isInteger = typeof Number.isInteger == "function" ? (e) => Number.isInteger(e) : (e) => typeof e == "number" && Number.isFinite(e) && Math.floor(e) === e;
	function r(e, t = " | ") {
		return e.map((e) => typeof e == "string" ? `'${e}'` : e).join(t);
	}
	e.joinValues = r, e.jsonStringifyReplacer = (e, t) => typeof t == "bigint" ? t.toString() : t;
})(zS ||= {});
var BS;
(function(e) {
	e.mergeShapes = (e, t) => ({
		...e,
		...t
	});
})(BS ||= {});
var G = zS.arrayToEnum([
	"string",
	"nan",
	"number",
	"integer",
	"float",
	"boolean",
	"date",
	"bigint",
	"symbol",
	"function",
	"undefined",
	"null",
	"array",
	"object",
	"unknown",
	"promise",
	"void",
	"never",
	"map",
	"set"
]), VS = (e) => {
	switch (typeof e) {
		case "undefined": return G.undefined;
		case "string": return G.string;
		case "number": return Number.isNaN(e) ? G.nan : G.number;
		case "boolean": return G.boolean;
		case "function": return G.function;
		case "bigint": return G.bigint;
		case "symbol": return G.symbol;
		case "object": return Array.isArray(e) ? G.array : e === null ? G.null : e.then && typeof e.then == "function" && e.catch && typeof e.catch == "function" ? G.promise : typeof Map < "u" && e instanceof Map ? G.map : typeof Set < "u" && e instanceof Set ? G.set : typeof Date < "u" && e instanceof Date ? G.date : G.object;
		default: return G.unknown;
	}
}, K = zS.arrayToEnum([
	"invalid_type",
	"invalid_literal",
	"custom",
	"invalid_union",
	"invalid_union_discriminator",
	"invalid_enum_value",
	"unrecognized_keys",
	"invalid_arguments",
	"invalid_return_type",
	"invalid_date",
	"invalid_string",
	"too_small",
	"too_big",
	"invalid_intersection_types",
	"not_multiple_of",
	"not_finite"
]), HS = class e extends Error {
	get errors() {
		return this.issues;
	}
	constructor(e) {
		super(), this.issues = [], this.addIssue = (e) => {
			this.issues = [...this.issues, e];
		}, this.addIssues = (e = []) => {
			this.issues = [...this.issues, ...e];
		};
		let t = new.target.prototype;
		Object.setPrototypeOf ? Object.setPrototypeOf(this, t) : this.__proto__ = t, this.name = "ZodError", this.issues = e;
	}
	format(e) {
		let t = e || function(e) {
			return e.message;
		}, n = { _errors: [] }, r = (e) => {
			for (let i of e.issues) if (i.code === "invalid_union") i.unionErrors.map(r);
			else if (i.code === "invalid_return_type") r(i.returnTypeError);
			else if (i.code === "invalid_arguments") r(i.argumentsError);
			else if (i.path.length === 0) n._errors.push(t(i));
			else {
				let e = n, r = 0;
				for (; r < i.path.length;) {
					let n = i.path[r];
					r === i.path.length - 1 ? (e[n] = e[n] || { _errors: [] }, e[n]._errors.push(t(i))) : e[n] = e[n] || { _errors: [] }, e = e[n], r++;
				}
			}
		};
		return r(this), n;
	}
	static assert(t) {
		if (!(t instanceof e)) throw Error(`Not a ZodError: ${t}`);
	}
	toString() {
		return this.message;
	}
	get message() {
		return JSON.stringify(this.issues, zS.jsonStringifyReplacer, 2);
	}
	get isEmpty() {
		return this.issues.length === 0;
	}
	flatten(e = (e) => e.message) {
		let t = {}, n = [];
		for (let r of this.issues) if (r.path.length > 0) {
			let n = r.path[0];
			t[n] = t[n] || [], t[n].push(e(r));
		} else n.push(e(r));
		return {
			formErrors: n,
			fieldErrors: t
		};
	}
	get formErrors() {
		return this.flatten();
	}
};
HS.create = (e) => new HS(e);
//#endregion
//#region node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/locales/en.js
var US = (e, t) => {
	let n;
	switch (e.code) {
		case K.invalid_type:
			n = e.received === G.undefined ? "Required" : `Expected ${e.expected}, received ${e.received}`;
			break;
		case K.invalid_literal:
			n = `Invalid literal value, expected ${JSON.stringify(e.expected, zS.jsonStringifyReplacer)}`;
			break;
		case K.unrecognized_keys:
			n = `Unrecognized key(s) in object: ${zS.joinValues(e.keys, ", ")}`;
			break;
		case K.invalid_union:
			n = "Invalid input";
			break;
		case K.invalid_union_discriminator:
			n = `Invalid discriminator value. Expected ${zS.joinValues(e.options)}`;
			break;
		case K.invalid_enum_value:
			n = `Invalid enum value. Expected ${zS.joinValues(e.options)}, received '${e.received}'`;
			break;
		case K.invalid_arguments:
			n = "Invalid function arguments";
			break;
		case K.invalid_return_type:
			n = "Invalid function return type";
			break;
		case K.invalid_date:
			n = "Invalid date";
			break;
		case K.invalid_string:
			typeof e.validation == "object" ? "includes" in e.validation ? (n = `Invalid input: must include "${e.validation.includes}"`, typeof e.validation.position == "number" && (n = `${n} at one or more positions greater than or equal to ${e.validation.position}`)) : "startsWith" in e.validation ? n = `Invalid input: must start with "${e.validation.startsWith}"` : "endsWith" in e.validation ? n = `Invalid input: must end with "${e.validation.endsWith}"` : zS.assertNever(e.validation) : n = e.validation === "regex" ? "Invalid" : `Invalid ${e.validation}`;
			break;
		case K.too_small:
			n = e.type === "array" ? `Array must contain ${e.exact ? "exactly" : e.inclusive ? "at least" : "more than"} ${e.minimum} element(s)` : e.type === "string" ? `String must contain ${e.exact ? "exactly" : e.inclusive ? "at least" : "over"} ${e.minimum} character(s)` : e.type === "number" || e.type === "bigint" ? `Number must be ${e.exact ? "exactly equal to " : e.inclusive ? "greater than or equal to " : "greater than "}${e.minimum}` : e.type === "date" ? `Date must be ${e.exact ? "exactly equal to " : e.inclusive ? "greater than or equal to " : "greater than "}${new Date(Number(e.minimum))}` : "Invalid input";
			break;
		case K.too_big:
			n = e.type === "array" ? `Array must contain ${e.exact ? "exactly" : e.inclusive ? "at most" : "less than"} ${e.maximum} element(s)` : e.type === "string" ? `String must contain ${e.exact ? "exactly" : e.inclusive ? "at most" : "under"} ${e.maximum} character(s)` : e.type === "number" ? `Number must be ${e.exact ? "exactly" : e.inclusive ? "less than or equal to" : "less than"} ${e.maximum}` : e.type === "bigint" ? `BigInt must be ${e.exact ? "exactly" : e.inclusive ? "less than or equal to" : "less than"} ${e.maximum}` : e.type === "date" ? `Date must be ${e.exact ? "exactly" : e.inclusive ? "smaller than or equal to" : "smaller than"} ${new Date(Number(e.maximum))}` : "Invalid input";
			break;
		case K.custom:
			n = "Invalid input";
			break;
		case K.invalid_intersection_types:
			n = "Intersection results could not be merged";
			break;
		case K.not_multiple_of:
			n = `Number must be a multiple of ${e.multipleOf}`;
			break;
		case K.not_finite:
			n = "Number must be finite";
			break;
		default: n = t.defaultError, zS.assertNever(e);
	}
	return { message: n };
}, WS = US;
function GS() {
	return WS;
}
//#endregion
//#region node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/helpers/parseUtil.js
var KS = (e) => {
	let { data: t, path: n, errorMaps: r, issueData: i } = e, a = [...n, ...i.path || []], o = {
		...i,
		path: a
	};
	if (i.message !== void 0) return {
		...i,
		path: a,
		message: i.message
	};
	let s = "", c = r.filter((e) => !!e).slice().reverse();
	for (let e of c) s = e(o, {
		data: t,
		defaultError: s
	}).message;
	return {
		...i,
		path: a,
		message: s
	};
};
function q(e, t) {
	let n = GS(), r = KS({
		issueData: t,
		data: e.data,
		path: e.path,
		errorMaps: [
			e.common.contextualErrorMap,
			e.schemaErrorMap,
			n,
			n === US ? void 0 : US
		].filter((e) => !!e)
	});
	e.common.issues.push(r);
}
var qS = class e {
	constructor() {
		this.value = "valid";
	}
	dirty() {
		this.value === "valid" && (this.value = "dirty");
	}
	abort() {
		this.value !== "aborted" && (this.value = "aborted");
	}
	static mergeArray(e, t) {
		let n = [];
		for (let r of t) {
			if (r.status === "aborted") return J;
			r.status === "dirty" && e.dirty(), n.push(r.value);
		}
		return {
			status: e.value,
			value: n
		};
	}
	static async mergeObjectAsync(t, n) {
		let r = [];
		for (let e of n) {
			let t = await e.key, n = await e.value;
			r.push({
				key: t,
				value: n
			});
		}
		return e.mergeObjectSync(t, r);
	}
	static mergeObjectSync(e, t) {
		let n = {};
		for (let r of t) {
			let { key: t, value: i } = r;
			if (t.status === "aborted" || i.status === "aborted") return J;
			t.status === "dirty" && e.dirty(), i.status === "dirty" && e.dirty(), t.value !== "__proto__" && (i.value !== void 0 || r.alwaysSet) && (n[t.value] = i.value);
		}
		return {
			status: e.value,
			value: n
		};
	}
}, J = Object.freeze({ status: "aborted" }), JS = (e) => ({
	status: "dirty",
	value: e
}), YS = (e) => ({
	status: "valid",
	value: e
}), XS = (e) => e.status === "aborted", ZS = (e) => e.status === "dirty", QS = (e) => e.status === "valid", $S = (e) => typeof Promise < "u" && e instanceof Promise, Y;
(function(e) {
	e.errToObj = (e) => typeof e == "string" ? { message: e } : e || {}, e.toString = (e) => typeof e == "string" ? e : e?.message;
})(Y ||= {});
//#endregion
//#region node_modules/.pnpm/zod@3.25.76/node_modules/zod/v3/types.js
var eC = class {
	constructor(e, t, n, r) {
		this._cachedPath = [], this.parent = e, this.data = t, this._path = n, this._key = r;
	}
	get path() {
		return this._cachedPath.length || (Array.isArray(this._key) ? this._cachedPath.push(...this._path, ...this._key) : this._cachedPath.push(...this._path, this._key)), this._cachedPath;
	}
}, tC = (e, t) => {
	if (QS(t)) return {
		success: !0,
		data: t.value
	};
	if (!e.common.issues.length) throw Error("Validation failed but no issues detected.");
	return {
		success: !1,
		get error() {
			if (this._error) return this._error;
			let t = new HS(e.common.issues);
			return this._error = t, this._error;
		}
	};
};
function X(e) {
	if (!e) return {};
	let { errorMap: t, invalid_type_error: n, required_error: r, description: i } = e;
	if (t && (n || r)) throw Error("Can't use \"invalid_type_error\" or \"required_error\" in conjunction with custom error map.");
	return t ? {
		errorMap: t,
		description: i
	} : {
		errorMap: (t, i) => {
			let { message: a } = e;
			return t.code === "invalid_enum_value" ? { message: a ?? i.defaultError } : i.data === void 0 ? { message: a ?? r ?? i.defaultError } : t.code === "invalid_type" ? { message: a ?? n ?? i.defaultError } : { message: i.defaultError };
		},
		description: i
	};
}
var Z = class {
	get description() {
		return this._def.description;
	}
	_getType(e) {
		return VS(e.data);
	}
	_getOrReturnCtx(e, t) {
		return t || {
			common: e.parent.common,
			data: e.data,
			parsedType: VS(e.data),
			schemaErrorMap: this._def.errorMap,
			path: e.path,
			parent: e.parent
		};
	}
	_processInputParams(e) {
		return {
			status: new qS(),
			ctx: {
				common: e.parent.common,
				data: e.data,
				parsedType: VS(e.data),
				schemaErrorMap: this._def.errorMap,
				path: e.path,
				parent: e.parent
			}
		};
	}
	_parseSync(e) {
		let t = this._parse(e);
		if ($S(t)) throw Error("Synchronous parse encountered promise.");
		return t;
	}
	_parseAsync(e) {
		let t = this._parse(e);
		return Promise.resolve(t);
	}
	parse(e, t) {
		let n = this.safeParse(e, t);
		if (n.success) return n.data;
		throw n.error;
	}
	safeParse(e, t) {
		let n = {
			common: {
				issues: [],
				async: t?.async ?? !1,
				contextualErrorMap: t?.errorMap
			},
			path: t?.path || [],
			schemaErrorMap: this._def.errorMap,
			parent: null,
			data: e,
			parsedType: VS(e)
		};
		return tC(n, this._parseSync({
			data: e,
			path: n.path,
			parent: n
		}));
	}
	"~validate"(e) {
		let t = {
			common: {
				issues: [],
				async: !!this["~standard"].async
			},
			path: [],
			schemaErrorMap: this._def.errorMap,
			parent: null,
			data: e,
			parsedType: VS(e)
		};
		if (!this["~standard"].async) try {
			let n = this._parseSync({
				data: e,
				path: [],
				parent: t
			});
			return QS(n) ? { value: n.value } : { issues: t.common.issues };
		} catch (e) {
			e?.message?.toLowerCase()?.includes("encountered") && (this["~standard"].async = !0), t.common = {
				issues: [],
				async: !0
			};
		}
		return this._parseAsync({
			data: e,
			path: [],
			parent: t
		}).then((e) => QS(e) ? { value: e.value } : { issues: t.common.issues });
	}
	async parseAsync(e, t) {
		let n = await this.safeParseAsync(e, t);
		if (n.success) return n.data;
		throw n.error;
	}
	async safeParseAsync(e, t) {
		let n = {
			common: {
				issues: [],
				contextualErrorMap: t?.errorMap,
				async: !0
			},
			path: t?.path || [],
			schemaErrorMap: this._def.errorMap,
			parent: null,
			data: e,
			parsedType: VS(e)
		}, r = this._parse({
			data: e,
			path: n.path,
			parent: n
		});
		return tC(n, await ($S(r) ? r : Promise.resolve(r)));
	}
	refine(e, t) {
		let n = (e) => typeof t == "string" || t === void 0 ? { message: t } : typeof t == "function" ? t(e) : t;
		return this._refinement((t, r) => {
			let i = e(t), a = () => r.addIssue({
				code: K.custom,
				...n(t)
			});
			return typeof Promise < "u" && i instanceof Promise ? i.then((e) => e ? !0 : (a(), !1)) : i ? !0 : (a(), !1);
		});
	}
	refinement(e, t) {
		return this._refinement((n, r) => e(n) ? !0 : (r.addIssue(typeof t == "function" ? t(n, r) : t), !1));
	}
	_refinement(e) {
		return new iw({
			schema: this,
			typeName: Q.ZodEffects,
			effect: {
				type: "refinement",
				refinement: e
			}
		});
	}
	superRefine(e) {
		return this._refinement(e);
	}
	constructor(e) {
		this.spa = this.safeParseAsync, this._def = e, this.parse = this.parse.bind(this), this.safeParse = this.safeParse.bind(this), this.parseAsync = this.parseAsync.bind(this), this.safeParseAsync = this.safeParseAsync.bind(this), this.spa = this.spa.bind(this), this.refine = this.refine.bind(this), this.refinement = this.refinement.bind(this), this.superRefine = this.superRefine.bind(this), this.optional = this.optional.bind(this), this.nullable = this.nullable.bind(this), this.nullish = this.nullish.bind(this), this.array = this.array.bind(this), this.promise = this.promise.bind(this), this.or = this.or.bind(this), this.and = this.and.bind(this), this.transform = this.transform.bind(this), this.brand = this.brand.bind(this), this.default = this.default.bind(this), this.catch = this.catch.bind(this), this.describe = this.describe.bind(this), this.pipe = this.pipe.bind(this), this.readonly = this.readonly.bind(this), this.isNullable = this.isNullable.bind(this), this.isOptional = this.isOptional.bind(this), this["~standard"] = {
			version: 1,
			vendor: "zod",
			validate: (e) => this["~validate"](e)
		};
	}
	optional() {
		return aw.create(this, this._def);
	}
	nullable() {
		return ow.create(this, this._def);
	}
	nullish() {
		return this.nullable().optional();
	}
	array() {
		return zC.create(this);
	}
	promise() {
		return rw.create(this, this._def);
	}
	or(e) {
		return HC.create([this, e], this._def);
	}
	and(e) {
		return KC.create(this, e, this._def);
	}
	transform(e) {
		return new iw({
			...X(this._def),
			schema: this,
			typeName: Q.ZodEffects,
			effect: {
				type: "transform",
				transform: e
			}
		});
	}
	default(e) {
		let t = typeof e == "function" ? e : () => e;
		return new sw({
			...X(this._def),
			innerType: this,
			defaultValue: t,
			typeName: Q.ZodDefault
		});
	}
	brand() {
		return new uw({
			typeName: Q.ZodBranded,
			type: this,
			...X(this._def)
		});
	}
	catch(e) {
		let t = typeof e == "function" ? e : () => e;
		return new cw({
			...X(this._def),
			innerType: this,
			catchValue: t,
			typeName: Q.ZodCatch
		});
	}
	describe(e) {
		let t = this.constructor;
		return new t({
			...this._def,
			description: e
		});
	}
	pipe(e) {
		return dw.create(this, e);
	}
	readonly() {
		return fw.create(this);
	}
	isOptional() {
		return this.safeParse(void 0).success;
	}
	isNullable() {
		return this.safeParse(null).success;
	}
}, nC = /^c[^\s-]{8,}$/i, rC = /^[0-9a-z]+$/, iC = /^[0-9A-HJKMNP-TV-Z]{26}$/i, aC = /^[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{12}$/i, oC = /^[a-z0-9_-]{21}$/i, sC = /^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*$/, cC = /^[-+]?P(?!$)(?:(?:[-+]?\d+Y)|(?:[-+]?\d+[.,]\d+Y$))?(?:(?:[-+]?\d+M)|(?:[-+]?\d+[.,]\d+M$))?(?:(?:[-+]?\d+W)|(?:[-+]?\d+[.,]\d+W$))?(?:(?:[-+]?\d+D)|(?:[-+]?\d+[.,]\d+D$))?(?:T(?=[\d+-])(?:(?:[-+]?\d+H)|(?:[-+]?\d+[.,]\d+H$))?(?:(?:[-+]?\d+M)|(?:[-+]?\d+[.,]\d+M$))?(?:[-+]?\d+(?:[.,]\d+)?S)?)??$/, lC = /^(?!\.)(?!.*\.\.)([A-Z0-9_'+\-\.]*)[A-Z0-9_+-]@([A-Z0-9][A-Z0-9\-]*\.)+[A-Z]{2,}$/i, uC = "^(\\p{Extended_Pictographic}|\\p{Emoji_Component})+$", dC, fC = /^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$/, pC = /^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\/(3[0-2]|[12]?[0-9])$/, mC = /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$/, hC = /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))\/(12[0-8]|1[01][0-9]|[1-9]?[0-9])$/, gC = /^([0-9a-zA-Z+/]{4})*(([0-9a-zA-Z+/]{2}==)|([0-9a-zA-Z+/]{3}=))?$/, _C = /^([0-9a-zA-Z-_]{4})*(([0-9a-zA-Z-_]{2}(==)?)|([0-9a-zA-Z-_]{3}(=)?))?$/, vC = "((\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]|[02468][048]00|[13579][26]00)-02-29|\\d{4}-((0[13578]|1[02])-(0[1-9]|[12]\\d|3[01])|(0[469]|11)-(0[1-9]|[12]\\d|30)|(02)-(0[1-9]|1\\d|2[0-8])))", yC = RegExp(`^${vC}$`);
function bC(e) {
	let t = "[0-5]\\d";
	e.precision ? t = `${t}\\.\\d{${e.precision}}` : e.precision ?? (t = `${t}(\\.\\d+)?`);
	let n = e.precision ? "+" : "?";
	return `([01]\\d|2[0-3]):[0-5]\\d(:${t})${n}`;
}
function xC(e) {
	return RegExp(`^${bC(e)}$`);
}
function SC(e) {
	let t = `${vC}T${bC(e)}`, n = [];
	return n.push(e.local ? "Z?" : "Z"), e.offset && n.push("([+-]\\d{2}:?\\d{2})"), t = `${t}(${n.join("|")})`, RegExp(`^${t}$`);
}
function CC(e, t) {
	return !!((t === "v4" || !t) && fC.test(e) || (t === "v6" || !t) && mC.test(e));
}
function wC(e, t) {
	if (!sC.test(e)) return !1;
	try {
		let [n] = e.split(".");
		if (!n) return !1;
		let r = n.replace(/-/g, "+").replace(/_/g, "/").padEnd(n.length + (4 - n.length % 4) % 4, "="), i = JSON.parse(atob(r));
		return !(typeof i != "object" || !i || "typ" in i && i?.typ !== "JWT" || !i.alg || t && i.alg !== t);
	} catch {
		return !1;
	}
}
function TC(e, t) {
	return !!((t === "v4" || !t) && pC.test(e) || (t === "v6" || !t) && hC.test(e));
}
var EC = class e extends Z {
	_parse(e) {
		if (this._def.coerce && (e.data = String(e.data)), this._getType(e) !== G.string) {
			let t = this._getOrReturnCtx(e);
			return q(t, {
				code: K.invalid_type,
				expected: G.string,
				received: t.parsedType
			}), J;
		}
		let t = new qS(), n;
		for (let r of this._def.checks) if (r.kind === "min") e.data.length < r.value && (n = this._getOrReturnCtx(e, n), q(n, {
			code: K.too_small,
			minimum: r.value,
			type: "string",
			inclusive: !0,
			exact: !1,
			message: r.message
		}), t.dirty());
		else if (r.kind === "max") e.data.length > r.value && (n = this._getOrReturnCtx(e, n), q(n, {
			code: K.too_big,
			maximum: r.value,
			type: "string",
			inclusive: !0,
			exact: !1,
			message: r.message
		}), t.dirty());
		else if (r.kind === "length") {
			let i = e.data.length > r.value, a = e.data.length < r.value;
			(i || a) && (n = this._getOrReturnCtx(e, n), i ? q(n, {
				code: K.too_big,
				maximum: r.value,
				type: "string",
				inclusive: !0,
				exact: !0,
				message: r.message
			}) : a && q(n, {
				code: K.too_small,
				minimum: r.value,
				type: "string",
				inclusive: !0,
				exact: !0,
				message: r.message
			}), t.dirty());
		} else if (r.kind === "email") lC.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "email",
			code: K.invalid_string,
			message: r.message
		}), t.dirty());
		else if (r.kind === "emoji") dC ||= new RegExp(uC, "u"), dC.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "emoji",
			code: K.invalid_string,
			message: r.message
		}), t.dirty());
		else if (r.kind === "uuid") aC.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "uuid",
			code: K.invalid_string,
			message: r.message
		}), t.dirty());
		else if (r.kind === "nanoid") oC.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "nanoid",
			code: K.invalid_string,
			message: r.message
		}), t.dirty());
		else if (r.kind === "cuid") nC.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "cuid",
			code: K.invalid_string,
			message: r.message
		}), t.dirty());
		else if (r.kind === "cuid2") rC.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "cuid2",
			code: K.invalid_string,
			message: r.message
		}), t.dirty());
		else if (r.kind === "ulid") iC.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "ulid",
			code: K.invalid_string,
			message: r.message
		}), t.dirty());
		else if (r.kind === "url") try {
			new URL(e.data);
		} catch {
			n = this._getOrReturnCtx(e, n), q(n, {
				validation: "url",
				code: K.invalid_string,
				message: r.message
			}), t.dirty();
		}
		else r.kind === "regex" ? (r.regex.lastIndex = 0, r.regex.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "regex",
			code: K.invalid_string,
			message: r.message
		}), t.dirty())) : r.kind === "trim" ? e.data = e.data.trim() : r.kind === "includes" ? e.data.includes(r.value, r.position) || (n = this._getOrReturnCtx(e, n), q(n, {
			code: K.invalid_string,
			validation: {
				includes: r.value,
				position: r.position
			},
			message: r.message
		}), t.dirty()) : r.kind === "toLowerCase" ? e.data = e.data.toLowerCase() : r.kind === "toUpperCase" ? e.data = e.data.toUpperCase() : r.kind === "startsWith" ? e.data.startsWith(r.value) || (n = this._getOrReturnCtx(e, n), q(n, {
			code: K.invalid_string,
			validation: { startsWith: r.value },
			message: r.message
		}), t.dirty()) : r.kind === "endsWith" ? e.data.endsWith(r.value) || (n = this._getOrReturnCtx(e, n), q(n, {
			code: K.invalid_string,
			validation: { endsWith: r.value },
			message: r.message
		}), t.dirty()) : r.kind === "datetime" ? SC(r).test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			code: K.invalid_string,
			validation: "datetime",
			message: r.message
		}), t.dirty()) : r.kind === "date" ? yC.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			code: K.invalid_string,
			validation: "date",
			message: r.message
		}), t.dirty()) : r.kind === "time" ? xC(r).test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			code: K.invalid_string,
			validation: "time",
			message: r.message
		}), t.dirty()) : r.kind === "duration" ? cC.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "duration",
			code: K.invalid_string,
			message: r.message
		}), t.dirty()) : r.kind === "ip" ? CC(e.data, r.version) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "ip",
			code: K.invalid_string,
			message: r.message
		}), t.dirty()) : r.kind === "jwt" ? wC(e.data, r.alg) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "jwt",
			code: K.invalid_string,
			message: r.message
		}), t.dirty()) : r.kind === "cidr" ? TC(e.data, r.version) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "cidr",
			code: K.invalid_string,
			message: r.message
		}), t.dirty()) : r.kind === "base64" ? gC.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "base64",
			code: K.invalid_string,
			message: r.message
		}), t.dirty()) : r.kind === "base64url" ? _C.test(e.data) || (n = this._getOrReturnCtx(e, n), q(n, {
			validation: "base64url",
			code: K.invalid_string,
			message: r.message
		}), t.dirty()) : zS.assertNever(r);
		return {
			status: t.value,
			value: e.data
		};
	}
	_regex(e, t, n) {
		return this.refinement((t) => e.test(t), {
			validation: t,
			code: K.invalid_string,
			...Y.errToObj(n)
		});
	}
	_addCheck(t) {
		return new e({
			...this._def,
			checks: [...this._def.checks, t]
		});
	}
	email(e) {
		return this._addCheck({
			kind: "email",
			...Y.errToObj(e)
		});
	}
	url(e) {
		return this._addCheck({
			kind: "url",
			...Y.errToObj(e)
		});
	}
	emoji(e) {
		return this._addCheck({
			kind: "emoji",
			...Y.errToObj(e)
		});
	}
	uuid(e) {
		return this._addCheck({
			kind: "uuid",
			...Y.errToObj(e)
		});
	}
	nanoid(e) {
		return this._addCheck({
			kind: "nanoid",
			...Y.errToObj(e)
		});
	}
	cuid(e) {
		return this._addCheck({
			kind: "cuid",
			...Y.errToObj(e)
		});
	}
	cuid2(e) {
		return this._addCheck({
			kind: "cuid2",
			...Y.errToObj(e)
		});
	}
	ulid(e) {
		return this._addCheck({
			kind: "ulid",
			...Y.errToObj(e)
		});
	}
	base64(e) {
		return this._addCheck({
			kind: "base64",
			...Y.errToObj(e)
		});
	}
	base64url(e) {
		return this._addCheck({
			kind: "base64url",
			...Y.errToObj(e)
		});
	}
	jwt(e) {
		return this._addCheck({
			kind: "jwt",
			...Y.errToObj(e)
		});
	}
	ip(e) {
		return this._addCheck({
			kind: "ip",
			...Y.errToObj(e)
		});
	}
	cidr(e) {
		return this._addCheck({
			kind: "cidr",
			...Y.errToObj(e)
		});
	}
	datetime(e) {
		return typeof e == "string" ? this._addCheck({
			kind: "datetime",
			precision: null,
			offset: !1,
			local: !1,
			message: e
		}) : this._addCheck({
			kind: "datetime",
			precision: e?.precision === void 0 ? null : e?.precision,
			offset: e?.offset ?? !1,
			local: e?.local ?? !1,
			...Y.errToObj(e?.message)
		});
	}
	date(e) {
		return this._addCheck({
			kind: "date",
			message: e
		});
	}
	time(e) {
		return typeof e == "string" ? this._addCheck({
			kind: "time",
			precision: null,
			message: e
		}) : this._addCheck({
			kind: "time",
			precision: e?.precision === void 0 ? null : e?.precision,
			...Y.errToObj(e?.message)
		});
	}
	duration(e) {
		return this._addCheck({
			kind: "duration",
			...Y.errToObj(e)
		});
	}
	regex(e, t) {
		return this._addCheck({
			kind: "regex",
			regex: e,
			...Y.errToObj(t)
		});
	}
	includes(e, t) {
		return this._addCheck({
			kind: "includes",
			value: e,
			position: t?.position,
			...Y.errToObj(t?.message)
		});
	}
	startsWith(e, t) {
		return this._addCheck({
			kind: "startsWith",
			value: e,
			...Y.errToObj(t)
		});
	}
	endsWith(e, t) {
		return this._addCheck({
			kind: "endsWith",
			value: e,
			...Y.errToObj(t)
		});
	}
	min(e, t) {
		return this._addCheck({
			kind: "min",
			value: e,
			...Y.errToObj(t)
		});
	}
	max(e, t) {
		return this._addCheck({
			kind: "max",
			value: e,
			...Y.errToObj(t)
		});
	}
	length(e, t) {
		return this._addCheck({
			kind: "length",
			value: e,
			...Y.errToObj(t)
		});
	}
	nonempty(e) {
		return this.min(1, Y.errToObj(e));
	}
	trim() {
		return new e({
			...this._def,
			checks: [...this._def.checks, { kind: "trim" }]
		});
	}
	toLowerCase() {
		return new e({
			...this._def,
			checks: [...this._def.checks, { kind: "toLowerCase" }]
		});
	}
	toUpperCase() {
		return new e({
			...this._def,
			checks: [...this._def.checks, { kind: "toUpperCase" }]
		});
	}
	get isDatetime() {
		return !!this._def.checks.find((e) => e.kind === "datetime");
	}
	get isDate() {
		return !!this._def.checks.find((e) => e.kind === "date");
	}
	get isTime() {
		return !!this._def.checks.find((e) => e.kind === "time");
	}
	get isDuration() {
		return !!this._def.checks.find((e) => e.kind === "duration");
	}
	get isEmail() {
		return !!this._def.checks.find((e) => e.kind === "email");
	}
	get isURL() {
		return !!this._def.checks.find((e) => e.kind === "url");
	}
	get isEmoji() {
		return !!this._def.checks.find((e) => e.kind === "emoji");
	}
	get isUUID() {
		return !!this._def.checks.find((e) => e.kind === "uuid");
	}
	get isNANOID() {
		return !!this._def.checks.find((e) => e.kind === "nanoid");
	}
	get isCUID() {
		return !!this._def.checks.find((e) => e.kind === "cuid");
	}
	get isCUID2() {
		return !!this._def.checks.find((e) => e.kind === "cuid2");
	}
	get isULID() {
		return !!this._def.checks.find((e) => e.kind === "ulid");
	}
	get isIP() {
		return !!this._def.checks.find((e) => e.kind === "ip");
	}
	get isCIDR() {
		return !!this._def.checks.find((e) => e.kind === "cidr");
	}
	get isBase64() {
		return !!this._def.checks.find((e) => e.kind === "base64");
	}
	get isBase64url() {
		return !!this._def.checks.find((e) => e.kind === "base64url");
	}
	get minLength() {
		let e = null;
		for (let t of this._def.checks) t.kind === "min" && (e === null || t.value > e) && (e = t.value);
		return e;
	}
	get maxLength() {
		let e = null;
		for (let t of this._def.checks) t.kind === "max" && (e === null || t.value < e) && (e = t.value);
		return e;
	}
};
EC.create = (e) => new EC({
	checks: [],
	typeName: Q.ZodString,
	coerce: e?.coerce ?? !1,
	...X(e)
});
function DC(e, t) {
	let n = (e.toString().split(".")[1] || "").length, r = (t.toString().split(".")[1] || "").length, i = n > r ? n : r;
	return Number.parseInt(e.toFixed(i).replace(".", "")) % Number.parseInt(t.toFixed(i).replace(".", "")) / 10 ** i;
}
var OC = class e extends Z {
	constructor() {
		super(...arguments), this.min = this.gte, this.max = this.lte, this.step = this.multipleOf;
	}
	_parse(e) {
		if (this._def.coerce && (e.data = Number(e.data)), this._getType(e) !== G.number) {
			let t = this._getOrReturnCtx(e);
			return q(t, {
				code: K.invalid_type,
				expected: G.number,
				received: t.parsedType
			}), J;
		}
		let t, n = new qS();
		for (let r of this._def.checks) r.kind === "int" ? zS.isInteger(e.data) || (t = this._getOrReturnCtx(e, t), q(t, {
			code: K.invalid_type,
			expected: "integer",
			received: "float",
			message: r.message
		}), n.dirty()) : r.kind === "min" ? (r.inclusive ? e.data < r.value : e.data <= r.value) && (t = this._getOrReturnCtx(e, t), q(t, {
			code: K.too_small,
			minimum: r.value,
			type: "number",
			inclusive: r.inclusive,
			exact: !1,
			message: r.message
		}), n.dirty()) : r.kind === "max" ? (r.inclusive ? e.data > r.value : e.data >= r.value) && (t = this._getOrReturnCtx(e, t), q(t, {
			code: K.too_big,
			maximum: r.value,
			type: "number",
			inclusive: r.inclusive,
			exact: !1,
			message: r.message
		}), n.dirty()) : r.kind === "multipleOf" ? DC(e.data, r.value) !== 0 && (t = this._getOrReturnCtx(e, t), q(t, {
			code: K.not_multiple_of,
			multipleOf: r.value,
			message: r.message
		}), n.dirty()) : r.kind === "finite" ? Number.isFinite(e.data) || (t = this._getOrReturnCtx(e, t), q(t, {
			code: K.not_finite,
			message: r.message
		}), n.dirty()) : zS.assertNever(r);
		return {
			status: n.value,
			value: e.data
		};
	}
	gte(e, t) {
		return this.setLimit("min", e, !0, Y.toString(t));
	}
	gt(e, t) {
		return this.setLimit("min", e, !1, Y.toString(t));
	}
	lte(e, t) {
		return this.setLimit("max", e, !0, Y.toString(t));
	}
	lt(e, t) {
		return this.setLimit("max", e, !1, Y.toString(t));
	}
	setLimit(t, n, r, i) {
		return new e({
			...this._def,
			checks: [...this._def.checks, {
				kind: t,
				value: n,
				inclusive: r,
				message: Y.toString(i)
			}]
		});
	}
	_addCheck(t) {
		return new e({
			...this._def,
			checks: [...this._def.checks, t]
		});
	}
	int(e) {
		return this._addCheck({
			kind: "int",
			message: Y.toString(e)
		});
	}
	positive(e) {
		return this._addCheck({
			kind: "min",
			value: 0,
			inclusive: !1,
			message: Y.toString(e)
		});
	}
	negative(e) {
		return this._addCheck({
			kind: "max",
			value: 0,
			inclusive: !1,
			message: Y.toString(e)
		});
	}
	nonpositive(e) {
		return this._addCheck({
			kind: "max",
			value: 0,
			inclusive: !0,
			message: Y.toString(e)
		});
	}
	nonnegative(e) {
		return this._addCheck({
			kind: "min",
			value: 0,
			inclusive: !0,
			message: Y.toString(e)
		});
	}
	multipleOf(e, t) {
		return this._addCheck({
			kind: "multipleOf",
			value: e,
			message: Y.toString(t)
		});
	}
	finite(e) {
		return this._addCheck({
			kind: "finite",
			message: Y.toString(e)
		});
	}
	safe(e) {
		return this._addCheck({
			kind: "min",
			inclusive: !0,
			value: -(2 ** 53 - 1),
			message: Y.toString(e)
		})._addCheck({
			kind: "max",
			inclusive: !0,
			value: 2 ** 53 - 1,
			message: Y.toString(e)
		});
	}
	get minValue() {
		let e = null;
		for (let t of this._def.checks) t.kind === "min" && (e === null || t.value > e) && (e = t.value);
		return e;
	}
	get maxValue() {
		let e = null;
		for (let t of this._def.checks) t.kind === "max" && (e === null || t.value < e) && (e = t.value);
		return e;
	}
	get isInt() {
		return !!this._def.checks.find((e) => e.kind === "int" || e.kind === "multipleOf" && zS.isInteger(e.value));
	}
	get isFinite() {
		let e = null, t = null;
		for (let n of this._def.checks) if (n.kind === "finite" || n.kind === "int" || n.kind === "multipleOf") return !0;
		else n.kind === "min" ? (t === null || n.value > t) && (t = n.value) : n.kind === "max" && (e === null || n.value < e) && (e = n.value);
		return Number.isFinite(t) && Number.isFinite(e);
	}
};
OC.create = (e) => new OC({
	checks: [],
	typeName: Q.ZodNumber,
	coerce: e?.coerce || !1,
	...X(e)
});
var kC = class e extends Z {
	constructor() {
		super(...arguments), this.min = this.gte, this.max = this.lte;
	}
	_parse(e) {
		if (this._def.coerce) try {
			e.data = BigInt(e.data);
		} catch {
			return this._getInvalidInput(e);
		}
		if (this._getType(e) !== G.bigint) return this._getInvalidInput(e);
		let t, n = new qS();
		for (let r of this._def.checks) r.kind === "min" ? (r.inclusive ? e.data < r.value : e.data <= r.value) && (t = this._getOrReturnCtx(e, t), q(t, {
			code: K.too_small,
			type: "bigint",
			minimum: r.value,
			inclusive: r.inclusive,
			message: r.message
		}), n.dirty()) : r.kind === "max" ? (r.inclusive ? e.data > r.value : e.data >= r.value) && (t = this._getOrReturnCtx(e, t), q(t, {
			code: K.too_big,
			type: "bigint",
			maximum: r.value,
			inclusive: r.inclusive,
			message: r.message
		}), n.dirty()) : r.kind === "multipleOf" ? e.data % r.value !== BigInt(0) && (t = this._getOrReturnCtx(e, t), q(t, {
			code: K.not_multiple_of,
			multipleOf: r.value,
			message: r.message
		}), n.dirty()) : zS.assertNever(r);
		return {
			status: n.value,
			value: e.data
		};
	}
	_getInvalidInput(e) {
		let t = this._getOrReturnCtx(e);
		return q(t, {
			code: K.invalid_type,
			expected: G.bigint,
			received: t.parsedType
		}), J;
	}
	gte(e, t) {
		return this.setLimit("min", e, !0, Y.toString(t));
	}
	gt(e, t) {
		return this.setLimit("min", e, !1, Y.toString(t));
	}
	lte(e, t) {
		return this.setLimit("max", e, !0, Y.toString(t));
	}
	lt(e, t) {
		return this.setLimit("max", e, !1, Y.toString(t));
	}
	setLimit(t, n, r, i) {
		return new e({
			...this._def,
			checks: [...this._def.checks, {
				kind: t,
				value: n,
				inclusive: r,
				message: Y.toString(i)
			}]
		});
	}
	_addCheck(t) {
		return new e({
			...this._def,
			checks: [...this._def.checks, t]
		});
	}
	positive(e) {
		return this._addCheck({
			kind: "min",
			value: BigInt(0),
			inclusive: !1,
			message: Y.toString(e)
		});
	}
	negative(e) {
		return this._addCheck({
			kind: "max",
			value: BigInt(0),
			inclusive: !1,
			message: Y.toString(e)
		});
	}
	nonpositive(e) {
		return this._addCheck({
			kind: "max",
			value: BigInt(0),
			inclusive: !0,
			message: Y.toString(e)
		});
	}
	nonnegative(e) {
		return this._addCheck({
			kind: "min",
			value: BigInt(0),
			inclusive: !0,
			message: Y.toString(e)
		});
	}
	multipleOf(e, t) {
		return this._addCheck({
			kind: "multipleOf",
			value: e,
			message: Y.toString(t)
		});
	}
	get minValue() {
		let e = null;
		for (let t of this._def.checks) t.kind === "min" && (e === null || t.value > e) && (e = t.value);
		return e;
	}
	get maxValue() {
		let e = null;
		for (let t of this._def.checks) t.kind === "max" && (e === null || t.value < e) && (e = t.value);
		return e;
	}
};
kC.create = (e) => new kC({
	checks: [],
	typeName: Q.ZodBigInt,
	coerce: e?.coerce ?? !1,
	...X(e)
});
var AC = class extends Z {
	_parse(e) {
		if (this._def.coerce && (e.data = !!e.data), this._getType(e) !== G.boolean) {
			let t = this._getOrReturnCtx(e);
			return q(t, {
				code: K.invalid_type,
				expected: G.boolean,
				received: t.parsedType
			}), J;
		}
		return YS(e.data);
	}
};
AC.create = (e) => new AC({
	typeName: Q.ZodBoolean,
	coerce: e?.coerce || !1,
	...X(e)
});
var jC = class e extends Z {
	_parse(e) {
		if (this._def.coerce && (e.data = new Date(e.data)), this._getType(e) !== G.date) {
			let t = this._getOrReturnCtx(e);
			return q(t, {
				code: K.invalid_type,
				expected: G.date,
				received: t.parsedType
			}), J;
		}
		if (Number.isNaN(e.data.getTime())) return q(this._getOrReturnCtx(e), { code: K.invalid_date }), J;
		let t = new qS(), n;
		for (let r of this._def.checks) r.kind === "min" ? e.data.getTime() < r.value && (n = this._getOrReturnCtx(e, n), q(n, {
			code: K.too_small,
			message: r.message,
			inclusive: !0,
			exact: !1,
			minimum: r.value,
			type: "date"
		}), t.dirty()) : r.kind === "max" ? e.data.getTime() > r.value && (n = this._getOrReturnCtx(e, n), q(n, {
			code: K.too_big,
			message: r.message,
			inclusive: !0,
			exact: !1,
			maximum: r.value,
			type: "date"
		}), t.dirty()) : zS.assertNever(r);
		return {
			status: t.value,
			value: new Date(e.data.getTime())
		};
	}
	_addCheck(t) {
		return new e({
			...this._def,
			checks: [...this._def.checks, t]
		});
	}
	min(e, t) {
		return this._addCheck({
			kind: "min",
			value: e.getTime(),
			message: Y.toString(t)
		});
	}
	max(e, t) {
		return this._addCheck({
			kind: "max",
			value: e.getTime(),
			message: Y.toString(t)
		});
	}
	get minDate() {
		let e = null;
		for (let t of this._def.checks) t.kind === "min" && (e === null || t.value > e) && (e = t.value);
		return e == null ? null : new Date(e);
	}
	get maxDate() {
		let e = null;
		for (let t of this._def.checks) t.kind === "max" && (e === null || t.value < e) && (e = t.value);
		return e == null ? null : new Date(e);
	}
};
jC.create = (e) => new jC({
	checks: [],
	coerce: e?.coerce || !1,
	typeName: Q.ZodDate,
	...X(e)
});
var MC = class extends Z {
	_parse(e) {
		if (this._getType(e) !== G.symbol) {
			let t = this._getOrReturnCtx(e);
			return q(t, {
				code: K.invalid_type,
				expected: G.symbol,
				received: t.parsedType
			}), J;
		}
		return YS(e.data);
	}
};
MC.create = (e) => new MC({
	typeName: Q.ZodSymbol,
	...X(e)
});
var NC = class extends Z {
	_parse(e) {
		if (this._getType(e) !== G.undefined) {
			let t = this._getOrReturnCtx(e);
			return q(t, {
				code: K.invalid_type,
				expected: G.undefined,
				received: t.parsedType
			}), J;
		}
		return YS(e.data);
	}
};
NC.create = (e) => new NC({
	typeName: Q.ZodUndefined,
	...X(e)
});
var PC = class extends Z {
	_parse(e) {
		if (this._getType(e) !== G.null) {
			let t = this._getOrReturnCtx(e);
			return q(t, {
				code: K.invalid_type,
				expected: G.null,
				received: t.parsedType
			}), J;
		}
		return YS(e.data);
	}
};
PC.create = (e) => new PC({
	typeName: Q.ZodNull,
	...X(e)
});
var FC = class extends Z {
	constructor() {
		super(...arguments), this._any = !0;
	}
	_parse(e) {
		return YS(e.data);
	}
};
FC.create = (e) => new FC({
	typeName: Q.ZodAny,
	...X(e)
});
var IC = class extends Z {
	constructor() {
		super(...arguments), this._unknown = !0;
	}
	_parse(e) {
		return YS(e.data);
	}
};
IC.create = (e) => new IC({
	typeName: Q.ZodUnknown,
	...X(e)
});
var LC = class extends Z {
	_parse(e) {
		let t = this._getOrReturnCtx(e);
		return q(t, {
			code: K.invalid_type,
			expected: G.never,
			received: t.parsedType
		}), J;
	}
};
LC.create = (e) => new LC({
	typeName: Q.ZodNever,
	...X(e)
});
var RC = class extends Z {
	_parse(e) {
		if (this._getType(e) !== G.undefined) {
			let t = this._getOrReturnCtx(e);
			return q(t, {
				code: K.invalid_type,
				expected: G.void,
				received: t.parsedType
			}), J;
		}
		return YS(e.data);
	}
};
RC.create = (e) => new RC({
	typeName: Q.ZodVoid,
	...X(e)
});
var zC = class e extends Z {
	_parse(e) {
		let { ctx: t, status: n } = this._processInputParams(e), r = this._def;
		if (t.parsedType !== G.array) return q(t, {
			code: K.invalid_type,
			expected: G.array,
			received: t.parsedType
		}), J;
		if (r.exactLength !== null) {
			let e = t.data.length > r.exactLength.value, i = t.data.length < r.exactLength.value;
			(e || i) && (q(t, {
				code: e ? K.too_big : K.too_small,
				minimum: i ? r.exactLength.value : void 0,
				maximum: e ? r.exactLength.value : void 0,
				type: "array",
				inclusive: !0,
				exact: !0,
				message: r.exactLength.message
			}), n.dirty());
		}
		if (r.minLength !== null && t.data.length < r.minLength.value && (q(t, {
			code: K.too_small,
			minimum: r.minLength.value,
			type: "array",
			inclusive: !0,
			exact: !1,
			message: r.minLength.message
		}), n.dirty()), r.maxLength !== null && t.data.length > r.maxLength.value && (q(t, {
			code: K.too_big,
			maximum: r.maxLength.value,
			type: "array",
			inclusive: !0,
			exact: !1,
			message: r.maxLength.message
		}), n.dirty()), t.common.async) return Promise.all([...t.data].map((e, n) => r.type._parseAsync(new eC(t, e, t.path, n)))).then((e) => qS.mergeArray(n, e));
		let i = [...t.data].map((e, n) => r.type._parseSync(new eC(t, e, t.path, n)));
		return qS.mergeArray(n, i);
	}
	get element() {
		return this._def.type;
	}
	min(t, n) {
		return new e({
			...this._def,
			minLength: {
				value: t,
				message: Y.toString(n)
			}
		});
	}
	max(t, n) {
		return new e({
			...this._def,
			maxLength: {
				value: t,
				message: Y.toString(n)
			}
		});
	}
	length(t, n) {
		return new e({
			...this._def,
			exactLength: {
				value: t,
				message: Y.toString(n)
			}
		});
	}
	nonempty(e) {
		return this.min(1, e);
	}
};
zC.create = (e, t) => new zC({
	type: e,
	minLength: null,
	maxLength: null,
	exactLength: null,
	typeName: Q.ZodArray,
	...X(t)
});
function BC(e) {
	if (e instanceof VC) {
		let t = {};
		for (let n in e.shape) {
			let r = e.shape[n];
			t[n] = aw.create(BC(r));
		}
		return new VC({
			...e._def,
			shape: () => t
		});
	}
	return e instanceof zC ? new zC({
		...e._def,
		type: BC(e.element)
	}) : e instanceof aw ? aw.create(BC(e.unwrap())) : e instanceof ow ? ow.create(BC(e.unwrap())) : e instanceof qC ? qC.create(e.items.map((e) => BC(e))) : e;
}
var VC = class e extends Z {
	constructor() {
		super(...arguments), this._cached = null, this.nonstrict = this.passthrough, this.augment = this.extend;
	}
	_getCached() {
		if (this._cached !== null) return this._cached;
		let e = this._def.shape(), t = zS.objectKeys(e);
		return this._cached = {
			shape: e,
			keys: t
		}, this._cached;
	}
	_parse(e) {
		if (this._getType(e) !== G.object) {
			let t = this._getOrReturnCtx(e);
			return q(t, {
				code: K.invalid_type,
				expected: G.object,
				received: t.parsedType
			}), J;
		}
		let { status: t, ctx: n } = this._processInputParams(e), { shape: r, keys: i } = this._getCached(), a = [];
		if (!(this._def.catchall instanceof LC && this._def.unknownKeys === "strip")) for (let e in n.data) i.includes(e) || a.push(e);
		let o = [];
		for (let e of i) {
			let t = r[e], i = n.data[e];
			o.push({
				key: {
					status: "valid",
					value: e
				},
				value: t._parse(new eC(n, i, n.path, e)),
				alwaysSet: e in n.data
			});
		}
		if (this._def.catchall instanceof LC) {
			let e = this._def.unknownKeys;
			if (e === "passthrough") for (let e of a) o.push({
				key: {
					status: "valid",
					value: e
				},
				value: {
					status: "valid",
					value: n.data[e]
				}
			});
			else if (e === "strict") a.length > 0 && (q(n, {
				code: K.unrecognized_keys,
				keys: a
			}), t.dirty());
			else if (e !== "strip") throw Error("Internal ZodObject error: invalid unknownKeys value.");
		} else {
			let e = this._def.catchall;
			for (let t of a) {
				let r = n.data[t];
				o.push({
					key: {
						status: "valid",
						value: t
					},
					value: e._parse(new eC(n, r, n.path, t)),
					alwaysSet: t in n.data
				});
			}
		}
		return n.common.async ? Promise.resolve().then(async () => {
			let e = [];
			for (let t of o) {
				let n = await t.key, r = await t.value;
				e.push({
					key: n,
					value: r,
					alwaysSet: t.alwaysSet
				});
			}
			return e;
		}).then((e) => qS.mergeObjectSync(t, e)) : qS.mergeObjectSync(t, o);
	}
	get shape() {
		return this._def.shape();
	}
	strict(t) {
		return Y.errToObj, new e({
			...this._def,
			unknownKeys: "strict",
			...t === void 0 ? {} : { errorMap: (e, n) => {
				let r = this._def.errorMap?.(e, n).message ?? n.defaultError;
				return e.code === "unrecognized_keys" ? { message: Y.errToObj(t).message ?? r } : { message: r };
			} }
		});
	}
	strip() {
		return new e({
			...this._def,
			unknownKeys: "strip"
		});
	}
	passthrough() {
		return new e({
			...this._def,
			unknownKeys: "passthrough"
		});
	}
	extend(t) {
		return new e({
			...this._def,
			shape: () => ({
				...this._def.shape(),
				...t
			})
		});
	}
	merge(t) {
		return new e({
			unknownKeys: t._def.unknownKeys,
			catchall: t._def.catchall,
			shape: () => ({
				...this._def.shape(),
				...t._def.shape()
			}),
			typeName: Q.ZodObject
		});
	}
	setKey(e, t) {
		return this.augment({ [e]: t });
	}
	catchall(t) {
		return new e({
			...this._def,
			catchall: t
		});
	}
	pick(t) {
		let n = {};
		for (let e of zS.objectKeys(t)) t[e] && this.shape[e] && (n[e] = this.shape[e]);
		return new e({
			...this._def,
			shape: () => n
		});
	}
	omit(t) {
		let n = {};
		for (let e of zS.objectKeys(this.shape)) t[e] || (n[e] = this.shape[e]);
		return new e({
			...this._def,
			shape: () => n
		});
	}
	deepPartial() {
		return BC(this);
	}
	partial(t) {
		let n = {};
		for (let e of zS.objectKeys(this.shape)) {
			let r = this.shape[e];
			n[e] = t && !t[e] ? r : r.optional();
		}
		return new e({
			...this._def,
			shape: () => n
		});
	}
	required(t) {
		let n = {};
		for (let e of zS.objectKeys(this.shape)) if (t && !t[e]) n[e] = this.shape[e];
		else {
			let t = this.shape[e];
			for (; t instanceof aw;) t = t._def.innerType;
			n[e] = t;
		}
		return new e({
			...this._def,
			shape: () => n
		});
	}
	keyof() {
		return ew(zS.objectKeys(this.shape));
	}
};
VC.create = (e, t) => new VC({
	shape: () => e,
	unknownKeys: "strip",
	catchall: LC.create(),
	typeName: Q.ZodObject,
	...X(t)
}), VC.strictCreate = (e, t) => new VC({
	shape: () => e,
	unknownKeys: "strict",
	catchall: LC.create(),
	typeName: Q.ZodObject,
	...X(t)
}), VC.lazycreate = (e, t) => new VC({
	shape: e,
	unknownKeys: "strip",
	catchall: LC.create(),
	typeName: Q.ZodObject,
	...X(t)
});
var HC = class extends Z {
	_parse(e) {
		let { ctx: t } = this._processInputParams(e), n = this._def.options;
		function r(e) {
			for (let t of e) if (t.result.status === "valid") return t.result;
			for (let n of e) if (n.result.status === "dirty") return t.common.issues.push(...n.ctx.common.issues), n.result;
			let n = e.map((e) => new HS(e.ctx.common.issues));
			return q(t, {
				code: K.invalid_union,
				unionErrors: n
			}), J;
		}
		if (t.common.async) return Promise.all(n.map(async (e) => {
			let n = {
				...t,
				common: {
					...t.common,
					issues: []
				},
				parent: null
			};
			return {
				result: await e._parseAsync({
					data: t.data,
					path: t.path,
					parent: n
				}),
				ctx: n
			};
		})).then(r);
		{
			let e, r = [];
			for (let i of n) {
				let n = {
					...t,
					common: {
						...t.common,
						issues: []
					},
					parent: null
				}, a = i._parseSync({
					data: t.data,
					path: t.path,
					parent: n
				});
				if (a.status === "valid") return a;
				a.status === "dirty" && !e && (e = {
					result: a,
					ctx: n
				}), n.common.issues.length && r.push(n.common.issues);
			}
			if (e) return t.common.issues.push(...e.ctx.common.issues), e.result;
			let i = r.map((e) => new HS(e));
			return q(t, {
				code: K.invalid_union,
				unionErrors: i
			}), J;
		}
	}
	get options() {
		return this._def.options;
	}
};
HC.create = (e, t) => new HC({
	options: e,
	typeName: Q.ZodUnion,
	...X(t)
});
var UC = (e) => e instanceof QC ? UC(e.schema) : e instanceof iw ? UC(e.innerType()) : e instanceof $C ? [e.value] : e instanceof tw ? e.options : e instanceof nw ? zS.objectValues(e.enum) : e instanceof sw ? UC(e._def.innerType) : e instanceof NC ? [void 0] : e instanceof PC ? [null] : e instanceof aw ? [void 0, ...UC(e.unwrap())] : e instanceof ow ? [null, ...UC(e.unwrap())] : e instanceof uw || e instanceof fw ? UC(e.unwrap()) : e instanceof cw ? UC(e._def.innerType) : [], WC = class e extends Z {
	_parse(e) {
		let { ctx: t } = this._processInputParams(e);
		if (t.parsedType !== G.object) return q(t, {
			code: K.invalid_type,
			expected: G.object,
			received: t.parsedType
		}), J;
		let n = this.discriminator, r = t.data[n], i = this.optionsMap.get(r);
		return i ? t.common.async ? i._parseAsync({
			data: t.data,
			path: t.path,
			parent: t
		}) : i._parseSync({
			data: t.data,
			path: t.path,
			parent: t
		}) : (q(t, {
			code: K.invalid_union_discriminator,
			options: Array.from(this.optionsMap.keys()),
			path: [n]
		}), J);
	}
	get discriminator() {
		return this._def.discriminator;
	}
	get options() {
		return this._def.options;
	}
	get optionsMap() {
		return this._def.optionsMap;
	}
	static create(t, n, r) {
		let i = /* @__PURE__ */ new Map();
		for (let e of n) {
			let n = UC(e.shape[t]);
			if (!n.length) throw Error(`A discriminator value for key \`${t}\` could not be extracted from all schema options`);
			for (let r of n) {
				if (i.has(r)) throw Error(`Discriminator property ${String(t)} has duplicate value ${String(r)}`);
				i.set(r, e);
			}
		}
		return new e({
			typeName: Q.ZodDiscriminatedUnion,
			discriminator: t,
			options: n,
			optionsMap: i,
			...X(r)
		});
	}
};
function GC(e, t) {
	let n = VS(e), r = VS(t);
	if (e === t) return {
		valid: !0,
		data: e
	};
	if (n === G.object && r === G.object) {
		let n = zS.objectKeys(t), r = zS.objectKeys(e).filter((e) => n.indexOf(e) !== -1), i = {
			...e,
			...t
		};
		for (let n of r) {
			let r = GC(e[n], t[n]);
			if (!r.valid) return { valid: !1 };
			i[n] = r.data;
		}
		return {
			valid: !0,
			data: i
		};
	}
	if (n === G.array && r === G.array) {
		if (e.length !== t.length) return { valid: !1 };
		let n = [];
		for (let r = 0; r < e.length; r++) {
			let i = e[r], a = t[r], o = GC(i, a);
			if (!o.valid) return { valid: !1 };
			n.push(o.data);
		}
		return {
			valid: !0,
			data: n
		};
	}
	return n === G.date && r === G.date && +e == +t ? {
		valid: !0,
		data: e
	} : { valid: !1 };
}
var KC = class extends Z {
	_parse(e) {
		let { status: t, ctx: n } = this._processInputParams(e), r = (e, r) => {
			if (XS(e) || XS(r)) return J;
			let i = GC(e.value, r.value);
			return i.valid ? ((ZS(e) || ZS(r)) && t.dirty(), {
				status: t.value,
				value: i.data
			}) : (q(n, { code: K.invalid_intersection_types }), J);
		};
		return n.common.async ? Promise.all([this._def.left._parseAsync({
			data: n.data,
			path: n.path,
			parent: n
		}), this._def.right._parseAsync({
			data: n.data,
			path: n.path,
			parent: n
		})]).then(([e, t]) => r(e, t)) : r(this._def.left._parseSync({
			data: n.data,
			path: n.path,
			parent: n
		}), this._def.right._parseSync({
			data: n.data,
			path: n.path,
			parent: n
		}));
	}
};
KC.create = (e, t, n) => new KC({
	left: e,
	right: t,
	typeName: Q.ZodIntersection,
	...X(n)
});
var qC = class e extends Z {
	_parse(e) {
		let { status: t, ctx: n } = this._processInputParams(e);
		if (n.parsedType !== G.array) return q(n, {
			code: K.invalid_type,
			expected: G.array,
			received: n.parsedType
		}), J;
		if (n.data.length < this._def.items.length) return q(n, {
			code: K.too_small,
			minimum: this._def.items.length,
			inclusive: !0,
			exact: !1,
			type: "array"
		}), J;
		!this._def.rest && n.data.length > this._def.items.length && (q(n, {
			code: K.too_big,
			maximum: this._def.items.length,
			inclusive: !0,
			exact: !1,
			type: "array"
		}), t.dirty());
		let r = [...n.data].map((e, t) => {
			let r = this._def.items[t] || this._def.rest;
			return r ? r._parse(new eC(n, e, n.path, t)) : null;
		}).filter((e) => !!e);
		return n.common.async ? Promise.all(r).then((e) => qS.mergeArray(t, e)) : qS.mergeArray(t, r);
	}
	get items() {
		return this._def.items;
	}
	rest(t) {
		return new e({
			...this._def,
			rest: t
		});
	}
};
qC.create = (e, t) => {
	if (!Array.isArray(e)) throw Error("You must pass an array of schemas to z.tuple([ ... ])");
	return new qC({
		items: e,
		typeName: Q.ZodTuple,
		rest: null,
		...X(t)
	});
};
var JC = class e extends Z {
	get keySchema() {
		return this._def.keyType;
	}
	get valueSchema() {
		return this._def.valueType;
	}
	_parse(e) {
		let { status: t, ctx: n } = this._processInputParams(e);
		if (n.parsedType !== G.object) return q(n, {
			code: K.invalid_type,
			expected: G.object,
			received: n.parsedType
		}), J;
		let r = [], i = this._def.keyType, a = this._def.valueType;
		for (let e in n.data) r.push({
			key: i._parse(new eC(n, e, n.path, e)),
			value: a._parse(new eC(n, n.data[e], n.path, e)),
			alwaysSet: e in n.data
		});
		return n.common.async ? qS.mergeObjectAsync(t, r) : qS.mergeObjectSync(t, r);
	}
	get element() {
		return this._def.valueType;
	}
	static create(t, n, r) {
		return n instanceof Z ? new e({
			keyType: t,
			valueType: n,
			typeName: Q.ZodRecord,
			...X(r)
		}) : new e({
			keyType: EC.create(),
			valueType: t,
			typeName: Q.ZodRecord,
			...X(n)
		});
	}
}, YC = class extends Z {
	get keySchema() {
		return this._def.keyType;
	}
	get valueSchema() {
		return this._def.valueType;
	}
	_parse(e) {
		let { status: t, ctx: n } = this._processInputParams(e);
		if (n.parsedType !== G.map) return q(n, {
			code: K.invalid_type,
			expected: G.map,
			received: n.parsedType
		}), J;
		let r = this._def.keyType, i = this._def.valueType, a = [...n.data.entries()].map(([e, t], a) => ({
			key: r._parse(new eC(n, e, n.path, [a, "key"])),
			value: i._parse(new eC(n, t, n.path, [a, "value"]))
		}));
		if (n.common.async) {
			let e = /* @__PURE__ */ new Map();
			return Promise.resolve().then(async () => {
				for (let n of a) {
					let r = await n.key, i = await n.value;
					if (r.status === "aborted" || i.status === "aborted") return J;
					(r.status === "dirty" || i.status === "dirty") && t.dirty(), e.set(r.value, i.value);
				}
				return {
					status: t.value,
					value: e
				};
			});
		}
		{
			let e = /* @__PURE__ */ new Map();
			for (let n of a) {
				let r = n.key, i = n.value;
				if (r.status === "aborted" || i.status === "aborted") return J;
				(r.status === "dirty" || i.status === "dirty") && t.dirty(), e.set(r.value, i.value);
			}
			return {
				status: t.value,
				value: e
			};
		}
	}
};
YC.create = (e, t, n) => new YC({
	valueType: t,
	keyType: e,
	typeName: Q.ZodMap,
	...X(n)
});
var XC = class e extends Z {
	_parse(e) {
		let { status: t, ctx: n } = this._processInputParams(e);
		if (n.parsedType !== G.set) return q(n, {
			code: K.invalid_type,
			expected: G.set,
			received: n.parsedType
		}), J;
		let r = this._def;
		r.minSize !== null && n.data.size < r.minSize.value && (q(n, {
			code: K.too_small,
			minimum: r.minSize.value,
			type: "set",
			inclusive: !0,
			exact: !1,
			message: r.minSize.message
		}), t.dirty()), r.maxSize !== null && n.data.size > r.maxSize.value && (q(n, {
			code: K.too_big,
			maximum: r.maxSize.value,
			type: "set",
			inclusive: !0,
			exact: !1,
			message: r.maxSize.message
		}), t.dirty());
		let i = this._def.valueType;
		function a(e) {
			let n = /* @__PURE__ */ new Set();
			for (let r of e) {
				if (r.status === "aborted") return J;
				r.status === "dirty" && t.dirty(), n.add(r.value);
			}
			return {
				status: t.value,
				value: n
			};
		}
		let o = [...n.data.values()].map((e, t) => i._parse(new eC(n, e, n.path, t)));
		return n.common.async ? Promise.all(o).then((e) => a(e)) : a(o);
	}
	min(t, n) {
		return new e({
			...this._def,
			minSize: {
				value: t,
				message: Y.toString(n)
			}
		});
	}
	max(t, n) {
		return new e({
			...this._def,
			maxSize: {
				value: t,
				message: Y.toString(n)
			}
		});
	}
	size(e, t) {
		return this.min(e, t).max(e, t);
	}
	nonempty(e) {
		return this.min(1, e);
	}
};
XC.create = (e, t) => new XC({
	valueType: e,
	minSize: null,
	maxSize: null,
	typeName: Q.ZodSet,
	...X(t)
});
var ZC = class e extends Z {
	constructor() {
		super(...arguments), this.validate = this.implement;
	}
	_parse(e) {
		let { ctx: t } = this._processInputParams(e);
		if (t.parsedType !== G.function) return q(t, {
			code: K.invalid_type,
			expected: G.function,
			received: t.parsedType
		}), J;
		function n(e, n) {
			return KS({
				data: e,
				path: t.path,
				errorMaps: [
					t.common.contextualErrorMap,
					t.schemaErrorMap,
					GS(),
					US
				].filter((e) => !!e),
				issueData: {
					code: K.invalid_arguments,
					argumentsError: n
				}
			});
		}
		function r(e, n) {
			return KS({
				data: e,
				path: t.path,
				errorMaps: [
					t.common.contextualErrorMap,
					t.schemaErrorMap,
					GS(),
					US
				].filter((e) => !!e),
				issueData: {
					code: K.invalid_return_type,
					returnTypeError: n
				}
			});
		}
		let i = { errorMap: t.common.contextualErrorMap }, a = t.data;
		if (this._def.returns instanceof rw) {
			let e = this;
			return YS(async function(...t) {
				let o = new HS([]), s = await e._def.args.parseAsync(t, i).catch((e) => {
					throw o.addIssue(n(t, e)), o;
				}), c = await Reflect.apply(a, this, s);
				return await e._def.returns._def.type.parseAsync(c, i).catch((e) => {
					throw o.addIssue(r(c, e)), o;
				});
			});
		}
		{
			let e = this;
			return YS(function(...t) {
				let o = e._def.args.safeParse(t, i);
				if (!o.success) throw new HS([n(t, o.error)]);
				let s = Reflect.apply(a, this, o.data), c = e._def.returns.safeParse(s, i);
				if (!c.success) throw new HS([r(s, c.error)]);
				return c.data;
			});
		}
	}
	parameters() {
		return this._def.args;
	}
	returnType() {
		return this._def.returns;
	}
	args(...t) {
		return new e({
			...this._def,
			args: qC.create(t).rest(IC.create())
		});
	}
	returns(t) {
		return new e({
			...this._def,
			returns: t
		});
	}
	implement(e) {
		return this.parse(e);
	}
	strictImplement(e) {
		return this.parse(e);
	}
	static create(t, n, r) {
		return new e({
			args: t || qC.create([]).rest(IC.create()),
			returns: n || IC.create(),
			typeName: Q.ZodFunction,
			...X(r)
		});
	}
}, QC = class extends Z {
	get schema() {
		return this._def.getter();
	}
	_parse(e) {
		let { ctx: t } = this._processInputParams(e);
		return this._def.getter()._parse({
			data: t.data,
			path: t.path,
			parent: t
		});
	}
};
QC.create = (e, t) => new QC({
	getter: e,
	typeName: Q.ZodLazy,
	...X(t)
});
var $C = class extends Z {
	_parse(e) {
		if (e.data !== this._def.value) {
			let t = this._getOrReturnCtx(e);
			return q(t, {
				received: t.data,
				code: K.invalid_literal,
				expected: this._def.value
			}), J;
		}
		return {
			status: "valid",
			value: e.data
		};
	}
	get value() {
		return this._def.value;
	}
};
$C.create = (e, t) => new $C({
	value: e,
	typeName: Q.ZodLiteral,
	...X(t)
});
function ew(e, t) {
	return new tw({
		values: e,
		typeName: Q.ZodEnum,
		...X(t)
	});
}
var tw = class e extends Z {
	_parse(e) {
		if (typeof e.data != "string") {
			let t = this._getOrReturnCtx(e), n = this._def.values;
			return q(t, {
				expected: zS.joinValues(n),
				received: t.parsedType,
				code: K.invalid_type
			}), J;
		}
		if (this._cache ||= new Set(this._def.values), !this._cache.has(e.data)) {
			let t = this._getOrReturnCtx(e), n = this._def.values;
			return q(t, {
				received: t.data,
				code: K.invalid_enum_value,
				options: n
			}), J;
		}
		return YS(e.data);
	}
	get options() {
		return this._def.values;
	}
	get enum() {
		let e = {};
		for (let t of this._def.values) e[t] = t;
		return e;
	}
	get Values() {
		let e = {};
		for (let t of this._def.values) e[t] = t;
		return e;
	}
	get Enum() {
		let e = {};
		for (let t of this._def.values) e[t] = t;
		return e;
	}
	extract(t, n = this._def) {
		return e.create(t, {
			...this._def,
			...n
		});
	}
	exclude(t, n = this._def) {
		return e.create(this.options.filter((e) => !t.includes(e)), {
			...this._def,
			...n
		});
	}
};
tw.create = ew;
var nw = class extends Z {
	_parse(e) {
		let t = zS.getValidEnumValues(this._def.values), n = this._getOrReturnCtx(e);
		if (n.parsedType !== G.string && n.parsedType !== G.number) {
			let e = zS.objectValues(t);
			return q(n, {
				expected: zS.joinValues(e),
				received: n.parsedType,
				code: K.invalid_type
			}), J;
		}
		if (this._cache ||= new Set(zS.getValidEnumValues(this._def.values)), !this._cache.has(e.data)) {
			let e = zS.objectValues(t);
			return q(n, {
				received: n.data,
				code: K.invalid_enum_value,
				options: e
			}), J;
		}
		return YS(e.data);
	}
	get enum() {
		return this._def.values;
	}
};
nw.create = (e, t) => new nw({
	values: e,
	typeName: Q.ZodNativeEnum,
	...X(t)
});
var rw = class extends Z {
	unwrap() {
		return this._def.type;
	}
	_parse(e) {
		let { ctx: t } = this._processInputParams(e);
		return t.parsedType !== G.promise && t.common.async === !1 ? (q(t, {
			code: K.invalid_type,
			expected: G.promise,
			received: t.parsedType
		}), J) : YS((t.parsedType === G.promise ? t.data : Promise.resolve(t.data)).then((e) => this._def.type.parseAsync(e, {
			path: t.path,
			errorMap: t.common.contextualErrorMap
		})));
	}
};
rw.create = (e, t) => new rw({
	type: e,
	typeName: Q.ZodPromise,
	...X(t)
});
var iw = class extends Z {
	innerType() {
		return this._def.schema;
	}
	sourceType() {
		return this._def.schema._def.typeName === Q.ZodEffects ? this._def.schema.sourceType() : this._def.schema;
	}
	_parse(e) {
		let { status: t, ctx: n } = this._processInputParams(e), r = this._def.effect || null, i = {
			addIssue: (e) => {
				q(n, e), e.fatal ? t.abort() : t.dirty();
			},
			get path() {
				return n.path;
			}
		};
		if (i.addIssue = i.addIssue.bind(i), r.type === "preprocess") {
			let e = r.transform(n.data, i);
			if (n.common.async) return Promise.resolve(e).then(async (e) => {
				if (t.value === "aborted") return J;
				let r = await this._def.schema._parseAsync({
					data: e,
					path: n.path,
					parent: n
				});
				return r.status === "aborted" ? J : r.status === "dirty" || t.value === "dirty" ? JS(r.value) : r;
			});
			{
				if (t.value === "aborted") return J;
				let r = this._def.schema._parseSync({
					data: e,
					path: n.path,
					parent: n
				});
				return r.status === "aborted" ? J : r.status === "dirty" || t.value === "dirty" ? JS(r.value) : r;
			}
		}
		if (r.type === "refinement") {
			let e = (e) => {
				let t = r.refinement(e, i);
				if (n.common.async) return Promise.resolve(t);
				if (t instanceof Promise) throw Error("Async refinement encountered during synchronous parse operation. Use .parseAsync instead.");
				return e;
			};
			if (n.common.async === !1) {
				let r = this._def.schema._parseSync({
					data: n.data,
					path: n.path,
					parent: n
				});
				return r.status === "aborted" ? J : (r.status === "dirty" && t.dirty(), e(r.value), {
					status: t.value,
					value: r.value
				});
			}
			return this._def.schema._parseAsync({
				data: n.data,
				path: n.path,
				parent: n
			}).then((n) => n.status === "aborted" ? J : (n.status === "dirty" && t.dirty(), e(n.value).then(() => ({
				status: t.value,
				value: n.value
			}))));
		}
		if (r.type === "transform") {
			if (n.common.async === !1) {
				let e = this._def.schema._parseSync({
					data: n.data,
					path: n.path,
					parent: n
				});
				if (!QS(e)) return J;
				let a = r.transform(e.value, i);
				if (a instanceof Promise) throw Error("Asynchronous transform encountered during synchronous parse operation. Use .parseAsync instead.");
				return {
					status: t.value,
					value: a
				};
			}
			return this._def.schema._parseAsync({
				data: n.data,
				path: n.path,
				parent: n
			}).then((e) => QS(e) ? Promise.resolve(r.transform(e.value, i)).then((e) => ({
				status: t.value,
				value: e
			})) : J);
		}
		zS.assertNever(r);
	}
};
iw.create = (e, t, n) => new iw({
	schema: e,
	typeName: Q.ZodEffects,
	effect: t,
	...X(n)
}), iw.createWithPreprocess = (e, t, n) => new iw({
	schema: t,
	effect: {
		type: "preprocess",
		transform: e
	},
	typeName: Q.ZodEffects,
	...X(n)
});
var aw = class extends Z {
	_parse(e) {
		return this._getType(e) === G.undefined ? YS(void 0) : this._def.innerType._parse(e);
	}
	unwrap() {
		return this._def.innerType;
	}
};
aw.create = (e, t) => new aw({
	innerType: e,
	typeName: Q.ZodOptional,
	...X(t)
});
var ow = class extends Z {
	_parse(e) {
		return this._getType(e) === G.null ? YS(null) : this._def.innerType._parse(e);
	}
	unwrap() {
		return this._def.innerType;
	}
};
ow.create = (e, t) => new ow({
	innerType: e,
	typeName: Q.ZodNullable,
	...X(t)
});
var sw = class extends Z {
	_parse(e) {
		let { ctx: t } = this._processInputParams(e), n = t.data;
		return t.parsedType === G.undefined && (n = this._def.defaultValue()), this._def.innerType._parse({
			data: n,
			path: t.path,
			parent: t
		});
	}
	removeDefault() {
		return this._def.innerType;
	}
};
sw.create = (e, t) => new sw({
	innerType: e,
	typeName: Q.ZodDefault,
	defaultValue: typeof t.default == "function" ? t.default : () => t.default,
	...X(t)
});
var cw = class extends Z {
	_parse(e) {
		let { ctx: t } = this._processInputParams(e), n = {
			...t,
			common: {
				...t.common,
				issues: []
			}
		}, r = this._def.innerType._parse({
			data: n.data,
			path: n.path,
			parent: { ...n }
		});
		return $S(r) ? r.then((e) => ({
			status: "valid",
			value: e.status === "valid" ? e.value : this._def.catchValue({
				get error() {
					return new HS(n.common.issues);
				},
				input: n.data
			})
		})) : {
			status: "valid",
			value: r.status === "valid" ? r.value : this._def.catchValue({
				get error() {
					return new HS(n.common.issues);
				},
				input: n.data
			})
		};
	}
	removeCatch() {
		return this._def.innerType;
	}
};
cw.create = (e, t) => new cw({
	innerType: e,
	typeName: Q.ZodCatch,
	catchValue: typeof t.catch == "function" ? t.catch : () => t.catch,
	...X(t)
});
var lw = class extends Z {
	_parse(e) {
		if (this._getType(e) !== G.nan) {
			let t = this._getOrReturnCtx(e);
			return q(t, {
				code: K.invalid_type,
				expected: G.nan,
				received: t.parsedType
			}), J;
		}
		return {
			status: "valid",
			value: e.data
		};
	}
};
lw.create = (e) => new lw({
	typeName: Q.ZodNaN,
	...X(e)
});
var uw = class extends Z {
	_parse(e) {
		let { ctx: t } = this._processInputParams(e), n = t.data;
		return this._def.type._parse({
			data: n,
			path: t.path,
			parent: t
		});
	}
	unwrap() {
		return this._def.type;
	}
}, dw = class e extends Z {
	_parse(e) {
		let { status: t, ctx: n } = this._processInputParams(e);
		if (n.common.async) return (async () => {
			let e = await this._def.in._parseAsync({
				data: n.data,
				path: n.path,
				parent: n
			});
			return e.status === "aborted" ? J : e.status === "dirty" ? (t.dirty(), JS(e.value)) : this._def.out._parseAsync({
				data: e.value,
				path: n.path,
				parent: n
			});
		})();
		{
			let e = this._def.in._parseSync({
				data: n.data,
				path: n.path,
				parent: n
			});
			return e.status === "aborted" ? J : e.status === "dirty" ? (t.dirty(), {
				status: "dirty",
				value: e.value
			}) : this._def.out._parseSync({
				data: e.value,
				path: n.path,
				parent: n
			});
		}
	}
	static create(t, n) {
		return new e({
			in: t,
			out: n,
			typeName: Q.ZodPipeline
		});
	}
}, fw = class extends Z {
	_parse(e) {
		let t = this._def.innerType._parse(e), n = (e) => (QS(e) && (e.value = Object.freeze(e.value)), e);
		return $S(t) ? t.then((e) => n(e)) : n(t);
	}
	unwrap() {
		return this._def.innerType;
	}
};
fw.create = (e, t) => new fw({
	innerType: e,
	typeName: Q.ZodReadonly,
	...X(t)
}), VC.lazycreate;
var Q;
(function(e) {
	e.ZodString = "ZodString", e.ZodNumber = "ZodNumber", e.ZodNaN = "ZodNaN", e.ZodBigInt = "ZodBigInt", e.ZodBoolean = "ZodBoolean", e.ZodDate = "ZodDate", e.ZodSymbol = "ZodSymbol", e.ZodUndefined = "ZodUndefined", e.ZodNull = "ZodNull", e.ZodAny = "ZodAny", e.ZodUnknown = "ZodUnknown", e.ZodNever = "ZodNever", e.ZodVoid = "ZodVoid", e.ZodArray = "ZodArray", e.ZodObject = "ZodObject", e.ZodUnion = "ZodUnion", e.ZodDiscriminatedUnion = "ZodDiscriminatedUnion", e.ZodIntersection = "ZodIntersection", e.ZodTuple = "ZodTuple", e.ZodRecord = "ZodRecord", e.ZodMap = "ZodMap", e.ZodSet = "ZodSet", e.ZodFunction = "ZodFunction", e.ZodLazy = "ZodLazy", e.ZodLiteral = "ZodLiteral", e.ZodEnum = "ZodEnum", e.ZodEffects = "ZodEffects", e.ZodNativeEnum = "ZodNativeEnum", e.ZodOptional = "ZodOptional", e.ZodNullable = "ZodNullable", e.ZodDefault = "ZodDefault", e.ZodCatch = "ZodCatch", e.ZodPromise = "ZodPromise", e.ZodBranded = "ZodBranded", e.ZodPipeline = "ZodPipeline", e.ZodReadonly = "ZodReadonly";
})(Q ||= {});
var pw = EC.create;
OC.create, lw.create, kC.create, AC.create, jC.create, MC.create, NC.create, PC.create, FC.create, IC.create, LC.create, RC.create, zC.create;
var mw = VC.create;
VC.strictCreate, HC.create, WC.create, KC.create, qC.create, JC.create, YC.create, XC.create, ZC.create, QC.create, $C.create, tw.create, nw.create, rw.create, iw.create, aw.create, ow.create, iw.createWithPreprocess, dw.create;
var hw = mw({
	name: pw().trim().min(1, "A manifest id is required").regex(/^[a-z0-9][a-z0-9-]*$/, "Lowercase letters, digits and hyphens only"),
	agent: pw().trim().optional(),
	egress: pw().trim().optional(),
	env: pw().trim().optional(),
	port: pw().trim().optional(),
	secrets: pw().trim().optional(),
	dockerfile: pw().trim().optional(),
	startCommand: pw().trim().optional(),
	binaries: pw().trim().optional(),
	harness: pw().trim().optional()
}), gw = class {
	generation = 0;
	begin() {
		return this.generation += 1, this.generation;
	}
	isCurrent(e) {
		return e === this.generation;
	}
	invalidate() {
		this.generation += 1;
	}
}, _w = (e) => e.type === "checkbox", vw = (e) => e instanceof Date, yw = (e) => e == null, bw = (e) => typeof e == "object", xw = (e) => !yw(e) && !Array.isArray(e) && bw(e) && !vw(e), Sw = (e) => xw(e) && e.target ? _w(e.target) ? e.target.checked : e.target.value : e, Cw = (e) => e.substring(0, e.search(/\.\d+(\.|$)/)) || e, ww = (e, t) => e.has(Cw(t)), Tw = (e) => {
	let t = e.constructor && e.constructor.prototype;
	return xw(t) && t.hasOwnProperty("isPrototypeOf");
}, Ew = typeof window < "u" && window.HTMLElement !== void 0 && typeof document < "u";
function Dw(e) {
	if (e instanceof Date) return new Date(e);
	let t = typeof FileList < "u" && e instanceof FileList;
	if (Ew && (e instanceof Blob || t)) return e;
	let n = Array.isArray(e);
	if (!n && !(xw(e) && Tw(e))) return e;
	let r = n ? [] : Object.create(Object.getPrototypeOf(e));
	for (let t in e) Object.prototype.hasOwnProperty.call(e, t) && (r[t] = Dw(e[t]));
	return r;
}
var Ow = (e) => /^\w*$/.test(e), kw = (e) => e === void 0, Aw = (e) => Array.isArray(e) ? e.filter(Boolean) : [], jw = (e) => Aw(e.replace(/["|']|\]/g, "").split(/\.|\[/)), $ = (e, t, n) => {
	if (!t || !xw(e)) return n;
	let r = (Ow(t) ? [t] : jw(t)).reduce((e, t) => yw(e) ? e : e[t], e);
	return kw(r) || r === e ? kw(e[t]) ? n : e[t] : r;
}, Mw = (e) => typeof e == "boolean", Nw = (e) => typeof e == "function", Pw = (e, t, n) => {
	let r = -1, i = Ow(t) ? [t] : jw(t), a = i.length, o = a - 1;
	for (; ++r < a;) {
		let t = i[r], a = n;
		if (r !== o) {
			let n = e[t];
			a = xw(n) || Array.isArray(n) ? n : isNaN(+i[r + 1]) ? {} : [];
		}
		if (t === "__proto__" || t === "constructor" || t === "prototype") return;
		e[t] = a, e = e[t];
	}
}, Fw = {
	BLUR: "blur",
	FOCUS_OUT: "focusout",
	CHANGE: "change"
}, Iw = {
	onBlur: "onBlur",
	onChange: "onChange",
	onSubmit: "onSubmit",
	onTouched: "onTouched",
	all: "all"
}, Lw = {
	max: "max",
	min: "min",
	maxLength: "maxLength",
	minLength: "minLength",
	pattern: "pattern",
	required: "required",
	validate: "validate"
}, Rw = t.createContext(null);
Rw.displayName = "HookFormControlContext";
var zw = (e, t, n, r = !0) => {
	let i = { defaultValues: t._defaultValues };
	for (let a in e) Object.defineProperty(i, a, { get: () => {
		let i = a;
		return t._proxyFormState[i] !== Iw.all && (t._proxyFormState[i] = !r || Iw.all), n && (n[i] = !0), e[i];
	} });
	return i;
}, Bw = typeof window < "u" ? t.useLayoutEffect : t.useEffect, Vw = (e) => typeof e == "string", Hw = (e, t, n, r, i) => Vw(e) ? (r && t.watch.add(e), $(n, e, i)) : Array.isArray(e) ? e.map((e) => (r && t.watch.add(e), $(n, e))) : (r && (t.watchAll = !0), n), Uw = (e) => yw(e) || !bw(e);
function Ww(e, t, n = /* @__PURE__ */ new WeakSet()) {
	if (Uw(e) || Uw(t)) return Object.is(e, t);
	if (vw(e) && vw(t)) return Object.is(e.getTime(), t.getTime());
	let r = Object.keys(e), i = Object.keys(t);
	if (r.length !== i.length) return !1;
	if (n.has(e) || n.has(t)) return !0;
	n.add(e), n.add(t);
	for (let a of r) {
		let r = e[a];
		if (!i.includes(a)) return !1;
		if (a !== "ref") {
			let e = t[a];
			if (vw(r) && vw(e) || xw(r) && xw(e) || Array.isArray(r) && Array.isArray(e) ? !Ww(r, e, n) : !Object.is(r, e)) return !1;
		}
	}
	return !0;
}
var Gw = t.createContext(null);
Gw.displayName = "HookFormContext";
var Kw = (e, t, n, r, i) => t ? {
	...n[e],
	types: {
		...n[e] && n[e].types ? n[e].types : {},
		[r]: i || !0
	}
} : {}, qw = (e) => Array.isArray(e) ? e : [e], Jw = () => {
	let e = [];
	return {
		get observers() {
			return e;
		},
		next: (t) => {
			for (let n of e) n.next && n.next(t);
		},
		subscribe: (t) => (e.push(t), { unsubscribe: () => {
			e = e.filter((e) => e !== t);
		} }),
		unsubscribe: () => {
			e = [];
		}
	};
};
function Yw(e, t) {
	let n = {};
	for (let r in e) if (e.hasOwnProperty(r)) {
		let i = e[r], a = t[r];
		if (i && xw(i) && a) {
			let e = Yw(i, a);
			xw(e) && (n[r] = e);
		} else e[r] && (n[r] = a);
	}
	return n;
}
var Xw = (e) => xw(e) && !Object.keys(e).length, Zw = (e) => e.type === "file", Qw = (e) => {
	if (!Ew) return !1;
	let t = e ? e.ownerDocument : 0;
	return e instanceof (t && t.defaultView ? t.defaultView.HTMLElement : HTMLElement);
}, $w = (e) => e.type === "select-multiple", eT = (e) => e.type === "radio", tT = (e) => eT(e) || _w(e), nT = (e) => Qw(e) && e.isConnected;
function rT(e, t) {
	let n = t.slice(0, -1).length, r = 0;
	for (; r < n;) e = kw(e) ? r++ : e[t[r++]];
	return e;
}
function iT(e) {
	for (let t in e) if (e.hasOwnProperty(t) && !kw(e[t])) return !1;
	return !0;
}
function aT(e, t) {
	let n = Array.isArray(t) ? t : Ow(t) ? [t] : jw(t), r = n.length === 1 ? e : rT(e, n), i = n.length - 1, a = n[i];
	return r && delete r[a], i !== 0 && (xw(r) && Xw(r) || Array.isArray(r) && iT(r)) && aT(e, n.slice(0, -1)), e;
}
var oT = (e) => {
	for (let t in e) if (Nw(e[t])) return !0;
	return !1;
};
function sT(e) {
	return Array.isArray(e) || xw(e) && !oT(e);
}
function cT(e, t = {}) {
	for (let n in e) {
		let r = e[n];
		sT(r) ? (t[n] = Array.isArray(r) ? [] : {}, cT(r, t[n])) : kw(r) || (t[n] = !0);
	}
	return t;
}
function lT(e, t, n) {
	n ||= cT(t);
	for (let r in e) {
		let i = e[r];
		if (sT(i)) kw(t) || Uw(n[r]) ? n[r] = cT(i, Array.isArray(i) ? [] : {}) : lT(i, yw(t) ? {} : t[r], n[r]);
		else {
			let e = t[r];
			n[r] = !Ww(i, e);
		}
	}
	return n;
}
var uT = {
	value: !1,
	isValid: !1
}, dT = {
	value: !0,
	isValid: !0
}, fT = (e) => {
	if (Array.isArray(e)) {
		if (e.length > 1) {
			let t = e.filter((e) => e && e.checked && !e.disabled).map((e) => e.value);
			return {
				value: t,
				isValid: !!t.length
			};
		}
		return e[0].checked && !e[0].disabled ? e[0].attributes && !kw(e[0].attributes.value) ? kw(e[0].value) || e[0].value === "" ? dT : {
			value: e[0].value,
			isValid: !0
		} : dT : uT;
	}
	return uT;
}, pT = (e, { valueAsNumber: t, valueAsDate: n, setValueAs: r }) => kw(e) ? e : t ? e === "" ? NaN : e && +e : n && Vw(e) ? new Date(e) : r ? r(e) : e, mT = {
	isValid: !1,
	value: null
}, hT = (e) => Array.isArray(e) ? e.reduce((e, t) => t && t.checked && !t.disabled ? {
	isValid: !0,
	value: t.value
} : e, mT) : mT;
function gT(e) {
	let t = e.ref;
	return Zw(t) ? t.files : eT(t) ? hT(e.refs).value : $w(t) ? [...t.selectedOptions].map(({ value: e }) => e) : _w(t) ? fT(e.refs).value : pT(kw(t.value) ? e.ref.value : t.value, e);
}
var _T = (e, t, n, r) => {
	let i = {};
	for (let n of e) {
		let e = $(t, n);
		e && Pw(i, n, e._f);
	}
	return {
		criteriaMode: n,
		names: [...e],
		fields: i,
		shouldUseNativeValidation: r
	};
}, vT = (e) => e instanceof RegExp, yT = (e) => kw(e) ? e : vT(e) ? e.source : xw(e) ? vT(e.value) ? e.value.source : e.value : e, bT = (e) => ({
	isOnSubmit: !e || e === Iw.onSubmit,
	isOnBlur: e === Iw.onBlur,
	isOnChange: e === Iw.onChange,
	isOnAll: e === Iw.all,
	isOnTouch: e === Iw.onTouched
}), xT = "AsyncFunction", ST = (e) => !!e && !!e.validate && !!(Nw(e.validate) && e.validate.constructor.name === xT || xw(e.validate) && Object.values(e.validate).find((e) => e.constructor.name === xT)), CT = (e) => e.mount && (e.required || e.min || e.max || e.maxLength || e.minLength || e.pattern || e.validate), wT = (e, t, n) => !n && (t.watchAll || t.watch.has(e) || [...t.watch].some((t) => e.startsWith(t) && /^\.\w+/.test(e.slice(t.length)))), TT = (e, t, n, r) => {
	for (let i of n || Object.keys(e)) {
		let n = $(e, i);
		if (n) {
			let { _f: e, ...a } = n;
			if (e) {
				if (e.refs && e.refs[0] && t(e.refs[0], i) && !r || e.ref && t(e.ref, e.name) && !r) return !0;
				if (TT(a, t)) break;
			} else if (xw(a) && TT(a, t)) break;
		}
	}
};
function ET(e, t, n) {
	let r = $(e, n);
	if (r || Ow(n)) return {
		error: r,
		name: n
	};
	let i = n.split(".");
	for (; i.length;) {
		let r = i.join("."), a = $(t, r), o = $(e, r);
		if (a && !Array.isArray(a) && n !== r) return { name: n };
		if (o && o.type) return {
			name: r,
			error: o
		};
		if (o && o.root && o.root.type) return {
			name: `${r}.root`,
			error: o.root
		};
		i.pop();
	}
	return { name: n };
}
var DT = (e, t, n, r) => {
	n(e);
	let { name: i, ...a } = e;
	return Xw(a) || Object.keys(a).length >= Object.keys(t).length || Object.keys(a).find((e) => t[e] === (!r || Iw.all));
}, OT = (e, t, n) => !e || !t || e === t || qw(e).some((e) => e && (n ? e === t : e.startsWith(t) || t.startsWith(e))), kT = (e, t, n, r, i) => i.isOnAll ? !1 : !n && i.isOnTouch ? !(t || e) : (n ? r.isOnBlur : i.isOnBlur) ? !e : !(n ? r.isOnChange : i.isOnChange) || e, AT = (e, t) => !Aw($(e, t)).length && aT(e, t), jT = (e, t, n) => {
	let r = qw($(e, n));
	return Pw(r, "root", t[n]), Pw(e, n, r), e;
};
function MT(e, t, n = "validate") {
	if (Vw(e) || Array.isArray(e) && e.every(Vw) || Mw(e) && !e) return {
		type: n,
		message: Vw(e) ? e : "",
		ref: t
	};
}
var NT = (e) => xw(e) && !vT(e) ? e : {
	value: e,
	message: ""
}, PT = async (e, t, n, r, i, a) => {
	let { ref: o, refs: s, required: c, maxLength: l, minLength: u, min: d, max: f, pattern: p, validate: m, name: h, valueAsNumber: g, mount: _ } = e._f, v = $(n, h);
	if (!_ || t.has(h)) return {};
	let y = s ? s[0] : o, b = (e) => {
		i && y.reportValidity && (y.setCustomValidity(Mw(e) ? "" : e || ""), y.reportValidity());
	}, x = {}, S = eT(o), C = _w(o), w = S || C, T = (g || Zw(o)) && kw(o.value) && kw(v) || Qw(o) && o.value === "" || v === "" || Array.isArray(v) && !v.length, E = Kw.bind(null, h, r, x), D = (e, t, n, r = Lw.maxLength, i = Lw.minLength) => {
		let a = e ? t : n;
		x[h] = {
			type: e ? r : i,
			message: a,
			ref: o,
			...E(e ? r : i, a)
		};
	};
	if (a ? !Array.isArray(v) || !v.length : c && (!w && (T || yw(v)) || Mw(v) && !v || C && !fT(s).isValid || S && !hT(s).isValid)) {
		let { value: e, message: t } = Vw(c) ? {
			value: !!c,
			message: c
		} : NT(c);
		if (e && (x[h] = {
			type: Lw.required,
			message: t,
			ref: y,
			...E(Lw.required, t)
		}, !r)) return b(t), x;
	}
	if (!T && (!yw(d) || !yw(f))) {
		let e, t, n = NT(f), i = NT(d);
		if (!yw(v) && !isNaN(v)) {
			let r = o.valueAsNumber || v && +v;
			yw(n.value) || (e = r > n.value), yw(i.value) || (t = r < i.value);
		} else {
			let r = o.valueAsDate || new Date(v), a = (e) => /* @__PURE__ */ new Date((/* @__PURE__ */ new Date()).toDateString() + " " + e), s = o.type == "time", c = o.type == "week";
			Vw(n.value) && v && (e = s ? a(v) > a(n.value) : c ? v > n.value : r > new Date(n.value)), Vw(i.value) && v && (t = s ? a(v) < a(i.value) : c ? v < i.value : r < new Date(i.value));
		}
		if ((e || t) && (D(!!e, n.message, i.message, Lw.max, Lw.min), !r)) return b(x[h].message), x;
	}
	if ((l || u) && !T && (Vw(v) || a && Array.isArray(v))) {
		let e = NT(l), t = NT(u), n = !yw(e.value) && v.length > +e.value, i = !yw(t.value) && v.length < +t.value;
		if ((n || i) && (D(n, e.message, t.message), !r)) return b(x[h].message), x;
	}
	if (p && !T && Vw(v)) {
		let { value: e, message: t } = NT(p);
		if (vT(e) && !v.match(e) && (x[h] = {
			type: Lw.pattern,
			message: t,
			ref: o,
			...E(Lw.pattern, t)
		}, !r)) return b(t), x;
	}
	if (m) {
		if (Nw(m)) {
			let e = MT(await m(v, n), y);
			if (e && (x[h] = {
				...e,
				...E(Lw.validate, e.message)
			}, !r)) return b(e.message), x;
		} else if (xw(m)) {
			let e = {};
			for (let t in m) {
				if (!Xw(e) && !r) break;
				let i = MT(await m[t](v, n), y, t);
				i && (e = {
					...i,
					...E(t, i.message)
				}, b(i.message), r && (x[h] = e));
			}
			if (!Xw(e) && (x[h] = {
				ref: y,
				...e
			}, !r)) return x;
		}
	}
	return b(!0), x;
}, FT = {
	mode: Iw.onSubmit,
	reValidateMode: Iw.onChange,
	shouldFocusError: !0
};
function IT(e = {}) {
	let t = {
		...FT,
		...e
	}, n = {
		submitCount: 0,
		isDirty: !1,
		isReady: !1,
		isLoading: Nw(t.defaultValues),
		isValidating: !1,
		isSubmitted: !1,
		isSubmitting: !1,
		isSubmitSuccessful: !1,
		isValid: !1,
		touchedFields: {},
		dirtyFields: {},
		validatingFields: {},
		errors: t.errors || {},
		disabled: t.disabled || !1
	}, r = {}, i = (xw(t.defaultValues) || xw(t.values)) && Dw(t.defaultValues || t.values) || {}, a = t.shouldUnregister ? {} : Dw(i), o = {
		action: !1,
		mount: !1,
		watch: !1,
		keepIsValid: !1
	}, s = {
		mount: /* @__PURE__ */ new Set(),
		disabled: /* @__PURE__ */ new Set(),
		unMount: /* @__PURE__ */ new Set(),
		array: /* @__PURE__ */ new Set(),
		watch: /* @__PURE__ */ new Set()
	}, c, l = 0, u = {
		isDirty: !1,
		dirtyFields: !1,
		validatingFields: !1,
		touchedFields: !1,
		isValidating: !1,
		isValid: !1,
		errors: !1
	}, d = { ...u }, f = { ...d }, p = {
		array: Jw(),
		state: Jw()
	}, m = t.criteriaMode === Iw.all, h = (e) => (t) => {
		clearTimeout(l), l = setTimeout(e, t);
	}, g = async (e) => {
		if (!o.keepIsValid && !t.disabled && (d.isValid || f.isValid || e)) {
			let e;
			t.resolver ? (e = Xw((await w()).errors), _()) : e = await E(r, !0), e !== n.isValid && p.state.next({ isValid: e });
		}
	}, _ = (e, r) => {
		!t.disabled && (d.isValidating || d.validatingFields || f.isValidating || f.validatingFields) && ((e || Array.from(s.mount)).forEach((e) => {
			e && (r ? Pw(n.validatingFields, e, r) : aT(n.validatingFields, e));
		}), p.state.next({
			validatingFields: n.validatingFields,
			isValidating: !Xw(n.validatingFields)
		}));
	}, v = (e, s = [], c, l, u = !0, m = !0) => {
		if (l && c && !t.disabled) {
			if (o.action = !0, m && Array.isArray($(r, e))) {
				let t = c($(r, e), l.argA, l.argB);
				u && Pw(r, e, t);
			}
			if (m && Array.isArray($(n.errors, e))) {
				let t = c($(n.errors, e), l.argA, l.argB);
				u && Pw(n.errors, e, t), AT(n.errors, e);
			}
			if ((d.touchedFields || f.touchedFields) && m && Array.isArray($(n.touchedFields, e))) {
				let t = c($(n.touchedFields, e), l.argA, l.argB);
				u && Pw(n.touchedFields, e, t);
			}
			(d.dirtyFields || f.dirtyFields) && (n.dirtyFields = lT(i, a)), p.state.next({
				name: e,
				isDirty: O(e, s),
				dirtyFields: n.dirtyFields,
				errors: n.errors,
				isValid: n.isValid
			});
		} else Pw(a, e, s);
	}, y = (e, t) => {
		Pw(n.errors, e, t), p.state.next({ errors: n.errors });
	}, b = (e) => {
		n.errors = e, p.state.next({
			errors: n.errors,
			isValid: !1
		});
	}, x = (e, t, n, s) => {
		let c = $(r, e);
		if (c) {
			let r = $(a, e, kw(n) ? $(i, e) : n);
			kw(r) || s && s.defaultChecked || t ? Pw(a, e, t ? r : gT(c._f)) : A(e, r), o.mount && !o.action && g();
		}
	}, S = (e, r, a, o, s) => {
		let c = !1, l = !1, u = { name: e };
		if (!t.disabled) {
			if (!a || o) {
				(d.isDirty || f.isDirty) && (l = n.isDirty, n.isDirty = u.isDirty = O(), c = l !== u.isDirty);
				let t = Ww($(i, e), r);
				l = !!$(n.dirtyFields, e), t ? aT(n.dirtyFields, e) : Pw(n.dirtyFields, e, !0), u.dirtyFields = n.dirtyFields, c ||= (d.dirtyFields || f.dirtyFields) && l !== !t;
			}
			if (a) {
				let t = $(n.touchedFields, e);
				t || (Pw(n.touchedFields, e, a), u.touchedFields = n.touchedFields, c ||= (d.touchedFields || f.touchedFields) && t !== a);
			}
			c && s && p.state.next(u);
		}
		return c ? u : {};
	}, C = (e, r, i, a) => {
		let o = $(n.errors, e), s = (d.isValid || f.isValid) && Mw(r) && n.isValid !== r;
		if (t.delayError && i ? (c = h(() => y(e, i)), c(t.delayError)) : (clearTimeout(l), c = null, i ? Pw(n.errors, e, i) : aT(n.errors, e)), (i ? !Ww(o, i) : o) || !Xw(a) || s) {
			let t = {
				...a,
				...s && Mw(r) ? { isValid: r } : {},
				errors: n.errors,
				name: e
			};
			n = {
				...n,
				...t
			}, p.state.next(t);
		}
	}, w = async (e) => (_(e, !0), await t.resolver(a, t.context, _T(e || s.mount, r, t.criteriaMode, t.shouldUseNativeValidation))), T = async (e) => {
		let { errors: t } = await w(e);
		if (_(e), e) for (let r of e) {
			let e = $(t, r);
			e ? Pw(n.errors, r, e) : aT(n.errors, r);
		}
		else n.errors = t;
		return t;
	}, E = async (r, i, o = { valid: !0 }) => {
		for (let c in r) {
			let l = r[c];
			if (l) {
				let { _f: r, ...c } = l;
				if (r) {
					let c = s.array.has(r.name), u = l._f && ST(l._f);
					u && d.validatingFields && _([r.name], !0);
					let f = await PT(l, s.disabled, a, m, t.shouldUseNativeValidation && !i, c);
					if (u && d.validatingFields && _([r.name]), f[r.name] && (o.valid = !1, i || e.shouldUseNativeValidation)) break;
					!i && ($(f, r.name) ? c ? jT(n.errors, f, r.name) : Pw(n.errors, r.name, f[r.name]) : aT(n.errors, r.name));
				}
				!Xw(c) && await E(c, i, o);
			}
		}
		return o.valid;
	}, D = () => {
		for (let e of s.unMount) {
			let t = $(r, e);
			t && (t._f.refs ? t._f.refs.every((e) => !nT(e)) : !nT(t._f.ref)) && ue(e);
		}
		s.unMount = /* @__PURE__ */ new Set();
	}, O = (e, n) => !t.disabled && (e && n && Pw(a, e, n), !Ww(ie(), i)), ee = (e, t, n) => Hw(e, s, { ...o.mount ? a : kw(t) ? i : Vw(e) ? { [e]: t } : t }, n, t), k = (e) => Aw($(o.mount ? a : i, e, t.shouldUnregister ? $(i, e, []) : [])), A = (e, t, n = {}) => {
		let i = $(r, e), o = t;
		if (i) {
			let n = i._f;
			n && (!n.disabled && Pw(a, e, pT(t, n)), o = Qw(n.ref) && yw(t) ? "" : t, $w(n.ref) ? [...n.ref.options].forEach((e) => e.selected = o.includes(e.value)) : n.refs ? _w(n.ref) ? n.refs.forEach((e) => {
				(!e.defaultChecked || !e.disabled) && (e.checked = Array.isArray(o) ? !!o.find((t) => t === e.value) : o === e.value || !!o);
			}) : n.refs.forEach((e) => e.checked = e.value === o) : Zw(n.ref) ? n.ref.value = "" : (n.ref.value = o, n.ref.type || p.state.next({
				name: e,
				values: Dw(a)
			})));
		}
		(n.shouldDirty || n.shouldTouch) && S(e, o, n.shouldTouch, n.shouldDirty, !0), n.shouldValidate && M(e);
	}, te = (e, t, n) => {
		for (let i in t) {
			if (!t.hasOwnProperty(i)) return;
			let a = t[i], o = e + "." + i, c = $(r, o);
			(s.array.has(e) || xw(a) || c && !c._f) && !vw(a) ? te(o, a, n) : A(o, a, n);
		}
	}, j = (e, t, c = {}) => {
		let l = $(r, e), u = s.array.has(e), m = Dw(t);
		Pw(a, e, m), u ? (p.array.next({
			name: e,
			values: Dw(a)
		}), (d.isDirty || d.dirtyFields || f.isDirty || f.dirtyFields) && c.shouldDirty && p.state.next({
			name: e,
			dirtyFields: lT(i, a),
			isDirty: O(e, m)
		})) : l && !l._f && !yw(m) ? te(e, m, c) : A(e, m, c), wT(e, s) ? p.state.next({
			...n,
			name: e,
			values: Dw(a)
		}) : p.state.next({
			name: o.mount ? e : void 0,
			values: Dw(a)
		});
	}, ne = async (e) => {
		o.mount = !0;
		let i = e.target, l = i.name, u = !0, h = $(r, l), v = (e) => {
			u = Number.isNaN(e) || vw(e) && isNaN(e.getTime()) || Ww(e, $(a, l, e));
		}, y = bT(t.mode), b = bT(t.reValidateMode);
		if (h) {
			let o, x, T = i.type ? gT(h._f) : Sw(e), D = e.type === Fw.BLUR || e.type === Fw.FOCUS_OUT, O = !CT(h._f) && !t.resolver && !$(n.errors, l) && !h._f.deps || kT(D, $(n.touchedFields, l), n.isSubmitted, b, y), ee = wT(l, s, D);
			Pw(a, l, T), D ? (!i || !i.readOnly) && (h._f.onBlur && h._f.onBlur(e), c && c(0)) : h._f.onChange && h._f.onChange(e);
			let k = S(l, T, D), A = !Xw(k) || ee;
			if (!D && p.state.next({
				name: l,
				type: e.type,
				values: Dw(a)
			}), O) return (d.isValid || f.isValid) && (t.mode === "onBlur" ? D && g() : D || g()), A && p.state.next({
				name: l,
				...ee ? {} : k
			});
			if (!D && ee && p.state.next({ ...n }), t.resolver) {
				let { errors: e } = await w([l]);
				if (_([l]), v(T), u) {
					let t = ET(n.errors, r, l), i = ET(e, r, t.name || l);
					o = i.error, l = i.name, x = Xw(e);
				}
			} else _([l], !0), o = (await PT(h, s.disabled, a, m, t.shouldUseNativeValidation))[l], _([l]), v(T), u && (o ? x = !1 : (d.isValid || f.isValid) && (x = await E(r, !0)));
			u && (h._f.deps && (!Array.isArray(h._f.deps) || h._f.deps.length > 0) && M(h._f.deps), C(l, x, o, k));
		}
	}, re = (e, t) => {
		if ($(n.errors, t) && e.focus) return e.focus(), 1;
	}, M = async (e, i = {}) => {
		let a, o, c = qw(e);
		if (t.resolver) {
			let t = await T(kw(e) ? e : c);
			a = Xw(t), o = e ? !c.some((e) => $(t, e)) : a;
		} else e ? (o = (await Promise.all(c.map(async (e) => {
			let t = $(r, e);
			return await E(t && t._f ? { [e]: t } : t);
		}))).every(Boolean), !(!o && !n.isValid) && g()) : o = a = await E(r);
		return p.state.next({
			...!Vw(e) || (d.isValid || f.isValid) && a !== n.isValid ? {} : { name: e },
			...t.resolver || !e ? { isValid: a } : {},
			errors: n.errors
		}), i.shouldFocus && !o && TT(r, re, e ? c : s.mount), o;
	}, ie = (e, t) => {
		let r = { ...o.mount ? a : i };
		return t && (r = Yw(t.dirtyFields ? n.dirtyFields : n.touchedFields, r)), kw(e) ? r : Vw(e) ? $(r, e) : e.map((e) => $(r, e));
	}, N = (e, t) => ({
		invalid: !!$((t || n).errors, e),
		isDirty: !!$((t || n).dirtyFields, e),
		error: $((t || n).errors, e),
		isValidating: !!$(n.validatingFields, e),
		isTouched: !!$((t || n).touchedFields, e)
	}), ae = (e) => {
		let t = e ? qw(e) : void 0;
		t?.forEach((e) => aT(n.errors, e)), t ? t.forEach((e) => {
			p.state.next({
				name: e,
				errors: n.errors
			});
		}) : p.state.next({ errors: {} });
	}, oe = (e, t, i) => {
		let a = ($(r, e, { _f: {} })._f || {}).ref, { ref: o, message: s, type: c, ...l } = $(n.errors, e) || {};
		Pw(n.errors, e, {
			...l,
			...t,
			ref: a
		}), p.state.next({
			name: e,
			errors: n.errors,
			isValid: !1
		}), i && i.shouldFocus && a && a.focus && a.focus();
	}, se = (e, t) => Nw(e) ? p.state.subscribe({ next: (n) => "values" in n && e(ee(void 0, t), n) }) : ee(e, t, !0), ce = (e) => p.state.subscribe({ next: (t) => {
		OT(e.name, t.name, e.exact) && DT(t, e.formState || d, ve, e.reRenderRoot) && e.callback({
			values: { ...a },
			...n,
			...t,
			defaultValues: i
		});
	} }).unsubscribe, le = (e) => (o.mount = !0, f = {
		...f,
		...e.formState
	}, ce({
		...e,
		formState: {
			...u,
			...e.formState
		}
	})), ue = (e, o = {}) => {
		for (let c of e ? qw(e) : s.mount) s.mount.delete(c), s.array.delete(c), o.keepValue || (aT(r, c), aT(a, c)), !o.keepError && aT(n.errors, c), !o.keepDirty && aT(n.dirtyFields, c), !o.keepTouched && aT(n.touchedFields, c), !o.keepIsValidating && aT(n.validatingFields, c), !t.shouldUnregister && !o.keepDefaultValue && aT(i, c);
		p.state.next({ values: Dw(a) }), p.state.next({
			...n,
			...o.keepDirty ? { isDirty: O() } : {}
		}), !o.keepIsValid && g();
	}, P = ({ disabled: e, name: t }) => {
		if (Mw(e) && o.mount || e || s.disabled.has(t)) {
			let n = s.disabled.has(t) !== !!e;
			e ? s.disabled.add(t) : s.disabled.delete(t), n && o.mount && !o.action && g();
		}
	}, de = (e, n = {}) => {
		let a = $(r, e), c = Mw(n.disabled) || Mw(t.disabled);
		return Pw(r, e, {
			...a || {},
			_f: {
				...a && a._f ? a._f : { ref: { name: e } },
				name: e,
				mount: !0,
				...n
			}
		}), s.mount.add(e), a ? P({
			disabled: Mw(n.disabled) ? n.disabled : t.disabled,
			name: e
		}) : x(e, !0, n.value), {
			...c ? { disabled: n.disabled || t.disabled } : {},
			...t.progressive ? {
				required: !!n.required,
				min: yT(n.min),
				max: yT(n.max),
				minLength: yT(n.minLength),
				maxLength: yT(n.maxLength),
				pattern: yT(n.pattern)
			} : {},
			name: e,
			onChange: ne,
			onBlur: ne,
			ref: (c) => {
				if (c) {
					de(e, n), a = $(r, e);
					let t = kw(c.value) && c.querySelectorAll && c.querySelectorAll("input,select,textarea")[0] || c, o = tT(t), s = a._f.refs || [];
					if (o ? s.find((e) => e === t) : t === a._f.ref) return;
					Pw(r, e, { _f: {
						...a._f,
						...o ? {
							refs: [
								...s.filter(nT),
								t,
								...Array.isArray($(i, e)) ? [{}] : []
							],
							ref: {
								type: t.type,
								name: e
							}
						} : { ref: t }
					} }), x(e, !1, void 0, t);
				} else a = $(r, e, {}), a._f && (a._f.mount = !1), (t.shouldUnregister || n.shouldUnregister) && !(ww(s.array, e) && o.action) && s.unMount.add(e);
			}
		};
	}, fe = () => t.shouldFocusError && TT(r, re, s.mount), F = (e) => {
		Mw(e) && (p.state.next({ disabled: e }), TT(r, (t, n) => {
			let i = $(r, n);
			i && (t.disabled = i._f.disabled || e, Array.isArray(i._f.refs) && i._f.refs.forEach((t) => {
				t.disabled = i._f.disabled || e;
			}));
		}, 0, !1));
	}, pe = (e, i) => async (o) => {
		let c;
		o && (o.preventDefault && o.preventDefault(), o.persist && o.persist());
		let l = Dw(a);
		if (p.state.next({ isSubmitting: !0 }), t.resolver) {
			let { errors: e, values: t } = await w();
			_(), n.errors = e, l = Dw(t);
		} else await E(r);
		if (s.disabled.size) for (let e of s.disabled) aT(l, e);
		if (aT(n.errors, "root"), Xw(n.errors)) {
			p.state.next({ errors: {} });
			try {
				await e(l, o);
			} catch (e) {
				c = e;
			}
		} else i && await i({ ...n.errors }, o), fe(), setTimeout(fe);
		if (p.state.next({
			isSubmitted: !0,
			isSubmitting: !1,
			isSubmitSuccessful: Xw(n.errors) && !c,
			submitCount: n.submitCount + 1,
			errors: n.errors
		}), c) throw c;
	}, me = (e, t = {}) => {
		$(r, e) && (kw(t.defaultValue) ? j(e, Dw($(i, e))) : (j(e, t.defaultValue), Pw(i, e, Dw(t.defaultValue))), t.keepTouched || aT(n.touchedFields, e), t.keepDirty || (aT(n.dirtyFields, e), n.isDirty = t.defaultValue ? O(e, Dw($(i, e))) : O()), t.keepError || (aT(n.errors, e), d.isValid && g()), p.state.next({ ...n }));
	}, he = (e, c = {}) => {
		let l = e ? Dw(e) : i, u = Dw(l), f = Xw(e), m = f ? i : u;
		if (c.keepDefaultValues || (i = l), !c.keepValues) {
			if (c.keepDirtyValues) {
				let e = /* @__PURE__ */ new Set([...s.mount, ...Object.keys(lT(i, a))]);
				for (let t of Array.from(e)) {
					let e = $(n.dirtyFields, t), r = $(a, t), i = $(m, t);
					e && !kw(r) ? Pw(m, t, r) : !e && !kw(i) && j(t, i);
				}
			} else {
				if (Ew && kw(e)) for (let e of s.mount) {
					let t = $(r, e);
					if (t && t._f) {
						let e = Array.isArray(t._f.refs) ? t._f.refs[0] : t._f.ref;
						if (Qw(e)) {
							let t = e.closest("form");
							if (t) {
								t.reset();
								break;
							}
						}
					}
				}
				if (c.keepFieldsRef) for (let e of s.mount) j(e, $(m, e));
				else r = {};
			}
			a = t.shouldUnregister ? c.keepDefaultValues ? Dw(i) : {} : Dw(m), p.array.next({ values: { ...m } }), p.state.next({ values: { ...m } });
		}
		s = {
			mount: c.keepDirtyValues ? s.mount : /* @__PURE__ */ new Set(),
			unMount: /* @__PURE__ */ new Set(),
			array: /* @__PURE__ */ new Set(),
			disabled: /* @__PURE__ */ new Set(),
			watch: /* @__PURE__ */ new Set(),
			watchAll: !1,
			focus: ""
		}, o.mount = !d.isValid || !!c.keepIsValid || !!c.keepDirtyValues || !t.shouldUnregister && !Xw(m), o.watch = !!t.shouldUnregister, o.keepIsValid = !!c.keepIsValid, o.action = !1, c.keepErrors || (n.errors = {}), p.state.next({
			submitCount: c.keepSubmitCount ? n.submitCount : 0,
			isDirty: f ? !1 : c.keepDirty ? n.isDirty : !!(c.keepDefaultValues && !Ww(e, i)),
			isSubmitted: c.keepIsSubmitted ? n.isSubmitted : !1,
			dirtyFields: f ? {} : c.keepDirtyValues ? c.keepDefaultValues && a ? lT(i, a) : n.dirtyFields : c.keepDefaultValues && e ? lT(i, e) : c.keepDirty ? n.dirtyFields : {},
			touchedFields: c.keepTouched ? n.touchedFields : {},
			errors: c.keepErrors ? n.errors : {},
			isSubmitSuccessful: c.keepIsSubmitSuccessful ? n.isSubmitSuccessful : !1,
			isSubmitting: !1,
			defaultValues: i
		});
	}, ge = (e, n) => he(Nw(e) ? e(a) : e, {
		...t.resetOptions,
		...n
	}), _e = (e, t = {}) => {
		let n = $(r, e), i = n && n._f;
		if (i) {
			let e = i.refs ? i.refs[0] : i.ref;
			e.focus && setTimeout(() => {
				e.focus(), t.shouldSelect && Nw(e.select) && e.select();
			});
		}
	}, ve = (e) => {
		n = {
			...n,
			...e
		};
	}, ye = {
		control: {
			register: de,
			unregister: ue,
			getFieldState: N,
			handleSubmit: pe,
			setError: oe,
			_subscribe: ce,
			_runSchema: w,
			_updateIsValidating: _,
			_focusError: fe,
			_getWatch: ee,
			_getDirty: O,
			_setValid: g,
			_setFieldArray: v,
			_setDisabledField: P,
			_setErrors: b,
			_getFieldArray: k,
			_reset: he,
			_resetDefaultValues: () => Nw(t.defaultValues) && t.defaultValues().then((e) => {
				ge(e, t.resetOptions), p.state.next({ isLoading: !1 });
			}),
			_removeUnmounted: D,
			_disableForm: F,
			_subjects: p,
			_proxyFormState: d,
			get _fields() {
				return r;
			},
			get _formValues() {
				return a;
			},
			get _state() {
				return o;
			},
			set _state(e) {
				o = e;
			},
			get _defaultValues() {
				return i;
			},
			get _names() {
				return s;
			},
			set _names(e) {
				s = e;
			},
			get _formState() {
				return n;
			},
			get _options() {
				return t;
			},
			set _options(e) {
				t = {
					...t,
					...e
				};
			}
		},
		subscribe: le,
		trigger: M,
		register: de,
		handleSubmit: pe,
		watch: se,
		setValue: j,
		getValues: ie,
		reset: ge,
		resetField: me,
		clearErrors: ae,
		unregister: ue,
		setError: oe,
		setFocus: _e,
		getFieldState: N
	};
	return {
		...ye,
		formControl: ye
	};
}
function LT(e = {}) {
	let n = t.useRef(void 0), r = t.useRef(void 0), [i, a] = t.useState({
		isDirty: !1,
		isValidating: !1,
		isLoading: Nw(e.defaultValues),
		isSubmitted: !1,
		isSubmitting: !1,
		isSubmitSuccessful: !1,
		isValid: !1,
		submitCount: 0,
		dirtyFields: {},
		touchedFields: {},
		validatingFields: {},
		errors: e.errors || {},
		disabled: e.disabled || !1,
		isReady: !1,
		defaultValues: Nw(e.defaultValues) ? void 0 : e.defaultValues
	});
	if (!n.current) {
		if (e.formControl) n.current = {
			...e.formControl,
			formState: i
		}, e.defaultValues && !Nw(e.defaultValues) && e.formControl.reset(e.defaultValues, e.resetOptions);
		else {
			let { formControl: t, ...r } = IT(e);
			n.current = {
				...r,
				formState: i
			};
		}
	}
	let o = n.current.control;
	return o._options = e, Bw(() => {
		let e = o._subscribe({
			formState: o._proxyFormState,
			callback: () => a({ ...o._formState }),
			reRenderRoot: !0
		});
		return a((e) => ({
			...e,
			isReady: !0
		})), o._formState.isReady = !0, e;
	}, [o]), t.useEffect(() => o._disableForm(e.disabled), [o, e.disabled]), t.useEffect(() => {
		e.mode && (o._options.mode = e.mode), e.reValidateMode && (o._options.reValidateMode = e.reValidateMode);
	}, [
		o,
		e.mode,
		e.reValidateMode
	]), t.useEffect(() => {
		e.errors && (o._setErrors(e.errors), o._focusError());
	}, [o, e.errors]), t.useEffect(() => {
		e.shouldUnregister && o._subjects.state.next({ values: o._getWatch() });
	}, [o, e.shouldUnregister]), t.useEffect(() => {
		if (o._proxyFormState.isDirty) {
			let e = o._getDirty();
			e !== i.isDirty && o._subjects.state.next({ isDirty: e });
		}
	}, [o, i.isDirty]), t.useEffect(() => {
		e.values && !Ww(e.values, r.current) ? (o._reset(e.values, {
			keepFieldsRef: !0,
			...o._options.resetOptions
		}), o._options.resetOptions?.keepIsValid || o._setValid(), r.current = e.values, a((e) => ({ ...e }))) : o._resetDefaultValues();
	}, [o, e.values]), t.useEffect(() => {
		o._state.mount || (o._setValid(), o._state.mount = !0), o._state.watch && (o._state.watch = !1, o._subjects.state.next({ ...o._formState })), o._removeUnmounted();
	}), n.current.formState = t.useMemo(() => zw(i, o), [o, i]), n.current;
}
//#endregion
//#region node_modules/.pnpm/@hookform+resolvers@4.1.3_react-hook-form@7.71.2_react@19.2.8_/node_modules/@hookform/resolvers/dist/resolvers.mjs
var RT = (e, t, n) => {
	if (e && "reportValidity" in e) {
		let r = $(n, t);
		e.setCustomValidity(r && r.message || ""), e.reportValidity();
	}
}, zT = (e, t) => {
	for (let n in t.fields) {
		let r = t.fields[n];
		r && r.ref && "reportValidity" in r.ref ? RT(r.ref, n, e) : r && r.refs && r.refs.forEach((t) => RT(t, n, e));
	}
}, BT = (e, t) => {
	t.shouldUseNativeValidation && zT(e, t);
	let n = {};
	for (let r in e) {
		let i = $(t.fields, r), a = Object.assign(e[r] || {}, { ref: i && i.ref });
		if (VT(t.names || Object.keys(e), r)) {
			let e = Object.assign({}, $(n, r));
			Pw(e, "root", a), Pw(n, r, e);
		} else Pw(n, r, a);
	}
	return n;
}, VT = (e, t) => {
	let n = HT(t);
	return e.some((e) => HT(e).match(`^${n}\\.\\d+`));
};
function HT(e) {
	return e.replace(/\]|\[/g, "");
}
//#endregion
//#region node_modules/.pnpm/@hookform+resolvers@4.1.3_react-hook-form@7.71.2_react@19.2.8_/node_modules/@hookform/resolvers/zod/dist/zod.mjs
function UT(e, t) {
	for (var n = {}; e.length;) {
		var r = e[0], i = r.code, a = r.message, o = r.path.join(".");
		if (!n[o]) {
			if ("unionErrors" in r) {
				var s = r.unionErrors[0].errors[0];
				n[o] = {
					message: s.message,
					type: s.code
				};
			} else n[o] = {
				message: a,
				type: i
			};
		}
		if ("unionErrors" in r && r.unionErrors.forEach(function(t) {
			return t.errors.forEach(function(t) {
				return e.push(t);
			});
		}), t) {
			var c = n[o].types, l = c && c[r.code];
			n[o] = Kw(o, t, n, i, l ? [].concat(l, r.message) : r.message);
		}
		e.shift();
	}
	return n;
}
function WT(e, t, n) {
	return n === void 0 && (n = {}), function(r, i, a) {
		try {
			return Promise.resolve(function(i, o) {
				try {
					var s = Promise.resolve(e[n.mode === "sync" ? "parse" : "parseAsync"](r, t)).then(function(e) {
						return a.shouldUseNativeValidation && zT({}, a), {
							errors: {},
							values: n.raw ? Object.assign({}, r) : e
						};
					});
				} catch (e) {
					return o(e);
				}
				return s && s.then ? s.then(void 0, o) : s;
			}(0, function(e) {
				if (function(e) {
					return Array.isArray(e?.errors);
				}(e)) return {
					values: {},
					errors: BT(UT(e.errors, !a.shouldUseNativeValidation && a.criteriaMode === "all"), a)
				};
				throw e;
			}));
		} catch (e) {
			return Promise.reject(e);
		}
	};
}
//#endregion
//#region src/routes/NewAgentHardenerManifestRoute/index.tsx
var GT = "The war-game runs your image in a sandbox, so it needs: a sandbox user and group; iproute2; a start command with absolute paths; the agent serving GET /health and POST /v1/chat/completions on the victim port; and NeMo Relay attached (nemo-relay middleware, plus nemo_relay.plugin.activate() held open for the life of the process).", KT = [
	"deepagents",
	"hermes",
	"langchain",
	"langgraph",
	"other"
].map((e) => ({
	value: e,
	children: e
})), qT = () => {
	let e = qi(), t = Ie(), n = Yi(), r = Ve(), [i, a] = d({}), [o, c] = d("agent"), [f, _] = d(null), [v, y] = d(""), [S, C] = d(null), [w, E] = d(!1), { data: D } = Po(e, { query: {} });
	Xi({ items: [
		{
			href: $i(e),
			slotLabel: "Agent Hardening"
		},
		{
			href: ta(e),
			slotLabel: "Manifests"
		},
		{ slotLabel: "New" }
	] });
	let { control: O, handleSubmit: k, watch: A, setError: te, setValue: j } = LT({
		defaultValues: {
			name: "",
			agent: "",
			egress: "",
			env: "",
			port: "",
			secrets: ""
		},
		resolver: WT(hw)
	}), ne = u(new gw()).current, re = A("agent"), M = Ja(), { mutate: ie } = M;
	s(() => {
		if (!re) return;
		let t = ne.begin();
		ie({
			workspace: e,
			agent: re
		}, { onSuccess: (e) => {
			ne.isCurrent(t) && (j("port", String(e.port)), j("secrets", e.secrets.join(", ")), j("egress", e.egress.join(", ")));
		} });
	}, [
		re,
		e,
		ie,
		j,
		ne
	]);
	let N = Ka(), ae = qa(), oe = (t) => {
		_(t), C(null);
		let r = ne.begin();
		N.mutate({
			workspace: e,
			manifestName: A("name") || "byo",
			file: t
		}, {
			onSuccess: (t) => {
				ne.isCurrent(r) && (y(t), ae.mutate({
					workspace: e,
					projectFileset: t
				}, {
					onSuccess: (e) => {
						ne.isCurrent(r) && (C(e), j("dockerfile", e.dockerfile), j("startCommand", e.start_command), j("binaries", e.binaries.join(", ")), j("port", String(e.port)), j("secrets", e.secrets.join(", ")), j("egress", e.egress.join(", ")), j("env", Object.entries(e.env).map(([e, t]) => `${e}=${t}`).join(", ")));
					},
					onError: oa(n, "Could not read the project bundle")
				}));
			},
			onError: oa(n, "Could not upload the project bundle")
		});
	}, se = (e) => S?.unresolved.includes(e) ?? !1, { data: le = [], isLoading: ue } = IS(e), de = l(() => le.flatMap((e) => e.name ? [{
		value: e.name,
		children: e.name
	}] : []), [le]), fe = _o({ mutation: {
		onSuccess: () => {
			r.invalidateQueries({ queryKey: fo(e) }), n.success("Manifest created."), t(ta(e));
		},
		onError: oa(n, "Failed to create the manifest")
	} }), pe = k((t) => {
		if (o === "agent" && !t.agent) {
			te("agent", { message: "Select a deployed agent" });
			return;
		}
		if (o === "project" && !v) {
			n.error("Upload the project bundle first.");
			return;
		}
		let r = LS(t.egress), a = LS(t.secrets), s = RS(t.env), c = t.port ? Number(t.port) : void 0;
		if (c !== void 0 && !Number.isInteger(c)) {
			te("port", { message: "Enter a whole number" });
			return;
		}
		let l = LS(t.binaries);
		fe.mutate({
			workspace: e,
			data: {
				name: t.name,
				...o === "agent" ? { agent: t.agent } : {
					source_type: "project",
					project_fileset: v,
					relay_integration_confirmed: w,
					...t.dockerfile ? { dockerfile: t.dockerfile } : {},
					...t.startCommand ? { start_command: t.startCommand } : {},
					...l.length ? { binaries: l } : {},
					...t.harness ? { harness: t.harness } : {}
				},
				...r.length ? { egress: r } : {},
				...a.length ? { secrets: a } : {},
				...Object.keys(s).length ? { env: s } : {},
				...c === void 0 ? {} : { port: c },
				models: i
			}
		});
	});
	return /* @__PURE__ */ p(h, {
		title: "New Agent Hardener manifest",
		children: /* @__PURE__ */ m(I, {
			className: "h-full overflow-auto",
			gap: "density-2xl",
			padding: "density-2xl",
			children: [/* @__PURE__ */ p(he, {
				className: "p-0",
				slotHeading: "New Manifest",
				slotDescription: "Scaffold a reusable war-game target. Give it an ID, then pick where the agent comes from."
			}), /* @__PURE__ */ p(ge, {
				className: "max-w-[720px]",
				children: /* @__PURE__ */ m(I, {
					gap: "density-xl",
					padding: "density-lg",
					children: [/* @__PURE__ */ p(x, {
						useControllerProps: {
							control: O,
							name: "name"
						},
						formFieldProps: {
							slotLabel: "Manifest ID",
							slotHelp: "Lowercase, e.g. clockbot-hardening."
						}
					}), /* @__PURE__ */ p("form", {
						onSubmit: pe,
						children: /* @__PURE__ */ m(I, {
							gap: "density-xl",
							children: [
								/* @__PURE__ */ p(_e, {
									name: "manifest-source",
									value: o,
									onValueChange: (e) => c(e),
									children: /* @__PURE__ */ m(F, {
										gap: "density-md",
										"data-testid": "manifest-source",
										children: [/* @__PURE__ */ p(ee, {
											value: "agent",
											checked: o === "agent",
											label: "Registered agent",
											description: "An agent already registered on the platform."
										}), /* @__PURE__ */ p(ee, {
											value: "project",
											checked: o === "project",
											label: "Bring your own",
											description: "An image whose Dockerfile you own. We read it and ask only for the rest."
										})]
									})
								}),
								o === "agent" ? /* @__PURE__ */ p(b, {
									useControllerProps: {
										control: O,
										name: "agent"
									},
									loading: ue,
									items: de,
									formFieldProps: { slotLabel: "Deployed Agent" }
								}) : /* @__PURE__ */ m(I, {
									gap: "density-lg",
									"data-testid": "project-upload",
									children: [
										/* @__PURE__ */ p(L, {
											kind: "body/regular/sm",
											className: "text-subtle",
											"data-testid": "image-requirements",
											children: GT
										}),
										/* @__PURE__ */ p(T, {
											accept: { "application/zip": [".zip"] },
											files: f ? [f] : [],
											onDropAccepted: (e) => e[0] && oe(e[0]),
											onRemoveFile: () => {
												ne.invalidate(), _(null), y(""), C(null);
											},
											helperText: N.isPending ? "Uploading…" : ae.isPending ? "Reading the Dockerfile…" : "A .zip of the directory holding your Dockerfile."
										}),
										S?.warnings.map((e) => /* @__PURE__ */ m(L, {
											kind: "body/regular/sm",
											className: "text-subtle",
											children: ["! ", e]
										}, e)),
										S && /* @__PURE__ */ p(x, {
											useControllerProps: {
												control: O,
												name: "dockerfile"
											},
											formFieldProps: {
												slotLabel: "Dockerfile",
												slotInfo: GT,
												slotHelp: se("dockerfile") ? "Several were found — name the one that builds the agent." : "Detected. Edit to override."
											}
										}),
										S && /* @__PURE__ */ p(x, {
											useControllerProps: {
												control: O,
												name: "startCommand"
											},
											formFieldProps: {
												slotLabel: "Start Command",
												slotInfo: "How the sandbox starts the agent. Use absolute paths, e.g. /app/.venv/bin/python server.py: the sandbox replaces the image PATH.",
												slotHelp: se("start_command") ? "The ENTRYPOINT is a shell form, so it cannot be read. Enter it here." : "Derived from the Dockerfile. Edit to override."
											}
										}),
										S && /* @__PURE__ */ p(x, {
											useControllerProps: {
												control: O,
												name: "binaries"
											},
											formFieldProps: {
												slotLabel: "Interpreter Globs",
												slotInfo: "The processes allowed to reach the egress allow-list, e.g. /app/.venv/bin/python*. A glob matching no process grants nothing while looking like it grants something.",
												slotHelp: "Comma-separated."
											}
										}),
										S && /* @__PURE__ */ p(b, {
											useControllerProps: {
												control: O,
												name: "harness"
											},
											items: KT,
											formFieldProps: {
												slotLabel: "Harness",
												slotInfo: "The framework the agent is built on. It decides whether a guardrail can refuse a tool call at all.",
												slotHelp: "Not readable from the project."
											}
										}),
										S && /* @__PURE__ */ m("label", {
											className: "flex items-center gap-2",
											"data-testid": "relay-confirmed",
											children: [/* @__PURE__ */ p("input", {
												type: "checkbox",
												checked: w,
												onChange: (e) => E(e.target.checked)
											}), /* @__PURE__ */ p(L, {
												kind: "body/regular/sm",
												children: "NeMo Relay is attached to this agent (middleware + plugin.activate() held open). Without it the victim emits no telemetry and the run cannot be scored."
											})]
										})
									]
								}),
								/* @__PURE__ */ p(x, {
									useControllerProps: {
										control: O,
										name: "egress"
									},
									formFieldProps: {
										slotLabel: "Egress Allow-list",
										slotInfo: "The sandbox blocks every other outbound connection. Auto-detected from the agent config (model endpoints, network MCP servers); add hosts only the packaged code knows about. Port 443 when none is given.",
										slotHelp: M.isPending ? "Detecting from the agent…" : "Comma-separated host[:port]."
									}
								}),
								/* @__PURE__ */ p(x, {
									useControllerProps: {
										control: O,
										name: "port"
									},
									formFieldProps: {
										slotLabel: "Victim Port",
										slotInfo: "Where the agent listens. It must answer GET /health within two minutes of starting and serve OpenAI-style POST /v1/chat/completions, which the attacks call.",
										slotHelp: M.isPending ? "Detecting from the agent…" : "Auto-detected from the deployment. Edit to override."
									}
								}),
								/* @__PURE__ */ p(x, {
									useControllerProps: {
										control: O,
										name: "secrets"
									},
									formFieldProps: {
										slotLabel: "Secret Names",
										slotInfo: "Credentials the agent reads, by name. Their values never go into the image: the sandbox gets placeholders, filled in only on requests to the endpoints they belong to.",
										slotHelp: M.isPending ? "Detecting from the agent…" : "Comma-separated; auto-detected from the agent config. Edit to override."
									}
								}),
								/* @__PURE__ */ p(x, {
									useControllerProps: {
										control: O,
										name: "env"
									},
									formFieldProps: {
										slotLabel: "Environment Variables (optional)",
										slotInfo: "Non-secret settings the agent reads, set as ENV in its image. Values here are stored in plain text, so credentials belong in Secret Names.",
										slotHelp: "Comma-separated KEY=VALUE."
									}
								}),
								/* @__PURE__ */ p(ce, { children: /* @__PURE__ */ p(g, {
									value: "models",
									title: "Models (optional)",
									children: /* @__PURE__ */ m(I, {
										gap: "density-md",
										children: [/* @__PURE__ */ p(L, {
											kind: "body/regular/sm",
											className: "text-subtle",
											children: "Defaults shown as placeholders; override any group for this target. Each run can still change them."
										}), /* @__PURE__ */ p(vs, {
											value: i,
											onChange: a,
											workspace: e,
											defaults: D
										})]
									})
								}) }),
								/* @__PURE__ */ m(F, {
									gap: "density-md",
									children: [/* @__PURE__ */ p(P, {
										color: "brand",
										type: "submit",
										disabled: fe.isPending,
										children: fe.isPending ? "Creating…" : "Create Manifest"
									}), /* @__PURE__ */ p(P, {
										asChild: !0,
										kind: "tertiary",
										children: /* @__PURE__ */ p(Me, {
											to: ta(e),
											children: "Cancel"
										})
									})]
								})
							]
						})
					})]
				})
			})]
		})
	});
};
//#endregion
//#region src/Root.tsx
function JT({ host: e }) {
	return Bi({
		getAccessToken: e.auth.getAccessToken,
		baseUrl: e.apiBaseUrl
	}), /* @__PURE__ */ p(Gi, {
		host: e,
		children: /* @__PURE__ */ m(Fe, { children: [
			/* @__PURE__ */ p(Pe, {
				path: Qi.runList,
				element: /* @__PURE__ */ p(PS, {})
			}),
			/* @__PURE__ */ p(Pe, {
				path: Qi.manifestList,
				element: /* @__PURE__ */ p(uc, {})
			}),
			/* @__PURE__ */ p(Pe, {
				path: Qi.manifestNew,
				element: /* @__PURE__ */ p(qT, {})
			}),
			/* @__PURE__ */ p(Pe, {
				path: Qi.manifestDetail,
				element: /* @__PURE__ */ p(cc, {})
			}),
			/* @__PURE__ */ p(Pe, {
				path: Qi.runDetails,
				element: /* @__PURE__ */ p(AS, {})
			})
		] })
	});
}
//#endregion
//#region src/Nav.tsx
var YT = (e) => [{
	group: "Governance",
	items: [{
		id: "agent-hardener",
		iconName: "sword",
		label: "Agent Hardening",
		href: $i(e)
	}]
}];
//#endregion
export { JT as Root, YT as navItems };
