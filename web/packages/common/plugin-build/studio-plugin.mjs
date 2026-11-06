// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import fs from "node:fs";
import path from "node:path";
//#region src/pluginContract/contract.ts
const PLUGIN_CONTRACT = {
	version: "1.0.0",
	entries: [
		{
			specifier: "react",
			package: "react",
			kind: "singleton"
		},
		{
			specifier: "react/jsx-runtime",
			package: "react",
			kind: "singleton"
		},
		{
			specifier: "react-dom",
			package: "react-dom",
			kind: "singleton"
		},
		{
			specifier: "react-dom/client",
			package: "react-dom",
			kind: "singleton"
		},
		{
			specifier: "react-router",
			package: "react-router",
			kind: "singleton"
		},
		{
			specifier: "@nvidia/foundations-react-core",
			package: "@nvidia/foundations-react-core",
			kind: "singleton"
		},
		{
			specifier: "@tanstack/react-query",
			package: "@tanstack/react-query",
			kind: "singleton"
		},
		{
			specifier: "@nemo/common",
			package: "@nemo/common",
			kind: "provided"
		}
	]
};
//#endregion
//#region src/pluginContract/remediate.ts
const CONTRACT_DOCS = "plugins/example-plugin/web/AGENTS.md";
const fix = (message, ...steps) => ({
	message,
	steps: [...steps, `Contract and shared-dependency rules: ${CONTRACT_DOCS}`]
});
const specifierUnmapped = (specifier) => fix(`Studio does not provide "${specifier}" (it is not in the plugin contract). This can also mean the plugin was built with a modified or older build helper.`, `Import "${specifier}" as a normal dependency so the plugin bundles its own copy, then rebuild.`, `If it must be shared with Studio, request that "${specifier}" be added to the plugin contract.`);
const singletonBundled = (pkg) => fix(`The singleton "${pkg}" is bundled into the plugin; a second copy breaks shared context.`, `Import "${pkg}" only by a contract specifier (e.g. react-router), never a deep path (e.g. react-router/dom) and never from node_modules directly. The build helper externalizes contract specifiers for you.`, "Rebuild the plugin.");
/** Render an issue as build-error text: code, message, optional importer, numbered steps. */
const formatIssue = (issue, importer) => [
	`${issue.code}: ${issue.message}`,
	...importer ? [`Imported from: ${importer}`] : [],
	...issue.steps.map((step, i) => `${i + 1}. ${step}`)
].join("\n");
//#endregion
//#region src/pluginContract/diffContract.ts
const issue = (code, subject, fix) => ({
	code,
	subject,
	...fix
});
//#endregion
//#region src/pluginContract/studioPlugin.ts
const MANIFEST_FILE = "plugin.manifest.json";
const { entries } = PLUGIN_CONTRACT;
const specifiers = new Set(entries.map((e) => e.specifier));
const packages = [...new Set(entries.map((e) => e.package))];
const singletons = new Set(entries.filter((e) => e.kind === "singleton").map((e) => e.package));
const bundledSingletonIn = (moduleIds) => {
	for (const id of moduleIds) {
		const segments = id.replaceAll("\\", "/").split("/");
		for (let i = segments.indexOf("node_modules"); i !== -1; i = segments.indexOf("node_modules", i + 1)) {
			const name = segments[i + 1]?.startsWith("@") ? `${segments[i + 1]}/${segments[i + 2]}` : segments[i + 1];
			if (singletons.has(name)) return name;
		}
	}
};
const readVersion = (root, pkg) => {
	try {
		const manifest = JSON.parse(fs.readFileSync(path.join(root, "node_modules", pkg, "package.json"), "utf8"));
		return typeof manifest.version === "string" ? manifest.version : void 0;
	} catch {
		return;
	}
};
/**
* Build plugin for Studio plugin bundles: externalizes the contract's shared
* specifiers, fails the build when a singleton would be bundled, and emits
* `plugin.manifest.json` next to the bundle.
*/
const studioPlugin = ({ root = process.cwd() } = {}) => ({
	name: "nemo-studio-plugin",
	enforce: "pre",
	resolveId(source, importer) {
		if (source.startsWith(".") || path.isAbsolute(source)) return null;
		if (specifiers.has(source)) return {
			id: source,
			external: true
		};
		const owner = packages.find((pkg) => source.startsWith(`${pkg}/`));
		if (owner) {
			const found = singletons.has(owner) ? issue("PLUGIN_SINGLETON_BUNDLED", owner, singletonBundled(owner)) : issue("PLUGIN_SPECIFIER_UNMAPPED", source, specifierUnmapped(source));
			this.error(formatIssue(found, importer));
		}
		return null;
	},
	generateBundle(_options, bundle) {
		const chunks = Object.values(bundle).filter((output) => output.type === "chunk");
		const singleton = bundledSingletonIn(chunks.flatMap((chunk) => chunk.moduleIds));
		if (singleton) this.error(formatIssue(issue("PLUGIN_SINGLETON_BUNDLED", singleton, singletonBundled(singleton))));
		const bare = new Set(chunks.flatMap((chunk) => [...chunk.imports, ...chunk.dynamicImports]).filter((id) => !(id in bundle)));
		const builtAgainst = {};
		for (const pkg of packages) {
			const version = readVersion(root, pkg);
			if (version) builtAgainst[pkg] = version;
		}
		const manifest = {
			contractVersion: PLUGIN_CONTRACT.version,
			specifiers: [...bare].sort(),
			bundledPackages: [],
			builtAgainst
		};
		this.emitFile({
			type: "asset",
			fileName: MANIFEST_FILE,
			source: `${JSON.stringify(manifest, null, 2)}\n`
		});
	}
});
//#endregion
export { studioPlugin };
