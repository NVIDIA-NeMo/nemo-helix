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
const DEV_RUNTIME = "react/jsx-dev-runtime";
/** `specifier` is what the plugin imported, when it differs from the package (a deep import). */
const singletonBundled = (pkg, specifier = pkg) => fix(`The singleton "${pkg}" is bundled into the plugin; a second copy breaks shared context.`, `Import "${pkg}" only by a contract specifier (e.g. react-router), never a deep path (e.g. react-router/dom) and never from node_modules directly. The build helper externalizes contract specifiers for you.`, ...specifier === DEV_RUNTIME ? [`A development-mode build emits the dev JSX runtime ("${DEV_RUNTIME}"), which Studio does not provide. Build in production mode (e.g. vite build without --mode development).`] : [], "Rebuild the plugin.");
/** A deep import of a provided package (e.g. `@nemo/common/components/Foo`). */
const providedDeepImport = (specifier, pkg) => fix(`Studio does not provide "${specifier}" (it is not in the plugin contract); "${pkg}" is provided only through its bare specifier.`, `Import from the bare "${pkg}" specifier instead of "${specifier}", then rebuild.`, `If the name you need is not exported from "${pkg}", add it to web/packages/common/src/plugin.ts.`);
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
	for (let dir = path.resolve(root);; dir = path.dirname(dir)) {
		try {
			const { version } = JSON.parse(fs.readFileSync(path.join(dir, "node_modules", pkg, "package.json"), "utf8"));
			if (typeof version === "string") return version;
		} catch {}
		if (path.dirname(dir) === dir) return;
	}
};
/**
* Build plugin for Studio plugin bundles: externalizes the contract's shared
* specifiers, fails the build when a singleton would be bundled or a deep import of
* a shared package (singleton or provided) is used, and emits `plugin.manifest.json`
* next to the bundle.
*/
const studioPlugin = (options = {}) => {
	let viteRoot;
	return {
		name: "nemo-studio-plugin",
		enforce: "pre",
		configResolved(config) {
			viteRoot = config.root;
		},
		resolveId(source, importer) {
			if (source.startsWith(".") || path.isAbsolute(source)) return null;
			if (specifiers.has(source)) return {
				id: source,
				external: true
			};
			const owner = packages.find((pkg) => source.startsWith(`${pkg}/`));
			if (owner) {
				const found = singletons.has(owner) ? issue("PLUGIN_SINGLETON_BUNDLED", owner, singletonBundled(owner, source)) : issue("PLUGIN_SPECIFIER_UNMAPPED", source, providedDeepImport(source, owner));
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
			const root = options.root ?? viteRoot ?? process.cwd();
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
	};
};
//#endregion
export { studioPlugin };
