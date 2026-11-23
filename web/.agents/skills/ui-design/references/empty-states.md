<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Studio Empty States

Studio renders every _empty_ state through **one** primitive: `EntityEmptyState`
in `@nemo/common`, driven by a central **entity registry**. Do not hand-roll
`StatusMessage`, copy `TableEmptyState`, or invent per-callsite empty markup.
Adding a new empty state means adding a registry entry and pointing a
callsite at it — nothing more. **Error states are separate**: keep routing the
error branch through the existing `ErrorPanel` (with `getErrorMessage(error)`),
which surfaces the actual failure — `EntityEmptyState` does not handle errors.

> Governing design rules: `kaizen-ui` skill →
> `references/patterns/empty-states.md`, `references/patterns/error-states.md`,
> `references/components/StatusMessage.md`. Read those before deviating.

## Prerequisite

This reference assumes the shared `EntityEmptyState` component and
`ENTITY_EMPTY_STATES` registry already exist in `@nemo/common` (delivered by
ASTD-394). If they do not yet exist, you are doing the initial build — follow
the Action Plan on the ticket, not this reference. This reference is the
go-forward guide for **every empty state after** that primitive lands.

Canonical locations:

- Component: `packages/common/src/components/EntityEmptyState/index.tsx`
- Registry: `packages/common/src/components/EntityEmptyState/registry.ts`
  (`ENTITY_EMPTY_STATES: Record<EntityKey, EmptyStateDescriptor>`)
- Coding-agent / CLI handoff: `packages/common/src/components/EntityAgentBridge/index.tsx`
- Icons: `packages/common/src/constants/entityIcons.ts` (`ENTITY_ICONS`)

## The two variants

Every empty state is exactly one of two governed variants. Never invent a
third idiom.

| Variant      | When                                                         | Required affordances                                                                                                                                                |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `first-use`  | Data source is genuinely empty; user hasn't created anything | icon, heading, subheading, primary create CTA, and the "Ask an agent · CLI" self-service snippet, labelled "Or use a coding agent or the CLI:" when the CTA renders |
| `no-results` | Items exist but current filters/search match zero            | heading naming the mismatch, **"Clear filters"** action; **no** create CTA                                                                                          |

**Compute the variant from signals — do not pick it manually.** Inside a
DataView the signals already exist: `hasFiltersApplied` / `hasSearchApplied`
(→ `no-results`); otherwise `first-use`. The error branch is handled separately
by `ErrorPanel`, not by a variant.

## Adding a new empty state

### 1. Add a registry entry

One entry per entity in `ENTITY_EMPTY_STATES`. Copy/CTA/CLI/prompt live here
only — never at the callsite.

```ts
filesets: {
  icon: FolderOpen,                                  // single size token, set by the component
  heading: 'No filesets yet',                        // sentence case, specific
  subheading: 'Filesets group the files your agents and jobs read from.',
  createAction: { label: 'Create fileset', to: ROUTES.FILESETS_NEW },
  cliCommand: 'nemo files filesets create --name <fileset-name>',
  skillPrompt: 'Help me create my first fileset with nemo-files',
},
```

Field rules:

- `icon` — a `lucide-react` icon. The component applies the one standard size
  token; do not set `size-*`, `h-[64px]`, etc. yourself.
- `heading` / `subheading` — sentence case ("No filesets yet", not
  "No Filesets Found" / "Manage Filesets"). Subheading answers "why would I add
  one?" in 1–2 sentences.
- `createAction?` — **omit** for entities with no in-app create flow (e.g.
  Agents, Members). Use `to` for route navigation; for imperative or
  modal-driven creation, omit `to` and pass `onCreate` at the callsite instead
  (`EmptyStateCreateAction` has no `onClick` field). Renders as
  `<Button color="brand">`.
- `cliCommand` — concrete, copy-pasteable, with `<placeholder>` args. Keep it
  accurate to the current CLI (see Accuracy below). Omit if the entity has no
  CLI equivalent.
- `skillPrompt` — copy-to-clipboard string that triggers the entity's skill.
  **Not wired to Copilot** (deferred by ticket decision 1) — it is stored for a
  future integration and surfaced under the "Ask an agent" snippet toggle today.
  Omit if no skill exists. Name a skill that actually ships (see
  `packages/nemo_helix_ext/src/nemo_helix_ext/skills/` and `plugins/*/skills/`);
  a prompt naming a skill the reader cannot install is worse than none.
- For a **read-only** entity (traces, spans, test-case results — anything the
  user inspects rather than creates), `cliCommand` / `skillPrompt` are still
  required where one exists. The handoff there is "examine/analyze this", not
  "create one" — that is the whole point of the bridge (ASTD-398). Only omit
  both when there is genuinely nothing to run, as with `baseModels`.

### 2. Wire the callsite

**DataView** (`StudioDataView` / any `DataView` table) — route
`renderEmptyState` through the shared component so the variant is derived:

```tsx
renderEmptyState={({ hasFiltersApplied, hasSearchApplied }) => (
  <EntityEmptyState
    entity="filesets"
    variant={hasFiltersApplied || hasSearchApplied ? 'no-results' : 'first-use'}
  />
)}
renderErrorState={() => (
  <ErrorPanel errorMessage={getErrorMessage(error ?? new Error('Failed to load'))} />
)}
```

Prefer the DataView/ScrollTable **defaults**: if the shared default already
renders `EntityEmptyState` for the given `entity`, pass only `entity` and let
the container derive the variant — don't re-specify the branching.

**Standalone (panel / non-table)** — render directly with an explicit variant:

```tsx
<EntityEmptyState entity="secrets" variant="first-use" />
```

**Chat** keeps its bespoke animated "Ready" state (a deliberate peak-end
moment). Only its "no models" branch adopts the shared create-CTA contract —
do not replace the animation.

### 3. Delete the old markup

When migrating a callsite: remove any hand-rolled `StatusMessage` and local
first-use/no-results `if` branching. Route the error branch through `ErrorPanel`
with `getErrorMessage(error)`. Do not leave the old empty-state path behind as a
fallback.

## The coding-agent bridge outside an empty state

`EntityAgentBridge` is the "Ask an agent · CLI" snippet on its own, driven by
the same registry entry. `EntityEmptyState` embeds it; render it directly only
when the surface genuinely cannot be an `EntityEmptyState`:

```tsx
<EntityAgentBridge entity="telemetryTraces" workspace={workspace} />
```

It renders `null` when the entity has neither a `cliCommand` nor a
`skillPrompt`, so mount it unconditionally rather than re-reading the registry
at the callsite to decide. Pass `workspace` wherever one is in scope — without
it the snippet hands the reader a literal `<workspace>` to fix up by hand.

### When a bespoke panel is allowed

Almost never. `AgentTraceStatistics`'s `TraceStatisticsEmptyState` is the
worked example of a legitimate exception, and it clears a high bar:

- It escalates through **three** exits (run the agent, set up tracing, widen the
  window), which overruns the two-action footer rule.
- Its "widen the window" exit is **not** `no-results`: the panel never learns
  whether traces exist outside the range, only that none landed inside it, so
  it must keep offering the create-equivalent action on both branches —
  something `no-results` forbids by design.

A panel that clears that bar still renders `EntityAgentBridge` rather than
hand-rolling a snippet, so the registry stays the only source of the commands.
Wanting different heading copy is **not** a reason to go bespoke — fix the
registry entry instead.

## Not an empty state

Do not route these through `EntityEmptyState`; they are separate idioms and
forcing them onto the primitive replaces accurate copy with generic copy:

| Situation                                                 | Use                                        |
| --------------------------------------------------------- | ------------------------------------------ |
| The request failed                                        | `ErrorPanel` with `getErrorMessage(error)` |
| The job has not finished yet ("…once the job completes")  | a pending/`Spinner` state                  |
| Nothing is selected yet ("Awaiting model selection")      | an instructional state                     |
| A precondition blocks the view ("File too large to edit") | an instructional state                     |
| A preview of a file's rows, or a chart with no series     | leave local                                |

The tell: an empty state answers _"there are none of these — here is how to get
one."_ If the honest answer is instead "not yet", "it broke", or "pick something
first", it is one of the rows above.

## Copy & CLI accuracy

- Headings are sentence case and name the entity. CTAs are verb + noun
  ("Create fileset", not "Get started").
- At most **2 buttons** in the footer (Kaizen empty-state rule). The CLI command
  and agent prompt live **below** the footer in a single KUI `CodeSnippet`
  (with its built-in copy button); a tiny `SegmentedControl` in the snippet's
  `slotActions` toggles between **Ask an agent** and **CLI**. This is not a
  footer button and does not count against the 2-action limit.
- CLI commands must match the shipping `nemo` CLI. Verify against the relevant
  plugin skill (`nemo files`, `nemo models`, `nemo guardrail`, `nemo secrets`,
  …) before committing. Because commands are centralized in the registry, a CLI
  change is a single-file fix.

## Verify

- `pnpm --filter @nemo/common test` — the `EntityEmptyState` unit tests cover
  the `first-use` and `no-results` variants, and `EntityAgentBridge`'s cover the
  CLI/prompt toggle, `<workspace>` resolution, and the render-nothing case.
  Add a case if you introduced a new descriptor shape or affordance, not for a
  plain new entry.
- `pnpm --filter nemo-studio-ui test` for migrated studio callsites.
- Storybook: check `first-use` and `no-results` render for a representative
  entity.

## Do / Don't

- **Do** add a registry entry + wire the callsite. **Don't** create a new
  `*EmptyState.tsx` component per entity.
- **Do** derive the variant from DataView signals. **Don't** branch on ad-hoc
  booleans or show a `first-use` state while filters are active.
- **Do** omit `createAction` / `cliCommand` / `skillPrompt` when the entity has
  none. **Don't** fabricate a CLI command or a create route.
- **Don't** re-introduce `TableEmptyState` or studio's `Empty` (both temporary
  migration shims, being removed) or raw `StatusMessage` for empty/error states.
- **Do** give a read-only entity an "examine/analyze this" bridge. **Don't**
  leave `cliCommand` and `skillPrompt` empty just because the entity has no
  create flow.
