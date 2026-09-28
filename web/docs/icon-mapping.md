<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Lucide → NVIDIA GUI Icons mapping

Reference for migrating Studio off `lucide-react` and onto
[`@nvidia/gui-icons`](https://github.com/NVIDIA/icons) (ASTD-664). This is the
mapping half of that work (ASTD-665); it changes no code.

## Method

- **Lucide inventory** — every named import from `'lucide-react'` under
  `web/packages/**` and `plugins/*/web/**` (excluding `node_modules`, `dist`,
  `generated`). 194 unique symbols across ~309 files. **Usage** below is the
  number of files importing the icon, not render sites.
- **NVIDIA inventory** — the 748 names in `IconNames` from
  `packages/gui-icons/src/IconName.ts`, plus `IconTagMap` keywords. Every icon
  ships `line` and `fill` variants; `line` is assumed throughout.
- **Confidence**
  - **exact** — same concept, near-identical glyph.
  - **close** — same concept, noticeably different glyph.
  - **approx** — semantic substitute; NVIDIA has no glyph for this concept.
  - **none** — no defensible substitute. Fallback recommended in _Notes_.

Matches were made from names and tag metadata, not side-by-side rendering.
Anything marked **close** or **approx** should get a visual check during
implementation.

## Mapping

| Lucide                  | NVIDIA GUI icon             | Confidence | Usage | Notes                                                                                                                          |
| ----------------------- | --------------------------- | ---------- | ----- | ------------------------------------------------------------------------------------------------------------------------------ |
| `Activity`              | `pulse`                     | close      | 1     | `ekg` alt.                                                                                                                     |
| `AlertTriangle`         | `warning`                   | exact      | 3     | Lucide alias of `TriangleAlert`.                                                                                               |
| `AlignLeft`             | `text-align-left`           | exact      | 1     |                                                                                                                                |
| `AppWindow`             | `window`                    | exact      | 1     |                                                                                                                                |
| `ArrowDown`             | `arrow-down`                | exact      | 4     |                                                                                                                                |
| `ArrowLeft`             | `arrow-left`                | exact      | 2     |                                                                                                                                |
| `ArrowRight`            | `arrow-right`               | exact      | 7     |                                                                                                                                |
| `ArrowUp`               | `arrow-up`                  | exact      | 8     |                                                                                                                                |
| `ArrowUpDown`           | `arrow-up-down`             | exact      | 2     | Use `sort` where it means "sortable column".                                                                                   |
| `BadgeCheck`            | `checkmark-badge`           | exact      | 2     |                                                                                                                                |
| `Ban`                   | `cancel`                    | exact      | 5     |                                                                                                                                |
| `BarChart3`             | `chart-bar`                 | exact      | 1     | Lucide alias of `ChartBar`.                                                                                                    |
| `Beaker`                | `beaker`                    | exact      | 1     |                                                                                                                                |
| `Binary`                | `numbers`                   | approx     | 1     | No 0/1 glyph; `data` (bits) alt.                                                                                               |
| `Book`                  | `book`                      | exact      | 1     |                                                                                                                                |
| `BookOpen`              | `book`                      | close      | 2     | No open-book glyph.                                                                                                            |
| `Bot`                   | `robot`                     | exact      | 6     | `chatbot` alt for conversational surfaces.                                                                                     |
| `Box`                   | `cube`                      | exact      | 3     |                                                                                                                                |
| `Boxes`                 | `cube-stack`                | exact      | 4     | Entity icon: `baseModels`.                                                                                                     |
| `Braces`                | `code`                      | approx     | 4     | No `{}` glyph.                                                                                                                 |
| `Brain`                 | `neural-network`            | close      | 1     |                                                                                                                                |
| `Bug`                   | `bug`                       | exact      | 1     |                                                                                                                                |
| `CalendarDays`          | `calendar`                  | exact      | 1     |                                                                                                                                |
| `ChartBar`              | `chart-bar`                 | exact      | 3     | Entity icon: evaluation results/sessions.                                                                                      |
| `ChartBarBig`           | `chart-bar`                 | close      | 1     |                                                                                                                                |
| `ChartLine`             | `chart`                     | exact      | 3     | NVIDIA `chart` is the time-series line chart.                                                                                  |
| `ChartNetwork`          | `chart-flow`                | close      | 1     | `graph-node-connect` alt.                                                                                                      |
| `ChartNoAxesCombined`   | `chart-performance`         | approx     | 1     |                                                                                                                                |
| `ChartScatter`          | `chart-scatterplot`         | exact      | 1     |                                                                                                                                |
| `ChartSpline`           | `chart`                     | close      | 1     |                                                                                                                                |
| `Check`                 | `check`                     | exact      | 4     | Also in micro set.                                                                                                             |
| `CheckCircle2`          | `check-circle`              | exact      | 3     | Lucide alias of `CircleCheck`.                                                                                                 |
| `CheckSquare`           | `check-circle`              | approx     | 1     | No checkbox glyph; prefer a KUI Checkbox if it is a control.                                                                   |
| `ChevronDown`           | `chevron-down`              | exact      | 9     | Also in micro set.                                                                                                             |
| `ChevronLeft`           | `chevron-left`              | exact      | 4     | Also in micro set.                                                                                                             |
| `ChevronRight`          | `chevron-right`             | exact      | 12    | Also in micro set.                                                                                                             |
| `ChevronUp`             | `chevron-up`                | exact      | 2     | Also in micro set.                                                                                                             |
| `ChevronsDown`          | `chevron-double-down`       | exact      | 1     |                                                                                                                                |
| `ChevronsDownUp`        | `collapse`                  | close      | 2     | Inward-pointing chevrons.                                                                                                      |
| `ChevronsUp`            | `chevron-double-up`         | exact      | 1     |                                                                                                                                |
| `ChevronsUpDown`        | `chevron-up-down`           | exact      | 2     |                                                                                                                                |
| `Circle`                | `shape-circle`              | exact      | 1     | Call site has a TODO to replace it anyway.                                                                                     |
| `CircleAlert`           | `error`                     | exact      | 13    |                                                                                                                                |
| `CircleCheck`           | `check-circle`              | exact      | 11    |                                                                                                                                |
| `CircleCheckBig`        | `check-circle`              | close      | 1     |                                                                                                                                |
| `CircleHelp`            | `help-circle`               | exact      | 9     |                                                                                                                                |
| `CirclePlay`            | `play`                      | close      | 1     | No circled variant.                                                                                                            |
| `CircleStop`            | `stop`                      | close      | 1     | No circled variant.                                                                                                            |
| `CircleX`               | `close-circle`              | exact      | 6     |                                                                                                                                |
| `ClipboardCheck`        | `clipboard`                 | close      | 1     | `list-checkmark` alt if it means "checklist".                                                                                  |
| `ClipboardList`         | `clipboard`                 | close      | 6     | `list-checkmark` alt.                                                                                                          |
| `Clock`                 | `clock`                     | exact      | 1     |                                                                                                                                |
| `Cloud`                 | `cloud`                     | exact      | 1     |                                                                                                                                |
| `Code2`                 | `code`                      | exact      | 1     | Lucide alias of `CodeXml`.                                                                                                     |
| `CodeXml`               | `code`                      | exact      | 1     |                                                                                                                                |
| `Cog`                   | `cog`                       | exact      | 4     |                                                                                                                                |
| `Columns3`              | `layout-columns`            | exact      | 4     |                                                                                                                                |
| `Command`               | `keyboard`                  | approx     | 1     | No ⌘ glyph; render the literal `⌘` if it is a shortcut hint.                                                                   |
| `Copy`                  | `copy-generic`              | exact      | 11    | `copy-doc` alt.                                                                                                                |
| `Cpu`                   | `cpu`                       | exact      | 2     |                                                                                                                                |
| `Database`              | `db`                        | exact      | 10    | Entity icon: `datasets`.                                                                                                       |
| `DatabaseCheck`         | `db`                        | approx     | 1     | Entity icon: `safeSynthesizerJobs`. No db-with-check; collides with `datasets` unless a different glyph is chosen (`secure`?). |
| `Dices`                 | `die-5`                     | close      | 1     | Single die; `die-1`…`die-6` available. `shuffle` alt.                                                                          |
| `Dot`                   | `shape-circle` (fill)       | close      | 2     | Use the `fill` variant at a small size.                                                                                        |
| `Download`              | `download`                  | exact      | 10    |                                                                                                                                |
| `EllipsisVertical`      | `more-vert`                 | exact      | 7     |                                                                                                                                |
| `Equal`                 | `equal`                     | exact      | 2     |                                                                                                                                |
| `ExternalLink`          | `open-external`             | exact      | 2     | Also in micro set.                                                                                                             |
| `Eye`                   | `eye`                       | exact      | 8     |                                                                                                                                |
| `EyeOff`                | `eye-off`                   | exact      | 2     |                                                                                                                                |
| `File`                  | `document`                  | exact      | 9     |                                                                                                                                |
| `FileCheck`             | `document-checkmark`        | exact      | 1     |                                                                                                                                |
| `FileJson`              | `code`                      | approx     | 3     | `document` alt if the file-ness matters more than JSON-ness.                                                                   |
| `FilePenLine`           | `rename`                    | close      | 2     |                                                                                                                                |
| `FilePlus`              | `document-new`              | exact      | 1     |                                                                                                                                |
| `FilePlus2`             | `document-new`              | exact      | 1     |                                                                                                                                |
| `FileSpreadsheet`       | `document`                  | approx     | 1     | No table/spreadsheet glyph; `layout-grid` alt.                                                                                 |
| `FileStack`             | `collection`                | close      | 1     | Entity icon: `filesets`. `library` / `copy-doc` alts.                                                                          |
| `FileText`              | `document`                  | close      | 2     | `text` alt.                                                                                                                    |
| `FileX`                 | `document`                  | approx     | 1     | No doc-with-x; pair with `close` or use `trash`.                                                                               |
| `Filter`                | `filter`                    | exact      | 10    |                                                                                                                                |
| `FlaskConical`          | `beaker`                    | close      | 7     | Entity icon: `experiments`.                                                                                                    |
| `FolderClosed`          | `folder-closed`             | exact      | 3     |                                                                                                                                |
| `FolderMinus`           | `folder-closed`             | approx     | 1     | No folder-minus.                                                                                                               |
| `FolderOpen`            | `folder-open`               | exact      | 11    | Entity icon: `filesetFiles`.                                                                                                   |
| `FolderPlus`            | `folder-closed`             | approx     | 1     | No folder-plus.                                                                                                                |
| `FolderTree`            | `chart-hierarchy`           | approx     | 1     |                                                                                                                                |
| `Form`                  | `layout-list`               | approx     | 1     | Entity icon: `dataDesignerJobs`. No form/input glyph.                                                                          |
| `Gauge`                 | `performance-medium`        | exact      | 3     | `performance-high` / `-low` for stateful gauges.                                                                               |
| `GitBranch`             | `scm-branch`                | exact      | 3     |                                                                                                                                |
| `GitCommitHorizontal`   | `commit`                    | exact      | 1     |                                                                                                                                |
| `GitCompare`            | `scm-compare`               | exact      | 1     | Entity icon: `evalComparison`.                                                                                                 |
| `Globe`                 | `world`                     | exact      | 2     |                                                                                                                                |
| `Gpu`                   | `gpu`                       | exact      | 1     |                                                                                                                                |
| `GraduationCap`         | `graduate`                  | exact      | 1     |                                                                                                                                |
| `GripVertical`          | `grip-edge-vertical`        | close      | 2     | `grip-mini` alt.                                                                                                               |
| `Hammer`                | `wrench`                    | approx     | 1     | No hammer; `toolbox` alt.                                                                                                      |
| `Hash`                  | `numbers`                   | approx     | 2     | No `#` glyph.                                                                                                                  |
| `HelpCircle`            | `help-circle`               | exact      | 2     | Lucide alias of `CircleHelp`.                                                                                                  |
| `Image`                 | `image`                     | exact      | 2     |                                                                                                                                |
| `ImagePlus`             | `import-image`              | close      | 2     | `camera-add` alt.                                                                                                              |
| `Info`                  | `info-circle`               | exact      | 16    | Also in micro set.                                                                                                             |
| `KeyRound`              | `key`                       | exact      | 2     |                                                                                                                                |
| `Layers`                | `layers`                    | exact      | 3     |                                                                                                                                |
| `LayoutDashboard`       | `apps`                      | close      | 1     | `window-grid` alt.                                                                                                             |
| `LayoutGrid`            | `layout-grid`               | exact      | 4     |                                                                                                                                |
| `LayoutList`            | `layout-list`               | exact      | 2     |                                                                                                                                |
| `LibraryBig`            | `library`                   | exact      | 1     |                                                                                                                                |
| `Lightbulb`             | `lightbulb`                 | exact      | 2     | Entity icon: `optimizerInsights`.                                                                                              |
| `Link2`                 | `link`                      | exact      | 1     |                                                                                                                                |
| `ListChecks`            | `list-checkmark`            | exact      | 3     | Entity icon: `jobs`.                                                                                                           |
| `ListTree`              | `chart-hierarchy`           | approx     | 5     | Entity icon: traces. `list-bullet` alt.                                                                                        |
| `Loader2`               | `circle-tick`               | close      | 4     | Spinner. Prefer a KUI spinner over an animated icon.                                                                           |
| `LoaderCircle`          | `circle-tick`               | close      | 2     | Same as above.                                                                                                                 |
| `Lock`                  | `lock-closed`               | exact      | 4     |                                                                                                                                |
| `LockKeyhole`           | `lock-closed`               | exact      | 1     | Entity icon: `secrets`.                                                                                                        |
| `LockOpen`              | `lock-open`                 | exact      | 1     |                                                                                                                                |
| `Logs`                  | `list-bullet`               | approx     | 1     | Entity icon: spans / monitor runs.                                                                                             |
| `LucideIcon` (type)     | `IconName` / component type | n/a        | 11    | Type-only import. See _Type and dynamic usage_.                                                                                |
| `MailWarning`           | `envelope`                  | approx     | 1     | No envelope-with-warning.                                                                                                      |
| `Maximize2`             | `expand`                    | exact      | 3     | `fullscreen` alt.                                                                                                              |
| `MessageSquare`         | `chat-single`               | exact      | 4     |                                                                                                                                |
| `MessageSquarePlus`     | `chat-new`                  | exact      | 2     |                                                                                                                                |
| `MessageSquareShare`    | `share`                     | approx     | 1     |                                                                                                                                |
| `MessageSquareText`     | `chat-message`              | exact      | 1     |                                                                                                                                |
| `MessagesSquare`        | `chat-multi`                | exact      | 4     |                                                                                                                                |
| `Metronome`             | `faders`                    | approx     | 1     | Entity icon: `customModels`. No metronome; `cube` alt to stay in the model family.                                             |
| `Minimize2`             | `collapse`                  | exact      | 1     | `fullscreen-exit` alt.                                                                                                         |
| `Minus`                 | `subtract`                  | exact      | 2     | Also in micro set.                                                                                                             |
| `Moon`                  | `moon`                      | exact      | 1     |                                                                                                                                |
| `Network`               | `network`                   | exact      | 1     |                                                                                                                                |
| `NotebookPen`           | `text`                      | approx     | 1     | NVIDIA `notes` is _musical_ notation — do not use.                                                                             |
| `Package`               | `package`                   | exact      | 1     |                                                                                                                                |
| `Palette`               | `palette`                   | exact      | 1     |                                                                                                                                |
| `PanelLeftClose`        | `chevron-double-left`       | approx     | 1     | No panel glyphs; `layout-detail` alt.                                                                                          |
| `PanelLeftOpen`         | `chevron-double-right`      | approx     | 1     |                                                                                                                                |
| `PanelRightClose`       | `chevron-double-right`      | approx     | 2     |                                                                                                                                |
| `PanelRightOpen`        | `chevron-double-left`       | approx     | 1     |                                                                                                                                |
| `Pause`                 | `pause`                     | exact      | 1     |                                                                                                                                |
| `Pencil`                | `pencil`                    | exact      | 8     |                                                                                                                                |
| `Pin`                   | `pin`                       | exact      | 1     |                                                                                                                                |
| `Play`                  | `play`                      | exact      | 3     |                                                                                                                                |
| `PlugZap`               | `plug-usb`                  | approx     | 1     | `connection` alt.                                                                                                              |
| `Plus`                  | `add`                       | exact      | 22    | Also in micro set.                                                                                                             |
| `Radar`                 | `radar`                     | exact      | 1     | Entity icon: `inferenceProviders`.                                                                                             |
| `RefreshCw`             | `refresh`                   | exact      | 9     |                                                                                                                                |
| `Repeat2`               | `loop`                      | exact      | 1     | `iterate` alt.                                                                                                                 |
| `Rocket`                | `rocket`                    | exact      | 2     | Entity icon: `deployments`.                                                                                                    |
| `RotateCcw`             | `reset`                     | close      | 5     | `undo` / `history` depending on intent.                                                                                        |
| `Route`                 | `route`                     | exact      | 2     |                                                                                                                                |
| `Rows3`                 | `layout-rows`               | exact      | 2     |                                                                                                                                |
| `Save`                  | `floppy`                    | exact      | 2     |                                                                                                                                |
| `Scale`                 | `scale-balance`             | exact      | 4     | NVIDIA `scale` is resize, not a balance.                                                                                       |
| `ScrollText`            | `document`                  | approx     | 4     |                                                                                                                                |
| `Search`                | `magnifying-glass`          | exact      | 7     | Also in micro set.                                                                                                             |
| `SearchCheck`           | `magnifying-glass`          | approx     | 1     |                                                                                                                                |
| `SearchCode`            | `document-preview`          | approx     | 1     | `window-code` alt.                                                                                                             |
| `Send`                  | `paperplane`                | exact      | 4     |                                                                                                                                |
| `Server`                | `datacenter`                | close      | 1     | `workstation-system` alt.                                                                                                      |
| `Settings`              | `cog`                       | exact      | 3     |                                                                                                                                |
| `ShieldCheck`           | `secure`                    | close      | 7     | Entity icon: `guardrails`. `shield` alt.                                                                                       |
| `Sliders`               | `faders`                    | close      | 3     | Vertical sliders. Verify orientation visually.                                                                                 |
| `SlidersHorizontal`     | `sliders`                   | close      | 3     | Entity icon: `agentOptimizations`. Verify orientation.                                                                         |
| `Sparkles`              | `sparkle`                   | exact      | 10    | `generate` alt for GenAI actions.                                                                                              |
| `SplinePointer`         | `line-segment`              | approx     | 1     | `keyframe` / `freehand` alts.                                                                                                  |
| `Split`                 | `split`                     | exact      | 2     |                                                                                                                                |
| `Square`                | `shape-square`              | exact      | 2     | Also in micro set.                                                                                                             |
| `SquareArrowOutUpRight` | `open-external`             | exact      | 1     |                                                                                                                                |
| `SquareFunction`        | `function`                  | exact      | 2     |                                                                                                                                |
| `Star`                  | `star`                      | exact      | 3     | Also in micro set.                                                                                                             |
| `Sun`                   | `sun-high`                  | exact      | 1     |                                                                                                                                |
| `Terminal`              | `window-terminal`           | exact      | 5     |                                                                                                                                |
| `ThumbsDown`            | `thumb-down`                | exact      | 2     |                                                                                                                                |
| `ThumbsUp`              | `thumb-up`                  | exact      | 2     |                                                                                                                                |
| `Timer`                 | `timer`                     | exact      | 2     |                                                                                                                                |
| `ToggleLeft`            | —                           | none       | 1     | Prefer a KUI Switch; `transfer-horizontal` if a glyph is unavoidable.                                                          |
| `Toolbox`               | `toolbox`                   | exact      | 1     |                                                                                                                                |
| `Trash`                 | `trash`                     | exact      | 25    |                                                                                                                                |
| `Trash2`                | `trash`                     | exact      | 15    | Collapses onto the same glyph as `Trash`.                                                                                      |
| `Triangle`              | `shape-triangle`            | exact      | 2     | Also in micro set.                                                                                                             |
| `TriangleAlert`         | `warning`                   | exact      | 6     |                                                                                                                                |
| `Unplug`                | `link-break`                | approx     | 1     | `plug-usb` alt.                                                                                                                |
| `Upload`                | `upload`                    | exact      | 3     |                                                                                                                                |
| `User`                  | `profile`                   | exact      | 1     |                                                                                                                                |
| `UserPen`               | `profile`                   | approx     | 1     | Entity icon: `anonymizerJobs`. No user-edit glyph; `rename` alt.                                                               |
| `UsersRound`            | `profile-group`             | exact      | 1     | Entity icon: `members`.                                                                                                        |
| `Wand2`                 | `wand`                      | exact      | 3     |                                                                                                                                |
| `Waypoints`             | `route`                     | close      | 1     | Entity icon: `virtualModels`. Collides with `Route` — `graph-node-connect` or `chart-flow` as alternates.                      |
| `Workflow`              | `chart-flow`                | close      | 2     | `graph-node-connect` alt.                                                                                                      |
| `WrapText`              | `return`                    | approx     | 1     |                                                                                                                                |
| `Wrench`                | `wrench`                    | exact      | 3     |                                                                                                                                |
| `X`                     | `close`                     | exact      | 22    | Also in micro set.                                                                                                             |
| `XCircle`               | `close-circle`              | exact      | 2     |                                                                                                                                |
| `icons` (record)        | `IconNames`                 | n/a        | 1     | Dynamic lookup. See _Type and dynamic usage_.                                                                                  |

## Summary

| Confidence           | Count |
| -------------------- | ----- |
| exact                | 128   |
| close                | 30    |
| approx               | 33    |
| none                 | 1     |
| n/a (type / dynamic) | 2     |

### Glyph collisions to resolve

`ENTITY_ICONS` (`web/packages/common/src/constants/entityIcons.ts`) requires
one glyph per unrelated entity. These mappings would merge previously distinct
glyphs and need a design call:

| Entities                              | Lucide today                      | NVIDIA collision                        |
| ------------------------------------- | --------------------------------- | --------------------------------------- |
| `datasets` / `safeSynthesizerJobs`    | `Database` / `DatabaseCheck`      | both → `db`                             |
| `virtualModels` / route UI            | `Waypoints` / `Route`             | both → `route`                          |
| `agentOptimizations` / `customModels` | `SlidersHorizontal` / `Metronome` | `sliders` vs `faders` — same family     |
| `members` / `anonymizerJobs`          | `UsersRound` / `UserPen`          | `profile-group` vs `profile` — adjacent |
| `telemetrySpans` / `agentMonitorRuns` | `Logs` (already shared)           | `list-bullet` — fine, intentional       |

Non-entity collapses that are probably fine: `Trash`+`Trash2` → `trash`,
`Code2`+`CodeXml`+`Braces`+`FileJson` → `code`, all `Chevron*` and `Circle*`
pairs, `Panel*` → double chevrons.

### Concepts with no NVIDIA glyph

`Braces`, `Hash`, `Binary`, `Command`, `ToggleLeft`, `Metronome`,
`FileSpreadsheet`, `FileX`, `FolderPlus`/`FolderMinus`, `Panel*`, `UserPen`,
`DatabaseCheck`, `Form`, `NotebookPen`, `MailWarning`. Options for the
implementation ticket: accept the approximations above, keep Lucide for that
handful, or request additions upstream in `NVIDIA/icons`.

## Type and dynamic usage

- `LucideIcon` is imported as a type in 11 files (e.g. `ENTITY_ICONS`,
  `AddColumnPalette/types.ts`, `CreateFilesetStart/types.ts`). The NVIDIA
  equivalent is either `IconName` (a string union) or the component type
  exported by `@nvidia/react-gui-icons`. Which one depends on whether the
  migration standardizes on `<NvidiaGuiIcon iconName="…" />` or on named
  components (`import { Gpu } from '@nvidia/react-gui-icons/icons'`).
- `web/packages/studio/src/plugins/iconMap.ts` resolves plugin `navItems`
  `iconName` strings (kebab-case) against the Lucide `icons` record. Plugins
  currently declare `flask-conical`, `key-round`, `building-2`, `table`, and
  `swords`; only `flask-conical` → `beaker` and `key-round` → `key` have
  matches. `building-2` → `business`, `table` → `layout-grid`, `swords` →
  `sword` are approximations. Because `iconName` is part of the plugin
  contract (`plugins/example-plugin/web/AGENTS.md`), switching the lookup
  to `IconNames` is a breaking change for plugin authors and should be
  flagged in ASTD-664.

## Micro GUI Icons

`@nvidia/micro-gui-icons` ships 24 utility glyphs tuned for small sizes:
`add`, `arrow-*` (8), `check`, `chevron-*` (4), `close`, `info-circle`,
`magnifying-glass`, `open-external`, `shape-circle`, `shape-square`,
`shape-triangle`, `star`, `subtract`, `sync`. Rows marked "Also in micro set"
above could use these for inline/table-cell contexts.
