# Skill entities + Harbor skill evals: hackathon plan

Status: plan v3, after two adversarial review rounds (gpt-6.2-sol, high). Nothing is implemented.

## Goal

Make agent skills a first-class, versioned platform entity, and evaluate them with Harbor-based
agent evals. Target matrix: no skill / skill V1 / skill V2 x harness x model x thinking level.

## Fixed decisions

- Skills live in **nemo-agents**. They are not eval-specific, and registered agents will reference
  them later.
- Follow the task/taskset revision design for consistency.
- Integrity: hash the content, store the hash, error on mismatch. No signatures.
- Do not assume GitHub-backed filesets or a local repo checkout.
- A skill revision links to a **digest-pinned** taskset revision.
- One eval run uses one skill configuration. A/B comparison happens above the eval (Experiments, reporting).
- No scores on the skill entity. Provenance goes on eval results and Intake.
- No optimizer hook.
- Shared code: only the content-hash helper and the revision ref syntax move to
  `nemo_helix_plugin`. The rest of `nemo_evals/revisions.py` stays in nemo-evals.
- The skill archive module lives in `nemo_helix_plugin` (nemo-agents publishes with it, the
  nemo-evals worker extracts with it). The SDK depends on neither package, so it does not use the
  archive module: it injects plain directories and only needs a mode-aware directory digest for its
  cache key.

## Phase 0: prove injection works

- Pin one **tested combination**: harness, CLI version (via `agent_kwargs.version`), execution
  mode, model route, effort, backend. Claude Code or Codex.
- Fixtures:
  - A test skill with V1 and V2 under the **same skill name**, one root `SKILL.md` each, where each
    version produces a different observable output.
  - One tiny Harbor task with a deterministic verifier. Fixtures live outside the skill.
  - A controlled task image with no `~/.claude/skills` and no `$HOME/.agents/skills`.
- SDK (`packages/nhx_evals_sdk/.../agent_eval/runtimes/harbor/runtime.py`):
  - `HarborRuntimeConfig.skills: list[Path]`: existing absolute directories only (Harbor treats a
    relative `org/name` as a git source).
  - Pass as `AgentConfig(skills=...)` alongside the existing `agent_options`.
  - Reject duplicate skill names after staging (Harbor expands collections and is last-wins).
  - Put skill names and a mode-aware tree digest into the field `_cache_is_stale` compares
    (~line 901). Keep raw paths out of the options hash.
- Done when:
  - All three runs complete with no producer errors.
  - The harness's discovered skills match the run: nothing for the no-skill run, the right revision's
    bytes for V1 and V2.
  - Each run produces its expected output.
  - A same-job-name re-run hits the cache; a byte change and a `chmod +x` change both miss.

## Phase 1: skill entity in nemo-agents

- **Shared code** (`nemo_helix_plugin`): content-hash helper and the revision ref syntax (`#tag` /
  `#<digest>` grammar, default `latest`, tag validation, fragment length). nemo-evals keeps its
  re-exports; stored task/taskset hashes must stay byte-for-byte identical. Align the tag length
  limit (publish accepts 256 chars; the taskset GET rejects above 255).
- **Archive module** (`nemo_helix_plugin`), one reader used at publish and on the worker:
  - Deterministic archive, one root, regular files and directories only (no symlinks), unique
    normalised paths, file modes normalised and included in the digest.
  - Limits on download size, decompressed size, entry count, total payload.
  - Exclusive writes into fresh private staging.
  - `evals/` excluded from the canonical capture.
- **Storage**: a service-owned fileset, `<skill>/<archive-sha256>/skill_archive`. Read back and
  verify any existing object before reuse. Retain archives used by accepted runs and their retries.
  Define whether skill readers can read the service-owned copy.
- **Entities**:
  - `SkillEntity` head holds pointers only (`latest_revision`, tags). Current content is served from
    the verified latest revision.
  - `SkillRevisionEntity`: `{archive_ref, files_hash, skill_name}`, frontmatter parsed from the
    archived bytes, optional digest-pinned `taskset`. `content_hash` covers the whole payload.
    Frontmatter tags stay separate from revision tags.
- **Publish**: authorize the caller's read of the source → validate tags → build, upload and verify
  the archive → create `rev.<n>` → move tags. On an identical existing revision, verify it and only
  finish moving pointers. Each step is recoverable within the same request. Create publishes
  revision 1 immediately. Do not copy the nemo-evals retry path (see "Separate issue").
- **API and CLI**: create, publish, revision lookup, tags. `nemo agents skills create --from-dir`
  is a convenience over uploading an archive or pointing at an existing fileset.

## Phase 2: platform wiring

- **Managed skills** on `HarborRunnerTarget`: `skills: list[SkillRef] | None`. Unset = the run does
  not manage skills (today's behaviour); a list (possibly empty) = exactly these skills. When set,
  reject `agent_kwargs.skills_dir` and any task `environment.skills_dir`, and accept only tested
  combinations.
- **Submission**: resolve refs in the shared `AgentEvalJob.to_spec` with a real read check against
  Agents and Files (the current `source_authorization` helper does not cover them). Freeze the
  revision digest and `{archive_ref, files_hash, skill_name}` into the internal spec.
  `_harbor_target` carries refs and rejects local skill paths.
- **Idempotency**: strip `skills` from `request_digest` when unset (the `closed_book` precedent) so
  existing replays keep their hash. Hash `[]` and non-empty lists. If revoked read access should
  block replays, check it before replay using the accepted pin.
- **Worker**: download only the frozen descriptor, verify `files_hash`, extract with the shared
  reader into a fresh per-job directory outside the result tree (the shared runner deletes the
  bundle tree after building its config, `jobs/agent_evaluate.py` ~703).
- **Provenance**: the run records which skills it ran with. The frozen spec holds
  `list[ResolvedSkill]` (pinned ref + archive descriptor); project that into the result entity and
  DTO and the run-owned Intake Evaluation (set at create, preserved through the final metadata
  patch), from the accepted plan rather than the first child's bundle (a run where every child fails
  writes none). Harness, model and settings are already recorded by the existing target provenance
  (`RunnerInfo`); do not add a parallel record.
- **Matrix**: lives entirely outside the eval, as a script that submits independent runs (distinct
  idempotency keys, the same pinned taskset). The eval has no notion of arms or comparisons.

## Phase 3: linked taskset

- `taskset` on publish must be a fully qualified `workspace/taskset#<digest>`, verified through the
  caller-authorized taskset revision GET. Tags are not accepted (that response has no digest).
- The link is optional; publishing an unlinked skill does not need the evals service.
- Stretch: generate a taskset from `evals/evals.json` (separate input, since `evals/` is excluded
  from the archive), with verifiers that grade task behaviour. Distractor and negative tasks.

## Out of the first matrix

- Fabric/DeepAgents and registered agents, enforced by validation. They need skills relocated into
  the agent workspace.
- `SkillUsedMetric` scoring.
- Further harness/model combinations; each needs a compatibility entry and a discovery smoke test.

## Separate issue (not today)

Possible existing race in the nemo-evals publish retry (`plugins/nemo-evals/src/nemo_evals/revisions.py`
~484): with three concurrent publishers, a task head could end up holding an older revision's
content. Plausible from reading the code; not reproduced.

## Contracts (wave 0, done)

Shared types every work unit builds against. Uncommitted in this worktree.

- `nemo_helix_plugin/content_hash.py`: moved from nemo-evals unchanged (evals re-exports it).
- `nemo_helix_plugin/revision_refs.py`: `#tag` / `#<digest>` grammar, `LATEST_TAG`,
  `SUBENTITY_REF_PATTERN`, `PINNED_SUBENTITY_REF_PATTERN`, `parse_subentity_ref`, `is_digest`,
  `validate_tag_name`, `validate_movable_tag`. nemo-evals `api/fields.py` and `revisions.py` import
  and re-export them. 199 evals ref/revision/hash tests pass unchanged. (Unit A is therefore done,
  except the 255/256 tag-length alignment.)
- `nemo_helix_plugin/agents/skills.py`: `SKILL_NAME_PATTERN`, `SkillRef`, `PinnedSkillRef`,
  `SkillArchiveDescriptor {archive_ref, files_hash, skill_name}`, `SkillFrontmatter`,
  `SkillSource {fileset}`, `SkillInput {source, taskset, tags}`, `Skill` DTO (carries
  `content_hash`), `ResolvedSkill {ref, archive}`, list/tag query params.
- `nemo_helix_plugin/agents/endpoints.py` + `client.py`: `create_skill` (POST `/skills/{name}`),
  `publish_skill` (PUT), `get_skill`, `list_skills`, `list_skill_revisions`, `get_skill_revision`,
  `tag_skill_revision` (PUT `/skills/{name}/tags/{tag}?revision=`), `delete_skill`. Routes mirror the
  evals taskset routes under `/apis/agents/v2/workspaces/{workspace}`.
- `nemo_helix_plugin/skill_archive.py`: format contract in the module docstring, `SkillArchiveLimits`,
  `SkillArchiveInfo`, error types, and `parse_skill_frontmatter`, `build_skill_archive`,
  `verify_skill_archive`, `extract_skill_archive` (bodies raise `NotImplementedError`; unit B).
- `nhx_evals_sdk.agent_eval.results`: `InjectedSkill {name, digest}` and
  `RunMetadata.skills: list[InjectedSkill] | None` (None = run did not manage skills, `[]` = none
  injected and none allowed in).

Field contracts not yet in code (owned by their units):

- Unit S: `HarborRuntimeConfig.skills: list[Path] | None = None`. None = unmanaged; a list (even
  empty) = exactly these skills, rejecting `agent_kwargs.skills_dir` and task
  `environment.skills_dir`. Fills `RunMetadata.skills`.
- Unit W: `HarborRunnerTarget.skills: list[SkillRef] | None = None` (same meaning); submission
  freezes `list[ResolvedSkill]` into the spec; the worker passes extracted dirs to
  `HarborRuntimeConfig.skills`.

## Work units and parallelism

| ID | Work unit | Depends on | Main files |
|----|-----------|-----------|------------|
| C | Contracts: `SkillRef`, archive module signatures, skill revision DTO | — | stubs only |
| A | Move content hash + ref syntax to `nemo_helix_plugin`; evals re-exports; tag length fix | C | `nemo_helix_plugin`, `nemo_evals/api/fields.py`, `revisions.py`, `content_hash.py` |
| B | Skill archive module + tests | C | new module in `nemo_helix_plugin` |
| S | SDK Harbor injection, duplicate check, cache digest | — | `nhx_evals_sdk/.../harbor/runtime.py` |
| F | Phase 0 fixtures: test skill V1/V2, Harbor task + verifier, controlled image | — | new fixture dir |
| P | Provenance: resolved skills into result entity/DTO and Intake | C | `taskset_runs/`, `jobs/result_persistence.py`, `api/service/result_service.py` |
| G | Skill entities, storage, publish service | A, B | `plugins/nemo-agents` (new modules) |
| R | Phase 0 proof run (live) | S, F | none |
| I | Skill API routes, CLI, taskset-link verification, OpenAPI regen | G | `plugins/nemo-agents` api/cli, `openapi/` |
| W | Evals wiring: target field, `to_spec` resolve + authz, `_harbor_target`, `request_digest`, worker staging | S, G, B | `jobs/agent_spec.py`, `jobs/runner_targets.py`, `api/taskset_run_schemas.py`, `jobs/agent_evaluate.py` |
| M | Matrix script + live end-to-end + negative tests | everything | none |
| X | Stretch: `evals.json` → taskset generator | — | new module |

Waves:

1. **Wave 0 (serial, short)**: C. Fixing the shared types first stops parallel work diverging.
2. **Wave 1 (5 parallel)**: A, B, S, F, P. Disjoint files except A and B both adding to
   `nemo_helix_plugin` (separate new modules). X can join as a sixth if there is spare capacity.
3. **Wave 2 (2 parallel)**: G, R.
4. **Wave 3 (2 parallel)**: I, W.
5. **Wave 4 (serial)**: M.

Critical path: C → A/B → G → W → M.

Shared generated files (`openapi/openapi.yaml`, `uv.lock`) are regenerated once at integration,
not per unit.
