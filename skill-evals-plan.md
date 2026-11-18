# Skill entities: hackathon plan

Scope: skills as a versioned platform entity. Harbor skill injection, eval metrics, provenance on
eval results and visualization belong to teammates and are out of scope here.

## Goal

Make agent skills a first-class, versioned entity in nemo-agents, retrievable by ref, with
searchable metadata, that eval work (and later registered agents) can reference.

## Fixed decisions

- Skills live in **nemo-agents**. They are not eval-specific.
- Follow the task/taskset revision design for consistency.
- Integrity: hash the content, store the hash, error on mismatch. No signatures.
- Do not assume GitHub-backed filesets or a local repo checkout.
- A skill revision can link to a **digest-pinned** taskset revision.
- No scores on the skill entity. No experiment concepts (arms, variants, baselines).
- Shared code: only the content-hash helper and the revision ref syntax move to
  `nemo_helix_plugin`. The rest of `nemo_evals/revisions.py` stays in nemo-evals.
- The skill archive module lives in `nemo_helix_plugin`; nemo-agents builds and verifies archives
  with it.

## Skill entity in nemo-agents

- **Archive format** (`nemo_helix_plugin/skill_archive.py`):
  - Deterministic archive, one root, regular files and directories only (no symlinks), unique
    normalised paths, file modes normalised and included in the digest.
  - Limits on download size, entry count, total payload, path length.
  - `evals/` excluded from the canonical capture.
  - `files_hash` = SHA-256 of the archive bytes.
- **Storage**: a service-owned fileset, `<skill>/<archive-sha256>/skill_archive`. Read back and
  verify any existing object before reuse. Define whether skill readers can read the
  service-owned copy.
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
- **Linked taskset**: optional, fully qualified `workspace/taskset#<digest>`, verified through the
  caller-authorized taskset revision GET (evals typed client; nemo-agents never imports nemo-evals).
  Tags are not accepted (that response has no digest).
- **API and CLI**: create, publish, get latest, revision lookup by tag or digest, list (with
  frontmatter search and filtering), list revisions, tags, delete.
  `nemo agents skills create --from-dir` uploads a local directory into a fileset first.

## Contracts (done)

- `nemo_helix_plugin/content_hash.py`: moved from nemo-evals unchanged (evals re-exports it).
- `nemo_helix_plugin/revision_refs.py`: `#tag` / `#<digest>` grammar, `LATEST_TAG`,
  `SUBENTITY_REF_PATTERN`, `PINNED_SUBENTITY_REF_PATTERN`, `parse_subentity_ref`, `is_digest`,
  `validate_tag_name`, `validate_movable_tag`. nemo-evals `api/fields.py` and `revisions.py` import
  and re-export them; evals ref, revision and hash tests pass unchanged.
- `nemo_helix_plugin/agents/skills.py`: `SKILL_NAME_PATTERN`, `SkillRef`, `PinnedSkillRef`,
  `SkillArchiveDescriptor {archive_ref, files_hash, skill_name}`, `SkillFrontmatter`,
  `SkillSource {fileset}`, `SkillInput {source, taskset, tags}`, `Skill` DTO (carries
  `content_hash`), list/tag query params.
- `nemo_helix_plugin/agents/endpoints.py` + `client.py`: `create_skill` (POST `/skills/{name}`),
  `publish_skill` (PUT), `get_skill`, `list_skills`, `list_skill_revisions`, `get_skill_revision`,
  `tag_skill_revision` (PUT `/skills/{name}/tags/{tag}?revision=`), `delete_skill`, mirroring the
  evals taskset routes under `/apis/agents/v2/workspaces/{workspace}`.
- `nemo_helix_plugin/skill_archive.py`: implemented `parse_skill_frontmatter`,
  `build_skill_archive`, `verify_skill_archive`.

## Work units

| ID | Work unit | Status |
|----|-----------|--------|
| A | Shared hash + ref syntax | Done (tag length 255/256 alignment open) |
| B | Skill archive build/verify | Done |
| G | Skill entities, storage, publish service | Done |
| I | Skill API routes, CLI, OpenAPI regen | Done |

## Known issues (tracked for later)

- **Skills fileset is not reserved.** The Files service cannot reserve a fileset for a service, so
  a workspace editor can pre-create `nemo-agents-skills` or overwrite or delete its objects. Every
  read verifies the archive against its pinned hash, so tampering is detected rather than served,
  but a tampered pin is then unusable. Today publish only refuses a pre-created fileset that is not
  on platform storage.
- **Executable bits are lost on publish.** The Files service does not keep file modes, so scripts
  published from a fileset are archived `0644`, and a change that only adds `+x` publishes no new
  revision. Fix: a mode-preserving source (e.g. uploading a pre-built archive).
- **Archive builder trusts its source directory not to change while it runs.** A directory swapped
  for a symlink mid-build could be followed (`O_NOFOLLOW` covers only the last path component), and
  hard-linked files are accepted. Publish stages into a private temp directory first, so this
  matters only for other callers of the builder.
- **Files listing is not paginated.** Source staging validates every entry before allocating
  anything, but the listing response itself is held in memory once.

## List filtering

Skills are filtered by frontmatter through the entities service's relationship filter: the
`agent_skill → revisions` relationship is registered in
`services/core/entities/.../utils/relationships.py`, and `SkillFilter` translates `description` and
`tags` into `{"revisions": {...}}` (`RelationshipOperation` in `nemo_helix_plugin/filter_ops.py`).
A skill matches when *any* of its published revisions matches, not only the latest. The head holds
pointers only.

## Separate issue (not today)

Possible existing race in the nemo-evals publish retry (`plugins/nemo-evals/src/nemo_evals/revisions.py`
~484): with three concurrent publishers, a task head could end up holding an older revision's
content. Plausible from reading the code; not reproduced.
