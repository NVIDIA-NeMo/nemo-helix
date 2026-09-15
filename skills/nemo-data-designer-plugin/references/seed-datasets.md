# Seed Datasets Reference

Seed datasets bootstrap synthetic data generation from existing data. Every column from the seed becomes a Jinja2 variable you can reference in prompts and expressions — the seed provides realism and domain specificity, and Data Designer adds volume and variation on top.

## Supported seed sources

Data Designer runs on NeMo Platform, which reads seed data over the network — not from your local disk. Only these seed sources work:

| Class                     | `seed_type`      | Path format                          | Use for                                                     |
| ------------------------- | ---------------- | ------------------------------------ | ----------------------------------------------------------- |
| `HuggingFaceSeedSource`   | `hf`             | `datasets/<owner>/<name>/**/*.parquet` | A dataset already published on HuggingFace.                 |
| `FilesetFileSeedSource`   | `nmp`            | `[<workspace>/]<fileset>#<file>`     | A single tabular file in the Files service. Most common.    |
| `DirectorySeedSource`     | `directory`      | `[<workspace>/]<fileset>[#<dir>]`    | Many tabular files under a fileset directory.               |
| `FileContentsSeedSource`  | `file_contents`  | `[<workspace>/]<fileset>[#<dir>]`    | Reading raw file contents into a `content` column.          |

`FilesetFileSeedSource` is **not** part of `data_designer.config` — import it from the plugin package:

```python
import data_designer.config as dd
from data_designer_nemo.fileset_file_seed_source import FilesetFileSeedSource

config_builder.with_seed_dataset(
    FilesetFileSeedSource(path="default/my-seed-data#records.parquet")
)
```

The other three are on `dd` as usual (`dd.HuggingFaceSeedSource`, `dd.DirectorySeedSource`, `dd.FileContentsSeedSource`).

Anything else — local files, in-memory dataframes, agent rollout directories — is rejected before the workload runs:

```text
  ✘ Seed source 'df' is not supported on the NeMo Platform.
    Use a serializable seed source such as a HuggingFace dataset
    or the Files service.
```

If the user has seed data on their own machine, they need to upload it to the Files service first (`nemo files --help`), then point a `FilesetFileSeedSource` at the resulting fileset.

## Before configuring a seed source

1. **Read the source code.** Read `seed_source.py` under the config root directory printed by `nemo data-designer agent context` for the parameters shared by all seed sources (`file_pattern`, `recursive`, `encoding`, …). Do not guess types or parameters. Note that `FilesetFileSeedSource` is not in that file — its only field is `path`.

2. **Confirm the seed data is reachable and fetch column names.** Run `nemo data-designer validate <path>`, which resolves the fileset or HuggingFace reference against the platform and reports a missing fileset, an unreachable path, or a path that matches no files. You need the seed's exact column names to write downstream prompts; list them with `nemo files` for a fileset, or from the dataset card for HuggingFace.

## Fileset path format

Fileset references are `[<workspace>/]<fileset>#<path-inside-fileset>`. The workspace defaults to the active one, and `#` separates the fileset name from the path within it:

- `default/my-seed-data#records.parquet` — explicit workspace, single file
- `my-seed-data#raw/` — active workspace, a directory
- `my-seed-data` — the whole fileset (directory-style sources only)

`FilesetFileSeedSource` points at one file, so its path always needs the `#` fragment.

## Notes

- Supported tabular formats: `.parquet`, `.csv`, `.json`, `.jsonl`.
- Seed columns are automatically registered as `SeedDatasetColumnConfig` entries — you do **not** add them manually. Just reference them by name in downstream prompts and expressions.
- A `HuggingFaceSeedSource.token` for a private dataset must be a NeMo Platform secret reference, not a literal token. See `nemo secrets --help`.
