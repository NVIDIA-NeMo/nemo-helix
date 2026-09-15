# Person Sampling Reference

## Sampler types

Prefer `"person"` — it provides census-grounded demographics and optional personality traits. Fall back to `"person_from_faker"` only when the locale you need has no fileset on the platform and you cannot get one created.

| `sampler_type`        | Params class                   | When to use                                                                                          |
| --------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `"person"`            | `PersonSamplerParams`          | **Preferred.** Reads Nemotron Personas data from a fileset in the `system` workspace.                 |
| `"person_from_faker"` | `PersonFromFakerSamplerParams` | Fallback when the locale's fileset is unavailable. Basic names/addresses via Faker, not demographically accurate. |

## Locale availability

Person sampling reads locale data from a NeMo Platform fileset named `system/nemotron-personas-dataset-<locale>` (lowercased locale). Nothing is downloaded to your machine, and the `installed` column in `nemo data-designer agent context` output describes a local library path that platform execution does not use — ignore it.

Supported locales: `en_IN`, `en_SG`, `en_US`, `fr_FR`, `hi_Deva_IN`, `hi_Latn_IN`, `ja_JP`, `ko_KR`, `pt_BR`.

`nemo data-designer validate <path>` checks that every locale your config samples from has a reachable fileset. If one is missing, validation fails and the fileset has to be created once, per locale, by someone with an NGC API key secret registered in the platform:

```bash
nemo data-designer personas make-fileset \
  --locale en_US \
  --api-key-secret <workspace>/<secret-name>
```

This is a one-time administrative step, not something to run per dataset. If the fileset is missing and you cannot create it, switch the column to `"person_from_faker"` and tell the user what they lost.

## Usage

The sampled person column is a nested dict. You can keep it as-is in the final dataset, or set `drop=True` to remove it and extract only the fields you need via `ExpressionColumnConfig`:

```python
# Keep the full person dict in the output
config_builder.add_column(dd.SamplerColumnConfig(
    name="person", sampler_type="person",
    params=dd.PersonSamplerParams(locale="en_US"),
))

# Or drop it and extract specific fields
config_builder.add_column(dd.SamplerColumnConfig(
    name="person", sampler_type="person",
    params=dd.PersonSamplerParams(locale="en_US"), drop=True,
))
config_builder.add_column(dd.ExpressionColumnConfig(
    name="full_name",
    expr="{{ person.first_name }} {{ person.last_name }}", dtype="str",
))
```

Set `with_synthetic_personas=True` when the dataset benefits from personality traits, interests, cultural background, or detailed persona descriptions (e.g., for realistic user simulation or persona-driven prompting). This option is only available with `"person"` — `"person_from_faker"` does not support it.

## Person Object Schema

Fields vary by locale. Always run the following script to get the exact schema for the locale you are using (script path is relative to this skill's directory):

```bash
python scripts/get_person_object_schema.py <locale>
```

This reads the locale's `system` fileset and prints the PII fields (always included) and synthetic persona fields (only included when `with_synthetic_personas=True`) available for that locale. Run it from an environment where the `nemo` CLI is installed, so it can reach the platform. If the locale's fileset does not exist yet, the script says so and prints the `personas make-fileset` command to create it.
