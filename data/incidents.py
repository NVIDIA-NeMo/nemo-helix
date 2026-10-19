# /// script
# dependencies = [
#   "data-designer",
# ]
# ///
"""Meridian Compute incident history, generated label-first with Data Designer.

Every row is one incident report plus the routing decision Meridian's policy
says is right for it. The answer is sampled first (runbook, scenario, action,
severity, reporter format); a model then writes the incident the way that
reporter would, given Meridian's codenames and private history but told never
to name the team, runbook or action. A second model judges whether the text
supports the label and whether it leaks the answer, and a deterministic check
flags the most common leaked label strings regardless of what the judge thinks;
the full regex sweep runs when the dataset is split.

Run it on NeMo Helix:

    nemo data-designer validate data/incidents.py
    nemo data-designer check-models data/incidents.py
    nemo data-designer preview data/incidents.py --save-results
    MERIDIAN_SPLIT=train nemo data-designer create data/incidents.py --num-records 3600
    MERIDIAN_SPLIT=test  nemo data-designer create data/incidents.py --num-records 500

MERIDIAN_SPLIT=train samples only the non-held-out scenario families.
MERIDIAN_SPLIT=test samples every family, so the test set contains scenario
types the fine-tuned model never saw. The `training_records` processor renders
the prompt/completion rows the post-training plugin trains on, using the exact same question
text the agent's route_incident tool sends at inference time.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import data_designer.config as dd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent"))
from mcps import prompting  # noqa: E402  (shared prompt rendering)

TAXONOMY = prompting.load_taxonomy(HERE.parent / "taxonomy.json")
SPLIT = os.environ.get("MERIDIAN_SPLIT", "train")
HOLDOUT = set(TAXONOMY["holdout_families"]["families"])

# Bring your own inference. Register any OpenAI-compatible provider with Helix
# (OpenRouter, Ollama, vLLM, NVIDIA endpoints) and point these settings at it;
# see data/README.md. Defaults are the OpenRouter names of the models used for the
# reference dataset: Nemotron 3.5 Lightning writes, Nemotron 3 Super judges. The
# writer and judge should be different models, and neither should be the model
# you later compare against.
PROVIDER = os.environ.get("MERIDIAN_PROVIDER", "default/openrouter")
WRITER_MODEL = os.environ.get("MERIDIAN_WRITER_MODEL", "nvidia/nemotron-3.5-lightning")
JUDGE_MODEL = os.environ.get("MERIDIAN_JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b")
# Extra request fields, as JSON. The reference run used NVIDIA endpoints, where this
# switches off reasoning so the writer returns only the report. Set "{}" for
# providers that reject unknown fields.
NO_THINK = json.loads(os.environ.get("MERIDIAN_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}'))

RUNBOOKS = {rb["id"]: rb for rb in TAXONOMY["runbooks"]}
FAMILY_DESC = TAXONOMY["family_descriptions"]

# Realistic skew: node faults and scheduler noise dominate, security is rare.
RUNBOOK_WEIGHTS = {
    "RB-01": 10, "RB-02": 9, "RB-03": 12, "RB-04": 14, "RB-05": 6, "RB-06": 7,
    "RB-07": 9, "RB-08": 8, "RB-09": 7, "RB-10": 8, "RB-11": 6, "RB-12": 4,
}

# Meridian's response policy is deterministic: severity decides the action for
# each runbook (taxonomy.json, runbooks[].policy). Severity is sampled per runbook
# with realistic weights; the writer conveys it; the action follows from both.
POLICY = {rb["id"]: rb["policy"] for rb in TAXONOMY["runbooks"]}
SEVERITY_WEIGHTS = {rb["id"]: rb["severity_weights"] for rb in TAXONOMY["runbooks"]}

# How the reporter would describe urgency, so the writer never prints "S1".
SEVERITY_WORDS = {
    "S1": "a production outage: many users or jobs affected right now, people are being pulled in",
    "S2": "serious and active: a whole pod, team or service is impaired and someone needs to look now",
    "S3": "real but contained: one node or job, no rush beyond today",
    "S4": "an annoyance or heads-up: nothing is broken for users",
}

STYLES = {s["id"]: s["description"] for s in TAXONOMY["reporter_styles"]}

CODENAMES = "\n".join(f"- {name}: {desc.split(' Public collision')[0]}" for name, desc in TAXONOMY["services"].items())

# Strings that must never appear in a report: they are the answer. Case matters
# for the severity codes ("S3" is a code, "s3" is usually a bucket), and single-word
# team names (Storage, Scheduler, Interconnect, Observability) only count when used
# as a team reference, since they are ordinary words in an incident report.
_MULTIWORD_TEAMS = [rb["team"] for rb in TAXONOMY["runbooks"] if " " in rb["team"]]
_SINGLEWORD_TEAMS = [rb["team"] for rb in TAXONOMY["runbooks"] if " " not in rb["team"]]
LEAK_PATTERNS = [
    r"(?i)\bRB-\d\d\b",
    r"(?i)\brunbook",
    r"(?i)\bauto-remediat\w*",
    r"(?i)\bpage-oncall\b",
    r"(?i)\bticket-business-hours\b",
    r"(?i)\bescalate-incident-commander\b",
    r"(?i)\bincident commander\b",
    r"\bS[1-4]\b",
] + [rf"(?i)\b{re.escape(t)}\b" for t in _MULTIWORD_TEAMS] + [
    rf"(?i)\b{re.escape(t)} (team|on-call|oncall|owners?)\b" for t in _SINGLEWORD_TEAMS
] + [rf"(?i)\b{re.escape(fam)}\b" for fam in FAMILY_DESC]


def _weighted_list(weights: dict[str, int | float], slots: int = 20) -> list[str]:
    """Expand {value: weight} into a list with proportional repeats (uniform sampling over it)."""
    total = float(sum(weights.values()))
    out: list[str] = []
    for value, w in weights.items():
        out += [value] * max(1, round(slots * w / total))
    return out


def _lookup_expr(column: str, table: dict[str, str]) -> str:
    """Jinja expression that maps a column's value through a literal table.
    repr() keeps single quotes, which survive the processor's JSON serialization."""
    return "{{ " + repr(table) + "[" + column + "] }}"


def _families() -> dict[str, list[str]]:
    out = {}
    for rb in TAXONOMY["runbooks"]:
        fams = rb["families"] if SPLIT == "test" else [f for f in rb["families"] if f not in HOLDOUT]
        out[rb["id"]] = fams
    return out


WRITER_SYSTEM = f"""You write realistic incident reports for Meridian Compute, which runs three GPU clusters (pod-a, pod-b, pod-c; nodes named dgx-a01..dgx-a24, dgx-b01..dgx-b24, dgx-c01..dgx-c24) for internal ML teams.

Meridian's internal services, as insiders refer to them:
{CODENAMES}

You are given background that the reporter would NOT write down. Use it only to decide what symptoms to describe.

Hard rules:
- Output only the report text. No title, no preamble, no explanation.
- Describe what the reporter actually sees: symptoms, metric names, error strings, node names, service names, times. Never state the diagnosis as a category.
- Never write any of these: a runbook ID (RB-xx), the word runbook, a team name, an action name (auto-remediate, page, ticket, escalate, incident commander), a severity code (S1 to S4), or the scenario label you were given. Monitoring payloads carry only technical labels (alertname, cluster, node, service, severity as critical/warning/info); never fields like team or response.
- Never quote or paraphrase the background notes. They are for you, not for the report.
- Stay on the one scenario you were given. Do not mix in other failure types from the service description.
- Match the requested format exactly. Engineers in chat are informal, abbreviate, sometimes guess a cause and are sometimes wrong. Pager alerts are one to three terse lines with metric names and thresholds. Tickets have Summary, Impact, Steps taken and Request sections. Monitoring payloads are one compact JSON object.
- Make the urgency in the text match the requested urgency description. Do not name the urgency; show it."""

WRITER_PROMPT = """Format: {{ reporter_style }} ({{ style_description }})
Cluster: {{ cluster }}. Node, if one is involved: {{ node }}.
Urgency to convey: {{ severity_words }}.

Background for you only. What is actually going on: {{ family_description }}.
Service involved and how Meridian thinks about it: {{ runbook_brief }}
{% if collision == "yes" %}Lean on the service codename so that an outsider could misread it as the public product with the same name.{% endif %}
{% if misdirect == "yes" and reporter_style == "slack-message" %}Have the engineer guess a wrong cause in passing.{% endif %}

Write the report now."""

JUDGE_PROMPT = """You audit synthetic incident reports for Meridian Compute, which uses these internal codenames:
{CODENAMES}

The report below was written to be owned by: {{{{ runbook_brief }}}}
The intended response is "{{{{ action_id }}}}" at urgency "{{{{ severity }}}}", in the format "{{{{ reporter_style }}}}".

Report:
<<<
{{{{ incident_text }}}}
>>>

Be strict. For no_leak, scan for any of: a runbook ID like RB-07, the word runbook, a team name (Storage, Interconnect, Scheduler, Node Health, Network Core, Identity and Access, Container Platform, Inference Serving, Data Platform, Observability, Capacity and Quota, Security Incident Response), an action name (auto-remediate, page-oncall, ticket-business-hours, escalate-incident-commander, incident commander), a severity code S1 to S4, or a hyphenated scenario label. Quote every leaked string you find in your reasoning. For realism, garbled text, non-English fragments or a wrong format score 1.""".format(CODENAMES=CODENAMES)


def _leak_hint_expr(needles: list[str]) -> str:
    """Jinja boolean: any needle appears in the lower-cased report. Kept to a few
    terms per column because the sandbox caps expression depth; the full regex
    sweep (team names, scenario slugs) runs at split time in scripts/split_dataset.py."""
    terms = " or ".join(f"('{n}' in incident_text | lower)" for n in needles)
    return "{{ " + terms + " }}"


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    families = _families()
    rb_ids = list(RUNBOOK_WEIGHTS)

    cb = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="writer",
                model=WRITER_MODEL,
                provider=PROVIDER,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.7, top_p=0.95, max_tokens=700, timeout=180, extra_body=NO_THINK, max_parallel_requests=8,
                ),
            ),
            dd.ModelConfig(
                alias="judge",
                model=JUDGE_MODEL,
                provider=PROVIDER,
                inference_parameters=dd.ChatCompletionInferenceParams(
                    temperature=0.0, max_tokens=700, timeout=180, extra_body=NO_THINK, max_parallel_requests=8,
                ),
            ),
        ],
    )

    # ── the answer, sampled first ────────────────────────────────────────
    cb.add_column(dd.SamplerColumnConfig(
        name="runbook_id", sampler_type="category",
        params=dd.CategorySamplerParams(values=rb_ids, weights=[RUNBOOK_WEIGHTS[r] for r in rb_ids]),
    ))
    cb.add_column(dd.SamplerColumnConfig(
        name="scenario_family", sampler_type="subcategory",
        params=dd.SubcategorySamplerParams(category="runbook_id", values=families),
    ))
    # Severity as a subcategory of the runbook (weights expressed by repeating
    # entries: Data Designer caps the complexity of conditional sampler rules).
    cb.add_column(dd.SamplerColumnConfig(
        name="severity", sampler_type="subcategory",
        params=dd.SubcategorySamplerParams(
            category="runbook_id",
            values={rb: _weighted_list({k: v for k, v in w.items() if v > 0}) for rb, w in SEVERITY_WEIGHTS.items()},
        ),
    ))
    # Action follows deterministically from runbook and severity.
    cb.add_column(dd.ExpressionColumnConfig(
        name="action_id",
        expr="{{ " + repr(POLICY) + "[runbook_id][severity] }}",
    ))
    cb.add_column(dd.ExpressionColumnConfig(name="severity_words", expr=_lookup_expr("severity", SEVERITY_WORDS), drop=True))

    # ── who is reporting, and how ────────────────────────────────────────
    cb.add_column(dd.SamplerColumnConfig(
        name="reporter_style", sampler_type="category",
        params=dd.CategorySamplerParams(values=list(STYLES), weights=[35, 30, 15, 20]),
    ))
    cb.add_column(dd.ExpressionColumnConfig(name="style_description", expr=_lookup_expr("reporter_style", STYLES), drop=True))
    # pod-b is where the slow-training-is-really-the-fabric history lives.
    cb.add_column(dd.SamplerColumnConfig(
        name="cluster", sampler_type="category",
        params=dd.CategorySamplerParams(values=["pod-a", "pod-b", "pod-c"], weights=[35, 40, 25]),
        conditional_params={
            "scenario_family == 'slow-allreduce-training-job'": dd.CategorySamplerParams(values=["pod-b"]),
        },
    ))
    cb.add_column(dd.SamplerColumnConfig(
        name="node_index", sampler_type="uniform",
        params=dd.UniformSamplerParams(low=1, high=24), convert_to="int", drop=True,
    ))
    cb.add_column(dd.ExpressionColumnConfig(
        name="node", expr="dgx-{{ cluster[-1] }}{% if node_index < 10 %}0{% endif %}{{ node_index }}",
    ))
    cb.add_column(dd.SamplerColumnConfig(
        name="collision", sampler_type="category",
        params=dd.CategorySamplerParams(values=["yes", "no"], weights=[30, 70]),
    ))
    cb.add_column(dd.SamplerColumnConfig(
        name="misdirect", sampler_type="category",
        params=dd.CategorySamplerParams(values=["yes", "no"], weights=[40, 60]),
    ))
    cb.add_column(dd.ExpressionColumnConfig(name="family_description", expr=_lookup_expr("scenario_family", FAMILY_DESC), drop=True))
    cb.add_column(dd.ExpressionColumnConfig(
        name="runbook_brief",
        expr=_lookup_expr("runbook_id", {rb_id: f"{rb['name']} ({rb['team']}). {rb['definition']} Meridian notes: {rb['private_context']}" for rb_id, rb in RUNBOOKS.items()}),
        drop=True,
    ))

    # ── the incident text ────────────────────────────────────────────────
    cb.add_column(dd.LLMTextColumnConfig(
        name="incident_text", model_alias="writer",
        system_prompt=WRITER_SYSTEM, prompt=WRITER_PROMPT,
    ))

    # ── quality gates ────────────────────────────────────────────────────
    # Deterministic leak hints the judge cannot talk its way around. Custom Python
    # columns are not available on the Helix service, so these are expressions.
    cb.add_column(dd.ExpressionColumnConfig(name="leak_runbook_words", dtype="bool", expr=_leak_hint_expr(["rb-0", "rb-1", "runbook", "incident commander"])))
    cb.add_column(dd.ExpressionColumnConfig(name="leak_action_words", dtype="bool", expr=_leak_hint_expr(["auto-remediat", "page-oncall", "ticket-business-hours", "escalate-incident"])))
    cb.add_column(dd.ExpressionColumnConfig(name="leak_severity_codes", dtype="bool", expr=_leak_hint_expr([" s1 ", " s2 ", " s3 ", " s4 "])))
    cb.add_column(dd.LLMJudgeColumnConfig(
        name="quality", model_alias="judge", prompt=JUDGE_PROMPT,
        scores=[
            dd.Score(
                name="label_support",
                description="Would a Meridian engineer who knows the codenames and history route this report to the intended owner, for the scenario described?",
                options={1: "No, it points elsewhere", 2: "Weakly", 3: "Plausibly", 4: "Clearly", 5: "Unambiguously"},
            ),
            dd.Score(
                name="no_leak",
                description="Does the report avoid every answer string listed in the instructions?",
                options={1: "Names the runbook, team or action outright", 2: "Contains a severity code or scenario label", 3: "Hints heavily", 5: "No leak"},
            ),
            dd.Score(
                name="realism",
                description="Does it read like a real report in the requested format, with urgency shown rather than named?",
                options={1: "Garbled, wrong format or non-English fragments", 3: "Passable", 5: "Indistinguishable from a real one"},
            ),
        ],
    ))

    # ── training records, same question text the agent's tool sends ─────
    cb.add_processor(dd.SchemaTransformProcessorConfig(
        name="training_records",
        template={
            "prompt": prompting.render_runbook_question("{{ incident_text }}", TAXONOMY),
            "completion": " " + _lookup_expr("runbook_id", {rb: prompting.letter_for_runbook(rb, TAXONOMY) for rb in RUNBOOKS}),
            "runbook_id": "{{ runbook_id }}",
            "action_id": "{{ action_id }}",
            "scenario_family": "{{ scenario_family }}",
            "reporter_style": "{{ reporter_style }}",
            "leak_hint": "{{ leak_runbook_words or leak_action_words or leak_severity_codes }}",
            # Judge scores stay in the main dataset only: inside a processor the
            # llm-judge column arrives as a JSON string, so its fields cannot be
            # addressed from a template. Filtering on them happens at split time.
        },
    ))
    return cb
