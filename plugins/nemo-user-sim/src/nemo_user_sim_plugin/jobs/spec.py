# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job specs. ``SimulateRequest`` is what users submit; ``SimulateSpec`` adds what the platform resolves."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from usersim.engine.core.probes import known_probes

# tool_calling simulates API responses through a Data Designer model facade, which a hosted run
# has to supply. Until the plugin provides one, it can't run here.
UNSUPPORTED_PROBES = frozenset({"tool_calling"})


class ModelTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["model"] = "model"
    model: str = Field(description="Model name as the inference gateway serves it.")


class AgentTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["agent"] = "agent"
    agent: str = Field(description="Name of an agent registered in this workspace.")
    use_agent_tools: bool = Field(
        default=False,
        description=(
            "Keep the agent's own tools (its MCP servers) during the run. "
            "Only for tools that are safe to call for real."
        ),
    )


Target = Annotated[ModelTarget | AgentTarget, Field(discriminator="kind")]


class RerunSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job: str = Field(description="The earlier simulate run the conversations come from.")
    conversations: list[str] = Field(
        min_length=1,
        max_length=2000,  # a rerun sets users to this many, so it takes the same limit
        description="Trajectory IDs of the conversations to run again, with the same persona and topic.",
    )


class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Target
    probes: list[str] = Field(default=["general_open_ended"], min_length=1, description="UserSim probes to run.")
    locale: str = "en_US"
    users: int = Field(default=50, ge=1, le=2000, description="Number of simulated conversations.")
    seed: int | None = Field(
        default=None, description="Sampling seed. UserSim doesn't guarantee the same personas for the same seed."
    )
    max_turns: int = Field(default=5, ge=1, le=50)
    publish_to_intake: bool = Field(default=True, description="Send each conversation to Intake for review.")
    evaluate: bool = Field(default=True, description="Score the conversations as a second job step.")
    preview: str | None = Field(
        default=None, description="Run the personas of this preview, as drawn, instead of drawing new ones."
    )
    exclude_personas: list[str] = Field(
        default_factory=list, description="Persona UUIDs from the preview to leave out of the run."
    )
    rerun: RerunSource | None = Field(
        default=None,
        description=(
            "Run these conversations of an earlier run again, with the same personas and topics, against the target "
            "as it is now. Sets users to the number of conversations."
        ),
    )
    scenarios: str | None = Field(
        default=None,
        description=(
            "Fileset of custom scenarios: topics.yaml, topics for the general_open_ended probe, or bank.yaml, "
            "an action-request bank for the safety_agentic probe in UserSim's schema with optional actions.yaml. "
            "The probe runs these instead of UserSim's own."
        ),
    )

    @model_validator(mode="after")
    def _a_rerun_runs_its_conversations(self) -> Self:
        if self.rerun is not None:
            if self.preview is not None:
                raise ValueError("A rerun runs an earlier run's personas, so it can't also run a preview.")
            self.users = len(self.rerun.conversations)
        return self

    @model_validator(mode="after")
    def _scenarios_need_their_probe(self) -> Self:
        if self.scenarios and not {"safety_agentic", "general_open_ended"} & set(self.probes):
            raise ValueError(
                "Custom scenarios run in the safety_agentic or general_open_ended probe, so probes must include one."
            )
        return self

    @field_validator("probes")
    @classmethod
    def _known(cls, probes: list[str]) -> list[str]:
        unknown = sorted(set(probes) - set(known_probes()))
        if unknown:
            raise ValueError(f"Unknown probes {unknown}. Known: {sorted(known_probes())}")
        unsupported = sorted(set(probes) & UNSUPPORTED_PROBES)
        if unsupported:
            raise ValueError(f"Probes {unsupported} aren't supported by the UserSim plugin yet.")
        return probes


class SimulateSpec(SimulateRequest):
    assistant_model: str
    fabric_config: dict | None = None


class EvaluateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assistant_model: str = Field(default="unknown", description="Recorded in the evaluator's models file only.")
    scenarios: str | None = Field(default=None, description="The run's custom scenarios, which its scoring reads.")


class ExportSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jobs: list[str] = Field(min_length=1, description="The simulate jobs whose conversations to export together.")
    quality: Literal["generous", "strict"] = Field(
        default="strict", description="Quality gate: strict drops any conversation a judge flagged."
    )
    require_review: bool = Field(
        default=False, description="Keep only conversations a reviewer gave positive feedback in Studio."
    )
    fileset: str = Field(description="Fileset to write the training data to.")
