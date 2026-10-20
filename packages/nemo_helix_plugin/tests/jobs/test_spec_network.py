# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import HelixJobStepSpec, StepIngressRule, StepNetwork, TcpPortRange
from pydantic import ValidationError


def test_step_network_defaults_to_unset():
    step = HelixJobStepSpec(name="train", executor=CPUExecutionProvider(container=ContainerSpec(image="img")))
    assert step.network is None


def test_step_network_round_trips():
    network = StepNetwork(
        ingress=[
            StepIngressRule(
                job_id_label="nemo-rl-job-id",
                tcp_ports=[51234],
                tcp_port_ranges=[TcpPortRange(low=3000, high=4998)],
            )
        ]
    )
    assert StepNetwork.model_validate(network.model_dump(mode="json")) == network


def test_tcp_port_range_rejects_an_inverted_span():
    with pytest.raises(ValidationError):
        TcpPortRange(low=4000, high=3000)


@pytest.mark.parametrize(
    "rule",
    [
        {"job_id_label": "nemo-rl-job-id", "tcp_ports": []},
        {"job_id_label": "nemo-rl-job-id", "tcp_ports": [0]},
        {"job_id_label": "nemo-rl-job-id", "tcp_ports": [65536]},
        {"job_id_label": "not a label", "tcp_ports": [8000]},
    ],
)
def test_step_ingress_rule_rejects_invalid_input(rule):
    with pytest.raises(ValidationError):
        StepIngressRule.model_validate(rule)
