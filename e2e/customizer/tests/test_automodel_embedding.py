# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU e2e: automodel bi_encoder fine-tune of Nemotron 3 Embed 1B on mined NVDocs, with BEIR uplift eval.

Trains with the validated NVDocs recipe from the embedding customization docs (``recipe:
bi_encoder``) on the full mined dataset, but serves the base and tuned checkpoints on vLLM
instead of Retriever NIM, then scores both on the frozen 20,909-query ``eval_beir`` split with
``retrieve-eval``. The evaluator adds the ``query: `` and ``passage: `` prefixes itself, so
plain vLLM ``/v1/embeddings`` needs no NIM ``input_type``.
"""

import json
import logging

import pytest
from nemo_automodel_plugin.schema import AutomodelJobInput
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.evals.client import EvaluatorClient
from nemo_helix_plugin.evals.types import (
    RetrieveEvalFilesetRef,
    RetrieveEvalInputSpec,
    RetrieveEvalModelRef,
    SubmitRetrieveEvalJobRequest,
)
from nemo_helix_plugin.models.client import ModelsClient
from nhx.testing.e2e import wait_for_platform_job

from e2e.customizer import customizer_eval as ceval
from e2e.customizer import customizer_jobs as jobs
from e2e.customizer.customization_helpers import unique_name

logger = logging.getLogger(__name__)

# Three epochs over the full mined set (~4,320 steps on 4 GPUs) take hours.
TRAINING_TIMEOUT = 10 * 3600
TEST_TIMEOUT = 12 * 3600

pytestmark = [
    pytest.mark.platform("docker", "kubernetes"),
    pytest.mark.feature("gpu", "automodel", "embedding", "uplift"),
    pytest.mark.timeout(TEST_TIMEOUT),
]

EVAL_METRIC = "ndcg_cut_10"
EVAL_RESULTS_RESULT_NAME = "eval-results"
RETRIEVE_EVAL_TIMEOUT = 3600
# The validated recipe moves nDCG@10 from 0.5671 to 0.6318 (+0.0647) on this split.
MIN_NDCG_UPLIFT = 0.05


def test_automodel_embedding_uplift(
    client: NemoClient,
    customizer_workspace: str,
    nvdocs_embedding_fileset: str,
    embed_base_entity: str,
) -> None:
    ws = customizer_workspace
    base_entity = embed_base_entity
    output_name = unique_name("nemotron-3-embed-1b-tuned")

    # The validated NVDocs recipe. warmup_steps 432 is ~10% of the run's steps on the full mined
    # set; rescale it if the dataset size changes. execution_profile is left to the cluster's
    # default training profile.
    spec = AutomodelJobInput.model_validate(
        {
            "model": f"{ws}/{base_entity}",
            "dataset": {"training": f"{ws}/{nvdocs_embedding_fileset}"},
            "training": {
                "training_type": "sft",
                "recipe": "bi_encoder",
                "finetuning_type": "all_weights",
                "max_seq_length": 512,
                "precision": "bf16",
                "attn_implementation": "flash_attention_2",
                # Keep the default ONNX-primary export: the vLLM compiler serves customized
                # embedding outputs from the Hugging Face checkpoint under alternates/hf.
                "retrieval": {
                    "query_max_length": 512,
                    "passage_max_length": 512,
                    "train_n_passages": 5,
                    "do_distributed_inbatch_negative": False,
                },
            },
            "schedule": {
                "epochs": 3,
                "seed": 42,
                "validation_split": 0.01,
                "val_check_interval": 0.25,
                "checkpoint_selection": "both",
            },
            "batch": {"global_batch_size": 128, "micro_batch_size": 8},
            "optimizer": {
                "learning_rate": 2.5e-5,
                "min_learning_rate": 1.25e-5,
                "warmup_steps": 432,
                "weight_decay": 0.01,
                "lr_decay_style": "cosine",
            },
            "parallelism": {"num_nodes": 1, "num_gpus_per_node": 4, "tensor_parallel_size": 1},
            "output": {"name": output_name},
        }
    )

    job_name, final = jobs.submit_and_wait_customization_job(client, "automodel", spec, ws, timeout=TRAINING_TIMEOUT)
    assert final.status == "completed", jobs.get_job_failure_details(client, job_name, ws)

    deployments: list[tuple[str, str]] = []
    try:
        for entity in (base_entity, output_name):
            deployments.append(jobs.deploy_vllm_model(client, ws, entity))
            _assert_has_model_provider(client, ws, entity)

        base_job = _submit_retrieve_eval(client, ws, nvdocs_embedding_fileset, base_entity)
        tuned_job = _submit_retrieve_eval(client, ws, nvdocs_embedding_fileset, output_name)
        base_score = _wait_for_retrieve_eval_score(client, ws, base_job)
        tuned_score = _wait_for_retrieve_eval_score(client, ws, tuned_job)
    finally:
        for deployment, config in deployments:
            jobs.delete_deployment(client, ws, deployment, config)

    result = ceval.UpliftResult(
        metric=EVAL_METRIC,
        base_score=base_score,
        tuned_score=tuned_score,
        tuned_label="bi_encoder",
    )
    logger.info("automodel embedding: base=%.4f tuned=%.4f uplift=%.4f", base_score, tuned_score, result.uplift)
    assert result.uplift >= MIN_NDCG_UPLIFT, (
        f"{EVAL_METRIC} uplift {result.uplift} is below {MIN_NDCG_UPLIFT}: base={base_score} tuned={tuned_score}"
    )


def _assert_has_model_provider(client: NemoClient, workspace: str, entity: str) -> None:
    """retrieve-eval only scores model entities with an attached provider."""
    model = ModelsClient.from_client(client).get_model(name=entity, workspace=workspace).data()
    assert model.model_providers, f"{workspace}/{entity} has no model_providers; retrieve-eval cannot score it"


def _submit_retrieve_eval(client: NemoClient, workspace: str, dataset_fileset: str, entity: str) -> str:
    job = (
        EvaluatorClient.from_client(client)
        .submit_retrieve_eval_job(
            workspace=workspace,
            body=SubmitRetrieveEvalJobRequest(
                name=unique_name(f"retrieve-eval-{entity}"),
                spec=RetrieveEvalInputSpec(
                    dataset=RetrieveEvalFilesetRef(f"{workspace}/{dataset_fileset}"),
                    target=RetrieveEvalModelRef(f"{workspace}/{entity}"),
                ),
            ),
        )
        .data()
    )
    logger.info("Submitted retrieve-eval job %s for %s/%s", job.name, workspace, entity)
    return job.name


def _wait_for_retrieve_eval_score(client: NemoClient, workspace: str, job_name: str) -> float:
    final = wait_for_platform_job(client, job_name, workspace, timeout=RETRIEVE_EVAL_TIMEOUT, poll_interval=30)
    assert final.status == "completed", jobs.get_job_failure_details(client, job_name, workspace)

    payload = (
        EvaluatorClient.from_client(client)
        .download_retrieve_eval_job_result(workspace=workspace, job=job_name, name=EVAL_RESULTS_RESULT_NAME)
        .read()
    )
    eval_results = json.loads(payload)
    logger.info("retrieve-eval %s results: %s", job_name, eval_results)
    score = eval_results.get(EVAL_METRIC)
    assert score is not None, f"retrieve-eval {job_name} did not report {EVAL_METRIC}: {eval_results}"
    return round(float(score), 4)
