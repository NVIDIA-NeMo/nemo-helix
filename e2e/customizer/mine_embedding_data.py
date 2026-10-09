# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mine hard negatives from NVDocs for the automodel embedding E2E test (off-CI).

Runs Stage 1 of the embedding customization tutorial against a live NeMo Helix with a
GPU and the Data Designer plugin: ``retrieval-prepare`` reads NVIDIA's published
Retrieval-Synthetic-NVDocs-v1 dump, mines hard negatives with Nemotron 3 Embed 1B, and
writes ``training.jsonl`` plus the frozen ``eval_beir/`` split. This script downloads
both, unique-keys the BEIR qrels, and writes them to
``<generation.output_dir>/embedding_nvdocs/`` where the test (and, later,
``publish_assets_to_s3.sh``) picks them up. By default it keeps every row and query, which
is the data the uplift test's recipe was validated on; ``--max-train-rows`` and
``--max-eval-queries`` subsample for quicker local runs. It also writes a
``--smoke-train-rows`` sample of ``training.jsonl`` to ``embedding_nvdocs_smoke/`` for the
smoke test.

    NHX_BASE_URL=http://localhost:8080 PYTHONPATH=. uv run --frozen \\
        python e2e/customizer/mine_embedding_data.py --workspace default

On an auth-enabled platform pass ``--token`` (bearer token) or ``--principal-id`` (when
reaching the API service directly). If the run is interrupted, pass ``--job-name`` to
download the results of the job already submitted instead of mining again.
"""

import argparse
import csv
import json
import logging
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import httpx
from nemo_data_designer_plugin.jobs.retrieval_spec import RetrievalMiningOptions, RetrievalPrepareJobConfig
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.data_designer.client import DataDesignerClient
from nemo_helix_plugin.data_designer.types import DataDesignerJobRequest
from nemo_helix_plugin.files.client import FilesClient
from nhx.testing.e2e import wait_for_platform_job

from e2e.customizer.assets_manifest import generation_config
from e2e.customizer.customization_helpers import create_hf_model_entity, create_hf_token_secret

logger = logging.getLogger(__name__)

FORMAT_NAME = "embedding_nvdocs"
SMOKE_FORMAT_NAME = "embedding_nvdocs_smoke"
EMBED_HF_REPO = "nvidia/Nemotron-3-Embed-1B-BF16"
EMBED_ENTITY = "nemotron-3-embed-1b"
NVDOCS_SDG = "hf://nvidia/Retrieval-Synthetic-NVDocs-v1@1c0d1856f3fb595b2dda98d4b61061fa6d782d51/nv_pp_dd_sdg.json"
ARTIFACTS_RESULT_NAME = "artifacts"
DOWNLOAD_CHUNK_BYTES = 64 * 1024 * 1024
DOWNLOAD_ATTEMPTS = 30


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("NHX_BASE_URL", "http://localhost:8080"))
    parser.add_argument("--workspace", default="default")
    parser.add_argument(
        "--token", default=os.environ.get("NHX_E2E_ACCESS_TOKEN"), help="Bearer token for an auth-enabled platform."
    )
    parser.add_argument(
        "--principal-id",
        default=os.environ.get("NHX_E2E_PRINCIPAL_ID"),
        help="X-NHX-Principal-Id header, for auth-enabled platforms reached without an edge proxy.",
    )
    parser.add_argument(
        "--job-name", default=None, help="Existing retrieval-prepare job to download instead of submitting a new one."
    )
    parser.add_argument(
        "--max-train-rows", type=int, default=None, help="Rows kept from training.jsonl (default: all)."
    )
    parser.add_argument("--max-eval-queries", type=int, default=None, help="Queries kept in eval_beir (default: all).")
    parser.add_argument(
        "--smoke-train-rows", type=int, default=512, help="Rows sampled into embedding_nvdocs_smoke/training.jsonl."
    )
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the manifest generation seed.")
    parser.add_argument("--timeout", type=float, default=4 * 3600, help="retrieval-prepare timeout in seconds.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <generation.output_dir>.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    cfg = generation_config()
    seed = cfg["seed"] if args.seed is None else args.seed
    output_root = args.output_dir or cfg["output_dir"]
    output_dir = output_root / FORMAT_NAME

    client = NemoClient(
        base_url=args.base_url,
        workspace=args.workspace,
        auth=args.token,
        default_headers={"X-NHX-Principal-Id": args.principal_id} if args.principal_id else None,
    )
    ws = args.workspace
    data_designer = DataDesignerClient.from_client(client)
    job_name = args.job_name or _submit_prepare_job(client, data_designer, ws, seed)

    final = wait_for_platform_job(client, job_name, ws, timeout=args.timeout, image_pull_timeout=1800, poll_interval=30)
    if final.status != "completed":
        raise SystemExit(f"retrieval-prepare {job_name} finished with status {final.status}")

    results = data_designer.list_job_results(workspace=ws, job_collection="retrieval-prepare", name=job_name).data()
    artifacts = next((row for row in results.data if row.name == ARTIFACTS_RESULT_NAME), None)
    if artifacts is None:
        raise SystemExit(f"retrieval-prepare {job_name} has no {ARTIFACTS_RESULT_NAME!r} result")

    # artifact_url is "<workspace>/<fileset>#<path>".
    fileset_ref, _, remote_prefix = artifacts.artifact_url.partition("#")
    artifacts_ws, _, job_fileset = fileset_ref.rpartition("/")
    artifacts_ws = artifacts_ws or ws
    eval_prefix = f"{remote_prefix}/eval_beir/"
    listed = FilesClient.from_client(client).list_files(name=job_fileset, workspace=artifacts_ws).data().data
    sizes = {entry.path: entry.size for entry in listed}
    training_path = f"{remote_prefix}/training.jsonl"
    eval_paths = [path for path in sizes if path.startswith(eval_prefix)]
    if training_path not in sizes or not eval_paths:
        raise SystemExit(f"retrieval-prepare {job_name} is missing {training_path} or files under {eval_prefix}")

    # Kept across runs so a download that dies part-way resumes; removed once the outputs exist.
    staging = Path(tempfile.gettempdir()) / f"nvdocs-artifacts-{job_name}"
    downloads = {training_path: staging / "training.jsonl"}
    downloads.update({path: staging / "eval_beir" / path.removeprefix(eval_prefix) for path in eval_paths})
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    if args.principal_id:
        headers["X-NHX-Principal-Id"] = args.principal_id
    for remote_path, local_path in downloads.items():
        url = (
            f"{args.base_url.rstrip('/')}/apis/files/v2/workspaces/{quote(artifacts_ws, safe='')}"
            f"/filesets/{quote(job_fileset, safe='')}/-/{quote(remote_path, safe='')}"
        )
        _download_resumable(url, local_path, sizes[remote_path], headers=headers)

    _write_subsampled(staging, output_dir, args.max_train_rows, args.max_eval_queries, seed)
    _write_training_sample(staging / "training.jsonl", output_root / SMOKE_FORMAT_NAME, args.smoke_train_rows, seed)
    shutil.rmtree(staging)

    logger.info("Wrote %s and %s", output_dir, output_root / SMOKE_FORMAT_NAME)
    return 0


def _download_resumable(
    url: str,
    local_path: Path,
    size: int,
    *,
    headers: dict[str, str],
    chunk_bytes: int = DOWNLOAD_CHUNK_BYTES,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """Download a ``size``-byte file in byte ranges, retrying a dropped range from where it stopped.

    One streamed GET of the ~800 MB training set does not survive a ``kubectl port-forward``;
    short ranged requests do. A partial file left by an earlier run is resumed. Ranges never
    run past ``size``: the Files API answers those with 416 rather than a shorter body. Each
    request gets its own connection, since the port-forward drops a reused one.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > size:
        local_path.unlink()
    offset = local_path.stat().st_size if local_path.exists() else 0
    failures = 0
    with (
        httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(120.0, connect=30.0),
            limits=httpx.Limits(max_keepalive_connections=0),
            transport=transport,
        ) as http,
        local_path.open("ab") as out,
    ):
        while offset < size:
            byte_range = f"bytes={offset}-{min(offset + chunk_bytes, size) - 1}"
            try:
                with http.stream("GET", url, headers={"Range": byte_range}) as response:
                    response.raise_for_status()
                    if response.status_code == 200:  # Range ignored, so the body is the whole file.
                        out.truncate(0)
                        offset = 0
                    for block in response.iter_bytes():
                        out.write(block)
                        offset += len(block)
                failures = 0
                logger.info("%s: %d / %d MB", local_path.name, offset >> 20, size >> 20)
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                failures += 1
                if failures >= DOWNLOAD_ATTEMPTS:
                    raise SystemExit(f"Gave up on {url} after {failures} failed attempts at byte {offset}") from exc
                delay = min(60, 2**failures)
                logger.warning("%s dropped at byte %d (%s); retrying in %ds", local_path.name, offset, exc, delay)
                time.sleep(delay)
    if local_path.stat().st_size != size:
        raise SystemExit(f"{local_path} is {local_path.stat().st_size} bytes; expected {size}")


def _submit_prepare_job(client: NemoClient, data_designer: DataDesignerClient, ws: str, seed: int) -> str:
    token_secret = create_hf_token_secret(client, ws)
    create_hf_model_entity(client, ws, entity_name=EMBED_ENTITY, hf_repo=EMBED_HF_REPO, token_secret=token_secret)

    prepare_spec = RetrievalPrepareJobConfig(
        sdg_input=NVDOCS_SDG,
        enable_mining=True,
        model=f"{ws}/{EMBED_ENTITY}",
        hard_negatives_to_mine=5,
        hard_neg_margin=0.95,
        seed=seed,
        mining=RetrievalMiningOptions(query_embedding_batch_size=64, document_embedding_batch_size=64),
    )
    job = data_designer.create_job(
        workspace=ws,
        job_collection="retrieval-prepare",
        body=DataDesignerJobRequest(spec=prepare_spec.model_dump(mode="json")),
    ).data()
    logger.info("Submitted retrieval-prepare job %s/%s (resume with --job-name %s)", ws, job.name, job.name)
    return job.name


def _write_subsampled(
    src: Path, dst: Path, max_train_rows: int | None, max_eval_queries: int | None, seed: int
) -> None:
    if not (src / "eval_beir" / "corpus.jsonl").is_file():
        raise SystemExit(f"eval_beir/corpus.jsonl missing under {src}")
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "eval_beir" / "qrels").mkdir(parents=True)
    rng = random.Random(seed)

    train_rows = (src / "training.jsonl").read_text(encoding="utf-8").splitlines()
    train_rows = [row for row in train_rows if row.strip()]
    if max_train_rows is not None and len(train_rows) > max_train_rows:
        train_rows = rng.sample(train_rows, max_train_rows)
    (dst / "training.jsonl").write_text("\n".join(train_rows) + "\n", encoding="utf-8")

    header, qrels = _read_unique_qrels(src / "eval_beir" / "qrels" / "test.tsv")
    query_ids = sorted({row["query-id"] for row in qrels})
    if not query_ids:
        raise SystemExit(f"No eval queries in {src / 'eval_beir' / 'qrels' / 'test.tsv'}")
    if max_eval_queries is not None and len(query_ids) > max_eval_queries:
        query_ids = sorted(rng.sample(query_ids, max_eval_queries))
    kept_queries = set(query_ids)

    with (dst / "eval_beir" / "qrels" / "test.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for row in qrels if row["query-id"] in kept_queries)

    with (
        (src / "eval_beir" / "queries.jsonl").open(encoding="utf-8") as src_queries,
        (dst / "eval_beir" / "queries.jsonl").open("w", encoding="utf-8") as dst_queries,
    ):
        for line in src_queries:
            if line.strip() and str(json.loads(line)["_id"]) in kept_queries:
                dst_queries.write(line if line.endswith("\n") else line + "\n")

    # Keep the full corpus so retrieval difficulty matches the published split.
    shutil.copyfile(src / "eval_beir" / "corpus.jsonl", dst / "eval_beir" / "corpus.jsonl")
    logger.info(
        "embedding_nvdocs: %d training rows, %d eval queries, %d qrels",
        len(train_rows),
        len(kept_queries),
        sum(1 for row in qrels if row["query-id"] in kept_queries),
    )


def _write_training_sample(src: Path, dst: Path, rows: int, seed: int) -> None:
    """Write ``rows`` randomly sampled training rows to ``dst/training.jsonl`` (no eval split)."""
    train_rows = [row for row in src.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(train_rows) > rows:
        train_rows = random.Random(seed).sample(train_rows, rows)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    (dst / "training.jsonl").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    logger.info("%s: %d training rows", dst.name, len(train_rows))


def _read_unique_qrels(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read BEIR qrels, dropping duplicate (query-id, corpus-id) rows the BEIR loader rejects."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        header = list(reader.fieldnames or [])
        seen: set[tuple[str, str]] = set()
        kept: list[dict[str, str]] = []
        for row in reader:
            key = (row["query-id"], row["corpus-id"])
            if key not in seen:
                seen.add(key)
                kept.append(row)
    return header, kept


if __name__ == "__main__":
    raise SystemExit(main())
