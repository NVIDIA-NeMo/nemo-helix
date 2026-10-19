# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImportImageIdentity(BaseModel):
    model_config = ConfigDict(extra="allow")

    image_ref: str = Field(min_length=1)
    image_digest: str = Field(pattern=r"^(?:[^@]+@)?sha256:[0-9a-f]{64}$")


class ImportTask(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    pack: str = Field(min_length=1)
    pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ImportBenchmark(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    tasks: list[str] = Field(min_length=1)


class BenchmarkImportManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: Literal[1]
    catalog_id: str = Field(min_length=1)
    visibility: Literal["private", "team", "org", "public"]
    source: dict[str, Any]
    tasks: list[ImportTask] = Field(min_length=1)
    benchmarks: list[ImportBenchmark] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_membership(self) -> "BenchmarkImportManifest":
        task_slugs = [task.slug for task in self.tasks]
        if len(task_slugs) != len(set(task_slugs)):
            raise ValueError("task slugs must be unique")
        benchmark_slugs = [benchmark.slug for benchmark in self.benchmarks]
        if len(benchmark_slugs) != len(set(benchmark_slugs)):
            raise ValueError("benchmark slugs must be unique")
        known = set(task_slugs)
        for benchmark in self.benchmarks:
            if len(benchmark.tasks) != len(set(benchmark.tasks)) or not set(benchmark.tasks) <= known:
                raise ValueError(f"benchmark {benchmark.slug} has duplicate or unknown tasks")
        return self


class BenchmarkImportCreate(BaseModel):
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest: BenchmarkImportManifest
    visibility: Literal["private", "team", "org", "public"] = "public"
    description: str | None = None
    images: dict[str, ImportImageIdentity] = Field(default_factory=dict)


class BenchmarkImportTask(BaseModel):
    slug: str
    task_id: str
    task_revision: int
    pack_path: str
    pack_sha256: str
    status: Literal["pending", "uploading", "building", "ready", "failed"]
    image_ref: str | None = None
    image_digest: str | None = None
    image_metadata: dict[str, Any] = Field(default_factory=dict)
    build_error: str | None = None
    upload: dict[str, Any] | None = None


class BenchmarkImportBenchmark(BaseModel):
    slug: str
    name: str
    task_slugs: list[str]
    benchmark_id: str | None = None
    benchmark_revision: int | None = None


class BenchmarkImport(BaseModel):
    id: str
    manifest_sha256: str
    visibility: Literal["private", "team", "org", "public"]
    description: str | None
    status: Literal["uploading", "preparing", "failed", "prepared", "ready"]
    created_at: datetime
    updated_at: datetime
    tasks: list[BenchmarkImportTask]
    benchmarks: list[BenchmarkImportBenchmark]


class BenchmarkImportPrepareResponse(BaseModel):
    import_id: str
    accepted: int
    skipped: int
    errors: dict[str, str] = Field(default_factory=dict)
    status: str


class BenchmarkImportImages(BaseModel):
    images: dict[str, ImportImageIdentity] = Field(default_factory=dict)


class BenchmarkImportPublishRequest(BaseModel):
    smoke_size: int | None = Field(default=None, ge=1)
