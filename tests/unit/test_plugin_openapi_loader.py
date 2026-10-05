# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter
from nemo_helix_plugin.jobs.openapi_utils import clear_query_param_schemas, generate_openapi_extra_params
from nemo_helix_plugin.service import NemoService, RouterSpec
from pydantic import BaseModel

from script.openapi_helper.openapi_tools import validate_refs
from script.openapi_helper.plugin_loader import build_plugin_app


class _DateFilter(BaseModel):
    gte: str | None = None


class _ListFilter(BaseModel):
    created_at: _DateFilter | None = None


class _PluginService(NemoService):
    name = "widgets"

    def get_routers(self) -> list[RouterSpec]:
        router = APIRouter()

        @router.get("/items", openapi_extra=generate_openapi_extra_params(filter_schema=_ListFilter))
        async def list_items() -> dict[str, list[object]]:
            return {"data": []}

        return [RouterSpec(router=router, prefix="/v1")]


def test_build_plugin_app_openapi_registers_rebased_query_param_schemas(monkeypatch):
    clear_query_param_schemas()
    monkeypatch.setattr(
        "script.openapi_helper.plugin_loader.discover_services",
        lambda: {"widgets": _PluginService},
    )
    try:
        app = build_plugin_app("widgets")
        spec = app.openapi()
        schemas = spec["components"]["schemas"]
        filter_param = next(
            param for param in spec["paths"]["/apis/widgets/v1/items"]["get"]["parameters"] if param["name"] == "filter"
        )

        assert filter_param["schema"] == {"$ref": "#/components/schemas/_ListFilter"}
        assert "_ListFilter" in schemas
        assert "_DateFilter" in schemas
        assert "$defs" not in schemas["_ListFilter"]
        assert validate_refs(spec) == []
    finally:
        clear_query_param_schemas()
