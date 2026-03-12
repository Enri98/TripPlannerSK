from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class TaskRequestParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    city: str
    weather: str | None = None
    cuisine_type: str | None = None
    budget: str | None = None


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    jsonrpc: str
    method: str
    params: TaskRequestParams
    id: int | str | None = None


class RpcError(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: int | str
    message: str
    data: dict[str, Any] | list[Any] | str | None = None


class AgentErrorPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    error: RpcError


class AgentTextResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    reply: str


class RpcEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    jsonrpc: str
    id: int | str | None = None
    result: dict[str, Any] | None = None
    error: RpcError | None = None
