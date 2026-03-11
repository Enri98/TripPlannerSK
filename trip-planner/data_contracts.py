from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class TaskRequestParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    city: str
    weather: str | None = None
    cuisine_type: str | None = None


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


class ActivityItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    type: str
    description: str


class ActivityResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    activities: list[ActivityItem]
    note: str | None = None


class RestaurantItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    type: str
    price_range: str


class RestaurantResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    restaurants: list[RestaurantItem]
    note: str | None = None


class RpcEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    jsonrpc: str
    id: int | str | None = None
    result: dict[str, Any] | None = None
    error: RpcError | None = None


class TripDirectorResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    weather_data: str | dict[str, Any] | list[Any]
    activity_suggestions: ActivityResponse | AgentErrorPayload
    restaurant_recommendations: RestaurantResponse | AgentErrorPayload
    note: str | None = None
