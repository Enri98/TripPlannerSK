from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any, Optional, Type

from pydantic import BaseModel


def normalize_json_schema(schema: dict) -> dict:
    normalized = deepcopy(schema)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)

            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())
                node["additionalProperties"] = False

            if "anyOf" in node and isinstance(node["anyOf"], list):
                non_null_variants = [
                    item
                    for item in node["anyOf"]
                    if not (isinstance(item, dict) and item.get("type") == "null")
                ]
                if len(non_null_variants) == 1 and isinstance(non_null_variants[0], dict):
                    node.pop("anyOf", None)
                    for key, value in non_null_variants[0].items():
                        node.setdefault(key, value)
                elif non_null_variants:
                    node["anyOf"] = non_null_variants

            for key in ("$defs", "definitions", "properties", "patternProperties", "dependentSchemas"):
                child_map = node.get(key)
                if isinstance(child_map, dict):
                    for child_key, child in child_map.items():
                        child_map[child_key] = walk(child)

            for key in ("items", "contains", "if", "then", "else", "not", "propertyNames"):
                child = node.get(key)
                if child is not None:
                    node[key] = walk(child)

            for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
                child_list = node.get(key)
                if isinstance(child_list, list):
                    node[key] = [walk(child) for child in child_list]

            return node

        if isinstance(node, list):
            return [walk(item) for item in node]

        return node

    return walk(normalized)


def get_structured_output_settings(model_class: Type[BaseModel]) -> dict:
    schema_name = re.sub(r"(?<!^)(?=[A-Z])", "_", model_class.__name__).lower()
    normalized_schema = normalize_json_schema(model_class.model_json_schema())
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": normalized_schema,
        },
    }


def create_rpc_error(code: int, message: str, rpc_id: Optional[int] = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": rpc_id,
    }


def is_schema_response_format_unsupported(exc: Exception) -> bool:
    _ = os.getenv("API_VERSION")

    def collect_messages(error: BaseException, seen: set[int]) -> list[str]:
        if id(error) in seen:
            return []
        seen.add(id(error))

        messages = [str(error), repr(error)]
        inner = getattr(error, "inner_exception", None)
        if isinstance(inner, BaseException):
            messages.extend(collect_messages(inner, seen))
        cause = getattr(error, "__cause__", None)
        if isinstance(cause, BaseException):
            messages.extend(collect_messages(cause, seen))
        context = getattr(error, "__context__", None)
        if isinstance(context, BaseException):
            messages.extend(collect_messages(context, seen))

        for arg in getattr(error, "args", ()):
            if isinstance(arg, BaseException):
                messages.extend(collect_messages(arg, seen))
            else:
                messages.append(str(arg))
                messages.append(repr(arg))
        return messages

    error_text = " ".join(collect_messages(exc, set())).lower()
    markers = (
        "response_format",
        "json_schema",
        "schema",
        "unsupported",
        "not supported",
        "invalid",
        "bad request",
        "required",
    )
    return any(marker in error_text for marker in markers)
