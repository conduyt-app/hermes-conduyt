"""The Conduyt listener protocol's schemas (contract/*.json, exported from the app's lib/agent-protocol-schema.ts) and a small
JSON Schema validator for them.

The validator FAILS CLOSED like the npm CLI's matchesSchema: a construct it does not understand never passes. It covers what the
protocol schemas use: type (a string or a union list), enum, const, required, properties, additionalProperties false, items,
maxItems, minimum, pattern, format uuid / date-time (only when ``formats``), oneOf (exactly one), anyOf, and if/then.
No Hermes import: this module is used by the tests without a Hermes install.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

CONTRACT_DIR = Path(__file__).resolve().parent / "contract"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_KNOWN_KEYS = {
    "type", "enum", "const", "required", "properties", "additionalProperties", "items", "maxItems", "minimum", "pattern",
    "format", "oneOf", "anyOf", "if", "then", "description", "title", "nullable",
}


@lru_cache(maxsize=1)
def protocol_schemas() -> dict[str, Any]:
    return json.loads((CONTRACT_DIR / "agent-protocol-schema.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def listener_contract() -> dict[str, Any]:
    return json.loads((CONTRACT_DIR / "agent-listener-contract.json").read_text(encoding="utf-8"))["endpoints"]


def schema(name: str) -> dict[str, Any]:
    return protocol_schemas()[name]


def _type_ok(t: str, value: Any) -> bool:
    if t == "null":
        return value is None
    if t == "boolean":
        return isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "string":
        return isinstance(value, str)
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, dict)
    raise ValueError(f"unknown type {t!r}")


def _format_ok(fmt: str, value: Any) -> bool:
    if not isinstance(value, str):
        return True  # a format constrains strings only
    if fmt == "uuid":
        return bool(UUID_RE.match(value))
    if fmt == "date-time":
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False
    return False  # an unknown format fails closed


def matches(s: Any, value: Any, *, formats: bool = False) -> bool:
    """True when ``value`` matches schema ``s``."""
    try:
        return _matches(s, value, formats)
    except ValueError:
        return False


def _matches(s: Any, value: Any, formats: bool, partial: bool = False) -> bool:
    """``partial``: an if/then branch, which may constrain only properties / required (JSON Schema applies those to objects)."""
    if not isinstance(s, dict):
        raise ValueError("schema is not an object")
    if s == {}:
        return True
    unknown = set(s) - _KNOWN_KEYS
    if unknown:
        raise ValueError(f"unsupported schema keys {sorted(unknown)}")
    if not any(k in s for k in ("type", "enum", "const", "oneOf", "anyOf")) and not (partial and set(s) <= {"properties", "required"}):
        raise ValueError("schema has no type, enum, const, oneOf or anyOf")
    if "const" in s and value != s["const"]:
        return False
    if "enum" in s and value not in s["enum"]:
        return False
    if "type" in s:
        types = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if value is None and s.get("nullable") is True:
            pass
        elif not any(_type_ok(t, value) for t in types):
            return False
    if "oneOf" in s and sum(1 for b in s["oneOf"] if _matches(b, value, formats)) != 1:
        return False
    if "anyOf" in s and not any(_matches(b, value, formats) for b in s["anyOf"]):
        return False
    if isinstance(value, dict):
        props = s.get("properties", {})
        for key in s.get("required", []):
            if key not in value:
                return False
        if s.get("additionalProperties") is False and set(value) - set(props):
            return False
        for key, sub in props.items():
            if key in value and not _matches(sub, value[key], formats):
                return False
        if "if" in s and _matches(s["if"], value, formats, True) and "then" in s and not _matches(s["then"], value, formats, True):
            return False
    if isinstance(value, list):
        if "maxItems" in s and len(value) > s["maxItems"]:
            return False
        if "items" in s and not all(_matches(s["items"], v, formats) for v in value):
            return False
    if "minimum" in s and isinstance(value, (int, float)) and not isinstance(value, bool) and value < s["minimum"]:
        return False
    if formats and isinstance(value, str):
        if "pattern" in s and not re.search(s["pattern"], value):
            return False
        if "format" in s and not _format_ok(s["format"], value):
            return False
    return True


def request_fits_contract(method: str, path_template: str, body: dict[str, Any] | None, params: dict[str, Any] | None) -> list[str]:
    """Problems with a request against the listener contract (empty: it fits). Body fields must be a subset of the endpoint's
    accepted fields; query params a subset of its params."""
    endpoints = listener_contract()
    key = f"{method} {path_template}"
    ep = endpoints.get(key)
    if ep is None:
        return [f"{key} is not a listener endpoint"]
    problems = []
    for field in (body or {}):
        if field not in ep["bodyFields"]:
            problems.append(f"{key}: body field {field!r} is not in the contract")
    if body is not None and not ep["hasRequestBody"]:
        problems.append(f"{key}: has no request body")
    for p in (params or {}):
        if p not in ep["params"]:
            problems.append(f"{key}: query param {p!r} is not in the contract")
    return problems
