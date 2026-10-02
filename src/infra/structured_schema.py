"""Compile the service's Pydantic contracts into a checked OpenAI wire schema.

This is a deliberately bounded dialect adapter, not a general JSON Schema
rewriter. Wire objects contain every field explicitly; domain defaults and
Python validators still belong to the model used after a response arrives.
The supported profile is OpenAI's standard structured output models; a
fine-tuned model's narrower constraints require an explicit future profile.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from pydantic import BaseModel
from pydantic.errors import PydanticInvalidForJsonSchema


SCHEMA_COMPILER_VERSION = "1"


class SchemaCompilationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Unsupported structured output schema at {path}: {reason}")


_ANNOTATIONS = {"title", "description", "default"}
_NUMERIC = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
_STRING = {"minLength", "maxLength", "pattern", "format"}
_ARRAY = {"minItems", "maxItems"}
_KEYWORDS = {
    "$defs", "$ref", "type", "properties", "required", "additionalProperties", "items",
    "anyOf", "oneOf", "discriminator", "enum", "const",
} | _ANNOTATIONS | _NUMERIC | _STRING | _ARRAY
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
_FORMATS = {"date-time", "time", "date", "duration", "email", "hostname", "ipv4", "ipv6", "uuid"}


def _path(parent: str, key: str | int) -> str:
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _fail(path: str, reason: str) -> None:
    raise SchemaCompilationError(path, reason)


def _primitive(value: Any) -> bool:
    return value is None or type(value) in {str, bool, int} or (type(value) is float and math.isfinite(value))


def _same_literal(left: Any, right: Any) -> bool:
    # JSON numbers 1 and 1.0 are equal; JSON booleans and numbers are not.
    numeric = type(left) in {int, float} and type(right) in {int, float}
    return (numeric or type(left) is type(right)) and left == right


def _normalize(node: Any, path: str) -> dict[str, Any]:
    if not isinstance(node, dict):
        _fail(path, "a schema must be an object")
    for key in node:
        if key not in _KEYWORDS:
            _fail(_path(path, key), "keyword is outside the supported OpenAI contract subset")
    result = {key: value for key, value in node.items() if key not in {"default", "const"}}
    for key in ("title", "description"):
        if key in node and not isinstance(node[key], str):
            _fail(_path(path, key), "annotation must be a string")
    if "$defs" in node:
        if not isinstance(node["$defs"], dict):
            _fail(_path(path, "$defs"), "definitions must map names to schemas")
        result["$defs"] = {name: _normalize(value, _path(_path(path, "$defs"), name))
                           for name, value in node["$defs"].items()}
    if "const" in node:
        if not _primitive(node["const"]):
            _fail(_path(path, "const"), "only primitive constants are supported")
        if "enum" in node:
            values = node["enum"]
            if not isinstance(values, list) or len(values) != 1 or not _same_literal(values[0], node["const"]):
                _fail(_path(path, "const"), "const and enum must express the same single value")
        result["enum"] = [node["const"]]
    if "enum" in result and (not isinstance(result["enum"], list) or not result["enum"]
                              or any(not _primitive(value) for value in result["enum"])):
        _fail(_path(path, "enum"), "enum must contain primitive JSON values")

    forms = [key for key in ("type", "$ref", "anyOf", "oneOf") if key in node]
    if len(forms) != 1:
        _fail(path, "exactly one type, reference, or union is required")
    form = forms[0]
    kind = node.get("type")
    if form == "type" and (not isinstance(kind, str) or kind not in _TYPES):
        _fail(_path(path, "type"), "type must be a supported primitive, array, or object")
    if "discriminator" in node and form != "oneOf":
        _fail(_path(path, "discriminator"), "a discriminator is supported only on a tagged oneOf")
    allowed = {form, "$defs", "enum", "const"} | _ANNOTATIONS
    if form == "$ref":
        # OpenAI rejects description/title siblings on references. They are
        # annotations, so removing them changes no accepted JSON values.
        result.pop("title", None)
        result.pop("description", None)
        allowed -= {"enum", "const"}
    if kind == "object":
        allowed |= {"properties", "required", "additionalProperties"}
        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            _fail(_path(path, "properties"), "properties must map names to schemas")
        if node.get("additionalProperties", False) is not False:
            _fail(_path(path, "additionalProperties"), "open objects and dynamic mappings are unsupported")
        required = node.get("required", [])
        if not isinstance(required, list) or any(not isinstance(key, str) or key not in properties for key in required):
            _fail(_path(path, "required"), "required must name existing properties")
        result["properties"] = {name: _normalize(value, _path(_path(path, "properties"), name))
                                for name, value in properties.items()}
        result["required"] = list(properties)
        result["additionalProperties"] = False
    elif kind == "array":
        allowed |= {"items"} | _ARRAY
        if "items" not in node:
            _fail(_path(path, "items"), "arrays require an item schema")
        result["items"] = _normalize(node["items"], _path(path, "items"))
    elif kind == "string":
        allowed |= _STRING
    elif kind in {"integer", "number"}:
        allowed |= _NUMERIC
    elif form in {"anyOf", "oneOf"}:
        if form == "oneOf":
            allowed.add("discriminator")
        branches = node[form]
        if not isinstance(branches, list) or not branches:
            _fail(_path(path, form), "a union requires nonempty schema branches")
        result[form] = [_normalize(branch, _path(_path(path, form), index))
                        for index, branch in enumerate(branches)]
    for key in node:
        if key not in allowed:
            _fail(_path(path, key), "keyword does not apply to this schema shape")
    for key in (_STRING | _ARRAY) & node.keys() - {"pattern", "format"}:
        if type(node[key]) is not int or node[key] < 0:
            _fail(_path(path, key), "length and item limits must be nonnegative integers")
    for key in _NUMERIC & node.keys():
        if type(node[key]) not in {int, float} or not math.isfinite(node[key]) or (key == "multipleOf" and node[key] <= 0):
            _fail(_path(path, key), "numeric constraints must be finite, with a positive multipleOf")
    if "pattern" in node and not isinstance(node["pattern"], str):
        _fail(_path(path, "pattern"), "pattern must be a string")
    if "format" in node and (not isinstance(node["format"], str) or node["format"] not in _FORMATS):
        _fail(_path(path, "format"), "format is outside the supported OpenAI contract subset")
    return result


def _resolve(node: dict[str, Any], root: dict[str, Any], path: str) -> dict[str, Any]:
    seen: set[str] = set()
    while "$ref" in node:
        reference = node["$ref"]
        if not isinstance(reference, str) or not (reference == "#" or reference.startswith("#/")):
            _fail(_path(path, "$ref"), "only local JSON Pointer references are supported")
        if reference in seen:
            _fail(_path(path, "$ref"), "reference aliases form a cycle without an object schema")
        seen.add(reference)
        target: Any = root
        for part in reference[2:].split("/") if reference != "#" else []:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                _fail(_path(path, "$ref"), f"unresolved local reference {reference}")
            target = target[part]
        if not isinstance(target, dict):
            _fail(_path(path, "$ref"), "reference must resolve to a schema object")
        forms = {"type", "$ref", "anyOf", "oneOf"} & target.keys()
        if (len(forms) != 1
                or ("type" in forms and (not isinstance(target["type"], str) or target["type"] not in _TYPES))
                or any(not isinstance(target[key], list) for key in forms & {"anyOf", "oneOf"})):
            _fail(_path(path, "$ref"), "reference must identify a schema, not a properties/definitions map")
        node = target
    return node


def _prove_disjoint(node: dict[str, Any], root: dict[str, Any], path: str) -> None:
    branches = [_resolve(branch, root, _path(_path(path, "oneOf"), index))
                for index, branch in enumerate(node["oneOf"])]
    candidates: set[str] | None = None
    for branch in branches:
        fields = set(branch.get("required", [])) if branch.get("type") == "object" else set()
        candidates = fields if candidates is None else candidates & fields
    discriminator = node.get("discriminator")
    if discriminator is not None:
        if (not isinstance(discriminator, dict) or set(discriminator) - {"propertyName", "mapping"}
                or not isinstance(discriminator.get("propertyName"), str)):
            _fail(_path(path, "discriminator"), "discriminator requires a propertyName")
        candidates = (candidates or set()) & {discriminator["propertyName"]}
    for field in sorted(candidates or []):
        observed: set[str] = set()
        for branch in branches:
            tag = _resolve(branch["properties"][field], root, _path(path, "oneOf"))
            # String tags are the service's tagged-union convention. Other
            # exclusivity proofs belong in an explicit future contract change.
            values = tag.get("enum")
            if not values or any(not isinstance(value, str) for value in values) or observed.intersection(values):
                break
            observed.update(values)
        else:
            return
    _fail(_path(path, "oneOf"), "branches need a common required string tag with disjoint const/enum values")


def _lower_unions(node: dict[str, Any], root: dict[str, Any], path: str) -> None:
    if "$ref" in node:
        _resolve(node, root, path)
    for key in ("properties", "$defs"):
        for name, child in node.get(key, {}).items():
            _lower_unions(child, root, _path(_path(path, key), name))
    for key in ("anyOf", "oneOf"):
        for index, child in enumerate(node.get(key, [])):
            _lower_unions(child, root, _path(_path(path, key), index))
    if "items" in node:
        _lower_unions(node["items"], root, _path(path, "items"))
    if "oneOf" in node:
        _prove_disjoint(node, root, path)
        node["anyOf"] = node.pop("oneOf")
        node.pop("discriminator", None)


def _schema_nodes(node: dict[str, Any], path: str):
    """Visit emitted schema nodes once per location; references add no bytes."""
    yield path, node
    for key in ("properties", "$defs"):
        for name, child in node.get(key, {}).items():
            yield from _schema_nodes(child, _path(_path(path, key), name))
    for index, child in enumerate(node.get("anyOf", [])):
        yield from _schema_nodes(child, _path(_path(path, "anyOf"), index))
    if "items" in node:
        yield from _schema_nodes(node["items"], _path(path, "items"))


def _check_provider_limits(schema: dict[str, Any]) -> None:
    property_count = enum_count = string_length = 0
    definitions: list[tuple[str, dict[str, Any]]] = []
    for path, node in _schema_nodes(schema, "#"):
        properties = node.get("properties", {})
        property_count += len(properties)
        string_length += sum(map(len, properties)) + sum(map(len, node.get("$defs", {})))
        values = node.get("enum", [])
        enum_count += len(values)
        enum_strings = sum(len(value) for value in values if isinstance(value, str))
        string_length += enum_strings
        if len(values) > 250 and enum_strings > 15000:
            _fail(_path(path, "enum"), "an enum with more than 250 values exceeds its 15000-character limit")
        definitions.extend((_path(_path(path, "$defs"), name), child) for name, child in node.get("$defs", {}).items())
    if property_count > 5000:
        _fail("#", "the schema exceeds the total limit of 5000 object properties")
    if enum_count > 1000:
        _fail("#", "the schema exceeds the total limit of 1000 enum values")
    if string_length > 120000:
        _fail("#", "property names, definition names and enum/const strings exceed 120000 characters")

    def check_depth(node: dict[str, Any], path: str, depth: int, ancestors: frozenset[int]) -> None:
        resolved = _resolve(node, schema, path)
        # A Pydantic recursive root is expanded from a definition and shares
        # its property map; count that identity once on the current path.
        identity = id(resolved["properties"]) if resolved.get("type") == "object" else id(resolved)
        if identity in ancestors:
            return
        ancestors = ancestors | {identity}
        depth += int(resolved.get("type") in {"object", "array"})
        if depth > 10:
            _fail(path, "the schema exceeds 10 levels of object/array nesting")
        for name, child in resolved.get("properties", {}).items():
            check_depth(child, _path(_path(path, "properties"), name), depth, ancestors)
        for index, child in enumerate(resolved.get("anyOf", [])):
            check_depth(child, _path(_path(path, "anyOf"), index), depth, ancestors)
        if "items" in resolved:
            check_depth(resolved["items"], _path(path, "items"), depth, ancestors)

    check_depth(schema, "#", 0, frozenset())
    # Unreferenced definitions are still part of the schema submitted to the
    # provider. Check their finite paths without rejecting recursive models.
    for path, definition in definitions:
        check_depth(definition, path, 0, frozenset())


def compile_output_schema(model: type[BaseModel], *, name: str | None = None) -> dict[str, Any]:
    """Return a strict response envelope, or fail before any provider request.

    Required wire fields intentionally narrow the domain's shorthand JSON input
    language. They do not add null to nonnullable fields or replace server-side
    validators; decode the returned object using the original domain model.
    """
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        _fail("#", "a Pydantic model class is required")
    schema_name = name if name is not None else model.__name__
    if not isinstance(schema_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", schema_name):
        _fail("#/name", "schema names require 1-64 letters, digits, underscores, or hyphens")
    try:
        source = model.model_json_schema(mode="validation")
    except PydanticInvalidForJsonSchema as exc:
        raise SchemaCompilationError("#", f"Pydantic could not generate a JSON Schema: {exc}") from exc
    schema = _normalize(source, "#")
    resolved_root = _resolve(schema, schema, "#")
    if resolved_root.get("type") != "object":
        _fail("#", "the output schema root must be an object")
    if "$ref" in schema:
        schema = {**resolved_root, **{key: value for key, value in schema.items() if key != "$ref"}}
    _lower_unions(schema, schema, "#")
    _check_provider_limits(schema)
    return {"name": schema_name, "strict": True, "schema": schema}


def schema_fingerprint(schema: dict[str, Any]) -> str:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
