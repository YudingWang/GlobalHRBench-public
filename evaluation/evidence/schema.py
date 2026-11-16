"""Versioned answer schema and mechanical validation."""

from __future__ import annotations

import re
import math

SCHEMA_VERSION = "globalhrbench.answer.v2"
DECISIONS = {"ANSWER", "CLARIFY", "INSUFFICIENT_EVIDENCE", "ESCALATE"}
CONFIDENCE = {"high", "medium", "low"}
SOURCE_ID = re.compile(r"^(?:US|UK|DE|JP|FR|NL|SG|BR|IN|CN|ZA)-[A-Za-z0-9.-]+$")

ANSWER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": sorted(DECISIONS)},
        "answer": {"type": "string"},
        "jurisdiction": {"type": ["string", "null"]},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": ["number", "string", "null"]},
                    "unit": {"type": "string"},
                },
                "required": ["name", "value", "unit"],
            },
        },
        "conditions": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": sorted(CONFIDENCE)},
    },
    "required": ["decision", "answer", "jurisdiction", "fields", "conditions", "citations", "confidence"],
}


def validate_answer(obj: object) -> tuple[bool, list[str]]:
    if not isinstance(obj, dict):
        return False, ["not_object"]
    errors = []
    if obj.get("decision") not in DECISIONS:
        errors.append("missing_or_invalid_decision")
    if not isinstance(obj.get("answer", ""), str):
        errors.append("answer_not_string")
    if obj.get("jurisdiction") is not None and not isinstance(obj.get("jurisdiction"), str):
        errors.append("jurisdiction_not_string")
    fields = obj.get("fields")
    if not isinstance(fields, list):
        errors.append("fields_not_list")
    else:
        names = set()
        for i, field in enumerate(fields):
            if not isinstance(field, dict) or not isinstance(field.get("name"), str) or "value" not in field or not isinstance(field.get("unit"), str):
                errors.append(f"invalid_field_{i}")
                continue
            if field["name"] in names:
                errors.append(f"duplicate_field_{field['name']}")
            names.add(field["name"])
            value = field.get("value")
            if isinstance(value, float) and not math.isfinite(value):
                errors.append(f"nonfinite_field_{i}")
            if isinstance(value, (dict, list)):
                errors.append(f"invalid_field_value_{i}")
    if not isinstance(obj.get("conditions", []), list):
        errors.append("conditions_not_list")
    cits = obj.get("citations", [])
    if not isinstance(cits, list) or any(not isinstance(c, str) or not SOURCE_ID.match(c) for c in cits):
        errors.append("invalid_citations")
    if obj.get("confidence") not in CONFIDENCE:
        errors.append("missing_or_invalid_confidence")
    if obj.get("decision") in {"CLARIFY", "INSUFFICIENT_EVIDENCE", "ESCALATE"} and fields:
        errors.append("nonanswer_decision_has_answer_fields")
    return not errors, errors
