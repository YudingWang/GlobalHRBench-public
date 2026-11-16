"""Closed-book / RAG / oracle evaluation with Ollama Llama 3, plus extractive baseline."""

from __future__ import annotations

import json
import hashlib
import re
import time
import urllib.error
import urllib.request
from typing import Any
from .schema import ANSWER_JSON_SCHEMA, SCHEMA_VERSION, validate_answer

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
WORD_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value) if isinstance(value, (int, float)) else False


def public_schema_hint(question: str) -> dict:
    """Derive output-format hints from public question text only.

    These hints constrain representation (country code, field name, unit); they
    do not expose evaluator gold values or passage IDs to a model.
    """
    q = (question or "").lower()
    country_patterns = {
        "US": ("united states", "u.s.", "american"),
        "UK": ("united kingdom", "great britain", "british"),
        "DE": ("germany", "german", "deutschland"),
        "FR": ("france", "french", "français"),
        "NL": ("netherlands", "dutch"),
        "JP": ("japan", "japanese"),
        "SG": ("singapore",),
        "BR": ("brazil", "brazilian", "brasil"),
    }
    country = next((code for code, pats in country_patterns.items() if any(p in q for p in pats)), None)
    # A false-premise query names a number that is deliberately wrong or
    # incomplete.  Do not turn that number's surface unit into an output
    # constraint; the corrected unit must come from the evidence/statute.
    false_premise = bool(
        re.search(r"\b(correct|right|true)\b", q)
        or "if not" in q
        or "is it" in q
    )
    if false_premise:
        unit = None
    elif "postpartum" in q or "after birth" in q or "after childbirth" in q:
        unit = "weeks_postpartum"
    elif "werktage" in q:
        unit = "Werktage"
    elif "jours ouvrables" in q:
        unit = "jours_ouvrables"
    elif "working days" in q:
        unit = "working_days"
    elif re.search(r"hours?\s+(?:per|a)\s+day|daily|workday|working day", q):
        unit = "hours_per_day"
    elif re.search(r"hours?|weekly|per week|workweek", q):
        unit = "hours_per_week"
    elif "percent" in q or "%" in q:
        unit = "percent"
    elif "workweeks" in q or "work weeks" in q:
        unit = "workweeks"
    elif "months" in q or "month" in q:
        unit = "months"
    elif "weeks" in q or "week" in q:
        unit = "weeks"
    elif "days" in q or "day" in q or "jours" in q:
        unit = "days"
    else:
        unit = None
    is_multiplier = any(x in q for x in ("multiplier", "times", "premium rate", "pay rate"))
    if is_multiplier:
        unit = "multiplier"
    field_name = "multiplier" if is_multiplier else ("threshold" if unit and unit.startswith("hours_") else "value")
    return {"jurisdiction_code_from_question": country, "canonical_field_name": field_name, "canonical_unit_from_question": unit}


def extractive_answer(query: str, passages: list[dict], must_abstain: bool) -> dict:
    if must_abstain or not passages:
        return {"decision": "INSUFFICIENT_EVIDENCE", "answer": "ABSTAIN", "fields": [], "conditions": [], "citations": [], "confidence": "low", "jurisdiction": None, "abstain": True}
    blob = " ".join(p["text"] for p in passages)
    nums = NUMBER.findall(blob)
    ans = nums[0] if nums else blob[:240]
    return {
        "decision": "ANSWER",
        "answer": str(ans),
        "fields": [],
        "conditions": [],
        "citations": [p["id"] for p in passages[:3]],
        "confidence": "low",
        "jurisdiction": passages[0]["country"] if passages else None,
        "abstain": False,
    }


PROMPT = """You are answering an employment-law question.
{instruction}
Reply with a single JSON object:
{{"decision": "ANSWER|CLARIFY|INSUFFICIENT_EVIDENCE|ESCALATE", "answer": "...", "jurisdiction": "US|UK|DE|JP|FR|NL|SG|BR|null", "fields": [{{"name":"...","value":0,"unit":"..."}}], "conditions": [], "citations": ["passage-id"], "confidence": "high|medium|low"}}

Output protocol: jurisdiction must be the two-letter code from the schema,
never a country name. For numeric claims use canonical field names: use
"threshold" for hour thresholds, "multiplier" for pay multipliers, and
"value" for other quantities. Use canonical units such as
"hours_per_week", "hours_per_day", "weeks", "months", "days", "percent",
"workweeks", "working_days", "weeks_postpartum", "weeks_equivalent",
"Werktage", or "jours_ouvrables"; do not substitute a generic unit such as
"hours" or "week". Include the numeric claim in fields even when the answer
sentence also contains it. If decision is CLARIFY, INSUFFICIENT_EVIDENCE, or
ESCALATE, fields must be [] and the answer must not invent a numeric claim.
When the question presents a number as a premise (for example, "is ...
correct? If not"), do not copy that number or its unit: correct it from the
supplied evidence, or state that the evidence is insufficient.

Question: {query}

Evidence:
{evidence}
"""


def prompt_for(condition: str, query: str, evidence: str) -> str:
    if condition == "closed":
        instruction = "Use your general knowledge. Do not invent citations; citations must be empty because no local evidence is supplied. If required business parameters are missing, ask for them or abstain."
    else:
        instruction = "Use only the supplied evidence for factual claims. If the evidence is missing or a required business parameter is absent, ask for it or abstain. Correct a false premise when the evidence supports the correction. Cite only passage IDs present in the evidence."
    return PROMPT.format(instruction=instruction, query=query, evidence=evidence)


def serialize_request(public_input: dict, passages: list[dict], config: dict) -> dict:
    """Canonical request serialization; only public input and explicit evidence enter it."""
    allowed = {"id", "question", "output_language", "user_context"}
    clean = {k: public_input.get(k) for k in allowed}
    evidence = [{"id": p.get("id"), "country": p.get("country"), "language": p.get("language"), "text": p.get("text")} for p in passages]
    schema_definition = {
        "decision": {"type": "string", "enum": ["ANSWER", "CLARIFY", "INSUFFICIENT_EVIDENCE", "ESCALATE"]},
        "answer": {"type": "string", "required": True, "unknown_policy": "Use an explicit uncertainty statement; never invent a value."},
        "jurisdiction": {"type": ["string", "null"], "required": True},
        "fields": {"type": "array", "items": {"name": "string", "value": "number|string|null", "unit": "string"}, "unique_by": "name"},
        "conditions": {"type": "array", "items": "string"},
        "citations": {"type": "array", "items": "visible evidence id only"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    }
    instruction = {
        "system": "Return exactly one JSON object. Follow the full schema definition below. Do not reveal hidden evaluator labels, gold answers, or unavailable evidence.",
        "schema_definition": schema_definition,
        "schema_rules": [
            "For closed-book, citations must be empty and no source claim may be presented as retrieved evidence.",
            "For evidence conditions, cite only IDs in the supplied evidence and use only that evidence for factual claims.",
            "Jurisdiction must be exactly one of US, UK, DE, JP, FR, NL, SG, BR, or null; never spell out a country name.",
            "Use field name threshold for hour thresholds, multiplier for pay multipliers, and value for other quantities; use the canonical unit that matches the question.",
            "Use CLARIFY when a missing user parameter is needed to answer; use INSUFFICIENT_EVIDENCE when the supplied evidence cannot support an answer; use ESCALATE for a genuinely high-risk or out-of-scope case. For any of these three decisions, fields must be an empty list.",
            "If the question contains a false or tentative numeric premise, do not repeat its number/unit as the answer; derive the corrected rule from evidence and label its actual canonical unit.",
            "A safe generic example is {\"decision\":\"CLARIFY\",\"answer\":\"Please provide the work location.\",\"jurisdiction\":null,\"fields\":[],\"conditions\":[\"work location required\"],\"citations\":[],\"confidence\":\"low\"}; do not copy its facts into the answer.",
        ],
        "public_format_hint": public_schema_hint(clean.get("question", "")),
        "provider": config.get("provider", "unknown"),
        "model": config.get("model", "unknown"),
        "schema_version": config.get("output_schema_version", "globalhrbench.answer.v1"),
        "condition": config.get("prompt_condition", config.get("condition", "rag")),
        "output_language": clean.get("output_language", "en"),
        "input": clean,
        "evidence": evidence,
        "max_tokens": config.get("max_tokens", 400),
        "temperature": config.get("temperature", 0.0),
        "prompt_version": config.get("prompt_version", "v3"),
    }
    # Keep the historical payload hash stable when the feature is disabled;
    # future structured-output runs hash the transport contract as well.
    if config.get("structured_outputs"):
        instruction["transport"] = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "globalhrbench_answer",
                    "strict": True,
                    "schema": ANSWER_JSON_SCHEMA,
                },
            },
            "require_parameters": bool(config.get("require_parameters", True)),
        }
    payload = json.dumps(instruction, ensure_ascii=False, sort_keys=True)
    # Conservative offline estimate: Unicode and compact JSON vary by tokenizer;
    # use two characters/token plus 25% safety rather than claiming an exact count.
    estimated = max(1, int((len(payload) / 2) * 1.25))
    return {"payload": instruction, "payload_hash": hashlib.sha256(payload.encode()).hexdigest(), "estimated_input_tokens": estimated, "token_estimator": "conservative_chars_per_token_2_plus_25pct"}


def generate_text(model: str, prompt: str, backend: str = "openrouter") -> str:
    if backend == "openrouter":
        from .openrouter import chat

        return chat(model, prompt)
    if backend == "ollama":
        return ollama_generate(model, prompt)
    raise ValueError(f"unknown backend {backend}")


def ollama_generate(model: str, prompt: str, timeout: int = 180) -> str:
    """Call the local Ollama HTTP API. `ollama run` is interactive and hung on 8GB RAM."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "num_predict": 256},
        }
    ).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise RuntimeError(f"ollama HTTP failed: {e}") from e
    return (data.get("response") or "").strip()


_PASSAGE_ID = re.compile(r"\b(?:US|UK|DE|JP|FR|NL|SG|BR)-[A-Za-z0-9.-]+\b")

_JURISDICTION_ALIASES = {
    "united states": "US", "united states of america": "US", "u.s.": "US", "usa": "US",
    "united kingdom": "UK", "great britain": "UK", "britain": "UK",
    "germany": "DE", "deutschland": "DE", "france": "FR", "netherlands": "NL",
    "the netherlands": "NL", "japan": "JP", "singapore": "SG", "brazil": "BR",
}
_UNIT_ALIASES = {
    "week": "weeks", "weeks per week": "weeks", "working days": "working_days",
    "jours ouvrables": "jours_ouvrables", "times": "multiplier", "time": "multiplier",
    "x": "multiplier", "multiple": "multiplier",
}
_FIELD_ALIASES = {"amount": "value", "number": "value", "quantity": "value", "rate": "multiplier"}


def _normalize_protocol_object(obj: dict) -> tuple[dict, list[str]]:
    """Repair representation-only slips while preserving the raw response.

    This is intentionally limited to schema vocabulary (country aliases,
    singular/plural units, and empty fields on an explicit non-answer).  It
    never changes a numeric value, decision, citation, or evidence claim.
    """
    out = dict(obj)
    actions: list[str] = []
    jur = out.get("jurisdiction")
    if isinstance(jur, str):
        key = jur.strip().casefold()
        normalized = _JURISDICTION_ALIASES.get(key)
        if normalized and normalized != jur:
            out["jurisdiction"] = normalized
            actions.append("jurisdiction_alias")
        elif jur.strip().upper() in {"US", "UK", "DE", "JP", "FR", "NL", "SG", "BR"} and jur != jur.strip().upper():
            out["jurisdiction"] = jur.strip().upper()
            actions.append("jurisdiction_case")
    decision = out.get("decision")
    if decision in {"CLARIFY", "INSUFFICIENT_EVIDENCE", "ESCALATE"} and out.get("fields"):
        out["fields"] = []
        actions.append("drop_fields_on_nonanswer")
    fields = out.get("fields")
    if isinstance(fields, list):
        normalized_fields = []
        for field in fields:
            if not isinstance(field, dict):
                normalized_fields.append(field)
                continue
            nf = dict(field)
            name = nf.get("name")
            if isinstance(name, str) and name.strip().casefold() in _FIELD_ALIASES:
                canonical = _FIELD_ALIASES[name.strip().casefold()]
                if canonical != name:
                    nf["name"] = canonical
                    actions.append("field_name_alias")
            unit = nf.get("unit")
            if isinstance(unit, str) and unit.strip().casefold() in _UNIT_ALIASES:
                canonical = _UNIT_ALIASES[unit.strip().casefold()]
                if canonical != unit:
                    nf["unit"] = canonical
                    actions.append("unit_alias")
            normalized_fields.append(nf)
        out["fields"] = normalized_fields
    citations = out.get("citations")
    if isinstance(citations, list):
        normalized_citations = []
        for citation in citations:
            if isinstance(citation, dict):
                # Some providers return the requested source object instead of
                # the schema's source ID string.  The ID is the only
                # evaluator-visible citation field; retaining the raw object
                # would make an otherwise parseable response schema-invalid.
                citation_id = citation.get("id") or citation.get("passage_id") or citation.get("source_id")
                if isinstance(citation_id, str):
                    normalized_citations.append(citation_id)
                    actions.append("citation_object_to_id")
                    continue
            normalized_citations.append(citation)
        out["citations"] = normalized_citations
    return out, sorted(set(actions))


def _json_obj(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i >= 0:
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = text.find("{", i + 1)
    return None


def parse_json_answer(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    obj = _json_obj(text)
    if obj is None:
        ans_m = re.search(r'"answer"\s*:\s*"((?:\\.|[^"\\])*)"', text)
        jur_m = re.search(r'"jurisdiction"\s*:\s*"([A-Z]{2})"', text)
        cits = _PASSAGE_ID.findall(text)
        if ans_m or cits:
            ans = (ans_m.group(1) if ans_m else text[:400]).replace('\\"', '"').replace("\\n", "\n")
            return {
                "answer": ans,
                "citations": cits,
                "jurisdiction": jur_m.group(1) if jur_m else None,
                "abstain": ans.upper().startswith("ABSTAIN"),
                "error_type": "parse_error",
            }
        abstain = "ABSTAIN" in text.upper()
        return {
            "answer": "ABSTAIN" if abstain else text[:400],
            "citations": cits,
            "jurisdiction": None,
            "abstain": abstain,
            "error_type": "parse_error",
        }
    ans = obj.get("answer", "")
    if isinstance(ans, dict):
        obj = {**obj, **ans}
        ans = obj.get("answer", "")
    elif isinstance(ans, str) and ans.lstrip().startswith("{"):
        inner = _json_obj(ans)
        if inner:
            merged = dict(obj)
            for k, v in inner.items():
                if k == "citations" and merged.get("citations"):
                    continue
                if v not in (None, "", []):
                    merged[k] = v
            obj = merged
            ans = obj.get("answer", ans)
    obj, normalization_actions = _normalize_protocol_object(obj)
    ans = obj.get("answer", "")
    citations = obj.get("citations") or []
    if isinstance(citations, str):
        citations = [citations]
    if not citations:
        citations = _PASSAGE_ID.findall(text)
    jur = obj.get("jurisdiction")
    if jur in (None, "", "null", "None"):
        jur = None
    ans_s = str(ans or "")
    schema_ok, schema_errors = validate_answer(obj)
    result = {
        "answer": ans_s,
        "citations": [str(c) for c in citations],
        "jurisdiction": jur,
        "decision": obj.get("decision"),
        "fields": obj.get("fields") if isinstance(obj.get("fields"), list) else [],
        "conditions": obj.get("conditions") if isinstance(obj.get("conditions"), list) else [],
        "confidence": obj.get("confidence"),
        "abstain": obj.get("decision") in {"CLARIFY", "INSUFFICIENT_EVIDENCE", "ESCALATE"} or _as_bool(obj.get("abstain")) or ans_s.upper().startswith("ABSTAIN"),
        "schema_valid": schema_ok,
        "schema_errors": schema_errors,
    }
    if normalization_actions:
        result["normalization_actions"] = normalization_actions
    return result


def _retry_delay_seconds(message: str, attempt: int) -> float:
    """Return a bounded provider-aware delay for a retryable transport error."""
    retry_after = re.search(r'"Retry-After"\s*:\s*"?(\d+(?:\.\d+)?)', message or "", re.I)
    if retry_after:
        # Provider hints can be very large; keep a single request from
        # blocking the whole matrix indefinitely while honoring short hints.
        return min(120.0, max(0.75, float(retry_after.group(1))))
    return min(8.0, 0.75 * (2**attempt))


def _is_retryable_api_error(message: str) -> bool:
    text = (message or "").lower()
    return any(token in text for token in (
        "http 408", "http 429", "http 500", "http 502", "http 503", "http 504",
        "in_flight_budget_exhausted", "network error", "timed out",
    ))


def score_qa(pred: dict, gold: dict, setting: str = "rag", visible_passage_ids: set[str] | None = None) -> dict:
    abstain_gold = bool(gold.get("must_abstain"))
    is_error = pred.get("error_type") in {"api_error", "parse_error"} or pred.get("status") == "error"
    schema_valid, schema_errors = validate_answer(pred)
    abstain_pred = _as_bool(pred.get("abstain")) or str(pred.get("answer", "")).upper().startswith("ABSTAIN")
    if is_error:
        correct = False
    elif abstain_gold:
        correct = abstain_pred
    else:
        correct = (not abstain_pred) and _match_prediction(pred, gold)
    gold_struct = gold.get("gold_answer_struct") or {}
    gold_decision = gold_struct.get("decision")
    pred_decision = pred.get("decision") or ("INSUFFICIENT_EVIDENCE" if abstain_pred else "ANSWER")
    decision_correct = (gold_decision == pred_decision) if gold_decision else (abstain_pred == abstain_gold)
    if gold_decision and (pred.get("decision") is not None or pred.get("fields")):
        correct = correct and decision_correct
    expected_fields = [f for f in (gold_struct.get("fields") or []) if isinstance(f, dict)]
    actual_fields = [f for f in (pred.get("fields") or []) if isinstance(f, dict)]
    by_name = {f.get("name"): f for f in actual_fields}
    field_checks = []
    quantity_checks = []
    unit_checks = []
    for expected in expected_fields:
        actual = by_name.get(expected.get("name"))
        unit_ok = bool(actual is not None and actual.get("unit") == expected.get("unit"))
        value_ok = False
        if actual is not None:
            try:
                value_ok = abs(float(actual.get("value")) - float(expected.get("value"))) <= 1e-6
            except (TypeError, ValueError):
                value_ok = actual.get("value") == expected.get("value")
        unit_checks.append(unit_ok)
        quantity_checks.append(value_ok)
        field_checks.append(value_ok and unit_ok)
    fields_correct = (all(field_checks) if expected_fields else correct) and not is_error
    applicability_correct = None
    gold_conditions = gold_struct.get("conditions")
    if isinstance(gold_conditions, list) and gold_conditions:
        applicability_correct = float(sorted(map(str, pred.get("conditions") or [])) == sorted(map(str, gold_conditions)) and not is_error)
    actor_correct = None
    if gold_struct.get("actor") is not None:
        actor_correct = float(pred.get("actor") == gold_struct.get("actor") and not is_error)
    gold_p = set(gold.get("gold_passages") or [])
    pred_c = {str(c) for c in (pred.get("citations") or [])}
    visible = visible_passage_ids
    valid_c = pred_c if visible is None else pred_c & visible
    if is_error:
        cit_hit = False
    elif abstain_gold:
        # Citation/support are not applicable to a correct refusal.
        cit_hit = False
    else:
        cit_hit = bool(gold_p & valid_c) if gold_p else False
    jur = pred.get("jurisdiction")
    if is_error:
        jur_ok = False
    elif abstain_gold:
        jur_ok = abstain_pred
    elif gold.get("country"):
        jur_ok = jur == gold["country"]
    else:
        jur_ok = True

    # Unsupported / legal hallucination (stricter than v0.1):
    # 1) answered an unanswerable
    # 2) named the wrong country
    # 3) closed-book wrong answer (invented the rule)
    # 4) RAG/oracle answer not citing a gold passage
    ungrounded = False
    if is_error:
        ungrounded = False
    elif not abstain_pred:
        if abstain_gold:
            ungrounded = True
        elif gold.get("country") and jur and jur not in (None, "null", "None") and jur != gold["country"]:
            ungrounded = True
        elif setting == "closed" and not correct:
            ungrounded = True
        elif setting in {"rag", "oracle"} and gold_p and not cit_hit:
            ungrounded = True
    citation_precision = (len(valid_c & gold_p) / len(pred_c)) if pred_c else 0.0
    citation_recall = (len(valid_c & gold_p) / len(gold_p)) if gold_p else 0.0
    core_joint_success = float(
        correct and decision_correct and bool(jur_ok)
        and (not expected_fields or fields_correct) and not is_error
    )
    # The historical ``joint_success`` metric is deliberately retained for
    # backward compatibility: it checks the answer/decision/jurisdiction/
    # required-field bundle, but not free-text applicability conditions.  The
    # strict metric makes that boundary explicit and is the one to use when a
    # claim requires exact applicability-condition agreement.
    full_joint_success = float(
        core_joint_success
        and (applicability_correct is None or bool(applicability_correct))
    )
    return {
        "correct": float(correct),
        "citation_hit": float(cit_hit),
        "jurisdiction_ok": float(bool(jur_ok)),
        "abstain_pred": float(abstain_pred),
        "hallucination": float(ungrounded),
        "ungrounded": float(ungrounded),
        "false_answer": float((not abstain_pred) and (not correct)),
        "citation_precision": float(citation_precision),
        "citation_recall": float(citation_recall),
        "citation_validity": float(len(valid_c) / len(pred_c)) if pred_c else 0.0,
        "answer_and_gold_citation_match": float(cit_hit and correct),
        "evidence_support": None,
        "answerability_correct": float((abstain_pred == abstain_gold) and not is_error),
        "decision_exact_match": float(decision_correct and not is_error),
        "fields_correct": float(fields_correct),
        "quantity_correct": float(all(quantity_checks) and not is_error) if expected_fields else None,
        "unit_correct": float(all(unit_checks) and not is_error) if expected_fields else None,
        "applicability_correct": applicability_correct,
        "actor_correct": actor_correct,
        "joint_success": core_joint_success,
        "full_joint_success": full_joint_success,
        "api_error": float(pred.get("error_type") == "api_error"),
        "parse_error": float(pred.get("error_type") == "parse_error"),
        "valid_response": float(schema_valid and not is_error),
        "schema_valid": float(schema_valid and not is_error),
        "schema_errors": schema_errors,
    }


def _match_prediction(pred_obj: dict, gold: dict) -> bool:
    expected = (gold.get("gold_answer_struct") or {}).get("fields") or []
    actual = pred_obj.get("fields") or []
    if actual or pred_obj.get("decision") is not None:
        schema_ok, _ = validate_answer(pred_obj)
        if not schema_ok:
            return False
    if expected and actual:
        for e in expected:
            matches = [a for a in actual if a.get("name") == e.get("name")]
            if not matches:
                return False
            a = matches[0]
            if a.get("unit") != e.get("unit"):
                return False
            try:
                if abs(float(a.get("value")) - float(e.get("value"))) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        return True
    return _match(pred_obj.get("answer", ""), gold)


def _match(pred: str, gold: dict) -> bool:
    p = (pred or "").strip().lower()
    if not p:
        return False
    if gold.get("numeric") is not None:
        normalized = (pred or "").lower().replace("one and one-half", "1.5").replace("one-half", "0.5")
        for word, value in WORD_NUMBERS.items():
            normalized = re.sub(rf"\b{word}\b", str(value), normalized)
        nums = [float(x) for x in NUMBER.findall(normalized)]
        g = float(gold["numeric"])
        # A numeric match must be stated with the expected unit and without
        # negating the matching number.  It is not enough to mention a number.
        unit = str(gold.get("unit") or "").lower()
        unit_terms = {
            "multiplier": ("times", "multiplier", "rate", "x"),
            "hours_per_week": ("hours", "week"),
            "hours_per_day": ("hours", "day"),
            "weeks": ("week",), "months": ("month",), "days": ("day",),
            "percent": ("percent", "%"),
        }
        terms = unit_terms.get(unit, tuple(unit.replace("_", " ").split()))
        if len(nums) > 1 and not re.search(r"\b(not|rather than|instead of|correct)\b", normalized):
            return False
        for n in nums:
            if abs(n - g) < 1e-6 or abs(n - g) / max(abs(g), 1) < 0.01:
                if len(nums) > 1 and re.search(rf"\b(not|incorrect|wrong)\b[^.]*\b{re.escape(str(int(n) if n.is_integer() else n))}\b", normalized):
                    continue
                number_text = str(int(n) if n.is_integer() else n)
                number_pos = normalized.find(number_text)
                local = normalized[number_pos:number_pos + 45].split(";")[0]
                if unit == "hours_per_week":
                    unit_at_number = bool(re.search(r"\b(?:40|\d+(?:\.\d+)?)\s*(?:hours?|h)\s*(?:per\s*)?week\b", local))
                elif unit == "hours_per_day":
                    unit_at_number = bool(re.search(r"\b(?:\d+(?:\.\d+)?)\s*(?:hours?|h)\s*(?:per\s*)?day\b", local))
                elif unit == "multiplier":
                    unit_at_number = bool(re.search(r"\b(?:\d+(?:\.\d+)?)\s*(?:x|times|multiplier|rate)\b", local))
                else:
                    unit_at_number = not unit or any(token in local for token in terms)
                if not unit_at_number:
                    continue
                prefix = normalized[max(0, number_pos - 18):number_pos]
                if not re.search(r"\b(not|n't|incorrect|wrong|below|under)\b", prefix):
                    return True
        # 1.5 written as one and one-half
        if g == 1.5 and ("time and a half" in normalized):
            return True
        # Numeric gold is authoritative: do not fall back to loose token
        # overlap after a number/unit/negation mismatch.
        return False
    ga = str(gold.get("gold_answer", "")).strip().lower()
    if ga and ga != "abstain":
        if ga in p:
            return True
        # require a non-trivial substring; empty/`p in ga` previously marked "" as correct
        if len(p) >= 12 and p in ga:
            return True
    # token overlap
    gt = set(re.findall(r"\w+", ga))
    pt = set(re.findall(r"\w+", p))
    if not gt:
        return False
    return len(gt & pt) / len(gt) >= 0.5


def format_evidence(passages: list[dict]) -> str:
    if not passages:
        return "(no evidence)"
    parts = []
    for p in passages:
        parts.append(f"[{p['id']}] country={p['country']} {p['statute']} {p['article']}\n{p['text']}")
    return "\n\n".join(parts)


def run_llm_slice(
    queries: list[dict],
    passages: list[dict],
    retrieval_rows: list[dict] | None,
    condition: str,
    model: str,
    max_items: int,
    backend: str = "extractive",
    run_id: str | None = None,
    max_tokens: int = 400,
    output_dir=None,
    frozen_evidence: dict[str, list[dict]] | None = None,
    structured_outputs: bool = False,
    require_parameters: bool = True,
    reasoning: dict | None = None,
    retry_limit: int = 0,
    truncated_retry_max_tokens: int | None = None,
    prompt_condition_override: str | None = None,
) -> dict:
    by_id = {p["id"]: p for p in passages}
    top_map = {}
    if retrieval_rows:
        for r in retrieval_rows:
            top_map[r["query_id"]] = r.get("top_ids") or []
    items = list(queries)[:max_items]
    rows = []
    t0 = time.time()
    ckpt = None
    done_ids: set[str] = set()
    if backend == "openrouter":
        from pathlib import Path as _P

        slug = model.replace("/", "_")
        suffix = f"_{run_id}" if run_id else ""
        ckpt_root = _P(output_dir) if output_dir is not None else _P(__file__).resolve().parents[2] / "experiments" / "results"
        ckpt = ckpt_root / f"qa_{slug}_{condition}{suffix}_rows.jsonl"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if ckpt.exists():
            query_by_id = {q["id"]: q for q in items}
            kept: list[str] = []
            for line in ckpt.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = rec.get("query_id")
                if not qid or qid in done_ids:
                    continue
                q0 = query_by_id.get(qid)
                if q0 is not None and rec.get("prompt_hash"):
                    if frozen_evidence is not None:
                        e0 = frozen_evidence.get(qid, [])
                    elif condition == "closed":
                        e0 = []
                    elif condition == "oracle":
                        e0 = [by_id[pid] for pid in q0.get("gold_passages", []) if pid in by_id]
                    elif condition == "country_filtered_rag":
                        from .retrieve import infer_query_country
                        requested = infer_query_country(q0.get("question") or q0.get("query"))
                        e0 = [by_id[pid] for pid in top_map.get(qid, [])[:5] if pid in by_id and requested and by_id[pid].get("country") == requested]
                    else:
                        e0 = [by_id[pid] for pid in top_map.get(qid, [])[:5] if pid in by_id]
                    p0 = serialize_request({"id": q0["id"], "question": q0.get("question") or q0["query"], "output_language": q0.get("output_language", q0.get("language", "en")), "user_context": q0.get("user_context") or {}}, e0, {"condition": condition, "prompt_condition": prompt_condition_override or condition, "model": model, "provider": backend, "output_schema_version": SCHEMA_VERSION, "max_tokens": max_tokens, "structured_outputs": structured_outputs, "require_parameters": require_parameters})
                    expected_hash = p0["payload_hash"]
                    if rec.get("prompt_hash") != expected_hash:
                        continue
                rows.append(rec)
                done_ids.add(qid)
                kept.append(json.dumps(rec, ensure_ascii=False))
            ckpt.write_text(("\n".join(kept) + ("\n" if kept else "")), encoding="utf-8")
            print(f"[llm {condition}] resume {len(done_ids)} from {ckpt.name}", flush=True)
    for idx, q in enumerate(items, 1):
        if q["id"] in done_ids:
            continue
        if frozen_evidence is not None:
            evid = frozen_evidence.get(q["id"], [])
        elif condition == "closed":
            evid = []
        elif condition == "oracle":
            evid = [by_id[pid] for pid in q.get("gold_passages", []) if pid in by_id]
        elif condition == "country_filtered_rag":
            from .retrieve import infer_query_country
            requested = infer_query_country(q.get("question") or q.get("query"))
            candidate_ids = top_map.get(q["id"], [])[:5]
            evid = [by_id[pid] for pid in candidate_ids if pid in by_id and requested and by_id[pid].get("country") == requested]
        else:
            evid = [by_id[pid] for pid in top_map.get(q["id"], [])[:5] if pid in by_id]
        response_meta = None
        if backend in {"openrouter", "ollama"}:
            request_payload = serialize_request({"id": q["id"], "question": q.get("question") or q["query"], "output_language": q.get("output_language", q.get("language", "en")), "user_context": q.get("user_context") or {}}, evid, {"condition": condition, "prompt_condition": prompt_condition_override or condition, "model": model, "provider": backend, "output_schema_version": SCHEMA_VERSION, "max_tokens": max_tokens, "structured_outputs": structured_outputs, "require_parameters": require_parameters})
            prompt = json.dumps(request_payload["payload"], ensure_ascii=False, sort_keys=True)
            retry_history = []
            attempt = 0
            truncation_retried = False
            while True:
                try:
                    call_max_tokens = max_tokens
                    if truncation_retried and truncated_retry_max_tokens:
                        call_max_tokens = max(max_tokens, truncated_retry_max_tokens)
                    if backend == "openrouter":
                        from .openrouter import chat_with_metadata
                        response_meta = chat_with_metadata(
                            model, prompt, max_tokens=call_max_tokens,
                            response_format=(
                                {"type": "json_schema", "json_schema": {
                                    "name": "globalhrbench_answer", "strict": True,
                                    "schema": ANSWER_JSON_SCHEMA,
                                }} if structured_outputs else None
                            ),
                            require_parameters=require_parameters if structured_outputs else False,
                            reasoning=reasoning,
                        )
                        raw = response_meta.get("content") or ""
                    else:
                        raw = generate_text(model, prompt, backend=backend)
                    pred = parse_json_answer(raw)
                    if (
                        not truncation_retried
                        and truncated_retry_max_tokens
                        and (response_meta or {}).get("finish_reason") == "length"
                        and attempt < retry_limit
                    ):
                        retry_history.append({
                            "attempt": attempt + 1,
                            "error": "truncated_output",
                            "retryable": True,
                            "next_max_tokens": max(max_tokens, truncated_retry_max_tokens),
                        })
                        truncation_retried = True
                        attempt += 1
                        continue
                    response_meta = dict(response_meta or {})
                    response_meta["attempt_count"] = attempt + 1
                    response_meta["retry_history"] = retry_history
                    response_meta["truncation_retry_count"] = int(truncation_retried)
                    break
                except Exception as e:  # noqa: BLE001
                    message = str(e)
                    retryable = _is_retryable_api_error(message)
                    if backend == "openrouter" and retryable and attempt < retry_limit:
                        delay = _retry_delay_seconds(message, attempt)
                        retry_history.append({"attempt": attempt + 1, "error": message[:500], "retryable": True, "delay_seconds": delay})
                        time.sleep(delay)
                        attempt += 1
                        continue
                    pred = {"answer": "ERROR", "citations": [], "jurisdiction": None, "abstain": True, "error": message, "error_type": "api_error"}
                    raw = message
                    response_meta = {"error": message, "attempt_count": attempt + 1, "retry_history": retry_history}
                    break
        else:
            pred = extractive_answer(q.get("question") or q["query"], evid, False)
            raw = None
        visible_ids = {p["id"] for p in evid}
        score_setting = "closed" if condition == "closed" else ("oracle" if condition == "oracle" else "rag")
        sc = score_qa(pred, q, setting=score_setting, visible_passage_ids=visible_ids if condition != "closed" else None)
        rec = {
            "query_id": q["id"],
            "condition": condition,
            "country": q.get("country"),
            "slice": q.get("slice"),
            "pred": pred,
            "error_type": pred.get("error_type"),
            "failure_type": (
                "api_error" if pred.get("error_type") == "api_error" else
                "truncated_output" if pred.get("error_type") == "parse_error" and (response_meta or {}).get("finish_reason") == "length" else
                "malformed_or_unrecoverable_json" if pred.get("error_type") == "parse_error" else
                "schema_invalid" if not pred.get("schema_valid", True) else None
            ),
            "raw": raw or "",
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest() if backend in {"openrouter", "ollama"} else None,
            "evidence_ids": [p["id"] for p in evid],
            "response_meta": response_meta,
            **sc,
        }
        rows.append(rec)
        if ckpt is not None:
            with ckpt.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if idx == 1 or idx % 10 == 0 or idx == len(items):
            print(f"[llm {condition}] {idx}/{len(items)} elapsed={time.time()-t0:.0f}s last_ok={sc['correct']}", flush=True)
    n = len(rows) or 1
    summary = {
        "condition": condition,
        "backend": model if backend in {"openrouter", "ollama"} else "extractive",
        "api": backend,
        "n": len(rows),
        "seconds": time.time() - t0,
        "correct": sum(r["correct"] for r in rows) / n,
        "citation_hit": sum(r["citation_hit"] for r in rows) / n,
        "jurisdiction_ok": sum(r["jurisdiction_ok"] for r in rows) / n,
        "hallucination": sum(r["hallucination"] for r in rows) / n,
        "ungrounded": sum(r.get("ungrounded", r["hallucination"]) for r in rows) / n,
        "false_answer": sum(r.get("false_answer", 0.0) for r in rows) / n,
    }
    return {"summary": summary, "rows": rows}


def rescore_rows(rows: list[dict], queries: list[dict], condition: str, reparse_raw: bool = True) -> list[dict]:
    """Re-apply score_qa to stored preds (no new model calls)."""
    by_id = {q["id"]: q for q in queries}
    out = []
    for rec in rows:
        gold = by_id.get(rec["query_id"])
        pred = rec.get("pred") or {}
        if reparse_raw and rec.get("raw") and not rec.get("error_type"):
            pred = parse_json_answer(str(rec["raw"]))
        if gold is None:
            out.append(rec)
            continue
        visible = set(rec.get("evidence_ids") or [])
        sc = score_qa(pred, gold, setting=condition, visible_passage_ids=visible if condition in {"rag", "oracle"} else None)
        new = dict(rec)
        new["pred"] = pred
        new.update(sc)
        out.append(new)
    return out


def summary_from_rows(rows: list[dict], condition: str, model: str, backend: str = "openrouter") -> dict:
    n = len(rows) or 1
    return {
        "condition": condition,
        "backend": model,
        "api": backend,
        "n": len(rows),
        "correct": sum(r["correct"] for r in rows) / n,
        "citation_hit": sum(r["citation_hit"] for r in rows) / n,
        "jurisdiction_ok": sum(r["jurisdiction_ok"] for r in rows) / n,
        "hallucination": sum(r["hallucination"] for r in rows) / n,
        "ungrounded": sum(r.get("ungrounded", r["hallucination"]) for r in rows) / n,
        "false_answer": sum(r.get("false_answer", 0.0) for r in rows) / n,
        "model": model,
    }
