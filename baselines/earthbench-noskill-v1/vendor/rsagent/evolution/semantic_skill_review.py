"""Structure and evidence-identity checks for independent semantic Skill review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rsagent.runtime.jsonio import dump
from rsagent.evolution.reference_skill_extraction import FIELDS, MAX_DRAFT
from rsagent.evolution.tool_registry import canonical, safe_bytes, strict_json


DECISIONS = {"accept", "revise", "reject"}
CLAIM_STATUSES = {"supported", "contradicted", "unverified"}


def check_review(document: Any, candidates: dict[str, dict], evidence: dict[str, dict]) -> dict[str, Any]:
    errors: list[str] = []
    reviews = document.get("reviews") if isinstance(document, dict) else None
    if not isinstance(reviews, list) or len(reviews) != len(candidates):
        return {"success": False, "errors": ["reviews must contain exactly one item per candidate"]}
    seen: set[str] = set()
    fields = {
        "candidate_id", "decision", "target_semantics", "claim_assessments",
        "decision_reason", "revised_skill", "remaining_uncertainties",
    }
    for index, item in enumerate(reviews):
        prefix = f"reviews[{index}]"
        if not isinstance(item, dict) or set(item) != fields:
            errors.append(prefix + ": use exactly candidate_id, decision, target_semantics, claim_assessments, decision_reason, revised_skill, remaining_uncertainties")
            continue
        candidate_id = item["candidate_id"]
        decision = item["decision"]
        if not isinstance(candidate_id, str) or candidate_id not in candidates or candidate_id in seen:
            errors.append(prefix + ".candidate_id: unknown or duplicate candidate")
        else:
            seen.add(candidate_id)
        if decision not in DECISIONS:
            errors.append(prefix + ".decision: accept, revise or reject required")
        if not isinstance(item["decision_reason"], str) or not item["decision_reason"].strip():
            errors.append(prefix + ".decision_reason: nonempty string required")
        semantics = item["target_semantics"]
        if not isinstance(semantics, list) or not 1 <= len(semantics) <= 12:
            errors.append(prefix + ".target_semantics: 1..12 task-semantic records required")
        else:
            for position, semantic in enumerate(semantics):
                label = prefix + f".target_semantics[{position}]"
                required = {"session_id", "product", "quantity", "unit", "threshold_meaning", "evidence_ids"}
                if not isinstance(semantic, dict) or set(semantic) != required:
                    errors.append(label + ": invalid fields")
                    continue
                if not all(isinstance(semantic[key], str) and semantic[key].strip()
                           for key in ("session_id", "product", "quantity", "unit", "threshold_meaning")):
                    errors.append(label + ": semantic strings must be nonempty")
                _check_evidence_list(semantic.get("evidence_ids"), evidence, label, errors)
        assessments = item["claim_assessments"]
        inventory = candidates.get(candidate_id, {}).get("claim_inventory", [])
        expected_claims = {claim["claim_id"]: claim["text"] for claim in inventory}
        if not isinstance(assessments, list) or len(assessments) != len(expected_claims):
            errors.append(prefix + ".claim_assessments: exactly one assessment per inventoried candidate claim required")
        else:
            assessed_claims: set[str] = set()
            for position, assessment in enumerate(assessments):
                label = prefix + f".claim_assessments[{position}]"
                if not isinstance(assessment, dict) or set(assessment) != {"candidate_claim_id", "claim", "status", "evidence_ids", "analysis"}:
                    errors.append(label + ": invalid fields")
                    continue
                claim_id = assessment.get("candidate_claim_id")
                if claim_id not in expected_claims or claim_id in assessed_claims:
                    errors.append(label + ".candidate_claim_id: unknown or duplicate claim")
                else:
                    assessed_claims.add(claim_id)
                    if assessment.get("claim") != expected_claims[claim_id]:
                        errors.append(label + ".claim: must exactly match the inventoried candidate claim")
                if assessment.get("status") not in CLAIM_STATUSES:
                    errors.append(label + ".status: supported, contradicted or unverified required")
                if not isinstance(assessment.get("claim"), str) or not assessment["claim"].strip():
                    errors.append(label + ".claim: nonempty string required")
                if not isinstance(assessment.get("analysis"), str) or not assessment["analysis"].strip():
                    errors.append(label + ".analysis: nonempty string required")
                _check_evidence_list(assessment.get("evidence_ids"), evidence, label, errors)
            if decision == "accept" and any(
                    assessment.get("status") != "supported" for assessment in assessments if isinstance(assessment, dict)):
                errors.append(prefix + ".decision: accept requires every inventoried claim to be supported")
        uncertainties = item["remaining_uncertainties"]
        if not isinstance(uncertainties, list) or not all(isinstance(value, str) and value.strip() for value in uncertainties) or len(uncertainties) > 12:
            errors.append(prefix + ".remaining_uncertainties: 0..12 nonempty strings required")
        revised = item["revised_skill"]
        if decision == "revise":
            if not isinstance(revised, dict) or set(revised) != FIELDS:
                errors.append(prefix + ".revised_skill: revise requires the six Skill fields")
            else:
                for key in ("title", "when", "rationale"):
                    if not isinstance(revised[key], str) or not revised[key].strip():
                        errors.append(prefix + f".revised_skill.{key}: nonempty string required")
                for key in ("guidance", "limitations"):
                    values = revised[key]
                    if not isinstance(values, list) or not 1 <= len(values) <= 12 or not all(isinstance(value, str) and value.strip() for value in values):
                        errors.append(prefix + f".revised_skill.{key}: 1..12 nonempty strings required")
                _check_evidence_list(revised.get("evidence_ids"), evidence, prefix + ".revised_skill", errors)
        elif revised is not None:
            errors.append(prefix + ".revised_skill: accept/reject requires null")
    return {
        "success": not errors,
        "errors": errors,
        "scope": "Structure and authorized evidence identity only; the host must still inspect semantic judgments and downstream behavior.",
    }


def _check_evidence_list(value: Any, evidence: dict[str, dict], label: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not 1 <= len(value) <= 12 or not all(isinstance(item, str) and item for item in value):
        errors.append(label + ".evidence_ids: 1..12 nonempty IDs required")
        return
    for evidence_id in value:
        if evidence_id not in evidence:
            errors.append(label + ".evidence_ids: unknown ID " + evidence_id)


class SemanticReviewProvider:
    def __init__(self, workspace: Path, output: Path, candidates: dict[str, dict], evidence: dict[str, dict], recorder=None):
        self.workspace = Path(workspace).resolve()
        self.output = Path(output).resolve()
        self.candidates = candidates
        self.evidence = evidence
        self.recorder = recorder
        path = {"type": "string", "description": "Workspace JSON file path, not inline JSON; at most 32768 UTF-8 bytes."}
        self.definitions = [
            {"name": "semantic_review_check", "description": "Check review structure and evidence IDs. This does not certify semantic correctness.",
             "input_schema": {"type": "object", "properties": {"source": path}, "required": ["source"], "additionalProperties": False}},
            {"name": "semantic_review_submit", "description": "Save one final independent candidate review after the structural check succeeds.",
             "input_schema": {"type": "object", "properties": {"source": path}, "required": ["source"], "additionalProperties": False}},
        ]
        self.names = {definition["name"] for definition in self.definitions}

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self.names or set(arguments) != {"source"} or not isinstance(arguments["source"], str):
            raise ValueError("Unknown semantic review operation or invalid source")
        path = (self.workspace / arguments["source"]).resolve(strict=True)
        if not path.is_relative_to(self.workspace):
            raise ValueError("Review source must stay inside workspace")
        document = strict_json(safe_bytes(path, MAX_DRAFT).decode())
        result = check_review(document, self.candidates, self.evidence)
        if name == "semantic_review_submit" and result["success"]:
            target = self.output / "review.json"
            if target.exists():
                raise ValueError("Final review already submitted")
            links = {
                evidence_id: self.evidence[evidence_id]["source"]
                for item in document["reviews"]
                for collection in (item["target_semantics"], item["claim_assessments"])
                for record in collection
                for evidence_id in record["evidence_ids"]
            }
            for item in document["reviews"]:
                if item["revised_skill"]:
                    for evidence_id in item["revised_skill"]["evidence_ids"]:
                        links[evidence_id] = self.evidence[evidence_id]["source"]
            dump(target, {
                "schema": "semantic_skill_review_v1",
                "document": document,
                "candidate_snapshot": self.candidates,
                "evidence_links": links,
                "check": result,
                "status": "semantic_review_pending_behavior_validation",
            })
        if self.recorder:
            self.recorder.log(
                "semantic_review_checked" if name.endswith("check") else "semantic_review_submitted",
                success=result["success"], errors=result["errors"], document=document,
            )
        return canonical(result)
