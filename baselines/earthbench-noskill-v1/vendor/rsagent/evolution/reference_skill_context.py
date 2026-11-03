"""Deterministic session ownership and observed tool states, never causal summaries."""
from rsagent.evolution.tool_registry import strict_json


def annotate_sessions(data):
    """Use paired, ordered actions; never borrow another session's active state.

    Descriptions are reported metadata, not host-certified capabilities.
    Unknown state stays unknown; a proposed/validated version is not active.
    """
    contexts = []
    data["records"]["task"]["scope"] = {"kind": "shared_task", "session_ids": []}
    for episode in data["episodes"]:
        sid = episode["episode_id"]
        scope = {"kind": "session", "session_ids": [sid], "session_name": episode["name"]}
        for key in ("start_evidence_id", "ending_evidence_id"):
            data["records"][episode[key]]["scope"] = scope
        first, last, operations = {}, {}, []
        for node in data["nodes"]:
            if node["episode_id"] != sid:
                continue
            for action in node["actions"]:
                call_id, result_id = action["call_evidence_id"], action["result_evidence_id"]
                data["records"][call_id]["scope"] = scope
                if result_id:
                    data["records"][result_id]["scope"] = scope
                for aid in action["artifact_ids"]:
                    data["artifacts"][aid]["scope"] = scope
                if not action["name"].startswith("evo_"):
                    continue
                call = strict_json(data["records"][call_id]["text"])
                result = strict_json(data["records"][result_id]["text"]) if result_id else {}
                op = {"operation": action["name"], "evidence_ids": [call_id] + ([result_id] if result_id else []),
                      "call_returned_success": result.get("success"), "tool": call.get("arguments", {}).get("name")}
                try:
                    value = result.get("output", {})
                    value = strict_json(value) if isinstance(value, str) else value
                except (ValueError, TypeError):
                    value = {}
                if not isinstance(value, dict):
                    value = {}
                if result.get("success") is True:
                    for key in ("revision", "parent", "previous", "active", "description", "success"):
                        if key in value:
                            op["reported_" + key] = value[key]
                    active = []
                    if action["name"] == "evo_list":
                        active = value.get("tools", [])
                    elif action["name"] in {"evo_checkout", "evo_run"}:
                        active = [dict(value, name=op["tool"])]
                    elif action["name"] in {"evo_activate", "evo_rollback"}:
                        active = [{"name": value.get("tool"), "revision": value.get("active")}]
                    if isinstance(active, list):
                        for item in active:
                            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("revision"), str):
                                continue
                            observed = {k: item[k] for k in ("revision", "description") if k in item}
                            observed.update(evidence_id=result_id, observation=action["name"])
                            first.setdefault(item["name"], observed)
                            last[item["name"]] = observed
                operations.append(op)
        context = {"session_id": sid, "session_name": episode["name"], "phase": episode["start"].get("phase", "unknown"),
                   "first_observed_active": first or "unknown", "last_observed_active": last or "unknown",
                   "operations": operations,
                   "scope": "Observed active states at cited events, not guaranteed session-start/end snapshots. Descriptions are reported claims; proposals/checks do not imply activation or capability."}
        episode["tool_context"] = {k: v for k, v in context.items() if k != "operations"}
        contexts.append(context)
    return contexts


def check_plan(plan, doc, evidence):
    """Structure and explicit ownership only. No truth/entailment or advice-quality gate."""
    decisions = doc.get("decisions", []) if isinstance(doc, dict) else []
    items = plan.get("candidates") if isinstance(plan, dict) else None
    if not isinstance(items, list) or len(items) != 2:
        return ["plan.candidates: require two category plans in extraction_plan.json"]
    expected = {d["category"]: d for d in decisions}
    errors, seen = [], set()
    fields = {"category", "action", "target_id", "observations", "advice", "applicability", "revision_review"}

    def strings(value, minimum=0, maximum=6):
        return isinstance(value, list) and minimum <= len(value) <= maximum and all(isinstance(x, str) and x.strip() for x in value)

    for i, item in enumerate(items):
        prefix = f"plan.candidates[{i}]"
        if not isinstance(item, dict) or set(item) != fields:
            errors.append(prefix + ": use exactly category, action, target_id, observations, advice, applicability, revision_review")
            continue
        category = item["category"]
        if not isinstance(category, str) or category not in expected or category in seen:
            errors.append(prefix + ": category must match one unique submitted decision")
            continue
        seen.add(category)
        decision = expected[category]
        if any(item[k] != decision[k] for k in ("action", "target_id")):
            errors.append(prefix + ": action/target_id must match the final draft; update the plan if your decision changes")
        retaining = decision["action"] != "none"
        if retaining and (not isinstance(item["advice"], str) or not item["advice"].strip()):
            errors.append(prefix + ".advice: state the proposed reusable advice")
        if not strings(item["applicability"], int(retaining)):
            errors.append(prefix + ".applicability: state 1..6 transfer conditions (0..6 for none)")
        review = item["revision_review"]
        if not isinstance(review, dict) or set(review) != {"retain", "revise", "remove", "add"} or not all(strings(v) for v in review.values()):
            errors.append(prefix + ".revision_review: retain/revise/remove/add each require 0..6 explained items; empty means no such change")
        elif decision["action"] == "update" and not any(review.values()):
            errors.append(prefix + ".revision_review: explain at least one retention or change for an update")
        observations = item["observations"]
        if not isinstance(observations, list) or not int(retaining) <= len(observations) <= 6:
            errors.append(prefix + ".observations: 1..6 observations for add/update, 0..6 for none")
            continue
        for j, observation in enumerate(observations):
            label = prefix + f".observations[{j}]"
            if not isinstance(observation, dict) or set(observation) != {"session_id", "evidence_ids", "fact"}:
                errors.append(label + ": fields must be session_id, evidence_ids, fact")
                continue
            if not isinstance(observation["fact"], str) or not observation["fact"].strip():
                errors.append(label + ".fact: describe the observation, not a universal rule")
            sid, ids = observation["session_id"], observation["evidence_ids"]
            if sid is not None and not isinstance(sid, str):
                errors.append(label + ".session_id: a session ID or null for shared/unknown material")
                continue
            if not strings(ids, 1):
                errors.append(label + ".evidence_ids: 1..6 current evidence IDs")
                continue
            scoped = False
            for eid in ids:
                if eid not in evidence:
                    errors.append(label + ": unknown current evidence ID " + eid)
                    continue
                owners = evidence[eid].get("scope", {}).get("session_ids", [])
                if owners:
                    scoped = True
                    if sid not in owners:
                        errors.append(label + ": evidence " + eid + " belongs to " + str(owners) + ", not " + str(sid))
            if sid is not None and not scoped:
                errors.append(label + ": shared/unknown evidence alone cannot establish a session-specific observation")
    return errors


PLAN_FORMAT = {"candidates": [{
    "category": "one of the two categories; provide one item per category",
    "action": "add | update | none", "target_id": "current ID for update, otherwise null",
    "observations": [{"session_id": "s0 etc.; null only for shared/unknown sources", "evidence_ids": ["current evidence ID"],
                      "fact": "what was observed in this session; case-specific numbers belong here"}],
    "advice": "proposed reusable recommendation; null is allowed for none",
    "applicability": ["conditions needed to transfer the observation into advice"],
    "revision_review": {"retain": ["old point retained and why"], "revise": ["old point narrowed/rewritten and why"],
                        "remove": ["old point dropped and why"], "add": ["new point added and why"]}
}]}
