"""Persistent host-side episode lifecycle, budgets and joint capability release.

This module is not an Agent tool. Solver/learner backends call its methods at
phase boundaries. It never treats a correct answer as proof that learning helped.
"""
from pathlib import Path
import re
from uuid import uuid4

from rsagent.evolution.capability_snapshots import CapabilitySnapshots
from rsagent.evolution.continual_skill import ContinualSkillRegistry
from rsagent.evolution.tool_registry import ToolRegistry, canonical, digest


class EvolutionCoordinator(ToolRegistry):
    def initialize(self, snapshot_root, initial_snapshot, max_model_requests):
        if type(max_model_requests) is not int or max_model_requests <= 0:
            raise ValueError("Need positive request budget")
        snapshots = CapabilitySnapshots(snapshot_root)
        snapshots.snapshot(initial_snapshot)
        if snapshots.current() != initial_snapshot:
            raise ValueError("Initial snapshot must be current in this arm's release store")
        with self.locked():
            if self.read("run.json") is not None:
                raise ValueError("Run already initialized")
            self.write("run.json", {"snapshots": str(snapshots.root), "current": initial_snapshot,
                                   "max_model_requests": max_model_requests, "reserved_requests": 0,
                                   "active_episode": None, "episodes": []})

    def begin(self, task_id, split, task_fingerprint, phase_budgets):
        if split not in {"learn", "probe"}:
            raise ValueError("Episode split must be learn or probe")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Need task ID")
        if not isinstance(task_fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", task_fingerprint):
            raise ValueError("Need task fingerprint")
        expected = {"solve", "learn", "validate"} if split == "learn" else {"solve"}
        if (not isinstance(phase_budgets, dict) or set(phase_budgets) != expected
                or any(type(n) is not int or n < 0 for n in phase_budgets.values()) or phase_budgets["solve"] == 0):
            raise ValueError("Invalid phase request budgets")
        with self.locked():
            state = self.read("run.json")
            if not state or state["active_episode"]:
                raise ValueError("Run missing or another episode is active")
            if state["reserved_requests"] + sum(phase_budgets.values()) > state["max_model_requests"]:
                raise ValueError("Run request budget exhausted")
            # Exact task reuse across learning and probe is blocked. Geographic
            # overlap and near-duplicates additionally require dataset auditing.
            for eid in state["episodes"]:
                prior = self.read(f"episodes/{eid}.json")
                if prior["split"] != split and (prior["task_id"] == task_id or prior["task_fingerprint"] == task_fingerprint):
                    raise ValueError("Task overlaps learning and probe")
            eid = uuid4().hex
            candidates = self.root / "candidates" / eid
            snapshots = CapabilitySnapshots(state["snapshots"])
            snapshots.materialize(state["current"], candidates)
            record = {"episode_id": eid, "task_id": task_id, "split": split, "task_fingerprint": task_fingerprint,
                      "base_snapshot": state["current"], "candidate_dir": str(candidates), "phase": "solve",
                      "phase_budgets": phase_budgets, "reports": {}}
            self.write(f"episodes/{eid}.json", record)
            state["active_episode"] = eid
            state["episodes"].append(eid)
            state["reserved_requests"] += sum(phase_budgets.values())
            self.write("run.json", state)
            self.event("evolution_episode_started", episode_id=eid, split=split, task_id=task_id)
        return record

    def finish_phase(self, episode_id, phase, report):
        """Host records actual usage/results, including failed phases.

The backend must enforce each reservation through its client request/step cap.
Reserved capacity is never refunded, so restart cannot create extra attempts.
"""
        if not isinstance(report, dict) or type(report.get("success")) is not bool:
            raise ValueError("Need structured phase result")
        usage = report.get("usage")
        if (not isinstance(usage, dict) or set(usage) != {"model_requests", "input_tokens", "output_tokens"}
                or any(type(n) is not int or n < 0 for n in usage.values())):
            raise ValueError("Need actual request/input/output token counts")
        if phase == "validate" and not isinstance(report.get("checks"), dict):
            raise ValueError("Validation needs named checks")
        with self.locked():
            state = self.read("run.json")
            record = self.read(f"episodes/{episode_id}.json")
            if not record or state["active_episode"] != episode_id or record["phase"] != phase:
                raise ValueError("Unexpected episode/phase")
            record["reports"][phase] = report
            over_budget = usage["model_requests"] > record["phase_budgets"][phase]
            if over_budget:
                record["phase"] = "budget_violation"
                state["active_episode"] = None
            elif record["split"] == "probe":
                record["phase"] = "completed"
                state["active_episode"] = None
            elif phase == "solve":
                record["phase"] = "learn"  # Failed attempts may still produce evidence.
            elif phase == "learn":
                no_update = report.get("updated") is False and report["success"]
                record["phase"] = "no_update" if no_update else "validate" if report["success"] else "discarded"
                if no_update or not report["success"]:
                    state["active_episode"] = None
            else:
                if report["success"] and report.get("checks") and all(v is True for v in report["checks"].values()):
                    record["phase"] = "ready_to_publish"
                else:
                    record["phase"] = "discarded"
                    state["active_episode"] = None
            self.write(f"episodes/{episode_id}.json", record)
            self.write("run.json", state)
            self.event("evolution_phase_finished", episode_id=episode_id, phase=phase,
                       report_sha256=digest(canonical(report).encode()), next_phase=record["phase"])
        return record

    def publish(self, episode_id):
        with self.locked():
            state = self.read("run.json")
            record = self.read(f"episodes/{episode_id}.json")
            if (not record or record["split"] != "learn" or record["phase"] != "ready_to_publish"
                    or state["active_episode"] != episode_id or state["current"] != record["base_snapshot"]):
                raise ValueError("Episode cannot publish")
            root = Path(record["candidate_dir"])
            snapshots = CapabilitySnapshots(state["snapshots"])
            snapshot = snapshots.current()
            if snapshot == record["base_snapshot"]:
                snapshot = snapshots.publish(ToolRegistry(root / "tools"), ContinualSkillRegistry(root / "skills"),
                                             parent=record["base_snapshot"], label=episode_id)
            else:
                # Recover if publication completed before the run checkpoint was written.
                committed = snapshots.snapshot(snapshot)
                if committed["parent"] != record["base_snapshot"] or committed["label"] != episode_id:
                    raise ValueError("Another episode changed the release")
            record.update(phase="completed", published_snapshot=snapshot)
            state.update(current=snapshot, active_episode=None)
            self.write(f"episodes/{episode_id}.json", record)
            self.write("run.json", state)
        return snapshot

    def abort(self, episode_id, reason):
        """Host recovery; retain all reports and reservations instead of auto-retrying."""
        with self.locked():
            state = self.read("run.json")
            if state["active_episode"] != episode_id:
                raise ValueError("Episode is not active")
            record = self.read(f"episodes/{episode_id}.json")
            record.update(phase="aborted", abort_reason=str(reason))
            state["active_episode"] = None
            self.write(f"episodes/{episode_id}.json", record)
            self.write("run.json", state)
