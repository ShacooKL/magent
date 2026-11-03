"""Host-owned trial snapshots, separate from legacy certified registries.

Execution checks are evidence with a scope, not task-utility certificates. The
host chooses tests and mounts; an Agent cannot publish by asserting success.
"""
from copy import deepcopy
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4

from rsagent.evolution.tool_registry import ToolRegistry, canonical, identifier, safe_bytes, strict_json
from rsagent.evolution.trial_execution import execute_tool


class TrialCapabilities:
    def __init__(self, root, recorder=None, *, timeout=30, runtime_prefix=None,
                 require_source_validation=False):
        # Reuse only the existing private storage, lock, atomic JSON and event IO.
        # No legacy activation policy, content hash or correctness gate is used.
        self._storage = ToolRegistry(root, recorder)
        self.root = self._storage.root
        self.timeout, self.runtime_prefix = timeout, runtime_prefix
        if type(require_source_validation) is not bool:
            raise ValueError("require_source_validation must be a boolean")
        with self._storage.locked():
            policy = self._storage.read("source-validation-policy.json", {})
            self.require_source_validation = require_source_validation or policy.get("required", False)
            if self.require_source_validation and not policy.get("required"):
                self._storage.write("source-validation-policy.json", {"required": True})

    def current(self):
        return self._storage.read("current.json", {}).get("snapshot")

    def _directory(self, snapshot):
        if not isinstance(snapshot, str) or not re.fullmatch(r"[a-f0-9]{32}", snapshot):
            raise ValueError("Unknown trial snapshot identifier")
        return self.root / "snapshots" / snapshot

    def snapshot(self, snapshot):
        return strict_json(safe_bytes(self._directory(snapshot) / "snapshot.json", 2 * 1024 * 1024).decode())

    def seed(self, baseline, initial_tools=()):
        """Register the frozen external baseline, without copying or altering it."""
        with self._storage.locked():
            if self.current() is not None:
                raise ValueError("This trial library already has a baseline")
            snapshot = uuid4().hex
            payload = {"schema": "trial_snapshot_v1", "snapshot": snapshot, "parent": None,
                       "baseline": baseline, "initial_tools": list(initial_tools),
                       "tools": {}, "skills": {}, "status": "baseline", "effectiveness_verified": False}
            self._storage.write(f"snapshots/{snapshot}/snapshot.json", payload)
            self._storage.write("current.json", {"snapshot": snapshot})
            self._storage.event("trial_baseline_registered", snapshot=snapshot, baseline=baseline)
            return snapshot

    def _execute(self, source, arguments, workspace, readonly=(), *, inspect_only=False):
        return execute_tool(source, arguments, workspace=workspace, readonly=readonly,
                            protected_roots=(self.root,), timeout=self.timeout,
                            runtime_prefix=self.runtime_prefix, inspect_only=inspect_only)

    def _check(self, source, case, readonly, origin):
        # Expected values never enter the sandbox. Test workspaces are separate,
        # including tests for the same tool, so files cannot mask later failures.
        with TemporaryDirectory(prefix="trial-check-") as temporary:
            result = self._execute(source, case["arguments"], Path(temporary), readonly)
        expected = "expected" in case
        passed = result["success"] and (not expected or canonical(result["value"]) == canonical(case["expected"]))
        return {"origin": origin, "case": deepcopy(case), "passed": passed,
                "checks_expected_value": expected, "execution": result}

    def publish(self, candidate, *, parent, agent_cases=None, independent_cases=None, readonly=()):
        """Inspect, optionally test, and atomically retain a whole proposed update.

        Cases are host-supplied maps: tool name -> [{arguments, expected?}]. Agent
        cases are never promoted to independent evidence. Absence of cases is
        explicit and does not prohibit an interface-only trial.
        """
        candidate = Path(candidate).resolve()
        proposal = strict_json(safe_bytes(candidate / "proposal.json", 2 * 1024 * 1024).decode())
        if proposal.get("schema") != "joint_update_v1":
            raise ValueError("Expected an archived joint_update_v1 candidate")
        agent_cases, independent_cases = agent_cases or {}, independent_cases or {}
        with self._storage.locked():
            if self.current() != parent:
                raise ValueError("Stale trial parent; no changes were published")
            previous = self.snapshot(parent)
            if proposal["base_snapshot"] not in {previous["baseline"], parent}:
                raise ValueError("Candidate evidence belongs to a different baseline")
            attempt = uuid4().hex
            folder = self.root / "attempts" / attempt
            folder.mkdir(parents=True)
            self._storage.write(f"attempts/{attempt}/proposal.json", proposal)
            report = {"attempt": attempt, "parent": parent, "source_baseline": proposal["base_snapshot"],
                      "published": False, "snapshot": parent, "status": "checking", "tools": {},
                      "errors": [], "effectiveness_verified": False, "retained": False}

            def finish(status):
                report["status"] = status
                self._storage.write(f"attempts/{attempt}/report.json", report)
                self._storage.event("trial_candidate_checked", attempt=attempt, parent=parent,
                                    status=status, errors=report["errors"])
                return report

            tools, skills = deepcopy(previous["tools"]), deepcopy(previous["skills"])
            changed_skills = {entry["name"] for entry in proposal["skills"]}
            changed_tools = {entry["name"] for entry in proposal["tools"] if entry["operation"] == "upsert"}
            if (set(agent_cases) | set(independent_cases)) - changed_tools:
                raise ValueError("Tests must name a tool upsert in this candidate")
            for kind, assets in (("tools", tools), ("skills", skills)):
                for entry in proposal[kind]:
                    name = identifier(entry["name"])
                    if kind == "tools" and name in previous["initial_tools"]:
                        report["errors"].append(f"Original tool {name} is frozen; use a new name")
                    elif entry["operation"] == "retire":
                        if name not in assets:
                            report["errors"].append(f"Cannot retire unknown {kind}: {name}")
                        else:
                            del assets[name]
                    elif entry["operation"] == "upsert":
                        assets[name] = {**entry, "updated_in": attempt}
                    else:
                        raise ValueError("Unknown update operation")
            available = set(previous["initial_tools"]) | set(tools)
            for name, skill in skills.items():
                missing = set(skill.get("requires_tools", [])) - available
                if missing:
                    report["errors"].append(f"Skill {name} has unavailable dependencies: {sorted(missing)}")
            if report["errors"]:
                return finish("rejected")
            if not proposal["tools"] and not proposal["skills"]:
                return finish("no_update")

            (folder / "tools").mkdir()
            for name, tool in tools.items():
                if name in changed_tools:
                    source = (candidate / tool["source"]).resolve(strict=True)
                    if not source.is_relative_to(candidate):
                        raise ValueError("Archived candidate source must stay inside its bundle")
                else:
                    source = self._directory(parent) / tool["source"]
                (folder / "tools" / f"{name}.py").write_bytes(safe_bytes(source))
                tool["source"] = f"tools/{name}.py"

            for name in sorted(changed_tools):
                source = folder / tools[name]["source"]
                with TemporaryDirectory(prefix="trial-interface-") as temporary:
                    interface = self._execute(source, {}, Path(temporary), readonly, inspect_only=True)
                checks = []
                if interface["success"]:
                    for origin, cases in (("agent_self_test", agent_cases.get(name, [])),
                                          ("independent_test", independent_cases.get(name, []))):
                        checks.extend(self._check(source, case, readonly, origin) for case in cases)
                retained = [key for key, skill in previous["skills"].items()
                            if key not in changed_skills and name in skill.get("requires_tools", [])]
                regression = []
                if retained and name in previous["tools"] and interface["success"]:
                    # Replay observed old examples for unchanged dependent Skills.
                    # A deliberate contract change can revise/retire those Skills
                    # in this same bundle; no exact-version binding is imposed.
                    old = previous["tools"][name]["validation"]
                    cases = {canonical(check["case"]): check["case"]
                             for check in [*old["checks"], *old["regression"]] if check["passed"]}
                    regression = [self._check(source, case, readonly, "retained_skill_regression")
                                  for case in cases.values()]
                passed = interface["success"] and all(c["passed"] for c in [*checks, *regression])
                validation = {"interface": interface, "checks": checks, "regression": regression,
                              "retained_skills": retained, "passed": passed,
                              "scope": "checked_examples" if checks or regression else "entrypoint_only",
                              "independent_test_count": sum(c["origin"] == "independent_test" for c in checks),
                              "effectiveness_verified": False}
                report["tools"][name] = validation
                tools[name]["validation"] = validation
                if not passed:
                    report["errors"].append(f"Tool {name} failed its interface or supplied example checks")
            if report["errors"]:
                return finish("rejected")

            pending = self.require_source_validation and bool(skills)
            payload = {"schema": "trial_snapshot_v1", "snapshot": attempt, "parent": parent,
                       "baseline": previous["baseline"], "initial_tools": previous["initial_tools"],
                       "source_task": proposal["task_id"], "source_baseline": proposal["base_snapshot"],
                       "tools": tools, "skills": skills,
                       "status": "pending_validation" if pending else "trial_snapshot",
                       "effectiveness_verified": False}
            if pending:
                payload["source_validation_required"] = True
            report.update(published=not pending, retained=not pending, snapshot=attempt, status=payload["status"])
            self._storage.write(f"attempts/{attempt}/report.json", report)
            self._storage.write(f"attempts/{attempt}/snapshot.json", payload)
            folder.rename(self._directory(attempt))
            if pending:
                self._storage.event("trial_snapshot_pending_validation", snapshot=attempt, parent=parent)
                return report
            # A crash before this replacement leaves the previous version usable;
            # a crash afterwards still leaves a complete, self-contained version.
            self._storage.write("current.json", {"snapshot": attempt})
            self._storage.event("trial_snapshot_published", snapshot=attempt, parent=parent)
            return report

    def assert_skill_reuse(self, snapshot, *, allow_pending_validation=False):
        """Host-only access check; ordinary providers cannot expose pending Skills."""
        payload = self.snapshot(snapshot)
        if not payload["skills"] or not (self.require_source_validation or payload.get("source_validation_required")):
            return
        audit = self._storage.read(f"snapshots/{snapshot}/source-validation.json")
        if audit and audit.get("promotion_completed") and audit["decision"].get("retained"):
            return
        if allow_pending_validation and payload.get("status") == "pending_validation" and audit is None:
            return
        raise ValueError("Skill snapshot has not passed host source validation")

    def resolve_source_validation(self, snapshot, results, *, policy):
        """Resolve one frozen host replay policy, never a model's self-assessment.

        The host supplies its planned session IDs and frozen result rows.
        Decisions (including incomplete/failed outcomes) are final for this
        candidate. Identical calls are idempotent; changing evidence needs a new
        candidate/version, not best-of-N selection on this gate.
        """
        if (not isinstance(policy, dict) or set(policy) != {"source_task", "session_ids"}
                or not isinstance(policy["source_task"], str) or not policy["source_task"]
                or not isinstance(policy["session_ids"], list) or not policy["session_ids"]
                or any(not isinstance(sid, str) or not sid for sid in policy["session_ids"])
                or len(set(policy["session_ids"])) != len(policy["session_ids"])):
            raise ValueError("Source validation requires one source task and nonempty unique planned session IDs")
        if not isinstance(results, list) or any(not isinstance(row, dict) for row in results):
            raise ValueError("Provide frozen host result rows")
        keys = ("id", "task_id", "snapshot", "status", "correct")
        observed = [{key: row.get(key) for key in keys} for row in results]
        observed.sort(key=canonical)
        policy = deepcopy(policy)
        with self._storage.locked():
            payload = self.snapshot(snapshot)
            if not payload.get("source_validation_required") or payload.get("source_task") != policy["source_task"]:
                raise ValueError("Candidate and source validation task do not match")
            relative = f"snapshots/{snapshot}"
            audit = self._storage.read(relative + "/source-validation.json")
            if audit is not None:
                if canonical(audit["policy"]) != canonical(policy) or canonical(audit["results"]) != canonical(observed):
                    raise ValueError("Source validation already resolved; do not replace its planned evidence")
                if audit.get("promotion_completed") or not audit["decision"]["retained"]:
                    return deepcopy(audit["decision"])
            else:
                ids = [row["id"] for row in observed]
                errors = []
                if len(ids) != len(set(map(canonical, ids))) or any(sid not in policy["session_ids"] for sid in ids):
                    errors.append("Unexpected or duplicate replay session IDs")
                if any(row["task_id"] != policy["source_task"] or row["snapshot"] != snapshot for row in observed):
                    errors.append("Replay result is not bound to this candidate and source task")
                incomplete = (len(ids) != len(policy["session_ids"]) or any(row["status"] != "completed" or row["correct"] is None
                                                  for row in observed))
                if any(row["status"] == "completed" and row["correct"] is not None and row["correct"] is not True
                       for row in observed):
                    errors.append("A completed replay did not receive a true host correctness result")
                if self.current() != payload["parent"]:
                    errors.append("Stale candidate parent; current library has changed")
                passed = not errors and not incomplete
                decision = {"snapshot": snapshot, "parent": payload["parent"], "published": passed,
                            "retained": passed, "status": "source_validated" if passed else
                            "validation_failed" if errors else "validation_incomplete",
                            "errors": errors, "source_task": policy["source_task"],
                            "required_results": len(policy["session_ids"]), "observed_results": len(observed),
                            "effectiveness_verified": False,
                            "scope": "Source-task replay admission only; not proof of transfer benefit or scientific correctness"}
                audit = {"schema": "skill_source_validation_v1", "policy": policy, "results": observed,
                         "decision": decision, "promotion_completed": False}
                # Journal the host decision before changing the live pointer.
                self._storage.write(relative + "/source-validation.json", audit)
            decision = audit["decision"]
            if decision["retained"] and self.current() not in {payload["parent"], snapshot}:
                raise ValueError("Cannot recover source validation after an unrelated library update")
            payload["status"] = decision["status"]
            self._storage.write(relative + "/snapshot.json", payload)
            report = self._storage.read(relative + "/report.json", {})
            report.update(decision)
            self._storage.write(relative + "/report.json", report)
            if decision["retained"]:
                self._storage.write("current.json", {"snapshot": snapshot})
                audit["promotion_completed"] = True
                self._storage.write(relative + "/source-validation.json", audit)
            self._storage.event("trial_source_validation_resolved", snapshot=snapshot,
                                status=decision["status"], retained=decision["retained"])
            return deepcopy(decision)

    def run(self, snapshot, name, arguments, *, workspace, readonly=()):
        payload = self.snapshot(snapshot)
        tool = payload["tools"][name]
        result = self._execute(self._directory(snapshot) / tool["source"], arguments, workspace, readonly)
        self._storage.event("trial_tool_result", snapshot=snapshot, name=name,
                            success=result["success"], exit_code=result["exit_code"],
                            timed_out=result["timed_out"], duration_ms=result["duration_ms"])
        return result

    def combine_independent(self, tool_library, tool_snapshot, skill_library, skill_snapshot, *, parent):
        """Host-only union AFTER independent learning, without a merging model.

        Learners retain their separate libraries; only a later solver sees this
        union. Existing validation is copied, not repeated or relabeled.
        """
        left, right = tool_library.snapshot(tool_snapshot), skill_library.snapshot(skill_snapshot)
        if self.require_source_validation and right["skills"]:
            raise ValueError("A combined Skill snapshot requires its own source replay; gated independent union is not supported")
        with self._storage.locked():
            previous = self.snapshot(parent)
            if self.current() != parent:
                raise ValueError("Stale combined snapshot parent")
            if left["skills"] or right["tools"]:
                raise ValueError("Independent inputs must be tool-only and skill-only")
            if any(p["baseline"] != previous["baseline"] or p["initial_tools"] != previous["initial_tools"]
                   for p in (left, right)):
                raise ValueError("Independent libraries must use the same external baseline")
            available = set(previous["initial_tools"]) | set(left["tools"])
            if any(set(skill.get("requires_tools", [])) - available for skill in right["skills"].values()):
                raise ValueError("Combined snapshot has unresolved Skill dependencies")
            snapshot = uuid4().hex
            payload = {**deepcopy(previous), "snapshot": snapshot, "parent": parent,
                       "tools": deepcopy(left["tools"]), "skills": deepcopy(right["skills"]),
                       "status": "trial_snapshot", "effectiveness_verified": False,
                       "independent_sources": {"tools": tool_snapshot, "skills": skill_snapshot}}
            with TemporaryDirectory(prefix=".combine-", dir=self.root) as temporary:
                staged = Path(temporary)
                (staged / "tools").mkdir()
                for name, tool in payload["tools"].items():
                    source = tool_library._directory(tool_snapshot) / tool["source"]
                    (staged / "tools" / f"{name}.py").write_bytes(safe_bytes(source))
                    tool["source"] = f"tools/{name}.py"
                (staged / "snapshot.json").write_text(canonical(payload) + "\n", encoding="utf-8")
                staged.rename(self._directory(snapshot))
            self._storage.write("current.json", {"snapshot": snapshot})
            self._storage.event("independent_snapshots_combined", snapshot=snapshot, parent=parent)
            return snapshot
