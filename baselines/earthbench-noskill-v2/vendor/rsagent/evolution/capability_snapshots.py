"""Host-only, content-addressed joint Tool/Skill releases.

A release captures validated code, evidence, policies and reports. Registries are
restored into new private directories for each episode, never edited in place.
"""
from contextlib import ExitStack
from pathlib import Path
import re

from rsagent.evolution.continual_skill import ContinualSkillRegistry
from rsagent.evolution.tool_registry import ToolRegistry, canonical, digest, safe_bytes


class CapabilitySnapshots(ToolRegistry):
    def current(self):
        return self.read("current.json", {}).get("snapshot")

    def publish(self, tools, skills, *, parent, label):
        if not isinstance(label, str) or not label or len(label) > 300:
            raise ValueError("Need release label")
        if not isinstance(skills, ContinualSkillRegistry):
            raise ValueError("Joint releases require decision_v3 skills")
        roots = [self.root, tools.root, skills.root]
        if any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]):
            raise ValueError("Release and registry directories must be disjoint")
        with ExitStack() as stack:
            for registry in sorted((self, tools, skills), key=lambda r: str(r.root)):
                stack.enter_context(registry.locked())
            if self.current() != parent:
                raise ValueError("Stale release parent")
            tool_records, skill_records, available = [], [], {}
            for name, entry in sorted(tools.index().items()):
                revision = entry["active"]
                manifest, source = tools.revision(name, revision)
                policy = tools.read(f"policies/{name}.json")
                validation = tools.read(f"validations/{name}/{revision}.json", {})
                if (not policy or validation.get("success") is not True
                        or validation.get("sha256") != manifest["sha256"]
                        or validation.get("policy_hash") != digest(canonical(policy).encode())):
                    raise ValueError(f"Tool {name} lacks current validation")
                available[name] = {k: manifest[k] for k in ("name", "revision", "sha256")}
                tool_records.append({"manifest": manifest, "source": safe_bytes(source).decode(),
                                     "policy": policy, "validation": validation})
            for name, entry in sorted(skills.index().items()):
                if entry.get("active") is None:
                    continue
                revision = entry["active"]
                manifest, _, contract = skills.behavior_report(name, revision)
                if not skills.compatible(manifest, available):
                    raise ValueError(f"Skill {name} needs dependency revalidation or retirement")
                source = safe_bytes(skills.root / "versions" / name / revision / "skill.json").decode()
                skill_records.append({"manifest": manifest, "source": source, "contract": contract,
                                      "validation": skills.read(f"validations/{name}/{revision}.json"),
                                      "behavior": skills.read(f"behavior/{name}/{revision}.json")})
            payload = {"schema": "capability_snapshot_v1", "parent": parent, "label": label,
                       "tools": tool_records, "skills": skill_records}
            encoded = canonical(payload).encode()
            if len(encoded) > 2 * 1024 * 1024 - 1:
                raise ValueError("Pilot release exceeds 2 MiB limit")
            snapshot = digest(encoded)
            self.write(f"snapshots/{snapshot}.json", payload)
            self.write("current.json", {"snapshot": snapshot})
            self.event("capability_published", snapshot=snapshot, parent=parent, label=label)
        return snapshot

    def snapshot(self, snapshot):
        if not isinstance(snapshot, str) or not re.fullmatch(r"[a-f0-9]{64}", snapshot):
            raise ValueError("Invalid snapshot identity")
        payload = self.read(f"snapshots/{snapshot}.json")
        if payload is None or digest(canonical(payload).encode()) != snapshot:
            raise ValueError("Unknown or changed snapshot")
        return payload

    def materialize(self, snapshot, target):
        payload = self.snapshot(snapshot)
        target = Path(target).resolve()
        if target.is_relative_to(self.root) or self.root.is_relative_to(target):
            raise ValueError("Candidate registries must be outside release storage")
        target.mkdir(parents=True, exist_ok=False)
        tools = ToolRegistry(target / "tools")
        skills = ContinualSkillRegistry(target / "skills")
        for kind, registry in (("tools", tools), ("skills", skills)):
            index = {}
            for record in payload[kind]:
                manifest = record["manifest"]
                name, revision = manifest["name"], manifest["revision"]
                if digest(record["source"].encode()) != manifest["sha256"]:
                    raise ValueError("Snapshot source hash differs")
                base = f"versions/{name}/{revision}"
                registry.write(base + "/manifest.json", manifest)
                filename = "tool.py" if kind == "tools" else "skill.json"
                (registry.root / base / filename).write_text(record["source"], encoding="utf-8")
                registry.write(f"validations/{name}/{revision}.json", record["validation"])
                index[name] = {"active": revision, "accepted": [revision]}
                if kind == "tools":
                    registry.write(f"policies/{name}.json", record["policy"])
                else:
                    registry.write(f"contracts/{manifest['contract_hash']}.json", record["contract"])
                    registry.write(f"behavior/{name}/{revision}.json", record["behavior"])
                    index[name]["status"] = "development_checked"
            registry.write("index.json", index)
        return tools, skills
