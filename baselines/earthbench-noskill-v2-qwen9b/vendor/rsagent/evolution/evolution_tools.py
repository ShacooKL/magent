"""Agent-facing tools for gated tool evolution (no skill extraction)."""
from pathlib import Path
import sys

from rsagent.evolution.tool_registry import ToolRegistry, canonical, safe_bytes


def disjoint(left, right):
    return not (left.is_relative_to(right) or right.is_relative_to(left))


class EvolutionToolProvider:
    def __init__(self, config, workspace, readonly, approve, recorder=None, registry=None):
        if set(config) - {"enabled", "registry_dir", "max_validations", "writable"}:
            raise ValueError("Unknown tool_evolution setting")
        self.workspace = Path(workspace).resolve()
        self.readonly = [Path(p).resolve() for p in readonly]
        self.approve = approve
        self.remaining = config.get("max_validations", 3)
        if type(self.remaining) is not int or not 1 <= self.remaining <= 10:
            raise ValueError("max_validations must be an integer in 1..10")
        root = Path(config["registry_dir"]).expanduser().resolve()
        exposed = [self.workspace, *self.readonly, Path(sys.prefix).resolve(),
                   *[Path(p).resolve() for p in ("/usr", "/bin", "/lib", "/lib64")]]
        if any(not disjoint(root, path) for path in exposed):
            raise ValueError("Registry must be outside workspace, references and sandbox runtime mounts")
        # An operator may inject a domain-specific gate. Never selected by model arguments.
        if registry is not None and registry.root != root:
            raise ValueError("Injected registry root differs from configured protected root")
        self.registry = registry if registry is not None else ToolRegistry(root, recorder)
        writable = config.get("writable", True)
        if type(writable) is not bool:
            raise ValueError("writable must be boolean")
        string = {"type": "string"}
        specifications = [
            ("evo_list", "Discover active versioned tools and names of operator validation contracts (including contracts with no initial tool). Expected outputs stay hidden.", {}, []),
            ("evo_checkout", "Copy active source to a workspace file for inspection/editing; stable source stays immutable.",
             {"name": string, "destination": string}, ["name", "destination"]),
            ("evo_propose", "Freeze a single .py candidate defining main(arguments)->JSON. Supply current parent revision when modifying a tool; omit for new tools. Does not activate code.",
             {"name": string, "source": string, "description": string, "parent": string}, ["name", "source", "description"]),
            ("evo_validate", "Run operator-owned JSON contract/regression tests in isolated workspaces. Cannot change tests. Validation budget is shared across this session.",
             {"name": string, "revision": string}, ["name", "revision"]),
            ("evo_activate", "Request activation; host requires current-policy validation and an unchanged parent. Does not self-certify correctness.",
             {"name": string, "revision": string}, ["name", "revision"]),
            ("evo_rollback", "Request a previously accepted version; host still requires current-policy validation.",
             {"name": string, "revision": string}, ["name", "revision"]),
            ("evo_run", "Execute an active versioned tool in the task sandbox. main(arguments) receives the JSON object; task workspace is /workspace and configured read-only inputs remain available. Returns JSON, not a scientific correctness certificate.",
             {"name": string, "arguments": {"type": "object"}}, ["name", "arguments"]),
        ]
        if not writable:
            specifications = [spec for spec in specifications if spec[0] in {"evo_list", "evo_checkout", "evo_run"}]
        self.definitions = [{"name": name, "description": description, "input_schema": {
            "type": "object", "properties": properties, "required": required,
            "additionalProperties": False}} for name, description, properties, required in specifications]
        self.names = {d["name"] for d in self.definitions}

    def workspace_path(self, value, missing=False):
        if not isinstance(value, str):
            raise ValueError("Expected workspace path string")
        path = Path(value)
        path = (self.workspace / path).resolve(strict=not missing)
        if not path.is_relative_to(self.workspace) or path == self.workspace:
            raise ValueError("Path must be inside writable task workspace")
        return path

    def execute(self, name, arguments):
        if name not in self.names:
            raise ValueError("Unknown evolution operation")
        schema = next(d["input_schema"] for d in self.definitions if d["name"] == name)
        if not isinstance(arguments, dict) or set(arguments) - set(schema["properties"]) or not set(schema["required"]) <= set(arguments):
            raise ValueError("Invalid operation arguments")
        for key, value in arguments.items():
            expected = schema["properties"][key]["type"]
            if (expected == "string" and not isinstance(value, str)) or (expected == "object" and not isinstance(value, dict)):
                raise ValueError("Invalid operation argument type")
        if name != "evo_list" and not self.approve(f"{name}: {arguments.get('name')} (isolated code; registry changes are gated)?"):
            raise PermissionError("Denied by user")
        a = arguments
        if name == "evo_list":
            result = {"tools": self.registry.discover(), "validation_contracts": self.registry.contracts(),
                      "remaining_validations": self.remaining}
        elif name == "evo_checkout":
            with self.registry.locked():
                entry = self.registry.index().get(a["name"])
                if not entry:
                    raise ValueError("No active tool")
                manifest, source = self.registry.revision(a["name"], entry["active"])
                destination = self.workspace_path(a["destination"], missing=True)
                if destination.exists():
                    raise ValueError("Checkout destination exists; use a new file")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(safe_bytes(source))
                result = {**manifest, "destination": str(destination.relative_to(self.workspace))}
        elif name == "evo_propose":
            source = self.workspace_path(a["source"])
            if source.suffix != ".py":
                raise ValueError("Candidate must be a .py file")
            result = self.registry.propose(a["name"], source, a["description"], a.get("parent"))
        elif name == "evo_validate":
            if self.remaining <= 0:
                raise ValueError("Validation budget exhausted")
            self.remaining -= 1
            result = self.registry.validate(a["name"], a["revision"])
        elif name in {"evo_activate", "evo_rollback"}:
            result = self.registry.activate(a["name"], a["revision"], rollback=name == "evo_rollback")
        else:
            result = self.registry.run(a["name"], a["arguments"], self.readonly, self.workspace)
            if not result["success"]:
                raise RuntimeError("Versioned tool execution failed: " + canonical(result))
        return canonical(result)


class CompositeToolProvider:
    """Compose existing Earth tools and versioned tools without changing their contracts."""
    def __init__(self, providers):
        self.providers = [p for p in providers if p is not None]
        self.definitions = [d for p in self.providers for d in p.definitions]
        self.names = {d["name"] for d in self.definitions}
        if len(self.names) != len(self.definitions):
            raise ValueError("Duplicate external tool names")
        # Earth tools have extra internal mounts beyond the task's read-only inputs.
        # A peer provider must not accidentally expose another provider's registry.
        for owner in self.providers:
            if not hasattr(owner, "registry"):
                continue
            for peer in self.providers:
                mounts = getattr(getattr(peer, "runner", None), "read_only_paths", [])
                if any(not disjoint(owner.registry.root, Path(path).resolve()) for path in mounts):
                    raise ValueError("Another tool provider would expose the private registry")

    def execute(self, name, arguments):
        for provider in self.providers:
            if name in provider.names:
                return provider.execute(name, arguments)
        raise ValueError("Unknown external tool")
