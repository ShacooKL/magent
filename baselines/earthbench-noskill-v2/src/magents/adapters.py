"""Reuse established API and sandbox adapters, not old experiment runners."""
import json
import os
from pathlib import Path

from rsagent.runtime.agent import (load_config, load_credentials, OpenAICompatibleClient,
                                   ToolBox, _tool, _string)
from rsagent.runtime.sandbox import NamespaceRunner
from rsagent.integrations.earth_tools import EarthToolProvider


def api_settings(path, profile):
    config = load_config(path, profile)
    if config.get("provider") != "openai":
        raise ValueError("This runner requires an OpenAI-compatible profile")
    return config


def client_for(path, config):
    credentials = {}
    if config.get("credentials_file"):
        location = Path(config["credentials_file"]).expanduser()
        credentials = load_credentials(location if location.is_absolute() else path.parent / location)
    key_name = config.get("api_key_env", "OPENAI_API_KEY")
    key = os.getenv(key_name) or credentials.get(key_name) or config.get("api_key")
    if not key:
        raise ValueError(f"Missing credential for {key_name}")
    return OpenAICompatibleClient(key, config["model"], config["base_url"],
                                  temperature=config.get("temperature"),
                                  enable_thinking=config.get("enable_thinking"),
                                  api_transport=config.get("api_transport"))


class WorkspaceTools:
    def __init__(self, workspace, data_paths, config, official=False, official_only=False, skill_dir=None, hypothesis_verification=False, source_workspace=None):
        self.workspace, self.earth = workspace, None
        self.official_only = official_only
        self.runner = NamespaceRunner(read_only_paths=data_paths, source_workspace=source_workspace)
        self.box = ToolBox(workspace, lambda _: True, self.runner, data_paths)
        if official:
            self.earth = EarthToolProvider(config["earth_tools"], workspace, data_paths, lambda _: True, source_workspace=source_workspace)
        self.definitions = list(self.box.definitions)
        if official_only:
            self.definitions = [t for t in self.definitions if t["name"] in {"list_files", "read_file"}]
        if self.earth:
            self.definitions.extend(self.earth.definitions)
        self.definitions.append(_tool("tool_search", "Find tools by words in their name or description; empty query lists all names.",
                                      {"query": _string("Keywords")}, ["query"]))
        self.definitions.append(_tool("tool_inspect", "Read exact declared tool input schema.",
                                      {"name": _string("Tool name")}, ["name"]))
        from magents.skills import SkillLibrary
        self.skills = SkillLibrary(skill_dir) if skill_dir else None
        if self.skills:
            self.definitions.extend(self.skills.definitions)
        from magents.hypotheses import Hypotheses
        self.hypotheses = Hypotheses(self.diagnostic_execute) if hypothesis_verification else None
        if self.hypotheses:
            self.definitions.extend(self.hypotheses.definitions)

    def diagnostic_execute(self, name, arguments):
        if name == 'run_command':
            if not self.earth:
                raise ValueError('Diagnostics require an official-tool environment')
            # Same mounts and Python prefix as the official bridge, fresh process.
            result = self.earth.runner.run(self.workspace, arguments['command'],
                {'PATH': str(self.earth.python.parent) + ':' + os.environ.get('PATH', '/usr/bin:/bin')}, timeout=120)
            return json.dumps({'stdout': result.stdout, 'exit_code': result.exit_code,
                               'timed_out': result.timed_out})
        if self.earth and name in self.earth.names:
            return self.earth.execute(name, arguments)
        if name in {'list_files', 'read_file', 'tool_inspect'}:
            return self.execute(name, arguments)
        raise ValueError('Diagnostic tests permit run_command, official tools, list_files, read_file or tool_inspect only')

    def execute(self, name, arguments):
        if self.official_only and name not in {t["name"] for t in self.definitions}:
            raise ValueError(f"Tool unavailable in official-only mode: {name}")
        if self.hypotheses and name in {t['name'] for t in self.hypotheses.definitions}:
            return self.hypotheses.call(name, arguments)
        if name in {"skill_search", "skill_read"} and self.skills:
            return self.skills.execute(name, arguments)
        if name == "tool_search":
            words = arguments["query"].lower().split()
            matches = [t for t in self.definitions if not words or any(
                word in (t["name"] + " " + t["description"]).lower() for word in words)]
            return json.dumps([{"name": t["name"], "description": t["description"]} for t in matches], ensure_ascii=False)
        if name == "tool_inspect":
            return json.dumps(next(t for t in self.definitions if t["name"] == arguments["name"]), ensure_ascii=False)
        if self.earth and name in self.earth.names:
            return self.earth.execute(name, arguments)
        if name == "run_command":
            result = self.runner.run(self.workspace, arguments["command"],
                                     {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, timeout=120)
            return json.dumps({"stdout": result.stdout, "exit_code": result.exit_code,
                               "timed_out": result.timed_out}, ensure_ascii=False)
        if name == "read_file":
            return self.box.resolve(arguments["path"]).read_text(encoding="utf-8")
        if name in {"list_files", "write_file", "edit_file"}:
            return getattr(self.box, name)(**arguments)
        raise ValueError(f"Unknown tool {name}")

    def close(self):
        if self.earth:
            self.earth.close()
