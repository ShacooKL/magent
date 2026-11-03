import unittest
from unittest.mock import patch, MagicMock

from magents.adapters import WorkspaceTools
from magents.engine import Team


class OfficialOnlyTests(unittest.TestCase):
    def test_catalog_and_dispatch_block_code(self):
        with patch("magents.adapters.NamespaceRunner"), patch("magents.adapters.ToolBox") as box:
            box.return_value.definitions = [
                {"name": name} for name in
                ("list_files", "read_file", "run_command", "write_file", "edit_file")]
            tools = WorkspaceTools(None, [], {}, official_only=True)
            names = {t["name"] for t in tools.definitions}
            self.assertEqual(names, {"list_files", "read_file", "tool_search", "tool_inspect"})
            for name in ("run_command", "write_file", "edit_file"):
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    tools.execute(name, {})

    def test_all_roles_receive_restriction(self):
        team = Team(None, None, MagicMock(), {"official_only": True})
        for prompt in team.prompts.values():
            self.assertIn("Do not write or execute auxiliary code", prompt)
            self.assertNotIn("temporary code may", prompt)


if __name__ == "__main__":
    unittest.main()
