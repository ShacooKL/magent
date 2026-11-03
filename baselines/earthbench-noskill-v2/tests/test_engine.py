import json
import tempfile
import unittest
from pathlib import Path

from magents.cli import DemoClient, EmptyTools
from magents.engine import Store, Team, BudgetExceeded, StageYield


def reply(text, tools):
    """Scripted model submissions, not production auto-conversion of prose."""
    names = {t["name"] for t in tools}
    value = None
    if "submit_plan" in names:
        name, value = "submit_plan", {"plan": text}
    elif "submit_progress" in names:
        name = "submit_progress"
        value = {"status": "complete", "completed": [text], "pending": [], "issues": [],
                 "artifacts": [], "evidence_ids": [1], "next_step": "review"}
    elif "submit_review" in names:
        name = "submit_review"
        verdict = "PASS" if text.startswith("PASS") else "INSUFFICIENT"
        value = {"verdict": verdict, "checked": [text], "issues": [],
                 "unchecked": [] if verdict == "PASS" else ["remaining checks"], "evidence_ids": [1]}
    elif "submit_final" in names:
        name = "submit_final"
        value = {"status": "completed", "answer": text, "reason": "scripted test", "evidence_ids": [1]}
    if value is not None:
        return {"content": [{"type": "tool_use", "id": "submit", "name": name, "input": value}]}
    return {"content": [{"type": "text", "text": text}]}


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "run"
        self.config = {"max_requests": 10, "max_role_requests": 3, "max_tokens": 128,
                       "tool_output_chars": 20, "review_rounds": 1}
        self.store = Store(self.root, "test task", {"mode": "test"})

    def test_full_workflow(self):
        state = Team(DemoClient(), EmptyTools(), self.store, self.config).run()
        self.assertEqual(state["status"], "partial")
        self.assertEqual(state["final_submission"]["status"], "incomplete")
        self.assertEqual(state["requests"], 5)
        self.assertEqual(len(list(self.root.iterdir())), 4)

    def test_shared_limit(self):
        self.config["max_requests"] = 2
        with self.assertRaises(BudgetExceeded):
            Team(DemoClient(), EmptyTools(), self.store, self.config).run()
        self.assertEqual(self.store.state["requests"], 2)
        self.assertEqual(self.store.state["status"], "budget_exhausted")

    def test_evidence_pagination(self):
        event = self.store.event("result", text="x" * 100)
        value = self.store.read(event, limit=10)
        self.assertEqual(value["next_offset"], 10)
        self.assertEqual(len(value["text"]), 10)

    def test_tool_trace_and_context_prefix(self):
        class Client:
            count = 0
            def create_message(inner, system, messages, tools, max_tokens):
                inner.count += 1
                if inner.count == 1:
                    inner.prefix = json.loads(json.dumps(messages))
                    return {"content": [{"type": "tool_use", "id": "1", "name": "read_file", "input": {}}]}
                self.assertEqual(messages[:len(inner.prefix)], inner.prefix)
                return {"content": [{"type": "text", "text": "done"}], "usage": {"prompt_tokens": 5}}
        class Tools:
            definitions = []
            def execute(inner, name, args):
                return "x" * 100
        team = Team(Client(), Tools(), self.store, self.config)
        self.assertEqual(team.ask("executor", "go"), "done")
        events = [json.loads(x) for x in (self.root / "events.jsonl").read_text().splitlines()]
        self.assertEqual(len(next(e for e in events if e["kind"] == "tool_result")["output"]), 100)
        self.assertEqual(self.store.state["usage"]["prompt_tokens"], 5)

    def test_failed_api_consumes_budget(self):
        class Client:
            def create_message(inner, *args, **kwargs):
                raise RuntimeError("network failure")
        with self.assertRaises(RuntimeError):
            Team(Client(), EmptyTools(), self.store, self.config).run()
        self.assertEqual(self.store.state["requests"], 1)
        self.assertEqual(self.store.state["status"], "failed")

    def test_revision_rechecked_before_final(self):
        class Client:
            replies = iter(["plan", "initial result", "ISSUES: missing date",
                            "REVISE include date", "corrected result", "PASS checked", "answer"])
            def create_message(inner, *args, **kwargs):
                return reply(next(inner.replies), args[2])
        team = Team(Client(), EmptyTools(), self.store, self.config)
        state = team.run()
        self.assertEqual(state["requests"], 7)
        self.assertEqual(json.loads(state["review"])["checked"], ["PASS checked"])
        self.assertEqual(json.loads(state["result"])["completed"], ["corrected result"])
        self.assertEqual(len(team.contexts["executor"]), 6)

    def test_truncated_tool_not_executed(self):
        class Client:
            count = 0
            def create_message(inner, *args, **kwargs):
                inner.count += 1
                if inner.count == 1:
                    return {"stop_reason": "length", "content": [
                        {"type": "tool_use", "id": "x", "name": "write_file", "input": {}}]}
                return {"content": [{"type": "text", "text": "short answer"}]}
        class Tools:
            definitions = []
            def execute(inner, *args):
                self.fail("Truncated tool call executed")
        self.assertEqual(Team(Client(), Tools(), self.store, self.config).ask("executor", "go"), "short answer")

    def test_yield_continue_preserves_context_and_does_not_repeat_tool(self):
        self.config["max_role_requests"] = 2
        snapshots = []
        class Client:
            count = 0
            def create_message(inner, system, messages, tools, max_tokens):
                inner.count += 1
                snapshots.append(json.loads(json.dumps(messages)))
                if inner.count == 1:
                    return {"content": [{"type": "tool_use", "id": "a", "name": "operation", "input": {}}]}
                if inner.count == 2:
                    response = reply("artifact created", tools)
                    response["content"][0]["input"].update(status="partial", pending=["finish"], next_step="finish")
                    return response
                return reply("CONTINUE finish remaining work" if inner.count == 3 else "finished", tools)
        class Tools:
            definitions = []
            calls = 0
            def execute(inner, *args):
                inner.calls += 1
                return "artifact created"
        tools = Tools()
        team = Team(Client(), tools, self.store, self.config)
        self.assertEqual(json.loads(team.work("executor", "work", reserve=2))["completed"], ["finished"])
        self.assertEqual(tools.calls, 1)
        self.assertEqual(snapshots[3][:len(snapshots[1])], snapshots[1])
        self.assertEqual(snapshots[3][2]["content"][0]["tool_use_id"], "a")
        self.assertEqual(self.store.state["result_status"], "returned")

    def test_stop_keeps_partial_evidence(self):
        self.config["max_role_requests"] = 1
        class Client:
            count = 0
            def create_message(inner, *args, **kwargs):
                inner.count += 1
                if inner.count == 1:
                    return {"content": [{"type": "tool_use", "id": "b", "name": "op", "input": {}}]}
                return {"content": [{"type": "text", "text": "STOP review incomplete work"}]}
        class Tools:
            definitions = []
            def execute(inner, *args):
                return "partial artifact"
        result = Team(Client(), Tools(), self.store, self.config).work("executor", "work")
        self.assertEqual(json.loads(result)["status"], "partial")
        self.assertEqual(len(json.loads(result)["evidence_ids"]), 2)

    def test_reserved_budget_reaches_review_and_final(self):
        self.config.update(max_requests=5, max_role_requests=1, downstream_reserve=3)
        class Client:
            count = 0
            def create_message(inner, *args, **kwargs):
                inner.count += 1
                if inner.count == 2:
                    return {"content": [{"type": "tool_use", "id": "c", "name": "op", "input": {}}]}
                text = {1: "plan", 3: "INSUFFICIENT partial work", 4: "Cannot complete calculation"}[inner.count]
                return reply(text, args[2])
        class Tools:
            definitions = []
            def execute(inner, *args):
                return "partial"
        state = Team(Client(), Tools(), self.store, self.config).run()
        self.assertEqual(state["status"], "partial")
        self.assertEqual(state["requests"], 4)
        self.assertIn("INSUFFICIENT", state["review"])
        self.assertTrue(state["answer"])

    def test_stage_yield_is_not_global_exhaustion(self):
        self.config["max_role_requests"] = 1
        class Client:
            def create_message(inner, *args, **kwargs):
                return {"stop_reason": "length", "content": []}
        with self.assertRaises(StageYield):
            Team(Client(), EmptyTools(), self.store, self.config).ask("executor", "work")
        self.assertEqual(self.store.state["requests"], 1)

    def test_plain_text_is_not_final_submission(self):
        class Client:
            def create_message(inner, *args, **kwargs):
                return {"content": [{"type": "text", "text": "<tool_code>submit_final</tool_code> done"}]}
        team = Team(Client(), EmptyTools(), self.store, self.config)
        with self.assertRaises(StageYield):
            team.ask("coordinator", "finish", text_only=True, submission="submit_final")
        self.assertNotIn("final_submission", self.store.state)
        self.assertEqual(self.store.state["answer"], "")

    def test_invalid_evidence_can_be_repaired(self):
        class Client:
            count = 0
            def create_message(inner, system, messages, tools, max_tokens):
                inner.count += 1
                response = reply("answer", tools)
                if inner.count == 1:
                    response["content"][0]["input"]["evidence_ids"] = [99999]
                return response
        team = Team(Client(), EmptyTools(), self.store, self.config)
        team.ask("coordinator", "finish", text_only=True, submission="submit_final")
        self.assertEqual(self.store.state["requests"], 2)
        self.assertEqual(self.store.state["final_submission"]["evidence_ids"], [1])
        events = [json.loads(x) for x in (self.root / "events.jsonl").read_text().splitlines()]
        self.assertEqual(len([e for e in events if e["kind"] == "submission"]), 1)

    def test_last_request_reserved_for_handoff_not_computation(self):
        self.config["max_role_requests"] = 1
        class Client:
            def create_message(inner, system, messages, tools, max_tokens):
                self.assertIn("Last request", messages[-1]["content"])
                return {"content": [{"type": "tool_use", "id": "a", "name": "op", "input": {}}]}
        class Tools:
            definitions = []
            def execute(inner, *args):
                self.fail("Reserved request executed computation")
        with self.assertRaises(StageYield) as caught:
            Team(Client(), Tools(), self.store, self.config).ask("executor", "go", submission="submit_progress")
        self.assertTrue(json.loads(caught.exception.progress)["handoff_missing"])

    def test_submission_cannot_be_combined_with_new_execution(self):
        self.config["max_role_requests"] = 1
        class Client:
            def create_message(inner, system, messages, tools, max_tokens):
                result = reply("finished", tools)
                result["content"].append({"type": "tool_use", "id": "op", "name": "op", "input": {}})
                return result
        class Tools:
            definitions = []
            def execute(inner, *args):
                self.fail("Reserved computation executed")
        with self.assertRaises(StageYield):
            Team(Client(), Tools(), self.store, self.config).ask("executor", "go", submission="submit_progress")
        self.assertNotIn("progress", self.store.state)

    def test_incomplete_final_is_an_explicit_submission(self):
        class Client:
            def create_message(inner, system, messages, tools, max_tokens):
                result = reply("", tools)
                result["content"][0]["input"].update(status="incomplete", reason="Missing calculation", evidence_ids=[])
                return result
        Team(Client(), EmptyTools(), self.store, self.config).ask(
            "coordinator", "finish", text_only=True, submission="submit_final")
        self.assertEqual(self.store.state["final_submission"]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
