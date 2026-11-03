"""Dependency-free terminal UI for the coding agent."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
from threading import Event, Thread
import time
from typing import Any, Iterator, TextIO


class TerminalUI:
    def __init__(self, *, verbose: bool = False, show_tools: bool = True, stream: TextIO = sys.stdout) -> None:
        self.verbose = verbose
        self.show_tools = show_tools
        self.stream = stream
        self.color = bool(getattr(stream, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ
        self.approve_all = False

    def _style(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def _print(self, text: str = "") -> None:
        print(text, file=self.stream, flush=True)

    def banner(self, model: str, workspace: Path, sandbox: str, read_only_paths: list[Path] | None = None, profile: str | None = None) -> None:
        width = min(max(shutil.get_terminal_size((88, 24)).columns, 60), 100)
        title = " Py Agent "
        top = "╭" + title + "─" * max(0, width - len(title) - 2) + "╮"
        bottom = "╰" + "─" * (width - 2) + "╯"
        self._print(self._style(top, "1;36"))
        self._print(f"│ profile    {profile or 'default'}")
        self._print(f"│ model      {self._style(model, '1')}" )
        self._print(f"│ workspace  {workspace}")
        for path in read_only_paths or []:
            self._print(f"│ read-only  {path}")
        self._print(f"│ sandbox    {sandbox} · network disabled")
        self._print(self._style(bottom, "1;36"))
        self._print(self._style("输入任务开始，/help 查看命令。写文件和执行命令默认需要确认。", "2"))
        self._print()

    def prompt(self, context: dict[str, Any] | None = None) -> str:
        if context is not None:
            self.context_bar(context)
        label = self._style("❯ ", "1;36")
        return input(label)

    def context_bar(self, context: dict[str, Any]) -> None:
        last = self._format_compression_time(context.get("last_compressed_at"))
        text = (
            f"上下文 {context['chars']:,}/{context['budget']:,} chars ({context['percent']}%)"
            f" · 压缩 {context['compression_count']} 次 · 最近 {last}"
        )
        self._print(self._style(text, "2;36"))

    def context_compacted(self, mode: str, turns: int, context: dict[str, Any]) -> None:
        self.info(
            f"{mode}压缩完成：已压缩 {turns} 个 turn；"
            f"上下文 {context['chars']:,}/{context['budget']:,} chars ({context['percent']}%)"
        )

    def approve(self, question: str) -> bool:
        if self.approve_all:
            return True
        self._print()
        self._print(self._style("╭─ 需要确认", "1;33"))
        for line in question.splitlines():
            self._print(f"│ {line}")
        answer = input(self._style("╰─ 允许？[y/N/a=本次会话全部允许] ", "1;33")).strip().lower()
        if answer == "a":
            self.approve_all = True
            self._print(self._style("  已允许本次会话后续操作", "33"))
            return True
        return answer in {"y", "yes"}

    @contextmanager
    def thinking(self) -> Iterator[None]:
        if not self.color:
            yield
            return
        stop = Event()
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

        def animate() -> None:
            index = 0
            while not stop.wait(0.08):
                text = self._style(f"{frames[index % len(frames)]} 正在思考…", "36")
                print(f"\r{text}", end="", file=self.stream, flush=True)
                index += 1

        thread = Thread(target=animate, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=0.2)
            print("\r\033[2K", end="", file=self.stream, flush=True)

    def assistant(self, text: str) -> None:
        self._print()
        self._print(self._style("◆ Agent", "1;32"))
        self._print(text.rstrip())

    def tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        if not self.show_tools:
            return
        detail = self._tool_detail(name, arguments)
        self._print(self._style(f"  ↳ {name}", "1;34") + (f"  {detail}" if detail else ""))
        if self.verbose and detail != _short_json(arguments, 500):
            self._print(self._style(f"    {_short_json(arguments, 500)}", "2"))

    def tool_result(self, name: str, output: str, is_error: bool) -> None:
        if not self.show_tools:
            return
        icon = "✗" if is_error else "✓"
        color = "31" if is_error else "32"
        first_line = next((line.strip() for line in output.splitlines() if line.strip()), "no output")
        self._print(self._style(f"    {icon} {first_line[:160]}", color))
        if self.verbose:
            lines = [line for line in output.strip().splitlines() if line.strip()]
            if len(lines) > 1:
                for line in "\n".join(lines[1:])[:1_200].splitlines():
                    self._print(self._style(f"      {line}", "2"))

    def info(self, text: str) -> None:
        self._print(self._style(f"● {text}", "36"))

    def success(self, text: str) -> None:
        self._print(self._style(f"✓ {text}", "32"))

    def warning(self, text: str) -> None:
        self._print(self._style(f"⚠ {text}", "33"))

    def error(self, text: str) -> None:
        self._print(self._style(f"✗ {text}", "31"))

    def help(self) -> None:
        self._print("""
可用命令
  /help      显示帮助
  /status    查看模型、沙箱和上下文状态
  /history   查看本次会话的任务历史
  /sessions  查看当前 workspace 保存的会话
  /compact   主动压缩上下文（尽量保留最近一个完整任务）
  /logs      显示当前运行日志
  /tools     切换工具调用显示；也可使用 /tools on 或 /tools off
  /paste     输入多行任务，以单独一行 . 结束
  /clear     清除对话上下文
  /exit      退出

确认操作时输入 a，可以批准本次会话后续所有写入和命令。""".strip())

    def status(self, *, model: str, workspace: Path, sandbox: str, turns: int, summaries: int, chars: int, budget: int, percent: int, compression_count: int, last_compressed_at: str | None, read_only_paths: list[Path] | None = None, session_id: str | None = None, profile: str | None = None) -> None:
        last = self._format_compression_time(last_compressed_at)
        references = ", ".join(str(path) for path in read_only_paths or []) or "none"
        self._print(f"""
当前状态
  session    {session_id or 'not persisted'}
  profile    {profile or 'default'}
  model      {model}
  workspace  {workspace}
  read-only  {references}
  sandbox    {sandbox}
  context    {turns} turns · {summaries} summaries · {chars:,}/{budget:,} chars ({percent}%)
  compact    {compression_count} times · last {last}
  tools      {'visible' if self.show_tools else 'hidden'} · {'verbose' if self.verbose else 'compact'}
  approvals  {'session auto-approve' if self.approve_all else 'ask before side effects'}""".rstrip())

    @staticmethod
    def _format_compression_time(value: str | None) -> str:
        if not value:
            return "从未"
        return value.replace("T", " ").removesuffix("+08:00") + " 北京时间"

    def set_tools_visible(self, enabled: bool) -> None:
        self.show_tools = enabled
        self.success(f"工具调用显示已{'开启' if enabled else '关闭'}")

    def history(self, turns: list[list[dict[str, Any]]]) -> None:
        if not turns:
            self.info("当前会话还没有历史任务")
            return
        self._print("\n任务历史")
        for index, turn in enumerate(turns, 1):
            prompt = next((message.get("content", "") for message in turn if message.get("role") == "user" and isinstance(message.get("content"), str)), "")
            self._print(f"  {index:>2}. {str(prompt).replace(chr(10), ' ')[:100]}")

    def sessions(self, rows: list[dict[str, Any]], current_session_id: str) -> None:
        if not rows:
            self.info("当前 workspace 还没有已保存的会话")
            return
        self._print("\n已保存会话")
        for row in rows:
            marker = "*" if row["session_id"] == current_session_id else " "
            updated = str(row.get("updated_at", "")).replace("T", " ")[:19]
            self._print(
                f" {marker} {row['session_id']} · {row['turns']} turns · "
                f"{row['summaries']} summaries · {updated}"
            )

    def multiline_prompt(self) -> str:
        self.info("输入多行任务；以单独一行 . 结束")
        lines: list[str] = []
        while True:
            line = input("│ ")
            if line == ".":
                return "\n".join(lines).strip()
            lines.append(line)

    @staticmethod
    def _tool_detail(name: str, arguments: dict[str, Any]) -> str:
        if name in {"read_file", "write_file", "edit_file", "list_files"}:
            path = arguments.get("path", ".")
            if name == "write_file":
                return f"{path} · {len(arguments.get('content', ''))} chars"
            if name == "edit_file":
                return f"{path} · {len(arguments.get('old_text', ''))}→{len(arguments.get('new_text', ''))} chars"
            return str(path)
        if name == "run_command":
            return str(arguments.get("command", ""))[:180]
        return _short_json(arguments, 180)


def _short_json(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "…"
