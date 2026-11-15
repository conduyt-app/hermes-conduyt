"""THE TOOL LOCK: a Conduyt Team Chat turn gets NO tool at all: no terminal, files, browser, code, network fetch, other
platforms, cron, delegation, memory or skill writes, MCP server or plugin tool. The allow-list is empty.

Hermes decides a platform's tools from ``platform_toolsets.<platform>`` (hermes_cli/tools_config.py ``_get_platform_tools``,
v0.23.1 :572-650). For a plugin platform with no saved list the default is the implicit ``hermes-<platform>`` bundle, which is
the FULL core set (toolsets.py :11-36, :335-352). Two more traps, verified against the installed source:
  * an explicit list without ``no_mcp`` still selects EVERY enabled MCP server (tools_config.py ``_merge_mcp_servers``);
  * every plugin toolset is on for every platform unless disabled for it (tools_config.py ``_enabled_plugin_toolsets`` :521-528).
So the lock resolves the toolset EXACTLY as Hermes does (``_get_platform_tools``), expands it to tool names exactly as the agent
does (``model_tools._select_tool_names``) and judges both levels. ``judge`` is pure (tested without Hermes); ``evaluate`` reads
the live config each time it is called (connect, every poll, every turn, every tool resolution).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

PLATFORM = "conduyt"
#: the line the paste message runs and every refusal names: no toolset at all, and no MCP server (an explicit list without
#: no_mcp still selects every enabled MCP server; an EMPTY list does too: measured, tools_config.py _merge_mcp_servers)
RECOMMENDED_LIST = ["no_mcp"]
RECOMMENDED_LINE = "hermes config set platform_toolsets.conduyt '[\"no_mcp\"]'"
#: what every Conduyt turn is built with (``toolsets_for_source``), whatever the config says
LOCKDOWN_LIST = ["no_mcp"]

ALLOWED_TOOLSETS: frozenset = frozenset()
ALLOWED_TOOLS: frozenset = frozenset()

FORBIDDEN_CLASSES: tuple[tuple[str, frozenset[str], tuple[str, ...]], ...] = (
    ("a terminal or shell", frozenset({"terminal", "process_manage"}), ()),
    ("reading or writing files", frozenset({"read_file", "write_file", "patch", "search_files"}), ()),
    ("a browser or computer use", frozenset({"computer_use", "browser_vault"}), ("browser_",)),
    ("running code", frozenset({"execute_code"}), ()),
    ("scheduled jobs", frozenset({"cronjob_manage"}), ()),
    ("delegating to other agents", frozenset({"delegate_task"}), ()),
    ("writing memory or skills", frozenset({"memory", "skill_manage"}), ()),
    ("searching your past sessions", frozenset({"session_search"}), ()),
    ("fetching the web", frozenset({"web_search", "web_extract"}), ()),
    ("messaging other platforms", frozenset({"send_message"}), ()),
)


def forbidden_class(tool: str) -> Optional[str]:
    for label, names, prefixes in FORBIDDEN_CLASSES:
        if tool in names or any(tool.startswith(p) for p in prefixes):
            return label
    return None


@dataclass
class Verdict:
    ok: bool
    problems: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    saved_list: Optional[list[str]] = None
    toolsets: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    other: bool = False   # the refusal is not the tool set's (hook compatibility, a memory provider): message() is the problem

    def message(self) -> str:
        """One plain paragraph for the log and Hermes's status: what is wrong and the exact line(s) that fix it."""
        if not self.ok and self.other:
            return ("Conduyt is not connected: " + "; ".join(self.problems) + ". Fix: "
                    + "  then  ".join(dict.fromkeys(self.fixes)) + "  (the plugin re-checks every 30 seconds).")
        if self.ok:
            return "Conduyt tool lock: OK (" + (", ".join(self.tools) if self.tools else "no tools") + ")"
        classes = list(dict.fromkeys(c for c in (forbidden_class(t) for t in self.tools) if c))
        others = [t for t in self.tools if not forbidden_class(t) and t not in ALLOWED_TOOLS]
        what = "; ".join(self.problems[:4]) + (f"; and {len(self.problems) - 4} more" if len(self.problems) > 4 else "")
        if classes:
            what = ", ".join(classes) + (f", and {len(others)} other tool(s) not on the allow-list" if others else "")
            if self.saved_list is None:
                what = "no tool list is saved for Conduyt, so Hermes would give its turns the full default set: " + what
        fixes = list(dict.fromkeys(self.fixes or [RECOMMENDED_LINE]))
        if len(fixes) > 4:
            fixes = fixes[:4] + [f"({len(fixes) - 4} more: run `hermes tools list --platform conduyt`)"]
        return ("Conduyt is not connected: its turns would get tools they must never have (" + what + "). Fix: "
                + "  then  ".join(fixes) + "  (the plugin re-checks every 30 seconds; no restart needed).")


def judge(saved: Any, toolsets: Iterable[str], tools: Iterable[str], mcp_servers: Any, conduyt_host: str = "") -> Verdict:
    """The lock's decision from what Hermes resolved. ``saved``: the raw ``platform_toolsets.conduyt`` value (after Hermes's own
    coercion; None when absent). ``toolsets``: ``_get_platform_tools`` for the platform. ``tools``: the tool names the agent
    would be built with. ``mcp_servers``: the config's ``mcp_servers`` mapping."""
    toolsets = sorted(set(toolsets))
    tools = sorted(set(tools))
    servers = mcp_servers if isinstance(mcp_servers, dict) else {}
    problems: list[str] = []
    fixes: list[str] = []
    if not isinstance(saved, list):
        problems.append("no tool list is saved for Conduyt, so Hermes would give its turns the full default set, terminal included")
        fixes.append(RECOMMENDED_LINE)
    for ts in toolsets:
        if ts in ALLOWED_TOOLSETS:
            continue
        name = ts[4:] if ts.startswith("mcp-") and ts[4:] in servers else ts
        if name in servers:
            problems.append(f"the MCP server {name!r}")
            fixes.append(RECOMMENDED_LINE)
        elif ts == "context_engine":
            problems.append("the context engine's tools")
            fixes.append("set context.engine back to compressor, or keep Conduyt disconnected")
        else:
            problems.append(f"the toolset {ts!r}")
            # a name the user saved is removed by saving the recommended list; a default-on plugin toolset needs Hermes to mark
            # it off for Conduyt first (``hermes tools disable`` rewrites the saved list without no_mcp, so the line follows it)
            fixes.append(RECOMMENDED_LINE if isinstance(saved, list) and ts in saved else f"hermes tools disable {ts} --platform conduyt")
    for tool in tools:
        if tool in ALLOWED_TOOLS:
            continue
        label = forbidden_class(tool)
        problems.append(f"{tool} ({label})" if label else f"{tool} (not on the Conduyt allow-list)")
        if not fixes:
            fixes.append(RECOMMENDED_LINE)
    if not isinstance(saved, list):
        fixes = [RECOMMENDED_LINE]
    elif fixes:
        fixes = [f for f in dict.fromkeys(fixes) if f != RECOMMENDED_LINE] + [RECOMMENDED_LINE]
    return Verdict(ok=not problems, problems=list(dict.fromkeys(problems)), fixes=fixes, saved_list=saved if isinstance(saved, list) else None,
                   toolsets=toolsets, tools=tools)


def tool_allowed(tool: str, verdict: Optional[Verdict] = None) -> bool:
    """The pre_tool_call guard's rule: no tool is ever allowed in a Conduyt turn."""
    return False


def evaluate(conduyt_host: str = "", config: Optional[dict] = None) -> Verdict:
    """The lock against the LIVE Hermes config (read fresh unless ``config`` is given). Any failure to resolve is a refusal."""
    try:
        from hermes_cli.tools_config import _coerce_platform_toolsets_value, _get_platform_tools
        from model_tools import _select_tool_names
        if config is None:
            from hermes_cli.config import load_config
            config = load_config()
        saved = _coerce_platform_toolsets_value((config.get("platform_toolsets") or {}).get(PLATFORM), PLATFORM)
        toolsets = _get_platform_tools(config, PLATFORM)
        disabled = (config.get("agent") or {}).get("disabled_toolsets")
        if disabled is not None and not isinstance(disabled, list):
            from agent.skill_utils import parse_config_string_list
            disabled = parse_config_string_list(disabled)
        tools = _select_tool_names(sorted(toolsets), disabled or None, True)
        return judge(saved, toolsets, tools, config.get("mcp_servers"), conduyt_host)
    except Exception as exc:  # fail closed
        logger.warning("Conduyt tool lock could not resolve Hermes's tools: %s", type(exc).__name__)
        return Verdict(ok=False, problems=[f"Hermes's tool configuration could not be read ({type(exc).__name__})"], fixes=[RECOMMENDED_LINE])
