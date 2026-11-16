"""THE MACHINE BOUNDARY: a Conduyt Team Chat turn never sees, and never posts, facts about the computer Hermes runs on.

CLOSED RULES (what the plugin guarantees; nothing here depends on guessing what a message asks for):

(A) INPUT. The model request of a Conduyt turn carries no fact about this machine. By construction: every SYSTEM carrier of
    the request (``messages`` / ``input`` items with a system or developer role, ``system``, ``instructions``) that holds this
    agent's prompt is rebuilt from an allow-list (``scrub_system_text``): the stable tier (persona, the plugin's Team Chat
    rule, guidance; minus the environment probe line), the agent's own memory blocks (matched byte for byte) and what follows
    the profile line (date, model lines, the gateway's ephemeral prompt) minus the runtime environment block. Project context
    files, other plugins' prompt sections, the profile line and the runtime block are dropped. Then EVERY system carrier
    (rebuilt or not) passes ONE redactor (``redact``): local path shapes and this machine's own concrete values
    (``MachineFacts``: home folder, user name, host name and its short form, Hermes home, working and scratch folders, OS
    version strings). After that no system carrier may hold one of Hermes's machine-detail markers, and the exact Team Chat
    rule must be present verbatim; anything else raises ScrubError and the turn's answer is never posted. User and assistant
    messages (the teammates' words, the thread's history) are NOT touched or checked here (rule E).
(B) FINAL PAYLOAD: enforced in the adapter (the scrub must be the only request middleware and nothing may rewrite the
    request after it).
(C) OUTPUT (``guard``): ONE deterministic filter over every answer the adapter would post: an ``ls -l`` line or a shell
    prompt -> REFUSAL; then the same redactor as (A) (path shapes -> PATH_MARK, the machine's values -> MACHINE_MARK). A secret
    shape is refused by the adapter before this runs. ``asks_to_run`` is only a HINT that swaps a non-refusal answer for
    REFUSAL when the message names a local target or a command with arguments; it is NOT part of the safety contract.
(D) HISTORY: enforced in the adapter (an answer the filter changed rolls the thread to a fresh Hermes session).
(E) Marker checks apply to system carriers only, never to user or assistant history.

A short machine value that is also an ordinary word (a user name such as "paul", a host name such as "studio") is removed
only in machine shapes (``paul@x``, ``x@studio``, ``studio.local``): removing it everywhere would shred the owner's own
name out of persona, memory and prose. A DISTINCTIVE value (by character class
only: a digit, ``-``, ``_`` or ``.``, or an inner capital) is removed everywhere, case-insensitive, at word boundaries.

Every match (paths, schemes, this machine's exact paths, its values) is made on ONE PROJECTION of the text (``project``):
Unicode NFKC per character, format characters (zero-width, bidi controls, word joiners) dropped, Markdown markers
``*`` `` ` `` ``_`` and ``~~`` and the backslash that escapes them dropped, with an index map back to the original; the
ORIGINAL span is what is removed. So ``/Us**er**s/x``, a fullwidth slash or a zero-width joiner inside a value cannot split
a match.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Iterable, Optional

# ------------------------------------------------------------------------------------------------------- Hermes's own markers
#: agent/prompt_builder.py build_context_files_prompt (its fixed opening)
PROJECT_CONTEXT_HEADER = "# Project Context\n\nThe following project context files have been loaded and should be followed:"
#: agent/prompt_builder.py RUNTIME_ENVIRONMENT_HEADING / RUNTIME_ENVIRONMENT_END (renderer-owned: Hermes escapes the heading
#: anywhere else, agent/system_prompt.py :805)
RUNTIME_HEADING = "# Hermes runtime environment"
RUNTIME_END = "<!-- End Hermes runtime environment -->"
#: agent/system_prompt.py _active_profile_line (always rendered, after every block of owner prose, before the date line)
PROFILE_PREFIX = "Active Hermes profile: "
#: tools/env_probe.py's toolchain line (its fixed opening): removed by shape too, whatever the probe's cached text is now
PROBE_PREFIX = "Python toolchain: "
#: gateway/session.py build_session_context_prompt: the gateway's "Current Session Context" block in the ephemeral system
#: prompt. It carries PARTICIPANT-controlled text (the channel name, display names) and owner details that are not persona
#: or memory (connected platforms, home channels, the Hermes home's cron folder), so it is removed whole: the plugin's own
#: turn text already names the sender and the channel, as user-tier data
SESSION_CONTEXT_START = "## Current Session Context"
SESSION_CONTEXT_END = '*For explicit targeting, use `"platform:chat_id"` format if the user provides a specific chat ID.*'
#: every marker that must never reach a Conduyt request's system carriers (the check after the scrub)
LEFTOVER_MARKERS = (PROJECT_CONTEXT_HEADER, RUNTIME_HEADING, RUNTIME_END, PROFILE_PREFIX, PROBE_PREFIX, SESSION_CONTEXT_START)

SYSTEM_ROLES = frozenset({"system", "developer"})


class ScrubError(ValueError):
    pass


def _collapse(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def scrub_system_text(text: str, stable: str, memory_blocks: Iterable[str] = (), probe_line: str = "") -> str:
    """The system prompt of a Conduyt request, rebuilt from an ALLOW-LIST of its own parts (see the module docstring). ``stable``
    is Hermes's stable tier for this agent (``_cached_system_prompt_static``), which must open ``text``; ``memory_blocks`` are
    Hermes's own memory blocks for this agent, each of which must appear between the stable tier and the profile line;
    ``probe_line`` is Hermes's environment probe line ("" when the host is clean). Raises ScrubError on any other shape."""
    if not isinstance(text, str) or not isinstance(stable, str) or not stable or not text.startswith(stable):
        raise ScrubError("the system prompt does not open with this agent's stable part")
    head, rest = stable, text[len(stable):]
    at = rest.rfind("\n\n" + PROFILE_PREFIX)
    if at < 0:
        raise ScrubError("the system prompt has no profile line")
    before, after = rest[:at], rest[at + 2:]
    end = after.find("\n\n")
    after = after[end + 2:] if end >= 0 else ""
    kept = []
    for block in memory_blocks:
        block = (block or "").strip()
        if not block:
            continue
        if block not in before:
            raise ScrubError("a memory block of this agent is not where Hermes puts it")
        kept.append(block)
    while SESSION_CONTEXT_START in after:   # the gateway's session context block: participant text, removed whole
        a0 = after.find(SESSION_CONTEXT_START)
        b0 = after.find(SESSION_CONTEXT_END, a0)
        if b0 < 0:
            raise ScrubError("the session context block has no end marker")
        after = after[:a0] + after[b0 + len(SESSION_CONTEXT_END):]
    i = after.rfind(RUNTIME_HEADING)
    if i >= 0:
        j = after.find(RUNTIME_END, i)
        if j < 0:
            raise ScrubError("the runtime environment block has no end marker")
        after = after[:i] + after[j + len(RUNTIME_END):]
    if probe_line and probe_line.strip():
        head = head.replace(probe_line.strip(), "")
    head = re.sub(r"(?m)^" + re.escape(PROBE_PREFIX) + r"[^\n]*$", "", head)   # any toolchain line, by its shape
    out = "\n\n".join(p for p in (_collapse(head), *kept, _collapse(after)) if p)
    problems = leftover_markers(out) + (["the environment probe line"] if probe_line.strip() and probe_line.strip() in out else [])
    if problems:
        raise ScrubError("local machine details remain after the scrub (" + ", ".join(problems) + ")")
    return out


def _walk_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)


def leftover_markers(obj: Any) -> list[str]:
    """Which of Hermes's machine-detail markers appear ANYWHERE in ``obj`` (every string, at any depth)."""
    found = []
    for s in _walk_strings(obj):
        for m in LEFTOVER_MARKERS:
            if m in s and m not in found:
                found.append(m)
    return [m.split("\n", 1)[0].strip("# ").strip() for m in found]


def _text_parts(value: Any) -> Optional[list]:
    """A system value's text holders: [("str", None)] for a plain string, or the list of part dicts that carry "text"."""
    if isinstance(value, str):
        return []
    if isinstance(value, list) and all(isinstance(p, dict) for p in value):
        return [p for p in value if isinstance(p.get("text"), str)]
    return None


def _scrub_value(value: Any, fn) -> tuple[Any, bool]:
    """(the scrubbed value, whether it held this agent's prompt). Plain string, or a list of content parts (Anthropic / Bedrock
    system blocks, chat content parts): the parts' text is joined, scrubbed once, and kept in the FIRST text part; the other
    text parts are removed (their cache markers move to the first one); non-text parts are kept."""
    if isinstance(value, str):
        return (fn(value), True) if _opens(value, fn) else (value, False)
    parts = _text_parts(value)
    if not parts:
        return value, False
    full = "".join(p["text"] for p in parts)
    if not _opens(full, fn):
        return value, False
    first = parts[0]
    first["text"] = fn(full)
    for p in parts[1:]:
        for k in ("cache_control",):
            if k in p and k not in first:
                first[k] = p[k]
    dropped = {id(p) for p in parts[1:]}
    return [p for p in value if id(p) not in dropped], True


def _opens(text: str, fn) -> bool:
    return isinstance(text, str) and text.startswith(fn.stable)


def _carriers(out: dict):
    """(container, key) of every SYSTEM carrier of a provider request: ``system`` (Anthropic Messages, Bedrock Converse),
    ``instructions`` (Responses), and the ``content`` of ``messages`` / ``input`` items with a system or developer role."""
    for key in ("system", "instructions"):
        if key in out:
            yield out, key
    for key in ("messages", "input"):
        items = out.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("role") in SYSTEM_ROLES and "content" in item:
                    yield item, "content"


def _redact_value(value: Any, facts: "Optional[MachineFacts]") -> Any:
    """EVERY string inside a system carrier, at any depth (text parts, nested lists or mappings of any shape)."""
    if isinstance(value, str):
        return redact(value, facts)[0]
    if isinstance(value, list):
        return [_redact_value(v, facts) for v in value]
    if isinstance(value, dict):
        return {k: _redact_value(v, facts) for k, v in value.items()}
    return value


#: stands in for the Team Chat rule while a carrier is redacted: private-use characters only, so no path shape and no
#: machine value (every value holds a word character) can match it
_RULE_SENTINEL = "\ue000\ue001\ue002"


def _protected_redact(value: Any, facts: "Optional[MachineFacts]", required: str) -> Any:
    """The redactor with the Team Chat rule as an ATOMIC span: it is swapped for a sentinel, the rest is redacted, then it is
    restored. A machine value that projects onto part of the rule (a user named ``no_reply`` vs the rule's ``NO_REPLY``)
    therefore cannot damage it; the rule is fixed plugin text, never a machine fact."""
    if not required:
        return _redact_value(value, facts)

    def walk(v):
        if isinstance(v, str):
            if _RULE_SENTINEL in v:
                raise ScrubError("the system prompt holds the plugin's internal sentinel")
            parts = v.split(required)
            return required.join(redact(p, facts)[0] for p in parts)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        return v
    return walk(value)


def content_free_request(request: Any, rule: str) -> dict:
    """What a Conduyt request becomes when it cannot be cleared: the SAME request shape with every content carrier replaced by
    content-free text (system = ``rule``, one user message "."), every other key dropped except the transport's own scalar
    settings, so not one byte of the original prompt, history or message reaches the provider. Hermes sends the ORIGINAL
    request when middleware returns nothing or raises (hermes_cli/middleware.py apply_llm_request_middleware), so this is
    what the scrub returns on every failure. Never raises."""
    base = request if isinstance(request, dict) else {}
    keep = ("model", "stream", "max_tokens", "max_completion_tokens", "max_output_tokens", "temperature", "top_p",
            "stream_options", "store", "inferenceConfig", "modelId", "anthropic_version", "timeout")
    out: dict = {}
    for k in keep:
        v = base.get(k)
        if isinstance(v, (str, int, float, bool)) or v is None and k in base:
            out[k] = v
        elif isinstance(v, dict) and all(isinstance(x, (str, int, float, bool)) for x in v.values()):
            out[k] = dict(v)
    if "instructions" in base or "input" in base:                       # Responses
        out["instructions"] = rule
        out["input"] = [{"role": "user", "content": "."}]
    elif "system" in base:                                              # Anthropic Messages / Bedrock Converse
        system = base.get("system")
        if isinstance(system, list) and any(isinstance(p, dict) and "type" not in p for p in system):
            out["system"] = [{"text": rule}]
            out["messages"] = [{"role": "user", "content": [{"text": "."}]}]
        else:
            out["system"] = rule
            out["messages"] = [{"role": "user", "content": "."}]
    else:                                                               # chat completions (and anything unknown)
        out["messages"] = [{"role": "system", "content": rule}, {"role": "user", "content": "."}]
    return out


def scrub_request(request: dict, stable: str, memory_blocks: Iterable[str] = (), probe_line: str = "",
                  facts: "Optional[MachineFacts]" = None, required: str = "") -> dict:
    """A copy of a provider request in which every SYSTEM carrier holding this agent's prompt is rebuilt from the allow-list,
    and every system carrier is redacted (path shapes, this machine's values). Raises ScrubError when a system carrier still
    holds a Hermes machine-detail marker, or when ``required`` (the exact Team Chat rule) is in no system carrier. User and
    assistant messages are returned unchanged (rule E)."""
    if not isinstance(request, dict):
        raise ScrubError("the request is not a mapping")
    blocks = [b for b in (memory_blocks or ()) if b]

    def fn(text):
        return scrub_system_text(text, stable, blocks, probe_line)
    fn.stable = stable
    out = deepcopy(request)
    carriers = list(_carriers(out))
    for holder, key in carriers:
        holder[key], rebuilt = _scrub_value(holder[key], fn)
        if not rebuilt and required and any(required in s for s in _walk_strings(holder[key])):
            # this agent's prompt in a shape the allow-list rebuild did not recognise: never sent as it is
            raise ScrubError("the system prompt is in a shape the plugin cannot rebuild")
        holder[key] = _protected_redact(holder[key], facts, required)
    system_values = [holder[key] for holder, key in carriers]
    left = leftover_markers(system_values)
    if left:
        raise ScrubError("local machine details remain in the system prompt (" + ", ".join(left) + ")")
    if required and not any(required in s for s in _walk_strings(system_values)):
        raise ScrubError("the Team Chat rule is not in the system prompt verbatim")
    return out


# ------------------------------------------------------------------------------------------------------------ output filter

REFUSAL = ("I can't run commands, read files or look at the computer I run on from Team Chat, so I can't share that here.")
PATH_MARK = "[local path removed]"

# --- THE PATH RULE: ONE table, used for the input (rule A) and the output (rule C) -------------------------------------------
#: every local root; matched CASE-INSENSITIVELY (the default macOS volume is case-insensitive, so /users/x is /Users/x)
PATH_ROOTS = ("Users", "home", "root", "private", "var", "etc", "tmp", "opt", "Volumes", "Library", "System", "Applications",
              "usr", "bin", "sbin", "dev", "proc", "mnt", "srv", "snap", "cores", "Network")
#: roots under which a folder name may contain spaces: an UNQUOTED path starting at one of these (and every ~ path, file URI,
#: Windows drive path, UNC share and this machine's own exact paths) is removed to the end of its line or sentence, since
#: where a spaced name ends cannot be known
SPACED_ROOTS = frozenset(r.lower() for r in ("Users", "home", "Volumes", "private", "etc", "var", "tmp", "opt", "usr",
                                             "Library", "Applications"))
#: the shapes, in one alternation (named groups say which); a match INSIDE a web URL (http, https, ftp, ws, wss) is not a
#: local path; a path right after an opening quote or backtick is removed through the matching closing one
PATH_SHAPES = (
    ("uri", r"(?<![\w])file:/"),                                    # file:///etc/hosts, FILE:/x (any case)
    ("tilde", r"(?<![\w.:~-])~[\w.-]*(?=[/\\])"),                    # ~/x, ~user/x, ~\x
    ("root", r"(?<![\w.:~-])/(?P<rootname>" + "|".join(PATH_ROOTS) + r")(?![\w-]|\.\w)"),   # /Users, //etc/hosts, /tmp/x
    ("drive", r"(?<![\w])[A-Za-z]:\\"),                              # C:\Users\x
    ("unc", r"\\\\[\w.-]+\\"),                                       # \\server\share
)
_START = re.compile("|".join(f"(?P<{name}>{rx})" for name, rx in PATH_SHAPES), re.I)
_START_ANY = re.compile("|".join(f"(?:{rx.replace('(?P<rootname>', '(?:')})" for _, rx in PATH_SHAPES), re.I)
_WEB_URL = re.compile(r"(?<![\w])(?:https?|ftp|wss?)://[^\s<>\"'`]+", re.I)
_QUOTED = re.compile(r"([\"'`])((?:" + _START_ANY.pattern + r")[^\n]*?)\1", re.I)
_BARE_END = re.compile(r"[\s`'\"<>()\[\]{},;|]")
_SPACED_END = re.compile(r"\n|[.!?;](?=\s|$)|[`\"<>|)\]]|'(?=\s|$|[.,;!?])")
_STOP = r"\s`'\"<>()\[\]{},;|"


def _spaced_end(text: str, i: int) -> int:
    m = _SPACED_END.search(text, i)
    return m.start() if m else len(text)


#: Unicode Default_Ignorable_Code_Point (DerivedCoreProperties.txt, Unicode 15.1), as a fixed table: these render as nothing
#: (variation selectors, joiners, bidi and tag characters, Hangul fillers, the combining grapheme joiner, soft hyphen ...)
DEFAULT_IGNORABLE = ((0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5),
                     (0x180B, 0x180F), (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x3164, 0x3164),
                     (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF), (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8), (0x1BCA0, 0x1BCA3),
                     (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF))
#: Markdown inline links / images and reference links: ``[label](dest)``, ``![alt](dest)``, ``[label][ref]``; only the label
#: is kept by the LINK projection (autolinks ``<https://...>`` are left as they are)
_MD_LINK = re.compile(r"!?\[([^\[\]\n]*)\](?:\([^()\n]*\)|\[[^\[\]\n]*\])")


def _ignorable(c: str) -> bool:
    import unicodedata
    o = ord(c)
    return (unicodedata.category(c) in ("Cf", "Mn", "Me")
            or any(a <= o <= b for a, b in DEFAULT_IGNORABLE))


def project(text: str, links: bool = False) -> tuple[str, list[int]]:
    """THE PROJECTION every match is made on: (projected text, original index of each projected character). Per original
    character: a default-ignorable code point (the DEFAULT_IGNORABLE table and every Unicode Cf) and every combining mark
    (Mn / Me) is dropped; the Markdown markers ``*``, `` ` `` and ``_`` are dropped, ``~~`` is dropped (a single ``~`` stays,
    it starts a home path), a backslash before one of those markers is dropped; every other character is NFKC-normalised
    (a fullwidth slash becomes ``/``). ``links``: a Markdown inline link, image or reference link is projected to its label
    only (matching runs on both projections, so a path in a link's DESTINATION is matched too). Not a Markdown parser: only
    what can split a word without changing how it reads. Needles are projected the same way (``project_needle``)."""
    import unicodedata
    t = str(text or "")
    drop = set()
    if links:
        for m in _MD_LINK.finditer(t):
            drop.update(range(m.start(), m.start(1)))           # "!" and "["
            drop.update(range(m.end(1), m.end()))               # "](dest)" or "][ref]"
    out: list[str] = []
    idx: list[int] = []
    i, n = 0, len(t)
    while i < n:
        c = t[i]
        if i in drop:
            i += 1
            continue
        if c == "\\" and i + 1 < n and t[i + 1] in "*_~`":
            i += 1
            continue
        if c in "*`_" or _ignorable(c):
            i += 1
            continue
        if c == "~" and i + 1 < n and t[i + 1] == "~":
            i += 2
            continue
        for ch in unicodedata.normalize("NFKC", c):
            if not _ignorable(ch):
                out.append(ch)
                idx.append(i)
        i += 1
    return "".join(out), idx


def project_needle(value: str) -> str:
    """A value to look for, projected exactly as the text is."""
    return project(value)[0]


def _back(idx: list[int], a: int, b: int) -> tuple[int, int]:
    return idx[a], idx[b - 1] + 1


def _merge(t: str, spans: list) -> list[tuple[int, int, str]]:
    """Sorted, merged (start, end, kind) spans of the ORIGINAL text; a sentence's own trailing punctuation is kept out of a
    path; overlapping spans merge, and a merged span that holds a path is a path."""
    out: list = []
    for a, b, kind in sorted(spans):
        while kind == "path" and b > a and t[b - 1] in " \t.,;:!?":
            b -= 1
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            pa, pb, pk = out[-1]
            out[-1] = (pa, max(pb, b), "path" if "path" in (pk, kind) else kind)
        else:
            out.append((a, b, kind))
    return out


def _path_spans_raw(t: str, proj: str, idx: list[int], exact_paths: Iterable[str]) -> list:
    quoted = []
    for m in _QUOTED.finditer(t):                       # quoted or backticked (ORIGINAL text): through the closing delimiter
        if not any(a <= m.start(2) < b for a, b in (u.span() for u in _WEB_URL.finditer(t))):
            quoted.append(m.span(2))
    spans = [(a, b, "path") for a, b in quoted]
    urls = [m.span() for m in _WEB_URL.finditer(proj)]

    def closed(b):   # a quoted path is complete when its closing delimiter is followed by a space, punctuation or the end
        return b + 1 >= len(t) or t[b + 1].isspace() or t[b + 1] in ".,;:!?)]}"

    def skip(pi):
        oi = idx[pi]
        return any(a <= pi < b for a, b in urls) or any(a <= oi < b and closed(b) for a, b in quoted)
    for m in _START.finditer(proj):
        if skip(m.start()):
            continue
        kind = m.lastgroup if m.lastgroup != "rootname" else "root"
        spaced = kind != "root" or (m.group("rootname") or "").lower() in SPACED_ROOTS
        if spaced:
            end = _spaced_end(proj, m.end())
        else:
            e = _BARE_END.search(proj, m.end())
            end = e.start() if e else len(proj)
        if end > m.start():
            spans.append((*_back(idx, m.start(), end), "path"))
    for p in exact_paths:                                # this machine's own paths, any case, projected
        needle = project_needle(p)
        if not needle.strip("/\\"):
            continue
        for m in re.finditer(re.escape(needle), proj, re.I):
            if not skip(m.start()):
                spans.append((*_back(idx, m.start(), _spaced_end(proj, m.end())), "path"))
    return spans


def path_spans(text: str, exact_paths: Iterable[str] = ()) -> list[tuple[int, int]]:
    """Where the local paths of ``text`` are, by THE PATH RULE on THE PROJECTION (merged spans of the original text)."""
    t = str(text or "")
    spans = []
    for links in (False, True):   # both projections: a link's destination and its rendered label
        proj, idx = project(t, links)
        spans += _path_spans_raw(t, proj, idx, exact_paths)
    return [(a, b) for a, b, _ in _merge(t, spans)]


def has_local_path(text: str) -> bool:
    return bool(path_spans(text))


def redact_paths(text: str, exact_paths: Iterable[str] = ()) -> tuple[str, int]:
    return _redact(str(text or ""), exact_paths, ())


_VERB = re.compile(r"\b(run|runs|running|execute|exec|list|ls|cat|open|read|show|print|paste|dump|tail|head|type|grep|find|"
                   r"browse|display|output|give\s+me|tell\s+me|what(?:'s|\s+is|\s+are)\s+(?:in|inside|on))\b", re.I)
_CODE_SPAN = re.compile(r"`[^`\n]+`|```")
#: command words (ordinary English words that also name commands, such as top, set, more, file or id, are left out: "the top
#: -5 deals" is not a command). Also left out: the words Hermes's own plugin and skill scanners flag in a plugin's source
#: (the environment dump, privilege and scheduler commands); a request for them is still caught by its argument ("ls ~"
#: after a prefix), by a local target or by the local-machine nouns ("environment variables").
_CMD = (r"ls|dir|cat|pwd|whoami|ps|df|du|uname|ifconfig|ipconfig|netstat|lsof|tree|history|find|grep|head|"
        r"tail|ssh|scp|curl|wget|chmod|chown|rm|mv|cp|mkdir|touch|echo|python3?|node|bash|zsh|sh|fish|"
        r"powershell|pwsh|cmd|git|hostname|launchctl|defaults|sqlite3|kill|pkill|open|xdg-open|"
        r"which|whereis|locate|stat|nslookup|dig|ping|traceroute|systemctl|journalctl|brew|apt|npm|pip3?|docker|kubectl")
#: a command word given an argument that is a flag, a path, a glob or a variable (``ls ~``, ``cat /etc/hosts``, ``ps -ef``)
_CMD_INVOCATION = re.compile(r"(?<![\w-])(?:" + _CMD + r")\s+(?:-{1,2}\w|~|/|\.{1,2}(?:/|(?=\s|$|`))|\*|\$|[A-Za-z]:\\)", re.I)
#: command words that are never ordinary CRM words: alone in a code span they still count (``\`pwd\```); every other command
#: word counts only with an argument (``open``, ``history``, ``find`` alone are CRM words: "what does `open` mean for a deal?")
_UNAMBIGUOUS = r"ls|pwd|whoami|uname|ifconfig|ipconfig|netstat|lsof|hostname"
_CMD_IN_SPAN = re.compile(r"`\s*(?:" + _UNAMBIGUOUS + r")(?:\s[^`]*)?`", re.I)
_LOCAL_NOUN = re.compile(r"\b(home\s+(?:dir|directory|folder)|your\s+(?:computer|machine|mac|laptop|pc|disk|drive|"
                         r"filesystem|file\s+system|terminal|shell|desktop|downloads|documents|files|folders|server|host|"
                         r"environment)|environment\s+variables?|env\s+vars?|local\s+(?:files?|folders?|disk|machine|path)|"
                         r"(?:the|this)\s+(?:filesystem|file\s+system|terminal|shell))\b", re.I)
_LOCAL_TARGET = re.compile(r"(?:^|[\s\"'(`])(?:~(?:[\w.-]+)?(?:[/\\]|(?=[\s`'\".,;:!?)]|$))|\.{1,2}/|/(?:"
                           + "|".join(PATH_ROOTS) + r")\b|[A-Za-z]:\\|file:/)", re.I)
_REFUSAL_PHRASE = re.compile(r"\b(can(?:'|’)?t|cannot|can\s+not|unable|not\s+able|won(?:'|’)?t|will\s+not|"
                             r"don(?:'|’)?t\s+have\s+(?:access|the\s+ability|a\s+way)|do\s+not\s+have\s+(?:access|the\s+ability)|"
                             r"no\s+access|not\s+allowed|not\s+permitted|isn(?:'|’)?t\s+something\s+i\s+can|"
                             r"not\s+something\s+i\s+can)\b", re.I)
_PERM_LINE = re.compile(r"(?m)^\s*[dlcbps-][rwxsStT-]{9}[@+.]?\s+\d+\s")
#: ``user@host:~/x$`` (bash) or ``user@host ~ %`` (zsh); an e-mail address followed by "$" or "#" is not one
_SHELL_PROMPT = re.compile(r"[\w.-]+@[\w.-]+(?::[~/][^\s$#%]*\s?|\s+~[\w/.-]*\s+)[$#%](?:\s|$)")
MAX_REFUSAL_CHARS = 600


def asks_to_run(text: str) -> bool:
    """A HINT, not part of the safety contract: whether a Team Chat message names a command with arguments (``ls ~``,
    ``cat /etc/hosts``, ``ps -ef``), an unambiguous command in a code span (``\\`pwd\\```), or an action verb (run, list, read,
    open, show, paste ...) together with a local target (``~``, ``~/x``, ``./x``, ``/Users``, ``/etc``, ``C:\\``, ``file://``)
    or a local-machine noun (home folder, your computer / terminal / files, environment variables). Web addresses alone are
    not local targets; ambiguous command words alone (``open``, ``history``, ``find``) are not commands."""
    t = str(text or "")
    if _CMD_INVOCATION.search(t) or _CMD_IN_SPAN.search(t):
        return True
    return bool(_VERB.search(t) and (_LOCAL_TARGET.search(t) or _LOCAL_NOUN.search(t)))


def is_refusal(answer: str) -> bool:
    """A short plain refusal: at most MAX_REFUSAL_CHARS characters, a refusal phrase (can't, cannot, unable, not able, won't,
    no access ...), no code block and no local path."""
    a = str(answer or "").strip()
    return (0 < len(a) <= MAX_REFUSAL_CHARS and bool(_REFUSAL_PHRASE.search(a)) and "```" not in a
            and not has_local_path(a) and not looks_like_command_output(a))


def looks_like_command_output(answer: str) -> bool:
    """An ``ls -l`` style line (permission bits, link count) or a ``user@host:~$`` shell prompt."""
    a = str(answer or "")
    return bool(_PERM_LINE.search(a) or _SHELL_PROMPT.search(a))


MACHINE_MARK = "[machine detail removed]"


def _distinctive(value: str) -> bool:
    """By character class only: a digit, "-", "_" or ".", or an inner capital (CamelCase) makes a value distinctive (removed
    everywhere); a plain word (letters, one case or Capitalised), however short or long, is removed only in machine shapes."""
    return any(c.isdigit() or c in "-_." for c in value) or any(c.isupper() for c in value[1:]) and any(c.islower() for c in value)


#: the values the inventory must hold, each gathered on its own; a missing one refuses the connection (fail closed)
REQUIRED_FACTS = ("home", "user", "host", "short_host", "hermes_home", "cwd", "temp", "os_version")
_PATH_FACTS = ("home", "hermes_home", "cwd", "temp")


class MachineFacts:
    """This machine's own concrete values: ``paths`` (removed as exact strings in any case, by the path rule), ``values``
    (distinctive names: removed everywhere, case-insensitive, word-bounded, MACHINE_MARK) and ``words`` (plain-word names of
    any length: removed only in machine shapes, ``word@``, ``@word``, ``word.local``, ``/Users/word``, ``/home/word``).
    ``required`` maps each REQUIRED_FACTS key to what was read; ``missing`` lists the keys nothing could be read for."""

    def __init__(self, paths: Iterable[str] = (), names: Iterable[str] = (), required: Optional[dict] = None,
                 missing: Iterable[str] = ()):
        self.paths = tuple(sorted({p for p in paths if isinstance(p, str) and p.strip("/\\")}, key=len, reverse=True))
        vals = {n.strip() for n in names if isinstance(n, str) and n.strip()}
        self.values = tuple(sorted((v for v in vals if _distinctive(v)), key=len, reverse=True))
        self.words = tuple(sorted((v for v in vals if not _distinctive(v)), key=len, reverse=True))
        self.required = dict(required or {})
        self.missing = tuple(missing)
        # compiled on the PROJECTED values: they are matched against the projected text
        self._value_res = [re.compile(r"(?<![\w])" + re.escape(project_needle(v)) + r"(?![\w])", re.I)
                           for v in self.values if project_needle(v)]
        self._word_res = [re.compile(r"(?<![\w.-])" + re.escape(pw) + r"@|@" + re.escape(pw) + r"(?![\w-])"
                                     r"|(?<![\w.-])" + re.escape(pw) + r"\.local\b|/(?:Users|home)/" + re.escape(pw) + r"(?![\w-])",
                                     re.I) for pw in (project_needle(w) for w in self.words) if pw]

    @property
    def complete(self) -> bool:
        return not self.missing

    def __repr__(self) -> str:   # never the values themselves
        return (f"MachineFacts({len(self.paths)} paths, {len(self.values)} values, {len(self.words)} words, "
                f"missing={list(self.missing)})")


def _default_readers(hermes_home: str) -> dict:
    import os
    import sys

    def env(name):
        return lambda: os.environ.get(name, "")

    def pw(field):
        def read():
            import pwd
            return getattr(pwd.getpwuid(os.getuid()), field)
        return read

    def mac_version():
        import platform
        return platform.mac_ver()[0] if sys.platform == "darwin" else ""
    import platform
    import socket
    return {
        "home": [lambda: os.path.expanduser("~") if os.path.expanduser("~") != "~" else "", env("HOME"), pw("pw_dir")],
        "user": [lambda: __import__("getpass").getuser(), pw("pw_name"), env("USER"), env("LOGNAME")],
        "host": [socket.gethostname, platform.node, lambda: os.uname().nodename],
        "hermes_home": [lambda: hermes_home],
        "cwd": [os.getcwd, env("PWD")],
        "temp": [lambda: __import__("tempfile").gettempdir(), env("TMPDIR")],
        "os_version": [mac_version, platform.release, lambda: os.uname().release, platform.version],
    }


def gather_machine_facts(hermes_home: str = "", extra_paths: Iterable[str] = (), readers: Optional[dict] = None) -> MachineFacts:
    """This machine's values (pure Python, no subprocess): EVERY reader of every REQUIRED_FACTS key is tried on its own and
    all the distinct values kept (a user name from getpass, pwd and the environment can differ); ``short_host`` is each
    host's first label. A key with no value is ``missing`` (the adapter refuses to connect). ``readers`` replaces the
    default readers (tests)."""
    import os
    readers = readers if readers is not None else _default_readers(hermes_home)
    got: dict = {}
    for key in REQUIRED_FACTS:
        if key == "short_host":
            continue
        vals = []
        for read in readers.get(key, []):
            try:
                v = read()
            except Exception:
                continue
            if isinstance(v, str) and v.strip():
                vals.append(v.strip())
        if key in _PATH_FACTS:
            for v in list(vals):
                try:
                    vals.append(os.path.realpath(v))
                except Exception:
                    pass
        got[key] = tuple(dict.fromkeys(vals))
    got["short_host"] = tuple(dict.fromkeys(h.split(".")[0] for h in got.get("host", ()) if h.split(".")[0]))
    missing = [k for k in REQUIRED_FACTS if not got.get(k)]
    paths = [v for k in _PATH_FACTS for v in got.get(k, ())] + [p for p in (extra_paths or ()) if p]
    names = [v for k in REQUIRED_FACTS if k not in _PATH_FACTS for v in got.get(k, ())]
    return MachineFacts(paths, names, got, missing)


def redact(text: str, facts: Optional[MachineFacts] = None) -> tuple[str, int]:
    """(``text`` with every local path (THE PATH RULE, this machine's own paths included, any case) and every machine value
    removed, how many)."""
    if facts is None:
        return _redact(str(text or ""), (), ())
    return _redact(str(text or ""), facts.paths, facts._value_res + facts._word_res)


def _redact(t: str, exact_paths: Iterable[str], value_res: list) -> tuple[str, int]:
    """One pass over THE PROJECTION: path spans and value spans, mapped back and merged; each merged span of the ORIGINAL
    text becomes PATH_MARK (it holds a path) or MACHINE_MARK."""
    spans = []
    for links in (False, True):   # both projections: a link's destination and its rendered label
        proj, idx = project(t, links)
        spans += _path_spans_raw(t, proj, idx, exact_paths)
        for rx in value_res:
            for m in rx.finditer(proj):
                if m.end() > m.start():
                    spans.append((*_back(idx, m.start(), m.end()), "value"))
    merged = _merge(t, spans)
    for a, b, kind in reversed(merged):
        t = t[:a] + (PATH_MARK if kind == "path" else MACHINE_MARK) + t[b:]
    return t, len(merged)


def redact_local_paths(answer: str, extra: Iterable[str] = ()) -> tuple[str, int]:
    """The path rule only, with the given exact paths (kept for callers without machine facts)."""
    return redact_paths(answer, extra)


def guard(requests: Iterable[str], answer: str, facts: Optional[MachineFacts] = None) -> tuple[str, Optional[str]]:
    """The text to post for ``answer`` and why it changed (None when it did not). Deterministic: command-output signatures ->
    REFUSAL; the hint (``asks_to_run`` on the turn's request texts) with a non-refusal answer -> REFUSAL; otherwise the
    redactor. Anything secret-shaped is the adapter's business BEFORE this runs (it is never posted)."""
    reqs = [str(r or "") for r in requests]
    if looks_like_command_output(answer):
        return REFUSAL, "the answer looked like command output"
    if any(asks_to_run(r) for r in reqs) and not is_refusal(answer):
        return REFUSAL, "the message named a command or a local target and the answer was not a refusal"
    text, n = redact(answer, facts)
    return text, (f"{n} local path(s) or machine detail(s) removed" if n else None)
