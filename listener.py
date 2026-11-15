"""The listener loop: the npm CLI listener's delivery contract (conduyt-cli src/agent-listener.ts runListenerLoop /
processInboxOnce / runClaimedTurnInner), run inside Hermes's gateway. No Hermes import: the adapter passes a dispatcher.

Delivery contract kept from the CLI:
  * every event is CLAIMED before the agent runs; the claim is completed by the ack, retried, never followed by a second run;
  * the cursor lives on the SERVER: this loop never sends a cursor of its own;
  * every reply carries ``reply:<message id>`` (a retried post gets the first reply back) and the claim token;
  * a turn runs under a lease renewed every minute; a failed or timed-out turn is FAILED (needs attention), never re-run here;
  * an event whose claim answer stayed unknown is HELD: nothing after it is claimed until the claim resolves;
  * repliedAckPending entries are acknowledged, needsAttention entries are reported once, the card's hello Test is answered;
  * transient failures back off exponentially with jitter (1 s base, 60 s cap); 401/403/404 stop the loop, 402 pauses it,
    the daily turn budget sleeps until it resets.
Added for Hermes:
  * bounded concurrency (default 2 turns), at most one running turn per Conduyt thread (one Hermes session per thread); an event
    that cannot start yet is left UNCLAIMED on the server (it is served again), so nothing is dropped and nothing is buffered;
  * the tool lock is re-checked before every poll and every claim; a refusal pauses the loop (no heartbeat: Conduyt shows the
    teammate as not listening) and resumes on its own once the config passes;
  * shutdown hands back (releases) every turn still running after a grace period, so the next start can serve it again.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .protocol import ApiError, ConduytClient, ConfigError, TransportError, backoff_delay, is_transient, redact

logger = logging.getLogger("conduyt_platform.listener")

LEASE_RENEW_S = 60.0          # the server lease is 10 minutes (app lib/agent-inbox.ts)
HEARTBEAT_S = 30.0            # "away" after 60 s without one (app lib/agent-auth.ts)
TURN_TIMEOUT_S = 8 * 60.0     # inside the 10-minute lease
AMBIGUITY_S = 10 * 60.0       # how long an unknown post/ack/silent is retried before the turn is marked possibly replied
LOCK_RECHECK_S = 30.0
PLAN_PAUSE_S = 5 * 60.0
BUDGET_SLEEP_CAP_S = 10 * 60.0
SHUTDOWN_GRACE_S = 20.0
MAX_REPLY_UNITS = 9_500       # Conduyt takes a message body under 10,000 characters, counted as JavaScript does (UTF-16 units)


@dataclass
class TurnOutcome:
    """What the dispatcher produced: ``reply`` (text), ``silent`` (no response warranted), ``failed`` (needs attention) or
    ``release`` (hand the turn back unrun, e.g. the tool lock refused it)."""
    kind: str
    text: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class Turn:
    event: dict
    generation: int

    @property
    def id(self) -> str:
        return self.event["id"]

    @property
    def thread_key(self) -> tuple[str, str]:
        return thread_key(self.event)


def thread_key(event: dict) -> tuple[str, str]:
    """The Conduyt thread a turn answers in: (channel, thread root). One Hermes session per key."""
    return (str(event["channelId"]), str(event.get("parentId") or event["id"]))


#: the turn reasons (contract AGENT_TURN_IDENTITY_SCHEMA.reason) that ADDRESS this teammate: a person mentioned it, wrote
#: to it directly, or another AI teammate handed it the message. Silence is not an answer to one of these (see _finish).
ADDRESSED_REASONS = frozenset({"mention", "dm", "handoff"})


def addressed(event: dict) -> bool:
    """Whether the turn's message is addressed to this teammate: the turn identity's reason when the server sends one, else
    the older ``addressedBy`` (a server from before turn identity) saying it was mentioned."""
    turn = event.get("turn")
    if isinstance(turn, dict):
        return turn.get("reason") in ADDRESSED_REASONS
    return event.get("addressedBy") == "mention"


def requires_answer(event: dict) -> bool:
    """Whether silence is no answer for this turn: it is addressed to this teammate (:func:`addressed`), or it carries folded
    messages (``alsoAnswers``: earlier messages to this teammate whose own turns Conduyt skipped and never serves again)."""
    return addressed(event) or bool(event.get("alsoAnswers"))


def _utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def shorten_to_limit(text: str, limit: int = MAX_REPLY_UNITS) -> str:
    """A PREFIX of ``text`` that fits ``limit`` UTF-16 units (Conduyt counts as JavaScript does). Only removes characters: no
    marker, no added character; trailing whitespace of the prefix is dropped.

    The answer is read ONCE with a CommonMark block parser (markdown-it, which Hermes ships through rich; tables enabled), as
    its top-level blocks with their line ranges, and the cut keeps WHOLE blocks: the longest run of whole top-level blocks
    that fits, less any trailing heading (ATX or setext, its title line included), thematic break, list or quote whose last
    block is one of those, list ending in an empty item, or table without a body row (unless only such blocks fit: then they stand). A code block, list, quote or table is therefore kept whole or dropped whole: no fence
    is left open and none is invented. Only a top-level PARAGRAPH, which holds no block structure, is cut inside: at its last
    line break or sentence end, else its last space, when that lies in the second half of the part that fits; else the whole
    blocks before it stand when they reach the second half; else the paragraph is cut at the limit. A cut inside the
    paragraph is used only when the result re-parses as the same blocks plus that one paragraph (:func:`_parses_as`). When not even the first
    block fits and it is not a paragraph (one giant code block or list), the plain cut stands (:func:`_plain_cut`; a known
    limit). Without the parser (it cannot be imported, or it fails on this text) every answer gets the plain cut."""
    if _utf16_units(text) <= limit:
        return text
    lo, hi = 0, len(text)
    while lo < hi:   # the longest code-point prefix within the limit
        mid = (lo + hi + 1) // 2
        if _utf16_units(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    head = text[:lo]
    blocks = _top_blocks(text)
    if blocks is None:
        return _plain_cut(head)
    floor = len(head) // 2
    fits = [b for b in blocks if len(text[:b[1]].rstrip()) <= lo]   # what is posted is right-stripped
    whole = list(fits)
    while whole and whole[-1][2]:
        whole.pop()   # a dangling heading, rule, empty last list item or bodiless table
    if fits and not whole:
        whole = fits  # only dangling blocks fit: they stand rather than a cut into the block after them
    kept = whole[-1][1] if whole else 0
    straddle = next((b for b in blocks if b not in fits and b[0] < lo), None)
    if straddle is not None and straddle[3] == "paragraph":
        candidates = []
        for marks in (("\n", ". ", "! ", "? ", "\u3002"), (" ",)):
            at = max(head.rfind(m, straddle[0]) + (len(m) if m != "\n" else 0) for m in marks)
            if at > max(floor, straddle[0]):
                candidates.append(at)
                break
        if kept > floor:
            candidates.append(kept)
        candidates.append(lo)
        before = [b[:2] for b in blocks if b[0] < straddle[0]]
        for at in candidates:
            cut = head[:at].rstrip()
            if at == kept or _parses_as(cut, before, straddle[0]):
                return cut
    if kept and text[:kept].strip():
        return text[:kept].rstrip()
    return _plain_cut(head)


def _parses_as(cut: str, before: list, para_start: int) -> bool:
    """Whether a cut INSIDE a paragraph changed no structure: ``cut`` parses as exactly the whole blocks before that paragraph
    (same offsets) and then one paragraph from its start. Shortening a line can make a block of it (a ``` line whose info
    string lost its backtick becomes a fence, a line cut to ``#`` a heading); such a cut is refused."""
    got = _top_blocks(cut)
    return (got is not None and [b[:2] for b in got[:-1]] == before and len(got) == len(before) + 1
            and got[-1][3] == "paragraph" and got[-1][0] == para_start)


def _plain_cut(head: str) -> str:
    """The plain cut of ``head`` (the longest prefix that fits): the last paragraph break, else the last line break or sentence
    end, else the last space, when that boundary lies in the second half of ``head``; else ``head`` itself."""
    floor = len(head) // 2   # a boundary in the first half would throw away too much: cut later instead
    for marks in (("\n\n",), ("\n", ". ", "! ", "? ", "\u3002"), (" ",)):
        at = max(head.rfind(m) + (len(m) if m != "\n\n" else 0) for m in marks)
        if at > floor:
            return head[:at].rstrip()
    return head.rstrip()


_CONTAINERS = frozenset({"blockquote_open", "bullet_list_open", "ordered_list_open", "list_item_open"})
_PARSER: Any = None        # the markdown-it parser, built on first use; False once it proved unavailable
_PARSER_NOTED = False


def _markdown():
    """The CommonMark parser with tables (markdown-it-py, a dependency of rich, which Hermes requires), or None, logged
    once, when it cannot be built: every cut is then the plain cut."""
    global _PARSER, _PARSER_NOTED
    if _PARSER is None:
        try:
            from markdown_it import MarkdownIt
            _PARSER = MarkdownIt("commonmark").enable("table")
        except Exception as exc:
            _PARSER = False
            if not _PARSER_NOTED:
                _PARSER_NOTED = True
                logger.warning("Conduyt: no Markdown parser (%s); long answers get the plain cut", type(exc).__name__)
    return _PARSER or None


def _top_blocks(text: str) -> Optional[list[tuple[int, int, bool, str]]]:
    """The top-level blocks of ``text`` as (start offset, end offset, dangling, kind), from markdown-it's line maps. Lines
    are counted as markdown-it counts them (it normalises CRLF and CR to LF before parsing). End offsets include the block's
    last line ending. None when there is no parser or it fails on this text."""
    md = _markdown()
    if md is None:
        return None
    try:
        tokens = md.parse(text)
    except Exception as exc:
        logger.warning("Conduyt: the Markdown parser failed (%s); this answer gets the plain cut", type(exc).__name__)
        return None
    starts = [0] + [m.end() for m in re.finditer(r"\r\n|\r|\n", text)]
    offset = lambda line: starts[line] if line < len(starts) else len(text)
    blocks: list[tuple[int, int, bool, str]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.level != 0 or tok.nesting == -1 or tok.map is None:
            i += 1
            continue
        j = i
        if tok.nesting == 1:   # walk to the matching close; collect what the dangling rule needs
            depth = 0
            while True:
                depth += tokens[j].nesting
                if depth == 0:
                    break
                j += 1
        inner = tokens[i:j + 1]
        kind = tok.type[:-5] if tok.type.endswith("_open") else tok.type
        if kind in ("heading", "hr"):
            dangling = True
        elif kind in ("bullet_list", "ordered_list", "blockquote"):
            # a container ending in a heading or rule (its last leaf block), or a list whose last item is empty
            leaf = next((t for t in reversed(inner[1:-1]) if t.nesting != -1 and t.type != "inline"
                         and t.type not in _CONTAINERS), None)
            dangling = leaf is not None and leaf.type in ("heading_open", "hr")
            if kind != "blockquote":
                last_item = max(k for k, t in enumerate(inner) if t.type == "list_item_open" and t.level == 1)
                dangling = dangling or inner[last_item + 1].type == "list_item_close"
        elif kind == "table":
            dangling = not any(t.type == "tbody_open" for t in inner)
        else:
            dangling = False
        blocks.append((offset(tok.map[0]), offset(tok.map[1]), dangling, kind))
        i = j + 1
    return blocks


def finalize_reply(text: str) -> Optional[str]:
    """What is posted, or None for NOTHING (the turn is marked needs attention). The plugin never adds a character of its
    own to a posted answer:
      1. leading and trailing whitespace is removed (str.strip: removal only);
      2. an answer holding anything secret-shaped (anything protocol.redact would replace: Conduyt-key-shaped or
         pair-code-shaped) is NOT posted at all (None);
      3. an answer over Conduyt's size is shortened to a prefix of itself (:func:`shorten_to_limit`); Conduyt's protocol has
         no non-chat field to report that, so it is only logged.
    The result is always a prefix of ``text.strip()``, or "" (silent), or None."""
    out = str(text or "").strip()
    if redact(out) != out:
        return None
    short = shorten_to_limit(out)
    if short != out:
        logger.info("Conduyt answer shortened from %d to %d characters to fit Conduyt's message size", len(out), len(short))
    return short


Dispatcher = Callable[[Turn], Awaitable[TurnOutcome]]
LockCheck = Callable[[], Any]   # returns an object with .ok and .message()
StatusSink = Callable[[str, Optional[str]], None]   # (state, message) for Hermes's status surface


class StopListener(Exception):
    """A fatal answer: the loop stops (the reason is published)."""


@dataclass
class Listener:
    client: ConduytClient
    dispatch: Dispatcher
    lock_check: LockCheck
    status: StatusSink = lambda state, message: None
    #: called for EVERY claim this listener ends as failed / needs attention, before the fail call: the adapter
    #: rolls that thread over to a fresh Hermes session, so the next message never continues a session holding a failed run
    on_failed: Callable[[Turn], None] = lambda turn: None
    concurrency: int = 2
    wait_s: int = 25
    turn_timeout_s: float = TURN_TIMEOUT_S
    lease_renew_s: float = LEASE_RENEW_S
    heartbeat_s: float = HEARTBEAT_S
    ambiguity_s: float = AMBIGUITY_S
    lock_recheck_s: float = LOCK_RECHECK_S
    shutdown_grace_s: float = SHUTDOWN_GRACE_S
    base_backoff_s: float = 1.0
    cap_backoff_s: float = 60.0
    rand: Callable[[], float] = random.random
    clock: Callable[[], float] = time.monotonic

    _stopping: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _turn_done: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _completions: int = field(default=0, init=False)
    _running: dict[str, tuple[asyncio.Task, Turn]] = field(default_factory=dict, init=False)
    _busy_threads: set = field(default_factory=set, init=False)
    _held: Optional[dict] = field(default=None, init=False)
    _reported: set = field(default_factory=set, init=False)
    _helloed: set = field(default_factory=set, init=False)
    _hello_refused: set = field(default_factory=set, init=False)
    _heartbeat_task: Optional[asyncio.Task] = field(default=None, init=False)
    _fatal: Optional[str] = field(default=None, init=False)
    stats: dict = field(default_factory=lambda: {"replied": 0, "silent": 0, "failed": 0, "released": 0, "skipped": 0}, init=False)

    # ---------------------------------------------------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        """Run until stop() or a fatal answer. Never raises for protocol failures (they are published as status)."""
        failures = 0
        try:
            while not self._stopping.is_set():
                if not await self._lock_ok():
                    continue
                try:
                    if self._heartbeat_task is None:
                        await self.client.heartbeat()   # the first beat is part of starting: a revoked key stops here
                        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                        self.status("connected", None)
                        logger.info("Conduyt listener connected (%s)", self.client.base_url)
                    await self._poll_once()
                    failures = 0
                except StopListener as stop:
                    self._fatal = str(stop)
                    self.status("fatal", str(stop))
                    logger.error("%s", stop)
                    return
                except ApiError as err:
                    if err.status == 402:
                        await self._stop_heartbeat()
                        self.status("paused", "Your Conduyt plan does not include AI teammates right now; checking again in 5 minutes.")
                        await self._sleep(PLAN_PAUSE_S)
                        continue
                    if err.fatal:
                        self._fatal = fatal_text(err)
                        self.status("fatal", self._fatal)
                        logger.error("%s", self._fatal)
                        return
                    if err.budget_exhausted:
                        await self._sleep(budget_sleep_s(err))
                        continue
                    failures += 1
                    await self._backoff(failures, err)
                except (TransportError, OSError) as err:
                    failures += 1
                    await self._backoff(failures, err)
                except ConfigError as err:
                    self._fatal = str(err)
                    self.status("fatal", str(err))
                    return
        finally:
            await self._stop_heartbeat()

    async def stop(self) -> None:
        """Clean shutdown: stop claiming, let running turns finish within the grace period, hand back the rest (release)."""
        self._stopping.set()
        self._turn_done.set()
        tasks = [t for t, _ in self._running.values()]
        if tasks:
            await asyncio.wait(tasks, timeout=self.shutdown_grace_s)
        for mid, (task, turn) in list(self._running.items()):
            if not task.done():
                task.cancel()
                try:
                    await self.client.release(turn.id, turn.generation)
                    self.stats["released"] += 1
                    logger.info("Conduyt turn %s handed back at shutdown (it is served again on the next start)", mid)
                except Exception as exc:  # the lease lapses on its own
                    logger.warning("Conduyt turn %s could not be handed back (%s); its lease lapses on its own", mid, type(exc).__name__)
        await self._stop_heartbeat()

    @property
    def fatal(self) -> Optional[str]:
        return self._fatal

    # ---------------------------------------------------------------------------------------------------------- helpers

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=max(0.0, seconds))
        except asyncio.TimeoutError:
            pass

    async def _backoff(self, failures: int, err: BaseException) -> None:
        delay = backoff_delay(failures, base=self.base_backoff_s, cap=self.cap_backoff_s, rand=self.rand)
        logger.warning("Conduyt listener error (attempt %d), retrying in %.1f s: %s", failures, delay, redact(err))
        self.status("retrying", f"Conduyt unreachable; retrying ({type(err).__name__})")
        await self._sleep(delay)

    async def _lock_ok(self) -> bool:
        verdict = self.lock_check()
        if verdict.ok:
            return True
        await self._stop_heartbeat()
        self.status("needs_setup", verdict.message())
        logger.error("%s", verdict.message())
        await self._sleep(self.lock_recheck_s)
        return False

    async def _stop_heartbeat(self) -> None:
        task, self._heartbeat_task = self._heartbeat_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _heartbeat_loop(self) -> None:
        while not self._stopping.is_set():
            await self._sleep(self.heartbeat_s)
            if self._stopping.is_set():
                return
            try:
                await self.client.heartbeat()
            except ApiError as err:
                if err.fatal and err.status != 402:
                    self._fatal = fatal_text(err)
                    self.status("fatal", self._fatal)
                    self._stopping.set()
                    return
                logger.warning("Conduyt heartbeat failed: %s", err)
            except Exception as err:
                logger.warning("Conduyt heartbeat failed: %s", type(err).__name__)

    async def _until_definite(self, label: str, call: Callable[[], Awaitable[Any]], deadline: list) -> Any:
        """Retry ``call`` while its outcome is unknown, inside one shared ambiguity window (``deadline`` holds its end)."""
        failures = 0
        while True:
            try:
                return await call()
            except Exception as err:
                if not is_transient(err):
                    raise
                if deadline[0] is None:
                    deadline[0] = self.clock() + self.ambiguity_s
                if self.clock() >= deadline[0]:
                    raise OutcomeUnknown(f"{label}: the outcome stayed unknown") from None
                failures += 1
                delay = min(backoff_delay(failures, base=self.base_backoff_s, cap=self.cap_backoff_s, rand=self.rand),
                            max(0.0, deadline[0] - self.clock()))
                logger.warning("%s failed (attempt %d, outcome unknown), retrying in %.1f s; the turn is not run again", label, failures, delay)
                await asyncio.sleep(delay)

    # ---------------------------------------------------------------------------------------------------------- the poll

    def _free_slots(self) -> int:
        return self.concurrency - len(self._running)

    async def _wait_for_capacity(self) -> None:
        while self._free_slots() <= 0 and not self._stopping.is_set():
            self._turn_done.clear()
            await self._turn_done.wait()

    async def _poll_once(self) -> None:
        await self._wait_for_capacity()
        if self._stopping.is_set():
            return
        if self._held is not None and not await self._resolve_held():
            return
        seen = self._completions
        page = await self.client.inbox(wait=self.wait_s)
        for stuck in page.get("needsAttention") or []:
            if stuck["messageId"] not in self._reported:
                self._reported.add(stuck["messageId"])
                logger.warning("Conduyt turn %s needs attention (%s); it is not re-run automatically", stuck["messageId"],
                               "possibly replied" if stuck.get("possiblyReplied") else "started and never completed")
        await self._answer_tests(page.get("tests") or [])
        for pending in page.get("repliedAckPending") or []:
            await self._ack_until_done(pending["seq"], pending["messageId"], pending["generation"])
        deferred = False
        for event in page["events"]:
            if self._stopping.is_set():
                return
            if event["id"] in self._running:
                self.stats["skipped"] += 1   # a duplicate delivery of a running turn: never a second run
                continue
            if self._free_slots() <= 0 or thread_key(event) in self._busy_threads:
                deferred = True             # left unclaimed on the server: served again, never dropped
                continue
            if not self.lock_check().ok:
                return                      # the next loop pauses and publishes why
            started = await self._claim_and_start(event)
            if started is None:
                return                      # held: nothing after it is claimed
        if not deferred and self._held is None:
            last = page["events"][-1]["agentInboxSeq"] if page["events"] else page.get("afterSeq", -1)
            if page["nextSeq"] > last:
                try:
                    await self.client.ack(page["nextSeq"])
                except ApiError as err:
                    if err.code not in ("cursor_ahead_of_unclaimed", "agent_inbox_seq_ahead"):
                        raise
        if deferred:
            # wait for a turn to finish before asking again (the deferred events are served again at once), unless one already
            # finished while this page was handled
            self._turn_done.clear()
            if self._completions == seen and self._running:
                await self._turn_done.wait()

    async def _resolve_held(self) -> bool:
        event = self._held
        try:
            answer = await self.client.claim(event["id"])
        except ApiError as err:
            if err.code in ("agent_event_done", "agent_event_claimed", "agent_event_not_in_inbox"):
                self._held = None
                return True
            raise
        self._held = None
        self._start(event, answer)
        return True

    async def _claim_and_start(self, event: dict) -> Optional[bool]:
        try:
            answer = await self._until_definite(f"claiming {event['id']}", lambda: self.client.claim(event["id"]), [None])
        except OutcomeUnknown:
            self._held = event
            logger.warning("Conduyt: paused at message %s; its claim's outcome is unknown (holding there)", event["id"])
            return None
        except ApiError as err:
            if err.code in ("agent_event_done", "agent_event_claimed", "agent_event_not_in_inbox"):
                self.stats["skipped"] += 1
                return False
            raise
        if self._stopping.is_set():
            # stop() ran while this claim was in flight: hand the turn back at once instead of starting it after shutdown
            try:
                await self.client.release(event["id"], answer["generation"])
                self.stats["released"] += 1
            except Exception as exc:
                logger.warning("Conduyt turn %s claimed during shutdown could not be handed back (%s)", event["id"], type(exc).__name__)
            return False
        self._start(event, answer)
        return True

    def _start(self, page_event: dict, answer: dict) -> None:
        # the claim's event is the one to run (the server rebuilt it after recording any fold); the page's copy is the fallback
        turn = Turn(event=answer.get("event") or page_event, generation=answer["generation"])
        self._busy_threads.add(turn.thread_key)
        task = asyncio.create_task(self._run_turn(turn))
        self._running[turn.id] = (task, turn)

    async def _answer_tests(self, tests: list) -> None:
        """The card's hello Test, as the reference listener answers it (conduyt-cli src/agent-listener.ts answerTests :1523): a
        test is remembered as answered only AFTER its hello posted; a definite refusal (isHelloRefusal :1519) is remembered so it
        is not retried; anything else (a network failure, a 5xx, a 429) is tried again on the next poll."""
        for test in tests:
            key = test.get("idempotencyKey")
            if not key or key in self._helloed or key in self._hello_refused:
                continue
            try:
                await self.client.hello(test)
            except ConfigError:
                self._hello_refused.add(key)
                logger.warning("Conduyt connection test asked for a text other than the fixed hello; not posted")
                continue
            except ApiError as err:
                if hello_refusal(err):
                    self._hello_refused.add(key)
                    logger.warning("Conduyt connection test refused (%s); it is not retried", err.code or err.status)
                    continue
                if err.fatal:
                    raise StopListener(fatal_text(err)) from None
                logger.warning("Conduyt connection test failed (%s); tried again on the next poll", err.status)
                continue
            except Exception as err:
                logger.warning("Conduyt connection test failed (%s); tried again on the next poll", type(err).__name__)
                continue
            self._helloed.add(key)
            logger.info("Conduyt connection test answered (hello posted in AI Team Chat)")

    async def _ack_until_done(self, seq: int, message_id: str, generation: int) -> None:
        failures = 0
        while True:
            try:
                await self.client.ack(seq, message_id, generation)
                return
            except ApiError as err:
                if err.code in ("agent_claim_missing", "agent_inbox_seq_ahead"):
                    return
                if err.fatal:
                    raise StopListener(fatal_text(err)) from None
                if not is_transient(err):
                    logger.warning("Conduyt acknowledgement of %s refused (%s)", message_id, err)
                    return
            except Exception as err:
                if not is_transient(err):
                    raise
            failures += 1
            if self._stopping.is_set() and failures > 3:
                return
            await asyncio.sleep(backoff_delay(failures, base=self.base_backoff_s, cap=self.cap_backoff_s, rand=self.rand))

    # ---------------------------------------------------------------------------------------------------------- one turn

    async def _renew_loop(self, turn: Turn) -> None:
        while True:
            await asyncio.sleep(self.lease_renew_s)
            try:
                await self.client.renew(turn.id, turn.generation)
            except Exception as err:
                logger.warning("Conduyt lease renewal for %s failed: %s", turn.id, type(err).__name__)

    async def _run_turn(self, turn: Turn) -> None:
        lease = asyncio.create_task(self._renew_loop(turn))
        deadline: list = [None]
        try:
            try:
                outcome = await asyncio.wait_for(self.dispatch(turn), timeout=self.turn_timeout_s)
            except asyncio.TimeoutError:
                outcome = TurnOutcome("failed", reason="turn_timeout")
            except asyncio.CancelledError:
                raise
            except Exception as err:
                logger.error("Conduyt turn %s failed in Hermes: %s", turn.id, type(err).__name__)
                outcome = TurnOutcome("failed")
            clean = False
            try:
                clean = await self._finish(turn, outcome, deadline)
            finally:
                # EVERY ending after the model ran that is not clean (see _finish) rolls the thread over: a
                # refused or failed silent / reply / fail call, a lost claim, an answer not posted, any exception
                if outcome.kind != "release" and not clean:
                    self._roll(turn)
        except OutcomeUnknown as unknown:
            logger.error("Conduyt turn %s: %s; marked possibly replied (not re-run)", turn.id, unknown)
            try:
                await self._mark_failed(turn, None, possibly_replied=True)
            except Exception:
                pass
            self.stats["failed"] += 1
        except StopListener as stop:
            self._fatal = str(stop)
            self.status("fatal", str(stop))
            self._stopping.set()
        except asyncio.CancelledError:
            raise
        except Exception as err:
            logger.error("Conduyt turn %s could not be completed: %s", turn.id, redact(err))
        finally:
            lease.cancel()
            self._running.pop(turn.id, None)
            self._busy_threads.discard(turn.thread_key)
            self._completions += 1
            self._turn_done.set()

    def _roll(self, turn: Turn) -> None:
        """The thread rollover (on_failed: the adapter's once-per-claim _turn_failed). Never raises."""
        try:
            self.on_failed(turn)
        except Exception as exc:
            logger.warning("Conduyt turn %s: the thread rollover failed (%s)", turn.id, type(exc).__name__)

    async def _mark_failed(self, turn: Turn, deadline: Optional[list], *, reason: Optional[str] = None,
                           possibly_replied: bool = False) -> None:
        """THE one place a claim is ended as failed / needs attention: first the thread's rollover (on_failed: the next
        message starts a fresh Hermes session), then the fail call (retried until definite when a deadline is given)."""
        self._roll(turn)
        c = self.client
        if deadline is None:
            await c.fail(turn.id, turn.generation, possibly_replied=possibly_replied, reason=reason)
            return
        await self._until_definite(f"failing {turn.id}",
                                   lambda: c.fail(turn.id, turn.generation, possibly_replied=possibly_replied, reason=reason),
                                   deadline)

    async def _finish(self, turn: Turn, outcome: TurnOutcome, deadline: list) -> bool:
        """End the claim for ``outcome``. Returns True ONLY for a clean ending, where the thread's Hermes session may go on:
        the turn was handed back unrun (release: the dispatcher returns it only before Hermes ran); the silent call
        answered a schema-valid silent result with outcome silent, or a schema-valid 409 turn_finished whose outcome is
        silent; the model's answer was posted (a 2xx) and acknowledged, or the post got a schema-valid 409 turn_finished
        whose outcome is replied (:func:`finished_as`). Every
        other exit (a failure, an answer not posted, a refusal, any exception) returns False or raises, and _run_turn then
        rolls the thread over (:meth:`_roll`), so the next message starts a fresh Hermes session."""
        c = self.client
        if outcome.kind == "release":
            await self._until_definite(f"handing back {turn.id}", lambda: c.release(turn.id, turn.generation), deadline)
            self.stats["released"] += 1
            return True
        if outcome.kind == "failed":
            reason = "turn_timeout" if outcome.reason == "turn_timeout" else None
            try:
                await self._mark_failed(turn, deadline, reason=reason)
            except ApiError as err:
                if err.code != "agent_claim_lost":
                    raise
            self.stats["failed"] += 1
            logger.warning("Conduyt turn %s needs attention (%s); it is not re-run automatically", turn.id, outcome.reason or "failed")
            return False
        if outcome.kind == "silent" and requires_answer(turn.event):
            # a mention / DM / hand-off / folded turn the model did not answer (NO_REPLY, or nothing left to post): not a silent ending,
            # which Conduyt shows as nothing at all; the typed failure instead (needs attention, the thread rolled over).
            # Nothing is posted: the plugin never writes an answer of its own
            logger.warning("Conduyt turn %s: the model gave no answer to a message addressed to it; nothing posted", turn.id)
            return await self._finish(turn, TurnOutcome("failed", reason="no_answer"), deadline)
        if outcome.kind == "silent":
            clean = False
            try:
                answer = await self._until_definite(f"going silent on {turn.id}", lambda: c.silent(turn.id, turn.generation),
                                                    deadline)
                # a schema-valid silent result (validated in protocol.parse_answer) that says the turn ended SILENT
                clean = isinstance(answer, dict) and answer.get("outcome") == "silent"
            except ApiError as err:
                if err.fatal:
                    raise StopListener(fatal_text(err)) from None
                if err.code != "turn_finished":
                    raise
                # turn_finished only proves the turn ended, not how: clean only when Conduyt says it ended
                # silent; replied / reacted / paused / trigger_deleted / no outcome roll the thread over
                clean = finished_as(err, "silent")
            self.stats["silent"] += 1
            return clean
        text = finalize_reply(outcome.text or "")
        if text is None:
            logger.warning("Conduyt turn %s: the answer holds something shaped like a Conduyt key or pair code; nothing "
                           "posted", turn.id)
            return await self._finish(turn, TurnOutcome("failed"), deadline)
        if not text:
            return await self._finish(turn, TurnOutcome("silent"), deadline)
        posted = True
        try:
            await self._until_definite(f"posting the reply to {turn.id}", lambda: c.reply(turn.event, text, turn.generation), deadline)
        except ApiError as err:
            if err.fatal:
                try:
                    await self._mark_failed(turn, None)
                except Exception:
                    pass
                raise StopListener(fatal_text(err)) from None
            # the turn already ended, the thread waits for a person, agents are muted there, or its message was deleted:
            # nothing (more) to post; the turn is acknowledged like any finished one
            if not (err.status == 409 and err.code in ("turn_finished", "agent_human_required", "agents_muted_in_channel", "parent_deleted")):
                return await self._finish(turn, TurnOutcome("failed"), deadline)
            # clean only when Conduyt says the turn ended REPLIED (our idempotent post already landed); any other
            # or missing outcome, and every other refusal, means this answer never reached Conduyt: roll over
            posted = err.code == "turn_finished" and finished_as(err, "replied")
        self.stats["replied"] += 1
        seq = turn.event.get("agentInboxSeq") or 0
        await self._ack_until_done(seq, turn.id, turn.generation)
        return posted


#: the turn_finished outcome that matches each requested ending (contract AGENT_CLAIM_CONFLICT_SCHEMA.properties.outcome:
#: replied, reacted, silent, paused, trigger_deleted; paused / trigger_deleted: the server ended it, nothing was posted)
TURN_FINISHED_MATCH = {"silent": "silent", "reply": "replied"}


def finished_as(err: ApiError, wanted: str) -> bool:
    """Whether a refusal CONFIRMS the turn ended exactly as ``wanted``: status exactly 409, a body that validates against the
    contract's whole AGENT_CLAIM_CONFLICT_SCHEMA (err.conflict, decided in protocol.parse_answer), code turn_finished and
    the outcome Conduyt reported equal to ``wanted``. Anything else (another status, an off-contract body, a missing or
    different outcome) is False."""
    return (err.status == 409 and getattr(err, "conflict", False) is True and err.code == "turn_finished"
            and (err.details or {}).get("outcome") == wanted)


class OutcomeUnknown(Exception):
    pass


def hello_refusal(err: ApiError) -> bool:
    """A definite refusal of the hello (the reference listener's isHelloRefusal): a 4xx other than 401/402/404/408/425/429, and a
    403 only when it carries a code."""
    return 400 <= err.status < 500 and err.status not in (401, 402, 404, 408, 425, 429) and (err.status != 403 or bool(err.code))


def budget_sleep_s(err: ApiError) -> float:
    from datetime import datetime, timezone
    reset = err.details.get("resetAt")
    try:
        until = (datetime.fromisoformat(str(reset).replace("Z", "+00:00")) - datetime.now(timezone.utc)).total_seconds()
    except ValueError:
        until = BUDGET_SLEEP_CAP_S
    return max(1.0, min(BUDGET_SLEEP_CAP_S, until))


def fatal_text(err: ApiError) -> str:
    if err.status == 401:
        return ("Conduyt refused this teammate's key (revoked, or paired again elsewhere). In Conduyt Settings > AI press Re-pair, "
                "paste the new pair code into Messaging > Conduyt in Hermes; it reconnects within a minute.")
    if err.status == 403:
        return "Conduyt refused this teammate's key for this action (permission). Check the teammate's owner still manages AI teammates."
    if err.status == 404:
        return "Conduyt AI teammates are switched off for this workspace (or the address is wrong)."
    return f"Conduyt answered {err.status}; the listener stopped."
