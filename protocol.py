"""Conduyt's AI-teammate LISTENER protocol, client side (outbound HTTPS only; no Hermes import).

The same wire contract as the npm CLI listener (conduyt-cli src/agent-listener.ts httpListenerApi) and the app routes under
app/api/v1/chat/agents/:
  POST /api/v1/chat/agents/pair                 {code}                         -> {apiKey, agent{...}, rateLimit}
  POST /api/v1/chat/agents/me/heartbeat         {listenerId, version}
  GET  /api/v1/chat/agents/me/inbox?wait=25[&afterSeq=]                        -> the inbox page (long-poll)
  POST /api/v1/chat/agents/me/inbox/claim       {listenerId, messageId, action, generation?, possiblyReplied?, reason?}
  POST /api/v1/chat/agents/me/inbox/ack         {seq, messageId?, listenerId?, generation?}
  POST /api/v1/chat/channels/{id}/messages      {body, parentId, idempotencyKey: reply:<messageId>, claimToken: <gen>:<listenerId>}
                                                 or the connection test's hello {body, testToken, idempotencyKey}
Answers are `{data: ...}`; every answer the loop acts on is validated against the published schemas (schema.py) and an answer
that fails them is an UNKNOWN outcome (TransportError), never a definite one.

Secrets: the key only ever travels in the Authorization header to the configured Conduyt origin. No error text from the server
is kept (a proxy may echo a request): only the status, a strict-pattern code and a few typed fields.
"""
from __future__ import annotations

import copy
import ipaddress
import json
import random
import re
import socket
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

from . import schema as S

DEFAULT_BASE_URL = "https://conduyt.app"
CLIENT_NAME = "hermes-conduyt-plugin"
#: this plugin's own version, sent with every heartbeat beside x-conduyt-client: hermes-conduyt-plugin, which the app keys the
#: Settings card's Test on (app lib/agent-webhook-connect.ts AGENT_LISTENER_TEST_CLIENTS); keep equal to plugin.yaml's version
PLUGIN_VERSION = "0.1.2"
REQUEST_TIMEOUT_S = 30.0
INBOX_WAIT_S = 25
KEY_RE = re.compile(r"^cdy_[0-9a-f]{48}$")
PAIR_CODE_RE = re.compile(r"^[0-9A-F]{16}$")
LISTENER_ID_RE = re.compile(r"^[A-Za-z0-9:_.-]{8,128}$")
_CODE_RE = re.compile(r"^[a-z0-9_]{1,64}$")
# anything shaped like a Conduyt key, even truncated or upper-cased, and a bare pair code
_SECRET_PATTERNS = (re.compile(r"cdy_[A-Za-z0-9]{4,}", re.I), re.compile(r"\b[0-9A-Fa-f]{16}\b"))


def redact(text: Any) -> str:
    """``text`` with every Conduyt-key-shaped and pair-code-shaped run replaced: for logs and status lines only. A posted
    answer is never altered: an answer this would change is not posted at all (listener.finalize_reply, adapter
    _dispatch_once)."""
    out = str(text)
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("[redacted]", out)
    return out


def new_listener_id() -> str:
    return f"hermes-{uuid.uuid4()}"


class ConduytError(Exception):
    """Base class."""


class ConfigError(ConduytError):
    """A local configuration problem (an unsafe address, a malformed credential): never retried."""


class TransportError(ConduytError):
    """The outcome is UNKNOWN: a network failure, a timeout, an unreadable or off-schema 2xx."""


class ApiError(ConduytError):
    """A definite refusal: status, a strict-pattern code and the few typed fields the loop acts on."""

    def __init__(self, status: int, code: Optional[str], details: dict[str, Any], *, conflict: bool = False):
        self.status, self.code, self.details = status, code, details
        #: True ONLY for a 409 whose whole body validates against the contract's AGENT_CLAIM_CONFLICT_SCHEMA
        #: (decided once, in parse_answer); a refusal the listener may read as a confirmed ending must carry it
        self.conflict = conflict
        super().__init__(f"Conduyt answered {status}{f' {code}' if code else ''}")

    @property
    def fatal(self) -> bool:
        """401 / 402 / 403 / 404: a revoked key, the plan, a permission, the feature switched off. Retrying cannot fix them."""
        return self.status in (401, 402, 403, 404)

    @property
    def budget_exhausted(self) -> bool:
        return self.status == 429 and self.code == "agent_daily_budget_exhausted"


def is_transient(error: BaseException) -> bool:
    """Retry with backoff: an unknown outcome, a 5xx, or a 429 that is not the daily turn budget."""
    if isinstance(error, ApiError):
        return error.status >= 500 or (error.status == 429 and not error.budget_exhausted)
    return isinstance(error, (TransportError, httpx.HTTPError, OSError))


def backoff_delay(failures: int, *, base: float = 1.0, cap: float = 60.0, rand=random.random) -> float:
    """Exponential backoff with full jitter in its upper half: base * 2^(n-1), capped, then 50-100% of that."""
    raw = min(cap, base * (2 ** max(0, failures - 1)))
    return raw * (0.5 + 0.5 * rand())


# ---------------------------------------------------------------------------------------------------------------------- address


def _is_private(ip: ipaddress._BaseAddress) -> bool:
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified)


def validate_base_url(raw: str, *, allow_insecure: bool = False) -> str:
    """The Conduyt origin to talk to (``https://host[:port]``). https only, no credentials or path, and never a private, loopback
    or link-local literal unless ``allow_insecure`` (a local development flag) is set. Raises ConfigError."""
    value = (raw or "").strip() or DEFAULT_BASE_URL
    parts = urlsplit(value)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise ConfigError("The Conduyt address must look like https://conduyt.app")
    if parts.scheme == "http" and not allow_insecure:
        raise ConfigError("The Conduyt address must use https (http is allowed only with CONDUYT_ALLOW_INSECURE_DEV=1)")
    if parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ConfigError("The Conduyt address must be the site address only, for example https://conduyt.app")
    host = parts.hostname.lower()
    if not allow_insecure:
        if host in ("localhost",) or host.endswith(".localhost") or host.endswith(".local") or host.endswith(".internal"):
            raise ConfigError("The Conduyt address points at this computer or a private network")
        try:
            if _is_private(ipaddress.ip_address(host)):
                raise ConfigError("The Conduyt address points at this computer or a private network")
        except ValueError:
            pass  # a name, checked again once resolved (check_resolved_host)
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}"


async def check_resolved_host(base_url: str, *, allow_insecure: bool = False, resolver=None) -> None:
    """Refuse a name that resolves to a private address (DNS pointing the plugin at the user's own network). Raises ConfigError."""
    if allow_insecure:
        return
    host = urlsplit(base_url).hostname or ""
    try:
        infos = await (resolver(host) if resolver else _resolve(host))
    except OSError as exc:
        raise TransportError(f"could not resolve the Conduyt address ({type(exc).__name__})") from exc
    for addr in infos:
        try:
            if _is_private(ipaddress.ip_address(addr)):
                raise ConfigError("The Conduyt address resolves to a private network address")
        except ValueError:
            continue


async def _resolve(host: str) -> list[str]:
    import asyncio
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


# ---------------------------------------------------------------------------------------------------------------------- client


def claim_token(generation: int, listener_id: str) -> str:
    """The reply's capability for the claimed turn: ``<generation>:<listenerId>`` (the app's parseAgentClaimToken)."""
    return f"{generation}:{listener_id}"


HELLO_TEXT = "Hello, I'm connected."


def reply_key(message_id: str) -> str:
    """Every reply carries ``reply:<source message id>``: a retried post gets the first reply back instead of a second one."""
    return f"reply:{message_id}"


def _safe_details(raw: Any, headers: httpx.Headers) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        raw = {}
    if isinstance(raw.get("messageId"), str) and S.UUID_RE.match(raw["messageId"]):
        out["messageId"] = raw["messageId"]
    for key in ("seq", "generation", "turnsToday", "budget"):
        if isinstance(raw.get(key), int) and not isinstance(raw.get(key), bool):
            out[key] = raw[key]
    if raw.get("outcome") in ("replied", "reacted", "silent", "paused", "trigger_deleted"):
        out["outcome"] = raw["outcome"]
    if isinstance(raw.get("resetAt"), str) and len(raw["resetAt"]) <= 40:
        out["resetAt"] = raw["resetAt"]
    retry_after = headers.get("retry-after")
    if retry_after and retry_after.strip().isdigit():
        out["retryAfter"] = int(retry_after.strip())
    return out


@dataclass
class ConduytClient:
    """One listener's HTTP calls. ``api_key`` is sent only as a Bearer header to ``base_url``; it is never logged or returned."""

    base_url: str
    api_key: str = field(repr=False)
    listener_id: str = field(default_factory=new_listener_id)
    version: str = PLUGIN_VERSION
    transport: Optional[httpx.AsyncBaseTransport] = None
    timeout_s: float = REQUEST_TIMEOUT_S
    _http: Optional[httpx.AsyncClient] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not KEY_RE.match(self.api_key or ""):
            raise ConfigError("The Conduyt teammate key is malformed")
        if not LISTENER_ID_RE.match(self.listener_id):
            raise ConfigError("listener id is malformed")

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            # follow_redirects=False: a redirect never carries the key to another origin
            self._http = httpx.AsyncClient(base_url=self.base_url, transport=self.transport, follow_redirects=False,
                                           timeout=self.timeout_s, headers={"x-conduyt-client": CLIENT_NAME})
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _call(self, method: str, path: str, *, body: Optional[dict] = None, params: Optional[dict] = None,
                    answer_schema: Optional[dict] = None, timeout: Optional[float] = None) -> Any:
        headers = {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        try:
            response = await self._client().request(method, path, json=body, params=params, headers=headers,
                                                    timeout=timeout or self.timeout_s)
            text = response.text
        except httpx.HTTPError as exc:
            raise TransportError(f"{path}: {type(exc).__name__} (outcome unknown)") from None
        return parse_answer(path, response.status_code, text, response.headers, answer_schema)

    # -- the protocol ----------------------------------------------------------------------------------------------------

    async def heartbeat(self) -> None:
        await self._call("POST", "/api/v1/chat/agents/me/heartbeat", body={"listenerId": self.listener_id, "version": self.version},
                         answer_schema=HEARTBEAT_ANSWER)

    async def inbox(self, *, wait: int = INBOX_WAIT_S, after_seq: Optional[int] = None) -> dict:
        params: dict[str, Any] = {"wait": wait}
        if after_seq is not None:
            params["afterSeq"] = after_seq
        return await self._call("GET", "/api/v1/chat/agents/me/inbox", params=params,
                                answer_schema=S.schema("AGENT_INBOX_PAGE_SCHEMA"), timeout=wait + 15)

    async def _claim(self, payload: dict, answer_schema: Optional[dict] = None) -> Any:
        return await self._call("POST", "/api/v1/chat/agents/me/inbox/claim", body={"listenerId": self.listener_id, **payload},
                                answer_schema=answer_schema)

    async def claim(self, message_id: str) -> dict:
        return await self._claim({"messageId": message_id, "action": "claim"}, S.schema("AGENT_CLAIM_ANSWER_SCHEMA"))

    async def renew(self, message_id: str, generation: int) -> None:  # answer: the contract's "renew / release / fail" branch
        await self._claim({"messageId": message_id, "action": "renew", "generation": generation},
                          claim_result("renew / release / fail"))

    async def release(self, message_id: str, generation: int) -> None:
        await self._claim({"messageId": message_id, "action": "release", "generation": generation},
                          claim_result("renew / release / fail"))

    async def fail(self, message_id: str, generation: int, *, possibly_replied: bool = False, reason: Optional[str] = None) -> None:
        payload: dict[str, Any] = {"messageId": message_id, "action": "fail", "generation": generation}
        if possibly_replied:
            payload["possiblyReplied"] = True
        if reason:
            payload["reason"] = reason
        await self._claim(payload, claim_result("renew / release / fail"))

    async def silent(self, message_id: str, generation: int) -> dict:
        """The silent ending; the 2xx answer validated against the contract's silent result: the caller reads
        its outcome to confirm the turn ended silent."""
        return await self._claim({"messageId": message_id, "action": "silent", "generation": generation}, claim_result("silent"))

    async def status(self, message_id: str, generation: Optional[int] = None) -> dict:
        payload: dict[str, Any] = {"messageId": message_id, "action": "status"}
        if generation is not None:
            payload["generation"] = generation
        return await self._claim(payload, S.schema("AGENT_STATUS_ANSWER_SCHEMA"))

    async def ack(self, seq: int, message_id: Optional[str] = None, generation: Optional[int] = None) -> None:
        body: dict[str, Any] = {"seq": seq}
        if message_id is not None:
            body.update({"messageId": message_id, "listenerId": self.listener_id, "generation": generation})
        await self._call("POST", "/api/v1/chat/agents/me/inbox/ack", body=body, answer_schema=ACK_ANSWER)

    async def reply(self, event: dict, text: str, generation: int) -> None:
        body = {"body": text, "parentId": event.get("parentId") or event["id"], "idempotencyKey": reply_key(event["id"]),
                "claimToken": claim_token(generation, self.listener_id)}
        await self._post_message(str(event["channelId"]), body)

    async def _post_message(self, channel_id: str, body: dict) -> dict:
        """POST a Team Chat message; a success is ONLY the pinned MESSAGE_POSTED shape for THIS channel (the contract has no
        schema for this endpoint), anything else is an unknown outcome."""
        data = await self._call("POST", f"/api/v1/chat/channels/{channel_id}/messages", body=body, answer_schema=MESSAGE_POSTED)
        if data.get("channelId") != channel_id:
            raise TransportError("a posted message's answer names another channel (outcome unknown)")
        return data

    async def hello(self, test: dict) -> None:
        """The connection test's hello: the ONE text the plugin posts that is not a model's answer, and only this fixed text
        (Conduyt's AGENT_WEBHOOK_HELLO_TEXT); a test asking for any other text is refused here, nothing posted."""
        if test.get("hello") != HELLO_TEXT:
            raise ConfigError("the connection test asked for a text other than the fixed hello; nothing posted")
        body = {"body": HELLO_TEXT, "testToken": test["testToken"], "idempotencyKey": test["idempotencyKey"]}
        await self._post_message(str(test["channelId"]), body)


#: answers the contract does NOT define, pinned to what the Conduyt app's routes return (read from the app's source; adding
#: them to the contract is a follow-up on the app side):
#:   MESSAGE_POSTED  app/api/v1/chat/channels/[id]/messages/route.ts POST: 201 jsonResponse(hydrated message) on a new post,
#:                   200 jsonResponse({...replayed, idempotentReplay: true}) on an idempotent replay; the message's id and
#:                   channelId are always present (uuids)
#:   ACK_ANSWER      app/api/v1/chat/agents/me/inbox/ack/route.ts: jsonResponse({ cursor }) with cursor a number (null is a 409)
#:   HEARTBEAT_ANSWER app/api/v1/chat/agents/me/heartbeat/route.ts: jsonResponse({ listening: true, at })
#: the only success statuses the Conduyt app sends (jsonResponse defaults to 200; a new message is 201)
SUCCESS_STATUSES = (200, 201)
MESSAGE_POSTED = {"type": "object", "required": ["id", "channelId"],
                  "properties": {"id": {"type": "string", "format": "uuid"}, "channelId": {"type": "string", "format": "uuid"}}}
ACK_ANSWER = {"type": "object", "required": ["cursor"], "properties": {"cursor": {"type": "integer"}}}
HEARTBEAT_ANSWER = {"type": "object", "required": ["listening"], "properties": {"listening": {"type": "boolean", "enum": [True]}}}


def claim_result(title: str) -> dict:
    """One branch of the contract's AGENT_CLAIM_RESULT_SCHEMA, by its title (the answer of one claim action)."""
    return next(b for b in S.schema("AGENT_CLAIM_RESULT_SCHEMA")["oneOf"] if b.get("title") == title)


def parse_answer(path: str, status: int, text: str, headers: httpx.Headers, answer_schema: Optional[dict]) -> Any:
    """A 2xx's data (validated) or the ApiError of a refusal. An unreadable or off-schema 2xx is UNKNOWN."""
    try:
        parsed = json.loads(text) if text else {}
    except ValueError:
        if 200 <= status < 300:
            raise TransportError(f"{path}: an unreadable {status} answer (outcome unknown)") from None
        parsed = {}
    if not 200 <= status < 300:
        code = parsed.get("code") if isinstance(parsed, dict) else None
        conflict = status == 409 and isinstance(parsed, dict) and S.matches(S.schema("AGENT_CLAIM_CONFLICT_SCHEMA"), parsed,
                                                                            formats=True)
        raise ApiError(status, code if isinstance(code, str) and _CODE_RE.match(code) else None, _safe_details(parsed, headers),
                       conflict=conflict)
    # every Conduyt success is a JSON OBJECT (the app's jsonResponse wraps it as {"data": {...}}): an empty
    # body, a 204, null, a list, a string or a non-object data is an UNKNOWN outcome for EVERY call, never a success
    if status not in SUCCESS_STATUSES:
        raise TransportError(f"{path}: a {status} answer, which Conduyt never sends for a success (outcome unknown)")
    if not text or not isinstance(parsed, dict):
        raise TransportError(f"{path}: an empty or non-object {status} answer (outcome unknown)")
    data = parsed.get("data", parsed)
    if not isinstance(data, dict):
        raise TransportError(f"{path}: a {status} answer whose data is not an object (outcome unknown)")
    if answer_schema is not None and not S.matches(answer_schema, data, formats=True):
        raise TransportError(f"{path}: a {status} answer that fails its schema (outcome unknown)")
    return data


def pair_answer_valid(data: Any) -> bool:
    """The pair answer: the published schema, except that ``agent.provider`` may be any string (a newer server may name a
    provider this copy of the contract does not list; the plugin never acts on it)."""
    pair = copy.deepcopy(S.schema("AGENT_PAIR_RESULT_SCHEMA"))
    pair["properties"]["agent"]["properties"]["provider"] = {"type": "string"}
    return S.matches(pair, data, formats=True)


async def pair(base_url: str, code: str, *, transport: Optional[httpx.AsyncBaseTransport] = None) -> dict:
    """Exchange a single-use pair code for the teammate key. Raises ApiError (401: invalid or expired) / TransportError."""
    code = (code or "").strip().upper()
    if not PAIR_CODE_RE.match(code):
        raise ConfigError("That is not a Conduyt pair code (16 letters and digits)")
    async with httpx.AsyncClient(base_url=base_url, transport=transport, follow_redirects=False, timeout=REQUEST_TIMEOUT_S,
                                 headers={"x-conduyt-client": CLIENT_NAME}) as http:
        try:
            response = await http.post("/api/v1/chat/agents/pair", json={"code": code})
            text = response.text
        except httpx.HTTPError as exc:
            raise TransportError(f"pairing: {type(exc).__name__} (outcome unknown)") from None
    data = parse_answer("/api/v1/chat/agents/pair", response.status_code, text, response.headers, None)
    if not pair_answer_valid(data):
        raise TransportError("The pair answer is incomplete; nothing was stored")
    return data
