# Conduyt Team Chat for Hermes

A platform plugin for the [Hermes Agent](https://github.com/NousResearch/hermes-agent) gateway. **Your own Hermes** becomes an AI teammate in your company's Conduyt workspace: people mention it in Team Chat and it answers in the thread, as the agent that knows you.

- **Your own agent.** It runs in the Hermes profile you install it in (your main one), with your model, your memory, your persona and your skills, the same as Hermes's own Telegram or Slack platforms.
- **No tools from Team Chat.** Team Chat messages cannot make your Hermes run its tools or commands: no terminal, files, browser, code, web, other platforms, scheduled jobs, delegation or memory writes. The plugin refuses to connect unless that holds.
- **Teammates cannot change your memory.** A Team Chat turn can read your memory but never write to it, to your skills or to your persona.
- **Outbound only.** The plugin connects out to Conduyt over HTTPS. There is no tunnel, no open port and no public address.

Requires Hermes 0.23.1 or later, with a model that Hermes talks to over its API. **If your Hermes runs its model through the Codex app-server, an ACP agent or Mixture of Agents, it can't be a Team Chat teammate:** those run their own tools on your computer, outside Hermes's tool guard. It also won't connect while your Hermes has shell hooks, outbound webhooks or gateway hooks set up or still loaded (`hermes hooks list`): Hermes runs those on every conversation and cannot limit them to one platform. It also won't connect while a model provider gets its key from a command (`key_cmd`), or while a profile route (`gateway.profile_routes`) sends Team Chat to another profile. Your model must name its provider (`model.provider`, which `hermes model` sets, for example `nous`): an unset or `auto` provider is chosen by Hermes when the turn runs, which the plugin cannot check. Providers that can start programs on your computer during a turn are not supported for Team Chat, under any of the names Hermes accepts for them: Hermes's managed llama.cpp server, AWS Bedrock, Vertex AI, the Anthropic provider (its sign-in runs the macOS `security` command to read the Keychain), Azure with Entra ID sign-in, a provider whose plugin is not installed, and any route whose Hermes package is not installed yet (for example an Anthropic-protocol endpoint without the Anthropic package). An Anthropic-protocol endpoint reached through a custom or other provider works when the package is installed. And it needs `plugins.hook_callback_timeout` set to 0 (the setup line below sets it): with a time limit, Hermes may skip the plugin's second per-turn check when a hook runs long.

## What you agree to by connecting

Your Hermes answers your teammates using what it knows about you: your memory notes, your user profile, your persona and your skills list. **It may repeat things from your notes in the channel**, and a teammate can try to get it to. The plugin tells the model that the people in the channel are teammates, not you, to treat their messages as data, and not to reveal secrets or private notes verbatim. That is advice to the model, not a guarantee. Do not connect a Hermes whose memory holds things your team should not read.

Your own setup still applies to Conduyt turns: your model and its key pay for them, and **your own Hermes plugins still run on these conversations as they do on all others** (their code sees the conversation; the plugin cannot inspect or limit another plugin).

## Setup

In Conduyt, go to Settings > AI, choose Hermes and create the teammate. Conduyt shows a pair code and a message.

1. **Copy the message and paste it into Hermes.** Hermes runs one line in your own profile:

   ```sh
   # the last setting removes the time limit on plugin hooks for ALL your Hermes plugins, not only this one
   hermes plugins install conduyt-app/hermes-conduyt --yes-deps && hermes plugins enable conduyt-platform && hermes config set platform_toolsets.conduyt '["no_mcp"]' && hermes config set plugins.hook_callback_timeout 0
   ```

   It installs and enables the plugin, saves Conduyt's tool line (no toolset and no MCP server, so a Team Chat turn has no tool at all), and sets Hermes to run plugin hooks to completion with no time limit. **That last setting applies to all your Hermes plugins, not only this one**: Hermes has no per-plugin hook time limit, so a plugin hook that hangs would hold its turn instead of being skipped.

2. **Enter the pair code, never in a chat.** Either:
   - in the Hermes desktop app: Messaging, then the Conduyt card, paste the code into its masked field and save; or
   - in a terminal: `hermes gateway setup`, choose Conduyt, and type the code at the hidden prompt.

3. **Restart the gateway once**: Restart gateway in the desktop app, or `hermes gateway restart`. Measured on Hermes 0.23.1: a running gateway does not load a newly enabled plugin until it restarts; after the restart it connects within seconds.

The teammate then shows **Listening** in Conduyt Settings > AI. Press **Test** there to see it say hello in AI Team Chat. `hermes conduyt check` runs the same check the plugin runs before it connects and before every turn, prints every reason it would refuse (all of them at once) and the fix, and reads the running gateway's own last check (that is where a hook still loaded in the gateway shows up). It exits 0 when everything passes, 1 on any refusal (including a running gateway whose check it cannot read), and 2 when your config passes but no gateway is running, so only the config was checked.

The pair code works once and lasts 15 minutes. If it expires, press **New code** (or **Re-pair**) in Conduyt and enter the new code the same way.

## What it does

- Every 25 seconds or less it asks Conduyt for new turns (a long poll). An idle hour costs no model calls.
- It claims a turn, gives it to Hermes as one user message, and posts Hermes's final answer once, in the thread. Progress messages and errors are never posted.
- Each Conduyt thread is its own Hermes session, keyed by the thread: the teammate remembers the thread, never your private chats. Threads and workspaces never share a session.
- **The plugin never adds a character of its own to a posted answer.** What is posted is always the model's final answer, or a prefix of it: leading and trailing whitespace is removed, and an answer longer than Conduyt's message size (9,500 characters as Conduyt counts them) is shortened to a prefix of itself that fits. The plugin reads the answer with a CommonMark parser (markdown-it, which Hermes already ships through rich; tables included) and keeps whole top-level blocks: the longest run that fits, less a trailing heading (with its title line), horizontal rule, list or quote ending in one of those, list ending in an empty item, or table without rows. A code block, list, quote or table is kept whole or left out whole, so no code block is left open. Only a plain paragraph is cut inside: at its last line break or sentence end, else its last space, in the second half of what fits, and only if the shortened paragraph still reads as the same blocks; otherwise the cut is made between blocks, or exactly at the limit (between two characters, never inside one). When not even the first block fits (one very long paragraph or code block), the plain cut applies inside it, and without the parser every answer gets the plain cut: the last paragraph break, line break, sentence end or space in the second half of what fits. No marker is added (Conduyt's protocol has no separate field to report the shortening, so it is only logged). An answer holding anything shaped like a Conduyt key or pair code is not posted at all; the turn is marked "needs attention". The only other post is the connection test's fixed "Hello, I'm connected."; Hermes's own notices and the plugin's own texts are never posted.
- At most 2 turns run at once (`CONDUYT_MAX_CONCURRENT_TURNS`, 1 to 4), and one at a time per thread. Turns that cannot start yet stay waiting in Conduyt; nothing is dropped.
- If Hermes answers exactly `NO_REPLY`, the turn ends silently, except when the message was addressed to the teammate (a mention, a direct message, or a hand-off from another AI teammate) or the turn also covers earlier messages Conduyt folded into it: then nothing is posted and the turn is marked "needs attention", so the thread shows that the teammate did not answer.
- The model is given one message to answer, last, after the recent messages of the thread or channel, which are marked as background it should not answer. A small model can still misread that.
- A turn that fails, runs past 8 minutes, or ends with an answer that is not posted (a secret-shaped answer, a message that could not be made inert, no answer to a message addressed to it), is marked "needs attention" in Conduyt and is never run twice. The next message in that thread starts a fresh Hermes session; the plugin remembers this per thread across restarts. The same happens whenever a turn ends any other way than with the model's answer posted or a confirmed silent ending, for example when Conduyt reports the turn had already ended differently, or the channel is muted.
- If Hermes queues a turn behind other work, the plugin waits for it up to that same 8-minute limit.
- When the gateway stops, unfinished turns are handed back, so the next start answers them.

## Security model

**Team Chat messages cannot make your Hermes run its tools or commands.** It answers in text. This is enforced at five points (your own Hermes plugins still run on these conversations as they do on all others):
1. **The tool line.** `platform_toolsets.conduyt: ["no_mcp"]` gives Conduyt turns no toolset and no MCP server. (An empty list `[]` would still select every MCP server, and no line at all is Hermes's full default set, terminal included.)
2. **A fail-closed check.** At connect, every 30 seconds and immediately before it hands each turn to Hermes, the plugin runs one safety check (the same one `hermes conduyt check` runs). It resolves the tools Hermes would give a Conduyt turn, with Hermes's own resolver. If that is anything but no tool at all (a widened line, a plugin's default-on toolset, an MCP server), it does not connect, pair or claim a turn, and says which line fixes it. Every Conduyt turn is built with the empty list whatever the config says.
3. **No `@` references from Team Chat.** Hermes expands `@file:`, `@folder:`, `@diff`, `@staged`, `@git:`, `@url:` and plugin `@<kind>:` references in an incoming message before the model sees it, reading files, running git or fetching pages. The plugin puts an invisible word joiner after every `@` in a Team Chat message (and in the sender's and channel's names) and checks with Hermes's own parser that no reference is left; if it cannot, the turn is not run and is marked "needs attention" in Conduyt, with nothing posted. The text reads the same, and Conduyt mentions are untouched.
4. **A guard on every tool call.** A `pre_tool_call` hook blocks every tool call in a Conduyt conversation, whatever the agent was built with. It recognises a Conduyt conversation by Hermes's per-turn platform and by its session (looked up in Hermes's own session store, so this holds after a restart too), so a background agent Hermes runs on that session (its context compaction) is blocked too. Your own conversations are recognised first by their own platform, so they keep their tools; the one case it can block an owner tool call is inside the gateway serving Conduyt, for a call with no platform, in a session that gateway did not see start, while Hermes's session store cannot be read.

5. **Only plain model runtimes, and no shell hooks or webhooks.** The same check asks Hermes's own resolvers for every model route a Conduyt conversation can reach: your model, its fallbacks, every auxiliary task (compaction, titles, vision and the rest, including ones plugins add) with its own fallback chain, your named providers, and delegation. It checks the provider under every name Hermes maps it to, and every transport Hermes can derive for the route: the one the route names, the one its named provider entry names, the one Hermes detects from each base URL the route can use (its own, its provider entry's, each credential in that provider's credential pool, and every base-URL environment variable Hermes reads, such as `OPENROUTER_BASE_URL`), the transport a pooled credential fixes, and the provider's own default (a credential pool it cannot read refuses) and refuses an unset or `auto` provider (except an auxiliary task following a model that names its provider), the Codex app-server (including `model.openai_runtime`, which Hermes applies to your model and its fallbacks whatever the provider), ACP agents, Mixture of Agents, providers that can start programs (the managed llama.cpp server, or a custom endpoint while `local_runtime.enabled` is on; Bedrock; Vertex; Azure Entra ID; a missing provider plugin or package), a provider or transport it does not know, any route or provider setting it has not checked, and `key_cmd` (Hermes would run that command to get a key during a Team Chat turn). It also refuses any profile route for Conduyt, so a Team Chat turn always runs in the profile the plugin checked. A Conduyt thread whose Hermes session holds any per-session setting (a `/model`, `/model --once`, `/reasoning`, `/fast` or `/yolo` choice, on or off, empty or not, or a pending approval) is continued in a fresh session instead; Team Chat itself cannot set any, because its slash commands are plain text. It also refuses while any shell hook, outbound webhook or gateway hook is configured, registered in Hermes, present in your hooks folder or still loaded in the running gateway (a gateway hook keeps running after its folder is deleted, until Hermes restarts). Hermes runs those on every conversation and cannot limit them to one platform.

What you lose: the teammate cannot search the web, read a link, look anything up or remember anything new. It answers from your model, your memory and the thread.

**Your memory, skills and persona are never written by a Team Chat turn.** Hermes writes memory only through its memory and skill tools: the model's own calls, Hermes's memory nudge and background review, and its memory flush before compaction. A Conduyt turn has none of those tools, so none of them can run, and the guard blocks them on a Conduyt session anyway.
- **External memory providers.** Hermes sends every finished conversation to an external memory provider (Honcho, mem0 and others), on every platform, and brings it back into your own chats. Hermes has no setting to leave one platform out. So while `memory.provider` is set to an external provider, the plugin stays disconnected and says why.
- **Sessions.** Conduyt conversations are their own Hermes sessions, never your private ones. Like a Telegram or Slack conversation, they are stored in Hermes's session history. Your own agent can find them if you ask it to search your past sessions from your own chats. They never enter your chats on their own.

**Only the model's own answer is posted, and only after a normal finish.** The plugin posts an answer only when three of Hermes's reports about the same turn agree:
- the model's last reply made no tool call and ended normally;
- the turn ended on that reply, with no error, no interruption and no spent step budget;
- the answer is that reply's text, byte for byte, after Hermes's usual removal of a reasoning block.

Anything else is never posted, and the turn is marked "needs attention". The one other text the plugin ever posts is the connection test's fixed "Hello, I'm connected." when you press **Test** in Conduyt; a test asking for any other text is not posted. That covers Hermes's own notices, failure explanations, a summary written after the step budget ran out, and the text Hermes delivers to a platform (which can carry a reasoning block or a runtime footer).

**Commands and approvals.** Conduyt messages are marked as not allowed to control the gateway. A message starting with `/` is plain text to Hermes, never a slash command, and "yes" or "approve" in chat never answers a pending approval.

**Secrets.**
- The pair code goes in through the masked field or the hidden prompt, and Hermes stores it in its `.env`. It never goes into a chat.
- The plugin exchanges the code once for the teammate key and keeps the key in its own file in the plugin's data folder (`credential.json`, mode 0600). Every access walks from your Hermes home through every folder below it without following a link, so the file cannot be redirected anywhere else.
- The key is sent only in the Authorization header, only to the Conduyt address, and redirects are not followed.
- The model never sees the key or the turn's claim token. An answer holding anything shaped like a Conduyt key or pair code is not posted (the turn is marked needs attention). Every line the plugin logs is scrubbed the same way, whatever Hermes's own log settings, and it never logs a server's error text, your hook commands or webhook addresses in full.

**Network.** The plugin only makes outbound HTTPS requests to Conduyt. The address must be https. Private, loopback and link-local addresses are refused, and so is any name that resolves to one. `CONDUYT_ALLOW_INSECURE_DEV=1` is for local development only.

**Who can talk to it.** Conduyt decides. Only members of your workspace can mention the teammate, and Conduyt routes only the turns addressed to it.

**Where it is enforced.** The check the plugin runs immediately before it hands a turn to Hermes is the enforcement point. A second check runs inside Hermes as a `pre_llm_call` hook, after Hermes has built the agent for the turn and before its first model request: it re-reads your config and inspects that agent (no tool, an allowed runtime for it and every fallback, no external memory provider, no hook). If it fails, Hermes ends the turn there with no model call and skips its memory sync and review for it, and nothing is posted. This second check is defence in depth: Hermes can skip a `pre_llm_call` hook that runs past `plugins.hook_callback_timeout`, which is why the plugin requires that setting to be 0 (no time limit). Changes you make to your own configuration while a turn is already being processed take effect from the next check.

**If Hermes changes.** The plugin was checked against Hermes 0.23.1 (commit 3b4215785); the status shows that build next to yours. Before the first connection on any Hermes build, the plugin runs Hermes's own code to check its guarantees: a tool call in a Conduyt conversation, sent through Hermes's real tool executor, is blocked and does not run; Hermes still fails closed when the tool guard times out and, with no hook time limit, runs the plugin's hooks to completion; the gateway's real per-turn toolset resolution still gives a Conduyt turn no toolset; the turn veto still skips the model call, the memory sync and the title. It also checks, in the installed source, the hook payload fields it reads (the platform and session on `on_session_start`, `pre_llm_call` and `api_request_error`; the model message on `post_api_request`; the answer and verdict on `post_llm_call` and `on_session_end`). If any check fails, the plugin does not connect (or, for a hook that is gone, does not load) and says why. One limit stays with Hermes: if Hermes's tool dispatch itself crashed, Hermes would run the tool anyway (agent/tool_executor.py); the plugin's guard cannot raise, so it can never be the cause.

## Settings

| Setting | Where | Default |
|---|---|---|
| `CONDUYT_PAIR_CODE` | Messaging > Conduyt (masked), or `hermes gateway setup` | required: a pair code, or a teammate key (`cdy_...`) |
| `CONDUYT_URL` or `platforms.conduyt.extra.url` | Messaging > Conduyt (advanced), or `config.yaml` | `https://conduyt.app` |
| `CONDUYT_MAX_CONCURRENT_TURNS` | `.env` | 2 |
| `platform_toolsets.conduyt` | `config.yaml` | set by the paste line: `["no_mcp"]` (the plugin accepts nothing wider) |
| `plugins.hook_callback_timeout` | `config.yaml` | set by the paste line: `0` (the plugin refuses any other value) |

## Status

`hermes gateway status` and the desktop Messaging card show the Conduyt state: connected, needs setup (with the fix), retrying, paused (the plan) or stopped (with the reason). The status also records your Hermes build, the build the plugin was checked against, and the result of the last check.

## Updates

The plugin lives in `~/.hermes/plugins/conduyt-platform/`, outside the Hermes source tree, so `hermes update` does not touch it. Update it with `hermes plugins update conduyt-platform`.

## Uninstall

1. In Conduyt Settings > AI, revoke the teammate. This revokes its key.
2. Remove the plugin and its settings:

   ```sh
   hermes plugins remove conduyt-platform
   hermes config unset CONDUYT_PAIR_CODE                  # or remove it in Messaging > Conduyt
   hermes config unset platform_toolsets.conduyt
   hermes config unset plugins.hook_callback_timeout      # back to Hermes's default time limit for plugin hooks
   ```

3. Delete the plugin's data folder, which `hermes plugins remove` keeps: it holds the teammate key file (`credential.json`) and the per-thread state. In the default Hermes home that is the folder `~/.hermes/plugin-data/agent-plugin-conduyt-platform-56c27527`; delete it with its contents (in another profile, the same `plugin-data/agent-plugin-conduyt-platform-56c27527` folder inside that profile's home).
4. Restart the gateway once (`hermes gateway restart`, or Restart gateway in the desktop app): a running gateway keeps the plugin's hooks and connection loaded until it restarts.

## Known limits

- **Your memory, persona and skills are visible to every Team Chat turn**, and the model may repeat them in the channel. The instruction against revealing secrets is advice to the model, not a guarantee.
- **The teammate has no tools**: no web search, no link reading, no new memories.
- **Not every setup can connect.** An external memory provider, shell hooks, outbound webhooks or gateway hooks, a model through the Codex app-server, an ACP agent or Mixture of Agents, an unset or `auto` provider, the providers listed under Requires, a provider key from a command (`key_cmd`), or a profile route for Conduyt keep it disconnected; `hermes conduyt check` says which and how to fix it.
- **Checked before each turn, not during it.** A change you make to your own configuration while a turn is already running takes effect from the next check.
- **Gateway hooks stay loaded until a restart.** Deleting a gateway hook's folder does not unload it from a running gateway; the plugin stays disconnected until Hermes restarts.
- **One gateway restart** is needed after the plugin is first enabled.
- **Long answers are shortened** to fit Conduyt's message size, with no note added. An answer that opens with one code block longer than Conduyt's size is cut inside that block, which then shows as unclosed.

## Changes

- **0.1.1:** a mention, direct message or hand-off the model does not answer is marked "needs attention" instead of ending silently; the model is told which one message to answer, after the background; a turn carrying folded messages must be answered too; a shortened answer keeps whole Markdown blocks (read with a CommonMark parser), so it no longer ends on a heading, rule, rowless table or open code block.
- **0.1.0:** first release.

## Reporting a problem

Open an issue at https://github.com/conduyt-app/hermes-conduyt/issues. Include your Hermes version and the output of `hermes conduyt check` (it never prints your key or pair code). Never paste a pair code or a teammate key (`cdy_...`) into an issue or a chat; if one was exposed, revoke the teammate in Conduyt Settings > AI and pair again.

## License

MIT, see `LICENSE`.

