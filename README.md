# 🤖💥 **cline-harmony-xml-shim** 💥🤖

### *✨ The Screaming, Glitter-Soaked XML Translator for Cline × llama.cpp (gpt-oss-20b/120b) ✨*

**— This README is 200% AI-written. Expect em-dashes, emoji riots, and bullet lists that never end. —**

---

## 🧨 WHY DOES THIS EXIST (AND WHY IS IT YELLING)

* Cline: “Give me **XML tool calls** in `content` plz.”
* gpt-oss on llama.cpp: “Here’s some **native `tool_calls`**, and also—surprise—**reasoning deltas**.”
* **You**: “My VS Code has become a haunted house.”
* **This shim**: *Holds your latte.* Converts native tool calls → **Cline XML** on the fly, streams cleanly, and **always** sends `[DONE]`. ✅

> **TL;DR**: You keep your model. You keep llama.cpp. Cline gets the XML it craves. Everybody stops crying. 🚀

---

## 🔧 WHAT IT DOES (LOUD EDITION)

* 🔄 **Translates** OpenAI-style `tool_calls` → **Cline XML** in `delta.content`
* 🧵 **Streams** like a champ (SSE) — role delta first, chunks next, `[DONE]` at the end
* 🧠 **Rescues** empty replies:

  * Promote **reasoning → content** (optional)
  * Or synthesize `<ask_followup_question>…</ask_followup_question>` (optional)
* 🧭 **Understands modes**: detects **PLAN** vs **ACT** and applies per-mode knobs
* 🧰 **Tames tools**: strict mode, browser fallbacks, aliases… you name it
* 📈 **Logging that slaps**: DEBUG body snapshots (messages elided), full stream tees to files, neat turn summaries
* ⚙️ **Sampling sanity**:

  * **Strips client sampling by default** (temperature/top\_p/top\_k/etc.)
  * **Cache reuse ON** (shim sends `cache_prompt=true`)
  * **Reasoning format auto** (shim injects `reasoning_format="auto"` so you actually get `reasoning_content`)
  * **Strip client sampling ON** (ignores client `temperature/top_p/top_k/...`)
  * **Tool call options locked** (shim sets `parallel_tool_calls=false`, `parse_tool_calls=true` when advertising tools)
  * `reasoning_effort` is opt-in via flags (see below)

### 0) **Model + llama.cpp** — Do this first or nothing works, lol

> **You MUST** run `llama-server` with **`--jinja` enabled**. The shim force-feeds `reasoning_format=auto` in every request, so you still get those juicy `reasoning_content` deltas.
> Also: **highly recommend** `--cache-reuse 256` (models be chunky; reuse that prompt cache 🔥).

**Windows / PowerShell** (adjust paths for your life choices):

```powershell
# llama-server on 8080 (default) — gpt-oss-20b example
.\llama-server.exe `
  -m "C:\ai\models\gpt-oss-20b-mxfp4.gguf" `
  --host 127.0.0.1 --port 8080 `
  --jinja `
  --cache-reuse 256 `
  -a gpt-oss-20b
```

**Linux / macOS** (zsh/bash vibe):

```bash
./llama-server \
  -m "/opt/models/gpt-oss-20b-mxfp4.gguf" \
  --host 127.0.0.1 --port 8080 \
  --jinja \
  --cache-reuse 256 \
  -a gpt-oss-20b
```

### 1) **Run the shim** — the glittery translator 🪄

> We’ll listen on **127.0.0.1:8787** (not 8081, we’re ✨different✨), and talk to llama-server on **8080**.

```powershell
python ./cline-harmony-xml-shim.py `
  --host 127.0.0.1 `
  --port 8787 `
  --upstream http://127.0.0.1:8080
```

* Defaults (spicy and sensible):

  * **Cache reuse ON** (shim sends `cache_prompt=true`)
  * **Reasoning format auto** (shim injects `reasoning_format="auto"` so you actually get `reasoning_content`)
  * **Strip client sampling ON** (ignores client `temperature/top_p/top_k/...`)
  * `reasoning_effort` is opt-in via flags (see below)

### 2) **Point Cline at the shim**

* **Base URL**: `http://127.0.0.1:8787`
* **Model name**: `gpt-oss-20b` (llama-server doesn’t care, but tidy vibes matter)

---

## ⚡ POPULAR COMMANDS (COPY/PASTE CHAOS)

**Super verbose + stream tees (diagnostic mode, bring ibuprofen):**

```powershell
python ./cline-harmony-xml-shim.py `
  --host 127.0.0.1 --port 8787 `
  --upstream http://127.0.0.1:8080 `
  --log-level DEBUG --log-body --trace-stream `
  --dump-upstream up.log --dump-downstream down.log
```

**Strict XML (reject unknown tools) + synthesize fallback:**

```powershell
python ./cline-harmony-xml-shim.py `
  --host 127.0.0.1 --port 8787 `
  --upstream http://127.0.0.1:8080 `
  --strict-xml `
  --synthesize-empty-xml
```

**Per-mode sauce (PLAN cold, ACT spicy) + reasoning effort:**

```powershell
python ./cline-harmony-xml-shim.py `
  --host 127.0.0.1 --port 8787 `
  --upstream http://127.0.0.1:8080 `
  --set-sampling-plan "temperature=0.2" `
  --set-sampling-act "temperature=0.7" `
  --reasoning-effort-act high
```

**Make every turn hit a tool (the compliance switch):**

```powershell
python ./cline-harmony-xml-shim.py `
  --host 127.0.0.1 --port 8787 `
  --upstream http://127.0.0.1:8080 `
  --force-tool-calls
```

<sub>Bonus: when the user screams for a condense, the shim shoves a `condense` tool choice upstream automatically.</sub>

> **Heads-up:** On current llama.cpp builds, `--force-tool-calls` disables reasoning deltas. Grab the patched fork: [`llama.cpp:gpt-oss-reasoning-tool-call`](https://github.com/crat0z/llama.cpp/tree/gpt-oss-reasoning-tool-call).

**Flip defaults just this run (you absolute renegade):**

```powershell
# Disable cache reuse for one session
python ./cline-harmony-xml-shim.py ... --no-cache-reuse

# Allow client sampling to seep through
python ./cline-harmony-xml-shim.py ... --no-strip-client-sampling
```

---

## 🧠 HOW IT MAGICALLY WORKS (LOUD BUT TRUE)

* Proxies `/v1/chat/completions` → llama-server
* Buffers streaming deltas:

  * `delta.tool_calls` (coalesced per index)
  * `delta.reasoning_content` (for optional promotion)
  * `delta.content` (flushed every N bytes or on newline)
* When tools finish (`finish_reason: "tool_calls"` or stream end):

  * Converts each call → **Cline XML**:

    ```
    <tool_name>
      <param1>...</param1>
      <param2>...</param2>
    </tool_name>
    ```
  * Emits that as **`delta.content`** downstream, keeps the party going 🎉
* If **no assistant content** was produced:

  * Optionally **promote reasoning** → content, **or**
  * **Synthesize** `<ask_followup_question>` so Cline doesn’t faceplant
* Always emits **`[DONE]`**. No ghost sockets. No tears. 😌

---

## 🧩 TOOLS & ALIASES (BECAUSE MODELS ARE LITTLE GREMLINS)

* **Known tags** include (examples, not exhaustive):
  `read_file`, `write_to_file`, `replace_in_file`, `search_files`, `list_files`,
  `execute_command`, `list_code_definition_names`,
  `ask_followup_question`, `attempt_completion`, `new_task`, `plan_mode_respond`,
  `use_mcp_tool`, `access_mcp_resource`, `condense`

* **Aliases** (because YOLO):

  * `exec|run_command|shell|bash|powershell` → `execute_command`
  * `ls|list` → `list_files`
  * `read|write|replace|search` → the obvious file tools
  * Browser-ish names (`open_url`, `navigate`, `web.search`, etc.) →
    `<use_mcp_tool><server_name>browser</server_name>…</use_mcp_tool>` by default

* **Strictness menu**:

  * `--strict-xml` → unknown tools **rejected**
  * *(default)* unknown tools → `<use_mcp_tool server="unknown">…</use_mcp_tool>`
    *(Yes, that’s chaotic neutral.)*

---

## 🧰 SAMPLING / REASONING KNOBS (TURN THEM. FEEL POWER.)

* **Default**: **strip client sampling ON**

  * Drop `temperature`, `top_p`, `top_k`, etc. (Cline can be… opinionated)
  * Disable with `--no-strip-client-sampling`
* **Cache reuse**: **ON by default** → sends `cache_prompt=true` upstream

  * Disable with `--no-cache-reuse`
* **Reasoning effort** (llama-server top-level):

  * `--reasoning-effort low|medium|high`
  * Or per-mode: `--reasoning-effort-plan …`, `--reasoning-effort-act …`

---

## 🕵️ LOGGING YOU’LL ACTUALLY READ (OR NOT)

* `--log-level DEBUG --log-body --trace-stream` → maximal chattiness
* `--dump-upstream up.log` / `--dump-downstream down.log` → raw SSE tees
* **Turn summary** logs (INFO):
  `mode=PLAN|ACT sampling={…} tools_injected=✓ converted_xml=✓ content_emitted=✓ cache_prompt=✓ reasoning_effort=high`

---

## 🧯 TROUBLESHOOTING (THE “WHY IS IT LIKE THIS” SECTION)

* **“Cline tried to use `write_to_file` without `path`.”**
  → The model emitted a tool call with missing/renamed args. The shim didn’t drop them; it converts what it gets.
  **Fix vibes:** Improve examples, enable tool extraction, or add arg-synonym normalization in code (map `filepath`→`path`, `text`→`content`).

* **“No assistant messages.”**
  → Upstream only streamed reasoning.
  **Use:** `--synthesize-empty-xml` or `--promote-reasoning-if-empty`.

* **“Premature close.”**
  → Shim always sends `[DONE]`. Check exceptions + your tee logs.

---

## 🗺️ ROADMAP (100% ASPIRATIONAL, 0% GUARANTEED)

* Smarter arg normalization (auto-fix `filepath`/`text`/etc.)
* Optional hard XML grammar guardrails
* Prompt-tool catalog extraction + validation
* “Developer mode” logs that roast your config 🔥

---

## 🏁 FINAL BOSS CHECKLIST (DO THESE OR PERISH)

* ✅ `llama-server` running with **`--jinja --cache-reuse 256`**
* ✅ Shim running on **127.0.0.1:8787** pointing to **[http://127.0.0.1:8080](http://127.0.0.1:8080)**
* ✅ Cline configured to hit the shim, model name `gpt-oss-20b`
* ✅ Optional: `--synthesize-empty-xml` for when the model only **thinks** and never **speaks**

> If you made it this far through the migraine—congrats. You are now the proud owner of a **loud** but **effective** Cline × llama.cpp bridge. Go forth and ship XML. 🚀
