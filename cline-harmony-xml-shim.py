#!/usr/bin/env python3
"""
Cline Harmony XML shim for llama.cpp (gpt-oss-20b/120b)
- Translates native OpenAI `tool_calls` -> Cline's XML in `content`
- Streaming + non-streaming
- Rich logging with levels
- Browser-ish calls fall back to <use_mcp_tool> until you give exact tags

CLI has precedence over env vars. Works fine on Windows/PowerShell.
"""

import os, os.path, json, time, logging, argparse
from typing import Any, Dict, Tuple
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# ---------- defaults (env) ----------
ENV = lambda k, d=None: os.getenv(k, d)

UPSTREAM               = ENV("UPSTREAM", "http://127.0.0.1:8081").rstrip("/")
PORT                   = int(ENV("PORT", "10000"))
HOST                   = ENV("HOST", "0.0.0.0")  # where the shim listens
MODEL_FALLBACK         = ENV("MODEL", "gpt-oss")
FORCE_TOOL_CHOICE_NONE = ENV("FORCE_TOOL_CHOICE_NONE", "0") == "1"

LOG_LEVEL     = ENV("LOG_LEVEL", "INFO").upper()
LOG_BODY      = ENV("LOG_BODY", "0") == "1"
TRACE_STREAM  = ENV("TRACE_STREAM", "0") == "1"
LOG_REASONING = ENV("LOG_REASONING", "0") == "1"

STRICT_XML            = ENV("STRICT_XML", "0") == "1"
ALLOW_UNKNOWN_AS_MCP  = ENV("ALLOW_UNKNOWN_AS_MCP", "1") == "1"
BROWSER_SERVER_NAME   = ENV("BROWSER_SERVER_NAME", "browser")
CUSTOM_ALIASES_JSON   = ENV("CUSTOM_ALIASES_JSON", "")

# ---------- argparse (CLI overrides env) ----------
def parse_args():
    p = argparse.ArgumentParser(description="Cline Harmony XML shim for llama.cpp")
    p.add_argument("--upstream", default=UPSTREAM,
                   help="Upstream OpenAI-compatible base (e.g. http://127.0.0.1:8081)")
    p.add_argument("--port", type=int, default=PORT, help="Port to listen on (default 10000)")
    p.add_argument("--host", default=HOST, help="Host to bind (default 0.0.0.0)")
    p.add_argument("--model", default=MODEL_FALLBACK, help="Model name to report if none given")
    # logging
    p.add_argument("--log-level", choices=["DEBUG","INFO","WARNING","ERROR"], default=LOG_LEVEL)
    p.add_argument("--log-body", action="store_true", default=LOG_BODY, help="Log full JSON bodies (truncated)")
    p.add_argument("--trace-stream", action="store_true", default=TRACE_STREAM, help="Echo upstream SSE lines")
    p.add_argument("--log-reasoning", action="store_true", default=LOG_REASONING, help="Print reasoning deltas")
    # behavior
    p.add_argument("--force-tool-choice-none", action="store_true", default=FORCE_TOOL_CHOICE_NONE,
                   help="Send tool_choice='none' upstream")
    p.add_argument("--strict-xml", action="store_true", default=STRICT_XML,
                   help="Refuse unknown tools (no conversion)")
    p.add_argument("--allow-unknown-as-mcp", action="store_true", default=ALLOW_UNKNOWN_AS_MCP,
                   help="Unknown tools -> <use_mcp_tool server=unknown>")
    p.add_argument("--no-allow-unknown-as-mcp", dest="allow_unknown_as_mcp", action="store_false")
    p.add_argument("--browser-server-name", default=BROWSER_SERVER_NAME,
                   help="MCP server name for browser fallbacks (default 'browser')")
    p.add_argument("--custom-aliases-json", default=CUSTOM_ALIASES_JSON,
                   help="Path to JSON dict of extra alias mappings")
    p.add_argument("--synthesize-empty-xml", action="store_true", default=False,
               help="If no content or tools were emitted, send <ask_followup_question>…</ask_followup_question> before [DONE].")
    p.add_argument("--promote-reasoning-if-empty", action="store_true", default=False,
               help="If only reasoning_content was received, move it to content when finishing.")
    p.add_argument("--dump-upstream", default="", help="Append raw upstream SSE lines to this file")
    p.add_argument("--dump-downstream", default="", help="Append raw SSE lines sent to client")

    return p.parse_args()

# ---------- logging ----------
def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

# ---------- tool mapping ----------
# Canonical tags present in your (plan+act) prompts
KNOWN = {
    # Files & CLI
    "read_file","write_to_file","replace_in_file","search_files","list_files",
    "execute_command","list_code_definition_names",
    # Interaction / planning
    "ask_followup_question","attempt_completion","new_task","plan_mode_respond","load_mcp_documentation",
    # MCP
    "use_mcp_tool","access_mcp_resource",
}

ALIASES = {
    # CLI
    "exec":"execute_command","run_command":"execute_command","command":"execute_command",
    "shell":"execute_command","bash":"execute_command","powershell":"execute_command",
    # Files
    "ls":"list_files","list":"list_files","read":"read_file","write":"write_to_file",
    "search":"search_files","replace":"replace_in_file",
    # Planning-ish
    "complete":"attempt_completion","ask":"ask_followup_question","question":"ask_followup_question",
}

def load_custom_aliases(path: str, logger: logging.Logger):
    global ALIASES
    if not path:
        return
    if not os.path.exists(path):
        logger.error("CUSTOM_ALIASES_JSON not found: %s", path)
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            extra = json.load(f)
        if isinstance(extra, dict):
            ALIASES.update({str(k): str(v) for k,v in extra.items()})
            logger.info("Loaded %d custom aliases from %s", len(extra), path)
    except Exception as e:
        logger.error("Failed to load custom aliases: %s", e)

def looks_browsery(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in [
        "browser","navigate","open_url","visit","go_to","click","type","scroll",
        "screenshot","close_browser","web.","web_","page_","tab_"
    ])

def normalize_name(name: str) -> Tuple[str, bool]:
    if not name:
        return "", False
    if name in KNOWN:
        return name, True
    if name in ALIASES:
        can = ALIASES[name]
        return can, (can in KNOWN)
    return name, (name in KNOWN)

def xml_escape(s: Any) -> str:
    s = "" if s is None else str(s)
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def args_to_xml(d: Dict[str, Any]) -> str:
    out=[]
    for k,v in (d or {}).items():
        if isinstance(v,(str,int,float,bool)) or v is None:
            out.append(f"<{k}>{xml_escape(v)}</{k}>")
        else:
            out.append(f"<{k}>{xml_escape(json.dumps(v, ensure_ascii=False))}</{k}>")
    return "".join(out)

def tool_to_xml_direct(tag: str, args_json: str, logger: logging.Logger) -> str:
    try:
        obj = json.loads(args_json) if args_json else {}
    except Exception:
        logger.warning("Arguments JSON parse failed for tool '%s': %s", tag, args_json)
        obj = {}
    return f"<{tag}>{args_to_xml(obj)}</{tag}>"

def tool_to_xml_mcp(server_name: str, tool_name: str, args_json: str) -> str:
    return (
        f"<use_mcp_tool>"
        f"<server_name>{xml_escape(server_name)}</server_name>"
        f"<tool_name>{xml_escape(tool_name)}</tool_name>"
        f"<arguments>{xml_escape(args_json or '{}')}</arguments>"
        f"</use_mcp_tool>"
    )

# ---------- app factory so we can inject args ----------
def create_app(cfg):
    app = FastAPI()
    log = logging.getLogger("clime-harmony-xml-shim")

    async def upstream_post(payload):
        upstream_body = dict(payload)
        if cfg.force_tool_choice_none:
            upstream_body["tool_choice"] = "none"
        if cfg.log_body and log.isEnabledFor(logging.DEBUG):
            log.debug(">>> Upstream request: %s", json.dumps(upstream_body, ensure_ascii=False)[:2000])
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{cfg.upstream}/v1/chat/completions", json=upstream_body)
            if cfg.trace_stream and upstream_body.get("stream"):
                log.debug("STREAM open -> %s", r.status_code)
            return r

    def convert_tool_call(native_name: str, args_json: str) -> str:
        name = native_name or ""
        canonical, known = normalize_name(name)
        if known and canonical in KNOWN:
            return tool_to_xml_direct(canonical, args_json, log)
        if looks_browsery(name):
            xml = tool_to_xml_mcp(cfg.browser_server_name, name, args_json)
            log.info("Browser fallback mapped '%s' -> <use_mcp_tool server=%s>", name, cfg.browser_server_name)
            return xml
        if cfg.strict_xml:
            log.error("Unknown tool '%s' (STRICT_XML=1) -- not converted.", name)
            return ""
        if cfg.allow_unknown_as_mcp:
            xml = tool_to_xml_mcp("unknown", name, args_json)
            log.warning("Unknown tool '%s' mapped to MCP fallback server=unknown", name)
            return xml
        log.warning("Unknown tool '%s' emitted as literal tag (may fail).", name)
        return tool_to_xml_direct(name, args_json, log)

    async def handle_nonstream(client_body: dict):
        if cfg.log_body and log.isEnabledFor(logging.DEBUG):
            log.debug("Client request (non-stream): %s", json.dumps(client_body, ensure_ascii=False)[:2000])
        r = await upstream_post(client_body)
        try:
            j = r.json()
        except Exception as e:
            log.error("Upstream non-stream JSON decode error: %s", e)
            return JSONResponse({"error":"bad_upstream_json"}, status_code=502)

        if not j.get("choices"):
            return JSONResponse(j, status_code=r.status_code)

        ch = j["choices"][0]
        msg = ch.get("message", {})
        tcs = msg.get("tool_calls") or []
        if tcs:
            parts=[]
            for tc in tcs:
                fn = tc.get("function") or {}
                name = fn.get("name","")
                args = fn.get("arguments","")
                xml = convert_tool_call(name, args)
                if not xml:
                    log.error("Failed to convert tool '%s'; leaving text-only.", name)
                parts.append(xml)
            content = (msg.get("content") or "") + "".join(parts)
            ch["message"] = {"role":"assistant","content": content}
            ch["message"].pop("tool_calls", None)
            return JSONResponse(j, status_code=r.status_code)

        return JSONResponse(j, status_code=r.status_code)

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        try:
            body = await req.json()
        except Exception:
            log.error("Failed to parse client JSON")
            return JSONResponse({"error": "bad_request"}, status_code=400)

        if cfg.log_body and log.isEnabledFor(logging.DEBUG):
            log.debug("Client request: %s", json.dumps(body, ensure_ascii=False)[:2000])

        if not body.get("stream"):
            return await handle_nonstream(body)

        up = await upstream_post(body)
        id_ = f"chatcmpl-xmlshim-{int(time.time())}"
        model = body.get("model") or cfg.model

        tool_buf: Dict[int, Dict[str, str]] = {}
        text_buf = ""
        reasoning_buf = []     # collect reasoning_content deltas (optional promotion)
        sent_any_content = False  # did we ever yield a chunk with delta.content?
        emitted_xml_tools = False # did we convert and emit any tool XML?

        async def gen():
            nonlocal text_buf, tool_buf, sent_any_content, emitted_xml_tools, reasoning_buf

            up_fp = open(cfg.dump_upstream, "a", encoding="utf-8") if getattr(cfg, "dump_upstream", "") else None
            down_fp = open(cfg.dump_downstream, "a", encoding="utf-8") if getattr(cfg, "dump_downstream", "") else None
            prefix = f"[{id_}] "  # session marker for log readability
            
            def tee_up(raw_line: str):
                if up_fp:
                    up_fp.write(prefix + raw_line + ("" if raw_line.endswith("\n") else "\n"))
                    up_fp.flush()
            
            def send_line(s: str):
                # s already includes trailing "\n\n"
                if down_fp:
                    down_fp.write(prefix + s)
                    down_fp.flush()
                return s

            def flush_text(mark_stop: bool = False):
                nonlocal text_buf, sent_any_content
                if not text_buf:
                    return None
                chunk = {
                    "id": id_, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{ "index": 0, "delta": {"content": text_buf}, "finish_reason": ("stop" if mark_stop else None) }],
                }
                text_buf = ""
                sent_any_content = True
                return chunk

            def flush_tools():
                nonlocal tool_buf, emitted_xml_tools, sent_any_content
                if not tool_buf:
                    return None
                xml = "".join(
                    convert_tool_call(v.get("name", ""), v.get("args", ""))
                    for _, v in sorted(tool_buf.items())
                )
                tool_buf.clear()
                emitted_xml_tools = True
                sent_any_content = True
                return {
                    "id": id_, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{ "index": 0, "delta": {"content": xml}, "finish_reason": None }],
                }

            def synthesize_if_empty():
                """Called right before DONE if nothing usable was produced.
                   Strategy (in order):
                     1) promote reasoning (if flag and we have any)
                     2) emit a tiny XML ask_followup (if flag)
                     3) emit a single space (last resort to satisfy 'assistant message exists')
                """
                nonlocal sent_any_content
                if sent_any_content:
                    return None

                # 1) promote reasoning to content if requested
                if cfg.promote_reasoning_if_empty and reasoning_buf:
                    promoted = "".join(reasoning_buf).strip()
                    if promoted:
                        log.warning("Promoting reasoning_content to content (empty content stream).")
                        return {
                            "id": id_, "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": model,
                            "choices": [{ "index": 0, "delta": {"content": promoted}, "finish_reason": None }],
                        }

                # 2) synthesize a minimal valid XML tool call (ask follow-up)
                if cfg.synthesize_empty_xml:
                    q = ("The upstream model returned no content this turn. "
                         "Would you like me to try again, adjust model settings, or proceed with a follow-up question?")
                    fallback = (
                        "<ask_followup_question>"
                        f"<question>{xml_escape(q)}</question>"
                        "</ask_followup_question>"
                    )
                    log.warning("Synthesizing minimal XML (ask_followup_question) because upstream produced no content/tool calls.")
                    return {
                        "id": id_, "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": model,
                        "choices": [{ "index": 0, "delta": {"content": fallback}, "finish_reason": None }],
                    }

                # 3) last resort: emit a single space so Cline sees an assistant message
                log.warning("Emitting a single-space content chunk to satisfy client (no content produced).")
                return {
                    "id": id_, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{ "index": 0, "delta": {"content": " "}, "finish_reason": None }],
                }

            # Start with a role delta
            head = {
                "id": id_, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model,
                "choices": [{ "index": 0, "delta": {"role": "assistant"}, "finish_reason": None }],
            }
            yield send_line(f"data: {json.dumps(head)}\n\n")

            try:
                async for line in up.aiter_lines():
                    if not line: 
                        continue
                    tee_up(line)
                    data = line[6:] if line.startswith("data: ") else line
                    if cfg.trace_stream:
                        log.debug("UPSTREAM: %s", data[:200])

                    if data == "[DONE]":
                        # finish: flush pending content/tools, then ensure not-empty
                        if (c := flush_text()):
                            yield send_line(f"data: {json.dumps(c)}\n\n")
                        if (c := flush_tools()):
                            yield send_line(f"data: {json.dumps(c)}\n\n")
                        if not sent_any_content:
                            if (c := synthesize_if_empty()):
                                yield send_line(f"data: {json.dumps(c)}\n\n")
                        yield send_line("data: [DONE]\n\n")
                        return

                    # parse upstream JSON line
                    try:
                        j = json.loads(data)
                    except Exception:
                        continue

                    ch0 = (j.get("choices") or [{}])[0]
                    d = ch0.get("delta") or {}

                    # capture reasoning (optional promotion at end)
                    if "reasoning_content" in d and d["reasoning_content"]:
                        if cfg.log_reasoning:
                            log.debug("[reasoning Δ] %s", d["reasoning_content"])
                        reasoning_buf.append(d["reasoning_content"])

                    # native tool_calls (buffer by index)
                    if "tool_calls" in d:
                        for tc in d["tool_calls"]:
                            idx = int(tc.get("index", 0))
                            buf = tool_buf.setdefault(idx, {"name": "", "args": ""})
                            fn = tc.get("function") or {}
                            if "name" in fn:
                                buf["name"] += fn["name"]
                            if "arguments" in fn:
                                buf["args"] += fn["arguments"]
                            log.info("tool_calls Δ idx=%s name+=%r args_len=%d",
                                     idx, fn.get("name", ""), len(buf["args"]))
                        continue

                    # upstream says tool_calls are done → emit converted XML
                    if ch0.get("finish_reason") in ("tool_calls", "tool_call"):
                        if (c := flush_text()):
                            yield send_line(f"data: {json.dumps(c)}\n\n")
                        if (c := flush_tools()):
                            yield send_line(f"data: {json.dumps(c)}\n\n")
                        continue

                    # normal text deltas
                    if isinstance(d.get("content"), str) and d["content"]:
                        text_buf += d["content"]
                        if len(text_buf) >= 256 or "\n" in text_buf:
                            if (c := flush_text()):
                                yield send_line(f"data: {json.dumps(c)}\n\n")

                    # graceful 'stop'
                    if ch0.get("finish_reason") == "stop":
                        # flush pending text as 'stop', ensure not-empty, then DONE
                        if (c := flush_text(mark_stop=True)):
                            yield send_line(f"data: {json.dumps(c)}\n\n")
                        elif not sent_any_content:
                            if (c := synthesize_if_empty()):
                                # mark the synthetic one as 'stop'
                                c["choices"][0]["finish_reason"] = "stop"
                                yield send_line(f"data: {json.dumps(c)}\n\n")
                            else:
                                # send a bare stop tail if we emitted nothing (Cline may still complain)
                                tail = {
                                    "id": id_, "object": "chat.completion.chunk",
                                    "created": int(time.time()), "model": model,
                                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                                }
                                yield send_line(f"data: {json.dumps(tail)}\n\n")
                        yield send_line("data: [DONE]\n\n")
                        return

            except Exception as e:
                log.error("Upstream stream error: %r", e)
                # best-effort flush + ensure at least one assistant message
                if (c := flush_text()):
                    yield send_line(f"data: {json.dumps(c)}\n\n")
                if (c := flush_tools()):
                    yield send_line(f"data: {json.dumps(c)}\n\n")
                if not sent_any_content:
                    if (c := synthesize_if_empty()):
                        yield send_line(f"data: {json.dumps(c)}\n\n")
                yield send_line("data: [DONE]\n\n")
                return
            
            finally:
                if up_fp: up_fp.close()
                if down_fp: down_fp.close()

        return StreamingResponse(gen(), media_type="text/event-stream")



    @app.get("/")
    def health():
        return {"ok": True, "upstream": cfg.upstream}

    return app

# ---------- main ----------
if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("clime-harmony-xml-shim")

    # Apply CLI overrides
    class Cfg:
        upstream = args.upstream.rstrip("/")
        port = args.port
        host = args.host
        model = args.model
        force_tool_choice_none = args.force_tool_choice_none
        log_level = args.log_level
        log_body = args.log_body
        trace_stream = args.trace_stream
        log_reasoning = args.log_reasoning
        strict_xml = args.strict_xml
        allow_unknown_as_mcp = args.allow_unknown_as_mcp
        browser_server_name = args.browser_server_name
        custom_aliases_json = args.custom_aliases_json
        synthesize_empty_xml = args.synthesize_empty_xml
        promote_reasoning_if_empty = args.promote_reasoning_if_empty
        dump_upstream = args.dump_upstream
        dump_downstream = args.dump_downstream

    load_custom_aliases(Cfg.custom_aliases_json, log)

    app = create_app(Cfg)

    log.info("Cline XML shim listening on %s:%s -> %s", Cfg.host, Cfg.port, Cfg.upstream)
    import uvicorn
    uvicorn.run(app, host=Cfg.host, port=Cfg.port)
