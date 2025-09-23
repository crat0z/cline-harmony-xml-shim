#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cline Harmony XML shim for llama.cpp (gpt-oss-20b/120b)

Goals
- Keep Cline's original system prompt intact (XML examples help).
- Parse tools from the system prompt -> attach OpenAI tools[] to upstream.
- Convert native OpenAI tool_calls -> Cline XML-in-content (1 tool/msg).
- Support streaming & non-streaming; guarantee at least one assistant message.
- Sampling: strip/override (global + per-mode), with sane defaults opt-in.
- Robust logging with levels, SSE tees, and end-of-turn summaries.
- Optional guardrail system nudge; optional XML grammar (experimental).

CLI has precedence over env vars. Works fine on Windows/PowerShell.

Author: you & me :)
"""

from __future__ import annotations

import os, json, time, logging, argparse, re
from typing import Any, Dict, List, Tuple, Optional
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# ---------- env helpers ----------
ENV = lambda k, d=None: os.getenv(k, d)

# ---------- defaults (env) ----------
UPSTREAM               = ENV("UPSTREAM", "http://127.0.0.1:8081").rstrip("/")
PORT                   = int(ENV("PORT", "10000"))
HOST                   = ENV("HOST", "0.0.0.0")
MODEL_FALLBACK         = ENV("MODEL", "gpt-oss")

# Logging
LOG_LEVEL     = ENV("LOG_LEVEL", "INFO").upper()
LOG_BODY      = ENV("LOG_BODY", "0") == "1"
TRACE_STREAM  = ENV("TRACE_STREAM", "0") == "1"
LOG_REASONING = ENV("LOG_REASONING", "0") == "1"

# Behavior toggles
EXTRACT_TOOLS          = ENV("EXTRACT_TOOLS", "1") == "1"
TOOL_EXAMPLES          = ENV("TOOL_EXAMPLES", "py")  # none|xml|py
FORCE_TOOL_CHOICE_NONE = ENV("FORCE_TOOL_CHOICE_NONE", "0") == "1"
TOOL_CHOICE            = ENV("TOOL_CHOICE", "auto")  # auto|none

STRICT_XML             = ENV("STRICT_XML", "1") == "1"      # default strict
ALLOW_UNKNOWN_AS_MCP   = ENV("ALLOW_UNKNOWN_AS_MCP", "0") == "1"
DEFAULT_MCP_SERVER     = ENV("DEFAULT_MCP_SERVER", "")      # e.g. "browser" or "" to disable
BROWSER_SERVER_NAME    = ENV("BROWSER_SERVER_NAME", "browser")
CUSTOM_ALIASES_JSON    = ENV("CUSTOM_ALIASES_JSON", "")
MULTI_TOOL_POLICY      = ENV("MULTI_TOOL_POLICY", "first")  # first|merge|error

SYNTH_EMPTY_XML        = ENV("SYNTHESIZE_EMPTY_XML", "1") == "1"  # default ON
PROMOTE_REASONING      = ENV("PROMOTE_REASONING_IF_EMPTY", "0") == "1"
FALLBACK_QUESTION      = ENV("FALLBACK_QUESTION", "The model returned no actionable content this turn. Would you like me to try again or ask a follow-up question?")

STRIP_CLIENT_SAMPLING  = ENV("STRIP_CLIENT_SAMPLING", "0") == "1"
SET_SAMPLING           = ENV("SET_SAMPLING", "")         # "temperature=0.3,top_p=0.9,top_k=40"
SET_SAMPLING_PLAN      = ENV("SET_SAMPLING_PLAN", "")    # per-mode overrides
SET_SAMPLING_ACT       = ENV("SET_SAMPLING_ACT", "")

GUARDRAIL_PROMPT       = ENV("GUARDRAIL_PROMPT", "0") == "1"

ENFORCE_XML            = ENV("ENFORCE_XML", "off")  # off|plan|act|both  (experimental)
FLUSH_BYTES            = int(ENV("FLUSH_BYTES", "256"))

# SSE tees
DUMP_UPSTREAM          = ENV("DUMP_UPSTREAM", "")
DUMP_DOWNSTREAM        = ENV("DUMP_DOWNSTREAM", "")

# ---------- argparse (CLI overrides env) ----------
def parse_args():
    p = argparse.ArgumentParser(description="Cline Harmony XML shim for llama.cpp")
    p.add_argument("--upstream", default=UPSTREAM, help="Upstream OpenAI-compatible base (e.g. http://127.0.0.1:8081)")
    p.add_argument("--port", type=int, default=PORT, help="Port to listen on (default 10000)")
    p.add_argument("--host", default=HOST, help="Host to bind (default 0.0.0.0)")
    p.add_argument("--model", default=MODEL_FALLBACK, help="Model name to report if none given")
    # logging
    p.add_argument("--log-level", choices=["DEBUG","INFO","WARNING","ERROR"], default=LOG_LEVEL)
    p.add_argument("--log-body", action="store_true", default=LOG_BODY, help="Log client/upstream JSON bodies (truncated)")
    p.add_argument("--trace-stream", action="store_true", default=TRACE_STREAM, help="Echo upstream SSE lines to console")
    p.add_argument("--log-reasoning", action="store_true", default=LOG_REASONING, help="Print reasoning deltas")
    p.add_argument("--dump-upstream", default=DUMP_UPSTREAM, help="Append raw upstream SSE lines to this file")
    p.add_argument("--dump-downstream", default=DUMP_DOWNSTREAM, help="Append raw SSE lines sent to client")
    # tools & conversions
    p.add_argument("--extract-tools", action="store_true", default=EXTRACT_TOOLS, help="Parse tools from system prompt and attach tools[]")
    p.add_argument("--tool-examples", choices=["none","xml","py"], default=TOOL_EXAMPLES, help="Append short example to tool description")
    p.add_argument("--tool-choice", choices=["auto","none"], default=TOOL_CHOICE, help="Upstream tool_choice override")
    p.add_argument("--force-tool-choice-none", action="store_true", default=FORCE_TOOL_CHOICE_NONE, help="Alias for --tool-choice none")
    p.add_argument("--strict-xml", action="store_true", default=STRICT_XML, help="Unknown tools error out -> synth follow-up")
    p.add_argument("--allow-unknown-as-mcp", action="store_true", default=ALLOW_UNKNOWN_AS_MCP, help="Unknown tools wrap as <use_mcp_tool>")
    p.add_argument("--default-mcp-server", default=DEFAULT_MCP_SERVER, help="Server name used when wrapping unknown tools (opt-in)")
    p.add_argument("--browser-server-name", default=BROWSER_SERVER_NAME, help="Heuristic mapping for browser-ish names (used if default MCP set)")
    p.add_argument("--custom-aliases-json", default=CUSTOM_ALIASES_JSON, help="Path to JSON dict of extra alias mappings")
    p.add_argument("--multi-tool-policy", choices=["first","merge","error"], default=MULTI_TOOL_POLICY, help="How to handle >1 native tool in a single reply")
    # fallbacks
    p.add_argument("--synthesize-empty-xml", action="store_true", default=SYNTH_EMPTY_XML, help="If no content/tools, emit <ask_followup_question>…</…>")
    p.add_argument("--promote-reasoning-if-empty", action="store_true", default=PROMOTE_REASONING, help="Promote reasoning_content to content if content empty")
    p.add_argument("--fallback-question", default=FALLBACK_QUESTION, help="Custom question text for synthesized <ask_followup_question>")
    # sampling
    p.add_argument("--strip-client-sampling", action="store_true", default=STRIP_CLIENT_SAMPLING, help="Drop client temperature/top_p/top_k/etc.")
    p.add_argument("--set-sampling", default=SET_SAMPLING, help='Global overrides, e.g. "temperature=0.3,top_p=0.9,top_k=40"')
    p.add_argument("--set-sampling-plan", default=SET_SAMPLING_PLAN, help='PLAN-mode overrides (take precedence)')
    p.add_argument("--set-sampling-act", default=SET_SAMPLING_ACT, help='ACT-mode overrides (take precedence)')
    # guardrails & grammar
    p.add_argument("--guardrail-prompt", action="store_true", default=GUARDRAIL_PROMPT, help="Insert a tiny system nudge after Cline system")
    p.add_argument("--enforce-xml", choices=["off","plan","act","both"], default=ENFORCE_XML, help="Attach a minimal XML grammar (experimental)")
    # streaming
    p.add_argument("--flush-bytes", type=int, default=FLUSH_BYTES, help="Flush text buffer when >= N bytes or newline")
    return p.parse_args()

# ---------- logging ----------
def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

# ---------- canonical tool tags (from Cline prompt) ----------
KNOWN = {
    "read_file","write_to_file","replace_in_file","search_files","list_files",
    "execute_command","list_code_definition_names",
    "ask_followup_question","attempt_completion","new_task","plan_mode_respond","load_mcp_documentation",
    "use_mcp_tool","access_mcp_resource",
}

ALIASES: Dict[str,str] = {
    # CLI
    "exec":"execute_command","run_command":"execute_command","command":"execute_command",
    "shell":"execute_command","bash":"execute_command","powershell":"execute_command",
    # Files
    "ls":"list_files","list":"list_files","read":"read_file","write":"write_to_file",
    "search":"search_files","replace":"replace_in_file",
    # Planning-ish
    "complete":"attempt_completion","ask":"ask_followup_question","question":"ask_followup_question",
}

SAMPLING_KEYS = {
    "temperature","top_p","top_k","min_p","tfs_z","typical_p",
    "presence_penalty","frequency_penalty","mirostat","mirostat_tau","mirostat_eta"
}

# ---------- small utils ----------
def load_custom_aliases(path: str, logger: logging.Logger):
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            extra = json.load(f)
        if isinstance(extra, dict):
            ALIASES.update({str(k): str(v) for k,v in extra.items()})
            logger.info("Loaded %d custom aliases from %s", len(extra), path)
    except FileNotFoundError:
        logger.error("CUSTOM_ALIASES_JSON not found: %s", path)
    except Exception as e:
        logger.error("Failed to load custom aliases: %s", e)

def xml_escape(s: Any) -> str:
    s = "" if s is None else str(s)
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def looks_browsery(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in [
        "browser","navigate","open_url","visit","go_to","click","type","scroll",
        "screenshot","close_browser","web.","web_","page_","tab_","search","html","link"
    ])

def normalize_name(name: str) -> Tuple[str,bool]:
    if not name:
        return "", False
    if name in KNOWN:
        return name, True
    if name in ALIASES:
        can = ALIASES[name]
        return can, (can in KNOWN)
    return name, (name in KNOWN)

def dict_from_overrides(s: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for part in filter(None, [p.strip() for p in s.split(",")]):
        if "=" in part:
            k,v = part.split("=",1)
            k = k.strip(); v = v.strip()
            try:
                out[k] = int(v) if re.fullmatch(r"[+-]?\d+", v) else float(v)
            except ValueError:
                out[k] = v
    return out

def content_to_text(m_content: Any) -> str:
    # OpenAI style can be str or list of {type,text|image_url|...}
    if isinstance(m_content, str):
        return m_content
    if isinstance(m_content, list):
        buf = []
        for seg in m_content:
            if isinstance(seg, dict) and seg.get("type") == "text":
                buf.append(str(seg.get("text","")))
        return "".join(buf)
    return str(m_content or "")

# ---------- tool spec parsing from system prompt ----------
class ToolSpec:
    def __init__(self, name: str, desc: str, params: List[Tuple[str,bool]], usage_xml: Optional[str]):
        self.name = name
        self.desc = desc
        self.params = params          # [(name, required)]
        self.usage_xml = usage_xml     # raw block or None

    def openai_schema(self, example_mode: str = "py") -> Dict[str, Any]:
        props = { p: {"type":"string"} for (p, _) in self.params }
        required = [ p for (p, req) in self.params if req ]
        desc = self.desc
        if example_mode == "xml" and self.usage_xml:
            one_line = " ".join(self.usage_xml.split())
            desc = (desc + " Example XML: " + one_line)[:1024]
        elif example_mode == "py":
            # Make a compact pythonic example: tool(p1="…", p2="…")
            arglist = []
            for (p, req) in self.params:
                placeholder = "..." if req else "..."
                arglist.append(f'{p}="{placeholder}"')
            ex = f"{self.name}(" + ", ".join(arglist) + ")"
            desc = (desc + " e.g. " + ex)[:1024]
        return {
            "type":"function",
            "function":{
                "name": self.name,
                "description": desc,
                "parameters":{
                    "type":"object",
                    "properties": props,
                    **({"required": required} if required else {})
                }
            }
        }

def parse_tools_from_system(system_text: str) -> Dict[str, ToolSpec]:
    """
    Parse between '# Tools' and the next top-level heading '# ' (e.g. '# Tool Use ...').
    Tool block begins with '## tool_name', followed by 'Description:' and 'Parameters:' and optional 'Usage:' xml.
    """
    # Isolate tools block
    m = re.search(r"(?is)#\s*Tools\s+(.*?)(?:\n#\s+|$)", system_text)
    if not m:
        return {}
    block = m.group(1)
    # Split into chunks beginning with '## toolname'
    chunks = re.split(r"\n(?=##\s+)", block)
    specs: Dict[str, ToolSpec] = {}
    for ch in chunks:
        mname = re.match(r"(?is)##\s+([a-zA-Z0-9_]+)\s*\n", ch)
        if not mname:
            continue
        name = mname.group(1).strip()
        mdesc = re.search(r"(?is)Description:\s*(.+?)(?:\n(?:Parameters:|Usage:)|$)", ch)
        desc = mdesc.group(1).strip() if mdesc else ""
        # Parameters (required/optional)
        params: List[Tuple[str,bool]] = []
        for p in re.findall(r"(?im)^\s*[-•]\s*([a-zA-Z0-9_]+)\s*:\s*\((required|optional)\)", ch):
            params.append((p[0], p[1].lower()=="required"))
        musage = re.search(r"(?is)Usage:\s*(<[^>]+>[\s\S]*?)\s*(?:\n##\s+|$)", ch)
        usage_xml = musage.group(1).strip() if musage else None
        specs[name] = ToolSpec(name, desc, params, usage_xml)
    return specs

# ---------- XML emit ----------
def args_to_xml_from_obj(obj: Dict[str, Any]) -> str:
    out=[]
    for k,v in (obj or {}).items():
        if isinstance(v,(str,int,float,bool)) or v is None:
            out.append(f"<{k}>{xml_escape(v)}</{k}>")
        else:
            out.append(f"<{k}>{xml_escape(json.dumps(v, ensure_ascii=False))}</{k}>")
    return "".join(out)

def tool_to_xml_direct(tag: str, args_json: str, logger: logging.Logger, param_order: Optional[List[str]]=None) -> str:
    try:
        obj = json.loads(args_json) if args_json else {}
    except Exception:
        logger.warning("Arguments JSON parse failed for tool '%s': %s", tag, args_json)
        obj = {}
    # Order by spec if given
    if param_order:
        ordered = {}
        for k in param_order:
            if k in obj:
                ordered[k] = obj[k]
        for k,v in obj.items():
            if k not in ordered:
                ordered[k] = v
        obj = ordered
    return f"<{tag}>{args_to_xml_from_obj(obj)}</{tag}>"

def tool_to_xml_mcp(server_name: str, tool_name: str, args_json: str) -> str:
    return (
        f"<use_mcp_tool>"
        f"<server_name>{xml_escape(server_name)}</server_name>"
        f"<tool_name>{xml_escape(tool_name)}</tool_name>"
        f"<arguments>{xml_escape(args_json or '{}')}</arguments>"
        f"</use_mcp_tool>"
    )

# ---------- grammar (experimental) ----------
def minimal_xml_grammar(which: str) -> str:
    """
    which: 'plan' | 'act' | 'both'
    Very permissive: one of known tool tags, with any content except literal '</'.
    """
    plan_tags = ["plan_mode_respond","ask_followup_question"]
    act_tags  = ["execute_command","read_file","write_to_file","replace_in_file","search_files","list_files",
                 "list_code_definition_names","use_mcp_tool","access_mcp_resource",
                 "ask_followup_question","attempt_completion","new_task","load_mcp_documentation"]
    if which == "plan":
        tags = plan_tags
    elif which == "act":
        tags = act_tags
    else:
        tags = sorted(set(plan_tags + act_tags))
    choices = " | ".join(tags)
    return f"""
root ::= ws tool ws
ws   ::= ( " " | "\\t" | "\\r" | "\\n" )*
tool ::= { ' | '.join(f'{t}_tag' for t in tags) }

blob ::= ( not_lt_slash | any_char_but_lt )*
not_lt_slash ::= "<" ~"/"
any_char_but_lt ::= ~"<"

{ ''.join(f'''
{t}_tag ::= "<{t}>" blob "</{t}>"
''' for t in tags) }
""".strip()

# ---------- mode detection ----------
def detect_mode(messages: List[Dict[str, Any]]) -> str:
    # Look at latest user message text for "Current Mode: PLAN/ACT"
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        text = content_to_text(m.get("content"))
        if "Current Mode" in text:
            if "PLAN MODE" in text or "PLAN" in text:
                return "PLAN"
            if "ACT MODE" in text or "ACT" in text:
                return "ACT"
        # Also allow a hint field sent by some clients
        if "mode: PLAN" in text:
            return "PLAN"
        if "mode: ACT" in text:
            return "ACT"
    return "ACT"

# ---------- app ----------
def create_app(cfg):
    app = FastAPI()
    log = logging.getLogger("cline-xml-shim")

    # load aliases once
    load_custom_aliases(cfg.custom_aliases_json, log)

    async def upstream_post(payload: Dict[str, Any], ctx: Dict[str, Any]):
        upstream_body = dict(payload)

        # tool_choice handling
        if cfg.force_tool_choice_none or cfg.tool_choice == "none":
            upstream_body["tool_choice"] = "none"
        elif cfg.tool_choice == "auto":
            upstream_body["tool_choice"] = "auto"

        # Insert guardrail prompt AFTER Cline system (optional)
        if cfg.guardrail_prompt:
            guard = {
                "role":"system",
                "content":"Output exactly one tool per message using the XML format shown above. Do not use native function calling. If you need to chat in plan mode, use <plan_mode_respond>."
            }
            upstream_body.setdefault("messages", [])
            # place after the first system (or at end if none)
            ins = 1 if upstream_body["messages"] and upstream_body["messages"][0].get("role")=="system" else len(upstream_body["messages"])
            upstream_body["messages"].insert(ins, guard)

        # Mode detection
        mode = detect_mode(upstream_body.get("messages", []))

        # Sampling policy
        eff_sampling = {}
        if cfg.strip_client_sampling:
            for k in list(upstream_body.keys()):
                if k in SAMPLING_KEYS:
                    upstream_body.pop(k, None)
        # globals
        eff_sampling.update(dict_from_overrides(cfg.set_sampling))
        # per-mode (override globals)
        if mode == "PLAN":
            eff_sampling.update(dict_from_overrides(cfg.set_sampling_plan))
        else:
            eff_sampling.update(dict_from_overrides(cfg.set_sampling_act))
        upstream_body.update(eff_sampling)

        # Extract tools from system prompt
        tool_specs: Dict[str, ToolSpec] = {}
        if cfg.extract_tools:
            # concat all system messages (Cline sends one big one)
            sys_texts = [m.get("content","") for m in upstream_body.get("messages",[]) if m.get("role")=="system" and isinstance(m.get("content"), str)]
            if sys_texts:
                merged = "\n\n".join(sys_texts)
                tool_specs = parse_tools_from_system(merged)
                if tool_specs:
                    tools_arr = [spec.openai_schema(cfg.tool_examples) for spec in tool_specs.values()]
                    upstream_body["tools"] = tools_arr

        # Grammar (experimental; some servers ignore when --jinja is enabled)
        if cfg.enforce_xml and cfg.enforce_xml != "off":
            which = "both" if cfg.enforce_xml == "both" else ("plan" if mode=="PLAN" else "act")
            upstream_body["grammar"] = minimal_xml_grammar(which)

        # For per-turn logging/summary
        ctx["mode"] = mode
        ctx["tool_specs"] = {k: {"params":[p for (p,_) in v.params]} for k,v in tool_specs.items()}
        ctx["effective_sampling"] = eff_sampling

        if cfg.log_body and log.isEnabledFor(logging.DEBUG):
            log.debug(">>> Upstream request: %s", json.dumps({k: (upstream_body[k] if k!="messages" else "[messages elided]") for k in upstream_body}, ensure_ascii=False)[:2000])

        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{cfg.upstream}/v1/chat/completions", json=upstream_body)
            if cfg.trace_stream and upstream_body.get("stream"):
                log.debug("STREAM open -> %s", r.status_code)
            return r

    # Conversion helpers (use parsed spec if available)
    def convert_tool_call(native_name: str, args_json: str, tool_specs: Dict[str, Any]) -> Tuple[str, bool]:
        """Return (xml_str, ok)."""
        name = native_name or ""
        canonical, known = normalize_name(name)
        # param order from spec (if available)
        order = None
        if tool_specs and canonical in tool_specs:
            order = tool_specs[canonical].get("params")

        # known canonical
        if known and canonical in KNOWN:
            return tool_to_xml_direct(canonical, args_json, logging.getLogger("cline-xml-shim"), order), True

        # browser-ish heuristic only if default server provided
        if cfg.default_mcp_server and looks_browsery(name):
            xml = tool_to_xml_mcp(cfg.default_mcp_server, name, args_json)
            return xml, True

        # unknown
        if cfg.allow_unknown_as_mcp and cfg.default_mcp_server:
            xml = tool_to_xml_mcp(cfg.default_mcp_server, name, args_json)
            return xml, True

        if cfg.strict_xml:
            return "", False

        # Literal fallback: emit a tag with same name (may fail on Cline)
        xml = tool_to_xml_direct(name, args_json, logging.getLogger("cline-xml-shim"), order)
        return xml, True

    # ---------- non-streaming path ----------
    async def handle_nonstream(client_body: dict):
        ctx: Dict[str, Any] = {}
        r = await upstream_post(client_body, ctx)
        try:
            j = r.json()
        except Exception as e:
            logging.getLogger("cline-xml-shim").error("Upstream non-stream JSON decode error: %s", e)
            return JSONResponse({"error":"bad_upstream_json"}, status_code=502)

        if not j.get("choices"):
            return JSONResponse(j, status_code=r.status_code)

        ch = j["choices"][0]
        msg = ch.get("message", {}) or {}
        tcs = msg.get("tool_calls") or []
        if tcs:
            # Multi-tool policy
            selected = []
            if cfg.multi_tool_policy == "first":
                selected = tcs[:1]
            elif cfg.multi_tool_policy == "merge":
                selected = tcs
            else:  # error
                return JSONResponse({
                    "id": j.get("id"), "object": "chat.completion",
                    "created": j.get("created"), "model": j.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "message": {"role":"assistant","content": f"<ask_followup_question><question>Multiple tool calls not allowed in one message. Please send only one.</question></ask_followup_question>"},
                        "finish_reason": "stop"
                    }]
                }, status_code=200)

            xml_parts=[]
            for tc in selected:
                fn = (tc or {}).get("function") or {}
                name = fn.get("name","")
                args = fn.get("arguments","")
                xml, ok = convert_tool_call(name, args, ctx.get("tool_specs",{}))
                if not ok:
                    # strict: synth question
                    xml = f"<ask_followup_question><question>Unknown tool '{xml_escape(name)}' in this context. Please use one of the documented tools.</question></ask_followup_question>"
                    logging.getLogger("cline-xml-shim").error("Unknown tool '%s'; synthesized follow-up.", name)
                xml_parts.append(xml)

            content = (msg.get("content") or "") + "".join(xml_parts)
            ch["message"] = {"role":"assistant","content": content}
            ch["message"].pop("tool_calls", None)
            return JSONResponse(j, status_code=200)

        return JSONResponse(j, status_code=r.status_code)

    # ---------- streaming path ----------
    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        log = logging.getLogger("cline-xml-shim")
        try:
            body = await req.json()
        except Exception:
            log.error("Failed to parse client JSON")
            return JSONResponse({"error": "bad_request"}, status_code=400)

        if cfg.log_body and log.isEnabledFor(logging.DEBUG):
            log.debug("Client request: %s", json.dumps(body, ensure_ascii=False)[:2000])

        # non-stream straight through
        if not body.get("stream"):
            return await handle_nonstream(body)

        # streaming
        ctx: Dict[str, Any] = {}
        up = await upstream_post(body, ctx)
        id_ = f"chatcmpl-xmlshim-{int(time.time())}"
        model = body.get("model") or cfg.model

        # buffers
        tool_buf: Dict[int, Dict[str, str]] = {}
        text_buf = ""
        reasoning_buf: List[str] = []
        sent_any_content = False
        emitted_xml_tools = False

        # tees
        up_fp = open(cfg.dump_upstream, "a", encoding="utf-8") if cfg.dump_upstream else None
        down_fp = open(cfg.dump_downstream, "a", encoding="utf-8") if cfg.dump_downstream else None
        prefix = f"[{id_}] "

        def tee_up(raw_line: str):
            if up_fp:
                up_fp.write(prefix + raw_line + ("" if raw_line.endswith("\n") else "\n"))
                up_fp.flush()

        def send_line(s: str) -> str:
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
            # multi-tool policy
            items = sorted(tool_buf.items())
            if cfg.multi_tool_policy == "first" and len(items) > 1:
                # DEBUG log dropped ones
                for idx, _rec in items[1:]:
                    log.debug("Dropping extra native tool_call idx=%s (policy=first)", idx)
                items = items[:1]
            elif cfg.multi_tool_policy == "error" and len(items) > 1:
                tool_buf.clear()
                xml = "<ask_followup_question><question>Multiple tool calls not allowed in one message. Please send only one.</question></ask_followup_question>"
                emitted_xml_tools = True
                sent_any_content = True
                return {
                    "id": id_, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{ "index": 0, "delta": {"content": xml}, "finish_reason": None }],
                }

            xml_parts=[]
            for _, v in items:
                name = v.get("name", "")
                args = v.get("args", "")
                # param order from parsed spec (if present)
                spec = ctx.get("tool_specs", {}).get(normalize_name(name)[0], None)
                order = spec.get("params") if isinstance(spec, dict) else None

                canonical, known = normalize_name(name)
                if known and canonical in KNOWN:
                    xml = tool_to_xml_direct(canonical, args, log, order)
                    ok = True
                else:
                    # heuristics / strict / mcp wrapping
                    if cfg.default_mcp_server and looks_browsery(name):
                        xml = tool_to_xml_mcp(cfg.default_mcp_server, name, args)
                        ok = True
                        log.warning("Browser-ish tool '%s' -> MCP server=%s", name, cfg.default_mcp_server)
                    elif cfg.allow_unknown_as_mcp and cfg.default_mcp_server:
                        xml = tool_to_xml_mcp(cfg.default_mcp_server, name, args)
                        ok = True
                        log.warning("Unknown tool '%s' -> MCP server=%s", name, cfg.default_mcp_server)
                    elif cfg.strict_xml:
                        xml = f"<ask_followup_question><question>Unknown tool '{xml_escape(name)}'. Please use one documented tool per message.</question></ask_followup_question>"
                        ok = False
                        log.error("Unknown tool '%s'; synthesized follow-up.", name)
                    else:
                        xml = tool_to_xml_direct(name, args, log, order)
                        ok = True
                        log.warning("Unknown tool '%s' emitted as literal tag (may fail).", name)

                xml_parts.append(xml)

            tool_buf.clear()
            emitted_xml_tools = True
            sent_any_content = True
            xml_all = "".join(xml_parts)
            return {
                "id": id_, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model,
                "choices": [{ "index": 0, "delta": {"content": xml_all}, "finish_reason": None }],
            }

        def synthesize_if_empty():
            """Called right before DONE if nothing usable was produced."""
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
                q = cfg.fallback_question or FALLBACK_QUESTION
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

        async def gen():
            nonlocal text_buf, tool_buf, sent_any_content, emitted_xml_tools, reasoning_buf

            # Start with a role delta (helps strict SSE clients)
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

                        # Tail with stop if we never wrote a stop
                        tail = {
                            "id": id_, "object": "chat.completion.chunk",
                            "created": int(time.time()), "model": model,
                            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                        }
                        yield send_line(f"data: {json.dumps(tail)}\n\n")
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
                    if "tool_calls" in d and d["tool_calls"]:
                        for tc in d["tool_calls"]:
                            idx = int(tc.get("index", 0))
                            buf = tool_buf.setdefault(idx, {"name": "", "args": ""})
                            fn = tc.get("function") or {}
                            if "name" in fn and fn["name"]:
                                buf["name"] += fn["name"]
                            if "arguments" in fn and fn["arguments"] is not None:
                                buf["args"] += fn["arguments"]
                            log.info("tool_calls Δ idx=%s name+=%r args_len=%d", idx, fn.get("name",""), len(buf["args"]))
                        continue

                    # upstream signals tool_calls completed → emit converted XML
                    if ch0.get("finish_reason") in ("tool_calls", "tool_call"):
                        if (c := flush_text()):
                            yield send_line(f"data: {json.dumps(c)}\n\n")
                        if (c := flush_tools()):
                            yield send_line(f"data: {json.dumps(c)}\n\n")
                        continue

                    # normal text deltas
                    if isinstance(d.get("content"), str) and d["content"]:
                        text_buf += d["content"]
                        if len(text_buf) >= cfg.flush_bytes or "\n" in text_buf:
                            if (c := flush_text()):
                                yield send_line(f"data: {json.dumps(c)}\n\n")

                    # graceful stop mid-stream
                    if ch0.get("finish_reason") == "stop":
                        if (c := flush_text(mark_stop=True)):
                            yield send_line(f"data: {json.dumps(c)}\n\n")
                        elif not sent_any_content:
                            if (c := synthesize_if_empty()):
                                c["choices"][0]["finish_reason"] = "stop"
                                yield send_line(f"data: {json.dumps(c)}\n\n")
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
                # Tail + DONE
                tail = {
                    "id": id_, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
                }
                yield send_line(f"data: {json.dumps(tail)}\n\n")
                yield send_line("data: [DONE]\n\n")
                return
            finally:
                if up_fp: up_fp.close()
                if down_fp: down_fp.close()
                # Turn summary
                try:
                    log.info(
                        "Turn summary | mode=%s sampling=%s tools_injected=%s converted_xml=%s content_emitted=%s",
                        ctx.get("mode","?"),
                        ctx.get("effective_sampling", {}),
                        bool(ctx.get("tool_specs")),
                        emitted_xml_tools,
                        sent_any_content
                    )
                except Exception:
                    pass

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/")
    def health():
        return {"ok": True, "upstream": cfg.upstream}

    return app

# ---------- main ----------
if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("cline-xml-shim")

    class Cfg:
        upstream = args.upstream.rstrip("/")
        port = args.port
        host = args.host
        model = args.model

        log_level = args.log_level
        log_body = args.log_body
        trace_stream = args.trace_stream
        log_reasoning = args.log_reasoning

        extract_tools = args.extract_tools
        tool_examples = args.tool_examples
        tool_choice = ("none" if args.force_tool_choice_none else args.tool_choice)
        force_tool_choice_none = (args.force_tool_choice_none or args.tool_choice == "none")

        strict_xml = args.strict_xml
        allow_unknown_as_mcp = args.allow_unknown_as_mcp
        default_mcp_server = args.default_mcp_server
        browser_server_name = args.browser_server_name
        custom_aliases_json = args.custom_aliases_json
        multi_tool_policy = args.multi_tool_policy

        synthesize_empty_xml = args.synthesize_empty_xml
        promote_reasoning_if_empty = args.promote_reasoning_if_empty
        fallback_question = args.fallback_question

        strip_client_sampling = args.strip_client_sampling
        set_sampling = args.set_sampling
        set_sampling_plan = args.set_sampling_plan
        set_sampling_act = args.set_sampling_act

        guardrail_prompt = args.guardrail_prompt
        enforce_xml = args.enforce_xml

        flush_bytes = args.flush_bytes

        dump_upstream = args.dump_upstream
        dump_downstream = args.dump_downstream

    app = create_app(Cfg)

    log.info("Cline XML shim listening on %s:%s -> %s", Cfg.host, Cfg.port, Cfg.upstream)

    import uvicorn
    uvicorn.run(app, host=Cfg.host, port=Cfg.port)
