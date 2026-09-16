"""Offline MCP protocol fixture; no third-party packages, network or shell."""
import json
import os
import sys
import time

mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
audit = sys.argv[2] if len(sys.argv) > 2 else None
if audit:
    with open(audit + ".pid", "w", encoding="utf8") as pidfile:
        pidfile.write(str(os.getpid()))


def send(value):
    print(json.dumps(value), flush=True)


for line in sys.stdin:
    request = json.loads(line)
    if audit:
        with open(audit, "a", encoding="utf8") as log:
            log.write(json.dumps(request) + "\n")
    method = request.get("method")
    if method is None:
        continue
    rid = request.get("id")
    params = request.get("params", {})
    if method == "initialize":
        if mode == "init_hang":
            time.sleep(10)
        version = "1900-01-01" if mode == "bad_version" else "2025-11-25"
        if mode == "older":
            version = "2025-06-18"
        result = {"protocolVersion": version, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "offline-fixture", "version": "1"}}
        if mode == "no_tools":
            result["capabilities"] = {}
    elif method == "notifications/initialized":
        send({"jsonrpc": "2.0", "id": "server-ping", "method": "ping"})
        send({"jsonrpc": "2.0", "id": "unsupported", "method": "sampling/createMessage", "params": {}})
        continue
    elif method == "tools/list":
        tool = {"name": "echo", "description": "Echo text", "inputSchema": {
            "type": "object", "properties": {"text": {"type": "string"}},
            "required": ["text"], "additionalProperties": False}}
        if mode == "bad_schema":
            tool["inputSchema"] = {"type": "object", "properties": 1}
        if mode == "bad_name":
            tool["name"] = "not portable.tool"
        if mode in ("structured", "bad_output"):
            tool["outputSchema"] = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
        result = {"tools": [tool]}
        if mode == "pages":
            if params.get("cursor"):
                result["tools"][0]["name"] = "second"
            else:
                result["nextCursor"] = "page2"
        if mode == "cursor_loop":
            result = {"tools": [], "nextCursor": "again"}
        if mode == "duplicate":
            result["tools"].append(tool)
    elif method == "tools/call":
        if mode == "exit":
            sys.exit(3)
        if mode == "hang":
            time.sleep(10)
        if mode == "bad_json":
            print("not-json", flush=True)
            continue
        if mode == "rpc_error":
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": "test remote error"}})
            continue
        text = params["arguments"]["text"]
        if mode == "env":
            text = json.dumps({"value": os.environ.get("AGENTSCOPE_MCP_TEST_ALLOWED", "missing"),
                               "has_home": "HOME" in os.environ, "has_path": "PATH" in os.environ})
            assert "_meta" not in params
        if mode in ("large", "oversize"):
            text = "large report data. " * 2000
        result = {"content": [{"type": "text", "text": text}]}
        if mode == "tool_error":
            result["isError"] = True
        if mode == "image":
            result["content"] = [{"type": "image", "data": "AA==", "mimeType": "image/png"}]
        if mode in ("structured", "bad_output"):
            result["structuredContent"] = {"answer": 42 if mode == "structured" else "wrong"}
        if mode == "wrong_id":
            rid += 10
        if mode == "stderr":
            sys.stderr.write("diagnostic only\n" * 10000)
            sys.stderr.flush()
    else:
        continue
    send({"jsonrpc": "2.0", "id": rid, "result": result})

if mode == "stubborn":
    time.sleep(10)
if audit:
    with open(audit, "a", encoding="utf8") as log:
        log.write(json.dumps({"event": "stdin_closed"}) + "\n")
