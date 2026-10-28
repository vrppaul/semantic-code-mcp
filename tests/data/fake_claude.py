"""Claude Code as the agents harness meets it, with no session: it records how it was called
in `calls.jsonl` beside itself, and prints the stream of a session on the login, every server
of its MCP config connected, that answers, whose result is an error where a file named
`fail` lies beside it.

A test copies it into a directory it puts first on PATH, as `claude`, under a line naming
the interpreter that runs it.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
ANSWER = {"path": "src/progress.py", "symbol": "progressbar"}
MCP_CONFIG = json.loads(Path(sys.argv[sys.argv.index("--mcp-config") + 1]).read_text())
INIT = {
    "type": "system",
    "subtype": "init",
    "model": "claude-sonnet-5-5",
    "claude_code_version": "2.1.312",
    "apiKeySource": "none",
    "mcp_servers": [
        {"name": name, "status": "connected", "source": "dynamic"}
        for name in MCP_CONFIG["mcpServers"]
    ],
}
RESULT = {
    "type": "result",
    "subtype": "success",
    "is_error": (HERE / "fail").exists(),
    "num_turns": 2,
    "result": "Found.\n" + json.dumps(ANSWER),
    "total_cost_usd": 0.05,
    "usage": {
        "input_tokens": 4,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 100,
        "output_tokens": 50,
    },
}

with (HERE / "calls.jsonl").open("a") as calls:
    calls.write(json.dumps({"arguments": sys.argv[1:], "environment": dict(os.environ)}) + "\n")
sys.stdout.write(json.dumps(INIT) + "\n" + json.dumps(RESULT) + "\n")
