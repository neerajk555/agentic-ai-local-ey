# Exercise 6: Observability and Cost Tracking

## What This Exercise Does

Module 9 covered CloudWatch and X-Ray for ordinary applications; agents need the same discipline, because their behaviour is less predictable and a bad decision several steps back is hard to spot from the final answer. Module 14 adds cost governance - tracking token usage. You'll build a LangChain **callback handler** (the real extension point that tracing products plug into) that records every tool call and every model call, including token counts, then read the trace back and estimate the cost.

## Time and Cost

- **Time:** ~15 min
- **Cost:** $0.00 - everything runs on your own machine.

## Prerequisites

- `00-START-HERE.md` complete - venv active, `llm_config.py` created, `smoke_test.py` passing
- Exercise 2 complete (reuses `tools.py` and `react_agent.py`)

## Steps

### Step 1: Write a callback handler that logs to a file

**In VS Code:** create a new file named `trace_logger.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
import json
import time
from langchain_core.callbacks import BaseCallbackHandler


class FileTraceLogger(BaseCallbackHandler):
    """Writes every tool call and every model call to a local JSON-lines file."""

    def __init__(self, log_path="agent_trace.jsonl"):
        self.log_path = log_path
        open(self.log_path, "w").close()  # start each run with a clean trace

    def _write(self, event):
        event["timestamp"] = time.time()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = kwargs.get("name") or (serialized or {}).get("name", "unknown")
        self._write({"event": "tool_start", "tool": name, "input": input_str})

    def on_tool_end(self, output, **kwargs):
        content = getattr(output, "content", output)
        self._write({"event": "tool_end", "output": str(content)})

    def on_llm_end(self, response, **kwargs):
        usage = {}
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (AttributeError, IndexError):
            pass
        self._write({
            "event": "llm_end",
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        })
```

`BaseCallbackHandler` has hooks for many events; this uses three: a tool starting, a tool finishing, and a model call finishing (where token usage is available). Writing one JSON object per line (JSON Lines) means each line parses independently, so a crashed run still leaves a readable partial trace. The handler empties the file when created, so each run starts clean.

### Step 2: Attach it to your Exercise 2 agent

**In VS Code:** create a new file named `06_traced_run.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
from langchain_core.messages import HumanMessage
from react_agent import graph
from trace_logger import FileTraceLogger

logger = FileTraceLogger("agent_trace.jsonl")
result = graph.invoke(
    {"messages": [HumanMessage(content="What is 12 times 12?")]},
    config={"callbacks": [logger]},
)
print("Final answer:", result["messages"][-1].content)
print("Trace written to agent_trace.jsonl")
```

```
python3 06_traced_run.py
```

Passing `config={"callbacks": [logger]}` attaches observability to any LangChain/LangGraph run without touching the agent's own code - the agent doesn't know the logger exists, which is exactly the point.

### Step 3: Read the trace back and see the cost

**In VS Code:** create a new file named `06_read_trace.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
import json
from llm_config import PRICE_PER_M_INPUT, PRICE_PER_M_OUTPUT

total_in = 0
total_out = 0
with open("agent_trace.jsonl") as f:
    for line in f:
        e = json.loads(line)
        if e["event"] == "tool_start":
            print(f"[{e['timestamp']:.2f}] TOOL   {e['tool']} <- {e['input']}")
        elif e["event"] == "tool_end":
            print(f"[{e['timestamp']:.2f}] TOOL   returned {e['output']}")
        elif e["event"] == "llm_end":
            total_in += e["input_tokens"]
            total_out += e["output_tokens"]
            print(f"[{e['timestamp']:.2f}] MODEL  {e['input_tokens']} tokens in, {e['output_tokens']} out")

cost = total_in / 1e6 * PRICE_PER_M_INPUT + total_out / 1e6 * PRICE_PER_M_OUTPUT
print(f"\nTotal tokens: {total_in} in / {total_out} out")
print(f"Estimated cost: ${cost:.6f}")
```

```
python3 06_read_trace.py
```

Ollama reports token counts for most models; if you see zeros your model/version did not report them, which does not affect the exercise. Local cost is always $0. The price constants come from `llm_config.py` - this is the same arithmetic as cost governance dashboards, at the smallest possible scale.

## Verify It Worked

- `agent_trace.jsonl` contains a `tool_start` and `tool_end` for `calculator`, plus `llm_end` lines
- The read-back prints a timestamped sequence of exactly what the agent did, then total tokens and an estimated cost

## Common Mistakes

- **Assuming callbacks change agent behaviour** - Callbacks only observe; they cannot and should not alter decisions.
- **Only logging tool calls** - The model's own calls are where token cost lives; `on_llm_end` is what makes cost tracking possible.
- **Reading a stale trace** - The trace is reset each run; if numbers look wrong, re-run `06_traced_run.py` first.

## Cleanup

Optional: `rm -f agent_trace.jsonl`
