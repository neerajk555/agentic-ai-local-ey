# Exercise 2: ReAct Agent with Tools

## What This Exercise Does

**ReAct** (Reason + Act) is the foundation of agentic AI: the model decides it needs a tool, calls it, reads the result, and repeats until it can answer - instead of guessing from its training data. This exercise builds a real ReAct agent from LangGraph's core pieces (`StateGraph`, `ToolNode`, `tools_condition`) so you see the actual graph: nodes, edges, and the conditional routing between them.

## Time and Cost

- **Time:** ~25 min
- **Cost:** $0.00 - everything runs on your own machine.

## Prerequisites

- `00-START-HERE.md` complete - venv active, `llm_config.py` created, `smoke_test.py` passing
- Exercise 1 complete

## Steps

### Step 1: Define two tools

**In VS Code:** create a new file named `tools.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, for example '47 * 89' or '340 * 0.15'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())
```

The `@tool` decorator turns a Python function into something a model can choose to call. **The docstring is not decoration** - it is what the model reads to decide when to use the tool. The calculator only allows digits and operators because `eval()` on arbitrary text would be a code-injection risk. (The file is `tools.py`, not `02_tools.py`: Python module names cannot start with a digit.)

### Step 2: Build the agent graph

**In VS Code:** create a new file named `react_agent.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from llm_config import get_llm
from tools import calculator, word_count

tools = [calculator, word_count]
llm_with_tools = get_llm().bind_tools(tools)


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile()

if __name__ == "__main__":
    question = (
        "What is 47 times 89? Also, how many words are in the sentence: "
        "The quick brown fox jumps over the lazy dog"
    )
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    for msg in result["messages"]:
        msg.pretty_print()
```

This is the ReAct loop made visible. `assistant` reasons and may request tools. `tools_condition` checks the assistant's last message: tool calls present -> go to the `tools` node, otherwise finish. `ToolNode` runs whichever tool was requested. The edge from `tools` back to `assistant` is what makes it a loop. `bind_tools` is what tells the model the tools exist at all.

### Step 3: Run it and read the whole trace

```
python3 react_agent.py
```

`pretty_print()` shows the entire conversation: your question, the model's tool-call requests, each tool's result, and the final answer built from those real results.

### Step 4: Check for over-eager tool calling

**In VS Code:** create a new file named `02b_greeting.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
from langchain_core.messages import HumanMessage
from react_agent import graph

result = graph.invoke({"messages": [HumanMessage(content="Say hello in one short sentence.")]})
for msg in result["messages"]:
    msg.pretty_print()

used_tools = any(getattr(m, "tool_calls", None) for m in result["messages"])
print("\nDid the agent call a tool?", used_tools)
```

```
python3 02b_greeting.py
```

A greeting needs no tool, so the expected answer is `False`. Larger models almost always get this right; small ones sometimes call a tool anyway. Seeing this yourself is the point - it is why production agents need guardrails and evaluation, not just a capable-looking demo.

## Verify It Worked

- The trace shows a `calculator` call and a `word_count` call
- The final answer contains the real tool results: 4183 for the multiplication and 9 for the word count
- You can explain which node decides to call a tool and which node actually runs it

## Common Mistakes

- **Vague tool docstrings** - The model chooses tools almost entirely from the docstring; a vague one is the top reason an agent 'ignores' a tool.
- **Using `create_react_agent` from an older tutorial** - That convenience function is deprecated; the `StateGraph` + `ToolNode` + `tools_condition` pattern used here is current.
- **Blaming your code for model flakiness** - Small models miss or misuse tools more often than large ones. Re-run or rephrase before debugging for an hour.

## Cleanup

Nothing to clean up.
