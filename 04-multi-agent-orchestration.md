# Exercise 4: Multi-Agent Orchestration: Supervisor and Workers

## What This Exercise Does

One agent handling every task tends to do all of them mediocrely. A **supervisor** pattern routes each task to a specialised **worker**, each with its own system prompt. This is Module 12's orchestrator/worker pattern, built for real: a supervisor classifies the request and a conditional edge sends it to the right worker.

## Time and Cost

- **Time:** ~20 min
- **Cost:** $0.00 - everything runs on your own machine.

## Prerequisites

- `00-START-HERE.md` complete - venv active, `llm_config.py` created, `smoke_test.py` passing
- Exercise 2 complete

## Steps

### Step 1: Build the supervisor, the workers, and the graph

**In VS Code:** create a new file named `04_multi_agent.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llm_config import get_llm

llm = get_llm()


def supervisor(state: MessagesState):
    task = state["messages"][-1].content
    decision = llm.invoke([HumanMessage(content=(
        "Classify this request as exactly one word, either 'math' or 'writing'. "
        f"Request: {task}"
    ))])
    return {"messages": [AIMessage(content=f"ROUTE:{decision.content.strip().lower()}")]}


def route_after_supervisor(state: MessagesState):
    return "math_worker" if "math" in state["messages"][-1].content else "writing_worker"


def make_worker(system_prompt: str):
    def worker(state: MessagesState):
        original_task = state["messages"][0].content
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=original_task)])
        return {"messages": [response]}
    return worker


builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor)
builder.add_node("math_worker", make_worker("You are a precise math tutor. Show your work step by step."))
builder.add_node("writing_worker", make_worker("You are a creative writing assistant. Be vivid and imaginative."))
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {"math_worker": "math_worker", "writing_worker": "writing_worker"},
)
builder.add_edge("math_worker", END)
builder.add_edge("writing_worker", END)
graph = builder.compile()

if __name__ == "__main__":
    for task in ["What is 25% of 480?", "Write two sentences about a lighthouse at night."]:
        print(f"\n=== Task: {task} ===")
        result = graph.invoke({"messages": [HumanMessage(content=task)]})
        print("Routed via:", result["messages"][1].content)
        print(result["messages"][-1].content)
```

The supervisor's only job is classification - it never answers the task itself. `make_worker` creates two workers from one function; they differ only in their `SystemMessage`, which is what makes them behave differently. Each worker re-reads `state["messages"][0]` (the *original* task), not the supervisor's routing message. `add_conditional_edges` with an explicit mapping makes the routing easy to read and extend.

### Step 2: Run it and compare the two routes

```
python3 04_multi_agent.py
```

Each task prints which route the supervisor chose, then the worker's answer. The math answer should show step-by-step working; the writing answer should be more descriptive.

## Verify It Worked

- The math task is routed via `ROUTE:math` and the writing task via `ROUTE:writing`
- The two answers have visibly different styles, coming from their different system prompts

## Common Mistakes

- **Workers reading the wrong message** - Reading `state["messages"][-1]` in a worker would hand it the routing label instead of the task.
- **Assuming multi-agent is always better** - For one simple task a single well-prompted agent is simpler and more reliable. Orchestration earns its cost only when roles genuinely differ.
- **Expecting perfect classification** - Free-form classification by a model can occasionally misroute an ambiguous request. Production systems constrain the output (structured output, an allowed-list) or add a fallback.

## Cleanup

Nothing to clean up.
