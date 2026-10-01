# Exercise 1: Your First LLM Call

## What This Exercise Does

Every agent is a plain model call with extra structure around it. Before adding tools, loops or memory, confirm you can call the model through LangChain and see exactly what comes back - including how many tokens the call used.

## Time and Cost

- **Time:** ~10 min
- **Cost:** $0.00 - everything runs on your own machine.

## Prerequisites

- `00-START-HERE.md` complete - venv active, `llm_config.py` created, `smoke_test.py` passing

## Steps

### Step 1: Activate your environment

```
cd ~/course/agentic-ai
source venv/bin/activate
```

You should see `(venv)` in your prompt. Every later import depends on this.

### Step 2: Make your first call

**In VS Code:** create a new file named `01_first_call.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
from llm_config import get_llm

llm = get_llm()
response = llm.invoke("Explain what an AI agent is, in exactly two sentences.")
print(response.content)
print("Token usage:", response.usage_metadata)
```

```
python3 01_first_call.py
```

`get_llm()` (from your `llm_config.py`) returns a LangChain chat model. `.invoke()` sends one message and waits for the full reply. `usage_metadata` is the token count for the call - the unit all model costs are measured in, which you'll track properly in Exercise 6.

### Step 3: Use a system message

**In VS Code:** create a new file named `01b_structured_call.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
from langchain_core.messages import SystemMessage, HumanMessage
from llm_config import get_llm

llm = get_llm()
messages = [
    SystemMessage(content="You are a terse assistant. Never use more than one sentence."),
    HumanMessage(content="What is LangGraph?"),
]
response = llm.invoke(messages)
print(response.content)
```

```
python3 01b_structured_call.py
```

Real agents pass a **list of messages**, not a plain string. A `SystemMessage` sets standing behaviour; a `HumanMessage` is the user's turn. Every later exercise builds on this message-list form - it is what lets you keep history (Exercise 3) and give different workers different roles (Exercise 4).

## Verify It Worked

- The first script prints a coherent two-sentence answer and a token-usage line
- The second script's answer is noticeably shorter, proving the system message changed behaviour

## Common Mistakes

- **Forgetting to activate the venv** - Every exercise assumes `(venv)` is showing in that specific terminal.
- **Putting model details in the exercise files** - Model name, region and limits live only in `llm_config.py`; if you find yourself typing a model name into an exercise, stop and edit the config instead.

## Cleanup

Nothing was created in the cloud or installed globally. The `.py` files are yours to keep or delete.
