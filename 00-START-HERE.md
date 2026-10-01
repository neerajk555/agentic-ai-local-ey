# Start Here: Local Agentic AI Exercises

Six hands-on exercises covering agentic AI fundamentals, orchestration, and governance (Modules 12-14), running **entirely on your own machine** - no cloud account, no API key, no cost. The agent code is real LangGraph, the same framework used in production; the only difference from a cloud setup is which model answers the prompts, and that lives in one file (`llm_config.py`).

## Pick a model that fits your machine

CPU-only inference is limited by core count and memory bandwidth, so model size matters more than anything else. Every model below supports tool-calling (needed for Exercise 2 onward).

| Your machine | Model | Download | Notes |
|---|---|---|---|
| 2 vCPU / 4 GB RAM | `qwen2.5:0.5b` | ~400 MB | Fastest; weakest tool-calling |
| **2 vCPU / 8 GB RAM (default)** | **`qwen2.5:1.5b`** | ~1 GB | Best quality for its size |
| 4+ vCPU / 8 GB RAM | `llama3.2:3b` or `qwen2.5:3b` | ~2 GB | Noticeably more reliable |
| 8+ vCPU / 16 GB RAM | `llama3.1:8b` | ~5 GB | Best local tool-calling here |

**Do not swap in `gemma2`, `gemma3`, or the original `llama3`** - those models do not support tool calling in Ollama and Exercise 2 will fail with a "does not support tools" error.

## Install Ollama and pull the models

```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

The installer starts Ollama as a background service. `nomic-embed-text` (~270 MB) is a separate, small model used only for Exercise 3's embeddings. Keep Ollama current (re-run the install line to update) - older versions lack native tool-calling.

Confirm both models are there:

```
ollama list
```

## Set up your Python environment

```
mkdir -p ~/course/agentic-ai && cd ~/course/agentic-ai
python3 -m venv venv
source venv/bin/activate
```

Run `source venv/bin/activate` again in every new terminal - the `(venv)` marker must appear in your prompt.

**In VS Code:** create a new file named `requirements.txt` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```text
langgraph
langchain
langchain-core
langchain-ollama
```

```
pip install -r requirements.txt
```

*Tested with: langgraph 1.2.12, langchain 1.4.3, langchain-core 1.6.6, langchain-ollama 1.1.0.*

## Create the one file every exercise shares

Every exercise gets its model from `llm_config.py`. Changing models later means editing this single file, never the exercises.

**In VS Code:** create a new file named `llm_config.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Change the model without editing code:  export LOCAL_MODEL="llama3.2:3b"
MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5:1.5b")
EMBED_MODEL = "nomic-embed-text"


def get_llm():
    # temperature=0 keeps tool-calling as predictable as a small model allows;
    # num_predict caps the length of any single reply.
    return ChatOllama(model=MODEL, temperature=0, num_predict=512)


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


# Local models are free, so cost tracking in Exercise 6 will show $0.
PRICE_PER_M_INPUT = 0.0
PRICE_PER_M_OUTPUT = 0.0
```

## Confirm everything works

**In VS Code:** create a new file named `smoke_test.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
from llm_config import get_llm

llm = get_llm()
reply = llm.invoke("Reply with exactly the word: OK")
print("Model replied:", reply.content)
print("Token usage:", reply.usage_metadata)
```

```
python3 smoke_test.py
```

You should see `Model replied: OK` (or very close to it) and a token-usage line. The first call is slower because Ollama loads the model into memory. If you get a connection error, Ollama is not running: try `sudo systemctl start ollama`.

## What to expect from small local models

- **Speed:** on 2 vCPUs expect several seconds per reply; Exercises 2 and 4 make two or three model calls per run.
- **Tool over-calling:** small models sometimes call a tool when none is needed (even for "hi"). Exercise 2 deliberately lets you observe this.
- **Occasional misroutes** in Exercise 4's classification step. This is a real, current limitation of small models, not a bug in your code.

If your machine is simply too slow, change `LOCAL_MODEL` to a smaller one from the table above, or use the AWS version of these exercises.

## The six exercises

| # | Exercise | What it covers |
|---|---|---|
| 1 | First LLM Call | The base model call everything else builds on |
| 2 | ReAct Agent with Tools | Reason-act-observe loop with `StateGraph`, `ToolNode`, `tools_condition` |
| 3 | Memory and RAG | Conversation memory (checkpointer) and retrieval over your own documents |
| 4 | Multi-Agent Orchestration | A supervisor routing work to specialist workers |
| 5 | Guardrails and Human-in-the-Loop | Input filtering plus a real `interrupt()` approval gate |
| 6 | Observability and Cost Tracking | A callback handler logging every step and the token cost |

Start with Exercise 1.
