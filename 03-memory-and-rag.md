# Exercise 3: Memory: Conversation History and RAG

## What This Exercise Does

An agent without memory forgets everything between calls. Real agents need two kinds: **short-term** memory (what was said in this conversation) and **long-term** memory (facts retrieved from a knowledge base when relevant - RAG). You'll build both: a LangGraph checkpointer for conversation memory, and a small vector store for retrieval-augmented generation.

## Time and Cost

- **Time:** ~25 min
- **Cost:** $0.00 - everything runs on your own machine.

## Prerequisites

- `00-START-HERE.md` complete - venv active, `llm_config.py` created, `smoke_test.py` passing
- Exercise 2 complete

## Steps

### Step 1: Short-term memory with a checkpointer

**In VS Code:** create a new file named `03_conversation_memory.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from llm_config import get_llm

llm = get_llm()


def assistant(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_edge(START, "assistant")
graph = builder.compile(checkpointer=InMemorySaver())

alex = {"configurable": {"thread_id": "alex-conversation"}}
fresh = {"configurable": {"thread_id": "brand-new-conversation"}}

r1 = graph.invoke({"messages": [HumanMessage(content="My name is Alex and I am learning LangGraph.")]}, alex)
print("Turn 1:", r1["messages"][-1].content)

r2 = graph.invoke({"messages": [HumanMessage(content="What is my name, and what am I learning?")]}, alex)
print("Turn 2 (same thread):", r2["messages"][-1].content)

r3 = graph.invoke({"messages": [HumanMessage(content="What is my name, and what am I learning?")]}, fresh)
print("Turn 3 (brand new thread):", r3["messages"][-1].content)
```

```
python3 03_conversation_memory.py
```

`InMemorySaver` saves the graph's state after every step, keyed by `thread_id`. Turn 2 never repeats your name - LangGraph reloads the saved history for that thread. Turn 3 uses a different `thread_id`, which is a completely separate, empty conversation, so the model has no idea who Alex is.

### Step 2: Long-term memory: embed a small knowledge base

**In VS Code:** create a new file named `03_knowledge_base.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from llm_config import get_embeddings

docs = [
    Document(page_content="The shared EKS cluster for this course runs 10 t3.small worker nodes."),
    Document(page_content="Each participant has a 5 dollar AWS budget for this course."),
    Document(page_content="The load balancer in the ALB exercise bills hourly regardless of traffic."),
    Document(page_content="Helm packages Kubernetes manifests into a reusable, versioned chart."),
]

store = InMemoryVectorStore.from_documents(docs, get_embeddings())
store.dump("knowledge_base.json")
print(f"Embedded {len(docs)} documents and saved knowledge_base.json")
```

```
python3 03_knowledge_base.py
```

`get_embeddings()` uses `nomic-embed-text` through Ollama - a small model that turns text into vectors. Texts with similar meaning get similar vectors, which is what makes search-by-meaning possible. `InMemoryVectorStore` is a simple, dependency-free vector database; production systems use something like OpenSearch or Aurora pgvector, but the retrieval concept is identical.

### Step 3: Retrieve and answer (the RAG pattern)

**In VS Code:** create a new file named `03_rag_query.py` in your `~/course/agentic-ai` folder (Explorer panel, right-click the folder > New File), paste this in, and save (`Ctrl+S`):

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage
from llm_config import get_llm, get_embeddings

store = InMemoryVectorStore.load("knowledge_base.json", get_embeddings())
llm = get_llm()

query = "How many worker nodes are in the shared cluster?"
hit = store.similarity_search(query, k=1)[0]
print("Retrieved:", hit.page_content)

prompt = (
    "Using ONLY the context below, answer the question in one sentence.\n\n"
    f"Context: {hit.page_content}\n\nQuestion: {query}"
)
print("Answer:", llm.invoke([HumanMessage(content=prompt)]).content)
```

```
python3 03_rag_query.py
```

This is RAG in full: embed the question, find the most similar document, and hand *that text* to the model as context. The model answers from your document rather than its training data (which knows nothing about this course's node count). "Using ONLY the context" reduces the chance the model blends in things it half-remembers.

## Verify It Worked

- Turn 2 recalls both Alex and LangGraph; Turn 3 does not
- The retrieval step returns the node-count document, not the budget or load-balancer one
- The final answer states 10 nodes, sourced from the retrieved context

## Common Mistakes

- **Reusing one `thread_id` for unrelated chats** - Everything under one thread_id shares memory; use a fresh id for a clean slate.
- **Confusing the two memory types** - The checkpointer remembers what was *said*; the vector store retrieves *facts* regardless of conversation. Production agents use both.
- **Mixing embedding models** - Documents and queries must be embedded by the same model. If you change `get_embeddings()`, re-run `03_knowledge_base.py` before querying.

## Cleanup

Optional: `rm -f knowledge_base.json`
