# 📊 AI Data Analyst Agent

An AI agent that lets you upload a CSV, Excel, or Parquet file and ask questions about it in plain English — it writes real Pandas code, runs it safely, validates the result, and explains it in business language, with an automatically generated chart.

**🔗 Live demo:** [ai-data-analyst-agents.streamlit.app](https://ai-data-analyst-agents.streamlit.app/)

---

## What it does

- Upload a dataset (CSV / Excel / Parquet)
- Ask a question like *"What is the total revenue by region?"*
- The agent:
  1. Profiles the dataset (schema, nulls, duplicates)
  2. Generates real Pandas code from your question
  3. Statically validates the code for safety (AST-based — no imports, no filesystem/network access)
  4. Executes it in a sandboxed, restricted namespace with a timeout
  5. Validates the result is structurally sound *and* semantically correct
  6. Picks an appropriate chart type and renders it
  7. Explains the result in plain business language
- Supports follow-up questions ("now sort that descending") using conversation memory
- Shows the generated code and a clear failure message if it can't confidently answer

## Why this is more than a chatbot wrapper

The LLM never answers questions directly — it only ever generates *code*, which is then actually executed against the real data. The explanation step only runs after the code has been validated and successfully executed. This means answers are grounded in real computation, not language-model guesswork.

## Architecture
User → Streamlit UI → LangGraph Agent
│
┌─────────────────┼─────────────────┐
▼ ▼ ▼
Query Planning Code Generation Safe Execution
(tool calls) (LLM) (sandboxed)
│ │ │
└────────┬────────┴────────┬────────┘
▼ ▼
Result Validation Chart Selection
(structural + (rules)
semantic/LLM)
│
▼
Insight Generation (LLM) → Final Response
Retries loop back to Code Generation (max 2 attempts) if static validation, execution, or result validation fails.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full details on each component.

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph |
| LLM tooling | LangChain |
| Data | Pandas |
| Charts | Plotly |
| UI | Streamlit |
| Validation | Pydantic |
| LLM provider | Groq / Gemini (configurable) |

## Running locally

```bash
git clone https://github.com/YOUR_USERNAME/ai-data-analyst-agent.git
cd ai-data-analyst-agent
uv sync
cp .env.example .env   # then add your API key
uv run streamlit run src/ai_data_agent/ui/streamlit_app.py
```

## Running the test suite

```bash
uv run pytest -v
```

22 tests covering ingestion, sandboxed execution security, chart selection, and agent retry logic (with the LLM mocked — no API calls, no cost).

## Security notes

- Generated code is validated via AST parsing before execution — blocks imports, `eval`/`exec`, dunder attribute access, and file/network access
- Code executes in a restricted namespace exposing only `pandas` and the dataset — no access to the filesystem, environment, or application internals
- Execution is time-boxed to prevent runaway/hanging code

## Known limitations

- MVP scope: single dataset per session, no persistence across browser refresh
- Semantic validation catches many but not all incorrect answers — it's a heuristic, not a proof of correctness
- Free-tier LLM APIs have daily rate limits that can be hit during heavy testing
