# Architecture

## Agent Graph (LangGraph)

The agent is a state graph with the following nodes:

1. **dataset_understanding** — no-op/logging step; the dataset profile is already computed at ingestion time
2. **query_planning** — calls `inspect_dataset` and `describe_dataset` tools to surface schema/stats context
3. **code_generation** — LLM call; generates Pandas code assigned to a `result` variable, given the schema and (if present) the previous conversation turn
4. **code_validation** — AST-based static security check + column-existence check
5. **safe_execution** — runs validated code in a restricted namespace (`pandas` + `df` only) inside a worker thread with a join-based timeout
6. **result_validation** — checks the result isn't empty/NaN, then asks the LLM a single YES/NO question: does this code fully answer the question?
7. **chart_selection** — rule-based (no LLM) selection of chart type based on the Python type/shape of the result
8. **insight_generation** — LLM call; explains the result in plain business language
9. **final_response** — assembles code, chart, insight, and success/error status; appends this turn to conversation history

Conditional edges route `code_validation` and `result_validation` failures back to `code_generation`, capped at 2 retries, after which the graph routes to `final_response` with a clear failure message.

## Security model (defense in depth)

1. **Static validation (AST)** — rejects imports, `eval`/`exec`, dunder attribute access, and code that doesn't assign to `result`
2. **Restricted execution namespace** — `__builtins__` is replaced with a small safe subset; only `pd` and `df` are in scope
3. **Thread-based timeout** — execution runs in a daemon worker thread with a bounded `join()`, preventing runaway code from hanging the app (chosen over `signal`-based timeouts, which are incompatible with multi-threaded hosting environments like Streamlit Cloud)

## Validation layers

1. **Structural** — result isn't `None`/empty/NaN (cheap, no LLM)
2. **Semantic** — a lightweight single-token LLM call checks whether the code's *intent* matches the question (only runs after structural checks pass, to minimize cost)

## Conversational memory

Each turn's `{question, code, result}` is appended to a bounded conversation history, passed into the next turn's code-generation prompt to resolve references like "that" or "instead." Full chat transcripts are not stored — only the structured code/result pair, to keep prompt size bounded.

## Known engineering decisions worth highlighting

- Chart selection is deliberately rule-based, not LLM-based — it's a deterministic function of Python type/shape, so an LLM call would be slower and less reliable for zero benefit
- The system prompt explicitly forbids `.plot()`/matplotlib calls — the LLM only ever returns data; charting is handled entirely by the application layer
- LLM response content is normalized across providers (some return `str`, others a list of content blocks) via a small adapter function, since the project has run against both Anthropic and Google's APIs during development
