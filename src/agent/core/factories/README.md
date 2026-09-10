# factories

Construct registered instances from startup configuration. Wiring happens once per process; nothing here re-reads config afterward.

## Contents

- `app.py` — `build_registries()`, builds the LLM, agent, tool, strategy, and guardrail registries plus process-wide config from an already-parsed `AppConfig`; `configure_tracing()`, builds and registers the process-wide `TracerProvider`.
