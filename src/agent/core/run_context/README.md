# run_context

Cross-cutting per-run state: the `(agent, session_id)` pair one run executes under, and the
usage of any supporting LLM call made during it.

## Contents

- `__init__.py` — `run_context()` and `current_run_context()`, a `ContextVar`-backed correlation context; `record_extra_usage()` and `collect_extra_usage()`, a `ContextVar`-backed per-run usage accumulator.
