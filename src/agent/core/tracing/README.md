# tracing

The process-wide content-capture flag, and small helpers shared across every
span-producing call site.

## Contents

- `__init__.py` — `set_capture_content()` and `capture_content_enabled()`, backed by a module-level flag; `record_gated_exception()`, shared by every span that can fail with a content-bearing exception; `stamp_run_context()`, shared by every span that attaches the active run's agent/session ids; `has_exported_span()`, shared by every log/request-id correlation site that reads the current span context; `truncate()` and `truncate_keeping_recent()`, shared by every call site that caps a piece of text before attaching or returning it, keeping its start or its end respectively.
