# protocols

Interfaces for anything with interchangeable implementations.

## Contents

- `illm.py` — `ILLM`, the interface for an outbound adapter that turns messages into a completion; also carries a `model` attribute naming the model it was constructed with
- `itool.py` — `ITool`, the interface for a tool an agent can expose to an LLM
- `istrategy.py` — `IStrategy`, the interface for a reasoning-loop algorithm
- `iguardrail.py` — `IGuardrail`, the interface for a content check run at a guardrail checkpoint; `run_guardrails()`, the shared pipeline that runs a list of them against content, applying each one's configured action
- `isession_store.py` — `ISessionStore`, the interface for per-conversation history storage
