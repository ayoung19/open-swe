# Managed memory respects Anthropic's cache budget

**Upstream as-is.**

- **What:** `packages/pypi/src/managed_deepagents/_memory_instructions.py` counts existing system, conversation, tool, and request-level cache breakpoints before caching the appended memory block.
- **Why:** Robin's Claude requests can already use all four breakpoints before managed memory is appended; adding a fifth rejects the entire request.
  Memory instructions still reach the model when the cache budget is full.
- **Upstream:** Drop this change once MDA's shared memory append path preserves Anthropic's four-breakpoint limit.
