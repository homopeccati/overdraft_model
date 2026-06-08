# ar1 · Surgical changes only `[scope]`

When refactoring, change ONLY what is explicitly requested. Do not rename unrelated variables, reorder imports, add comments, or restructure logic outside the target scope. Every line you touch that was not asked about is a regression risk.

**Where to use:** Paste at the top of any refactoring system prompt or Claude Project instructions.
**Risk if omitted:** The model will "clean up" surrounding code, changing behaviour silently.
