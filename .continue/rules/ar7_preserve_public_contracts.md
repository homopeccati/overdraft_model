# ar7 · Preserve public contracts `[safety]`

Do not change function signatures, return types, class interfaces, or exported names unless the task explicitly requires it. Refactoring is an internal concern. If simplification requires a breaking change, flag it as a separate step and stop.

**Where to use:** Any refactor of library code or shared modules.
**Risk if omitted:** Silent signature changes cause runtime failures invisible to the LLM.
