# ar5 · Justify every abstraction `[architecture]`

Before introducing any new function, class, interface, or module, state explicitly: (a) what it replaces, (b) how many call sites will use it, (c) why inlining is insufficient. If you cannot answer all three, do not introduce the abstraction.

**Where to use:** Architectural refactoring sessions.
**Risk if omitted:** Abstraction is the primary vehicle for LLM-generated complexity.
