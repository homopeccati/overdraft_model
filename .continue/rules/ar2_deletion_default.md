# ar2 · Deletion is the default answer `[scope]`

When reviewing code for simplification, your first instinct must be to delete, not to restructure. Before proposing a refactor, ask: can this be removed entirely? A function nobody calls, a comment that restates the code, a config option never toggled — these should be deleted, not refactored.

**Where to use:** Any "clean up this code" session.
**Risk if omitted:** LLMs default to restructuring because it looks like work. Deletion is usually the right call.
