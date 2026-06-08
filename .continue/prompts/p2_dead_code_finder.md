# p2 · Dead Code Finder `[code]`

*Identify what can be deleted outright*

## Prompt

```
Audit the following code for dead weight. Produce a numbered list of everything that can be deleted without affecting behaviour:

- Unused imports / variables / parameters
- Functions never called
- Comments that restate the code
- Error handling for errors that cannot occur in this context
- Abstractions used only once (name the call site)
- Configuration options never toggled

Do NOT suggest rewrites yet. Deletions only.

[PASTE CODE HERE]
```

## Usage

Use before any refactor. Deletion should precede restructuring — you often discover the refactor isn't needed once dead code is gone.
