# p4 · Targeted Single-Function Refactor `[code]`

*Clean up one function without touching the rest*

## Prompt

```
Refactor ONLY the function named [FUNCTION_NAME] in the code below.

Constraints:
- Do not touch any other function
- Preserve the exact signature: [PASTE SIGNATURE]
- Reduce line count if possible
- Remove inline comments unless they explain non-obvious logic
- Do not add error handling, logging, or type annotations

Output: the refactored function only. No preamble.

[PASTE CODE HERE]
```

## Usage

The most important prompt in the toolkit. Always scope to a single function. Asking for a file-wide refactor in one pass produces cascading unreviewed changes.
