# p1 · Compression Pass `[code]`

*Reduce volume without changing behaviour*

## Prompt

```
Here is a code block. Your task is to reduce its line count by at least 20% while preserving identical behaviour and public interface.

Rules:
- Delete dead code, redundant comments, and single-use helper functions
- Inline trivial wrappers
- Do not add new abstractions
- Do not change names of public symbols
- Output: the compressed block only, then a single line: "Before: X lines → After: Y lines"

[PASTE CODE HERE]
```

## Usage

Run this as a first pass on any LLM-generated file before reviewing logic. The 20% floor is a forcing function — if the model can't find 20%, the code was probably already lean.
