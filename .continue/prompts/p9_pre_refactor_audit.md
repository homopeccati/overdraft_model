# p9 · Pre-Refactor Audit `[process]`

*Understand the code before touching it*

## Prompt

```
Audit the following code without changing anything. Produce:

1. A one-sentence description of what this code actually does
2. A list of its problems, ranked: [CRITICAL | MINOR | COSMETIC]
3. Your recommended sequence of changes (what to do first, second, third)
4. An estimate: after cleanup, roughly how many lines should remain?

Do not write any refactored code yet.

[PASTE CODE HERE]
```

## Usage

Run this before any refactoring session. The "lines remaining" estimate is a useful anchor — if the model thinks 80 lines can become 40, you know what to hold it to.
