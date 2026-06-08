# p10 · Regression Safety Check `[process]`

*Verify a refactor didn't break behaviour*

## Prompt

```
Compare the original and refactored versions of this code. For each public function or exported symbol:

1. Is the signature identical? (yes/no)
2. Are there any behaviour changes, even minor ones? Describe them precisely.
3. Are there any edge cases the original handled that the refactor does not?

Original:
[PASTE ORIGINAL]

Refactored:
[PASTE REFACTORED]
```

## Usage

Use after every significant refactor before running tests. The model is good at spotting subtle signature drift and silent logic changes when given both versions.
