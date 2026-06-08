# p7 · Docstring Triage `[docs]`

*Fix over-verbose function documentation*

## Prompt

```
For each docstring in the following code, rewrite it to a single line that states only what the function returns or does that isn't already obvious from its name and signature.

Delete the docstring entirely if the function name is self-explanatory.
Do not include parameter descriptions unless a parameter name is ambiguous.
Do not include return type (that belongs in the type signature).

[PASTE CODE HERE]
```

## Usage

Most docstrings are a parameter list dressed up as prose. If your signature is typed, the docstring should say what isn't already encoded there.
