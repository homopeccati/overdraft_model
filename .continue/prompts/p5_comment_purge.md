# p5 · Comment Purge `[code]`

*Strip redundant documentation from code*

## Prompt

```
Remove all comments from the following code that merely restate what the code does. Keep only:
- Comments that explain WHY a non-obvious decision was made
- Warnings about external dependencies or side effects
- Legal headers (do not modify)

Output the cleaned code only.

[PASTE CODE HERE]
```

## Usage

LLMs generate comments compulsively. If you can read the comment and immediately see it in the code, the comment is noise.
