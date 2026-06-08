# p3 · Abstraction Audit `[code]`

*Find over-engineered structure*

## Prompt

```
Review the following code for unnecessary abstraction. For each class, interface, helper function, or module you find, answer:

1. How many call sites use this? (list them)
2. If only one: paste what the inlined version would look like
3. If zero: mark DELETE

Do not rewrite anything yet. Output as a structured list.

[PASTE CODE HERE]
```

## Usage

LLMs love to introduce Manager, Handler, Factory, and Wrapper classes. This prompt exposes them. If a class has one method and one call site, it should almost always be inlined.
