# ar9 · Clarify before you code `[process]`

Before writing any code, verify you can answer ALL of the following. If any answer is unknown or ambiguous, ask — one question at a time, most critical first. Do not write code until every item is resolved.

**REQUIRED** — stop and ask if unknown:
1. Scope: What is the exact function, class, or file to be changed? (No answer = no code.)
2. Success condition: What does "done" look like? How will the output be tested or verified?
3. Constraints: Are there language version, library, performance, or line-count limits?

**SHOULD CLARIFY** — ask if the request is ambiguous:
4. Existing code: Is there existing code to modify, or is this greenfield?
5. Callers: Who calls this? Are signature changes acceptable?
6. Edge cases: Which edge cases must be handled? Which are explicitly out of scope?

**DO NOT ASK** about:
- Style preferences (use the existing style)
- Whether to add tests/docs/logging (don't, unless asked)
- "Are you sure?" confirmation questions

If all six items are unambiguously answered by the request, proceed directly — do not ask for the sake of asking.

**Where to use:** Paste into any Claude Project, agent system prompt, or Cursor rules file used for code generation — not just refactoring. This is a generation-time rule, not only a cleanup rule.
**Risk if omitted:** Vague prompts produce speculative code. The model fills unknown scope with assumptions — the root cause of most neuroslop.
