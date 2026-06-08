# p8 · Architecture Doc Triage `[docs]`

*Compress an AI-generated design doc*

## Prompt

```
The following architecture document was AI-generated and is too long. Compress it:

1. Remove any section that restates the code structure (the code is the source of truth)
2. Remove any "future considerations" or "out of scope" sections
3. Keep only: decisions made, and the reason each decision was made over the alternative
4. Target: half the current word count

Output the compressed document.

[PASTE DOC HERE]
```

## Usage

Architecture docs should record decisions, not describe implementations. "We use Redis for session state because X" is worth keeping. "The system has a Redis component" is not.
