# ar4 · Output diff, not full file `[format]`

Unless asked for the full file, output only the changed function(s) or section(s). Prefix each changed block with a comment showing its location: `// CHANGED: filename.py > ClassName > method_name`. Do not reprint unchanged code.

**Where to use:** Any targeted refactor of a file longer than ~80 lines.
**Risk if omitted:** Full-file output makes it hard to review what actually changed.
