Log an issue, progress update, or finding to the experiment documentation in `ppio/docs/experiments/`.

## Behavior

1. Identify the current experiment from context (branch name, running pods, or ask if unclear)
2. Find the corresponding experiment doc in `ppio/docs/experiments/`
3. Append the entry to the appropriate section:
   - **Issue Log**: For bugs, crashes, race conditions, deployment failures — include When, Symptom, Root cause, Fix
   - **Progress**: For status updates, epoch completions, metric milestones
   - **Notes**: For observations, design decisions, or learnings
4. Update the **Status** line at the top if the status has changed

## Entry Format

### For issues:
```markdown
### Issue N: <Short Title>

**When**: <context — which epoch, which node, what was happening>
**Symptom**: <what went wrong, error messages>
**Root cause**: <why it happened>
**Fix**: <what was changed to resolve it>
```

### For progress updates:
```markdown
### <Date> — <Summary>

<Details: metrics, epoch count, files processed, etc.>
```

## Guidelines

- Be concise but include enough detail to reproduce/understand the issue later
- Include file paths and line numbers for code-related issues
- For fixes, list the specific files changed
- Keep the Status line at the top of the doc accurate
