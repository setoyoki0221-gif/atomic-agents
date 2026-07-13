# Live rules

These rules are re-injected into Claude's context every turn (and before relevant edits) so they stay
in front of the model. Each rule below is a frontmatter block plus body. Edit freely; changes take
effect on the next prompt. Commit this file so the team shares the rules.

---
description: Clean code principles
priority: 5
---
- Follow the clean-code principles in `.claude/clean-code-principles.md` (Uncle Bob, Fowler, Beck, Metz, Feathers) when writing, reviewing, or refactoring code.
- If you have not read that file yet this session, read it before your next code change, then apply it.
- Core convention regardless: no inline comments unless they capture a real hidden constraint; lean on naming and structure.
