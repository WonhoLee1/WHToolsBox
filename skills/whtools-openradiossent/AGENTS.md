<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-04 | Updated: 2026-06-04 -->

# skills/whtools-openradiossent

## Purpose
OMC skill package for OpenRadiOSS explicit dynamics agent workflows. Mirrors the structure of `whtools-calculixent` but targets OpenRadiOSS deck conventions, starter/engine split, and the `openradioss_rag.db` knowledge base.

## Key Files

| File | Description |
|------|-------------|
| `SKILL.md` | Skill entry point — OpenRadiOSS agent rules, deck format conventions, and RAG pointers |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `resources/` | Static reference data and RAG database assets for this skill |
| `scripts/` | Build or utility scripts supporting the skill |

## For AI Agents

### Working In This Directory
- Read `SKILL.md` before generating any OpenRadiOSS deck content
- The RAG database (`openradioss_rag.db` at repo root) covers the cloned OpenRadiOSS source in `wht_openradiossent_doc/`
- OpenRadiOSS uses a starter (`_0001.rad`) + engine (`_0001.rad` continuation) split — do not conflate with CalculiX single-file INP format

<!-- MANUAL: -->
