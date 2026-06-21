<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-04 | Updated: 2026-06-04 -->

# skills/whtools-calculixent

## Purpose
OMC skill package for CalculiX FEA agent workflows. Defines a 3-tier agent decision tree (LLM fast-path → RAG search → ask user), Abaqus/CalculiX INP coding conventions, and pointers to the SQLite RAG database for resolving multi-physics edge cases.

## Key Files

| File | Description |
|------|-------------|
| `SKILL.md` | Skill entry point — agent decision tree, INP deck standards, element conventions, multi-physics step templates |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `resources/` | Static reference data and RAG database assets for this skill |
| `scripts/` | Build or utility scripts supporting the skill |

## For AI Agents

### Working In This Directory
- Read `SKILL.md` before generating any CalculiX INP content — it contains mandatory conventions (UTF-8 BOM-free, uppercase NSET/ELSET names, quadratic elements preferred, `*INCLUDE` mesh separation)
- The RAG database (`calculix_rag.db` at repo root) is the fallback when LLM knowledge is uncertain — query it for verified solver keywords and example patterns
- When encountering ambiguous engineering decisions (mesh density, rigid body mode suppression), surface 2+ options rather than assuming

<!-- MANUAL: -->
