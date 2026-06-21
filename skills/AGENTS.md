<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-04 | Updated: 2026-06-04 -->

# skills

## Purpose
Claude Code OMC skill packages for this project. Each subdirectory is a self-contained skill that can be loaded by the OMC plugin, providing AI agents with domain-specific decision trees, coding conventions, and RAG-backed knowledge for CalculiX and OpenRadiOSS workflows.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `whtools-calculixent/` | OMC skill for CalculiX FEA tasks — agent decision tree, INP conventions, RAG DB pointer (see `whtools-calculixent/AGENTS.md`) |
| `whtools-openradiossent/` | OMC skill for OpenRadiOSS explicit dynamics tasks (see `whtools-openradiossent/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Skills are loaded by the OMC plugin from `~/.claude/skills/` or the project `skills/` directory
- Each skill's entry point is its `SKILL.md` file
- Do not rename skill directories — the directory name is the skill identifier used in invocation

<!-- MANUAL: -->
