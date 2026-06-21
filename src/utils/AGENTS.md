<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-04 | Updated: 2026-06-04 -->

# src/utils

## Purpose
Utility scripts for building the RAG documentation indexes and performing auxiliary format conversions. Not part of the main analysis pipeline — run manually when documentation sources change or new conversion formats are needed.

## Key Files

| File | Description |
|------|-------------|
| `build_calculix_index.py` | `LightweightCodeRAGBuilder` — scans CalculiX source/doc files in `wht_calculixent_doc/`, extracts keyword-tagged chunks, and writes them to `calculix_rag.db` (SQLite) |
| `build_openradioss_index.py` | Same pattern for OpenRadiOSS source in `wht_openradiossent_doc/`, outputs `openradioss_rag.db` |
| `vtkhdf_converter.py` | Converts VTK/VTU output to VTK HDF5 format for large-model workflows |

## For AI Agents

### Working In This Directory
- Run the index builders from the repo root: `python src/utils/build_calculix_index.py`
- The resulting `.db` files are written to the repo root (not `workspace/`) because they are persistent indexes, not run artifacts
- RAG keywords are defined as module-level constants in each builder — extend them if new CalculiX physics domains are added

### Testing Requirements
- After rebuilding an index, verify it is non-empty: `sqlite3 calculix_rag.db "SELECT COUNT(*) FROM chunks;"`

## Dependencies

### Internal
- Reads from `wht_calculixent_doc/` and `wht_openradiossent_doc/` source trees
- Writes `calculix_rag.db` and `openradioss_rag.db` at repo root

### External
- `sqlite3` (stdlib), `numpy`, `re`, `os`

<!-- MANUAL: -->
