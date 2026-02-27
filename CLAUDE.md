# Project: assumption-mining

## Goal

Get the paper **"Assumption Mining: Extracting and Updating Implicit Requirements in LLM Code Generation"** accepted at **ASE 2026** (41st IEEE/ACM International Conference on Automated Software Engineering, Munich, October 12–16, 2026).

**Submission deadline**: April 15, 2026 (research track)

### Current state (as of 2026-02-27)
- Paper skeleton: ~60% complete — framework, methodology, related work, and threats are written; all result tables are empty (`\todo{}` placeholders)
- Scripts: written but not yet run (`scripts/`)
- Data: empty (`data/` has only `.gitkeep`)
- Figures: 6 figures described but not yet created

### Critical path to submission
1. **Phase 1 (Week 1, Feb 27–Mar 7)**: Run `collect_pilot.py` → `run_extractor.py` → `export_for_annotation.py`; recruit 2 annotators; begin pilot annotation
2. **Phase 2 (Week 2, Mar 8–14)**: Build AmbigBench benchmark (`select_benchmark_tasks.py`); implement 4 extraction baselines (DG, CE, CQ, CoT) and 2 dependency baselines (KH, FF); implement `run_rq3.py` (missing)
3. **Phase 3 (Weeks 3–4, Mar 15–28)**: Run RQ1, RQ2, RQ3 experiments (3 LLM backends: GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro); conduct RQ4 user study (20+ participants, within-subjects)
4. **Phase 4 (Week 5, Mar 29–Apr 6)**: Create all 6 figures; fill all result tables; write RQ summary answers
5. **Phase 5 (Week 6, Apr 7–14)**: Full revision, update abstract with numbers, fix ACM metadata to ASE 2026, submit

### Highest-risk items
- **IRB approval** for user study (RQ4) — check requirements immediately; can take 6–8 weeks
- **Annotator recruitment** — need 2 grad students with 2+ yr Python experience; start now
- **Circular self-consistency** (ASE reviewer concern) — GPT-4o evaluating itself in RQ1; mitigate with held-out tasks + Claude/Gemini ablations
- **`run_rq3.py` not yet implemented** — build in Week 2
- **No system demo** — a screenshot or short video of the UI (§4) strengthens the submission

### ASE-specific priorities
- Release benchmark + code at an anonymized URL (reviewers expect replication package)
- Keep within 10-page ACM SIGCONF limit (move overflow to supplemental)
- Make multi-model ablation results prominent to counter LLM self-evaluation concern

## TeX / LaTeX Setup

**System**: BasicTeX 2025 (TeX Live 2025basic) on macOS, binary path `/Library/TeX/texbin/`.

**User texmf tree**: `~/Library/texmf/tex/latex/` (user-mode packages installed here via `tlmgr --usermode install`)

### Compiling the paper

```bash
cd assumption-mining
/Library/TeX/texbin/pdflatex -interaction=nonstopmode main.tex
# Verify: "Output written on main.pdf"
open main.pdf
```

### Kernel/package compatibility fix (done 2026-02)

`tcolorbox 6.9.0` and `hyperref 7.01p` require a LaTeX kernel ≥ 2025-06.
BasicTeX 2025.0308 shipped with a 2024-11-01 kernel — **fixed by running in Terminal**:

```bash
sudo /Library/TeX/texbin/tlmgr update --self
sudo /Library/TeX/texbin/tlmgr update --all
sudo /Library/TeX/texbin/fmtutil-sys --byfmt pdflatex
```

After the kernel update, three additional missing deps of tcolorbox 6.9.0 were installed in user mode (no sudo):

```bash
/Library/TeX/texbin/tlmgr --usermode install tikzfill pdfcol listingsutf8
```

### Diagnosing missing packages

```bash
# Find which tlmgr package provides a missing .sty:
/Library/TeX/texbin/tlmgr search --global --file missing.sty

# Install in user mode (no sudo):
/Library/TeX/texbin/tlmgr --usermode install <package>

# Batch-compile to capture all errors at once:
/Library/TeX/texbin/pdflatex -interaction=batchmode main.tex
grep "File \`.*' not found" main.log | sed "s/.*File \`//;s/'.*$//" | sort -u
```

## Claude Code Skills

- Type `/` at the prompt to open the interactive skills/commands menu
- Filter by typing after `/` (e.g., `/latex` shows LaTeX-related skills)
- Personal skills: `~/.claude/skills/`; project skills: `.claude/skills/`
