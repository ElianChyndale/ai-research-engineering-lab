# Reproduction Template

Forces the researcher to write BEFORE coding:
1. target paper claim
2. nearest owner
3. possible IPW moment
4. strongest baseline
5. reproduction success criterion
6. novelty kill condition

**Files:**
- `paper_claim.md` — the 6 pre-coding fields (start here).
- `baseline_map.md` — MUST/SHOULD/NOT-YET baselines with kill-claims.
- `config_pilot.yaml` — pre-registered config (seed ranges must not overlap).
- `run.py` — provenance seal + seed-disjointness check + common run interface.
- `analyse.py` — aggregation + kill-gate evaluation + KillTestReport.
- `report.md` — the filled-in result.

**Usage:** copy this directory per reproduction; fill `paper_claim.md` first;
then implement `run.py`'s `main()` and `analyse.py`'s `main()`.
