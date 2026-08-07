# C3 Owner Map — who already owns this problem

| Lineage | Owner | What it already covers | Collision |
|---|---|---|---|
| Scientific chart QA / reasoning | ChartQA, PlotQA, MathVista, SciGraphQA | reading charts; some unit awareness | **direct owner** |
| Unit/dimensional analysis in LLMs | various unit-conversion benchmarks | arithmetic unit conversion | medium |
| Visually-rich document understanding | DocVQA, M3DocRAG | layout/OCR; not typed measurement semantics | low-medium |
| Scientific figure consistency | prior "scientific figure forensics" work | inconsistency detection | medium-high |
| Typed/structured scientific reasoning | Neurosymbolic / measurement-graph work | explicit measurement graphs | high (conceptual) |

**IPW moment for C3:** A **chain-of-thought prompt** or a **simple
unit-compatibility constraint checker** may already fix the measurement-semantic
failures — if so, the "missing typed representation" is not needed. Test CoT +
constraint checker BEFORE claiming the mechanism.

**Strongest baseline:** strongest accessible frontier multimodal model on
originals + CoT prompt + a deterministic unit/axis consistency checker.
