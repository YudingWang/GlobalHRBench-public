# GlobalHRBench

**Multilingual Employment-Law QA with Broad Model Evaluation and Deep Human Validation**

Yuding Wang · [Farlinko](https://farlinko.com)  
Yuhang Yao · Carnegie Mellon University

[Release v2026.11.15.1](https://github.com/YudingWang/GlobalHRBench-public/releases/tag/v2026.11.15.1) · [Read the paper](paper/GlobalHRBench.pdf) · [Dataset guide](docs/DATA.md) · [Evaluation and reproduction](docs/REPRODUCIBILITY.md) · [LaTeX source](paper/source/)

GlobalHRBench evaluates whether models answer employment-law questions correctly, preserve the conditions of the governing rule, and support their claims with the supplied evidence. It covers **501 questions across 8 jurisdictions and 6 languages**, with a **138-question reviewed core**.

Correctness, rule completeness, and source fidelity reveal different failures. An answer can preserve the full required rule while adding claims that its evidence packet does not support. “Unsupported” here is relative to the supplied evidence, not a judgment that a claim is legally false.

## What is evaluated?

| Evaluation | Questions | Models | Comparison | Scoring |
|---|---:|---:|---|---|
| Broad retrieval | 60 canonical tasks | 14 | BM25 vs. jurisdiction-aware routing; 840 pairs | Automated target-answer matching |
| Broad evidence access | The same 60 tasks | 14 | Closed book, the same BM25 baseline, designated evidence; 2,520 outputs | Automated target-answer matching |
| Deep human validation | 138 reviewed questions | 4 | Retrieved vs. designated evidence; 1,104 outputs | Human correctness, rule completeness, source fidelity |

The two broad views use one **14-model × 60-task × 4-condition matrix (3,360 observations)**. Figure 4 compares BM25 with routing; Figure 5 compares closed-book, BM25 and designated evidence. Both use the same BM25 observations. The completion adds 840 previously missing closed-book/designated outputs and preserves all retained observations. Runs were collected at different times, so this is a matched task matrix rather than a single contemporaneous randomized experiment.

The human evaluation contains **2,208 independent initial review forms and 113 adjudications**. Broad and human evaluation use different question versions and scoring protocols; their scores should be read within each study.

## Start here

- **Read:** [current English preprint](paper/GlobalHRBench.pdf), dated 15 November 2026. This repository does not claim conference acceptance or an arXiv identifier.
- **Explore:** [501-question inventory](data/questions.jsonl), [reviewed core](data/reviewed_core.jsonl), [required-rule annotations](data/required_rules.jsonl), and [sources](data/sources.jsonl).
- **Inspect:** [human response text and requests](results/human/responses.jsonl), [anonymous initial reviews](results/human/initial_reviews.jsonl), [final labels](results/human/final_labels.jsonl), and [per-output scores](results/human/answer_items.jsonl).
- **Reproduce:** run the offline check below. It makes no model calls and needs no API key.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

Use Python 3.10 or 3.11. The check validates file hashes, recomputes human metrics and agreement, checks reported aggregate rows and paired intervals, and re-scores 1,920 available main-comparison predictions with the original automated scorers.

## Repository map

```text
paper/       Current PDF and self-contained LaTeX source, including vector figures
data/       Questions, required rules, evidence sets, and source records
results/     Broad comparisons and anonymous human-evaluation records
evaluation/ Frozen scoring and statistical functions
scripts/     Offline verification entry point
docs/       Data dictionary, metric definitions, reproduction scope, and rights
```

The public snapshot contains the research materials needed for the documented checks. Private development history, earlier submissions, original Office workbooks, reviewer identities, provider-account records, and credentials are kept outside this repository. Original answer strings and scientific labels are not rewritten for publication.

## Reproduction scope

All 1,104 human-evaluated answers and their initial and final annotations are included. The package verifies released human scores from annotations; it does not replace independent semantic review. The original twelve-model retrieval comparison retains item scores but not a complete set of original answer text. Its 720 pairs can be reaggregated; complete answer rescoring is not claimed. The two previously added models contribute 120 further retrieval pairs and have saved predictions. All 840 newly completed closed-book/designated outputs also have retained answers, requests and attempt records. The earlier seven-model evidence-access BM25 observations remain in `results/broad/historical_seven_model/`; they are not pooled into the common main baseline.

The paper also reports historical resource-wide and response-contract diagnostics. This compact public snapshot focuses on its main broad and human comparisons; those ancillary runs are not reconstructed by the quick check. See [the exact coverage](docs/REPRODUCIBILITY.md).

## Citation

```bibtex
@misc{wangyao2026globalhrbench,
  title = {GlobalHRBench: Multilingual Employment-Law QA with Broad Model Evaluation and Deep Human Validation},
  author = {Wang, Yuding and Yao, Yuhang},
  year = {2026},
  howpublished = {Preprint}
}
```

## Use and licensing

The code is MIT licensed. Third-party legal text, translations, and model outputs are not granted new rights by the code license. See [rights and privacy](docs/RIGHTS_AND_PRIVACY.md) for source attribution and the release boundary. The benchmark is a research resource, not legal advice or a certification of current law.
