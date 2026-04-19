---
language:
- en
license: mit
size_categories:
- n<1K
pretty_name: CUAVerifierBench
tags:
- cua
- agent-evaluation
- verifier
- arxiv:2604.06240
task_categories:
- image-text-to-text
configs:
- config_name: trajectories
  data_files:
  - split: fara7b_om2w_browserbase
    path: trajectories/fara7b_om2w_browserbase-*
  - split: internal
    path: trajectories/internal-*
- config_name: annotations
  data_files:
  - split: fara7b_om2w_browserbase
    path: annotations/fara7b_om2w_browserbase-*
  - split: internal
    path: annotations/internal-*
---

# CUAVerifierBench: A Human-Annotated Benchmark for Computer-Using-Agent Verifiers

[![Microsoft](https://img.shields.io/badge/Microsoft-Project-0078D4?logo=microsoft)](https://aka.ms/msaif/fara)
[![Hugging Face Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/microsoft/fara-7b)
[![Github](https://img.shields.io/badge/Github-181717?logo=github&logoColor=white)](https://github.com/microsoft/fara)

Universal Verifier paper: *The Art of Building Verifiers for Computer Use Agents*

## Dataset Summary

**CUAVerifierBench** is an evaluation benchmark for **verifiers** of computer-using agents (CUAs) — i.e. judges that read an agent's trajectory (screenshots + actions + final answer) and decide whether the task was completed correctly. Where benchmarks like WebTailBench measure *agents*, CUAVerifierBench measures the *judges that score those agents*.

Each row pairs a Fara-7B agent trajectory with one human reviewer's verdict, plus the verdicts produced by the **Universal Verifier (MMRubricAgent)** and several legacy verifiers. Researchers can use the dataset to:

- Compute verifier–human agreement (Cohen's κ, accuracy, F1) on a fixed corpus of trajectories
- Study disagreement between judges and how it changes when reviewers see the verifier's output (the "UV-informed" stage)
- Iterate on new verifier prompts/architectures against a frozen ground-truth set

## Splits

Both configs (`trajectories`, `annotations`) carry the same two splits:

| Split | Source | Trajectories | Annotation rows | Annotation stages |
|---|---|---|---|---|
| `fara7b_om2w_browserbase` | Fara-7B trajectories on the [Online-Mind2Web](https://huggingface.co/datasets/osunlp/Online-Mind2Web) tasks executed via the [Browserbase](https://www.browserbase.com/) remote browser | 106 | 215 (≈2 reviewers/task) | UV-blind **and** UV-informed |
| `internal` | Microsoft-internal task suite — heldout aurora-v2 task definitions evaluated with the same WebSurfer + verifier stack | 154 | 154 (1 reviewer/task) | UV-blind only |

The two splits share the same column schema. The `internal` split was annotated in a single UV-blind stage, so its `informed_*` fields and `human_process_score` are empty / null.

## Dataset Structure

The dataset is exposed as **two HuggingFace configs** that are joinable on `task_id`:

| Config | Granularity | Contents |
|---|---|---|
| `trajectories` | one row per task | The agent run — instruction, screenshots, web_surfer log, final answer, plus all verifier outputs and task-level human aggregates |
| `annotations` | one row per (task, judge) | Free-text and structured human judgments from one reviewer |

Reviewer identities are anonymized as `Judge1` … `JudgeN` using a single shared map across both splits — the same human always gets the same `Judge` ID.

Storing screenshots only in `trajectories` (rather than duplicating across judges) cuts the on-disk size roughly in half.

### Config: `trajectories`

| Field | Type | Description |
|---|---|---|
| `task_id` | string | **PK.** Online-Mind2Web task identifier (e.g. `Adidas--11857213`) |
| `instruction` | string | Natural-language task given to the agent |
| `init_url` | string | Starting URL |
| `start_timestamp`, `end_timestamp` | string | Wall-clock bounds of the run |
| `final_answer` | string | The agent's submitted answer (or `<no_answer>` if it never called `terminate`) |
| `is_aborted` | bool | Whether the run was aborted before completion |
| `web_surfer_log` | string | Full JSONL action/observation log from `web_surfer.log` |
| `screenshots` | sequence of `Image` | Inline PNG screenshots in chronological order, decoded to PIL automatically |
| `n_screenshots` | int32 | Length of the `screenshots` list |
| `gpt_eval_json` | string | Raw JSON of the original Online-Mind2Web GPT judge verdict |
| `uv_rubric_score` | float32 | **Universal Verifier (current)** rubric score in [0, 1] |
| `uv_outcome_success` | int32 | **Universal Verifier (current)** binary outcome verdict |
| `mm_is_success` | int32 | **Legacy (deprecated)** — verdict from the original WebTailBench multimodal grounded verifier (see note below) |
| `verifier_is_success` | int32 | **Legacy (deprecated)** — verdict from the original WebTailBench text-only task verifier (see note below) |
| `final_human_outcome_label` | int32 | Final adjudicated outcome label across all reviewers of this task |
| `final_human_process_label` | int32 | Final adjudicated process label across all reviewers of this task |
| `median_human_rubric_score_agnostic` | float32 | Median of UV-blind process scores across reviewers |
| `majority_human_outcome_vote` | int32 | Majority vote of UV-blind outcome labels |

> **About the legacy verifiers.** `mm_is_success` and `verifier_is_success` come from the **original WebTailBench verifier suite** used in the Fara-7B tech report (a 3-judge ensemble: text-only task verifier, multimodal grounded verifier, early rubric agent). The entire suite has since been **deprecated and replaced by the [Universal Verifier (MMRubricAgent)](https://github.com/microsoft/fara/blob/main/webeval/src/webeval/rubric_agent/mm_rubric_agent.py)** in [`microsoft/fara`](https://github.com/microsoft/fara). They are included only for backwards-compatible analysis against numbers from the original Fara-7B paper. **New work should use `uv_rubric_score` / `uv_outcome_success`.**

### Config: `annotations`

| Field | Type | Description |
|---|---|---|
| `task_id` | string | **FK** → `trajectories.task_id` |
| `annotator` | string | Anonymized reviewer (`Judge1` … `Judge6`) |
| `human_judgement_outcome` | string | UV-blind outcome label (`Correct` / `Incorrect` / etc.) |
| `human_judgement_process` | string | UV-blind process label |
| `human_process_score` | float32 | UV-blind continuous process score in [0, 1] |
| `outcome_comment` | string | UV-blind free-text justification for the outcome label |
| `process_comment` | string | UV-blind free-text justification for the process label |
| `informed_outcome_agreement` | string | UV-informed: agreement with the Universal Verifier's outcome verdict |
| `informed_process_agreement` | string | UV-informed: agreement with the Universal Verifier's process verdict |
| `informed_outcome_comment` | string | UV-informed free-text justification |
| `informed_process_comment` | string | UV-informed free-text justification |

**UV-blind vs. UV-informed.** Reviewers labeled each trajectory in two stages: first *without* seeing any verifier output (`human_*` and `*_comment` fields), then *after* being shown the Universal Verifier's verdict (`informed_*` fields).

## Loading

Each config is loaded separately and joined on `task_id`. Pass either `fara7b_om2w_browserbase` or `internal` as the split:

```python
from datasets import load_dataset

split = "fara7b_om2w_browserbase"  # or "internal"
trajs = load_dataset("microsoft/CUAVerifierBench", "trajectories", split=split)
anns  = load_dataset("microsoft/CUAVerifierBench", "annotations",  split=split)

# Per-judge analysis: join in pandas
import pandas as pd
df = anns.to_pandas().merge(trajs.to_pandas(), on="task_id")

# Or look up trajectories on demand:
by_id = {r["task_id"]: r for r in trajs}
print(by_id[anns[0]["task_id"]]["screenshots"][0])  # PIL.Image
```

## Dataset Creation

### Source trajectories

Trajectories were generated by running [Fara-7B](https://huggingface.co/microsoft/fara-7b) on the public [Online-Mind2Web](https://huggingface.co/datasets/osunlp/Online-Mind2Web) task set, executed inside a [Browserbase](https://www.browserbase.com/)-hosted Chromium instance. Each trajectory contains the screenshots the model saw, the structured actions it issued, and the final answer it submitted.

### Annotation protocol

Each task was independently reviewed by ~2 human annotators in two stages:

1. **UV-blind (agnostic)** — Reviewers read the instruction and trajectory and assign outcome / process labels and a continuous process score, *without* seeing any verifier output.
2. **UV-informed** — Reviewers are then shown the Universal Verifier's verdict and asked whether they agree, with free-text justifications.

Reviewer identities are anonymized as `Judge1`…`Judge6`.

### Universal Verifier outputs

For each trajectory we also include the verdicts of the **MMRubricAgent** (the Universal Verifier shipped with Fara) and two legacy verifiers, so users can directly compute verifier–human agreement.

## Considerations for Using the Data

### Intended Use

- Evaluating CUA verifiers against human judgment
- Studying inter-annotator agreement and the effect of showing model verdicts to humans
- Developing new judge prompts / architectures for trajectory evaluation

### Limitations

- 106 tasks is a relatively small corpus; results should be reported with confidence intervals
- All trajectories come from a single agent (Fara-7B); verifier behavior on trajectories from other agents may differ
- Tasks inherit the temporal validity and domain biases of Online-Mind2Web

### Licensing

MIT License

### Citation

If you use CUAVerifierBench in your research, please cite the Universal Verifier paper:

```bibtex
@article{UniversalVerifier2026,
  title={The Art of Building Verifiers for Computer Use Agents},
  journal={arXiv preprint arXiv:2604.06240},
  year={2026},
  url={https://arxiv.org/abs/2604.06240v1}
}
```

### Contributions

Created by Microsoft Research AI Frontiers.
