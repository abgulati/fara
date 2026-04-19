"""Build the CUAVerifierBench HuggingFace dataset locally.

Emits two HF *configs* — ``trajectories`` and ``annotations`` — each with
two splits: ``fara7b_om2w_browserbase`` and ``internal``. Tables are
joinable on ``task_id``:

* ``trajectories`` — one row per task (instruction, screenshots,
  web_surfer log, verifier outputs, task-level aggregates).
* ``annotations`` — one row per (task_id, annotator) human review.

Annotator names are anonymized to ``Judge1`` … ``JudgeN`` using a single
shared map across both splits, so the same human gets the same ID.

Run:
    python build_dataset.py [--out OUT_DIR] [--push REPO_ID]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_cuaverifierbench")

# --- om2w_browserbase split -------------------------------------------------
OM2W_ANN_TSV = Path(
    "/mnt/users/wangzhe/code/agento/webeval_next/verify_verifiers/"
    "trajectory_human_annotations/browserbase_om2w_v2/annotations_fixed.tsv"
)
OM2W_TRAJ_ROOT = Path("/data/data/Agento/eval/browserbase-om2w/traj")
OM2W_SCORE_FILE = "gpt_eval.json"

# --- internal split ---------------------------------------------------------
INTERNAL_ANN_TSV = Path(
    "/mnt/users/wangzhe/code/agento/webeval_next/verify_verifiers/"
    "trajectory_human_annotations/"
    "internal_combined_annotations.outcome_process.fixed.merged.tsv"
)
INTERNAL_TRAJ_ROOT = Path(
    "/data/data/Agento/eval/verify_verifiers/human_annotated_traj/"
    "internal_combined_traj/runs/"
    "WebSurfer-orca_qwen25vl_aurorav2_solver_history-100-max_n_images-3/"
    "models_dummy/corbyrosset/"
    "HoldOut__data_data_Agento_eval_verify_verifiers_human_annotated_traj_"
    "internal_combined_traj/"
    "gpt5.2v4_MMv20_gpt5.2_scored_soft_outcome_verifier/traj"
)
INTERNAL_SCORE_FILE = "0.8-5-3.json"

SCREENSHOT_RE = re.compile(r"^screenshot(\d+)\.png$")


# ---------------------------------------------------------------------------
# Trajectory loader
# ---------------------------------------------------------------------------
def _load_trajectory(traj_dir: Path, score_filename: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    td_path = traj_dir / "task_data.json"
    task_data = json.loads(td_path.read_text()) if td_path.exists() else {}
    out["instruction"] = task_data.get("task_proposal") or task_data.get("task_summary") or ""
    out["init_url"] = task_data.get("init_url") or ""
    out["start_timestamp"] = task_data.get("start_timestamp") or ""
    out["end_timestamp"] = task_data.get("end_timestamp") or ""

    fa_files = list(traj_dir.glob("*_final_answer.json"))
    if fa_files:
        fa = json.loads(fa_files[0].read_text())
        out["final_answer"] = fa.get("final_answer", "<no_answer>")
        out["is_aborted"] = bool(fa.get("is_aborted", False))
    else:
        out["final_answer"] = "<no_answer>"
        out["is_aborted"] = True

    wsl_path = traj_dir / "web_surfer.log"
    out["web_surfer_log"] = wsl_path.read_text(encoding="utf-8") if wsl_path.exists() else ""

    score_path = traj_dir / "scores" / score_filename
    out["gpt_eval_json"] = score_path.read_text(encoding="utf-8") if score_path.exists() else ""

    screenshots: List[tuple] = []
    for fname in traj_dir.iterdir():
        m = SCREENSHOT_RE.match(fname.name)
        if m:
            screenshots.append((int(m.group(1)), str(fname)))
    screenshots.sort()
    out["screenshots"] = [p for _, p in screenshots]
    out["n_screenshots"] = len(screenshots)
    return out


def _parse_uv_outcome_from_score_json(raw: str) -> Optional[int]:
    """Extract rubric_outcome_verification.output_success (bool) → int."""
    if not raw:
        return None
    try:
        outer = json.loads(raw)
        inner = json.loads(outer.get("gpt_response_text", "{}"))
        ov = inner.get("rubric_outcome_verification") or {}
        v = ov.get("output_success")
        if v is None:
            return None
        return int(bool(v))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Per-row helpers
# ---------------------------------------------------------------------------
def _f(d: Dict[str, Any], name: str) -> str:
    return (d.get(name) or "").strip()


def _floatish(d: Dict[str, Any], name: str) -> Optional[float]:
    v = _f(d, name)
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _binint(d: Dict[str, Any], name: str) -> Optional[int]:
    v = _f(d, name)
    if v == "":
        return None
    s = v.lower()
    if s in ("true", "1", "1.0", "yes", "correct"):
        return 1
    if s in ("false", "0", "0.0", "no", "incorrect"):
        return 0
    try:
        return int(float(v))
    except ValueError:
        return None


def _normalize_outcome_label(raw: str) -> str:
    """Map numeric ('1.0'/'0.0') and string variants to 'Correct'/'Incorrect'."""
    s = raw.strip().lower()
    if s in ("1", "1.0", "correct", "true", "yes"):
        return "Correct"
    if s in ("0", "0.0", "incorrect", "false", "no"):
        return "Incorrect"
    return raw  # leave anything else (e.g. 'partial') as-is


# ---------------------------------------------------------------------------
# Annotation TSV adapters: normalize each TSV's columns into a canonical dict
# ---------------------------------------------------------------------------
def _read_om2w_annotations(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    logger.info(f"loaded {len(rows)} om2w_browserbase annotation rows from {path.name}")
    return rows  # already in canonical om2w schema


def _read_internal_annotations(path: Path) -> List[Dict[str, Any]]:
    """Map internal TSV cols → same canonical keys used by the om2w loader.

    Internal TSV has no UV-informed stage and no continuous process score;
    those fields are emitted as empty strings so downstream coercion → None.
    """
    with open(path, encoding="utf-8") as f:
        # QUOTE_NONE: NOTES sometimes contains stray double-quotes
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    # Strip surrounding quotes from column names (QUOTE_NONE keeps them literal).
    rows = [{k.strip('"'): v for k, v in r.items()} for r in rows]
    out: List[Dict[str, Any]] = []
    for r in rows:
        # Some internal files include a header artifact row; skip if Annotator literally is "Annotator".
        if (r.get("Annotator") or "").strip() == "Annotator":
            continue
        if not (r.get("TASK") or "").strip():
            continue
        out.append({
            "TASK": r.get("TASK", ""),
            "VERIFIER": r.get("Annotator", ""),
            "HUMAN JUDGEMENT OUTCOME": _normalize_outcome_label(r.get("HUMAN JUDGEMENT OUTCOME", "")),
            "HUMAN JUDGEMENT PROCESS": _normalize_outcome_label(r.get("HUMAN JUDGEMENT PROCESS", "")),
            "HUMAN PROCESS SCORE": "",  # not in internal
            "OUTCOME_COMMENT": r.get("NOTES", ""),
            "PROCESS_COMMENT": "",
            "INFORMED_OUTCOME_AGREEMENT": "",
            "INFORMED_PROCESS_AGREEMENT": "",
            "INFORMED_OUTCOME_COMMENT": "",
            "INFORMED_PROCESS_COMMENT": "",
            "rubric_score": r.get("rubric_score", ""),
            "outcome_success": "",  # parsed from score JSON instead
            "mm_is_success": r.get('MM_verifier_response[""is_success""]', "")
                              or r.get('MM_verifier_response["is_success"]', ""),
            "verifier_is_success": r.get("Verifier_bool", ""),
        })
    logger.info(f"loaded {len(out)} internal annotation rows from {path.name}")
    return out


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
_TRAJ_FEATURES = Features({
    "task_id": Value("string"),
    "instruction": Value("string"),
    "init_url": Value("string"),
    "start_timestamp": Value("string"),
    "end_timestamp": Value("string"),
    "final_answer": Value("string"),
    "is_aborted": Value("bool"),
    "web_surfer_log": Value("string"),
    "screenshots": Sequence(Image()),
    "n_screenshots": Value("int32"),
    "gpt_eval_json": Value("string"),
    "uv_rubric_score": Value("float32"),
    "uv_outcome_success": Value("int32"),
    "mm_is_success": Value("int32"),
    "verifier_is_success": Value("int32"),
    "final_human_outcome_label": Value("int32"),
    "final_human_process_label": Value("int32"),
    "median_human_rubric_score_agnostic": Value("float32"),
    "majority_human_outcome_vote": Value("int32"),
})

_ANN_FEATURES = Features({
    "task_id": Value("string"),
    "annotator": Value("string"),
    "human_judgement_outcome": Value("string"),
    "human_judgement_process": Value("string"),
    "human_process_score": Value("float32"),
    "outcome_comment": Value("string"),
    "process_comment": Value("string"),
    "informed_outcome_agreement": Value("string"),
    "informed_process_agreement": Value("string"),
    "informed_outcome_comment": Value("string"),
    "informed_process_comment": Value("string"),
})


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------
def _majority(values: List[Optional[int]]) -> Optional[int]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return collections.Counter(vals).most_common(1)[0][0]


def _trajectory_row(
    task_id: str,
    traj: Dict[str, Any],
    anns: List[Dict[str, Any]],
    *,
    derive_aggregates: bool,
    uv_outcome_from_score_json: bool,
) -> Dict[str, Any]:
    head = anns[0]
    if uv_outcome_from_score_json:
        uv_out = _parse_uv_outcome_from_score_json(traj["gpt_eval_json"])
    else:
        uv_out = _binint(head, "outcome_success")

    if derive_aggregates:
        outcome_votes = [_binint(a, "HUMAN JUDGEMENT OUTCOME") for a in anns]
        process_votes = [_binint(a, "HUMAN JUDGEMENT PROCESS") for a in anns]
        final_outcome = _majority(outcome_votes)
        final_process = _majority(process_votes)
        median_score = None
        majority_outcome = final_outcome
    else:
        final_outcome = _binint(head, "final_human_outcome_label")
        final_process = _binint(head, "final_human_process_label")
        median_score = _floatish(head, "median_human_rubric_score_agnostic")
        majority_outcome = _binint(head, "majority_human_outcome_vote")

    return {
        "task_id": task_id,
        "instruction": traj["instruction"],
        "init_url": traj["init_url"],
        "start_timestamp": traj["start_timestamp"],
        "end_timestamp": traj["end_timestamp"],
        "final_answer": traj["final_answer"],
        "is_aborted": traj["is_aborted"],
        "web_surfer_log": traj["web_surfer_log"],
        "screenshots": traj["screenshots"],
        "n_screenshots": traj["n_screenshots"],
        "gpt_eval_json": traj["gpt_eval_json"],
        "uv_rubric_score": _floatish(head, "rubric_score"),
        "uv_outcome_success": uv_out,
        "mm_is_success": _binint(head, "mm_is_success"),
        "verifier_is_success": _binint(head, "verifier_is_success"),
        "final_human_outcome_label": final_outcome,
        "final_human_process_label": final_process,
        "median_human_rubric_score_agnostic": median_score,
        "majority_human_outcome_vote": majority_outcome,
    }


def _annotation_row(task_id: str, annotator: str, ann: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "annotator": annotator,
        "human_judgement_outcome": _f(ann, "HUMAN JUDGEMENT OUTCOME"),
        "human_judgement_process": _f(ann, "HUMAN JUDGEMENT PROCESS"),
        "human_process_score": _floatish(ann, "HUMAN PROCESS SCORE"),
        "outcome_comment": _f(ann, "OUTCOME_COMMENT"),
        "process_comment": _f(ann, "PROCESS_COMMENT"),
        "informed_outcome_agreement": _f(ann, "INFORMED_OUTCOME_AGREEMENT"),
        "informed_process_agreement": _f(ann, "INFORMED_PROCESS_AGREEMENT"),
        "informed_outcome_comment": _f(ann, "INFORMED_OUTCOME_COMMENT"),
        "informed_process_comment": _f(ann, "INFORMED_PROCESS_COMMENT"),
    }


# ---------------------------------------------------------------------------
# Per-split build
# ---------------------------------------------------------------------------
def _build_split(
    *,
    split_name: str,
    annotations: List[Dict[str, Any]],
    traj_root: Path,
    score_filename: str,
    alias_map: Dict[str, str],
    derive_aggregates: bool,
    uv_outcome_from_score_json: bool,
) -> Dict[str, Dataset]:
    annotated_task_ids = sorted({(a.get("TASK") or "").strip() for a in annotations if a.get("TASK")})
    have_traj = {tid for tid in annotated_task_ids if (traj_root / tid).is_dir()}
    missing = sorted(set(annotated_task_ids) - have_traj)
    if missing:
        logger.warning(f"[{split_name}] {len(missing)} annotated task_ids have no trajectory dir: {missing[:5]}…")
    logger.info(f"[{split_name}] task_ids w/ both annotation and trajectory: {len(have_traj)} / {len(annotated_task_ids)}")

    anns_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for ann in annotations:
        tid = (ann.get("TASK") or "").strip()
        if tid in have_traj:
            anns_by_task.setdefault(tid, []).append(ann)

    traj_rows: List[Dict[str, Any]] = []
    ann_rows: List[Dict[str, Any]] = []
    for tid in sorted(have_traj):
        traj = _load_trajectory(traj_root / tid, score_filename)
        traj_rows.append(_trajectory_row(
            tid, traj, anns_by_task[tid],
            derive_aggregates=derive_aggregates,
            uv_outcome_from_score_json=uv_outcome_from_score_json,
        ))
        for ann in anns_by_task[tid]:
            judge = alias_map.get(_f(ann, "VERIFIER"), _f(ann, "VERIFIER"))
            ann_rows.append(_annotation_row(tid, judge, ann))

    logger.info(f"[{split_name}] emitted {len(traj_rows)} trajectory rows, {len(ann_rows)} annotation rows")
    return {
        "trajectories": Dataset.from_list(traj_rows, features=_TRAJ_FEATURES),
        "annotations": Dataset.from_list(ann_rows, features=_ANN_FEATURES),
    }


def _build_alias_map(*ann_lists: List[Dict[str, Any]]) -> Dict[str, str]:
    """Single shared map across all splits: same human → same JudgeN."""
    raw = set()
    for anns in ann_lists:
        for a in anns:
            name = (a.get("VERIFIER") or "").strip()
            if name:
                raw.add(name)
    return {name: f"Judge{i + 1}" for i, name in enumerate(sorted(raw))}


def build_all() -> Dict[str, DatasetDict]:
    """Returns {config_name: DatasetDict({split_name: Dataset})}."""
    om2w_anns = _read_om2w_annotations(OM2W_ANN_TSV)
    internal_anns = _read_internal_annotations(INTERNAL_ANN_TSV)

    alias_map = _build_alias_map(om2w_anns, internal_anns)
    logger.info(f"anonymized {len(alias_map)} unique annotators → Judge1..Judge{len(alias_map)}")
    logger.info(f"  alias map: {alias_map}")

    om2w_split = _build_split(
        split_name="fara7b_om2w_browserbase",
        annotations=om2w_anns,
        traj_root=OM2W_TRAJ_ROOT,
        score_filename=OM2W_SCORE_FILE,
        alias_map=alias_map,
        derive_aggregates=False,                # om2w TSV has pre-computed aggregates
        uv_outcome_from_score_json=False,       # om2w TSV has outcome_success column
    )
    internal_split = _build_split(
        split_name="internal",
        annotations=internal_anns,
        traj_root=INTERNAL_TRAJ_ROOT,
        score_filename=INTERNAL_SCORE_FILE,
        alias_map=alias_map,
        derive_aggregates=True,                 # compute majority votes from per-judge labels
        uv_outcome_from_score_json=True,        # parse from rubric_outcome_verification.output_success
    )

    return {
        "trajectories": DatasetDict({
            "fara7b_om2w_browserbase": om2w_split["trajectories"],
            "internal": internal_split["trajectories"],
        }),
        "annotations": DatasetDict({
            "fara7b_om2w_browserbase": om2w_split["annotations"],
            "internal": internal_split["annotations"],
        }),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/cuaverifierbench_local", help="Local save_to_disk root")
    p.add_argument("--push", default=None, help="HF repo (e.g. microsoft/CUAVerifierBench)")
    args = p.parse_args()

    configs = build_all()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg_name, dd in configs.items():
        local = out_dir / cfg_name
        dd.save_to_disk(str(local))
        for split_name, ds in dd.items():
            logger.info(f"  config={cfg_name!r} split={split_name!r} rows={len(ds)} → {local}")

    if args.push:
        for cfg_name, dd in configs.items():
            logger.info(f"push_to_hub({args.push!r}, config={cfg_name!r}) …")
            dd.push_to_hub(args.push, config_name=cfg_name, private=True)
        logger.info(f"pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
