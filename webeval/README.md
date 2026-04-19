# webeval

Evaluation framework for [Fara-7B](https://huggingface.co/microsoft/Fara-7B)
on agentic web benchmarks (WebVoyager, OnlineMind2Web, WebTailBench).

For installation + the full reproducibility CLI, see the repo-root
[`README.md`](../README.md). Activate the **`fara_webeval`** conda env
before running anything below.

This document focuses on a single internal contract that every
benchmark, system, and verifier reads or writes: the **trajectory** —
i.e. the on-disk record of one agent run on one task.

## Trajectory format

Every executed task lives in its own directory under
`<out_url>/runs/<system.hash>/<model_prefix>/<user>/<benchmark.exec_hash>/<run_id>/traj/<task_id>/`,
e.g.:

```
/data/data/Fara/eval/runs/WebSurfer-fara-100-max_n_images-3/
  fara-7b/<user>/WebTailBench_hf/<run_id>/traj/<task_id>/
    web_surfer.log                 # newline-delimited JSON action+observation log
    <task_id>_final_answer.json    # FinalAnswer dump (final_answer, screenshots, token_usage)
    screenshot0.png ... screenshotN.png
    times.json                     # {start, end, duration} of the solver run
    core.log                       # webeval.core per-task log (Execution / Evaluation lifecycle)
    scores/
      <benchmark.eval_hash>.json   # {score: float, gpt_response_text: str|json}
```

### `web_surfer.log` (one JSON object per line)

Written by the `LogHandler` in `webeval/utils.py`, attached to the
`WebSurferLogger` inside `WebSurferSystem.get_answer()`. The handler
runs at `DEBUG` level so the `FaraAgent.{generate_model_call,
execute_action}` methods (`src/fara/fara_agent.py`) — which emit at
`logger.debug(...)` — actually flow through. Each line is one of three
shapes:

| `type`              | Source                                         | Key fields |
|---------------------|------------------------------------------------|---|
| `OtherEvent`        | `FaraAgent` thought/action prints + observation| `message` (free text — `Thought #N: ...\nAction #N: executing tool '<tool>' with arguments {...}` or `Observation#N: ...`), `timestamp` |
| `WebSurferEvent`    | `FaraAgent.execute_action`                     | `source`, `message`, `url`, `action`, `arguments`, `timestamp` |
| `OrchestrationEvent` / `AgentEvent` | reserved for orchestrator-side messages | `source`, `message` |

`Trajectory.__init__` reads every line and either:

1. Uses each event's top-level `action` field directly (the
   `WebSurferEvent` shape), OR
2. If no event has an `action` field (the all-`OtherEvent` case), runs
   `parse_text_based_event` to extract `action` + `arguments` (with
   `thoughts`) from the `Thought #N: ... Action #N: executing tool …`
   text. This is how trajectories from older `WebSurferSystem` runs
   are recovered.

After parsing, `Trajectory.actions` is a list of JSON-encoded action
arg dicts (with `thoughts` stripped) and `Trajectory.thoughts` is the
parallel list of thought strings.

> **Gotcha**: if `web_surfer.log` is empty, `Trajectory.events == []`
> and `Trajectory.actions == []`. Downstream evaluators (e.g.
> `WebTailBenchBenchmark.evaluator`) treat that as
> `verifier_reasoning="No actions found in the trajectory"` and return
> score 0 **without** invoking MMRubricAgent. The fix is in
> `WebSurferSystem.get_answer()`, which now bumps the logger to `DEBUG`
> before the `FaraAgent` runs.

### `<task_id>_final_answer.json` (`FinalAnswer` dataclass)

Written by `WebSurferSystem.get_answer()` once the agent terminates or
hits `max_rounds`. Field reference (`webeval/trajectory.py:FinalAnswer`):

| Field            | Type                       | Notes |
|------------------|----------------------------|---|
| `final_answer`   | `str`                      | The agent's final response text. Defaults to `"<no_answer>"` — see the next section for what that means. |
| `env_state_json` | `str` (JSON)               | Parsed JSON dump of the final webpage `<pre>` element at `/finish` (synthetic envs only). |
| `env_state_raw`  | `str`                      | Raw text from that `<pre>` element (debugging / fallback if `env_state_json` parse failed). |
| `screenshots`    | `List[str]`                | Screenshot filenames captured during the run, **relative** to the trajectory dir if `is_rel_paths` is true (the default), else absolute. |
| `is_aborted`     | `bool`                     | True if the trajectory was killed mid-execution by an environment error. Aborted trajectories are typically retried up to `--max_error_task_retries` (see `webeval/core.py`). |
| `is_rel_paths`   | `bool`                     | Whether `screenshots` paths are relative to `trajectory.path`. New runs use `True`. |
| `token_usage`    | `Dict[str, RequestUsage]`  | Per-component LLM token usage (e.g. `{"FaraAgent": RequestUsage(prompt_tokens=…, completion_tokens=…)}`). Aggregated via `RequestUsage.__add__`. |

Persisted via `FinalAnswer.save(path)`; loaded via `FinalAnswer.load(path)`.
`Trajectory.__init__` calls `FinalAnswer.load(...)` on the single
`*_final_answer.json` it finds in the dir (raises `ValueError` if 0
or >1 are present), reattaches the loaded `FinalAnswer` as
`trajectory.answer`, and exposes the screenshot list directly as
`trajectory.screenshots`. `trajectory.is_aborted` is just a
pass-through to `trajectory.answer.is_aborted`.

#### `final_answer == "<no_answer>"` is the load-bearing flag

`FinalAnswer` is constructed with `final_answer = "<no_answer>"` as the
default, and `WebSurferSystem.get_answer()` only **overwrites it with
the agent's actual answer string when the agent calls `terminate(...)`
within `--max_rounds` steps**. Three trajectory outcomes therefore look
different on disk:

| Outcome                                    | `is_aborted` | `final_answer`        | Notes |
|--------------------------------------------|--------------|-----------------------|---|
| Agent called `terminate(...)` within budget| `False`      | the model's response  | Only this case has a real string to score. |
| Agent never called `terminate`, hit `--max_rounds` | `False` | `"<no_answer>"`       | Trajectory ran to completion-by-cutoff; downstream is free to score the final state but the model itself produced no answer. |
| Run errored mid-execution (browser crash, timeout, …) | `True`  | `"<no_answer>"`       | Will be retried up to `--max_error_task_retries` (`webeval/core.py:run_eval_single_example`); only the final attempt's `_final_answer.json` is kept. |

**Filtering "actually completed" runs**: anywhere you want trajectories
that finished within step budget AND produced a real model answer, the
canonical check is

```python
candidate.answer.final_answer != "<no_answer>" and not candidate.is_aborted
```

This is what `webeval/post_eval_analysis.py` uses when computing
"average score across non-aborted, non-step-budget-exceeded
trajectories", and it's also why `WebTailBenchBenchmark.evaluator`
short-circuits with `verifier_reasoning='The last action must be
"terminate"'` for any trajectory whose last logged action is not
`terminate` — those are guaranteed to have `final_answer ==
"<no_answer>"` and therefore nothing the rubric judge can grade.

`screenshot{i}.png` files keep accumulating regardless of how the
trajectory ended, so screenshot count alone is **not** a proxy for
"in budget" — always combine it with the `final_answer` check above.

### `times.json`

`{"start": <iso8601>, "end": <iso8601>, "duration": <float-seconds>}`,
written around the solver call in `webeval/core.py:run_single_task`.
`evaluate_single_example` reads `duration` to populate
`EvalResult.duration`.

### `scores/<benchmark.eval_hash>.json`

Written by `webeval/core.py:evaluate_single_example` after a successful
`Benchmark.evaluate_example()` call. Single writer, single shape across
all benchmarks (top-level `{score, gpt_response_text}`); the
benchmark-specific verdict lives inside `gpt_response_text`.

| Field               | Type             | Notes |
|---------------------|------------------|---|
| `score`             | `float` or `int` | Top-line success metric (0/1 for outcome-style judges; fraction for rubric-style judges). |
| `gpt_response_text` | `str` (often a JSON-encoded dict) | Free-form judge output. See per-benchmark table below. |

#### Where each benchmark's verdict lives

The score file is the **single source of truth for an evaluation
result** — neither `task_data.json` (does not exist for new fara-private
runs) nor `final_answer.json` get overwritten by the verifier.
Re-running the verifier (`--redo_eval` or `scripts/verify_trajectories.py`)
rewrites this file in place.

| Benchmark | Score file path under `traj/<task_id>/` | `gpt_response_text` payload |
|---|---|---|
| `webvoyager` | `scores/gpt_eval.json` | gpt-4o judge text — verdict is `"SUCCESS"` / `"NOT SUCCESS"` parsed back to `score ∈ {0.0, 1.0}` (`benchmarks/webvoyager/webvoyager.py`). |
| `om2w` | `scores/<eval_method>-<threshold>.json` (e.g. `WebJudge_Online_Mind2Web_eval-3.json`) | o4-mini judge text — verdict via `extract_predication(...)` from the OnlineMind2Web judge methods (`benchmarks/om2w/om2w.py`). |
| `webtailbench` | `scores/mmrubric_<rubric_threshold>-<max_imgs>-<keypt>.json` | JSON-encoded dict with `rubric_*` (per-criterion scoring), `outcome_*` (output_success / primary_intent / reasoning), `error_taxonomy.*` (first-point-of-failure, ambiguity / invalid-task verdicts). Top-level `score` is `int(outcome_success)`. |
| `verify_trajectories.py` (stand-alone) | same path + shape as `webtailbench` | same Universal Verifier output. |

### `core.log`

Per-task log lines from `webeval.core.<task_id>` — `[Execution …]
Start/Completed/Error`, `[Evaluation …] Start/Completed/Error`. Useful
when a task hangs or errors and the rest of the pipeline keeps moving.

## `Trajectory` consumers

```python
from webeval.trajectory import Trajectory
traj = Trajectory.from_folder("/path/to/<task_id>/")
# Returns None if the dir is missing the answer file or the log can't be parsed.
traj.events                # list[dict] of parsed action/observation events
traj.actions               # list[str]  of JSON-encoded action-arg dicts
traj.thoughts              # list[str]  parallel to .actions
traj.screenshots           # list[Path] absolute paths to screenshot files
traj.answer                # FinalAnswer
traj.is_aborted            # bool (delegates to .answer.is_aborted)
traj.latest_screenshot     # convenience: trajectory.path / "screenshot_scaled.png"
```

Three places consume it today:

1. **Benchmark verifiers** (`webeval/benchmarks/*/<bench>.py`): each
   benchmark's `evaluator(task_data, candidate)` receives a
   `Trajectory` and returns `(score, gpt_response_text)`. WebTailBench
   wraps `MMRubricAgent` (the Universal Verifier) for this; webvoyager
   / om2w call their own GPT judges.
2. **`webeval/scripts/verify_trajectories.py`**: stand-alone parallel
   runner that points `MMRubricAgent` at any directory of
   `Trajectory`-shaped folders and writes per-task
   `scores/mmrubric_<…>.json`. Useful for re-scoring an existing
   eval run with new judge parameters.
3. **`webeval/post_eval_analysis.py`**: aggregates
   `_final_answer.json` + `scores/<…>.json` + counts of
   `WebSurferEvent` lines in `web_surfer.log` for run-level reporting.

## Where each file is written

| File                              | Writer                                                                                    |
|-----------------------------------|-------------------------------------------------------------------------------------------|
| `web_surfer.log`                  | `LogHandler` attached in `WebSurferSystem.get_answer()` (capturing `FaraAgent` log calls) |
| `<task_id>_final_answer.json`     | `WebSurferSystem.get_answer()` → `FinalAnswer.save()`                                     |
| `screenshot{i}.png`               | `FaraAgent.execute_action` (browser screenshots; one per action by default)               |
| `times.json`                      | `webeval/core.py:run_single_task`                                                         |
| `core.log`                        | `_logger = logger.getChild(task_id)` in `webeval/core.py:run_eval_single_example`         |
| `scores/<benchmark.eval_hash>.json` | `webeval/core.py:evaluate_single_example`                                                 |
| `scores/mmrubric_<…>.json`        | `webeval/scripts/verify_trajectories.py`                                                  |
