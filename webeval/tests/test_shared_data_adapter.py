"""Tests for ``webtailbench.shared_data_adapter.create_datapoint``.

Builds a minimal fara ``Trajectory`` on disk (action log + fake
screenshots + FinalAnswer JSON) and verifies the DataPoint it produces
matches what ``MMRubricAgent._extract_input_from_datapoint`` expects.
"""

from __future__ import annotations

import json
from pathlib import Path


def _make_trajectory_dir(root: Path, n_actions: int = 2) -> Path:
    d = root / "sample_traj"
    d.mkdir(parents=True, exist_ok=True)

    # web_surfer.log: one action per line, fara's format (dict with action/arguments/url).
    log = d / "web_surfer.log"
    events = []
    for i in range(n_actions):
        events.append(
            {
                # ``gpt_solver=True`` in ``Trajectory`` filters on source==WebSurfer.
                "source": "WebSurfer",
                "action": "click" if i < n_actions - 1 else "stop_execution",
                "url": f"https://example.com/step_{i}",
                "arguments": {
                    "action": "click" if i < n_actions - 1 else "stop_execution",
                    "thoughts": f"thinking at step {i}",
                    "screen_description": f"screen at step {i}",
                    "target": f"button_{i}" if i < n_actions - 1 else None,
                },
            }
        )
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    # Fake screenshots on disk — content not validated by the adapter.
    screenshots = []
    for i in range(n_actions):
        p = d / f"screenshot_{i}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG signature, harmless for tests
        screenshots.append(p.name)

    # FinalAnswer JSON — follows fara's FinalAnswer schema.
    answer = {
        "final_answer": "done",
        "env_state_json": "{}",
        "env_state_raw": "",
        "screenshots": screenshots,
        "is_aborted": False,
        "is_rel_paths": True,
        "token_usage": {},
    }
    (d / "_answer.json").write_text(json.dumps(answer))
    return d


def test_create_datapoint_normalizes_screenshots_and_actions(tmp_path):
    from webeval.benchmarks.webtailbench.shared_data_adapter import create_datapoint
    from webeval.trajectory import Trajectory

    traj_dir = _make_trajectory_dir(tmp_path, n_actions=3)
    traj = Trajectory(traj_dir, gpt_solver=True)

    task_data = {
        "id": "sample_traj",
        "question": "pretend to click buttons",
        "init_url": "https://example.com",
    }
    dp = create_datapoint(task_data, traj)

    assert dp.task.task_id == "sample_traj"
    assert dp.task.instruction == "pretend to click buttons"
    assert dp.task.environment_config["init_url"] == "https://example.com"

    summaries = dp.solver_log.get_step_summaries()
    assert len(summaries) == 3
    # Last action must have been normalized stop_execution → terminate.
    assert summaries[-1].action_name == "terminate"
    # Screenshot indices must be 1-based and match action index.
    for i, s in enumerate(summaries, start=1):
        assert s.index == i
        assert s.screenshot_path.endswith(f"screenshot_{i}.png")


def test_create_datapoint_handles_missing_init_url(tmp_path):
    """Task_data without ``init_url`` should default to an empty string
    rather than crashing — WebTailBench TSV does not ship init_url."""
    from webeval.benchmarks.webtailbench.shared_data_adapter import create_datapoint
    from webeval.trajectory import Trajectory

    traj_dir = _make_trajectory_dir(tmp_path, n_actions=1)
    traj = Trajectory(traj_dir, gpt_solver=True)

    dp = create_datapoint({"id": "x", "question": "noop"}, traj)
    assert dp.task.environment_config["init_url"] == ""
