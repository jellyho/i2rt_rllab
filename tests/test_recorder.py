"""Recorder data-collection (mock) + DAgger-source assembly + outcome sidecar tests."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from workstation.lerobot_recorder import outcomes as _outcomes
from workstation.lerobot_recorder.config import ACTION_DIM, CameraSpec, RecorderConfig
from workstation.lerobot_recorder.dataset_writer import dataset_dir
from workstation.lerobot_recorder.portal_bridge import PortalBridge
from workstation.lerobot_recorder.recorder import Recorder


class _FakeWriter(SimpleNamespace):
    """Stand-in writer that records what the recorder streams to it.

    The recorder no longer keeps an episode in memory -- frames go to the writer one at a
    time -- so a fake that cannot receive them tests the wrong path. `frames` is the episode
    in flight, exactly what `rec._episode` used to hold.
    """

    def __init__(self, **kw):
        super().__init__(
            num_episodes=0,
            total_episodes=0,
            outcome_totals={"success": 0, "fail": 0},
            queue_depth=0,
            low_disk=False,
            finalized=False,
            progress={"saving": False, "queued": 0},
            **kw,
        )
        self.frames = []
        self.tasks = []
        self.saved = []

    def supports_streaming(self):
        return True

    def stream_frame(self, frame, task=""):
        self.frames.append(frame)
        self.tasks.append(task)

    def end_episode(self, outcome, task):
        self.saved.append((list(self.frames), outcome, task))
        self.frames = []

    def abort_episode(self):
        self.frames = []


def _in_flight(rec):
    """The frames of the episode in progress, whichever path the recorder took."""
    writer = rec.writer
    return getattr(writer, "frames", None) if hasattr(writer, "frames") else rec._episode


def test_recorder_start_failure_releases_hardware(tmp_path, monkeypatch):
    cfg = RecorderConfig(repo_id="test/yam", root=str(tmp_path), mock=True)
    rec = Recorder(cfg)
    calls = []
    monkeypatch.setattr(rec.cameras, "start", lambda: calls.append("cameras.start"))
    monkeypatch.setattr(rec.cameras, "stop", lambda: calls.append("cameras.stop"))
    monkeypatch.setattr(rec.robot, "start", lambda: calls.append("robot.start"))
    monkeypatch.setattr(rec.robot, "stop", lambda: calls.append("robot.stop"))

    def fail_writer():
        raise RuntimeError("dataset initialization failed")

    monkeypatch.setattr(rec, "_open_writer", fail_writer)
    try:
        rec.start()
    except RuntimeError as exc:
        assert str(exc) == "dataset initialization failed"
    else:
        raise AssertionError("startup should fail")

    assert calls == ["cameras.start", "robot.start", "robot.stop", "cameras.stop"]
    assert rec.writer is None


def test_recorder_records_episode_and_outcome(tmp_path):
    cfg = RecorderConfig(repo_id="test/yam", root=str(tmp_path), fps=60, mock=True)
    rec = Recorder(cfg)
    rec.start()
    rec.arm()

    captured, seen = False, set()
    t0 = time.time()
    while time.time() - t0 < 10:
        st = rec.get_status()
        seen.add(st["teleop"])
        if st["pending"]:
            captured = True
            rec.keep_episode(outcome="success")  # submits to the async writer queue
            break
        time.sleep(0.05)
    rec.shutdown()  # drains the queue + finalizes

    assert captured, "gate never produced a pending episode"
    assert "ENGAGED" in seen and "IDLE" in seen
    assert rec.writer.num_episodes >= 1  # worker saved it off the queue
    final = rec.get_status()
    assert final["kept"] >= 1 and final["success"] >= 1  # live stats counted the keep
    assert final["robot_ok"] is True  # mock bridge reports connected

    # the dataset lives at <root>/<name>; the verdict went into it with the frames
    assert dataset_dir(str(tmp_path), "test/yam") == str(tmp_path / "yam")
    saved = rec.writer.saved_episodes
    assert saved[0]["outcome"] == "success"
    assert saved[0]["episode"] == 0


def test_eval_records_at_the_loop_rate_once_the_rollout_has_started(tmp_path):
    """Eval samples at the record rate, like teleop -- NOT once per action the runner sends.

    It used to be send-driven, and that made the recording's time axis a fiction: no action is
    sent while the policy infers, so the hold left no frames while LeRobot writes
    timestamp = frame_index / fps regardless. What is preserved is the START: nothing is recorded
    until the first action, because the ticks before it are a stationary arm waiting on a JAX
    compile.
    """
    cfg = RecorderConfig(
        repo_id="test/eval", root=str(tmp_path), fps=60, mock=True, record_source="eval", review_before_save=False
    )
    rec = Recorder(cfg)
    rec.start()
    rec.arm()
    time.sleep(0.3)
    assert rec.get_status()["frames"] == 0  # armed, but nothing commanded yet -> no frames

    rec.note_action_sent(np.zeros(ACTION_DIM, dtype=np.float32))  # one chunk, then hold
    time.sleep(0.4)
    frames = rec.get_status()["frames"]
    rec.disarm()  # eval: ends the rollout and submits it
    rec.shutdown()
    # One send, many ticks: the hold is recorded rather than vanishing, which is the whole change.
    assert frames > 1, f"the hold after a single send should still be recorded, got {frames}"
    assert rec.writer.num_episodes >= 1
    assert rec.writer.saved_episodes[0]["frames"] == frames


def test_note_action_sent_is_a_noop_until_armed_in_eval(tmp_path):
    # Sends that arrive before arming (or in a non-eval source) must not leak into any episode.
    cfg = RecorderConfig(
        repo_id="test/eval_idle", root=str(tmp_path), fps=60, mock=True, record_source="eval", review_before_save=False
    )
    rec = Recorder(cfg)
    rec.start()
    try:
        rec.note_action_sent(np.zeros(ACTION_DIM, dtype=np.float32))  # not armed -> dropped
        time.sleep(0.2)
        assert rec.get_status()["frames"] == 0
    finally:
        rec.shutdown()


def test_manual_save_finalizes_and_next_episode_reopens_writer(tmp_path):
    cfg = RecorderConfig(repo_id="test/manual", root=str(tmp_path), fps=60, mock=True, review_before_save=False)
    rec = Recorder(cfg)
    rec.start()
    try:
        rec._episode = [rec._sample_frame()]
        rec._submit("success")
        rec.save_dataset()
        first_writer = rec.writer
        assert first_writer.finalized
        assert first_writer.num_episodes == 1

        rec._episode = [rec._sample_frame()]
        rec._submit("fail")
        assert rec.writer is not first_writer
        rec.save_dataset()
    finally:
        rec.shutdown()

    assert [row["episode"] for row in first_writer.saved_episodes] == [0]
    assert [row["outcome"] for row in first_writer.saved_episodes] == ["success"]
    assert [row["episode"] for row in rec.writer.saved_episodes] == [1]
    assert [row["outcome"] for row in rec.writer.saved_episodes] == ["fail"]


def test_manual_save_preserves_armed_idle_state(tmp_path):
    cfg = RecorderConfig(repo_id="test/armed_save", root=str(tmp_path), fps=60, mock=True, review_before_save=False)
    rec = Recorder(cfg)
    rec.writer = rec._open_writer()
    rec.arm()
    try:
        rec._episode = [rec._sample_frame()]
        rec._submit("success")
        rec.save_dataset()
        st = rec.get_status()
        assert rec.gate.armed is True
        assert st["armed"] is True
        assert st["recording"] is False
    finally:
        rec.shutdown()


def _real_dataset_cfg(**kw) -> RecorderConfig:
    """Mock hardware, real LeRobot writer: what a cross-session count has to be read back from.
    One tiny camera keeps the AV1 encode of a one-frame episode to a fraction of a second
    (and above the 16x16 floor the SVT encoder crashes on)."""
    return RecorderConfig(
        fps=60,
        mock=True,
        mock_writer=False,
        review_before_save=False,
        cameras=[CameraSpec("agentview", serial="", width=32, height=32, fps=60)],
        **kw,
    )


def test_status_reports_dataset_total_across_sessions(tmp_path):
    # session 1: fresh dataset — total grows with the saves
    cfg = _real_dataset_cfg(repo_id="test/total", root=str(tmp_path))
    rec = Recorder(cfg)
    rec.start()
    try:
        rec._episode = [rec._sample_frame()]
        rec._submit("success")
        rec._episode = [rec._sample_frame()]
        rec._submit("success")
        rec.save_dataset()
        assert rec.writer.total_episodes == 2
        assert rec.writer.new_episodes == 2
        assert rec.get_status()["episodes_total"] == 2
    finally:
        rec.shutdown()

    # session 2: resume — the dashboard total starts at the EXISTING count, not 0
    cfg2 = _real_dataset_cfg(repo_id="test/total", root=str(tmp_path), resume=True)
    rec2 = Recorder(cfg2)
    rec2.start()
    try:
        assert rec2.writer.total_episodes == 2  # before any new episode this session
        assert rec2.writer.new_episodes == 0
        rec2._episode = [rec2._sample_frame()]
        rec2._submit("fail")
        rec2.save_dataset()
        assert rec2.writer.total_episodes == 3
        assert rec2.writer.new_episodes == 1
        assert rec2.get_status()["episodes_total"] == 3
    finally:
        rec2.shutdown()


def test_status_reports_dataset_outcome_totals_across_sessions(tmp_path):
    # session 1: one success + one fail
    cfg = _real_dataset_cfg(repo_id="test/outcomes", root=str(tmp_path))
    rec = Recorder(cfg)
    rec.start()
    try:
        rec._episode = [rec._sample_frame()]
        rec._submit("success")
        rec._episode = [rec._sample_frame()]
        rec._submit("fail")
        rec.save_dataset()
        assert rec.writer.outcome_totals == {"success": 1, "fail": 1}
        st = rec.get_status()
        assert st["success_total"] == 1 and st["fail_total"] == 1
    finally:
        rec.shutdown()

    # session 2: resume — totals seed from the verdicts the dataset itself carries, then grow
    cfg2 = _real_dataset_cfg(repo_id="test/outcomes", root=str(tmp_path), resume=True)
    rec2 = Recorder(cfg2)
    rec2.start()
    try:
        assert rec2.writer.outcome_totals == {"success": 1, "fail": 1}  # before recording anything
        rec2._episode = [rec2._sample_frame()]
        rec2._submit("success")
        rec2.save_dataset()
        assert rec2.writer.outcome_totals == {"success": 2, "fail": 1}
        st = rec2.get_status()
        assert st["success_total"] == 2 and st["fail_total"] == 1
    finally:
        rec2.shutdown()
    # ...and the verdicts are in the dataset's own schema, readable without either session
    assert _outcomes.episode_outcomes(str(tmp_path / "outcomes")) == {0: "success", 1: "fail", 2: "success"}


def test_streaming_encoding_reaches_the_writer_without_being_asked_for(tmp_path):
    """Both the create and the resume path stream, with nothing set in the config."""
    from workstation.lerobot_recorder.dataset_writer import AsyncDatasetWriter

    cfg = RecorderConfig(repo_id="t/stream", root=str(tmp_path), mock=True)
    w = AsyncDatasetWriter(cfg, [], {})
    assert w._create_encoding_kwargs().get("streaming_encoding") is True
    assert w._dataset_encoding_kwargs().get("streaming_encoding") is True  # resume path too


def test_control_mode_in_frame():
    cfg = RecorderConfig(record_source="teleop", mock=False)
    rec = Recorder(cfg)
    snap = {
        "state": np.zeros(42, np.float32),
        "action": np.zeros(14, np.float32),
        "leader": np.zeros(12, np.float32),
        "eef": np.zeros(14, np.float32),
        "control_mode": 2,
    }
    frame = rec._frame({"agentview": np.zeros((4, 4, 3), np.uint8)}, snap)
    assert frame["observation.control_mode"].tolist() == [2.0]
    assert frame["observation.state"].shape == (42,)
    assert frame["observation.leader"].shape == (12,)
    assert frame["observation.eef"].shape == (14,)  # zeros if the robot can't FK
    assert frame["action"].shape == (14,)
    assert "agentview" in frame["images"]


def test_recenter_pauses_appends_without_closing_episode():
    cfg = RecorderConfig(record_source="teleop", mock=True)
    rec = Recorder(cfg)
    rec.writer = _FakeWriter()
    rec.gate.arm()
    images = {"agentview": np.zeros((4, 4, 3), np.uint8)}
    snap = {
        "teleop_state": "ENGAGED",
        "state": np.zeros(42, np.float32),
        "action": np.zeros(14, np.float32),
        "leader": np.zeros(12, np.float32),
        "eef": np.zeros(14, np.float32),
        "control_mode": 0,
        "buttons": {},
        "leader_recentering": False,
    }

    rec._step(images, snap)
    assert len(_in_flight(rec)) == 1
    assert rec.gate.recording is True

    snap["leader_recentering"] = True
    rec._step(images, snap)
    rec._step(images, snap)
    assert len(_in_flight(rec)) == 1  # no frame, state, or action => no dataset time
    assert rec.gate.recording is True  # same episode remains open internally
    assert rec.get_status()["recording"] is False

    snap["leader_recentering"] = False
    rec._step(images, snap)
    assert len(_in_flight(rec)) == 2
    assert rec.get_status()["recording"] is True


def test_dagger_source_assembly():
    cfg = RecorderConfig(record_source="dagger", mock=False)
    bridge = PortalBridge(cfg)
    human_l, human_r = np.arange(7, dtype=float), np.arange(7, 14, dtype=float)
    applied_l, applied_r = human_l + 20, human_r + 20
    pose = {"pos": [0.0] * 7, "vel": [0.0] * 7, "eff": [0.0] * 7}

    intervening = {
        "intervention": True,
        "policy_running": True,
        "left": {**pose, "human": human_l.tolist(), "applied": applied_l.tolist()},
        "right": {**pose, "human": human_r.tolist(), "applied": applied_r.tolist()},
        "t": 1.0,
    }
    snap = bridge._assemble(intervening)
    assert snap["teleop_state"] == "ENGAGED"
    assert snap["action"] is not None and snap["action"].shape == (14,)
    assert np.allclose(snap["action"], np.concatenate([applied_l, applied_r]))
    assert snap["control_mode"] == 2

    policy = {
        "intervention": False,
        "policy_running": True,
        "left": {**pose, "applied": human_l.tolist()},
        "right": {**pose, "applied": human_r.tolist()},
        "t": 2.0,
    }
    snap_policy = bridge._assemble(policy)
    assert snap_policy["teleop_state"] == "ENGAGED"
    assert np.allclose(snap_policy["action"], np.concatenate([human_l, human_r]))
    assert snap_policy["control_mode"] == 1

    stopped = {"intervention": False, "policy_running": False, "left": pose, "right": pose, "t": 3.0}
    snap_stopped = bridge._assemble(stopped)
    assert snap_stopped["teleop_state"] == "IDLE"
    assert snap_stopped["action"] is None


def test_portal_bridge_queues_policy_action_for_its_client_thread():
    bridge = PortalBridge(RecorderConfig(record_source="dagger", mock=False))
    action = {"left": np.arange(7), "right": np.arange(7, 14)}

    bridge.set_policy_action(action)
    action["left"][0] = 99

    assert bridge._policy_action_seq == 1
    assert bridge._policy_action_req is not None
    assert bridge._policy_action_req["left"][0] == 0

    bridge.set_policy_running(False)
    assert bridge._policy_action_req is None


def test_portal_bridge_reissues_ui_request_after_handle_changed_robot_state():
    bridge = PortalBridge(RecorderConfig(record_source="dagger", mock=False))
    bridge._policy_running_req = True
    bridge._policy_running_sent = True

    bridge.set_policy_running(True)

    assert bridge._policy_running_req is True
    assert bridge._policy_running_sent is None


def test_finish_clears_policy_action_and_latched_rollout_request():
    bridge = PortalBridge(RecorderConfig(record_source="dagger", mock=False))
    bridge.set_policy_running(True)
    bridge.set_policy_action({"left": np.zeros(7), "right": np.zeros(7)})

    bridge.finish_dagger_run("keep")

    assert bridge._finish_req == "keep"
    assert bridge._policy_running_req is False
    assert bridge._intervention_req is False
    assert bridge._policy_action_req is None


def test_dagger_records_one_rollout_across_interventions():
    cfg = RecorderConfig(record_source="dagger", mock=False)
    rec = Recorder(cfg)
    rec.writer = _FakeWriter()
    submitted = rec.writer.saved
    rec.gate.arm()
    images = {"agentview": np.zeros((4, 4, 3), np.uint8)}

    def snap(*, running=True, intervention=False, mode=1, event=None):
        return {
            "teleop_state": "ENGAGED" if running else "IDLE",
            "state": np.zeros(42, np.float32),
            "action": np.full(14, mode, np.float32) if running else None,
            "leader": np.zeros(12, np.float32),
            "eef": np.zeros(14, np.float32),
            "control_mode": mode,
            "buttons": {},
            "intervention": intervention,
            "leader_recentering": False,
            "last_dagger_event": event,
        }

    rec._step(images, snap())
    rec._step(images, snap(intervention=True, mode=2))
    rec._step(images, snap())
    assert rec.gate.recording is True
    assert len(_in_flight(rec)) == 3
    assert [int(f["observation.control_mode"][0]) for f in _in_flight(rec)] == [1, 2, 1]
    assert rec.get_status()["interventions"] == 1

    rec._step(images, snap(running=False, event={"seq": 1, "action": "keep"}))
    assert rec.gate.recording is False
    assert rec._pending is False
    assert rec._btn_outcome == "success"
    assert len(submitted) == 1
    assert len(submitted[0][0]) == 3
    # The label that reaches the writer, and so next.success. It used to be "keep", which
    # training reads as neither success nor fail -- success_only skipped every episode a DAgger
    # operator had kept.
    assert submitted[0][1] == "success"


def test_dagger_discard_drops_the_complete_rollout():
    cfg = RecorderConfig(record_source="dagger", mock=False, review_before_save=False)
    rec = Recorder(cfg)
    rec.writer = _FakeWriter()
    rec.gate.arm()
    images = {"agentview": np.zeros((4, 4, 3), np.uint8)}
    base = {
        "state": np.zeros(42, np.float32),
        "leader": np.zeros(12, np.float32),
        "eef": np.zeros(14, np.float32),
        "buttons": {},
        "intervention": False,
        "leader_recentering": False,
    }
    rec._step(images, {**base, "teleop_state": "ENGAGED", "action": np.zeros(14), "control_mode": 1})
    rec._step(
        images,
        {
            **base,
            "teleop_state": "IDLE",
            "action": None,
            "control_mode": 1,
            "last_dagger_event": {"seq": 1, "action": "discard"},
        },
    )
    assert _in_flight(rec) == []
    assert rec.get_status()["discarded"] == 1
    assert rec.get_status()["kept"] == 0


def test_dagger_snapshot_carries_state_and_event():
    cfg = RecorderConfig(record_source="dagger", mock=False)
    bridge = PortalBridge(cfg)
    pose = {"pos": [0.0] * 7, "vel": [0.0] * 7, "eff": [0.0] * 7}
    snap = bridge._assemble(
        {
            "intervention": False,
            "policy_running": True,
            "homing": False,
            "dagger_state": "policy",
            "last_dagger_event": {"seq": 3, "action": "keep"},
            "fine_grained": True,
            "leader_recentering": True,
            "recenter_fault": False,
            "left": pose,
            "right": pose,
            "t": 2.0,
        }
    )
    assert snap["policy_running"] is True
    assert snap["dagger_state"] == "policy"
    assert snap["last_dagger_event"] == {"seq": 3, "action": "keep"}
    assert snap["fine_grained"] is True
    assert snap["leader_recentering"] is True
    assert snap["recenter_fault"] is False


def test_dagger_recorder_events_do_not_use_expert_button_map():
    cfg = RecorderConfig(record_source="dagger", mock=False, button_map={"left.0": "discard"})
    rec = Recorder(cfg)
    rec.gate.arm()
    rec.gate.update("ENGAGED")
    rec._scan_buttons({"buttons": {"left": [1]}})
    assert rec._btn_outcome is None

    rec._scan_dagger_event({"last_dagger_event": {"seq": 1, "action": "keep"}})
    assert rec._btn_outcome == "success"
    rec._scan_dagger_event({"last_dagger_event": {"seq": 2, "action": "discard"}})
    assert rec._btn_outcome == "discard"


# --------------------------------------------------------------- deploy (no recording)
def test_deploy_source_opens_no_dataset(tmp_path):
    """`deploy` runs the policy but must not create a dataset anywhere on disk."""
    cfg = RecorderConfig(repo_id="test/deployonly", root=str(tmp_path), fps=60, mock=True, record_source="deploy")
    rec = Recorder(cfg)
    rec.start()
    time.sleep(0.5)  # let the loop run — in dagger/eval this would be buffering frames
    st = rec.get_status()
    rec.shutdown()

    assert rec.writer is None  # no writer was ever opened
    assert not (tmp_path / "deployonly").exists()  # and nothing was written to disk
    assert st["frames"] == 0 and st["recording"] is False and st["armed"] is False
    assert st["robot_ok"] is True  # the robot link is live all the same


def test_deploy_source_ignores_arm_and_save(tmp_path):
    """Arming/saving are meaningless without a dataset; they must be safe no-ops, not crashes."""
    cfg = RecorderConfig(repo_id="test/deploynoop", root=str(tmp_path), fps=60, mock=True, record_source="deploy")
    rec = Recorder(cfg)
    rec.start()
    rec.arm()
    time.sleep(0.4)
    armed = rec.get_status()["armed"]
    rec.save_dataset()  # would raise/AttributeError if it reached the writer
    rec.disarm()
    rec.shutdown()

    assert armed is False  # arm() did not arm anything
    assert rec.writer is None
    assert not (tmp_path / "deploynoop").exists()


def test_deploy_source_still_reports_rollout_state(tmp_path):
    """The UI needs policy/intervention state in deploy mode — that is the whole point."""
    cfg = RecorderConfig(repo_id="test/deploystate", root=str(tmp_path), fps=60, mock=True, record_source="deploy")
    rec = Recorder(cfg)
    snap = {
        "teleop_state": "IDLE",
        "state": np.zeros(4),
        "action": np.zeros(4),
        "policy_running": True,
        "intervention": True,
        "dagger_state": "intervention",
        "fine_grained": False,
        "leader_recentering": False,
        "recenter_fault": False,
        "homing": False,
        "estop": False,
        "buttons": {},
    }
    rec._step({}, snap)
    st = rec.get_status()
    assert st["policy_running"] is True
    assert st["intervention"] is True
    assert st["dagger_state"] == "intervention"
    assert st["frames"] == 0  # ...but still nothing buffered


def test_deploy_source_does_not_use_expert_button_map():
    """Handle buttons drive the robot's rollout state machine here, not outcome labels."""
    cfg = RecorderConfig(record_source="deploy", mock=False, button_map={"left.0": "discard"})
    rec = Recorder(cfg)
    rec._scan_buttons({"buttons": {"left": [1]}})
    assert rec._btn_outcome is None


# ------------------------------------------------------- robot-server mode mismatch
def test_wrong_robot_mode_refuses_to_start_with_an_actionable_message(tmp_path, monkeypatch):
    """A crossed robot server otherwise fails SILENTLY (a teleop server just ignores
    policy actions), so starting must refuse and name the command to run."""
    cfg = RecorderConfig(
        repo_id="test/mode", root=str(tmp_path), mock=False, record_source="deploy", expected_robot_mode="deploy"
    )
    rec = Recorder(cfg)
    monkeypatch.setattr(rec.cameras, "start", lambda: None)
    monkeypatch.setattr(rec.cameras, "stop", lambda: None)
    monkeypatch.setattr(rec.robot, "start", lambda: None)
    monkeypatch.setattr(rec.robot, "stop", lambda: None)
    monkeypatch.setattr(type(rec.robot), "robot_mode", property(lambda _self: "teleop"))

    try:
        rec.start()
    except RuntimeError as exc:
        assert "'teleop'" in str(exc) and "'deploy'" in str(exc)
        assert "robot/yam deploy" in str(exc)
    else:
        raise AssertionError("a teleop server must not satisfy a deployment session")


def test_matching_robot_mode_starts_normally(tmp_path, monkeypatch):
    cfg = RecorderConfig(
        repo_id="test/modeok", root=str(tmp_path), mock=False, record_source="deploy", expected_robot_mode="deploy"
    )
    rec = Recorder(cfg)
    monkeypatch.setattr(rec.cameras, "start", lambda: None)
    monkeypatch.setattr(rec.cameras, "stop", lambda: None)
    monkeypatch.setattr(rec.robot, "start", lambda: None)
    monkeypatch.setattr(rec.robot, "stop", lambda: None)
    monkeypatch.setattr(type(rec.robot), "robot_mode", property(lambda _self: "deploy"))
    rec.start()
    rec.shutdown()
    assert rec.writer is None  # deploy source: still no dataset


def test_unknown_robot_mode_does_not_block_startup(tmp_path, monkeypatch):
    """An older server that reports no mode must not become un-startable."""
    cfg = RecorderConfig(
        repo_id="test/modenone", root=str(tmp_path), mock=False, record_source="deploy", expected_robot_mode="deploy"
    )
    rec = Recorder(cfg)
    monkeypatch.setattr(rec.cameras, "start", lambda: None)
    monkeypatch.setattr(rec.cameras, "stop", lambda: None)
    monkeypatch.setattr(rec.robot, "start", lambda: None)
    monkeypatch.setattr(rec.robot, "stop", lambda: None)
    monkeypatch.setattr(type(rec.robot), "robot_mode", property(lambda _self: None))
    rec._check_robot_mode(timeout=0.1)  # warns, does not raise
    rec.shutdown()


def test_mock_sessions_skip_the_mode_check(tmp_path):
    cfg = RecorderConfig(
        repo_id="test/modemock", root=str(tmp_path), mock=True, record_source="deploy", expected_robot_mode="deploy"
    )
    rec = Recorder(cfg)
    rec.start()  # mock has no robot at all — the check must not fire
    rec.shutdown()


def test_older_robot_reporting_the_pre_rename_mode_is_accepted(tmp_path, monkeypatch):
    """The controller was renamed dagger -> deploy once it also served plain deployment.
    An un-updated robot server still says "dagger"; that skew must not read as a mismatch."""
    cfg = RecorderConfig(
        repo_id="test/modeold", root=str(tmp_path), mock=False, record_source="deploy", expected_robot_mode="deploy"
    )
    rec = Recorder(cfg)
    monkeypatch.setattr(rec.cameras, "start", lambda: None)
    monkeypatch.setattr(rec.cameras, "stop", lambda: None)
    monkeypatch.setattr(rec.robot, "start", lambda: None)
    monkeypatch.setattr(rec.robot, "stop", lambda: None)
    monkeypatch.setattr(type(rec.robot), "robot_mode", property(lambda _self: "dagger"))
    rec.start()  # must not raise
    rec.shutdown()


# --------------------------------------------------------------- memory guard
def test_low_memory_refuses_to_start_an_episode(tmp_path):
    """Better to not start than to buffer an episode there is no room for: a truncated
    take is data nobody asked for, and the OOM kill lands mid-write."""
    cfg = RecorderConfig(
        repo_id="test/ram",
        root=str(tmp_path),
        fps=60,
        mock=True,
        record_source="eval",
        review_before_save=False,
        min_free_ram_gb=1e9,
    )  # nothing will ever satisfy this
    rec = Recorder(cfg)
    rec.start()
    rec.arm()
    st = rec.get_status()
    rec.shutdown()

    assert st["low_ram"] is True
    assert st["armed"] is False, "arming must not proceed when the check fails"
    assert st["recording"] is False


def test_normal_memory_starts_as_usual(tmp_path):
    cfg = RecorderConfig(
        repo_id="test/ramok",
        root=str(tmp_path),
        fps=60,
        mock=True,
        record_source="eval",
        review_before_save=False,
        min_free_ram_gb=0.001,
    )
    rec = Recorder(cfg)
    rec.start()
    rec.arm()
    time.sleep(0.5)
    st = rec.get_status()
    rec.disarm()
    rec.shutdown()
    assert st["low_ram"] is False and st["armed"] is True


def test_guard_is_off_when_the_threshold_is_zero(tmp_path):
    cfg = RecorderConfig(
        repo_id="test/ramoff",
        root=str(tmp_path),
        fps=60,
        mock=True,
        record_source="eval",
        review_before_save=False,
        min_free_ram_gb=0.0,
    )
    rec = Recorder(cfg)
    rec.start()
    rec.arm()
    time.sleep(0.3)
    st = rec.get_status()
    rec.disarm()
    rec.shutdown()
    assert st["low_ram"] is False and st["armed"] is True


def test_it_is_checked_once_per_episode_not_per_frame(tmp_path, monkeypatch):
    """One read of /proc/meminfo an episode -- the record loop should not carry it."""
    calls = []
    monkeypatch.setattr(Recorder, "available_ram_gb", staticmethod(lambda: calls.append(1) or 999.0))
    cfg = RecorderConfig(
        repo_id="test/ramonce",
        root=str(tmp_path),
        fps=60,
        mock=True,
        record_source="eval",
        review_before_save=False,
        min_free_ram_gb=1.0,
    )
    rec = Recorder(cfg)
    rec.start()
    rec.arm()
    for i in range(30):  # eval is send-driven: drive some frames into the episode
        rec.note_action_sent(np.full(ACTION_DIM, float(i), dtype=np.float32))
        time.sleep(0.02)
    time.sleep(0.2)  # let the loop drain
    n_during = len(calls)
    rec.disarm()  # submits -> the after-episode report reads it once more
    rec.shutdown()
    assert n_during == 1, f"expected one check at arm, got {n_during}"
    assert len(calls) == 2, "expected exactly one more when the episode was handed over"


def test_available_ram_is_read_from_proc():
    """A plausible number, and inf rather than a crash where /proc/meminfo is absent."""
    free = Recorder.available_ram_gb()
    assert free > 0
    if free != float("inf"):
        assert free < 10_000


def test_no_episode_starts_while_the_first_inference_is_still_running():
    """End to end through the bridge: the compile stall must not become recorded frames."""
    from workstation.lerobot_recorder.portal_bridge import PortalBridge

    cfg = RecorderConfig(record_source="dagger", mock=False)
    bridge = PortalBridge(cfg)
    side = {"pos": np.zeros(7), "vel": np.zeros(7), "eff": np.zeros(7), "applied": np.zeros(7)}

    requested = bridge._assemble(
        {
            "left": side,
            "right": side,
            "policy_running": True,
            "policy_driving": False,  # asked for, not yet driving
        }
    )
    assert requested["teleop_state"] == "IDLE"

    driving = bridge._assemble(
        {
            "left": side,
            "right": side,
            "policy_running": True,
            "policy_driving": True,
        }
    )
    assert driving["teleop_state"] == "ENGAGED"


def test_an_older_robot_server_still_records():
    """A server that predates `policy_driving` must not silently stop producing episodes."""
    from workstation.lerobot_recorder.portal_bridge import PortalBridge

    cfg = RecorderConfig(record_source="dagger", mock=False)
    bridge = PortalBridge(cfg)
    side = {"pos": np.zeros(7), "vel": np.zeros(7), "eff": np.zeros(7), "applied": np.zeros(7)}
    old = bridge._assemble({"left": side, "right": side, "policy_running": True})
    assert old["teleop_state"] == "ENGAGED"


def test_the_recorder_streams_the_active_task_with_every_frame():
    """The task the operator set has to reach the writer, or the dataset's task column is blank."""
    cfg = RecorderConfig(record_source="dagger", mock=True, task="assemble lego blocks")
    rec = Recorder(cfg)
    rec.writer = _FakeWriter()
    rec.gate.arm()

    images = {"agentview": np.zeros((4, 4, 3), np.uint8)}
    snap = {
        "teleop_state": "ENGAGED",
        "state": np.zeros(42, np.float32),
        "action": np.zeros(14, np.float32),
        "leader": np.zeros(12, np.float32),
        "eef": np.zeros(14, np.float32),
        "control_mode": 1,
        "buttons": {},
        "intervention": False,
        "leader_recentering": False,
    }
    for _ in range(3):
        rec._step(images, snap)

    assert rec.writer.tasks == ["assemble lego blocks"] * 3


def test_a_homing_frame_is_never_captured():
    """Going home is not part of the rollout, gripper release included.

    The runner stops streaming when the robot reports homing, but the flag can flip between that
    check and the capture -- which put a single frame of the homing pose (arm travelling, gripper
    commanded shut) at the end of a recording."""
    cfg = RecorderConfig(repo_id="test/homing", root="/tmp/x", fps=30, mock=True, record_source="eval")
    rec = Recorder(cfg)
    rec.gate.arm()
    try:
        real = rec.robot.get_snapshot

        rec.robot.get_snapshot = lambda: {**real(), "homing": True}
        rec.note_action_sent(np.zeros(ACTION_DIM, dtype=np.float32))
        assert len(rec._sent_frames) == 0, "a homing frame must not be captured"

        rec.robot.get_snapshot = lambda: {**real(), "teleop_state": "HOMING"}
        rec.note_action_sent(np.zeros(ACTION_DIM, dtype=np.float32))
        assert len(rec._sent_frames) == 0, "teleop HOMING is the same thing by another name"
    finally:
        rec.robot.get_snapshot = real


def test_each_eval_rollout_is_its_own_episode(tmp_path):
    """An eval episode is one ROLLOUT, not one arming session.

    The operator arms once and runs several rollouts; between them the robot goes home, which is
    correctly not recorded (nothing is sent while homing). Concatenated, they became a single
    episode in which the arm teleports back to home mid-trajectory -- measured on a real recording
    as 7 rollouts inside one 9052-frame episode, with 6 jumps of 1.7-2.1 rad.

    The boundary comes from the runner (`on_rollout_end` -> `end_rollout`), which is the component
    that decides whether to send and therefore knows when the rollout ended. Polling the robot
    snapshot for `policy_running` would be a second, laggier source of the same fact.
    """
    cfg = RecorderConfig(
        repo_id="test/evalseg", root=str(tmp_path), fps=30, mock=True, record_source="eval", review_before_save=False
    )
    rec = Recorder(cfg)
    # Cameras and one cached frame, but no record loop: _step is driven by hand so the rollout
    # boundary is exercised deterministically instead of raced against the loop.
    rec.cameras.start()
    rec.robot.start()
    rec._last_images = rec.cameras.read()
    rec.writer = rec._open_writer()
    rec.gate.arm()
    try:
        for rollout in range(2):
            for i in range(4):
                rec.note_action_sent(np.full(ACTION_DIM, float(rollout * 10 + i), dtype=np.float32))
                rec._step(rec.get_last_images(), rec.robot.get_snapshot())
            assert rec.get_status()["frames"] == 4, "this rollout's own frames"

            rec.end_rollout()  # the runner: the policy stopped driving
            rec._step(rec.get_last_images(), rec.robot.get_snapshot())
            assert rec.get_status()["frames"] == 0, "a finished rollout must not carry into the next"
            assert rec.get_status()["kept"] == rollout + 1, "each rollout is handed over as it ends"
        assert rec.gate.armed is True, "the gate stays armed for the next rollout"
    finally:
        rec.shutdown()  # drains the async writer
    # ...and both really reached the writer, as two episodes rather than one.
    rows = rec.writer.saved_episodes
    assert [r["episode"] for r in rows] == [0, 1]
    assert all(r["frames"] == 4 for r in rows), rows


def test_end_rollout_is_ignored_when_there_is_no_rollout_to_end(tmp_path):
    """It arrives on every stop, including ones with nothing recorded (a rollout that never
    streamed, or a non-eval source), and must not open or close anything then."""
    cfg = RecorderConfig(repo_id="test/evalidle", root=str(tmp_path), fps=30, mock=True, record_source="eval")
    rec = Recorder(cfg)
    rec.end_rollout()  # not armed
    assert rec._rollout_ended is False
    rec.gate.arm()
    rec.end_rollout()
    assert rec._rollout_ended is True

    dag = Recorder(RecorderConfig(repo_id="test/dag", root=str(tmp_path), fps=30, mock=True, record_source="dagger"))
    dag.gate.arm()
    dag.end_rollout()
    assert dag._rollout_ended is False, "dagger has its own episode boundary"


def test_eval_takes_its_verdict_from_the_robot_not_the_teleop_button_map():
    """Two systems must not read the same press under different maps.

    In eval the policy drives, so the handle buttons belong to the robot's rollout state machine,
    which reports the verdict back through the dagger event. The recorder also used to scan them
    with the TELEOP map, where right.0 means discard -- while the robot now has right.0 as
    success_home. So a rollout ended with that button was closed as a success by the robot and
    thrown away by the recorder in the same tick, and the log said `leader button right.0 ->
    discard` while the operator was pressing what the screen called Success."""
    cfg = RecorderConfig(record_source="eval", mock=False)
    rec = Recorder(cfg)
    assert rec._button_outcome == {}, "eval must not carry the teleop outcome map"

    # ...and the verdict still arrives, from the robot's event
    rec._scan_dagger_event({"last_dagger_event": {"seq": 1, "action": "success"}})
    assert rec._btn_outcome == "success"

    rec._btn_outcome = None
    rec._scan_dagger_event({"last_dagger_event": {"seq": 2, "action": "fail"}})
    assert rec._btn_outcome == "fail"


def test_teleop_still_uses_the_button_map():
    """Only the policy-driven modes hand the buttons to the robot; a human-driven teleop episode
    is still labelled by the leader buttons."""
    rec = Recorder(RecorderConfig(record_source="teleop", mock=False))
    assert rec._button_outcome, "teleop keeps its outcome map"
    assert rec._button_outcome.get("right.0") == "discard", "the teleop map, unchanged"


# --------------------------------------------------------------- eval records the way teleop does
def test_eval_records_one_frame_per_tick_not_one_per_sent_action(tmp_path):
    """The recording's time axis has to be real, and the way to get that is to sample like teleop.

    Send-driven capture wrote one frame per action the runner pushed, so the stretch where the
    policy is inferring -- during which nothing is sent -- left NO frames, while LeRobot writes
    timestamp = frame_index / fps regardless. A 150 ms hold replayed as 0 ms and the motion tore.
    Here the loop ticks four times and only two sends land; four frames must be recorded, because
    the arm was there for all four.
    """
    cfg = RecorderConfig(
        repo_id="test/tick", root=str(tmp_path), fps=30, mock=True, record_source="eval", review_before_save=False
    )
    rec = Recorder(cfg)
    rec.cameras.start()
    rec.robot.start()
    rec._last_images = rec.cameras.read()
    rec.writer = rec._open_writer()
    rec.gate.arm()
    try:
        for tick in range(4):
            if tick in (0, 2):  # a replan on two of the four ticks
                rec.note_action_sent(np.zeros(ACTION_DIM, dtype=np.float32))
            rec._step(rec.get_last_images(), rec.robot.get_snapshot())
        assert rec.get_status()["frames"] == 4, "a tick with no send is still a moment the arm existed"
    finally:
        rec.shutdown()


def test_nothing_is_recorded_before_the_first_action_of_a_rollout(tmp_path):
    """The episode still starts at the first commanded action.

    The ticks before it are a stationary arm waiting on the first inference -- a JAX compile is
    tens of seconds -- and recording those with control_mode=policy is what teaches a policy to
    hold still. Only the holds BETWEEN chunks are new.
    """
    cfg = RecorderConfig(
        repo_id="test/pre", root=str(tmp_path), fps=30, mock=True, record_source="eval", review_before_save=False
    )
    rec = Recorder(cfg)
    rec.cameras.start()
    rec.robot.start()
    rec._last_images = rec.cameras.read()
    rec.writer = rec._open_writer()
    rec.gate.arm()
    try:
        for _ in range(3):  # armed, nothing sent yet
            rec._step(rec.get_last_images(), rec.robot.get_snapshot())
        assert rec.get_status()["frames"] == 0
        rec.note_action_sent(np.zeros(ACTION_DIM, dtype=np.float32))
        rec._step(rec.get_last_images(), rec.robot.get_snapshot())
        assert rec.get_status()["frames"] == 1
    finally:
        rec.shutdown()


def test_action_seq_marks_the_chunk_boundary(tmp_path):
    """Nothing is dropped, so the send stream has to be recoverable from the recording itself.

    `action_seq` is constant inside a chunk and increments at a replan, which is what tells a
    held frame from a fresh one -- the information send-driven capture used to carry by omitting
    the held frames entirely.
    """
    cfg = RecorderConfig(
        repo_id="test/seq", root=str(tmp_path), fps=30, mock=True, record_source="eval", review_before_save=False
    )
    rec = Recorder(cfg)
    rec.cameras.start()
    rec.robot.start()
    rec._last_images = rec.cameras.read()
    rec.writer = rec._open_writer()
    rec.gate.arm()
    try:
        sends = [True, False, False, True, False]
        for send in sends:
            if send:
                rec.note_action_sent(np.zeros(ACTION_DIM, dtype=np.float32))
            rec._step(rec.get_last_images(), rec.robot.get_snapshot())
        seq = [int(f["action_seq"][0]) for f in _in_flight(rec)]
        assert seq == [1, 1, 1, 2, 2], seq
    finally:
        rec.shutdown()


def test_action_seq_restarts_each_rollout(tmp_path):
    """Chunk numbering is per episode; rollout two must not continue rollout one's count."""
    cfg = RecorderConfig(
        repo_id="test/seq2", root=str(tmp_path), fps=30, mock=True, record_source="eval", review_before_save=False
    )
    rec = Recorder(cfg)
    rec.cameras.start()
    rec.robot.start()
    rec._last_images = rec.cameras.read()
    rec.writer = rec._open_writer()
    rec.gate.arm()
    try:
        for _ in range(2):
            rec.note_action_sent(np.zeros(ACTION_DIM, dtype=np.float32))
            rec._step(rec.get_last_images(), rec.robot.get_snapshot())
        rec.end_rollout()
        rec._step(rec.get_last_images(), rec.robot.get_snapshot())
        rec.note_action_sent(np.zeros(ACTION_DIM, dtype=np.float32))
        rec._step(rec.get_last_images(), rec.robot.get_snapshot())
        assert [int(f["action_seq"][0]) for f in _in_flight(rec)] == [1]
    finally:
        rec.shutdown()


def test_a_tick_that_records_nothing_is_counted_not_swallowed(tmp_path):
    """A dropped tick is a hole in a fixed-rate recording, so it has to be visible.

    cameras.healthy is all-or-nothing across cameras: one camera's hiccup drops the whole frame,
    and at a rollout's start the sensor warm-up can drop the first chunk outright -- which is what
    "the first one or two chunks did not save" looks like from the outside. The count and the
    reason go in the episode log.
    """
    cfg = RecorderConfig(
        repo_id="test/skip", root=str(tmp_path), fps=30, mock=True, record_source="eval", review_before_save=False
    )
    rec = Recorder(cfg)
    rec.cameras.start()
    rec.robot.start()
    rec._last_images = rec.cameras.read()
    rec.writer = rec._open_writer()
    rec.gate.arm()
    try:
        rec.note_action_sent(np.zeros(ACTION_DIM, dtype=np.float32))
        snap = rec.robot.get_snapshot()
        rec._step(rec.get_last_images(), snap)  # a good tick
        blind = dict(snap)
        blind["state"] = None  # the robot stopped reporting for one tick
        rec._step(rec.get_last_images(), blind)
        assert rec.get_status()["frames"] == 1, "the blind tick must not become a frame"
        assert rec._skipped["state"] == 1, rec._skipped
    finally:
        rec.shutdown()


def test_the_sample_frame_declares_every_key_a_real_frame_carries(tmp_path):
    """The sample frame IS the declared schema, so it has to match what _frame() emits.

    LeRobot rejects an add_frame whose keys differ from the schema in EITHER direction, and it
    rejects it per frame -- so a key added to _frame() and not to _sample_frame() does not fail at
    open, it fails on every save and stops the writer. That is exactly what `action_seq` did on a
    brand-new dataset. This compares the two directly so the next added column cannot repeat it.
    """
    cfg = RecorderConfig(repo_id="test/schema", root=str(tmp_path), fps=30, mock=True, record_source="eval")
    rec = Recorder(cfg)
    rec.cameras.start()
    rec.robot.start()
    try:
        sample = set(rec._sample_frame())
        real = set(rec._frame(rec.cameras.read(), rec.robot.get_snapshot()))
        assert real - sample == set(), f"_frame emits keys the schema does not declare: {real - sample}"
        assert sample - real == set(), f"the schema declares keys no frame carries: {sample - real}"
    finally:
        rec.shutdown()
