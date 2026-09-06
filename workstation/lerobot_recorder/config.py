"""Configuration for the LeRobot recorder (cameras, topics, dataset schema)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from i2rt.serving.control_config import DEFAULT_TELEOP_BUTTON_OUTCOMES

# ----------------------------------------------------------------------------
# Robot / dataset dimensions
# ----------------------------------------------------------------------------
ARMS = ("left", "right")
ARM_DOF = 7  # 6 joints + gripper per arm (follower)
LEADER_ARM_DOF = 6  # teaching-handle leader exposes 6 arm joints (gripper is a passive trigger)

# The recorded schema is FIXED from the robot's known outputs (no runtime probe).
# MotorChainRobot.get_observations() gives joint_pos/vel/eff (+ gripper_*), so:
#   observation.state  = per arm [pos(7), vel(7), eff(7)]  -> 42
#   observation.leader = per arm leader joint positions    -> 12
#   action             = per arm applied/human action [7]  -> 14
# (The robot exposes no end-effector pose, so observation.eef is not recorded.)
STATE_DIM = len(ARMS) * ARM_DOF * 3  # 42
ACTION_DIM = len(ARMS) * ARM_DOF  # 14
LEADER_DIM = len(ARMS) * LEADER_ARM_DOF  # 12
EEF_POSE_DIM = 7  # per arm: position(3) + quaternion wxyz(4)
EEF_DIM = len(ARMS) * EEF_POSE_DIM  # 14 (zeros when the robot can't provide FK)

# Per-frame control-mode label (always written as observation.control_mode), so a
# dataset records whether each frame came from teleop, a policy, an intervention,
# replay, or the automatic HOMING return — useful for HG-DAgger filtering and for
# dropping the homing tail at train time (filter control_mode == homing). The editor
# can also set frames to `homing` to trim the tail of already-collected episodes.
CONTROL_MODE = {"teleop": 0, "policy": 1, "intervention": 2, "replay": 3, "homing": 4}


def state_names() -> List[str]:
    names: List[str] = []
    for arm in ARMS:
        for field_ in ("pos", "vel", "eff"):
            names += [f"{arm}.{field_}.{i}" for i in range(ARM_DOF)]
    return names


def action_names() -> List[str]:
    return [f"{arm}.act.{i}" for arm in ARMS for i in range(ARM_DOF)]


def leader_names() -> List[str]:
    return [f"{arm}.leader.{i}" for arm in ARMS for i in range(LEADER_ARM_DOF)]


def eef_names() -> List[str]:
    comp = ["x", "y", "z", "qw", "qx", "qy", "qz"]
    return [f"{arm}.eef.{c}" for arm in ARMS for c in comp]


# ----------------------------------------------------------------------------
# Cameras (RealSense). Map each physical serial to a role/image key.
# Two D405 on the wrists + one D455 agentview.
# ----------------------------------------------------------------------------
@dataclass
class CameraSpec:
    key: str  # dataset image key, e.g. "wrist_left"
    serial: str  # RealSense serial number ("" = pick any remaining device)
    width: int = 640
    height: int = 480
    fps: int = 60  # native stream fps (>= record fps so frames don't repeat)
    # RealSense color-sensor options applied after the stream starts (and on every
    # reconnect), e.g. {"enable_auto_exposure": 0, "exposure": 300, "gain": 64} to
    # lock exposure so brightness doesn't drift across an episode. Names are
    # `rs.option` members; set from config.yaml's per-camera `options:` map.
    options: dict = field(default_factory=dict)


def default_cameras() -> List[CameraSpec]:
    # Fill in the real serials (run `python -m workstation.lerobot_recorder.cameras --list`).
    return [
        CameraSpec("wrist_left", serial="", width=640, height=480, fps=60),
        CameraSpec("wrist_right", serial="", width=640, height=480, fps=60),
        CameraSpec("agentview", serial="", width=640, height=480, fps=60),
    ]


# ----------------------------------------------------------------------------
# Top-level recorder config
# ----------------------------------------------------------------------------
@dataclass
class RecorderConfig:
    repo_id: str = "user/yam_bimanual"
    root: str = "~/lerobot_data"
    task: str = "do the task"  # ACTIVE language instruction (persists until changed)
    tasks: List[str] = field(default_factory=list)  # quick-switch templates shown in the GUI
    fps: int = 60  # dataset / record-loop rate (matched to the 60 fps cameras)
    # Output dataset format: "lerobot" (LeRobotDataset, default) or "abcdl" (the abcdl
    # MP4+binary training cache — one episode dir per save, loadable via AbcdlDataset /
    # pushable with AbcdlDataset.push_to_hub). Needs the `abcdl` package installed.
    record_format: str = "lerobot"
    abcdl_size: int = 224  # square px the cameras are resized to for the abcdl format
    # Per-frame RL signals computed from the episode outcome and stored as dataset
    # features (success / reward / mc_return) — for value/RL learners.
    rl_features: bool = False  # enable per-frame success/reward/mc_return
    reward_mode: str = "sparse"  # "sparse" (0/+1 at success) | "step" (-1/step, 0 at success)
    discount_factor: float = 0.99  # γ for the Monte-Carlo return-to-go
    # Video encoding speed (LeRobot). The default 'libsvtav1' (AV1) is SLOW to encode;
    # 'h264' is much faster, 'auto' picks a hardware encoder (e.g. nvenc) if available.
    vcodec: str = "libsvtav1"
    # Workstation encoder implementation. TorchCodec is preferred, with an automatic
    # PyAV fallback if GPU usage is already at/above the configured safety threshold.
    encoding_backend: str = "torchcodec"  # torchcodec | pyav
    torchcodec_max_used_vram_gb: float = 5.0
    encoder_threads: int = 0  # threads per video encode (0 = let LeRobot decide)
    # >1 accumulates that many episodes and encodes their videos in PARALLEL (LeRobot's
    # ProcessPoolExecutor) — the supported way to use more than one encoder "worker".
    batch_encoding_size: int = 1
    # Async image writer: LeRobot writes every frame as a temp PNG BEFORE encoding the
    # video; off by default = single-threaded and usually the slowest part of a save.
    # ~4 threads per camera parallelizes it (0 = synchronous; processes>0 adds subprocesses).
    image_writer_threads: int = 0
    image_writer_processes: int = 0
    # Keyframe interval. LeRobot hard-codes 2 in its streaming encoder -- a keyframe every
    # other frame -- which makes a random frame decode fastest and the file about 2.8x larger
    # than the same footage at 10. Measured on this rig: 0.90 ms per random frame at g=2
    # against 1.23 ms at g=10, a difference a dataloader worker hides, while the disk and the
    # upload do not hide 2.8x. 10 still leaves three keyframes a second at 30 fps.
    gop: int = 10
    # (There is no streaming_encoding knob. Frames always stream straight into the PyAV
    # encoder; the alternative round-trips every frame of every camera through a ~1.5 MB PNG
    # on disk, which is not a mode worth being one config key away from. See
    # AsyncDatasetWriter._dataset_encoding_kwargs.)
    robot_type: str = "yam_bimanual"
    cameras: List[CameraSpec] = field(default_factory=default_cameras)
    # Robot link: the YAM robot machine running i2rt.serving.run_robot_server (portal).
    robot_host: str = "127.0.0.1"
    robot_port: int = 11331
    # What drives episode gating + the recorded action:
    #   "teleop" -> gate on teleop_state (ENGAGED..IDLE), action = applied command
    #   "dagger" -> gate on the complete policy rollout, action = executed command;
    #               observation.control_mode marks policy vs human intervention
    #   "eval"   -> one continuous rollout from arm to disarm, action = executed command
    #   "deploy" -> run the policy WITHOUT recording: no dataset is created or opened and
    #               no frames are buffered. Everything else (cameras, robot link, live
    #               view, e-stop, human takeover) behaves exactly as in the other modes,
    #               so a rollout can be watched and interrupted without producing data.
    record_source: str = "teleop"
    # Which robot-server controller this tool needs ("teleop" / "dagger"); "" skips the
    # check. The robot modes are mutually exclusive and started separately on the robot
    # machine, so a mismatch otherwise fails *silently* — a teleop server simply ignores
    # policy actions, and a dagger server never reports the teleop engage gate.
    expected_robot_mode: str = ""
    resume: bool = False  # append to an existing dataset at `root` instead of creating a new one
    min_free_gb: float = 1.0  # refuse to save an episode when free disk drops below this
    # An episode is buffered whole in RAM before it is handed to the writer, and at
    # 640x480x3 on three cameras that is ~2.8 MB per frame -- roughly 7 GB for a 40 s
    # episode. Past this much free memory the recorder ENDS the episode and saves it,
    # because the alternative is the OOM killer taking the process mid-write and leaving
    # a dataset that will not open. 0 disables the check.
    min_free_ram_gb: float = 3.0
    # Eval only. False (the default) records a frame only where the rollout actually advanced:
    # a tick contributes nothing while the policy is inferring (no new action) or while the camera
    # has not produced a new image (the loop can outrun it). True keeps those, which makes the
    # video's elapsed time real at the cost of frames that repeat the previous one.
    record_holds: bool = False
    mock: bool = False  # synthetic cameras + teleop stream (no hardware / robot)
    # None -> the writer follows ``mock``. False with ``mock=True`` records the synthetic stream
    # into a real LeRobot dataset (a dry run of the writer with no hardware; what the tests use).
    mock_writer: Optional[bool] = None
    review_before_save: bool = True  # hold each episode for Keep/Delete instead of auto-saving
    auto_arm: bool = False  # arm collection automatically on Start (record on the next teleop engage)
    review_cam: str = "agentview"  # which camera to buffer (downsampled) for review playback
    # Leader-handle button -> episode outcome, keyed "<side>.<button_index>" (upper=0,
    # lower=1). Left upper is reserved for fine-grained control; right upper discards.
    button_map: dict = field(default_factory=lambda: dict(DEFAULT_TELEOP_BUTTON_OUTCOMES))
    # Where the run page's overlay lists past demonstrations from. Empty = the recording root.
    # They differ whenever rollouts are written somewhere other than where the demos were collected.
    reference_root: str = ""
    review_downscale: int = 4  # spatial stride for the review preview (640x480 -> 160x120)
    # Operator-preview blend when a past dataset episode is selected.  1.0 is
    # live cameras only; 0.0 is the reference episode only.  This never changes
    # recorded images or policy inputs.
    reference_live_alpha: float = 0.5


def apply_recorder_section(cfg: RecorderConfig, rec_section) -> RecorderConfig:
    """Copy the ``recorder:`` section of config.yaml onto ``cfg``, in place.

    Both entry points -- ``yam-data record`` and ``yam-data deploy`` -- build the same
    RecorderConfig from the same YAML section, and each used to read it field by field on its
    own. The two lists drifted: deploy read 12 keys where record read 18, so ``format``,
    ``rl_features``, ``reward_mode``, ``discount_factor``, ``record_format`` and ``abcdl_size``
    were silently ignored for a deployment that recorded a dataset. Nothing announced that --
    the dataset just came out shaped differently from a recorded one.

    Anything genuinely specific to one entry point (``expected_robot_mode``, the teleop button
    map) stays with that entry point; this is only the shared part.
    """
    g = rec_section.get
    cfg.record_format = str(g("format", g("record_format", cfg.record_format)))
    cfg.abcdl_size = int(g("abcdl_size", cfg.abcdl_size))
    # per-frame RL signals (success / reward / mc_return)
    cfg.rl_features = bool(g("rl_features", cfg.rl_features))
    cfg.reward_mode = str(g("reward_mode", cfg.reward_mode))
    cfg.discount_factor = float(g("discount_factor", cfg.discount_factor))
    # video-encoding knobs (saving speed)
    cfg.vcodec = str(g("vcodec", cfg.vcodec))
    cfg.encoding_backend = str(g("encoding_backend", cfg.encoding_backend))
    cfg.torchcodec_max_used_vram_gb = float(g("torchcodec_max_used_vram_gb", cfg.torchcodec_max_used_vram_gb))
    cfg.gop = max(1, int(g("gop", cfg.gop)))
    cfg.encoder_threads = int(g("encoder_threads", cfg.encoder_threads))
    cfg.batch_encoding_size = int(g("batch_encoding_size", cfg.batch_encoding_size))
    cfg.image_writer_threads = int(g("image_writer_threads", cfg.image_writer_threads))
    cfg.image_writer_processes = int(g("image_writer_processes", cfg.image_writer_processes))
    cfg.reference_live_alpha = min(max(float(g("reference_live_alpha", cfg.reference_live_alpha)), 0.0), 1.0)
    return cfg
