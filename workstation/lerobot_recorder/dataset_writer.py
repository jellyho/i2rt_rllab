"""Async, queued ``LeRobotDataset`` (v3.0) writer for bimanual YAM episodes.

The recorder buffers each episode's frames in memory and **submits the whole
episode** to this writer. A single background worker thread then encodes/saves
episodes **one at a time** off a queue — so LeRobot's per-trajectory processing
(``save_episode``: parquet + video encode) never blocks the next collection.

Targets the official LeRobot Dataset **v3.0** API (``lerobot >= 0.4.0``):

* ``LeRobotDataset.create(...)`` / load existing for ``--resume``
* ``add_frame(frame)`` — per-frame ``task`` is a key inside the frame dict
* ``save_episode()`` / ``clear_episode_buffer()`` / ``finalize()``

Each frame is a dict of ``{feature_key: np.ndarray}`` plus ``"images"`` (a
``{cam: HxWx3}`` dict) and ``"task"``. The feature schema is built from a sample
frame so new fields (leader pose, eef, control_mode, …) flow through with no
schema edits here. ``mock=True`` skips ``lerobot`` and just counts.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import tempfile
import threading
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional

import numpy as np

from workstation.lerobot_recorder import outcomes as _outcomes
from workstation.lerobot_recorder.config import RecorderConfig
from workstation.lerobot_recorder.video_encoding import (
    EncodingBackendDecision,
    encode_frames_torchcodec,
    select_encoding_backend,
)

logger = logging.getLogger(__name__)


def _import_lerobot_dataset() -> type:
    try:
        from lerobot.datasets import LeRobotDataset  # lerobot >= 0.4.0 (v3.0)
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def dataset_dir(root: str, repo_id: str) -> str:
    """The actual dataset folder: ``<root>/<name>`` where ``name`` is the last segment
    of ``repo_id`` (e.g. root=~/lerobot_data, repo_id=hello/pick_and_place ->
    ~/lerobot_data/pick_and_place). ``root`` is a PARENT dir holding one folder per
    dataset, so several datasets can live side by side."""
    name = repo_id.strip("/").split("/")[-1] or "dataset"
    return os.path.join(os.path.expanduser(root), name)


def list_datasets(root: str) -> List[str]:
    """Dataset folder names under the parent ``root`` (for the setup-page picker).

    Every non-hidden subdirectory counts — a fresh dataset has no metadata yet.
    Sorted; [] when the root doesn't exist."""
    path = os.path.expanduser(root)
    try:
        return sorted(d for d in os.listdir(path) if not d.startswith(".") and os.path.isdir(os.path.join(path, d)))
    except OSError:
        return []


def dataset_tasks(ds_dir: str) -> List[str]:
    """Task strings already used by the dataset at ``ds_dir`` (best-effort).

    Reads LeRobot v3 ``meta/tasks.parquet`` (tasks are the index) or the v2
    ``meta/tasks.jsonl`` fallback. [] when neither exists or parsing fails."""
    meta = os.path.join(os.path.expanduser(ds_dir), "meta")
    parquet = os.path.join(meta, "tasks.parquet")
    if os.path.exists(parquet):
        try:
            import pandas as pd

            return [str(t) for t in pd.read_parquet(parquet).index]
        except Exception:
            return []
    jsonl = os.path.join(meta, "tasks.jsonl")
    if os.path.exists(jsonl):
        try:
            with open(jsonl) as fh:
                return [json.loads(line)["task"] for line in fh if line.strip()]
        except Exception:
            return []
    return []


def dataset_info(root: str) -> Dict:
    """Inspect the dataset dir at ``root`` for the setup page — no lerobot import.

    ``{"exists": bool, "episodes": int|None}``. Episode count is a best-effort read
    of LeRobot metadata (``meta/info.json``)."""
    path = os.path.expanduser(root)
    if not os.path.isdir(path) or not os.listdir(path):
        return {"exists": False, "episodes": None}
    episodes: Optional[int] = None
    info_path = os.path.join(path, "meta", "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path) as fh:
                meta_episodes = json.load(fh).get("total_episodes")
            if meta_episodes is not None:
                episodes = int(meta_episodes)
        except Exception:
            episodes = None
    return {"exists": True, "episodes": episodes}


def remove_dataset_root(root: str) -> None:
    """Delete the dataset dir at ``root`` (used by the GUI's confirmed overwrite)."""
    path = os.path.expanduser(root)
    if os.path.isdir(path):
        shutil.rmtree(path)
        logger.info("removed existing dataset dir %s", path)


class AsyncDatasetWriter:
    """Queued episode writer. ``submit()`` returns immediately; a worker saves."""

    def __init__(
        self,
        cfg: RecorderConfig,
        image_keys: List[str],
        image_shapes: Dict[str, tuple],
        extra_features: Optional[Dict[str, tuple]] = None,
    ) -> None:
        self.cfg = cfg
        self.image_keys = image_keys
        self.image_shapes = image_shapes
        # Per-step columns a policy declared at handshake, kept at the shape it declared so the
        # dataset describes itself. Flattening them would store [N, action_dim] candidates as an
        # anonymous 56-vector and leave the reader to know the layout out of band.
        self.extra_features = {str(k): tuple(v) for k, v in (extra_features or {}).items()}
        mock_writer = getattr(cfg, "mock_writer", None)
        self._mock = bool(cfg.mock) if mock_writer is None else bool(mock_writer)
        # Output format: "lerobot" (LeRobotDataset) or "abcdl" (the abcdl MP4+binary
        # training cache; one episode dir per submit, written via abcdl.EpisodeWriter).
        self._abcdl = str(getattr(cfg, "record_format", "lerobot")).lower() == "abcdl"
        self._ds = None
        self._features: Optional[dict] = None
        #: Columns this build writes that a resumed dataset does not declare; dropped per frame.
        #: See _note_undeclared_features.
        self._drop_keys: set = set()
        self._active_episode_frames: Optional[List[dict]] = None
        self._pyav_encode_temporary = None
        if not self._mock and not self._abcdl:
            self._encoding_decision = select_encoding_backend(
                getattr(cfg, "encoding_backend", "torchcodec"),
                float(getattr(cfg, "torchcodec_max_used_vram_gb", 5.0)),
            )
        else:
            self._encoding_decision = EncodingBackendDecision(
                str(getattr(cfg, "encoding_backend", "torchcodec")),
                "pyav",
                "video encoding is inactive",
            )
        self._effective_vcodec = str(cfg.vcodec)
        if (
            self._encoding_decision.requested == "torchcodec"
            and self._encoding_decision.effective == "pyav"
            and self._effective_vcodec == "auto"
        ):
            # The fallback is specifically meant to protect a policy already using
            # the GPU, so do not resolve auto back to NVENC through PyAV.
            self._effective_vcodec = "h264"
        # The dataset lives in <root>/<name>; root is just the parent directory.
        self._root = dataset_dir(cfg.root, cfg.repo_id)
        # Episodes already in the dataset before this session (resume); re-synced to
        # the authoritative count in open(). total/new_episodes build on this.
        self._initial_episodes = int(dataset_info(self._root)["episodes"] or 0) if cfg.resume else 0
        # Whole-dataset ✓/✗ counts, read off the outcome columns' per-episode stats on resume
        # (grow as episodes save).
        self._outcome_totals = self._read_outcome_totals() if cfg.resume else {"success": 0, "fail": 0}

        self._queue: "queue.Queue" = queue.Queue()
        # Streamed frames go to their own BOUNDED queue, which is the whole point: it is the
        # thing that caps how much of an episode is ever resident. 64 frames of three 640x480
        # cameras is ~180 MB, against ~7 GB for a 40 s episode held as a list. Encoding runs
        # ~6.5x faster than capture, so it should sit near empty; if it ever fills,
        # stream_frame blocks rather than dropping a frame.
        self._frame_queue: "queue.Queue" = queue.Queue(maxsize=64)
        self._streamed_frames = 0
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._n_episodes = 0  # saved episodes (incremented by the worker)
        self._n_submitted = 0
        self._saving = False  # True while the worker is encoding/writing one episode
        self._saving_index: Optional[int] = None  # episode index currently being saved
        self._saving_frames = 0  # frame count of the episode currently being saved
        self._finalized = False
        self._failed = False
        self._last_error = ""
        self._failed_episodes = 0
        # What this session saved, in order: {episode, outcome, task, frames, source}. The
        # in-memory successor of the sidecar row -- for the status line and for tests; the
        # verdict itself is in the dataset.
        self._saved_episodes: List[dict] = []
        # Episodes whose encoded video came out short (dropped frames). Not a save failure —
        # recording continues and finalize() repairs it — but the GUI surfaces the count so a
        # failing encoder is noticed at the rig instead of at training time.
        self._video_issues = 0
        self._finalize_lock = threading.Lock()
        self.low_disk = False  # set when an episode was refused for lack of free space

    def _free_gb(self) -> float:
        path = self._root if os.path.isdir(self._root) else (os.path.dirname(self._root) or ".")
        try:
            return shutil.disk_usage(path).free / 1e9
        except Exception:
            return float("inf")

    # ------------------------------------------------------------------ schema
    @staticmethod
    def _vector_names(key: str, dim: int) -> List[str]:
        from workstation.lerobot_recorder.config import action_names, eef_names, leader_names, state_names

        if key == "observation.state":
            return state_names()
        if key == "action":
            return action_names()
        if key == "observation.leader":
            return leader_names()
        if key == "observation.eef":
            return eef_names()
        return [f"{key.rsplit('.', 1)[-1]}.{i}" for i in range(dim)]

    def _build_features(self, sample: dict) -> dict:
        # Always video. Storing frames as images is not an option here: three 640x480
        # cameras at 30 fps is ~4.5 GB a minute of PNGs on disk.
        img_dtype = "video"
        feats: dict = {}
        for key in self.image_keys:
            feats[f"observation.images.{key}"] = {
                "dtype": img_dtype,
                "shape": self.image_shapes[key],
                "names": ["height", "width", "channels"],
            }
        for key, val in sample.items():
            if key in ("images", "task"):
                continue
            if key in self.extra_features:
                shape = self.extra_features[key]
                feats[key] = {"dtype": "float32", "shape": shape, "names": None}
                continue
            vec = np.asarray(val, dtype=np.float32).reshape(-1)
            feats[key] = {"dtype": "float32", "shape": (vec.size,), "names": self._vector_names(key, vec.size)}
        # The episode's verdict, as two per-frame bool features -- unconditionally. This is
        # where success/fail lives now; there is no sidecar. See outcomes.py for the shape.
        for key, spec in _outcomes.OUTCOME_FEATURES.items():
            feats[key] = dict(spec)
        if getattr(self.cfg, "rl_features", False):
            for key in ("success", "reward", "mc_return"):  # per-frame RL signals
                feats[key] = {"dtype": "float32", "shape": (1,), "names": None}
        return feats

    def _dataset_encoding_kwargs(self) -> dict:
        torchcodec = self._encoding_decision.effective == "torchcodec"
        enc = {
            "vcodec": self._effective_vcodec,
            # TorchCodec 0.10's VideoEncoder is episode-batch based. Encoding one
            # episode/camera at a time bounds host memory and uses our adapter.
            "batch_encoding_size": 1 if torchcodec else max(1, int(self.cfg.batch_encoding_size)),
        }
        if int(self.cfg.encoder_threads) > 0:
            enc["encoder_threads"] = int(self.cfg.encoder_threads)
        # Direct-to-encoder streaming, always, whenever PyAV is the encoder. This used to be
        # opt-in via recorder.streaming_encoding, defaulting to off -- and off is the one
        # configuration where LeRobot stages every frame of every camera as a ~1.5 MB PNG,
        # because that is how the non-streaming PyAV encoder gets its input. One deploy run
        # left 7926 PNGs and 2.7 GB behind that way. There is no reason to let a config key
        # choose that: StreamingVideoEncoder ships in stock lerobot 0.4.4, so the fast path
        # is always available, and _suppress_image_staging can then hold unconditionally.
        if not torchcodec:
            enc["streaming_encoding"] = True
        return enc

    @property
    def encoding_backend(self) -> str:
        return self._encoding_decision.effective

    def _configure_torchcodec_encoder(self) -> None:
        if self._encoding_decision.effective != "torchcodec" or self._ds is None:
            return
        original = getattr(self._ds, "_encode_temporary_episode_video", None)
        if original is None:
            logger.warning("LeRobot has no episode encoder hook; falling back to PyAV")
            self._encoding_decision = EncodingBackendDecision(
                self._encoding_decision.requested,
                "pyav",
                "installed LeRobot has no episode encoder hook",
                self._encoding_decision.gpu_memory,
            )
            return
        self._pyav_encode_temporary = original

        def encode_episode(_dataset: object, video_key: str, episode_index: int) -> Path:
            return self._encode_torchcodec_episode(video_key, episode_index)

        self._ds._encode_temporary_episode_video = MethodType(encode_episode, self._ds)

    def _apply_gop(self) -> None:
        """Set the keyframe interval on LeRobot's streaming encoder.

        ``LeRobotDataset.create`` hard-codes ``g=2`` and offers no way to pass another, so the
        only place to say otherwise is the encoder it just built. Two means a keyframe every
        other frame: the fastest random-frame decode there is, and about 2.8x the file. The
        trade is measured in RecorderConfig.gop; what matters here is that it is a decision
        rather than a default nobody noticed, and that it is said out loud in the log.
        """
        encoder = getattr(self._ds, "_streaming_encoder", None)
        if encoder is None:
            return
        want = max(1, int(getattr(self.cfg, "gop", 10)))
        current = getattr(encoder, "g", None)
        if current == want:
            return
        try:
            encoder.g = want
            logger.info("keyframe interval: g=%s (lerobot's default is %s)", want, current)
        except Exception as e:
            logger.warning("could not set the keyframe interval: %s", e)

    def _suppress_image_staging(self) -> None:
        """Stop LeRobot staging every video frame to disk as a PNG.

        `add_frame` writes one PNG per frame per camera unless a streaming encoder is
        installed — ~1.5 MB a frame, so a 2300-frame episode wrote 9741 files and 3.4 GB in
        the middle of the record loop. Neither encoder we actually run reads them:

          * TorchCodec is batch-only, so `streaming_encoding` is not passed to LeRobot and
            the frames come from `_active_episode_frames` in memory instead;
          * PyAV with `streaming_encoding` feeds the encoder directly as frames arrive.

        The one path that genuinely needs the PNGs is PyAV with streaming *off*, whose
        encoder reads the staging directory. `_dataset_encoding_kwargs` no longer produces
        that combination -- streaming is forced on whenever PyAV is the encoder -- so this
        raises instead of proceeding. Reaching it would mean LeRobot silently dropped the
        kwarg, and writing gigabytes of PNGs mid-episode is worse than refusing to start.

        Safe elsewhere because neither place LeRobot removes the directory needs it to
        exist: the rmtree after encoding lives inside the PyAV encoder (not called under
        TorchCodec, and not reached when streaming), and `clear_episode_buffer` guards its
        own with `is_dir()`.
        """
        ds = self._ds
        if ds is None or not hasattr(ds, "_save_image"):
            return
        streaming = getattr(ds, "_streaming_encoder", None) is not None
        torchcodec = self._encoding_decision.effective == "torchcodec"
        if not (streaming or torchcodec):
            raise RuntimeError(
                "PyAV encoder has no streaming encoder installed, so LeRobot would stage every "
                "frame of every camera as a ~1.5 MB PNG. This build passes streaming_encoding "
                "unconditionally under PyAV, so reaching here means lerobot ignored it — refusing "
                "to record rather than filling the disk mid-episode."
            )

        self._skipped_images = 0

        def _skip(_self, *_args, **_kwargs) -> None:
            self._skipped_images += 1

        ds._save_image = MethodType(_skip, ds)
        logger.info(
            "per-frame PNG staging disabled (%s reads frames directly)",
            "TorchCodec" if torchcodec else "the streaming encoder",
        )

    def _encode_torchcodec_episode(self, video_key: str, episode_index: int) -> Path:
        frames = self._active_episode_frames
        if frames is None:
            raise RuntimeError("TorchCodec encoder called without an active episode")
        current = select_encoding_backend(
            self._encoding_decision.requested,
            float(getattr(self.cfg, "torchcodec_max_used_vram_gb", 5.0)),
        )
        if current.effective != "torchcodec":
            if self._pyav_encode_temporary is None:
                raise RuntimeError(f"TorchCodec became unsafe and PyAV is unavailable: {current.reason}")
            logger.warning("encode-time VRAM check selected PyAV for %s: %s", video_key, current.reason)
            original_vcodec = getattr(self._ds, "vcodec", self._effective_vcodec)
            try:
                if str(self.cfg.vcodec) == "auto":
                    self._ds.vcodec = "h264"
                return self._pyav_encode_temporary(video_key, episode_index)
            finally:
                self._ds.vcodec = original_vcodec
        camera = video_key.split("observation.images.", 1)[-1]
        temp_dir = Path(tempfile.mkdtemp(dir=self._root))
        output = temp_dir / f"{video_key}_{episode_index:03d}.mp4"
        try:
            encode_frames_torchcodec(
                (frame["images"][camera] for frame in frames),
                output,
                fps=int(self.cfg.fps),
                vcodec=str(getattr(self._ds, "vcodec", self._effective_vcodec)),
            )
            return output
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if self._pyav_encode_temporary is None:
                raise
            logger.exception("TorchCodec failed for %s; retrying this camera with PyAV", video_key)
            return self._pyav_encode_temporary(video_key, episode_index)

    @staticmethod
    def _accepted_kwargs(func) -> set:
        """Parameter names ``func`` will accept, or an empty set when it takes **kwargs."""
        import inspect

        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            return set()
        if any(q.kind is inspect.Parameter.VAR_KEYWORD for q in params.values()):
            return set()
        return set(params)

    def _fit_encoding_kwargs(self, enc: dict, func) -> dict:
        """Drop only the keys ``func`` cannot take, folding vcodec/gop into RGBEncoderConfig.

        lerobot 0.5 replaced the flat ``vcodec=`` argument with an ``rgb_encoder`` config object
        that carries the codec AND the keyframe interval. Passing the flat kwargs and retrying
        with NONE of them on the TypeError meant one unsupported key silently dropped
        ``batch_encoding_size`` and ``streaming_encoding`` too -- so recording fell back to
        LeRobot's defaults: PNG staging for every frame of every camera, g=2, and whatever codec
        it likes. The log said "using defaults" and the run carried on.

        Filtering by signature keeps the supported keys, and the version that wants a config
        object gets one instead of losing the setting.
        """
        accepted = self._accepted_kwargs(func)
        if not accepted:
            return enc
        out = {k: v for k, v in enc.items() if k in accepted}
        if "vcodec" in enc and "vcodec" not in accepted and "rgb_encoder" in accepted:
            try:
                from lerobot.configs.video import RGBEncoderConfig
            except ImportError:
                return out
            # g is the keyframe interval _apply_gop monkeypatches onto the encoder after the
            # fact on 0.4.x; here it is simply part of the config the encoder is built from.
            out["rgb_encoder"] = RGBEncoderConfig(
                vcodec=str(enc["vcodec"]),
                g=max(1, int(getattr(self.cfg, "gop", 10))),
            )
        dropped = sorted(set(enc) - set(out) - {"vcodec"})
        if dropped:
            logger.warning("this lerobot does not accept %s; leaving them at its defaults", dropped)
        return out

    def _create_encoding_kwargs(self) -> dict:
        enc = self._dataset_encoding_kwargs()
        # Async PNG writer (parallelizes the slow pre-encode step). Only pass when
        # enabled so older LeRobot versions without these kwargs keep working.
        if int(self.cfg.image_writer_threads) > 0 or int(self.cfg.image_writer_processes) > 0:
            enc["image_writer_threads"] = int(self.cfg.image_writer_threads)
            enc["image_writer_processes"] = int(self.cfg.image_writer_processes)
        return enc

    # ------------------------------------------------------- RL per-frame features
    def _rl_config_path(self) -> str:
        return os.path.join(self._root, "rl_config.json")

    def _read_rl_config_on_resume(self) -> None:
        """Continuing an existing dataset must reuse its reward scheme — load the
        sidecar and override the (possibly different) current config."""
        path = self._rl_config_path()
        if self.cfg.resume and os.path.exists(path):
            try:
                rc = json.load(open(path))
                self.cfg.rl_features = bool(rc.get("rl_features", self.cfg.rl_features))
                self.cfg.reward_mode = str(rc.get("reward_mode", self.cfg.reward_mode))
                self.cfg.discount_factor = float(rc.get("discount_factor", self.cfg.discount_factor))
                logger.info("resume: inheriting RL settings %s", rc)
            except Exception as e:
                logger.error("could not read rl_config.json on resume: %s", e)

    def _write_rl_config_once(self) -> None:
        """Persist the reward scheme next to the dataset so resume can inherit it."""
        path = self._rl_config_path()
        if os.path.exists(path):
            return
        try:
            os.makedirs(self._root, exist_ok=True)
            json.dump(
                {
                    "rl_features": bool(getattr(self.cfg, "rl_features", False)),
                    "reward_mode": getattr(self.cfg, "reward_mode", "sparse"),
                    "discount_factor": float(getattr(self.cfg, "discount_factor", 0.99)),
                },
                open(path, "w"),
                indent=2,
            )
        except Exception as e:
            logger.error("could not write rl_config.json: %s", e)

    def _frame_features(self, n_frames: int, outcome: Optional[str]) -> Optional[dict]:
        """Per-frame success/reward/mc_return for one episode, or None when disabled."""
        if not getattr(self.cfg, "rl_features", False):
            return None
        try:
            from abcdl.rewards import compute_frame_features
        except ImportError as e:
            raise RuntimeError("rl_features needs the abcdl package — pip install -e '.[abcdl]'") from e
        return compute_frame_features(
            n_frames,
            success=(outcome == "success"),
            mode=getattr(self.cfg, "reward_mode", "sparse"),
            discount=float(getattr(self.cfg, "discount_factor", 0.99)),
        )

    # ------------------------------------------------------------------ lifecycle
    def open(self, sample_frame: dict) -> None:
        """Open the dataset using ``sample_frame`` to derive the feature schema."""
        # On resume, inherit the dataset's original RL settings so the per-frame
        # reward/return scheme stays consistent across a continued collection.
        self._read_rl_config_on_resume()
        self._features = self._build_features(sample_frame)
        logger.warning(
            "recording encoder: requested=%s effective=%s vcodec=%s (%s)",
            self._encoding_decision.requested,
            self._encoding_decision.effective,
            self._effective_vcodec,
            self._encoding_decision.reason,
        )
        if self._abcdl:
            os.makedirs(self._root, exist_ok=True)
            if self.cfg.resume and os.path.isdir(self._root):
                self._n_episodes = sum(
                    1
                    for d in os.listdir(self._root)
                    if d.startswith("episode_") and os.path.exists(os.path.join(self._root, d, "states_actions.bin"))
                )
                logger.info("abcdl dataset resuming at %s (%d episodes)", self._root, self._n_episodes)
            else:
                logger.info(
                    "abcdl dataset at %s (format=abcdl, size=%d)",
                    self._root,
                    int(getattr(self.cfg, "abcdl_size", 224)),
                )
            self._initial_episodes = self._n_episodes  # authoritative pre-session count
            self._outcome_totals = self._read_outcome_totals()
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
            return
        if not self._mock:
            resume_local = self.cfg.resume and os.path.isdir(self._root)
            if resume_local:
                info_path = os.path.join(self._root, "meta", "info.json")
                if not os.path.isfile(info_path):
                    raise RuntimeError(
                        f"Cannot continue local dataset at {self._root}: "
                        "meta/info.json is missing. Uncheck 'Continue collecting' "
                        "and confirm overwrite to create it again."
                    )
                try:
                    with open(info_path) as fh:
                        local_info = json.load(fh)
                    empty_local = (
                        int(local_info.get("total_episodes", 0)) == 0 and int(local_info.get("total_frames", 0)) == 0
                    )
                    if not empty_local and _outcomes.predates_outcome_schema(self._root):
                        # Appending to a dataset that predates the outcome columns would produce a
                        # parquet whose rows disagree about their schema -- refuse before opening it.
                        # The fix is one command; the error names it.
                        raise _outcomes.OutcomeColumnsMissing(self._root)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Cannot read local dataset metadata at {info_path}: {exc}") from exc
                if empty_local:
                    logger.warning("recreating empty local dataset at %s", self._root)
                    shutil.rmtree(self._root)
                    resume_local = False

            if resume_local:
                # A previous run that died mid-episode leaves files the metadata never
                # learned about, and a half-written data parquet has no footer. LeRobot
                # globs data/ rather than reading the file list from the metadata, so that
                # one file makes the whole dataset -- including every good episode --
                # refuse to open. Clear the orphans before trying.
                self._recover_from_crash()
                LeRobotDataset = _import_lerobot_dataset()
                enc = self._fit_encoding_kwargs(self._dataset_encoding_kwargs(), LeRobotDataset.__init__)
                self._ds = LeRobotDataset(self.cfg.repo_id, root=self._root, **enc)
                self._note_undeclared_features()
                self._n_episodes = int(
                    getattr(self._ds, "num_episodes", getattr(getattr(self._ds, "meta", None), "total_episodes", 0))
                )
                self._cleanup_interrupted_image_dirs(self._n_episodes)
                writer = getattr(self._ds, "writer", None)
                if writer is not None and hasattr(writer, "cleanup_interrupted_episode"):
                    writer.cleanup_interrupted_episode(self._n_episodes)
                elif hasattr(self._ds, "clear_episode_buffer"):
                    # LeRobot releases before DatasetWriter stored the in-progress
                    # buffer directly on LeRobotDataset.  A failed encode can leave
                    # PNGs for the next episode index behind; remove them before new
                    # frames reuse that same index.
                    if getattr(self._ds, "episode_buffer", None) is None and hasattr(
                        self._ds, "create_episode_buffer"
                    ):
                        self._ds.episode_buffer = self._ds.create_episode_buffer()
                    self._ds.clear_episode_buffer(delete_images=True)
                logger.info(
                    "dataset resuming at %s (%d existing episodes, vcodec=%s, batch=%s)",
                    self._root,
                    self._n_episodes,
                    self.cfg.vcodec,
                    self.cfg.batch_encoding_size,
                )
            else:
                LeRobotDataset = _import_lerobot_dataset()
                enc = self._fit_encoding_kwargs(self._create_encoding_kwargs(), LeRobotDataset.create)
                self._ds = LeRobotDataset.create(
                    repo_id=self.cfg.repo_id,
                    fps=self.cfg.fps,
                    features=self._features,
                    root=self._root,
                    robot_type=self.cfg.robot_type,
                    use_videos=True,
                    **enc,
                )
                logger.info(
                    "dataset created at %s (repo_id=%s, vcodec=%s, batch=%s)",
                    self._root,
                    self.cfg.repo_id,
                    self.cfg.vcodec,
                    self.cfg.batch_encoding_size,
                )
            self._configure_torchcodec_encoder()
            self._apply_gop()
            self._suppress_image_staging()
        else:
            if self.cfg.resume:
                self._n_episodes = len(self._read_mock_manifest())
            logger.info("MOCK writer (repo_id=%s); features=%s", self.cfg.repo_id, sorted(self._features))

        self._initial_episodes = self._n_episodes  # authoritative pre-session count
        self._outcome_totals = self._read_outcome_totals()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _recover_from_crash(self) -> None:
        """Quarantine anything a previous crash left behind, before opening the dataset.

        Best-effort: if recovery itself fails we still try to open, so a bug here can only
        cost the clear error message, never the session."""
        try:
            from workstation.lerobot_recorder.crash_recovery import describe, recover

            summary = describe(self._root)
            if summary == "no crash leftovers":
                return
            logger.warning(
                "found leftovers from an interrupted recording — a truncated data file "
                "would stop this dataset opening at all, so they are being moved aside:\n%s",
                summary,
            )
            result = recover(self._root)
            if result["moved"]:
                logger.warning(
                    "moved %d item(s) to %s — committed episodes are untouched",
                    len(result["moved"]),
                    result["quarantine"],
                )
        except Exception as e:
            logger.error("crash recovery skipped: %s", e)

    def _cleanup_interrupted_image_dirs(self, episode_index: int) -> None:
        """Remove temp PNG directories for the next, not-yet-committed episode.

        Some LeRobot releases report video-backed cameras only in ``video_keys``
        while their interrupted-episode cleanup iterates ``image_keys``.  Address
        the on-disk scratch paths directly so stale frames cannot be mixed into a
        resumed episode with the same index.
        """
        removed = 0
        for image_key in self.image_keys:
            image_dir = os.path.join(
                self._root,
                "images",
                f"observation.images.{image_key}",
                f"episode-{int(episode_index):06d}",
            )
            if os.path.isdir(image_dir):
                shutil.rmtree(image_dir)
                removed += 1
        if removed:
            logger.warning(
                "removed stale scratch images for uncommitted episode #%d (%d camera directories)",
                episode_index,
                removed,
            )

    def _read_outcome_totals(self) -> Dict[str, int]:
        """Whole-dataset success/fail counts, from the outcome columns' per-episode stats."""
        if not os.path.isdir(self._root):
            return {"success": 0, "fail": 0}
        if self._mock:
            rows = self._read_mock_manifest()
            return {k: sum(1 for r in rows if r.get("outcome") == k) for k in ("success", "fail")}
        return _outcomes.outcome_totals(self._root)

    # A mock writer records nothing, so across a save-and-continue it would forget how many
    # episodes it "saved" and restart at #0. This manifest is that memory -- a list of the
    # saved_episodes rows -- and nothing else reads it. It is not a dataset.
    _MOCK_MANIFEST = "mock_manifest.json"

    def _read_mock_manifest(self) -> List[dict]:
        try:
            with open(os.path.join(self._root, self._MOCK_MANIFEST)) as fh:
                rows = json.load(fh)
            return rows if isinstance(rows, list) else []
        except (OSError, ValueError):
            return []

    def _append_mock_manifest(self, row: dict) -> None:
        os.makedirs(self._root, exist_ok=True)
        rows = self._read_mock_manifest() + [row]
        path = os.path.join(self._root, self._MOCK_MANIFEST)
        with open(path + ".tmp", "w") as fh:
            json.dump(rows, fh)
        os.replace(path + ".tmp", path)

    # ------------------------------------------------------------------ submit
    def _check_writable(self) -> None:
        if self._failed:
            raise RuntimeError(f"dataset writer stopped after a save failure: {self._last_error}")
        if self._finalized:
            raise RuntimeError("dataset writer has already been finalized")

    def submit(self, frames: List[dict], outcome: Optional[str], task: str) -> None:
        """Enqueue a complete episode (list of frame dicts) for background saving."""
        self._check_writable()
        if frames:
            self._queue.put(("episode", frames, outcome, task))
            with self._lock:
                self._n_submitted += 1

    # -------------------------------------------------------------- streaming
    def supports_streaming(self) -> bool:
        """Whether frames can be handed over one at a time instead of an episode at a time.

        The verdict itself streams fine: ``next.success`` / ``next.done`` are False on every
        frame but the last, so frames go in with False and only the final one is rewritten
        once the outcome is known (see ``_handle_stream_item``). The optional RL signals
        (reward / mc_return) are the exception -- they depend on the episode's length at every
        frame -- so those recordings keep buffering the whole episode.
        """
        return not (self._mock or self._abcdl or bool(getattr(self.cfg, "rl_features", False)))

    def stream_frame(self, frame: dict, task: str) -> None:
        """Hand one captured frame to the writer thread, with the task it belongs to.

        The task is passed per frame rather than waiting for ``end_episode``, because LeRobot
        wants it on every ``add_frame`` and by the time the episode ends its frames are already
        inside the dataset. Taking it from the frame dict instead -- which carries no task --
        silently wrote an empty string for every streamed episode.


        The point of this path is that no episode is ever held whole in memory. Three cameras
        at 640x480 are ~2.8 MB a frame, so a 40 s episode buffered as a list is ~7 GB -- enough
        to push the machine into swap and get the recorder OOM-killed mid-episode. Queued, the
        resident set is bounded by the queue instead.

        The queue is bounded and this call BLOCKS when it is full. That is deliberate: dropping
        a frame silently is how videos end up shorter than the metadata claims. Blocking is
        near-hypothetical anyway -- the streaming encoder measures ~196 frames/s on three
        640x480 cameras against a 30 fps capture rate.
        """
        self._check_writable()
        self._frame_queue.put(("frame", frame, task))

    def end_episode(self, outcome: Optional[str], task: str) -> None:
        """Close the streamed episode and save it."""
        self._check_writable()
        self._frame_queue.put(("end", outcome, task))
        with self._lock:
            self._n_submitted += 1

    def abort_episode(self) -> None:
        """Throw away the streamed episode in flight (the review 'Delete')."""
        if self._failed or self._finalized:
            return
        self._frame_queue.put(("abort",))

    @property
    def queue_depth(self) -> int:
        """Episodes WAITING in the queue (not counting the one being saved)."""
        return self._queue.qsize()

    @property
    def saving(self) -> bool:
        """True while the worker is encoding/writing an episode (qsize is 0 then)."""
        with self._lock:
            return self._saving

    @property
    def pending_total(self) -> int:
        """Episodes not yet on disk: queued + the one currently being saved."""
        with self._lock:
            return self._queue.qsize() + (1 if self._saving else 0)

    @property
    def progress(self) -> dict:
        """Detailed writer state for the GUI: how much is saved, what's encoding now,
        and how much is waiting. One worker by design (a LeRobotDataset is single-writer)."""
        with self._lock:
            return {
                "workers": 1,
                "saved": self._n_episodes,
                "submitted": self._n_submitted,
                "saving": self._saving,
                "saving_index": self._saving_index,
                "saving_frames": self._saving_frames,
                "queued": self._queue.qsize(),
                "finalized": self._finalized,
                "failed": self._failed,
                "last_error": self._last_error,
                "failed_episodes": self._failed_episodes,
                "video_issues": self._video_issues,
                "encoding_backend": self._encoding_decision.effective,
                "encoding_backend_reason": self._encoding_decision.reason,
            }

    @property
    def num_episodes(self) -> int:
        with self._lock:
            return self._n_episodes

    @property
    def total_episodes(self) -> int:
        """Episodes in the WHOLE dataset: pre-session (resume) + saved this session."""
        with self._lock:
            return max(self._n_episodes, self._initial_episodes)

    @property
    def new_episodes(self) -> int:
        """Episodes saved THIS session (excludes what a resumed dataset already had)."""
        with self._lock:
            return max(0, self._n_episodes - self._initial_episodes)

    @property
    def outcome_totals(self) -> Dict[str, int]:
        """Whole-dataset {success, fail} counts (dataset history on resume + this session)."""
        with self._lock:
            return dict(self._outcome_totals)

    @property
    def saved_episodes(self) -> List[dict]:
        """Episodes this session saved, oldest first: {episode, outcome, task, frames, source}."""
        with self._lock:
            return [dict(e) for e in self._saved_episodes]

    @property
    def finalized(self) -> bool:
        return self._finalized

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        while not (self._stop.is_set() and self._queue.empty() and self._frame_queue.empty()):
            # Streamed frames first: they are the live capture, and letting them wait behind a
            # whole-episode save is what the queue exists to avoid.
            if self._drain_frame_queue():
                continue
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            _kind, frames, outcome, task = item
            with self._lock:
                self._saving = True
                self._saving_index = self._n_episodes
                self._saving_frames = len(frames)
            try:
                self._save_episode(frames, outcome, task)
            except Exception as e:
                logger.error("episode save failed: %s", e)
                self._recover_failed_episode_buffer()
                with self._lock:
                    self._failed = True
                    self._last_error = f"{type(e).__name__}: {e}"
                    self._failed_episodes += 1
                self._discard_queued_after_failure()
                return
            finally:
                with self._lock:
                    self._saving = False
                    self._saving_index = None
                    self._saving_frames = 0
                self._queue.task_done()

    def _note_undeclared_features(self) -> None:
        """On resume, find the columns this build would write that the dataset does not declare.

        LeRobot rejects an add_frame whose keys differ from the declared schema in EITHER direction,
        and it rejects it per frame -- so a recorder that gained a column since the dataset was
        created does not fail at open, it fails on every save and the writer stops. That is how a
        new `action_seq` turned a rollout into a wall of identical errors.

        The dataset's schema wins: a column it never declared cannot be added without rewriting it,
        and dropping the column costs one diagnostic while refusing to record costs the session. So
        the extras are dropped, once, loudly, naming both the column and the fix.
        """
        self._drop_keys = set()
        declared = getattr(getattr(self._ds, "meta", None), "features", None) or getattr(self._ds, "features", None)
        if not declared or not self._features:
            return
        extra = {k for k in self._features if k not in declared}
        if not extra:
            return
        self._drop_keys = extra
        logger.warning(
            "%s was created before this recorder had %s, so %s will NOT be recorded into it. "
            "Everything else is unaffected. Record into a new --repo-id to get the column, or "
            "migrate the dataset if you need it on the existing episodes.",
            self._root,
            ", ".join(sorted(extra)),
            "they" if len(extra) > 1 else "it",
        )

    def _to_lerobot_frame(self, f: dict, task: str, extra: Optional[dict] = None) -> dict:
        frame = {f"observation.images.{k}": v for k, v in f["images"].items()}
        for key, val in f.items():
            if key in ("images", "task") or key in self._drop_keys:
                continue
            array = np.asarray(val, dtype=np.float32)
            # Declared extras go in at their declared shape; they travel flattened so nothing
            # in between has to care what the layout is.
            frame[key] = array.reshape(self.extra_features[key]) if key in self.extra_features else array
        if extra:
            for k, v in extra.items():
                frame[k] = np.asarray([v], dtype=np.float32)
        # The verdict is not known while frames stream in; every frame carries False and the
        # terminal frame is rewritten at end_episode. LeRobot validates that every declared
        # feature is present on every add_frame, so the placeholder is not optional.
        for key in _outcomes.OUTCOME_FEATURES:
            frame.setdefault(key, _outcomes.frame_value(False))
        frame["task"] = task
        return frame

    def _drain_frame_queue(self) -> bool:
        """Consume streamed frames / episode boundaries. True if anything was handled.

        Runs on the writer thread, so the encode cost stays off the capture loop exactly as it
        did when whole episodes were queued -- what changed is that the frames are no longer
        all resident at once.
        """
        handled = False
        while True:
            try:
                item = self._frame_queue.get_nowait()
            except queue.Empty:
                return handled
            handled = True
            try:
                self._handle_stream_item(item)
            except Exception as e:
                logger.error("streamed episode failed: %s", e)
                self._recover_failed_episode_buffer()
                self._streamed_frames = 0
                with self._lock:
                    self._failed = True
                    self._last_error = f"{type(e).__name__}: {e}"
                    self._failed_episodes += 1
                self._discard_queued_after_failure()
                raise
            finally:
                self._frame_queue.task_done()

    def _handle_stream_item(self, item: tuple) -> None:
        kind = item[0]
        if kind == "frame":
            _, frame, task = item
            self._ds.add_frame(self._to_lerobot_frame(frame, task))
            self._streamed_frames += 1
            with self._lock:
                self._saving_frames = self._streamed_frames
            return
        if kind == "abort":
            if self._streamed_frames:
                self._ds.clear_episode_buffer()
                logger.info("discarded %d streamed frame(s)", self._streamed_frames)
            self._streamed_frames = 0
            with self._lock:
                self._saving_frames = 0
            return

        _, outcome, task = item
        n_frames, self._streamed_frames = self._streamed_frames, 0
        if not n_frames:
            return
        free = self._free_gb()
        if free < self.cfg.min_free_gb:
            # Frames are already inside LeRobot's buffer, so "not saved" means dropping them
            # rather than declining to start -- say so plainly.
            self.low_disk = True
            self._ds.clear_episode_buffer()
            logger.warning(
                "LOW DISK: %.1f GB free (< %s GB) — %d streamed frame(s) DISCARDED",
                free,
                self.cfg.min_free_gb,
                n_frames,
            )
            with self._lock:
                self._saving_frames = 0
            return
        self.low_disk = False
        self._stamp_streamed_verdict(outcome, n_frames)
        with self._lock:
            self._saving = True
            self._saving_index = self._n_episodes
        try:
            self._ds.save_episode()
            self._verify_episode_video(self._n_episodes)
        finally:
            with self._lock:
                self._saving = False
                self._saving_index = None
                self._saving_frames = 0
        self._write_rl_config_once()
        with self._lock:
            episode_index = self._n_episodes
            self._n_episodes += 1
        self._record_outcome(episode_index, outcome, task, n_frames)
        logger.info("saved episode #%d (%d frames, outcome=%s, streamed)", episode_index, n_frames, outcome)

    def _recover_failed_episode_buffer(self) -> None:
        """Reset LeRobot's mutated buffer so finalize cannot raise a secondary error."""
        if self._mock or self._abcdl or self._ds is None:
            return
        try:
            self._ds.clear_episode_buffer(delete_images=True)
        except Exception as e:
            logger.error("could not clean failed episode buffer: %s", e)

    def _discard_queued_after_failure(self) -> None:
        discarded = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
            discarded += 1
        if discarded:
            logger.error("discarded %d queued episode(s) after writer failure", discarded)

    def _save_episode(self, frames: List[dict], outcome: Optional[str], task: str) -> None:
        free = self._free_gb()
        if free < self.cfg.min_free_gb:
            self.low_disk = True
            logger.warning("LOW DISK: %.1f GB free (< %s GB) — episode NOT saved", free, self.cfg.min_free_gb)
            return
        self.low_disk = False
        # Per-frame RL signals from the episode outcome (None when disabled).
        ff = self._frame_features(len(frames), outcome)
        verdict = _outcomes.terminal_flags(len(frames), outcome)
        if not self._mock:
            if self._abcdl:
                self._save_episode_abcdl(frames, task, self._n_episodes, ff)
            else:
                self._active_episode_frames = frames
                try:
                    for i, f in enumerate(frames):
                        frame = {f"observation.images.{k}": v for k, v in f["images"].items()}
                        for key, val in f.items():
                            if key in ("images", "task") or key in self._drop_keys:
                                continue
                            frame[key] = np.asarray(val, dtype=np.float32)
                        if ff:
                            for k, v in ff.items():
                                frame[k] = np.asarray([v[i]], dtype=np.float32)
                        for k, v in verdict.items():
                            frame[k] = v[i]
                        frame["task"] = task
                        self._ds.add_frame(frame)
                    if self._encoding_decision.effective == "torchcodec":
                        self._ds.save_episode(parallel_encoding=False)
                    else:
                        self._ds.save_episode()
                finally:
                    self._active_episode_frames = None
                # The episode's video is on disk now — check it really holds the frames
                # LeRobot just recorded for it, whichever backend encoded it.
                self._verify_episode_video(self._n_episodes)
        self._write_rl_config_once()
        with self._lock:
            episode_index = self._n_episodes
            self._n_episodes += 1
        self._record_outcome(episode_index, outcome, task, len(frames))
        logger.info("saved episode #%d (%d frames, outcome=%s)", episode_index, len(frames), outcome)

    def _verify_episode_video(self, episode_index: int) -> None:
        """Right after an episode is encoded, check every camera file really holds the frames
        LeRobot just recorded for it — and shout NOW if it doesn't.

        The stream-copy concat that appends this episode to the shared camera file can silently
        drop a frame at the join. Nothing in the metadata notices: the file simply ends up
        shorter than the metadata believes, every later episode in that file is misaligned, and
        the last one's window runs off the end — which only surfaces much later as
        ``IndexError: Invalid frame index=N`` when someone loads the dataset to train.

        Comparing the file's real frame count (read from the mp4 sample table — one ffprobe per
        camera, ~30 ms, no decoding) against the ``to_timestamp`` LeRobot just wrote catches it
        at the rig, while the operator can still switch encoder. ``finalize()`` repairs whatever
        slipped through; this is the early warning. Best-effort — never breaks a recording."""
        if int(self.cfg.batch_encoding_size) > 1:
            return  # video for this episode isn't encoded yet; finalize() does the checking
        try:
            from workstation.lerobot_recorder.video_integrity import video_frame_count

            meta = getattr(self._ds, "meta", None)
            latest = getattr(meta, "latest_episode", None)
            if not latest:
                return

            def _scalar(v):
                return v[0] if isinstance(v, (list, tuple, np.ndarray)) else v

            fps = int(self.cfg.fps)
            for key in self.image_keys:
                vkey = f"observation.images.{key}"
                if f"videos/{vkey}/to_timestamp" not in latest:
                    continue
                path = os.path.join(
                    meta.root,
                    meta.video_path.format(
                        video_key=vkey,
                        chunk_index=int(_scalar(latest[f"videos/{vkey}/chunk_index"])),
                        file_index=int(_scalar(latest[f"videos/{vkey}/file_index"])),
                    ),
                )
                claimed = int(round(float(_scalar(latest[f"videos/{vkey}/to_timestamp"])) * fps))
                actual = video_frame_count(path)
                if actual is None or actual >= claimed:
                    continue
                with self._lock:
                    self._video_issues += 1
                logger.error(
                    "DROPPED VIDEO FRAMES on episode #%d camera %s: %s holds %d frames but the "
                    "metadata expects %d. Later episodes in this file are now misaligned and the "
                    "dataset will raise 'IndexError: Invalid frame index' when loaded. It is "
                    "auto-repaired when you stop recording; if this keeps happening, %s",
                    episode_index,
                    key,
                    os.path.basename(path),
                    actual,
                    claimed,
                    self._encoder_remedy(),
                )
        except Exception as e:
            logger.debug("per-episode video verification skipped: %s", e)

    def _encoder_remedy(self) -> str:
        """What to change in config.yaml when an encoder keeps dropping frames.

        The advice depends on which backend actually encoded. It never suggests turning
        streaming off: that is the PNG-staging path, which no longer exists."""
        if self._encoding_decision.effective == "torchcodec":
            return "try recorder.encoding_backend: pyav in config.yaml."
        return "set recorder.vcodec: h264 in config.yaml (CPU encode; slower but reliable)."

    def _save_episode_abcdl(
        self, frames: List[dict], task: str, ep_index: int, frame_features: Optional[dict] = None
    ) -> None:
        """Write one episode as an abcdl dir via abcdl.EpisodeWriter (images square-resized)."""
        import cv2

        try:
            from abcdl.writer import EpisodeWriter
        except ImportError as e:
            raise RuntimeError("format: abcdl needs the abcdl package — pip install -e '.[abcdl]'") from e

        size = int(getattr(self.cfg, "abcdl_size", 224))
        out_dir = os.path.join(self._root, f"episode_{ep_index:06d}")
        tick = int(1e9 / max(1, int(self.cfg.fps)))
        w = EpisodeWriter(out_dir, formats=("abcdl",), fps=int(self.cfg.fps), cameras=list(self.image_keys))
        for i, f in enumerate(frames):
            imgs = {
                k: cv2.resize(np.asarray(v), (size, size), interpolation=cv2.INTER_AREA)
                for k, v in f["images"].items()
            }
            w.add_frame(
                i * tick, np.asarray(f["observation.state"], np.float64), np.asarray(f["action"], np.float64), imgs
            )
        w.save(task=task, frame_features=frame_features)

    def _record_outcome(self, episode_index: int, outcome: Optional[str], task: str, n_frames: int) -> None:
        """Only the live counters. The verdict itself went into the dataset with the frames."""
        state = _outcomes.normalize(outcome)
        row = {
            "episode": int(episode_index),
            "outcome": state,
            "task": task,
            "frames": int(n_frames),
            "source": self.cfg.record_source,
        }
        with self._lock:
            if state in ("success", "fail"):
                self._outcome_totals[state] += 1
            self._saved_episodes.append(row)
        if self._mock:
            self._append_mock_manifest(row)

    def _stamp_streamed_verdict(self, outcome: Optional[str], n_frames: int) -> None:
        """Rewrite the terminal frame's ``next.success`` / ``next.done`` in LeRobot's buffer.

        Streamed frames were added with both False (the verdict did not exist yet). Only the
        last frame carries it, so only one element per column changes -- and it has to change
        BEFORE ``save_episode`` computes the episode's stats, or ``stats/next.success/max``
        would say the episode was never judged.
        """
        buf = getattr(self._ds, "episode_buffer", None)
        if not buf:
            return
        flags = _outcomes.terminal_flags(n_frames, outcome)
        for key, column in flags.items():
            if buf.get(key):
                buf[key][-1] = column[-1]

    # ------------------------------------------------------------------ shutdown
    def finalize(self) -> None:
        """Drain the queue, stop the worker, then close the LeRobot dataset."""
        with self._finalize_lock:
            if self._finalized:
                return
            self._stop.set()
            if self._worker is not None:
                self._worker.join(timeout=600.0)
            if not self._mock and self._ds is not None and self._n_episodes > 0:
                self._flush_pending_batch()
                try:
                    self._ds.finalize()
                    logger.info("dataset finalized (parquet/metadata closed)")
                except Exception as e:
                    logger.error("dataset finalize failed: %s", e)
                self._verify_video_lengths()
            self._finalized = True

    def _verify_video_lengths(self) -> None:
        """Verify each camera's video against its metadata after finalize, and self-heal.

        Two independent things go wrong when the GPU/streaming encoder is under load, and
        they need different checks (see ``video_integrity`` for the full write-up):

        1. **The file is shorter than the metadata claims** — the stream-copy concat that
           appends an episode to the shared camera file dropped a frame at the join. This is
           invisible to metadata (every window still spans ``length`` frames), silently
           misaligns every later episode in that file, and makes the *last* episode's window
           run off the end — which is what raises ``IndexError: Invalid frame index=N`` when
           the dataset is later loaded for training. Needs the real frame count to spot.
        2. **One episode's window is a frame short** — the encoder returned a short clip for
           that episode. Harmless to load but it blocks ``delete_episodes`` (LeRobot asserts
           length == video-frame count before re-encoding a shared file).

        (1) is repaired first, by realigning the affected episodes onto the frames that are
        really theirs (or appending a duplicate frame when the loss is off the end of the
        file); (2) is then snapped back to consistency. Both WARN loudly so the operator can
        switch encoder if it recurs. Best-effort: a check failure never breaks a finished
        recording."""
        try:
            from workstation.lerobot_recorder.dataset_editor import (
                repair_length_consistency,
                video_length_mismatches,
            )
            from workstation.lerobot_recorder.video_integrity import (
                repair_short_videos,
                video_file_shortfalls,
            )

            short = video_file_shortfalls(self._root)
            if short:
                logger.warning(
                    "TRUNCATED VIDEO on %d file(s): %s — the encoder/concat dropped %d frame(s), "
                    "which would raise 'IndexError: Invalid frame index' when this dataset is "
                    "loaded. Repairing now; if this recurs, %s",
                    len(short),
                    ", ".join(
                        f"{s['video_key'].split('.')[-1]} file-{s['file']:03d} ({s['actual']}/{s['claimed']} frames)"
                        for s in short
                    ),
                    sum(s["missing"] for s in short),
                    self._encoder_remedy(),
                )
                for r in repair_short_videos(self._root):
                    if r["status"] == "repaired":
                        logger.warning(
                            "repaired %s file-%03d: realigned %d episode(s) %s, appended %d frame(s)",
                            r["video_key"].split(".")[-1],
                            r["file"],
                            len(r["shifted_episodes"]),
                            r["shifted_episodes"],
                            r["appended_frames"],
                        )
                    else:
                        logger.error(
                            "COULD NOT repair %s file-%03d (%s) — episodes %s will fail to load; "
                            "run: workstation/yam-data check-videos --fix",
                            r["video_key"].split(".")[-1],
                            r["file"],
                            r.get("error", r["status"]),
                            r["overrun_episodes"],
                        )

            bad = video_length_mismatches(self._root)
            if not bad:
                return
            cams = sorted({b["camera"].split(".")[-1] for b in bad})
            eps = sorted({b["episode"] for b in bad})
            logger.warning(
                "video/length mismatch on %d episode(s) %s (cameras: %s) — the video encoder "
                "dropped a trailing frame. Repairing metadata so the dataset stays editable; "
                "if this recurs often, %s",
                len(eps),
                eps,
                ", ".join(cams),
                self._encoder_remedy(),
            )
            n = repair_length_consistency(self._root)
            logger.warning("repaired %d episode-video length field(s) to match frame length", n)
        except Exception as e:
            logger.error("video-length verification skipped: %s", e)

    def _flush_pending_batch(self) -> None:
        """With batch_encoding_size > 1, LeRobot defers video encoding and its finalize()
        does NOT flush the trailing (< batch) episodes — they'd stay as temp PNGs and the
        dataset would be part-video/part-images. Encode that remainder here before closing."""
        ds = self._ds
        pending = int(getattr(ds, "episodes_since_last_encoding", 0) or 0)
        if pending <= 0 or not hasattr(ds, "_batch_save_episode_video"):
            return
        try:
            end = int(ds.num_episodes)
            ds._batch_save_episode_video(end - pending, end)
            ds.episodes_since_last_encoding = 0
            logger.info("flushed %d trailing episode(s) to video before finalize", pending)
        except Exception as e:
            logger.error("could not flush pending batch encode (%d episodes): %s", pending, e)
