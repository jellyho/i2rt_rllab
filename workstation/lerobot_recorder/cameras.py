"""RealSense camera manager — opens the configured cameras, grabs latest frames.

Each :class:`~workstation.lerobot_recorder.config.CameraSpec` maps a serial to a
dataset image key (``wrist_left`` / ``wrist_right`` / ``agentview``). ``read()``
returns the most recent color frame per key as an ``HxWx3 uint8`` array.

``mock=True`` returns synthetic frames so the pipeline runs without hardware.

CLI: ``python -m workstation.lerobot_recorder.cameras --list`` prints connected
RealSense serials so you can fill them into the config.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from workstation.lerobot_recorder.config import CameraSpec, RecorderConfig

logger = logging.getLogger(__name__)


def _rs():
    """pyrealsense2, imported lazily so a machine without it can still run mock cameras."""
    import pyrealsense2 as rs

    return rs


def color_control_sensor(device):
    """The sensor that owns exposure/gain for a device's COLOR stream.

    Not always the color sensor: the D455 has a real ``RGB Camera`` endpoint, but the
    D405 has no RGB sensor at all — its color stream is derived from the ``Stereo
    Module``, so ``device.first_color_sensor()`` raises "Could not find requested
    sensor type!" and exposure must be set on the stereo sensor instead.

    Consequence for units: RealSense color sensors count exposure in 100 us steps
    (D455 RGB range 1..10000), while depth/stereo sensors count in 1 us steps (D405
    range 1..165000). The SAME numeric exposure means a 100x different time on a D405
    than on a D455 — never copy an exposure value between the two models.
    """
    try:
        return device.first_color_sensor()
    except Exception:
        pass
    import pyrealsense2 as rs

    for sensor in device.query_sensors():
        if sensor.supports(rs.option.exposure):
            return sensor
    raise RuntimeError("no sensor exposing exposure/gain controls")


class CameraManager:
    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        self.specs: List[CameraSpec] = cfg.cameras
        self._pipelines: Dict[str, object] = {}
        self._serials: Dict[str, str] = {}  # resolved serial per key (for reconnect)
        self._last: Dict[str, np.ndarray] = {}  # latest frame per key (written by the capture thread)
        self._stamp: Dict[str, float] = {}
        self._healthy: Dict[str, bool] = {}
        # Per-camera pinhole intrinsics, filled in on open; see _read_intrinsics.
        self._intrinsics: Dict[str, dict] = {}
        self._next_retry: Dict[str, float] = {}
        self._fps: Dict[str, int] = {}  # resolved color fps per key (decided once, reused on reconnect)
        self._frame_t = 0
        # Camera grabbing (and the blocking reconnect) runs on its own thread so a slow
        # frame or a pipe re-open never stalls the record loop or freezes the GUI; both
        # just read the latest cached frame via read().
        self._cap_lock = threading.Lock()
        self._cap_stop = threading.Event()
        self._cap_thread: "threading.Thread | None" = None

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        if self.cfg.mock:
            return
        import pyrealsense2 as rs

        available = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
        for spec in self.specs:
            serial = spec.serial or self._pick_unused(available)
            if not serial:
                # Don't abort the whole recorder for one missing camera: leave it
                # unhealthy (read() yields black) so the others still display/record.
                logger.warning("camera '%s': no RealSense serial (available: %s)", spec.key, sorted(available))
                self._healthy[spec.key] = False
                continue
            available.discard(serial)
            self._serials[spec.key] = serial
            try:
                self._open_pipe(spec, serial)
            except Exception as e:  # a bad profile / busy device shouldn't blank every camera
                self._healthy[spec.key] = False
                logger.warning("camera '%s' (%s) could not open: %s", spec.key, serial, e)

        self._cap_stop.clear()
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()

    def _capture_loop(self) -> None:
        """Continuously grab the latest frame per camera and cache it. Owns the camera
        API and the (blocking) reconnect, so consumers never block on hardware."""
        while not self._cap_stop.is_set():
            for spec in self.specs:
                if self._cap_stop.is_set():
                    break
                try:
                    pipe = self._pipelines.get(spec.key)
                    if pipe is None:
                        raise RuntimeError("pipeline not open")
                    frames = pipe.wait_for_frames(timeout_ms=1000)
                    img = np.asanyarray(frames.get_color_frame().get_data())  # HxWx3 uint8 (rgb8)
                    with self._cap_lock:
                        self._last[spec.key] = img
                        # When this frame arrived, so a consumer can ask how stale the image it
                        # just paired with an action was. The recorder samples the LATEST cached
                        # frame, so a consecutive pair of reads inside one camera period gets the
                        # same image -- which reads as a freeze in the video and is invisible
                        # without this. Nothing is written to the dataset; see age_s().
                        self._stamp[spec.key] = time.monotonic()
                    if not self._healthy.get(spec.key, True):  # was down -> recovered
                        logger.info("camera '%s' recovered", spec.key)
                    self._healthy[spec.key] = True
                except Exception:
                    if self._healthy.get(spec.key, True):  # log the drop once, on the transition
                        logger.warning("camera '%s' stopped delivering frames (link unstable?)", spec.key)
                    self._healthy[spec.key] = False
                    self._try_reconnect(spec)  # blocking pipe re-open, but off the record/GUI threads

    def _supported_color_fps(self, serial: str, spec: CameraSpec) -> list:
        """Color fps values the device actually offers at (width, height) in rgb8."""
        import pyrealsense2 as rs

        fps = set()
        for dev in rs.context().query_devices():
            if dev.get_info(rs.camera_info.serial_number) != serial:
                continue
            for sensor in dev.query_sensors():
                for p in sensor.get_stream_profiles():
                    try:
                        vp = p.as_video_stream_profile()
                        if (
                            p.stream_type() == rs.stream.color
                            and p.format() == rs.format.rgb8
                            and vp.width() == spec.width
                            and vp.height() == spec.height
                        ):
                            fps.add(p.fps())
                    except Exception:
                        pass
        return sorted(fps)

    def _resolve_fps(self, spec: CameraSpec, serial: str) -> int:
        """Decide the color fps for this camera ONCE (querying device profiles), cache
        it, and log any fallback a single time. Reconnects reuse the cached value, so
        we never re-query the RealSense context mid-stream (which can disrupt the other
        cameras and trigger a reconnect cascade)."""
        if spec.key in self._fps:
            return self._fps[spec.key]
        # Many RealSense models (D405/D455) cap 640x480 color at 30 fps, so a 60 fps
        # request fails with "Couldn't resolve requests". Fall back to the highest
        # supported fps <= the requested one instead of blanking the camera.
        fps = spec.fps
        supported = self._supported_color_fps(serial, spec)
        if supported and spec.fps not in supported:
            usable = [f for f in supported if f <= spec.fps] or supported
            fps = max(usable)
            logger.info(
                "camera '%s': %d fps unsupported; using %d fps (available: %s)", spec.key, spec.fps, fps, supported
            )
        self._fps[spec.key] = fps
        return fps

    def _open_pipe(self, spec: CameraSpec, serial: str) -> None:
        rs = _rs()

        fps = self._resolve_fps(spec, serial)
        pipe = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_device(serial)
        rs_cfg.enable_stream(rs.stream.color, spec.width, spec.height, rs.format.rgb8, fps)
        profile = pipe.start(rs_cfg)
        self._pipelines[spec.key] = pipe
        self._healthy[spec.key] = True
        self._read_intrinsics(spec, profile)
        self._apply_options(spec, profile)

    def _read_intrinsics(self, spec: CameraSpec, profile) -> None:
        """Record this camera's pinhole intrinsics, straight off the device.

        Needed to draw anything positioned in 3-D onto a frame -- a predicted gripper path,
        say. The camera knows its own fx/fy/cx/cy from factory calibration, so asking beats
        both a hand calibration and a guess from the field of view. Re-read on every open, and
        never fatal: a camera that will not report them still records perfectly well.
        """
        try:
            intr = profile.get_stream(_rs().stream.color).as_video_stream_profile().get_intrinsics()
            self._intrinsics[spec.key] = {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "cx": float(intr.ppx),
                "cy": float(intr.ppy),
                "width": int(intr.width),
                "height": int(intr.height),
            }
        except Exception as e:
            self._intrinsics.pop(spec.key, None)
            logger.warning("camera '%s': could not read intrinsics: %s", spec.key, e)

    def intrinsics(self, key: str) -> Optional[dict]:
        """``{fx, fy, cx, cy, width, height}`` for a camera, or None if it never reported them."""
        return self._intrinsics.get(key)

    def _apply_options(self, spec: CameraSpec, profile) -> None:
        """Push ``spec.options`` onto the camera's exposure-owning sensor (best effort).

        Runs on every open, so a reconnect restores the same settings. Auto-* toggles
        are applied first: writing `exposure` while auto-exposure is still on is a
        no-op on RealSense, so the order decides whether locking works at all. One
        unsupported option (model-dependent) must not blank the camera, so failures
        are logged and skipped individually.
        """
        if not spec.options:
            return
        import pyrealsense2 as rs

        try:
            sensor = color_control_sensor(profile.get_device())
        except Exception as e:
            logger.warning("camera '%s': no sensor to apply options to: %s", spec.key, e)
            return
        # auto-exposure/white-balance off before the manual values they gate
        ordered = sorted(spec.options.items(), key=lambda kv: not kv[0].startswith("enable_auto"))
        for name, value in ordered:
            try:
                option = getattr(rs.option, name)
            except AttributeError:
                logger.warning("camera '%s': unknown RealSense option '%s' (ignored)", spec.key, name)
                continue
            try:
                if not sensor.supports(option):
                    logger.warning("camera '%s': option '%s' unsupported by this model (ignored)", spec.key, name)
                    continue
                rng = sensor.get_option_range(option)
                if not (rng.min <= value <= rng.max):
                    logger.warning(
                        "camera '%s': option '%s'=%g out of range [%g, %g] (ignored)",
                        spec.key,
                        name,
                        value,
                        rng.min,
                        rng.max,
                    )
                    continue
                sensor.set_option(option, value)
                logger.info("camera '%s': %s = %g", spec.key, name, value)
            except Exception as e:
                logger.warning("camera '%s': could not set option '%s'=%g: %s", spec.key, name, value, e)

    def age_s(self) -> Dict[str, float]:
        """Seconds since each camera's cached frame arrived. Empty in mock mode.

        Diagnostic only: the recorder logs the worst of these per episode so "the image was N ms
        old when it was paired with that action" is answerable without a schema change.
        """
        if self.cfg.mock:
            return {}
        now = time.monotonic()
        with self._cap_lock:
            return {k: now - t for k, t in self._stamp.items()}

    def read(self) -> Dict[str, np.ndarray]:
        """Return {key: HxWx3 uint8 RGB} — the latest cached frame per camera, copied.
        Non-blocking: never touches the camera API (the capture thread owns that), so
        the record loop and GUI stay responsive even while a camera is reconnecting."""
        if self.cfg.mock:
            return self._mock_frames()
        out: Dict[str, np.ndarray] = {}
        with self._cap_lock:
            for spec in self.specs:
                img = self._last.get(spec.key)
                out[spec.key] = img.copy() if img is not None else np.zeros((spec.height, spec.width, 3), np.uint8)
        return out

    def _try_reconnect(self, spec: CameraSpec) -> None:
        """Throttled best-effort reconnection for a faulted camera (silent — the capture
        loop logs the down/recovered transitions; reuses the cached fps, no device re-query)."""
        now = time.monotonic()
        if now < self._next_retry.get(spec.key, 0.0):
            return
        self._next_retry[spec.key] = now + 2.0
        try:
            old = self._pipelines.pop(spec.key, None)
            if old is not None:
                try:
                    old.stop()
                except Exception:
                    pass
            self._open_pipe(spec, self._serials.get(spec.key, spec.serial))
        except Exception:
            pass  # stay unhealthy; will retry on the next interval

    @property
    def healthy(self) -> bool:
        """True iff every camera delivered a frame on the latest read (always True in mock)."""
        return all(self._healthy.values()) if self._healthy else True

    def stop(self) -> None:
        self._cap_stop.set()
        if self._cap_thread is not None:
            self._cap_thread.join(timeout=2.0)
            self._cap_thread = None
        for pipe in self._pipelines.values():
            try:
                pipe.stop()
            except Exception:
                pass
        self._pipelines.clear()

    @property
    def image_keys(self) -> List[str]:
        return [s.key for s in self.specs]

    def shape_of(self, key: str) -> tuple:
        spec = next(s for s in self.specs if s.key == key)
        return (spec.height, spec.width, 3)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _pick_unused(available: set) -> str:
        return next(iter(sorted(available)), "")

    def _mock_frames(self) -> Dict[str, np.ndarray]:
        self._frame_t += 1
        out = {}
        for spec in self.specs:
            img = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
            x = (self._frame_t * 4) % spec.width
            img[:, max(0, x - 8) : x + 8, :] = 200  # a moving bar so frames differ
            out[spec.key] = img
        return out


def detect_cameras(cfg: RecorderConfig) -> Dict:
    """Lightweight presence check for the configured cameras — no streaming.

    Returns ``{"found", "total", "missing"}`` so the setup page can show camera
    status before anything is opened. In mock mode all cameras report present.
    """
    total = len(cfg.cameras)
    if cfg.mock:
        return {"found": total, "total": total, "missing": []}
    try:
        import pyrealsense2 as rs

        available = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
    except Exception as e:
        return {"found": 0, "total": total, "missing": [c.key for c in cfg.cameras], "error": str(e)}

    pool = set(available)
    found, missing = 0, []
    for spec in cfg.cameras:
        if spec.serial:
            if spec.serial in available:
                found += 1
            else:
                missing.append(spec.key)
        elif pool:  # unpinned camera: any remaining device satisfies it
            pool.pop()
            found += 1
        else:
            missing.append(spec.key)
    return {"found": found, "total": total, "missing": missing}


def _list_devices() -> None:
    import pyrealsense2 as rs

    devs = rs.context().query_devices()
    if len(devs) == 0:
        print("No RealSense devices found.")
        return
    for d in devs:
        print(f"{d.get_info(rs.camera_info.name):24s}  serial={d.get_info(rs.camera_info.serial_number)}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="RealSense helper")
    p.add_argument("--list", action="store_true", help="list connected RealSense serials")
    args = p.parse_args()
    if args.list:
        _list_devices()
