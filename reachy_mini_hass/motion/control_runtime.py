"""Pose and control loop helpers for `MovementManager`."""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from ..core.config import Config
from .pose_composer import clamp_body_yaw, compose_poses, create_head_pose_matrix, extract_yaw_from_pose
from .state_machine import RobotState

if TYPE_CHECKING:
    from .movement_manager import MovementManager

logger = logging.getLogger(__name__)

# Deadbands for skipping redundant set_target sends. A send is skipped when
# every component moved less than its epsilon since the last successful send
# (subject to the idle keepalive below, so the daemon still sees traffic).
POSE_EPS = 1e-3  # Max element delta in 4x4 pose matrix
ANTENNA_EPS = 0.005  # Radians (~0.29 deg)
BODY_YAW_EPS = 0.005  # Radians (~0.29 deg)

# How long the WS must stay lost (with plain set_target retries failing)
# before we escalate to rebuilding the WebSocket client. Transient daemon
# stalls recover on their own within ~2s; an actually-closed socket never
# does, because the SDK's receive loop has exited and nothing re-opens it.
WS_REBUILD_AFTER_LOST_S = 10.0
# Minimum spacing between rebuild attempts while the connection stays lost,
# so a down daemon isn't hammered with back-to-back handshakes.
WS_REBUILD_RETRY_INTERVAL_S = 10.0


def update_emotion_move(manager: "MovementManager") -> tuple[np.ndarray, tuple[float, float], float] | None:
    with manager._emotion_move_lock:
        if manager._emotion_move is None:
            return None
        elapsed = manager._now() - manager._emotion_start_time
        if elapsed >= manager._emotion_move.duration:
            emotion_name = manager._emotion_move.emotion_name
            manager._emotion_move = None
            logger.info("Emotion move complete: %s", emotion_name)
            _fire_emotion_complete(manager, emotion_name)
            _return_to_rest_if_deep_sleep(manager)
            return None
        try:
            head_pose, antennas, body_yaw = manager._emotion_move.evaluate(elapsed)
            antenna_tuple = (float(antennas[0]), float(antennas[1]))
            clamped_body_yaw = clamp_body_yaw(float(body_yaw))
            return (head_pose, antenna_tuple, clamped_body_yaw)
        except Exception as e:
            logger.error("Error sampling emotion pose: %s", e)
            manager._emotion_move = None
            _fire_emotion_complete(manager, "<error>")
            _return_to_rest_if_deep_sleep(manager)
            return None


def _fire_emotion_complete(manager: "MovementManager", emotion_name: str) -> None:
    """Invoke the (optional) emotion-completion callback outside the lock."""
    callback = getattr(manager, "_on_emotion_complete_callback", None)
    if callback is None:
        return
    try:
        callback(emotion_name)
    except Exception as e:
        logger.debug("Emotion-complete callback error: %s", e)


def _return_to_rest_if_deep_sleep(manager: "MovementManager") -> None:
    """Smoothly transition back into the deep sleep rest pose if the user
    has the deep sleep mode active (idle_behavior disabled) and the robot
    is now in IDLE state. Called at the end of an emotion and on
    voice-phase return to IDLE so the head settles back to its resting
    position. No-op if the user has chosen the raised idle mode.
    """
    if manager.state.robot_state != RobotState.IDLE:
        return
    if manager._idle_behavior_enabled():
        return
    # Lazy import avoids a circular dependency between control_runtime
    # and idle_runtime at module load.
    from .idle_runtime import transition_or_apply_idle_rest_pose

    transition_or_apply_idle_rest_pose(manager, duration=2.0)


def compose_final_pose(manager: "MovementManager") -> tuple[np.ndarray, tuple[float, float], float]:
    primary_head = create_head_pose_matrix(
        x=manager.state.target_x,
        y=manager.state.target_y,
        z=manager.state.target_z,
        roll=manager.state.target_roll,
        pitch=manager.state.target_pitch,
        yaw=manager.state.target_yaw,
    )
    # Face tracking offsets are no longer composed here: the SDK daemon-side
    # head tracker blends its own aim into the IK output via start_head_tracking,
    # so our `set_target` calls provide only the "base" pose. Speech sway is
    # likewise handled by the daemon when ReachyMini.enable_wobbling() is used,
    # so only the local animation layer is composed here.
    anim_blend = manager.state.animation_blend
    secondary_head = create_head_pose_matrix(
        x=manager.state.anim_x * anim_blend,
        y=manager.state.anim_y * anim_blend,
        z=manager.state.anim_z * anim_blend,
        roll=manager.state.anim_roll * anim_blend,
        pitch=manager.state.anim_pitch * anim_blend,
        yaw=manager.state.anim_yaw * anim_blend,
    )
    final_head = compose_poses(primary_head, secondary_head)

    anim_antenna_left = manager.state.anim_antenna_left * anim_blend
    anim_antenna_right = manager.state.anim_antenna_right * anim_blend
    target_antenna_left = manager.state.target_antenna_left + anim_antenna_left
    target_antenna_right = manager.state.target_antenna_right + anim_antenna_right
    antenna_left, antenna_right = manager._antenna_controller.get_blended_positions(target_antenna_left, target_antenna_right)

    if manager.state.robot_state != RobotState.IDLE:
        manager._idle_antenna_smoothed = None
        manager._last_idle_antenna_update = 0.0

    final_head_yaw = extract_yaw_from_pose(final_head)
    if manager._user_body_yaw_override is not None:
        # User has manually set body yaw via the HA entity. Honor it
        # persistently instead of letting the auto-derivation (head-yaw
        # coupling + idle-with-no-face zero) overwrite it on every tick.
        # The override is cleared on transition out of IDLE (so voice
        # phases / face tracking re-couple body to head naturally).
        target_body_yaw = clamp_body_yaw(manager._user_body_yaw_override)
    else:
        target_body_yaw = clamp_body_yaw(final_head_yaw)
        if manager.state.robot_state == RobotState.IDLE and not manager.state.face_detected:
            target_body_yaw = 0.0

    now = manager._now()
    if manager._body_yaw_smoothed is None:
        manager._body_yaw_smoothed = target_body_yaw
        manager._last_body_yaw_update = now
    else:
        dt = max(1e-6, now - manager._last_body_yaw_update)
        max_rate_rad_s = math.radians(Config.motion.body_yaw_max_rate_deg_s)
        if manager.state.face_detected or manager.state.robot_state != RobotState.IDLE:
            max_rate_rad_s *= 1.15
        max_step = max_rate_rad_s * dt
        delta = target_body_yaw - manager._body_yaw_smoothed
        if abs(delta) <= Config.motion.body_yaw_deadband_rad:
            manager._body_yaw_smoothed = target_body_yaw
        else:
            step = max(-max_step, min(max_step, delta))
            manager._body_yaw_smoothed = clamp_body_yaw(manager._body_yaw_smoothed + step)
        manager._last_body_yaw_update = now

    return final_head, (antenna_right, antenna_left), manager._body_yaw_smoothed


def _pose_unchanged(manager: "MovementManager", head_pose: np.ndarray, antennas: tuple[float, float], body_yaw: float) -> bool:
    """True when the pose is within deadband of the last successfully sent one."""
    last_pose = manager._last_sent_head_pose
    last_antennas = manager._last_sent_antennas
    last_body_yaw = manager._last_sent_body_yaw
    if last_pose is None or last_antennas is None or last_body_yaw is None:
        return False
    return bool(
        np.max(np.abs(head_pose - last_pose)) < POSE_EPS
        and abs(antennas[0] - last_antennas[0]) < ANTENNA_EPS
        and abs(antennas[1] - last_antennas[1]) < ANTENNA_EPS
        and abs(body_yaw - last_body_yaw) < BODY_YAW_EPS
    )


def issue_control_command(manager: "MovementManager", head_pose: np.ndarray, antennas: tuple[float, float], body_yaw: float) -> None:
    if manager._draining_event.is_set() or manager._emotion_playing_event.is_set() or manager._robot_paused_event.is_set():
        return
    now = manager._now()

    # Cap actual WS sends to Config.motion.max_send_rate_hz (default 15Hz).
    # The control loop itself runs at 100Hz to keep animation and pose
    # composition fresh, but ~15Hz is enough to drive the motors smoothly.
    # Sending at the full loop rate has been observed (see ae13179's log
    # capture) to starve the daemon's outbound publishes under combined
    # audio + motion load; the SDK's WSClient then misses its 1s heartbeat
    # window, flips `_is_alive` to False, and every subsequent set_target
    # raises "Lost connection" — motion freezes while audio keeps working.
    min_send_interval = 1.0 / max(1.0, float(Config.motion.max_send_rate_hz))
    if not manager._connection_lost and (now - manager._last_send_time) < min_send_interval:
        return

    # When the pose has not meaningfully changed, drop to a slow keepalive
    # instead of re-sending identical targets. The keepalive send keeps the
    # command path exercised so a broken connection is still detected while
    # the robot sits parked (deep sleep spends hours at a static pose).
    if (
        not manager._connection_lost
        and _pose_unchanged(manager, head_pose, antennas, body_yaw)
        and (now - manager._last_send_time) < max(min_send_interval, float(Config.motion.idle_heartbeat_interval_s))
    ):
        return

    if manager._connection_lost:
        _maybe_start_ws_rebuild(manager, now)
        if now - manager._last_reconnect_attempt < manager._reconnect_attempt_interval:
            return
        manager._last_reconnect_attempt = now
        logger.debug("Attempting to send command after connection loss...")
    try:
        manager.robot.set_target(head=head_pose, antennas=list(antennas), body_yaw=body_yaw)
        manager._last_successful_command = now
        manager._consecutive_errors = 0
        manager._last_sent_head_pose = head_pose.copy()
        manager._last_sent_antennas = antennas
        manager._last_sent_body_yaw = body_yaw
        manager._last_send_time = now
        if manager._connection_lost:
            logger.info("✓ Connection to robot restored")
            manager._connection_lost = False
            manager._reconnect_attempt_interval = manager._reconnect_backoff_initial
            manager._suppressed_errors = 0
            _reenergize_motors_after_restore(manager)
    except Exception as e:
        error_msg = str(e)
        manager._consecutive_errors += 1
        is_connection_error = manager._is_connection_error(e)
        if is_connection_error:
            if not manager._connection_lost:
                if manager._consecutive_errors >= manager._max_consecutive_errors:
                    logger.warning(f"Connection unstable after {manager._consecutive_errors} errors: {error_msg}")
                    logger.warning("  Will retry connection every %.1fs...", manager._reconnect_attempt_interval)
                    manager._connection_lost = True
                    manager._connection_lost_since = now
                    manager._last_reconnect_attempt = now
                else:
                    manager._log_error_throttled(
                        f"Transient connection error ({manager._consecutive_errors}/{manager._max_consecutive_errors}): {error_msg}"
                    )
            else:
                manager._log_error_throttled(f"Connection still lost: {error_msg}")
                manager._reconnect_attempt_interval = min(
                    manager._reconnect_backoff_max,
                    manager._reconnect_attempt_interval * manager._reconnect_backoff_multiplier,
                )
        else:
            manager._log_error_throttled(f"Failed to set robot target: {error_msg}")


def _reenergize_motors_after_restore(manager: "MovementManager") -> None:
    """Re-enable motor torque after a connection outage.

    During an outage the per-motor watchdogs can drop torque on motors that
    need continuous correction (the antennas especially); once torque is off,
    set_target has no effect on them until enable_motors() re-energizes.
    enable_motors() is idempotent for motors that are still energized. The
    known trade-off: if the user disabled motors via HA right before a WS
    hiccup, this silently re-enables them — a narrow, recoverable window.
    """
    try:
        manager.robot.enable_motors()
        logger.info("Motors re-enabled after connection restore")
    except Exception as e:
        logger.warning("Failed to re-enable motors after restore: %s", e)


def _maybe_start_ws_rebuild(manager: "MovementManager", now: float) -> None:
    """Escalate to a full WebSocket rebuild when plain retries aren't enough.

    The SDK's WSClient never re-opens its socket: once the receive loop exits
    (actual close, daemon restart), the heartbeat stays dead and every
    set_target raises forever. Retrying set_target on that client — what the
    backoff path does — can only recover *stalls*, not closures. After the
    connection has been lost for WS_REBUILD_AFTER_LOST_S we rebuild the
    client in a background thread so the control loop never blocks on the
    connection handshake.
    """
    if now - manager._connection_lost_since < WS_REBUILD_AFTER_LOST_S:
        return
    thread = manager._ws_rebuild_thread
    if thread is not None and thread.is_alive():
        return
    if now - manager._last_ws_rebuild_attempt < WS_REBUILD_RETRY_INTERVAL_S:
        return
    manager._last_ws_rebuild_attempt = now
    thread = threading.Thread(
        target=_rebuild_ws_connection,
        args=(manager,),
        daemon=True,
        name="WSRebuild",
    )
    manager._ws_rebuild_thread = thread
    thread.start()


def _rebuild_ws_connection(manager: "MovementManager") -> None:
    """Replace the dead WSClient on manager.robot with a freshly connected one.

    Runs on a dedicated thread. On success the new client is swapped into
    `manager.robot.client` (all SDK calls go through that attribute, so the
    swap heals set_target, get_status, head tracking, etc. in one move) and
    daemon-side state that a restart would have reset is re-asserted.
    """
    old_client = manager.robot.client
    host, port = old_client.host, old_client.port
    logger.warning("Rebuilding daemon WebSocket connection to %s:%s ...", host, port)
    try:
        from reachy_mini.io.ws_client import WSClient

        new_client = WSClient(host=host, port=port)
        new_client.wait_for_connection(timeout=5.0)
    except Exception as e:
        logger.warning("WebSocket rebuild failed (%s); will retry while connection stays lost", e)
        return

    try:
        old_client.disconnect()
    except Exception:
        pass
    manager.robot.client = new_client

    # A daemon restart resets these; both calls are safe if it didn't.
    # The app constructs ReachyMini with the default automatic_body_yaw=True.
    try:
        manager.robot.set_automatic_body_yaw(True)
    except Exception as e:
        logger.debug("Failed to re-assert automatic body yaw after rebuild: %s", e)
    _reenergize_motors_after_restore(manager)

    manager._connection_lost = False
    manager._consecutive_errors = 0
    manager._reconnect_attempt_interval = manager._reconnect_backoff_initial
    manager._suppressed_errors = 0
    manager._last_successful_command = manager._now()
    logger.info("✓ Daemon WebSocket rebuilt — motion control restored")


def run_control_loop(manager: "MovementManager", *, max_control_dt_s: float) -> None:
    logger.info("Movement manager control loop started (%.1f Hz)", manager._control_loop_hz)
    last_time = manager._now()
    while not manager._stop_event.is_set():
        loop_start = manager._now()
        dt = min(max(0.0, loop_start - last_time), max_control_dt_s)
        last_time = loop_start
        try:
            manager._poll_commands()
            if manager._robot_paused_event.is_set():
                manager._robot_resumed_event.wait(timeout=0.5)
                continue
            emotion_pose = manager._update_emotion_move()
            if emotion_pose is not None:
                head_pose, antennas, body_yaw = emotion_pose
                manager._issue_control_command(head_pose, antennas, body_yaw)
            else:
                manager._update_action(dt)
                manager._update_animation(dt)
                manager._update_antenna_blend(dt)
                manager._update_animation_blend()
                manager._update_idle_look_around()
                head_pose, antennas, body_yaw = manager._compose_final_pose()
                manager._issue_control_command(head_pose, antennas, body_yaw)
            _publish_deep_sleep_state_if_changed(manager)
        except Exception as e:
            manager._log_error_throttled(f"Control loop error: {e}")
        sleep_time = max(0.0, manager._target_period - (manager._now() - loop_start))
        if sleep_time > 0:
            time.sleep(sleep_time)
    logger.info("Movement manager control loop stopped")


def _publish_deep_sleep_state_if_changed(manager: "MovementManager") -> None:
    """Detect transitions of `is_in_deep_sleep_state()` and fire the
    registered publish callback so the HA "Deep Sleep" switch entity can
    push its new value to Home Assistant. No-op when no callback is
    registered (i.e., before HA connects)."""
    callback = manager._deep_sleep_state_callback
    if callback is None:
        return
    current = manager.is_in_deep_sleep_state()
    if current == manager._last_published_deep_sleep_state:
        return
    manager._last_published_deep_sleep_state = current
    try:
        callback()
    except Exception:
        logger.exception("deep sleep state publish callback failed")
