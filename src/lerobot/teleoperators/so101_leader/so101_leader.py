#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import time
from queue import Empty, Queue

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_so101_leader import SO101LeaderConfig

logger = logging.getLogger(__name__)


PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logger.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")
    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:  # pragma: no cover - defensive
    keyboard = None
    PYNPUT_AVAILABLE = False
    logger.info(f"Could not import pynput: {e}")


class SO101Leader(Teleoperator):
    """
    SO-101 Leader Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = SO101LeaderConfig
    name = "so101_leader"

    def __init__(self, config: SO101LeaderConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        self._keyboard_listener = None
        self._keyboard_events: Queue[str] = Queue()
        self._pressed_keys: set[object] = set()
        self._intervention_active = False
        self._keyboard_warning_logged = False

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        self._start_keyboard_listener()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """Return teleoperation event flags collected from the keyboard."""
        events = {
            TeleopEvents.IS_INTERVENTION: self._intervention_active,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

        if not PYNPUT_AVAILABLE or self._keyboard_listener is None:
            return events

        terminate_episode = False
        success = False
        rerecord_episode = False

        while True:
            try:
                event = self._keyboard_events.get_nowait()
            except Empty:
                break

            if event == "toggle_intervention":
                self._intervention_active = not self._intervention_active
            elif event == "success":
                success = True
                terminate_episode = True
            elif event == "failure":
                success = False
                terminate_episode = True
            elif event == "rerecord":
                success = False
                rerecord_episode = True
                terminate_episode = True

        events[TeleopEvents.IS_INTERVENTION] = self._intervention_active
        events[TeleopEvents.SUCCESS] = success
        events[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        events[TeleopEvents.TERMINATE_EPISODE] = terminate_episode or success or rerecord_episode
        return events

    def _start_keyboard_listener(self) -> None:
        """Initialize keyboard listener used to capture teleop events."""
        if not PYNPUT_AVAILABLE or keyboard is None:
            if not self._keyboard_warning_logged:
                logger.info("pynput not available - teleop events will be disabled.")
                self._keyboard_warning_logged = True
            return

        if self._keyboard_listener is not None and self._keyboard_listener.is_alive():
            return

        try:
            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            )
            self._keyboard_listener.start()
        except Exception as exc:  # pragma: no cover - defensive
            if not self._keyboard_warning_logged:
                logger.error(f"Failed to start keyboard listener: {exc}")
                self._keyboard_warning_logged = True
            self._keyboard_listener = None
            return

        logger.info("Keyboard listener started for teleop event annotations.")
        logger.info("Keyboard controls - space: toggle intervention, s: success, esc: failure, r: rerecord.")

    def _stop_keyboard_listener(self) -> None:
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

        self._pressed_keys.clear()
        while True:
            try:
                self._keyboard_events.get_nowait()
            except Empty:
                break
        self._intervention_active = False

    def _on_key_press(self, key) -> None:
        if key in self._pressed_keys:
            return
        self._pressed_keys.add(key)

        try:
            if key == keyboard.Key.space:
                self._keyboard_events.put("toggle_intervention")
            elif key == keyboard.Key.esc:
                self._keyboard_events.put("failure")
            elif hasattr(key, "char") and key.char is not None:
                char = key.char.lower()
                if char == "s":
                    self._keyboard_events.put("success")
                elif char == "r":
                    self._keyboard_events.put("rerecord")
        except AttributeError:
            pass

    def _on_key_release(self, key) -> None:
        self._pressed_keys.discard(key)


    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self._stop_keyboard_listener()
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
