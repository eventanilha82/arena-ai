from __future__ import annotations

import hashlib
import json
import math
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping, Self


class Poc2ContractError(ValueError):
    """Raised when the POC 2 motion contract or one of its assets is invalid."""


@dataclass(frozen=True, slots=True)
class Poc2FrameMetadata:
    index: int
    file: str
    source_frame: int
    phase: str
    support: str | None
    source_duration_seconds: float
    effective_duration_seconds: float
    flight: bool
    visible_bbox: tuple[int, int, int, int]
    pelvis_x: float
    root_offset_x: float
    support_foot: tuple[float, float] | None
    support_target_x: float | None
    support_target_y: float | None
    foot_lock_error_px: float | None
    foot_lock_error_y_px: float | None
    sha256: str
    sheet_source_rect: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Poc2DirectionMetadata:
    uniform_code: str
    direction: str
    sheet_path: Path
    sheet_sha256: str
    source_sha256: str
    contact_offsets_px: tuple[float, float]
    frames: tuple[Poc2FrameMetadata, ...]


@dataclass(frozen=True, slots=True)
class Poc2MotionMetadata:
    contract_path: Path
    contract_sha256: str
    version: int
    status: str
    artifact: str
    animation: str
    source_provenance: str
    wordmark_provenance: str
    canvas_size_px: int
    canvas_ground_y_px: int
    frame_count: int
    sheet_columns: int
    sheet_rows: int
    source_cycle_seconds: float
    playback_speed: float
    cycle_seconds: float
    cycle_distance_px: float
    ball_canvas_size_px: int
    ball_visible_diameter_px: int
    ball_contact_gap_px: float
    ball_free_roll_excursion_px: float
    ball_visual_roll_radius_px: float
    ball_angular_response_seconds: float
    ball_max_angular_accel_deg_s2: float
    ball_rotation_integration_hz: float
    uniform_codes: tuple[str, ...]
    direction_names: tuple[str, str]


@dataclass(frozen=True, slots=True)
class Poc2FrameSample:
    metadata: Poc2FrameMetadata
    cycle_elapsed_seconds: float
    cycle_phase: float
    elapsed_in_frame_seconds: float
    frame_progress: float

    @property
    def index(self) -> int:
        return self.metadata.index

    @property
    def phase(self) -> str:
        return self.metadata.phase

    @property
    def flight(self) -> bool:
        return self.metadata.flight


@dataclass(frozen=True, slots=True)
class Poc2PlayerGeometry:
    canvas_size_px: float
    canvas_anchor_x_px: float
    canvas_ground_y_px: float
    scene_center_x: float
    scene_ground_y: float
    scene_left: float
    scene_top: float
    scene_size: float
    scale: float


@dataclass(frozen=True, slots=True)
class Poc2BallSample:
    canvas_center_x_px: float
    canvas_center_y_px: float
    canvas_ground_y_px: float
    canvas_radius_px: float
    canvas_diameter_px: float
    scene_center_x: float
    scene_center_y: float
    scene_ground_y: float
    scene_radius: float
    scene_diameter: float
    relative_offset_px: float
    signed_relative_offset_px: float
    touch_phase: float
    touch_slot: int
    free_roll: float
    forward_speed_px_s: float
    velocity_x_px_s: float
    scene_velocity_x_px_s: float
    rotation_degrees: float
    angular_velocity_deg_s: float


@dataclass(frozen=True, slots=True)
class Poc2DribbleSample:
    uniform_code: str
    direction: str
    elapsed_seconds: float
    cycle_index: int
    frame: Poc2FrameSample
    player: Poc2PlayerGeometry
    ball: Poc2BallSample

    @property
    def frame_index(self) -> int:
        return self.frame.index

    @property
    def phase(self) -> str:
        return self.frame.phase

    @property
    def flight(self) -> bool:
        return self.frame.flight


class CinematicDribbleRuntime:
    """Deterministic, render-agnostic runtime for the approved POC 2 motion."""

    VERSION: Final = 1
    ARTIFACT: Final = "arena_poc2_dribble_motion_contract"
    STATUS: Final = "promoted"
    ANIMATION: Final = "approved_poc2_variable_timing_no_morph_no_crossfade"
    SOURCE_PROVENANCE: Final = (
        "gpt_image_pose_matched_per_uniform_per_direction"
    )
    WORDMARK_PROVENANCE: Final = "native_gpt_image_jersey_pixels_no_overlay"

    CANVAS_SIZE_PX: Final = 320
    CANVAS_GROUND_Y_PX: Final = 306
    FRAME_COUNT: Final = 8
    SHEET_COLUMNS: Final = 4
    SHEET_ROWS: Final = 2
    SOURCE_CYCLE_SECONDS: Final = 0.8
    PLAYBACK_SPEED: Final = 0.5
    CYCLE_SECONDS: Final = 1.6
    CYCLE_DISTANCE_PX: Final = 300.0

    BALL_CANVAS_SIZE_PX: Final = 52
    BALL_VISIBLE_DIAMETER_PX: Final = 42
    BALL_CONTACT_GAP_PX: Final = 2.0
    BALL_FREE_ROLL_EXCURSION_PX: Final = 22.0
    BALL_VISUAL_ROLL_RADIUS_PX: Final = 27.0
    BALL_ANGULAR_RESPONSE_SECONDS: Final = 0.075
    BALL_MAX_ANGULAR_ACCEL_DEG_S2: Final = 4320.0
    BALL_ROTATION_INTEGRATION_HZ: Final = 240.0

    # The POC measured translational speed with a fixed 60 Hz backward probe.
    # Keeping it numeric here preserves that motion without consulting game FPS.
    VELOCITY_PROBE_HZ: Final = 60.0

    UNIFORM_CODES: Final = (
        "black",
        "blue",
        "burgundy",
        "gold",
        "green",
        "orange",
        "red",
        "sky",
        "white",
    )
    DIRECTION_NAMES: Final = ("right", "left")
    FRAME_PHASES: Final = (
        "contact_a",
        "compression_a",
        "toe_off_a",
        "flight_a",
        "contact_b",
        "compression_b",
        "toe_off_b",
        "flight_b",
    )
    SOURCE_FRAME_DURATIONS: Final = (
        0.11,
        0.19,
        0.04,
        0.06,
        0.13,
        0.18,
        0.04,
        0.05,
    )
    FRAME_FLIGHT: Final = (
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    )
    FRAME_SUPPORT: Final = (
        "front",
        "front",
        "back",
        None,
        "front",
        "front",
        "back",
        None,
    )

    _TOP_LEVEL_KEYS: Final = frozenset(
        {
            "version",
            "status",
            "artifact",
            "animation",
            "source_provenance",
            "wordmark_provenance",
            "canvas_size",
            "canvas_ground_y",
            "frame_count",
            "sheet_columns",
            "sheet_rows",
            "source_cycle_seconds",
            "playback_speed",
            "cycle_seconds",
            "cycle_distance_px",
            "ball_canvas_size_px",
            "ball_visible_diameter_px",
            "ball_contact_gap_px",
            "ball_free_roll_excursion_px",
            "ball_visual_roll_radius_px",
            "ball_angular_response_seconds",
            "ball_max_angular_accel_deg_s2",
            "ball_rotation_integration_hz",
            "uniforms",
        }
    )
    _DIRECTION_KEYS: Final = frozenset(
        {
            "sheet",
            "sheet_sha256",
            "source_sha256",
            "contact_offsets_px",
            "frames",
        }
    )
    _FRAME_KEYS: Final = frozenset(
        {
            "file",
            "source_frame",
            "root_offset_x",
            "phase",
            "support",
            "duration",
            "flight",
            "visible_bbox",
            "pelvis_x",
            "support_foot",
            "support_target_x",
            "support_target_y",
            "foot_lock_error_px",
            "foot_lock_error_y_px",
            "sha256",
        }
    )
    _PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
    _FLOAT_TOLERANCE: Final = 1e-9

    def __init__(
        self,
        contract_path: Path,
        *,
        verify_assets: bool = True,
    ) -> None:
        path = Path(contract_path).expanduser().resolve()
        if not path.is_file():
            raise Poc2ContractError(f"missing POC 2 motion contract: {path}")

        try:
            raw_contract = path.read_bytes()
            payload = json.loads(raw_contract.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Poc2ContractError(
                f"cannot read POC 2 motion contract: {path}"
            ) from exc

        root = self._mapping(payload, "contract")
        self._exact_keys(root, self._TOP_LEVEL_KEYS, "contract")
        self._validate_root_constants(root)

        directions = self._load_directions(
            path,
            root,
            verify_assets=verify_assets,
        )
        contract_sha256 = hashlib.sha256(raw_contract).hexdigest()
        self.metadata = Poc2MotionMetadata(
            contract_path=path,
            contract_sha256=contract_sha256,
            version=self.VERSION,
            status=self.STATUS,
            artifact=self.ARTIFACT,
            animation=self.ANIMATION,
            source_provenance=self.SOURCE_PROVENANCE,
            wordmark_provenance=self.WORDMARK_PROVENANCE,
            canvas_size_px=self.CANVAS_SIZE_PX,
            canvas_ground_y_px=self.CANVAS_GROUND_Y_PX,
            frame_count=self.FRAME_COUNT,
            sheet_columns=self.SHEET_COLUMNS,
            sheet_rows=self.SHEET_ROWS,
            source_cycle_seconds=self.SOURCE_CYCLE_SECONDS,
            playback_speed=self.PLAYBACK_SPEED,
            cycle_seconds=self.CYCLE_SECONDS,
            cycle_distance_px=self.CYCLE_DISTANCE_PX,
            ball_canvas_size_px=self.BALL_CANVAS_SIZE_PX,
            ball_visible_diameter_px=self.BALL_VISIBLE_DIAMETER_PX,
            ball_contact_gap_px=self.BALL_CONTACT_GAP_PX,
            ball_free_roll_excursion_px=self.BALL_FREE_ROLL_EXCURSION_PX,
            ball_visual_roll_radius_px=self.BALL_VISUAL_ROLL_RADIUS_PX,
            ball_angular_response_seconds=self.BALL_ANGULAR_RESPONSE_SECONDS,
            ball_max_angular_accel_deg_s2=(
                self.BALL_MAX_ANGULAR_ACCEL_DEG_S2
            ),
            ball_rotation_integration_hz=(
                self.BALL_ROTATION_INTEGRATION_HZ
            ),
            uniform_codes=self.UNIFORM_CODES,
            direction_names=self.DIRECTION_NAMES,
        )
        self.directions: Mapping[
            str, Mapping[str, Poc2DirectionMetadata]
        ] = MappingProxyType(
            {
                uniform_code: MappingProxyType(dict(uniform_directions))
                for uniform_code, uniform_directions in directions.items()
            }
        )

        self._rotation_lock = threading.RLock()
        self._cycle_checkpoints: dict[
            tuple[str, str], list[tuple[float, float]]
        ] = {}
        steps_per_cycle = self.CYCLE_SECONDS * self.BALL_ROTATION_INTEGRATION_HZ
        if not math.isclose(
            steps_per_cycle,
            round(steps_per_cycle),
            rel_tol=0.0,
            abs_tol=self._FLOAT_TOLERANCE,
        ):
            raise Poc2ContractError(
                "cycle_seconds must contain an integral number of 240 Hz steps"
            )
        self._steps_per_cycle = round(steps_per_cycle)
        self._integration_step_seconds = (
            1.0 / self.BALL_ROTATION_INTEGRATION_HZ
        )

    @classmethod
    def load(
        cls,
        contract_path: Path,
        *,
        verify_assets: bool = True,
    ) -> Self:
        return cls(contract_path, verify_assets=verify_assets)

    @property
    def uniform_codes(self) -> tuple[str, ...]:
        return self.metadata.uniform_codes

    @property
    def direction_names(self) -> tuple[str, str]:
        return self.metadata.direction_names

    def direction(
        self,
        uniform_code: str,
        left: bool,
    ) -> Poc2DirectionMetadata:
        if type(left) is not bool:
            raise TypeError("left must be a bool")
        try:
            uniform_directions = self.directions[uniform_code]
        except (KeyError, TypeError) as exc:
            available = ", ".join(self.uniform_codes)
            raise KeyError(
                f"unknown POC 2 uniform {uniform_code!r}; available: {available}"
            ) from exc
        return uniform_directions["left" if left else "right"]

    def sample(
        self,
        uniform_code: str,
        left: bool,
        elapsed: float,
        player_center_x: float,
        ground_y: float,
        scale: float = 1.0,
    ) -> Poc2DribbleSample:
        elapsed_seconds = self._finite_number(elapsed, "elapsed")
        center_x = self._finite_number(player_center_x, "player_center_x")
        scene_ground_y = self._finite_number(ground_y, "ground_y")
        uniform_scale = self._finite_number(scale, "scale")
        if elapsed_seconds < 0.0:
            raise ValueError("elapsed must be non-negative")
        if uniform_scale <= 0.0:
            raise ValueError("scale must be greater than zero")

        direction = self.direction(uniform_code, left)
        direction_sign = -1.0 if left else 1.0
        cycle_index, cycle_elapsed = self._split_elapsed(elapsed_seconds)
        frame_sample = self._frame_sample(direction, cycle_elapsed)

        relative_offset, touch_phase, touch_slot, free_roll = (
            self._relative_ball_offset(direction, elapsed_seconds)
        )
        forward_speed = self._forward_speed(direction, elapsed_seconds)
        angular_velocity, rotation_degrees = self._rotation_state(
            direction,
            elapsed_seconds,
            cycle_index,
            cycle_elapsed,
        )

        canvas_anchor_x = self.CANVAS_SIZE_PX / 2.0
        ball_radius = self.BALL_VISIBLE_DIAMETER_PX / 2.0
        signed_offset = direction_sign * relative_offset
        player_left = center_x - canvas_anchor_x * uniform_scale
        player_top = (
            scene_ground_y - self.CANVAS_GROUND_Y_PX * uniform_scale
        )
        ball_canvas_x = canvas_anchor_x + signed_offset
        ball_canvas_y = self.CANVAS_GROUND_Y_PX - ball_radius
        velocity_x = direction_sign * forward_speed

        player = Poc2PlayerGeometry(
            canvas_size_px=float(self.CANVAS_SIZE_PX),
            canvas_anchor_x_px=canvas_anchor_x,
            canvas_ground_y_px=float(self.CANVAS_GROUND_Y_PX),
            scene_center_x=center_x,
            scene_ground_y=scene_ground_y,
            scene_left=player_left,
            scene_top=player_top,
            scene_size=self.CANVAS_SIZE_PX * uniform_scale,
            scale=uniform_scale,
        )
        ball = Poc2BallSample(
            canvas_center_x_px=ball_canvas_x,
            canvas_center_y_px=ball_canvas_y,
            canvas_ground_y_px=float(self.CANVAS_GROUND_Y_PX),
            canvas_radius_px=ball_radius,
            canvas_diameter_px=float(self.BALL_VISIBLE_DIAMETER_PX),
            scene_center_x=center_x + signed_offset * uniform_scale,
            scene_center_y=scene_ground_y - ball_radius * uniform_scale,
            scene_ground_y=scene_ground_y,
            scene_radius=ball_radius * uniform_scale,
            scene_diameter=self.BALL_VISIBLE_DIAMETER_PX * uniform_scale,
            relative_offset_px=relative_offset,
            signed_relative_offset_px=signed_offset,
            touch_phase=touch_phase,
            touch_slot=touch_slot,
            free_roll=free_roll,
            forward_speed_px_s=forward_speed,
            velocity_x_px_s=velocity_x,
            scene_velocity_x_px_s=velocity_x * uniform_scale,
            rotation_degrees=rotation_degrees,
            angular_velocity_deg_s=angular_velocity,
        )
        return Poc2DribbleSample(
            uniform_code=uniform_code,
            direction=direction.direction,
            elapsed_seconds=elapsed_seconds,
            cycle_index=cycle_index,
            frame=frame_sample,
            player=player,
            ball=ball,
        )

    def _load_directions(
        self,
        contract_path: Path,
        root: Mapping[str, Any],
        *,
        verify_assets: bool,
    ) -> dict[str, dict[str, Poc2DirectionMetadata]]:
        uniforms = self._mapping(root["uniforms"], "contract.uniforms")
        if set(uniforms) != set(self.UNIFORM_CODES):
            raise Poc2ContractError(
                "contract.uniforms must contain exactly: "
                + ", ".join(self.UNIFORM_CODES)
            )

        result: dict[str, dict[str, Poc2DirectionMetadata]] = {}
        for uniform_code in self.UNIFORM_CODES:
            uniform_path = f"contract.uniforms.{uniform_code}"
            uniform = self._mapping(uniforms[uniform_code], uniform_path)
            self._exact_keys(uniform, {"directions"}, uniform_path)
            direction_payloads = self._mapping(
                uniform["directions"],
                f"{uniform_path}.directions",
            )
            if set(direction_payloads) != set(self.DIRECTION_NAMES):
                raise Poc2ContractError(
                    f"{uniform_path}.directions must contain right and left"
                )

            result[uniform_code] = {}
            for direction_name in self.DIRECTION_NAMES:
                result[uniform_code][direction_name] = self._load_direction(
                    contract_path,
                    uniform_code,
                    direction_name,
                    direction_payloads[direction_name],
                    verify_assets=verify_assets,
                )
        return result

    def _load_direction(
        self,
        contract_path: Path,
        uniform_code: str,
        direction_name: str,
        value: Any,
        *,
        verify_assets: bool,
    ) -> Poc2DirectionMetadata:
        location = (
            f"contract.uniforms.{uniform_code}.directions.{direction_name}"
        )
        payload = self._mapping(value, location)
        self._exact_keys(payload, self._DIRECTION_KEYS, location)

        expected_sheet = f"poc2_runner_{direction_name}_{uniform_code}.png"
        sheet_name = self._string(payload["sheet"], f"{location}.sheet")
        if sheet_name != expected_sheet or Path(sheet_name).name != sheet_name:
            raise Poc2ContractError(
                f"{location}.sheet must be {expected_sheet!r}"
            )
        sheet_path = (contract_path.parent / sheet_name).resolve()
        if sheet_path.parent != contract_path.parent:
            raise Poc2ContractError(f"{location}.sheet escapes its asset directory")

        sheet_sha256 = self._sha256(
            payload["sheet_sha256"],
            f"{location}.sheet_sha256",
        )
        source_sha256 = self._sha256(
            payload["source_sha256"],
            f"{location}.source_sha256",
        )
        if verify_assets:
            self._verify_sheet(sheet_path, sheet_sha256, location)

        contact_offsets = self._number_tuple(
            payload["contact_offsets_px"],
            2,
            f"{location}.contact_offsets_px",
        )
        if any(offset <= 0.0 for offset in contact_offsets):
            raise Poc2ContractError(
                f"{location}.contact_offsets_px must be positive"
            )

        frame_payloads = self._list(payload["frames"], f"{location}.frames")
        if len(frame_payloads) != self.FRAME_COUNT:
            raise Poc2ContractError(
                f"{location}.frames must contain {self.FRAME_COUNT} entries"
            )
        frames = tuple(
            self._load_frame(
                frame_value,
                frame_index,
                f"{location}.frames[{frame_index}]",
            )
            for frame_index, frame_value in enumerate(frame_payloads)
        )

        source_duration = sum(
            frame.source_duration_seconds for frame in frames
        )
        effective_duration = sum(
            frame.effective_duration_seconds for frame in frames
        )
        self._equal_number(
            source_duration,
            self.SOURCE_CYCLE_SECONDS,
            f"{location}.frames source duration",
        )
        self._equal_number(
            effective_duration,
            self.CYCLE_SECONDS,
            f"{location}.frames effective duration",
        )
        self._validate_contact_offsets(
            direction_name,
            contact_offsets,
            frames,
            location,
        )
        return Poc2DirectionMetadata(
            uniform_code=uniform_code,
            direction=direction_name,
            sheet_path=sheet_path,
            sheet_sha256=sheet_sha256,
            source_sha256=source_sha256,
            contact_offsets_px=(contact_offsets[0], contact_offsets[1]),
            frames=frames,
        )

    def _load_frame(
        self,
        value: Any,
        index: int,
        location: str,
    ) -> Poc2FrameMetadata:
        payload = self._mapping(value, location)
        self._exact_keys(payload, self._FRAME_KEYS, location)

        expected_file = f"frame_{index:02d}.png"
        file_name = self._string(payload["file"], f"{location}.file")
        if file_name != expected_file:
            raise Poc2ContractError(
                f"{location}.file must be {expected_file!r}"
            )
        source_frame = self._integer(
            payload["source_frame"],
            f"{location}.source_frame",
        )
        if source_frame != index:
            raise Poc2ContractError(
                f"{location}.source_frame must equal its frame index"
            )

        phase = self._string(payload["phase"], f"{location}.phase")
        if phase != self.FRAME_PHASES[index]:
            raise Poc2ContractError(
                f"{location}.phase must be {self.FRAME_PHASES[index]!r}"
            )
        flight = self._boolean(payload["flight"], f"{location}.flight")
        if flight is not self.FRAME_FLIGHT[index]:
            raise Poc2ContractError(
                f"{location}.flight is inconsistent with phase {phase!r}"
            )
        support_value = payload["support"]
        if support_value is not None:
            support_value = self._string(
                support_value,
                f"{location}.support",
            )
        if support_value != self.FRAME_SUPPORT[index]:
            raise Poc2ContractError(
                f"{location}.support is inconsistent with phase {phase!r}"
            )

        source_duration = self._number(
            payload["duration"],
            f"{location}.duration",
        )
        self._equal_number(
            source_duration,
            self.SOURCE_FRAME_DURATIONS[index],
            f"{location}.duration",
        )
        effective_duration = source_duration / self.PLAYBACK_SPEED

        visible_bbox = self._int_tuple(
            payload["visible_bbox"],
            4,
            f"{location}.visible_bbox",
        )
        bbox_x, bbox_y, bbox_width, bbox_height = visible_bbox
        if (
            bbox_x < 0
            or bbox_y < 0
            or bbox_width <= 0
            or bbox_height <= 0
            or bbox_x + bbox_width > self.CANVAS_SIZE_PX
            or bbox_y + bbox_height > self.CANVAS_SIZE_PX
        ):
            raise Poc2ContractError(
                f"{location}.visible_bbox falls outside the POC 2 canvas"
            )

        support_foot = self._optional_point(
            payload["support_foot"],
            f"{location}.support_foot",
        )
        support_target_x = self._optional_number(
            payload["support_target_x"],
            f"{location}.support_target_x",
        )
        support_target_y = self._optional_number(
            payload["support_target_y"],
            f"{location}.support_target_y",
        )
        foot_lock_error = self._optional_number(
            payload["foot_lock_error_px"],
            f"{location}.foot_lock_error_px",
        )
        foot_lock_error_y = self._optional_number(
            payload["foot_lock_error_y_px"],
            f"{location}.foot_lock_error_y_px",
        )
        support_values = (
            support_foot,
            support_target_x,
            support_target_y,
            foot_lock_error,
            foot_lock_error_y,
        )
        if flight and any(item is not None for item in support_values):
            raise Poc2ContractError(
                f"{location} is a flight frame and cannot define foot lock data"
            )
        if not flight and any(item is None for item in support_values):
            raise Poc2ContractError(
                f"{location} must define complete foot lock data"
            )

        column = index % self.SHEET_COLUMNS
        row = index // self.SHEET_COLUMNS
        return Poc2FrameMetadata(
            index=index,
            file=file_name,
            source_frame=source_frame,
            phase=phase,
            support=support_value,
            source_duration_seconds=source_duration,
            effective_duration_seconds=effective_duration,
            flight=flight,
            visible_bbox=visible_bbox,
            pelvis_x=self._number(payload["pelvis_x"], f"{location}.pelvis_x"),
            root_offset_x=self._number(
                payload["root_offset_x"],
                f"{location}.root_offset_x",
            ),
            support_foot=support_foot,
            support_target_x=support_target_x,
            support_target_y=support_target_y,
            foot_lock_error_px=foot_lock_error,
            foot_lock_error_y_px=foot_lock_error_y,
            sha256=self._sha256(payload["sha256"], f"{location}.sha256"),
            sheet_source_rect=(
                column * self.CANVAS_SIZE_PX,
                row * self.CANVAS_SIZE_PX,
                self.CANVAS_SIZE_PX,
                self.CANVAS_SIZE_PX,
            ),
        )

    def _validate_root_constants(self, root: Mapping[str, Any]) -> None:
        exact_strings = {
            "status": self.STATUS,
            "artifact": self.ARTIFACT,
            "animation": self.ANIMATION,
            "source_provenance": self.SOURCE_PROVENANCE,
            "wordmark_provenance": self.WORDMARK_PROVENANCE,
        }
        for key, expected in exact_strings.items():
            actual = self._string(root[key], f"contract.{key}")
            if actual != expected:
                raise Poc2ContractError(
                    f"contract.{key} must be {expected!r}"
                )

        exact_integers = {
            "version": self.VERSION,
            "canvas_size": self.CANVAS_SIZE_PX,
            "canvas_ground_y": self.CANVAS_GROUND_Y_PX,
            "frame_count": self.FRAME_COUNT,
            "sheet_columns": self.SHEET_COLUMNS,
            "sheet_rows": self.SHEET_ROWS,
            "ball_canvas_size_px": self.BALL_CANVAS_SIZE_PX,
            "ball_visible_diameter_px": self.BALL_VISIBLE_DIAMETER_PX,
        }
        for key, expected in exact_integers.items():
            actual = self._integer(root[key], f"contract.{key}")
            if actual != expected:
                raise Poc2ContractError(
                    f"contract.{key} must be {expected}"
                )

        exact_numbers = {
            "source_cycle_seconds": self.SOURCE_CYCLE_SECONDS,
            "playback_speed": self.PLAYBACK_SPEED,
            "cycle_seconds": self.CYCLE_SECONDS,
            "cycle_distance_px": self.CYCLE_DISTANCE_PX,
            "ball_contact_gap_px": self.BALL_CONTACT_GAP_PX,
            "ball_free_roll_excursion_px": (
                self.BALL_FREE_ROLL_EXCURSION_PX
            ),
            "ball_visual_roll_radius_px": self.BALL_VISUAL_ROLL_RADIUS_PX,
            "ball_angular_response_seconds": (
                self.BALL_ANGULAR_RESPONSE_SECONDS
            ),
            "ball_max_angular_accel_deg_s2": (
                self.BALL_MAX_ANGULAR_ACCEL_DEG_S2
            ),
            "ball_rotation_integration_hz": (
                self.BALL_ROTATION_INTEGRATION_HZ
            ),
        }
        for key, expected in exact_numbers.items():
            actual = self._number(root[key], f"contract.{key}")
            self._equal_number(actual, expected, f"contract.{key}")

        self._equal_number(
            self.SOURCE_CYCLE_SECONDS / self.PLAYBACK_SPEED,
            self.CYCLE_SECONDS,
            "effective POC 2 cycle",
        )
        if self.SHEET_COLUMNS * self.SHEET_ROWS != self.FRAME_COUNT:
            raise Poc2ContractError("POC 2 sheet grid does not fit eight frames")
        if self.CANVAS_GROUND_Y_PX >= self.CANVAS_SIZE_PX:
            raise Poc2ContractError("POC 2 ground line is outside the canvas")
        if self.BALL_VISIBLE_DIAMETER_PX > self.BALL_CANVAS_SIZE_PX:
            raise Poc2ContractError(
                "visible ball diameter exceeds its material canvas"
            )

    def _validate_contact_offsets(
        self,
        direction_name: str,
        contact_offsets: tuple[float, ...],
        frames: tuple[Poc2FrameMetadata, ...],
        location: str,
    ) -> None:
        direction_sign = 1.0 if direction_name == "right" else -1.0
        canvas_center = self.CANVAS_SIZE_PX / 2.0
        radius = self.BALL_VISIBLE_DIAMETER_PX / 2.0
        for contact_slot, frame_index in enumerate((0, 4)):
            bbox_x, _bbox_y, bbox_width, _bbox_height = (
                frames[frame_index].visible_bbox
            )
            forward_edge = (
                bbox_x + bbox_width
                if direction_name == "right"
                else bbox_x
            )
            forward_extent = direction_sign * (forward_edge - canvas_center)
            expected = forward_extent + radius + self.BALL_CONTACT_GAP_PX
            self._equal_number(
                contact_offsets[contact_slot],
                expected,
                f"{location}.contact_offsets_px[{contact_slot}]",
            )

    def _verify_sheet(
        self,
        sheet_path: Path,
        expected_sha256: str,
        location: str,
    ) -> None:
        if not sheet_path.is_file():
            raise Poc2ContractError(f"missing sheet for {location}: {sheet_path}")
        actual_sha256 = self._file_sha256(sheet_path)
        if actual_sha256 != expected_sha256:
            raise Poc2ContractError(
                f"SHA-256 mismatch for {location}.sheet: {sheet_path}"
            )
        width, height = self._png_dimensions(sheet_path)
        expected_size = (
            self.CANVAS_SIZE_PX * self.SHEET_COLUMNS,
            self.CANVAS_SIZE_PX * self.SHEET_ROWS,
        )
        if (width, height) != expected_size:
            raise Poc2ContractError(
                f"invalid sheet dimensions for {location}: "
                f"{width}x{height}, expected {expected_size[0]}x{expected_size[1]}"
            )

    def _frame_sample(
        self,
        direction: Poc2DirectionMetadata,
        cycle_elapsed: float,
    ) -> Poc2FrameSample:
        frame_start = 0.0
        selected = direction.frames[-1]
        for frame in direction.frames:
            frame_end = frame_start + frame.effective_duration_seconds
            if cycle_elapsed < frame_end:
                selected = frame
                break
            frame_start = frame_end

        elapsed_in_frame = max(0.0, cycle_elapsed - frame_start)
        progress = min(
            1.0,
            elapsed_in_frame / selected.effective_duration_seconds,
        )
        return Poc2FrameSample(
            metadata=selected,
            cycle_elapsed_seconds=cycle_elapsed,
            cycle_phase=cycle_elapsed / self.CYCLE_SECONDS,
            elapsed_in_frame_seconds=elapsed_in_frame,
            frame_progress=progress,
        )

    def _relative_ball_offset(
        self,
        direction: Poc2DirectionMetadata,
        elapsed: float,
    ) -> tuple[float, float, int, float]:
        cycle_elapsed = elapsed % self.CYCLE_SECONDS
        half_cycle = self.CYCLE_SECONDS / 2.0
        touch_slot = 0 if cycle_elapsed < half_cycle else 1
        touch_phase = (
            cycle_elapsed - touch_slot * half_cycle
        ) / half_cycle

        contact_start = direction.contact_offsets_px[touch_slot]
        contact_end = direction.contact_offsets_px[(touch_slot + 1) % 2]
        contact_track = contact_start + (
            contact_end - contact_start
        ) * touch_phase
        free_roll = math.sin(math.pi * touch_phase)
        relative_offset = (
            contact_track
            + self.BALL_FREE_ROLL_EXCURSION_PX * free_roll
        )
        return relative_offset, touch_phase, touch_slot, free_roll

    def _forward_speed(
        self,
        direction: Poc2DirectionMetadata,
        elapsed: float,
    ) -> float:
        probe_seconds = 1.0 / self.VELOCITY_PROBE_HZ
        relative_offset = self._relative_ball_offset(direction, elapsed)[0]
        previous_offset = self._relative_ball_offset(
            direction,
            elapsed - probe_seconds,
        )[0]
        player_speed = self.CYCLE_DISTANCE_PX / self.CYCLE_SECONDS
        return player_speed + (
            relative_offset - previous_offset
        ) / probe_seconds

    def _target_angular_velocity(
        self,
        direction: Poc2DirectionMetadata,
        elapsed: float,
    ) -> float:
        direction_sign = 1.0 if direction.direction == "right" else -1.0
        return (
            -direction_sign
            * self._forward_speed(direction, elapsed)
            / self.BALL_VISUAL_ROLL_RADIUS_PX
            * (180.0 / math.pi)
        )

    def _rotation_state(
        self,
        direction: Poc2DirectionMetadata,
        elapsed: float,
        cycle_index: int,
        cycle_elapsed: float,
    ) -> tuple[float, float]:
        angular_velocity, angle = self._cycle_checkpoint(
            direction,
            cycle_index,
        )
        full_steps, remainder = self._split_integration_steps(cycle_elapsed)
        if full_steps >= self._steps_per_cycle:
            angular_velocity, angle = self._cycle_checkpoint(
                direction,
                cycle_index + 1,
            )
            return angular_velocity, angle % 360.0

        for step_index in range(full_steps):
            sample_elapsed = (
                step_index + 1
            ) * self._integration_step_seconds
            angular_velocity, angle = self._advance_rotation_step(
                direction,
                angular_velocity,
                angle,
                sample_elapsed,
                self._integration_step_seconds,
            )
        if remainder > 0.0:
            sample_elapsed = (
                full_steps * self._integration_step_seconds + remainder
            )
            angular_velocity, angle = self._advance_rotation_step(
                direction,
                angular_velocity,
                angle,
                sample_elapsed,
                remainder,
            )
        return angular_velocity, angle % 360.0

    def _cycle_checkpoint(
        self,
        direction: Poc2DirectionMetadata,
        cycle_index: int,
    ) -> tuple[float, float]:
        key = (direction.uniform_code, direction.direction)
        with self._rotation_lock:
            checkpoints = self._cycle_checkpoints.get(key)
            if checkpoints is None:
                checkpoints = [
                    (
                        self._target_angular_velocity(direction, 0.0),
                        0.0,
                    )
                ]
                self._cycle_checkpoints[key] = checkpoints

            while len(checkpoints) <= cycle_index:
                angular_velocity, angle = checkpoints[-1]
                for step_index in range(self._steps_per_cycle):
                    sample_elapsed = (
                        step_index + 1
                    ) * self._integration_step_seconds
                    angular_velocity, angle = self._advance_rotation_step(
                        direction,
                        angular_velocity,
                        angle,
                        sample_elapsed,
                        self._integration_step_seconds,
                    )
                checkpoints.append((angular_velocity, angle))
            return checkpoints[cycle_index]

    def _advance_rotation_step(
        self,
        direction: Poc2DirectionMetadata,
        current_velocity: float,
        current_angle: float,
        sample_elapsed: float,
        dt: float,
    ) -> tuple[float, float]:
        target_velocity = self._target_angular_velocity(
            direction,
            sample_elapsed,
        )
        response = 1.0 - math.exp(
            -dt / self.BALL_ANGULAR_RESPONSE_SECONDS
        )
        requested_change = (target_velocity - current_velocity) * response
        maximum_change = self.BALL_MAX_ANGULAR_ACCEL_DEG_S2 * dt
        applied_change = max(
            -maximum_change,
            min(maximum_change, requested_change),
        )
        next_velocity = current_velocity + applied_change
        return next_velocity, current_angle + next_velocity * dt

    def _split_elapsed(self, elapsed: float) -> tuple[int, float]:
        cycle_index = int(elapsed // self.CYCLE_SECONDS)
        cycle_elapsed = elapsed - cycle_index * self.CYCLE_SECONDS
        if cycle_elapsed >= self.CYCLE_SECONDS:
            cycle_index += 1
            cycle_elapsed = 0.0
        return cycle_index, max(0.0, cycle_elapsed)

    def _split_integration_steps(self, duration: float) -> tuple[int, float]:
        exact_steps = duration * self.BALL_ROTATION_INTEGRATION_HZ
        nearest_steps = round(exact_steps)
        if math.isclose(
            exact_steps,
            nearest_steps,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            return nearest_steps, 0.0
        full_steps = math.floor(exact_steps)
        remainder = duration - full_steps * self._integration_step_seconds
        return full_steps, remainder

    @classmethod
    def _mapping(cls, value: Any, location: str) -> Mapping[str, Any]:
        if type(value) is not dict:
            raise Poc2ContractError(f"{location} must be an object")
        return value

    @classmethod
    def _list(cls, value: Any, location: str) -> list[Any]:
        if type(value) is not list:
            raise Poc2ContractError(f"{location} must be an array")
        return value

    @classmethod
    def _exact_keys(
        cls,
        value: Mapping[str, Any],
        expected: set[str] | frozenset[str],
        location: str,
    ) -> None:
        actual = set(value)
        if actual == set(expected):
            return
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise Poc2ContractError(
            f"{location} has invalid keys ({'; '.join(details)})"
        )

    @classmethod
    def _string(cls, value: Any, location: str) -> str:
        if type(value) is not str or not value:
            raise Poc2ContractError(f"{location} must be a non-empty string")
        return value

    @classmethod
    def _boolean(cls, value: Any, location: str) -> bool:
        if type(value) is not bool:
            raise Poc2ContractError(f"{location} must be a bool")
        return value

    @classmethod
    def _integer(cls, value: Any, location: str) -> int:
        if type(value) is not int:
            raise Poc2ContractError(f"{location} must be an integer")
        return value

    @classmethod
    def _number(cls, value: Any, location: str) -> float:
        if type(value) not in {int, float}:
            raise Poc2ContractError(f"{location} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise Poc2ContractError(f"{location} must be finite")
        return result

    @classmethod
    def _finite_number(cls, value: Any, location: str) -> float:
        if type(value) not in {int, float}:
            raise TypeError(f"{location} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{location} must be finite")
        return result

    @classmethod
    def _optional_number(
        cls,
        value: Any,
        location: str,
    ) -> float | None:
        if value is None:
            return None
        return cls._number(value, location)

    @classmethod
    def _number_tuple(
        cls,
        value: Any,
        length: int,
        location: str,
    ) -> tuple[float, ...]:
        items = cls._list(value, location)
        if len(items) != length:
            raise Poc2ContractError(
                f"{location} must contain {length} entries"
            )
        return tuple(
            cls._number(item, f"{location}[{index}]")
            for index, item in enumerate(items)
        )

    @classmethod
    def _int_tuple(
        cls,
        value: Any,
        length: int,
        location: str,
    ) -> tuple[int, ...]:
        items = cls._list(value, location)
        if len(items) != length:
            raise Poc2ContractError(
                f"{location} must contain {length} entries"
            )
        return tuple(
            cls._integer(item, f"{location}[{index}]")
            for index, item in enumerate(items)
        )

    @classmethod
    def _optional_point(
        cls,
        value: Any,
        location: str,
    ) -> tuple[float, float] | None:
        if value is None:
            return None
        point = cls._number_tuple(value, 2, location)
        return point[0], point[1]

    @classmethod
    def _sha256(cls, value: Any, location: str) -> str:
        digest = cls._string(value, location)
        if (
            len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Poc2ContractError(
                f"{location} must be a lowercase SHA-256 digest"
            )
        return digest

    @classmethod
    def _equal_number(
        cls,
        actual: float,
        expected: float,
        location: str,
    ) -> None:
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=cls._FLOAT_TOLERANCE,
        ):
            raise Poc2ContractError(
                f"{location} must be {expected}, got {actual}"
            )

    @classmethod
    def _file_sha256(cls, path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise Poc2ContractError(f"cannot read POC 2 sheet: {path}") from exc
        return digest.hexdigest()

    @classmethod
    def _png_dimensions(cls, path: Path) -> tuple[int, int]:
        try:
            with path.open("rb") as handle:
                header = handle.read(24)
        except OSError as exc:
            raise Poc2ContractError(f"cannot read POC 2 sheet: {path}") from exc
        if (
            len(header) != 24
            or header[:8] != cls._PNG_SIGNATURE
            or struct.unpack(">I", header[8:12])[0] != 13
            or header[12:16] != b"IHDR"
        ):
            raise Poc2ContractError(f"invalid PNG sheet: {path}")
        return struct.unpack(">II", header[16:24])


Poc2DribbleRuntime = CinematicDribbleRuntime


__all__ = [
    "CinematicDribbleRuntime",
    "Poc2BallSample",
    "Poc2ContractError",
    "Poc2DirectionMetadata",
    "Poc2DribbleRuntime",
    "Poc2DribbleSample",
    "Poc2FrameMetadata",
    "Poc2FrameSample",
    "Poc2MotionMetadata",
    "Poc2PlayerGeometry",
]
