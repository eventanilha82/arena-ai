from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Final


POC_WIDTH: Final = 1280.0
POC_HEIGHT: Final = 760.0
POC_HEADER_HEIGHT: Final = 76.0
POC_CONTENT_HEIGHT: Final = POC_HEIGHT - POC_HEADER_HEIGHT
POC_GROUND_Y: Final = 656.0
POC_BALL_CANVAS_SIZE: Final = 52.0
POC_RUNNER_CANVAS_SIZE: Final = 288.0
POC_RUNNER_ROOT: Final = (144.0, 279.0)
POC_RUNNER_REFERENCE_VISIBLE_HEIGHT: Final = 229.0
POC_APPROVED_REFERENCE_VISIBLE_HEIGHT: Final = 244.0


@dataclass(frozen=True)
class PocViewport:
    x: float
    y: float
    width: float
    height: float
    scale: float
    content_x: float
    content_y: float

    @classmethod
    def fit(cls, rect: object) -> "PocViewport":
        x = float(getattr(rect, "x"))
        y = float(getattr(rect, "y"))
        width = float(getattr(rect, "w"))
        height = float(getattr(rect, "h"))
        scale = min(width / POC_WIDTH, height / POC_CONTENT_HEIGHT)
        content_width = POC_WIDTH * scale
        content_height = POC_CONTENT_HEIGHT * scale
        return cls(
            x=x,
            y=y,
            width=width,
            height=height,
            scale=scale,
            content_x=x + (width - content_width) * 0.5,
            content_y=y + (height - content_height) * 0.5,
        )

    def point(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.content_x + x * self.scale,
            self.content_y + (y - POC_HEADER_HEIGHT) * self.scale,
        )

    def size(self, width: float, height: float) -> tuple[int, int]:
        return (
            max(1, round(width * self.scale)),
            max(1, round(height * self.scale)),
        )

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> tuple[int, int, int, int]:
        left, top = self.point(x, y)
        scaled_width, scaled_height = self.size(width, height)
        return (
            round(left),
            round(top),
            scaled_width,
            scaled_height,
        )


@dataclass(frozen=True)
class PocSequenceSample:
    elapsed: float
    actor_visible: bool
    actor_source: int
    actor_frame: int
    actor_x: float
    actor_ground_y: float
    actor_shadow_x: float
    actor_shadow_y: float
    actor_shadow_w: float
    actor_shadow_h: float
    actor_shadow_alpha: float
    ball_visible: bool
    ball_x: float
    ball_y: float
    ball_ground_y: float
    ball_rotation: float
    ball_phase: int
    ball_trajectory_progress: float
    ball_shadow_x: float
    ball_shadow_y: float
    ball_shadow_w: float
    ball_shadow_h: float
    ball_shadow_alpha: float
    keeper_frame: int
    keeper_x: float
    keeper_y: float
    keeper_shadow_x: float
    keeper_shadow_y: float
    keeper_shadow_w: float
    keeper_shadow_h: float
    keeper_shadow_alpha: float
    goal_x: float
    goal_y: float
    goal_w: float
    goal_h: float
    net_strength: float
    ball_after_keeper: bool


@dataclass(frozen=True)
class PocSequence:
    key: str
    attack_direction: str
    keeper_direction: str
    goal_side: str
    profile: str
    outcome: str
    variant: int
    weight_bps: int
    release_seconds: float
    impact_seconds: float
    line_seconds: float
    duration_seconds: float
    audio_cues: tuple[tuple[str, float], ...]
    net_static_back: str | None
    net_static_back_sha256: str | None
    net_keyframes: tuple["PocNetKeyframe", ...]
    net_contact_frames: tuple["PocNetContactFrame", ...]
    actor_segments: tuple["PocActorSegment", ...]
    sample_times: tuple[float, ...]
    samples: tuple[tuple[float | int, ...], ...]


@dataclass(frozen=True)
class PocActorSegment:
    start_seconds: float
    end_seconds: float
    visible: bool
    source: int
    frame: int


@dataclass(frozen=True)
class PocNetKeyframe:
    seconds_after_impact: float
    source_elapsed_seconds: float
    back_roi: str
    back_roi_sha256: str
    back_roi_source_rect: tuple[int, int, int, int]
    back_roi_rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class PocNetContactFrame:
    seconds_after_impact: float
    front_contact: str
    front_contact_sha256: str
    front_contact_source_rect: tuple[int, int, int, int]
    front_contact_rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class PocLayerAsset:
    file: str
    sha256: str


class PocSequenceBank:
    VERSION: Final = 5
    GOAL_VARIANTS: Final = 3
    PLAN_SAMPLE_COUNT: Final = 1_000
    NET_KEYFRAME_HZ: Final = 30
    NET_KEYFRAME_COUNT: Final = 70
    NET_CONTACT_SAMPLE_HZ: Final = 60
    NET_CONTACT_END_SECONDS: Final = 0.18
    CONTINUOUS_INDICES: Final = tuple(
        index
        for index in range(37)
        if index
        not in {
            1,
            2,
            3,
            11,
            16,
            23,
            36,
        }
    )

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise RuntimeError(f"missing approved POC runtime contract: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != self.VERSION:
            raise RuntimeError(f"invalid approved POC runtime contract: {path}")
        if (
            payload.get("artifact")
            != "arena_cinematic_runtime_contract"
            or payload.get("status") != "promoted"
        ):
            raise RuntimeError(
                f"invalid promoted cinematic runtime contract: {path}"
            )
        if payload.get("sample_fields") != list(self.sample_fields()):
            raise RuntimeError(f"invalid POC runtime sample schema: {path}")
        if payload.get("plan_sample_count") != self.PLAN_SAMPLE_COUNT:
            raise RuntimeError(f"invalid POC plan sample count: {path}")
        if (
            payload.get("net_keyframe_hz") != self.NET_KEYFRAME_HZ
            or payload.get("net_keyframe_count")
            != self.NET_KEYFRAME_COUNT
            or payload.get("net_contact_sample_hz")
            != self.NET_CONTACT_SAMPLE_HZ
            or abs(
                float(payload.get("net_contact_end_seconds", -1.0))
                - self.NET_CONTACT_END_SECONDS
            )
            > 1e-9
        ):
            raise RuntimeError(f"invalid POC net cadence: {path}")
        goal_layers = payload.get("goal_layers")
        if not isinstance(goal_layers, dict):
            raise RuntimeError(f"missing POC goal layers: {path}")
        self.goal_base_layers = self._load_directional_layers(
            goal_layers.get("base"),
            "base",
        )
        self.goal_front_layers = self._load_directional_layers(
            goal_layers.get("front"),
            "front",
        )
        sequences = payload.get("sequences")
        expected_keys = self.expected_keys()
        if (
            not isinstance(sequences, dict)
            or set(sequences) != expected_keys
        ):
            raise RuntimeError(f"incomplete approved POC runtime contract: {path}")
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.sample_hz = int(payload["sample_hz"])
        self.ball_phase_labels = tuple(payload["ball_phase_labels"])
        self.sequences: dict[str, PocSequence] = {}
        for key, entry in sequences.items():
            samples = entry.get("samples")
            if not isinstance(samples, list) or len(samples) < 2:
                raise RuntimeError(f"empty POC runtime sequence: {key}")
            if any(len(sample) != len(self.sample_fields()) for sample in samples):
                raise RuntimeError(f"invalid POC runtime sample width: {key}")
            attack_direction = str(entry["attack_direction"])
            profile = str(entry["profile"])
            outcome = str(entry["outcome"])
            variant = int(entry["variant"])
            if key != self.key(
                attack_direction,
                profile,
                outcome,
                variant,
            ):
                raise RuntimeError(
                    f"incoherent POC runtime sequence key: {key}"
                )
            net_keyframes = tuple(
                self._load_net_keyframe(frame, key)
                for frame in entry.get("net_keyframes", [])
            )
            net_contact_frames = tuple(
                self._load_net_contact_frame(frame, key)
                for frame in entry.get("net_contact_frames", [])
            )
            if outcome == "goal":
                if (
                    not entry.get("net_static_back")
                    or len(net_keyframes) != self.NET_KEYFRAME_COUNT
                    or not net_contact_frames
                ):
                    raise RuntimeError(
                        f"incomplete POC net animation: {key}"
                    )
            elif (
                entry.get("net_static_back")
                or net_keyframes
                or net_contact_frames
            ):
                raise RuntimeError(
                    f"non-goal sequence carries net animation: {key}"
                )
            sample_rows = tuple(tuple(sample) for sample in samples)
            serialized_sample_times = tuple(
                float(sample[0])
                for sample in sample_rows
            )
            duration_seconds = float(entry["duration_seconds"])
            sample_times = tuple(
                min(
                    duration_seconds,
                    index / self.sample_hz,
                )
                for index in range(len(sample_rows))
            )
            if (
                abs(serialized_sample_times[0]) > 1e-9
                or any(
                    following <= previous
                    for previous, following in zip(
                        serialized_sample_times,
                        serialized_sample_times[1:],
                    )
                )
                or abs(
                    serialized_sample_times[-1]
                    - duration_seconds
                )
                > 1e-8
                or any(
                    abs(serialized - reconstructed) > 1e-8
                    for serialized, reconstructed in zip(
                        serialized_sample_times,
                        sample_times,
                    )
                )
            ):
                raise RuntimeError(
                    f"invalid POC runtime sample timeline: {key}"
                )
            if any(
                abs(
                    frame.seconds_after_impact
                    - index / self.NET_KEYFRAME_HZ
                )
                > 1e-5
                for index, frame in enumerate(net_keyframes)
            ):
                raise RuntimeError(
                    f"invalid POC net frame indices: {key}"
                )
            if any(
                following.seconds_after_impact
                <= previous.seconds_after_impact
                for previous, following in zip(
                    net_contact_frames,
                    net_contact_frames[1:],
                )
            ):
                raise RuntimeError(
                    f"invalid POC contact timeline: {key}"
                )
            self.sequences[key] = PocSequence(
                key=key,
                attack_direction=attack_direction,
                keeper_direction=str(entry["keeper_direction"]),
                goal_side=str(entry["goal_side"]),
                profile=profile,
                outcome=outcome,
                variant=variant,
                weight_bps=int(entry["weight_bps"]),
                release_seconds=float(entry["release_seconds"]),
                impact_seconds=float(entry["impact_seconds"]),
                line_seconds=float(entry["line_seconds"]),
                duration_seconds=duration_seconds,
                audio_cues=tuple(
                    (str(name), float(seconds))
                    for name, seconds in entry["audio_cues"]
                ),
                net_static_back=(
                    str(entry["net_static_back"])
                    if entry.get("net_static_back")
                    else None
                ),
                net_static_back_sha256=(
                    str(entry["net_static_back_sha256"])
                    if entry.get("net_static_back_sha256")
                    else None
                ),
                net_keyframes=net_keyframes,
                net_contact_frames=net_contact_frames,
                actor_segments=self._actor_segments(
                    sample_rows,
                    duration_seconds,
                ),
                sample_times=sample_times,
                samples=sample_rows,
            )
        self._validate_variant_weights(path)

    @staticmethod
    def _actor_segments(
        samples: tuple[tuple[float | int, ...], ...],
        duration_seconds: float,
    ) -> tuple[PocActorSegment, ...]:
        segments: list[PocActorSegment] = []
        start = float(samples[0][0])
        visible = bool(samples[0][1])
        source = int(samples[0][2])
        frame = int(samples[0][3])
        for sample in samples[1:]:
            current = (
                bool(sample[1]),
                int(sample[2]),
                int(sample[3]),
            )
            if current == (visible, source, frame):
                continue
            end = float(sample[0])
            segments.append(
                PocActorSegment(
                    start_seconds=start,
                    end_seconds=end,
                    visible=visible,
                    source=source,
                    frame=frame,
                )
            )
            start = end
            visible, source, frame = current
        segments.append(
            PocActorSegment(
                start_seconds=start,
                end_seconds=max(start, duration_seconds),
                visible=visible,
                source=source,
                frame=frame,
            )
        )
        return tuple(segments)

    @staticmethod
    def _load_directional_layers(
        payload: object,
        label: str,
    ) -> dict[str, PocLayerAsset]:
        if not isinstance(payload, dict) or set(payload) != {"left", "right"}:
            raise RuntimeError(f"incomplete POC {label} goal layers")
        result: dict[str, PocLayerAsset] = {}
        for side, entry in payload.items():
            if not isinstance(entry, dict):
                raise RuntimeError(f"invalid POC {label} layer: {side}")
            result[side] = PocLayerAsset(
                file=str(entry["file"]),
                sha256=str(entry["sha256"]),
            )
        return result

    @staticmethod
    def _load_net_keyframe(
        payload: object,
        sequence_key: str,
    ) -> PocNetKeyframe:
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"invalid POC net keyframe: {sequence_key}"
            )
        return PocNetKeyframe(
            seconds_after_impact=float(payload["seconds_after_impact"]),
            source_elapsed_seconds=float(
                payload["source_elapsed_seconds"]
            ),
            back_roi=str(payload["back_roi"]),
            back_roi_sha256=str(payload["back_roi_sha256"]),
            back_roi_source_rect=tuple(
                int(value)
                for value in payload["back_roi_source_rect"]
            ),
            back_roi_rect=tuple(int(value) for value in payload["back_roi_rect"]),
        )

    @staticmethod
    def _load_net_contact_frame(
        payload: object,
        sequence_key: str,
    ) -> PocNetContactFrame:
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"invalid POC contact frame: {sequence_key}"
            )
        return PocNetContactFrame(
            seconds_after_impact=float(
                payload["seconds_after_impact"]
            ),
            front_contact=str(payload["front_contact"]),
            front_contact_sha256=str(
                payload["front_contact_sha256"]
            ),
            front_contact_source_rect=tuple(
                int(value)
                for value in payload[
                    "front_contact_source_rect"
                ]
            ),
            front_contact_rect=tuple(
                int(value)
                for value in payload["front_contact_rect"]
            ),
        )

    @classmethod
    def expected_keys(cls) -> set[str]:
        keys: set[str] = set()
        for direction in ("right", "left"):
            for profile in ("high", "mid", "low"):
                for outcome in ("goal", "save", "wide"):
                    variants = (
                        range(cls.GOAL_VARIANTS)
                        if outcome == "goal"
                        else range(1)
                    )
                    for variant in variants:
                        keys.add(
                            cls.key(
                                direction,
                                profile,
                                outcome,
                                variant,
                            )
                        )
        return keys

    def _validate_variant_weights(self, path: Path) -> None:
        for direction in ("right", "left"):
            for profile in ("high", "mid", "low"):
                for outcome in ("goal", "save", "wide"):
                    variants = self.variants(
                        direction,
                        profile,
                        outcome,
                    )
                    if sum(sequence.weight_bps for sequence in variants) != 10_000:
                        raise RuntimeError(
                            "invalid POC variant weights: "
                            f"{direction}/{profile}/{outcome} in {path}"
                        )

    @staticmethod
    def sample_fields() -> tuple[str, ...]:
        return (
            "elapsed",
            "actor_visible",
            "actor_source",
            "actor_frame",
            "actor_x",
            "actor_ground_y",
            "actor_shadow_x",
            "actor_shadow_y",
            "actor_shadow_w",
            "actor_shadow_h",
            "actor_shadow_alpha",
            "ball_visible",
            "ball_x",
            "ball_y",
            "ball_ground_y",
            "ball_rotation",
            "ball_phase",
            "ball_trajectory_progress",
            "ball_shadow_x",
            "ball_shadow_y",
            "ball_shadow_w",
            "ball_shadow_h",
            "ball_shadow_alpha",
            "keeper_frame",
            "keeper_x",
            "keeper_y",
            "keeper_shadow_x",
            "keeper_shadow_y",
            "keeper_shadow_w",
            "keeper_shadow_h",
            "keeper_shadow_alpha",
            "goal_x",
            "goal_y",
            "goal_w",
            "goal_h",
            "net_strength",
            "ball_after_keeper",
        )

    @staticmethod
    def key(
        attack_direction: str,
        profile: str,
        outcome: str,
        variant: int = 0,
    ) -> str:
        return f"{attack_direction}:{profile}:{outcome}:v{variant}"

    def sequence(
        self,
        attack_direction: str,
        profile: str,
        outcome: str,
        variant: int = 0,
    ) -> PocSequence:
        key = self.key(
            attack_direction,
            profile,
            outcome,
            variant,
        )
        try:
            return self.sequences[key]
        except KeyError as exc:
            raise RuntimeError(f"missing approved POC sequence: {key}") from exc

    def variants(
        self,
        attack_direction: str,
        profile: str,
        outcome: str,
    ) -> tuple[PocSequence, ...]:
        variant_count = self.GOAL_VARIANTS if outcome == "goal" else 1
        return tuple(
            self.sequence(
                attack_direction,
                profile,
                outcome,
                variant,
            )
            for variant in range(variant_count)
        )

    def select_sequence(
        self,
        *,
        attack_direction: str,
        profile: str,
        outcome: str,
        match_seed: int,
        event_minute: int,
        side: str,
    ) -> PocSequence:
        variants = self.variants(
            attack_direction,
            profile,
            outcome,
        )
        identity = (
            f"{match_seed}:{event_minute}:{side}:{outcome}:"
            f"{profile}:poc7-variant-v1"
        ).encode("utf-8")
        draw = int.from_bytes(
            hashlib.sha256(identity).digest()[:4],
            "big",
        ) % 10_000
        accumulated = 0
        for sequence in variants:
            accumulated += sequence.weight_bps
            if draw < accumulated:
                return sequence
        return variants[-1]

    @staticmethod
    def select_profile(
        match_seed: int,
        event_minute: int,
        side: str,
        outcome: str,
    ) -> str:
        identity = f"{match_seed}:{event_minute}:{side}:{outcome}".encode("utf-8")
        index = int.from_bytes(hashlib.sha256(identity).digest()[:4], "big") % 3
        return ("high", "mid", "low")[index]

    @staticmethod
    def event_elapsed(
        raw_progress: float,
        sequence: PocSequence,
        event_window_seconds: float,
    ) -> float:
        if raw_progress <= 1.0:
            return max(0.0, raw_progress) * sequence.impact_seconds
        return min(
            sequence.duration_seconds,
            sequence.impact_seconds
            + (raw_progress - 1.0) * event_window_seconds,
        )

    def sample(
        self,
        sequence: PocSequence,
        elapsed: float,
    ) -> PocSequenceSample:
        clamped = max(0.0, min(sequence.duration_seconds, elapsed))
        following_index = min(
            len(sequence.samples) - 1,
            bisect_right(sequence.sample_times, clamped),
        )
        first_index = max(0, following_index - 1)
        first_seconds = sequence.sample_times[first_index]
        following_seconds = sequence.sample_times[following_index]
        blend = (
            0.0
            if following_index == first_index
            else (clamped - first_seconds)
            / max(1e-9, following_seconds - first_seconds)
        )
        first = sequence.samples[first_index]
        following = sequence.samples[following_index]
        discrete = first if blend < 0.5 else following
        values: list[float | int] = list(discrete)
        for index in self.CONTINUOUS_INDICES:
            values[index] = (
                float(first[index])
                + (float(following[index]) - float(first[index])) * blend
            )
        values[0] = clamped
        first_angle = float(first[15])
        following_angle = float(following[15])
        angle_delta = (
            (following_angle - first_angle + 180.0) % 360.0
        ) - 180.0
        values[15] = (first_angle + angle_delta * blend) % 360.0
        return PocSequenceSample(
            elapsed=float(values[0]),
            actor_visible=bool(values[1]),
            actor_source=int(values[2]),
            actor_frame=int(values[3]),
            actor_x=float(values[4]),
            actor_ground_y=float(values[5]),
            actor_shadow_x=float(values[6]),
            actor_shadow_y=float(values[7]),
            actor_shadow_w=float(values[8]),
            actor_shadow_h=float(values[9]),
            actor_shadow_alpha=float(values[10]),
            ball_visible=bool(values[11]),
            ball_x=float(values[12]),
            ball_y=float(values[13]),
            ball_ground_y=float(values[14]),
            ball_rotation=float(values[15]),
            ball_phase=int(values[16]),
            ball_trajectory_progress=float(values[17]),
            ball_shadow_x=float(values[18]),
            ball_shadow_y=float(values[19]),
            ball_shadow_w=float(values[20]),
            ball_shadow_h=float(values[21]),
            ball_shadow_alpha=float(values[22]),
            keeper_frame=int(values[23]),
            keeper_x=float(values[24]),
            keeper_y=float(values[25]),
            keeper_shadow_x=float(values[26]),
            keeper_shadow_y=float(values[27]),
            keeper_shadow_w=float(values[28]),
            keeper_shadow_h=float(values[29]),
            keeper_shadow_alpha=float(values[30]),
            goal_x=float(values[31]),
            goal_y=float(values[32]),
            goal_w=float(values[33]),
            goal_h=float(values[34]),
            net_strength=float(values[35]),
            ball_after_keeper=bool(values[36]),
        )

    def previous_sample(
        self,
        sequence: PocSequence,
        elapsed: float,
        seconds: float,
    ) -> PocSequenceSample:
        return self.sample(sequence, max(0.0, elapsed - seconds))

    @staticmethod
    def actor_frame_position(
        sequence: PocSequence,
        elapsed: float,
    ) -> float:
        clamped = max(0.0, min(sequence.duration_seconds, elapsed))
        segments = sequence.actor_segments
        segment_index = len(segments) - 1
        for index, segment in enumerate(segments):
            if clamped < segment.end_seconds or index == len(segments) - 1:
                segment_index = index
                break
        segment = segments[segment_index]
        position = float(segment.frame)
        if segment_index + 1 >= len(segments):
            return position
        following = segments[segment_index + 1]
        if (
            not segment.visible
            or not following.visible
            or following.source != segment.source
        ):
            return position
        following_position = float(following.frame)
        if (
            segment.source == 0
            and segment.frame == 7
            and following.frame == 0
        ):
            following_position = 8.0
        duration = max(
            1e-6,
            segment.end_seconds - segment.start_seconds,
        )
        progress = max(
            0.0,
            min(
                1.0,
                (clamped - segment.start_seconds) / duration,
            ),
        )
        return position + (following_position - position) * progress

    @staticmethod
    def nearest_net_keyframe(
        sequence: PocSequence,
        elapsed: float,
    ) -> PocNetKeyframe | None:
        if not sequence.net_keyframes:
            return None
        local = max(0.0, elapsed - sequence.impact_seconds)
        index = max(
            0,
            min(
                len(sequence.net_keyframes) - 1,
                round(local * PocSequenceBank.NET_KEYFRAME_HZ),
            ),
        )
        return sequence.net_keyframes[index]

    @staticmethod
    def nearest_net_contact_frame(
        sequence: PocSequence,
        elapsed: float,
    ) -> PocNetContactFrame | None:
        if not sequence.net_contact_frames:
            return None
        local = max(0.0, elapsed - sequence.impact_seconds)
        if local > PocSequenceBank.NET_CONTACT_END_SECONDS:
            return None
        return min(
            sequence.net_contact_frames,
            key=lambda frame: abs(
                frame.seconds_after_impact - local
            ),
        )
