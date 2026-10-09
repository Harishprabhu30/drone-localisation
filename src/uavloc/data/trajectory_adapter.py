from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_ID = "uavloc.trajectory.v1"

ALLOWED_ROLES = {
    "development",
    "validation",
    "blind_stress",
    "held_out",
}

ALLOWED_SOURCE_TYPES = {
    "video",
    "image_directory",
    "canonical_manifest",
}

ALLOWED_SAMPLING_MODES = {
    "uniform_time",
    "existing_manifest",
    "native",
}

ALLOWED_SIGNAL_SOURCES = {
    "assumed",
    "constant_metadata",
    "per_frame_file",
    "provider",
    "unavailable",
}

ALLOWED_REFERENCE_MODES = {
    "unavailable",
    "postfreeze_optional",
    "postfreeze_required",
}

CANONICAL_BLIND_COLUMNS = (
    "trajectory_id",
    "query_id",
    "frame_index",
    "timestamp_s",
    "image_path",
    "image_width",
    "image_height",
    "reference_available",
)

FORBIDDEN_BLIND_REFERENCE_COLUMNS = {
    "lat",
    "lon",
    "latitude",
    "longitude",
    "gps_lat",
    "gps_lon",
    "srt_lat",
    "srt_lon",
    "reference_x_m",
    "reference_y_m",
    "reference_cumulative_distance_m",
    "eval_ref_lat",
    "eval_ref_lon",
    "ground_truth_error",
    "error_m",
    "oracle_tile_identity",
    "hit_le_40m",
}


class TrajectorySpecError(ValueError):
    pass


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrajectorySpecError(f"{name} must be a mapping.")
    return dict(value)


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrajectorySpecError(f"{name} must be a non-empty string.")
    return value.strip()


def _require_positive(value: Any, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise TrajectorySpecError(f"{name} must be numeric.") from exc
    if out <= 0:
        raise TrajectorySpecError(f"{name} must be > 0.")
    return out


@dataclass(frozen=True)
class SignalSpec:
    name: str
    source: str
    config: dict[str, Any]

    @property
    def is_assumed(self) -> bool:
        return self.source == "assumed"

    @property
    def is_dynamic(self) -> bool:
        return self.source in {"per_frame_file", "provider"}

    @property
    def value(self) -> Any:
        return self.config.get("value")


@dataclass(frozen=True)
class ReferenceSpec:
    mode: str
    provider: str | None
    path: str | None
    config: dict[str, Any]

    @property
    def available(self) -> bool:
        return self.mode != "unavailable"

    @property
    def required(self) -> bool:
        return self.mode == "postfreeze_required"


@dataclass(frozen=True)
class TrajectorySpec:
    path: Path
    raw: dict[str, Any]
    trajectory_id: str
    role: str
    source_type: str
    signals: dict[str, SignalSpec]
    reference: ReferenceSpec

    @property
    def root(self) -> Path:
        return self.path.parent.resolve()

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    def signal(self, name: str) -> SignalSpec:
        if name not in self.signals:
            raise KeyError(name)
        return self.signals[name]

    def capability_summary(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_ID,
            "trajectory_id": self.trajectory_id,
            "role": self.role,
            "source_type": self.source_type,
            "reference_available": self.reference.available,
            "reference_mode": self.reference.mode,
            "signals": {
                name: {
                    "source": spec.source,
                    "assumed": spec.is_assumed,
                    "dynamic": spec.is_dynamic,
                }
                for name, spec in sorted(self.signals.items())
            },
        }

    def blind_runtime_contract(self) -> dict[str, Any]:
        source = dict(self.raw["source"])
        sampling = dict(self.raw["sampling"])
        map_cfg = dict(self.raw["map"])
        signals = {
            name: {
                "source": spec.source,
                **spec.config,
            }
            for name, spec in self.signals.items()
        }

        return {
            "schema": SCHEMA_ID,
            "trajectory": {
                "id": self.trajectory_id,
                "role": self.role,
            },
            "source": source,
            "sampling": sampling,
            "signals": signals,
            "map": map_cfg,
            "reference": {
                "available_to_localization": False,
                "mode": self.reference.mode,
            },
        }


def _validate_signal(name: str, payload: Any) -> SignalSpec:
    cfg = _require_mapping(payload, f"signals.{name}")
    source = _require_nonempty_string(
        cfg.pop("source", None),
        f"signals.{name}.source",
    )

    if source not in ALLOWED_SIGNAL_SOURCES:
        raise TrajectorySpecError(
            f"signals.{name}.source={source!r} is unsupported. "
            f"Allowed: {sorted(ALLOWED_SIGNAL_SOURCES)}"
        )

    if source == "assumed":
        if "value" not in cfg:
            raise TrajectorySpecError(
                f"signals.{name} with source='assumed' requires value."
            )

    elif source == "constant_metadata":
        if "value" not in cfg:
            raise TrajectorySpecError(
                f"signals.{name} with source='constant_metadata' requires value."
            )

    elif source == "per_frame_file":
        _require_nonempty_string(
            cfg.get("path"),
            f"signals.{name}.path",
        )
        _require_nonempty_string(
            cfg.get("column"),
            f"signals.{name}.column",
        )

    elif source == "provider":
        _require_nonempty_string(
            cfg.get("provider"),
            f"signals.{name}.provider",
        )

    elif source == "unavailable":
        if "value" in cfg:
            raise TrajectorySpecError(
                f"signals.{name} source='unavailable' cannot define value."
            )

    return SignalSpec(
        name=name,
        source=source,
        config=cfg,
    )


def _validate_reference(payload: Any) -> ReferenceSpec:
    cfg = _require_mapping(payload, "reference")
    mode = _require_nonempty_string(
        cfg.pop("mode", None),
        "reference.mode",
    )

    if mode not in ALLOWED_REFERENCE_MODES:
        raise TrajectorySpecError(
            f"reference.mode={mode!r} is unsupported. "
            f"Allowed: {sorted(ALLOWED_REFERENCE_MODES)}"
        )

    provider = cfg.pop("provider", None)
    path = cfg.pop("path", None)

    if mode == "unavailable":
        if provider not in (None, ""):
            raise TrajectorySpecError(
                "reference.provider must be omitted when reference.mode='unavailable'."
            )
        if path not in (None, ""):
            raise TrajectorySpecError(
                "reference.path must be omitted when reference.mode='unavailable'."
            )
        return ReferenceSpec(
            mode=mode,
            provider=None,
            path=None,
            config=cfg,
        )

    provider = _require_nonempty_string(
        provider,
        "reference.provider",
    )

    if path is not None:
        path = _require_nonempty_string(
            path,
            "reference.path",
        )

    if mode == "postfreeze_required" and path is None:
        raise TrajectorySpecError(
            "reference.path is required when reference.mode='postfreeze_required'."
        )

    return ReferenceSpec(
        mode=mode,
        provider=provider,
        path=path,
        config=cfg,
    )


def validate_trajectory_dict(raw: Mapping[str, Any]) -> dict[str, Any]:
    data = _require_mapping(raw, "trajectory spec")

    schema = _require_nonempty_string(
        data.get("schema"),
        "schema",
    )
    if schema != SCHEMA_ID:
        raise TrajectorySpecError(
            f"schema must be {SCHEMA_ID!r}, got {schema!r}."
        )

    trajectory = _require_mapping(
        data.get("trajectory"),
        "trajectory",
    )
    trajectory_id = _require_nonempty_string(
        trajectory.get("id"),
        "trajectory.id",
    )
    role = _require_nonempty_string(
        trajectory.get("role"),
        "trajectory.role",
    )
    if role not in ALLOWED_ROLES:
        raise TrajectorySpecError(
            f"trajectory.role={role!r} is unsupported. "
            f"Allowed: {sorted(ALLOWED_ROLES)}"
        )

    source = _require_mapping(
        data.get("source"),
        "source",
    )
    source_type = _require_nonempty_string(
        source.get("type"),
        "source.type",
    )
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise TrajectorySpecError(
            f"source.type={source_type!r} is unsupported. "
            f"Allowed: {sorted(ALLOWED_SOURCE_TYPES)}"
        )

    if source_type == "video":
        video = _require_mapping(
            source.get("video"),
            "source.video",
        )
        _require_nonempty_string(
            video.get("path"),
            "source.video.path",
        )
    elif source_type == "image_directory":
        _require_nonempty_string(
            source.get("path"),
            "source.path",
        )
    elif source_type == "canonical_manifest":
        _require_nonempty_string(
            source.get("path"),
            "source.path",
        )

    sampling = _require_mapping(
        data.get("sampling"),
        "sampling",
    )
    sampling_mode = _require_nonempty_string(
        sampling.get("mode"),
        "sampling.mode",
    )
    if sampling_mode not in ALLOWED_SAMPLING_MODES:
        raise TrajectorySpecError(
            f"sampling.mode={sampling_mode!r} is unsupported. "
            f"Allowed: {sorted(ALLOWED_SAMPLING_MODES)}"
        )
    if sampling_mode == "uniform_time":
        _require_positive(
            sampling.get("rate_hz"),
            "sampling.rate_hz",
        )

    signals_payload = _require_mapping(
        data.get("signals", {}),
        "signals",
    )
    signals = {
        str(name): _validate_signal(
            str(name),
            payload,
        )
        for name, payload in signals_payload.items()
    }

    map_cfg = _require_mapping(
        data.get("map"),
        "map",
    )
    _require_nonempty_string(
        map_cfg.get("id"),
        "map.id",
    )
    _require_nonempty_string(
        map_cfg.get("crs"),
        "map.crs",
    )

    variants = _require_mapping(
        map_cfg.get("variants"),
        "map.variants",
    )
    if not variants:
        raise TrajectorySpecError(
            "map.variants must contain at least one variant."
        )
    for name, payload in variants.items():
        variant = _require_mapping(
            payload,
            f"map.variants.{name}",
        )
        _require_nonempty_string(
            variant.get("tile_index"),
            f"map.variants.{name}.tile_index",
        )

    reference = _validate_reference(
        data.get(
            "reference",
            {"mode": "unavailable"},
        )
    )

    policy = _require_mapping(
        data.get("research_policy", {}),
        "research_policy",
    )
    if role == "held_out" and policy.get(
        "allow_method_changes_after_run",
        False,
    ):
        raise TrajectorySpecError(
            "held_out trajectories cannot allow method changes after the run."
        )

    return {
        "trajectory_id": trajectory_id,
        "role": role,
        "source_type": source_type,
        "signals": signals,
        "reference": reference,
    }


def load_trajectory_spec(
    path: str | Path,
    *,
    check_paths: bool = False,
) -> TrajectorySpec:
    config_path = Path(path)
    raw = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8",
        )
    )
    validated = validate_trajectory_dict(raw)

    spec = TrajectorySpec(
        path=config_path.resolve(),
        raw=dict(raw),
        trajectory_id=validated["trajectory_id"],
        role=validated["role"],
        source_type=validated["source_type"],
        signals=validated["signals"],
        reference=validated["reference"],
    )

    if check_paths:
        validate_declared_paths(spec)

    return spec


def validate_declared_paths(spec: TrajectorySpec) -> None:
    source = spec.raw["source"]

    path_values: list[tuple[str, str]] = []

    if spec.source_type == "video":
        path_values.append(
            ("source.video.path", source["video"]["path"])
        )
    elif spec.source_type in {
        "image_directory",
        "canonical_manifest",
    }:
        path_values.append(
            ("source.path", source["path"])
        )

    for name, signal in spec.signals.items():
        if signal.source == "per_frame_file":
            path_values.append(
                (f"signals.{name}.path", signal.config["path"])
            )

    for variant_name, variant in spec.raw["map"]["variants"].items():
        path_values.append(
            (
                f"map.variants.{variant_name}.tile_index",
                variant["tile_index"],
            )
        )
        if variant.get("descriptor_cache"):
            path_values.append(
                (
                    f"map.variants.{variant_name}.descriptor_cache",
                    variant["descriptor_cache"],
                )
            )

    if spec.reference.path:
        path_values.append(
            ("reference.path", spec.reference.path)
        )

    missing = []
    for label, value in path_values:
        path = spec.resolve_path(value)
        if not path.exists():
            missing.append(
                f"{label}: {path}"
            )

    if missing:
        raise FileNotFoundError(
            "Declared trajectory paths are missing:\n"
            + "\n".join(missing)
        )
