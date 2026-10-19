"""Role 3 (@Sans): experimental design for the main study.

This module turns a study configuration into explicit, reproducible run
specifications. Every random object (rotation, initial direction) comes from
its own ``SeedSequence`` whose spawn key names the stage, purpose, study,
condition number, regime, pilot round, orientation and index. Streams for the
pilot, tuning and evaluation stages therefore never overlap, and any single
instance can be rebuilt from its key without replaying the rest of the study.

Run specifications are plain JSON-compatible dictionaries, so worker processes
rebuild their own problem from seeds instead of receiving pickled matrices.
The optimizers themselves are owned by modules 1, 2, 4 and 5 and are only
called through the shared ``run`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np

from .problems import QuadraticProblem, random_orthogonal, rotation_2d

# Spawn-key vocabulary. Changing any value changes every generated instance.
STAGE_CODES = {"pilot": 1, "tuning": 2, "evaluation": 3}
PURPOSE_ROTATION = 1
PURPOSE_INIT = 2
REGIME_CODES = {"none": 0, "transfer": 1, "retune": 2}
NORMALIZATIONS = ("loss", "unit_norm")


def integer_setting(value: Any, name: str, minimum: int = 0) -> int:
    """Validate a discrete setting without bool acceptance or numeric truncation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, excluding booleans")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def study_code(name: str) -> int:
    """Stable 32-bit code for a study's seed namespace."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")


def kappa_code(kappa: float) -> int:
    if isinstance(kappa, (bool, np.bool_)):
        raise ValueError("condition numbers cannot be booleans")
    if not isinstance(kappa, (int, float, np.integer, np.floating)):
        raise ValueError("condition numbers must be finite real numbers")
    try:
        value = float(kappa)
    except OverflowError as error:
        raise ValueError("condition number is outside the finite float64 range") from error
    if not math.isfinite(value) or value < 1 or value != round(value):
        raise ValueError("condition numbers must be finite integers >= 1 so seed keys stay exact")
    return int(round(value))


def seed_key(stage: str, purpose: int, study: str, kappa: float, regime: str = "none",
             round_index: int = 0, orientation_code: int = 0, index: int = 0) -> list[int]:
    if stage not in STAGE_CODES:
        raise ValueError(f"unknown stage {stage!r}")
    if regime not in REGIME_CODES:
        raise ValueError(f"unknown regime {regime!r}")
    purpose = integer_setting(purpose, "purpose")
    round_index = integer_setting(round_index, "round_index")
    orientation_code = integer_setting(orientation_code, "orientation_code")
    index = integer_setting(index, "index")
    return [STAGE_CODES[stage], purpose, study_code(study), kappa_code(kappa),
            REGIME_CODES[regime], round_index, orientation_code, index]


def generator(root_seed: int, key: list[int] | tuple[int, ...]) -> np.random.Generator:
    root_seed = integer_setting(root_seed, "root_seed")
    checked_key = tuple(integer_setting(value, "seed key") for value in key)
    return np.random.default_rng(np.random.SeedSequence(entropy=root_seed, spawn_key=checked_key))


def spectrum(dimension: int, kappa: float) -> np.ndarray:
    """Eigenvalues in [1/kappa, 1], log-spaced; dimension two gives diag(1/kappa, 1)."""
    dimension = integer_setting(dimension, "dimension", 1)
    kappa_code(kappa)
    if dimension == 1:
        if kappa != 1:
            raise ValueError("a one-dimensional spectrum has condition number one")
        return np.ones(1)
    values = np.geomspace(1.0 / float(kappa), 1.0, dimension)
    values[-1] = 1.0  # L = 1 exactly
    values[0] = 1.0 / float(kappa)
    return values


@dataclass(frozen=True)
class Orientation:
    """Identity, an exact planar angle, or a Haar rotation drawn from a seed key."""

    orientation_id: str
    kind: str
    degrees: float | None = None
    key: tuple[int, ...] | None = None

    def matrix(self, dimension: int, root_seed: int) -> np.ndarray:
        if self.kind == "identity":
            return np.eye(dimension)
        if self.kind == "angle":
            if dimension != 2:
                raise ValueError("angle orientations are only defined in two dimensions")
            return rotation_2d(np.deg2rad(self.degrees))
        if self.kind == "haar":
            return random_orthogonal(dimension, generator(root_seed, self.key))
        raise ValueError(f"unknown orientation kind {self.kind!r}")

    def to_json(self) -> dict[str, Any]:
        return {"orientation_id": self.orientation_id, "kind": self.kind,
                "degrees": self.degrees, "key": list(self.key) if self.key is not None else None}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Orientation":
        key = value.get("key")
        return cls(value["orientation_id"], value["kind"], value.get("degrees"),
                   tuple(key) if key is not None else None)


def canonical_initial(eigenvalues: np.ndarray, direction: np.ndarray,
                      normalization: str = "loss", scale: float = 1.0) -> np.ndarray:
    """Proposal eq. (3): z0 with f_I(z0)=scale (loss) or ||z0||=scale (unit_norm).

    With normalization='loss' and scale=1 this is exactly ``matched_pair``'s z0
    for the same Gaussian direction u.
    """
    d = np.asarray(eigenvalues, dtype=np.float64)
    u = np.asarray(direction, dtype=np.float64)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("initial scale must be finite and positive")
    if normalization == "loss":
        energy = float(u @ (d * u))
        if energy <= 0:
            raise ValueError("zero initial direction; choose another seed")
        return np.sqrt(2 * scale / energy) * u
    if normalization == "unit_norm":
        norm = float(np.linalg.norm(u))
        if norm == 0:
            raise ValueError("zero initial direction; choose another seed")
        return scale * u / norm
    raise ValueError(f"normalization must be one of {NORMALIZATIONS}")


def build_instance(spec: dict[str, Any], root_seed: int) -> tuple[QuadraticProblem, np.ndarray, np.ndarray]:
    """Rebuild (problem, initial point w0, canonical point z0) from a run specification."""
    dimension = integer_setting(spec["dimension"], "dimension", 1)
    eigenvalues = spectrum(dimension, spec["kappa"])
    orientation = Orientation.from_json(spec["orientation"])
    q = orientation.matrix(dimension, root_seed)
    init = spec["init"]
    direction = generator(root_seed, init["key"]).normal(size=dimension)
    z0 = canonical_initial(eigenvalues, direction, init.get("normalization", "loss"),
                           float(init.get("scale", 1.0)))
    return QuadraticProblem(eigenvalues, q), q @ z0, z0


# ---------------------------------------------------------------------------
# Study plans
# ---------------------------------------------------------------------------

def is_planar(study: dict[str, Any]) -> bool:
    return "angles_degrees" in study.get("evaluation", {})


def validate_study(study: dict[str, Any]) -> None:
    for field in ("name", "dimension", "condition_numbers", "evaluation"):
        if field not in study:
            raise ValueError(f"study is missing {field!r}")
    if not isinstance(study["name"], str) or not study["name"] or "|" in study["name"]:
        raise ValueError("study name must be a nonempty string without '|'")
    dimension = integer_setting(study["dimension"], "dimension", 1)
    if not isinstance(study["condition_numbers"], list) or not study["condition_numbers"]:
        raise ValueError("condition_numbers must be a nonempty list")
    for kappa in study["condition_numbers"]:
        # Keep the pre-existing namespace restriction, rather than claiming that
        # condition numbers are mathematically discrete. Integer-valued floats
        # remain supported and generate exactly the same original seed keys.
        kappa_code(kappa)
        if dimension == 1 and kappa != 1:
            raise ValueError("a one-dimensional study has condition number one")
    if len(set(study["condition_numbers"])) != len(study["condition_numbers"]):
        raise ValueError("condition_numbers must be unique")
    if not isinstance(study["evaluation"], dict):
        raise ValueError("study evaluation must be an object")
    if is_planar(study):
        if study["dimension"] != 2:
            raise ValueError("angle grids require dimension 2")
        angles = study["evaluation"]["angles_degrees"]
        if not isinstance(angles, list) or not angles or any(
            isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) for a in angles
        ):
            raise ValueError("angles_degrees must be a nonempty finite numeric list")
        if len({f"{a:g}" for a in angles}) != len(angles):
            raise ValueError("angles_degrees must have unique orientation identifiers")
        if 0 not in angles:
            raise ValueError("planar studies need angle 0, the diagonal reference")
        integer_setting(study["evaluation"]["inits"], "evaluation.inits", 1)
    else:
        integer_setting(study["evaluation"]["rotations"], "evaluation.rotations", 1)
        integer_setting(study["evaluation"]["inits_per_rotation"], "evaluation.inits_per_rotation", 1)
    if study.get("initial_normalization", "loss") not in NORMALIZATIONS:
        raise ValueError("initial_normalization must be 'loss' or 'unit_norm'")
    scale = study.get("initial_scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("initial_scale must be finite and positive")
    if "seed_study" in study and (not isinstance(study["seed_study"], str) or not study["seed_study"]):
        raise ValueError("seed_study must be a nonempty string")


def _seed_namespace(study: dict[str, Any]) -> str:
    """Studies may share instances (e.g. a unit-norm variant) through ``seed_study``."""
    return study.get("seed_study", study["name"])


def _init(study: dict[str, Any], init_id: str, key: list[int]) -> dict[str, Any]:
    return {"init_id": init_id, "key": key,
            "normalization": study.get("initial_normalization", "loss"),
            "scale": float(study.get("initial_scale", 1.0))}


def _random_orientations(study: dict[str, Any], kappa: float, root_seed: int, stage: str,
                         regime: str, round_index: int, count: int, prefix: str) -> list[Orientation]:
    """Identity plus ``count-1`` random orientations (uniform angle in 2D, Haar otherwise)."""
    ns = _seed_namespace(study)
    result = [Orientation("identity", "identity")]
    for j in range(1, count):
        key = tuple(seed_key(stage, PURPOSE_ROTATION, ns, kappa, regime, round_index, 0, j))
        if is_planar(study):
            degrees = float(generator(root_seed, key).uniform(0.0, 90.0))
            result.append(Orientation(f"{prefix}{j:02d}-deg{degrees:.4f}", "angle", degrees, key))
        else:
            result.append(Orientation(f"{prefix}{j:02d}", "haar", None, key))
    return result


def search_units(study: dict[str, Any], kappa: float, root_seed: int, *, stage: str, regime: str,
                 round_index: int, orientations: int, inits: int) -> list[dict[str, Any]]:
    """Orientation/initialization units for a transfer-style search (pilot or transfer tuning)."""
    ns = _seed_namespace(study)
    units = []
    prefix = "pilot-rot" if stage == "pilot" else "tune-rot"
    for j, orientation in enumerate(_random_orientations(study, kappa, root_seed, stage, regime,
                                                         round_index, orientations, prefix)):
        for i in range(inits):
            key = seed_key(stage, PURPOSE_INIT, ns, kappa, regime, round_index, j, i)
            units.append({"orientation": orientation.to_json(),
                          "init": _init(study, f"{stage[:4]}-o{j}-i{i:02d}", key)})
    return units


def evaluation_orientations(study: dict[str, Any], kappa: float) -> list[tuple[Orientation, int]]:
    """Evaluation orientations with their orientation code (0 = diagonal reference)."""
    ns = _seed_namespace(study)
    if is_planar(study):
        return [(Orientation("identity" if a == 0 else f"deg{a:g}", "identity" if a == 0 else "angle",
                             None if a == 0 else float(a)), k)
                for k, a in enumerate(study["evaluation"]["angles_degrees"])]
    result = [(Orientation("identity", "identity"), 0)]
    for r in range(int(study["evaluation"]["rotations"])):
        key = tuple(seed_key("evaluation", PURPOSE_ROTATION, ns, kappa, "none", 0, 0, r))
        result.append((Orientation(f"haar{r:02d}", "haar", None, key), r + 1))
    return result


def retune_units(study: dict[str, Any], kappa: float, orientation: Orientation, code: int,
                 inits: int) -> list[dict[str, Any]]:
    """Tuning units for one evaluation orientation in the retuning regime.

    Planar studies share retuning initial directions across angles (mirroring the
    shared evaluation directions); higher-dimensional studies draw them per orientation.
    """
    ns = _seed_namespace(study)
    scope = 0 if is_planar(study) else code
    return [{"orientation": orientation.to_json(),
             "init": _init(study, f"retune-s{scope}-i{i:02d}",
                           seed_key("tuning", PURPOSE_INIT, ns, kappa, "retune", 0, scope, i))}
            for i in range(inits)]


def evaluation_units(study: dict[str, Any], kappa: float) -> list[dict[str, Any]]:
    """Every evaluation run unit, each annotated with its diagonal reference.

    Planar: ten shared directions; angle 0 is the reference for every angle and
    is not duplicated. Rotation studies: each (rotation r, direction i) has its own
    z0; the identity run with that same z0 is its reference. ``cluster_id`` is the
    resampling unit planned in proposal Sec. 5.4.
    """
    ns = _seed_namespace(study)
    tag = f"{study['name']}|k{kappa_code(kappa)}"
    units = []
    if is_planar(study):
        for orientation, _ in evaluation_orientations(study, kappa):
            for i in range(int(study["evaluation"]["inits"])):
                init_id = f"eval-i{i:02d}"
                units.append({
                    "orientation": orientation.to_json(),
                    "init": _init(study, init_id, seed_key("evaluation", PURPOSE_INIT, ns, kappa, "none", 0, 0, i)),
                    "role": "reference" if orientation.kind == "identity" else "rotated",
                    "pair_id": f"{tag}|{orientation.orientation_id}|{init_id}",
                    "reference_key": f"{tag}|identity|{init_id}",
                    "cluster_id": f"{tag}|{init_id}",
                })
        return units
    orientations = evaluation_orientations(study, kappa)
    identity = orientations[0][0]
    for orientation, code in orientations[1:]:
        for i in range(int(study["evaluation"]["inits_per_rotation"])):
            init_id = f"eval-r{code - 1:02d}-i{i:02d}"
            init = _init(study, init_id, seed_key("evaluation", PURPOSE_INIT, ns, kappa, "none", 0, code, i))
            pair = f"{tag}|{orientation.orientation_id}|{init_id}"
            reference_key = f"{tag}|identity|{init_id}"
            for unit_orientation, role in ((identity, "reference"), (orientation, "rotated")):
                units.append({"orientation": unit_orientation.to_json(), "init": init, "role": role,
                              "pair_id": pair, "reference_key": reference_key,
                              "cluster_id": f"{tag}|{orientation.orientation_id}"})
    return units


def theory_parameters(algorithm: str, eigenvalues: np.ndarray) -> dict[str, float] | None:
    """Spectral-information references, proposal eqs. (7) and (10); not equal-budget tuning."""
    mu, large = float(np.min(eigenvalues)), float(np.max(eigenvalues))
    if algorithm == "gd":
        return {"learning_rate": 2 / (large + mu)}
    if algorithm == "heavy_ball":
        a, b = math.sqrt(large), math.sqrt(mu)
        return {"learning_rate": 4 / (a + b) ** 2, "beta": ((a - b) / (a + b)) ** 2}
    return None
