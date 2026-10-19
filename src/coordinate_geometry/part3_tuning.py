"""Role 3 (@Sans): learning-rate grids, the selection rule and pilot range revision.

Pure functions only; ``part3_experiments`` supplies the runs. The rules follow
proposal Sec. 5.1-5.2:

* every algorithm has the same number of log-spaced candidates (16 by default);
* the score of a candidate is the median of log10 max(r_B, floor) over its tuning
  runs, r_B = f(w_B) / f(w_0) at the tuning budget B;
* any failed tuning run (non-finite or explosive) gives the candidate the worst
  score; a run that merely misses the threshold is *not* a failure;
* ties go to the smaller learning rate; if every candidate fails, the selection
  is recorded as a tuning failure instead of a chosen rate;
* a selected edge candidate is a boundary optimum only if it beats its
  neighbour; a tied floor plateau that starts at the edge is not expanded;
* GD and heavy-ball grids stay strictly inside their analytical stability
  intervals, so an optimum at a stability-capped upper end is reported as
  ``stability_capped`` and never expanded past the cap.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

FAILED_STATUSES = ("nonfinite", "explosive")


def stability_cap(algorithm: str, params: dict[str, Any], largest_eigenvalue: float = 1.0) -> float | None:
    """Open upper bound of the stable step-size interval, or None when not spectrum-free.

    GD: eta < 2/L (proposal eq. 6). Heavy ball: eta < 2(1+beta)/L (Sec. 1.3).
    Diagonal-preconditioned GD and the Adam variants have no common analytical cap.
    """
    if algorithm == "gd":
        return 2.0 / largest_eigenvalue
    if algorithm == "heavy_ball":
        return 2.0 * (1.0 + float(params.get("beta", 0.9))) / largest_eigenvalue
    return None


def candidate_grid(low: float, high: float, count: int) -> list[float]:
    if not (math.isfinite(low) and math.isfinite(high)) or not 0 < low < high:
        raise ValueError("a learning-rate range needs 0 < low < high")
    if not isinstance(count, int) or count < 2:
        raise ValueError("at least two candidates are required")
    grid = np.geomspace(low, high, count)
    grid[0], grid[-1] = low, high
    return [float(x) for x in grid]


def check_range(algorithm: str, params: dict[str, Any], low: float, high: float,
                cap_fraction: float) -> None:
    cap = stability_cap(algorithm, params)
    if cap is not None and high > cap * cap_fraction * (1 + 1e-12):
        raise ValueError(f"{algorithm}: upper rate {high} exceeds {cap_fraction} x stability cap {cap}")


def run_failed(status: str) -> bool:
    return status in FAILED_STATUSES


def candidate_score(relative_losses: Sequence[float | None], statuses: Sequence[str],
                    floor: float = 1e-16) -> float:
    """Median of log10 max(r_B, floor); +inf (worst) if any run failed."""
    if len(relative_losses) != len(statuses) or not statuses:
        raise ValueError("need one status per tuning run")
    if any(run_failed(s) for s in statuses):
        return math.inf
    values = []
    for r in relative_losses:
        if r is None or not math.isfinite(r):
            raise ValueError("a non-failed tuning run must have a finite relative loss")
        values.append(math.log10(max(r, floor)))
    return float(np.median(values))


TIE_BREAKS = ("smaller_rate", "earliest_attainment")


def select_rate(rates: Sequence[float], scores: Sequence[float], tie_atol: float = 1e-12,
                attainment: Sequence[float] | None = None, tie_break: str = "smaller_rate") -> dict[str, Any]:
    """Apply the selection rule to scores aligned with ascending ``rates``.

    ``tie_break='smaller_rate'`` is the proposal's rule. ``'earliest_attainment'``
    first breaks score ties by the smaller median first-attainment time
    (``attainment``, inf when not reached), then by the smaller rate.
    """
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {TIE_BREAKS}")
    if tie_break == "earliest_attainment" and (attainment is None or len(attainment) != len(rates)):
        raise ValueError("earliest_attainment needs one median attainment per candidate")
    if len(rates) != len(scores) or not rates:
        raise ValueError("rates and scores must align")
    if any(b <= a for a, b in zip(rates, rates[1:])):
        raise ValueError("rates must be strictly ascending so ties favour the smaller rate")
    finite = [i for i, s in enumerate(scores) if math.isfinite(s)]
    n_failed = len(rates) - len(finite)
    if not finite:
        return {"tuning_failure": True, "selected_index": None, "selected_rate": None,
                "selected_score": None, "boundary": None, "edge_plateau": None,
                "candidates_failed": n_failed, "tied_candidates": 0}
    best = min(scores[i] for i in finite)
    tied = [i for i in finite if scores[i] <= best + tie_atol]

    def rank(i: int) -> tuple[float, int]:
        return (attainment[i], i) if tie_break == "earliest_attainment" else (0.0, i)

    index = min(tied, key=rank)
    edge = "lower" if index == 0 else "upper" if index == len(rates) - 1 else None
    # An edge counts as a boundary optimum only if it is strictly better than its
    # neighbour. When a tied plateau (typically candidates saturating the loss
    # floor) merely begins at the edge, the optimum set extends into the grid, so
    # widening the range cannot locate it; this is recorded as ``edge_plateau``.
    neighbour = 1 if edge == "lower" else len(rates) - 2 if edge == "upper" else None
    strict = edge is not None and (neighbour not in tied or rank(neighbour)[0] > rank(index)[0])
    return {"tuning_failure": False, "selected_index": index, "selected_rate": float(rates[index]),
            "selected_score": float(scores[index]), "boundary": edge if strict else None,
            "edge_plateau": None if strict else edge, "candidates_failed": n_failed,
            "tied_candidates": len(tied)}


def revise_range(algorithm: str, params: dict[str, Any], low: float, high: float,
                 boundary_hits: set[str], factor: float, cap_fraction: float) -> dict[str, Any]:
    """Expand a range whose pilot optimum sat on a boundary (proposal Sec. 5.2).

    The lower end moves down by ``factor``. The upper end moves up by ``factor``
    but never beyond ``cap_fraction`` of the stability cap; an upper hit that is
    already at that limit is labelled ``stability_capped`` and left alone.
    """
    if factor <= 1:
        raise ValueError("expansion factor must exceed one")
    cap = stability_cap(algorithm, params)
    limit = cap * cap_fraction if cap is not None else math.inf
    new_low, new_high, notes = low, high, []
    if "lower" in boundary_hits:
        new_low = low / factor
        notes.append("lower_expanded")
    if "upper" in boundary_hits:
        if high >= limit * (1 - 1e-12):
            notes.append("stability_capped")
        else:
            new_high = min(high * factor, limit)
            notes.append("upper_expanded")
    changed = new_low != low or new_high != high
    return {"low": new_low, "high": new_high, "changed": changed, "notes": notes}
