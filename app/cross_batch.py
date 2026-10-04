"""Cross-batch comparison layout plans over ACTIVE batch certificates.

``POST /api/v1/certificates/comparison-plans`` takes ``request_key``, 2–4
ACTIVE batch references (``batch_id`` + ``expected_revision``), one or more
comparisons (``comparison_id`` → one ``sample_id`` per batch), the target
plate geometry, blocked wells and a seed.

Before any placement is computed, the service checks, inside one SQLite
transaction, that every named certificate is ACTIVE and still at its
``expected_revision``, and confirms from each batch's immutable revision
snapshot that every named sample:

* exists in the frozen plan;
* still occupies exactly its planned well in the frozen scan (a displaced or
  duplicated sample never qualifies);
* sits in a well that is not blocked in the snapshot;
* carries a valid (present, non-null, finite) reading.

Any stale revision, duplicated/missing/unknown/ineligible sample, or an
instance admitting no arrangement fails the whole request; nothing is saved.

Placement optimality is lexicographic (CP-SAT):

1. maximise the number of orthogonal (up/down/left/right) plate edges whose
   two wells carry samples of the *same* comparison;
2. maximise the rows plus columns covered per batch and per
   (batch, experiment group) — spreading every batch and every group over the
   target plate;
3. a deterministic, seed-governed tie-break (the stage-2 optimum is fed back
   as a complete hint and the first solution instantiates it), so identical
   (snapshots, comparisons, plate, blocked wells, seed) always yields the
   identical plan.

The certificate state reads, the snapshot lookups, the plan insert and the
idempotency record all commit (or roll back) in one SQLite transaction.
State-dependent rejections (no certificate / REVOKED / stale revision) are not
recorded under the ``request_key``; content-dependent failures are recorded
like any other response.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

from ortools.sat.python import cp_model

from .certificates import (
    STATUS_ACTIVE,
    STATUS_REVOKED,
    _content_hash,
    _get_certificate,
    _idem_record,
    _idem_replay,
    _sha256,
    _utcnow,
    get_store,
)
from .plates import MAX_CELLS, PLATE_PRESETS, parse_well, well_name
from .planner import SolverTimeout, _make_solver
from .schemas import (
    CERTIFICATE_NOT_FOUND,
    CERTIFICATE_REVISION_MISMATCH,
    CERTIFICATE_REVOKED,
    VIOLATION_CROSS_BATCH_BATCH_DUPLICATED,
    VIOLATION_CROSS_BATCH_CAPACITY_EXCEEDED,
    VIOLATION_CROSS_BATCH_COMPARISON_DUPLICATED,
    VIOLATION_CROSS_BATCH_COMPARISON_INCOMPLETE,
    VIOLATION_CROSS_BATCH_NO_SOLUTION,
    VIOLATION_CROSS_BATCH_SAMPLE_DUPLICATED,
    VIOLATION_CROSS_BATCH_SAMPLE_NOT_ELIGIBLE,
    VIOLATION_CROSS_BATCH_SAMPLE_UNKNOWN,
    VIOLATION_PLATE_DIMENSION_INVALID,
    VIOLATION_WELL_REFERENCE_INVALID,
    CrossBatchPlanRequest,
)

_ENDPOINT_CROSS_BATCH = "POST /api/v1/certificates/comparison-plans"


class CrossBatchPlanInvalid(Exception):
    """No cross-batch comparison plan can be produced (HTTP 422).

    No plan row and no idempotency record is written by the pure builder;
    the endpoint transaction records the 422 under the request key.
    """

    def __init__(self, categories: list[str], messages: list[str]):
        self.categories = categories
        self.messages = messages
        super().__init__("; ".join(messages) if messages else "invalid cross-batch plan request")


def cross_batch_invalid_body(exc: CrossBatchPlanInvalid) -> dict:
    """Canonical 422 envelope for an unproducible comparison plan."""
    return {
        "detail": "No cross-batch comparison layout can be produced; no plan was saved.",
        "infeasibility_categories": exc.categories,
        "messages": exc.messages,
    }


# --------------------------------------------------------------- input model


@dataclass(frozen=True)
class _Item:
    """One placed sample: a (batch, comparison, sample) triple."""

    index: int
    batch_id: str
    batch_order: int
    comparison_id: str
    comparison_order: int
    sample_id: str
    group: str
    src_row: int
    src_col: int

    @property
    def qualified_id(self) -> str:
        return f"{self.batch_id}:{self.sample_id}"


@dataclass
class ComparisonPlanLayout:
    rows: int
    columns: int
    available_wells: list[str]
    blocked: list[str]
    items: list[_Item]
    placement: dict[str, str]  # qualified sample id -> target well
    adjacent_edges: list[tuple[str, str, str]]  # (comparison_id, w1, w2)
    coverage: dict[tuple[str, Optional[str]], tuple[list[int], list[int]]]
    stages: list[tuple[str, int, str, bool, float]]


# ------------------------------------------------------------- plate parsing


def _resolve_target_plate(plate) -> tuple[int, int]:
    if plate.type == "custom":
        if not plate.rows or not plate.columns:
            raise CrossBatchPlanInvalid(
                [VIOLATION_PLATE_DIMENSION_INVALID],
                ["custom plate requires both 'rows' and 'columns'"],
            )
        rows, cols = plate.rows, plate.columns
        if rows * cols > MAX_CELLS:
            raise CrossBatchPlanInvalid(
                [VIOLATION_PLATE_DIMENSION_INVALID],
                [f"custom plate {rows}x{cols} exceeds {MAX_CELLS} wells"],
            )
        return rows, cols
    preset_rows, preset_cols = PLATE_PRESETS[plate.type]
    if plate.rows is not None and plate.rows != preset_rows:
        raise CrossBatchPlanInvalid(
            [VIOLATION_PLATE_DIMENSION_INVALID],
            [f"plate type '{plate.type}' has {preset_rows} rows, got {plate.rows}"],
        )
    if plate.columns is not None and plate.columns != preset_cols:
        raise CrossBatchPlanInvalid(
            [VIOLATION_PLATE_DIMENSION_INVALID],
            [f"plate type '{plate.type}' has {preset_cols} columns, got {plate.columns}"],
        )
    return preset_rows, preset_cols


def _parse_blocked(blocked_wells: list[str], rows: int, cols: int) -> set[tuple[int, int]]:
    """Canonicalise the target blocked-well list; a bad well name is a 422."""
    categories: list[str] = []
    messages: list[str] = []
    blocked: set[tuple[int, int]] = set()
    for raw in blocked_wells:
        try:
            blocked.add(parse_well(raw, rows, cols))
        except ValueError as exc:
            if VIOLATION_WELL_REFERENCE_INVALID not in categories:
                categories.append(VIOLATION_WELL_REFERENCE_INVALID)
            messages.append(str(exc))
    if categories:
        raise CrossBatchPlanInvalid(categories, messages)
    return blocked


# ------------------------------------------------------ snapshot eligibility


def _sample_eligibility(snapshot: dict, sample_id: str, rows: int, cols: int) -> Optional[str]:
    """Return None when the sample is usable, else a human-readable reason.

    Mirrors the re-test plan rule: the sample must exist in the frozen plan,
    its well must not be blocked in the snapshot, the frozen scan must place
    exactly this sample at exactly its planned well, and the well must carry
    a valid (present, non-null, finite) reading.
    """
    target: dict[str, str] = snapshot["target"]
    scan: dict[str, str] = snapshot["scan"]
    readings: dict = snapshot.get("readings", {})
    group_of = {s["id"]: s["group"] for s in snapshot["samples"]}

    raw_well = target.get(sample_id)
    if raw_well is None:
        return "is not part of the batch's frozen plan"
    pos = parse_well(raw_well, rows, cols)
    well = well_name(*pos)

    frozen_blocked: set[tuple[int, int]] = set()
    for raw in snapshot.get("blocked_wells", []):
        frozen_blocked.add(parse_well(raw, rows, cols))
    if pos in frozen_blocked:
        return f"planned well {well} is blocked in the frozen snapshot"

    if group_of.get(sample_id) is None:
        return "has no experiment group in the frozen sample grouping"

    scanned_wells: set[str] = set()
    for raw_scan_well, sid in scan.items():
        if sid != sample_id:
            continue
        try:
            scanned_wells.add(well_name(*parse_well(raw_scan_well, rows, cols)))
        except ValueError:
            continue
    if scan.get(well) != sample_id or scanned_wells != {well}:
        return (
            f"is not confirmed at its planned well {well} in the frozen scan "
            "(misplaced, missing or duplicated)"
        )

    value = readings.get(well)
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return f"has no valid (finite, non-null) reading at its planned well {well}"
    return None


# -------------------------------------------------------------- validation


def _validate_and_build_items(
    req: CrossBatchPlanRequest, contexts: dict[str, dict]
) -> list[_Item]:
    """Check every structural/eligibility rule; return the sorted item list.

    All problems are accumulated and reported in one 422; nothing is saved.
    """
    categories: list[str] = []
    messages: list[str] = []

    def add(category: str, msg: str) -> None:
        if category not in categories:
            categories.append(category)
        messages.append(msg)

    # Batches, in request order (de-duplicated for lookups).
    batch_ids: list[str] = []
    batch_revision: dict[str, int] = {}
    for ref in req.batches:
        if ref.batch_id not in batch_revision:
            batch_revision[ref.batch_id] = ref.expected_revision
            batch_ids.append(ref.batch_id)
        elif batch_revision[ref.batch_id] != ref.expected_revision:
            add(
                VIOLATION_CROSS_BATCH_BATCH_DUPLICATED,
                f"batch '{ref.batch_id}' is listed more than once with "
                "different expected_revision values",
            )
        else:
            add(
                VIOLATION_CROSS_BATCH_BATCH_DUPLICATED,
                f"batch '{ref.batch_id}' is listed more than once",
            )

    # Comparison ids must be unique.
    comparison_order: dict[str, int] = {}
    for entry in req.comparisons:
        if entry.comparison_id in comparison_order:
            add(
                VIOLATION_CROSS_BATCH_COMPARISON_DUPLICATED,
                f"comparison_id '{entry.comparison_id}' appears more than once",
            )
        else:
            comparison_order[entry.comparison_id] = len(comparison_order)

    # Structural per-comparison checks: exactly one sample per batch.
    for entry in req.comparisons:
        for bid in batch_ids:
            if bid not in entry.samples:
                add(
                    VIOLATION_CROSS_BATCH_COMPARISON_INCOMPLETE,
                    f"comparison '{entry.comparison_id}' is missing the sample "
                    f"for batch '{bid}'",
                )
        for bid in entry.samples:
            if bid not in batch_revision:
                add(
                    VIOLATION_CROSS_BATCH_COMPARISON_INCOMPLETE,
                    f"comparison '{entry.comparison_id}' names batch '{bid}', "
                    "which is not among the request's batches",
                )

    # A sample may serve at most one comparison within its batch.
    usage: dict[str, dict[str, str]] = {bid: {} for bid in batch_ids}
    for entry in req.comparisons:
        for bid, sid in entry.samples.items():
            if bid not in batch_revision:
                continue
            if sid in usage[bid]:
                add(
                    VIOLATION_CROSS_BATCH_SAMPLE_DUPLICATED,
                    f"sample '{sid}' of batch '{bid}' is used by comparisons "
                    f"'{usage[bid][sid]}' and '{entry.comparison_id}'; each "
                    "sample may serve at most one comparison",
                )
            else:
                usage[bid][sid] = entry.comparison_id

    if categories:
        raise CrossBatchPlanInvalid(categories, messages)

    # Existence/eligibility against the immutable revision snapshots.
    items: list[_Item] = []
    for bid in batch_ids:
        ctx = contexts[bid]
        snapshot = ctx["snapshot"]
        rows = snapshot["plate"]["rows"]
        cols = snapshot["plate"]["columns"]
        group_of = {s["id"]: s["group"] for s in snapshot["samples"]}
        for entry in req.comparisons:
            sid = entry.samples[bid]
            if sid not in snapshot["target"]:
                add(
                    VIOLATION_CROSS_BATCH_SAMPLE_UNKNOWN,
                    f"sample '{sid}' of batch '{bid}' (comparison "
                    f"'{entry.comparison_id}') is unknown to the batch's "
                    "current revision snapshot",
                )
                continue
            reason = _sample_eligibility(snapshot, sid, rows, cols)
            if reason is not None:
                add(
                    VIOLATION_CROSS_BATCH_SAMPLE_NOT_ELIGIBLE,
                    f"sample '{sid}' of batch '{bid}' (comparison "
                    f"'{entry.comparison_id}') {reason}",
                )
                continue
            pos = parse_well(snapshot["target"][sid], rows, cols)
            items.append(
                _Item(
                    index=0,
                    batch_id=bid,
                    batch_order=batch_ids.index(bid),
                    comparison_id=entry.comparison_id,
                    comparison_order=comparison_order[entry.comparison_id],
                    sample_id=sid,
                    group=group_of[sid],
                    src_row=pos[0],
                    src_col=pos[1],
                )
            )

    if categories:
        raise CrossBatchPlanInvalid(categories, messages)

    # Deterministic item order: comparison, then batch (then sample id).
    items.sort(key=lambda it: (it.comparison_order, it.batch_order, it.sample_id))
    return [
        _Item(i, it.batch_id, it.batch_order, it.comparison_id, it.comparison_order,
              it.sample_id, it.group, it.src_row, it.src_col)
        for i, it in enumerate(items)
    ]


# ------------------------------------------------------------------- CP-SAT


class _ComparisonModel:
    """One CP-SAT model for a lexicographic stage of the comparison layout."""

    def __init__(self, rows: int, cols: int, blocked: set[tuple[int, int]],
                 items: list[_Item], batch_ids: list[str]):
        self.rows = rows
        self.cols = cols
        self.items = items
        self.n = len(items)
        self.batch_ids = list(batch_ids)
        self.batch_index = {b: i for i, b in enumerate(self.batch_ids)}
        self.comparison_ids = sorted(
            {it.comparison_id for it in items},
            key=lambda cid: min(it.comparison_order for it in items
                                if it.comparison_id == cid),
        )
        self.comparison_index = {c: i for i, c in enumerate(self.comparison_ids)}

        self.model = cp_model.CpModel()
        self.wells = [
            (r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked
        ]
        self.well_index = {w: i for i, w in enumerate(self.wells)}

        # x[i][j]: item i occupies available well j.
        self.x = [
            [self.model.new_bool_var(f"x_{i}_{j}") for j in range(len(self.wells))]
            for i in range(self.n)
        ]
        self.occ = {w: self.model.new_bool_var(f"occ_{w[0]}_{w[1]}") for w in self.wells}
        for i in range(self.n):
            self.model.add_exactly_one(self.x[i])
        for j, w in enumerate(self.wells):
            col = [self.x[i][j] for i in range(self.n)]
            self.model.add(sum(col) <= 1)
            self.model.add(self.occ[w] == sum(col))

        self._build_adjacency()
        self._build_coverage()

    # ------------------------------------------------------------- adjacency
    def _build_adjacency(self) -> None:
        # Comparison index carried by each well (-1 when empty).
        n_c = max(len(self.comparison_ids) - 1, 0)
        self.cmp_at = {}
        for w in self.wells:
            var = self.model.new_int_var(-1, n_c, f"cmp_{w[0]}_{w[1]}")
            self.model.add(var == -1).only_enforce_if(self.occ[w].Not())
            for i, it in enumerate(self.items):
                j = self.well_index[w]
                self.model.add(
                    var == self.comparison_index[it.comparison_id]
                ).only_enforce_if(self.x[i][j])
            self.cmp_at[w] = var

        well_set = set(self.wells)
        self.edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r, c in self.wells:
            if (r + 1, c) in well_set:
                self.edges.append(((r, c), (r + 1, c)))
            if (r, c + 1) in well_set:
                self.edges.append(((r, c), (r, c + 1)))

        # p[e] <=> occ[u] AND occ[v] AND cmp[u] == cmp[v]
        self.edge_same: list = []
        self.edge_p: list = []
        for ei, (u, v) in enumerate(self.edges):
            same = self.model.new_bool_var(f"adjsame_{ei}")
            self.model.add(self.cmp_at[u] == self.cmp_at[v]).only_enforce_if(same)
            self.model.add(self.cmp_at[u] != self.cmp_at[v]).only_enforce_if(same.Not())
            p = self.model.new_bool_var(f"adj_{ei}")
            self.model.add(p <= self.occ[u])
            self.model.add(p <= self.occ[v])
            self.model.add(p <= same)
            self.model.add(p >= self.occ[u] + self.occ[v] + same - 2)
            self.edge_same.append(same)
            self.edge_p.append(p)
        self.adjacency = self.model.new_int_var(0, max(len(self.edges), 0), "adjacency")
        self.model.add(self.adjacency == sum(self.edge_p))

    # -------------------------------------------------------------- coverage
    def _build_coverage(self) -> None:
        """Batch and (batch, group) row/column coverage indicator variables."""
        self.groups_by_batch: dict[str, list[str]] = {}
        for it in self.items:
            gs = self.groups_by_batch.setdefault(it.batch_id, [])
            if it.group not in gs:
                gs.append(it.group)
        for gs in self.groups_by_batch.values():
            gs.sort()

        # Per-batch row/column occupancy.
        self.batch_row: dict[str, list] = {}
        self.batch_col: dict[str, list] = {}
        self.batch_rr: dict[str, list] = {}
        self.batch_cc: dict[str, list] = {}
        # Per-(batch, group) row/column occupancy.
        self.group_row: dict[tuple[str, str], list] = {}
        self.group_col: dict[tuple[str, str], list] = {}
        self.group_rr: dict[tuple[str, str], list] = {}
        self.group_cc: dict[tuple[str, str], list] = {}

        for bid in self.batch_ids:
            members = [i for i, it in enumerate(self.items) if it.batch_id == bid]
            r_lists = {r: [] for r in range(self.rows)}
            c_lists = {c: [] for c in range(self.cols)}
            for i in members:
                for j, (r, c) in enumerate(self.wells):
                    r_lists[r].append(self.x[i][j])
                    c_lists[c].append(self.x[i][j])
            rr, cc = [], []
            for r in range(self.rows):
                v = self.model.new_bool_var(f"brr_{bid}_{r}")
                self.model.add(v <= sum(r_lists[r]))
                for t in r_lists[r]:
                    self.model.add(v >= t)
                rr.append(v)
            for c in range(self.cols):
                v = self.model.new_bool_var(f"bcc_{bid}_{c}")
                self.model.add(v <= sum(c_lists[c]))
                for t in c_lists[c]:
                    self.model.add(v >= t)
                cc.append(v)
            self.batch_row[bid] = [sum(r_lists[r]) for r in range(self.rows)]
            self.batch_col[bid] = [sum(c_lists[c]) for c in range(self.cols)]
            self.batch_rr[bid] = rr
            self.batch_cc[bid] = cc

            for g in self.groups_by_batch.get(bid, []):
                g_members = [
                    i for i, it in enumerate(self.items)
                    if it.batch_id == bid and it.group == g
                ]
                gr_lists = {r: [] for r in range(self.rows)}
                gc_lists = {c: [] for c in range(self.cols)}
                for i in g_members:
                    for j, (r, c) in enumerate(self.wells):
                        gr_lists[r].append(self.x[i][j])
                        gc_lists[c].append(self.x[i][j])
                grr, gcc = [], []
                for r in range(self.rows):
                    v = self.model.new_bool_var(f"grr_{bid}_{g}_{r}")
                    self.model.add(v <= sum(gr_lists[r]))
                    for t in gr_lists[r]:
                        self.model.add(v >= t)
                    grr.append(v)
                for c in range(self.cols):
                    v = self.model.new_bool_var(f"gcc_{bid}_{g}_{c}")
                    self.model.add(v <= sum(gc_lists[c]))
                    for t in gc_lists[c]:
                        self.model.add(v >= t)
                    gcc.append(v)
                key = (bid, g)
                self.group_row[key] = [sum(gr_lists[r]) for r in range(self.rows)]
                self.group_col[key] = [sum(gc_lists[c]) for c in range(self.cols)]
                self.group_rr[key] = grr
                self.group_cc[key] = gcc

        total_max = len(self.batch_ids) * (self.rows + self.cols) + sum(
            len(gs) * (self.rows + self.cols)
            for gs in self.groups_by_batch.values()
        )
        self.coverage = self.model.new_int_var(0, max(total_max, 0), "coverage")
        expr = []
        for bid in self.batch_ids:
            expr += self.batch_rr[bid] + self.batch_cc[bid]
        for key in self.group_rr:
            expr += self.group_rr[key] + self.group_cc[key]
        self.model.add(self.coverage == sum(expr))

    # ------------------------------------------------------------- objectives
    def maximize_adjacency(self) -> None:
        self.model.maximize(self.adjacency)

    def pin_adjacency(self, value: int) -> None:
        self.model.add(self.adjacency == value)

    def maximize_coverage(self) -> None:
        self.model.maximize(self.coverage)

    def pin_coverage(self, value: int) -> None:
        self.model.add(self.coverage == value)

    # ----------------------------------------------------------------- hints
    def add_hint(self, assignment: dict[int, tuple[int, int]]) -> None:
        """Pin every variable from a complete feasible item → well map."""
        occupied = set(assignment.values())
        item_at = {pos: self.items[i] for i, pos in assignment.items()}

        for i in range(self.n):
            j = self.well_index[assignment[i]]
            for k in range(len(self.wells)):
                self.model.add_hint(self.x[i][k], 1 if k == j else 0)
        for w in self.wells:
            self.model.add_hint(self.occ[w], 1 if w in occupied else 0)
            self.model.add_hint(
                self.cmp_at[w],
                self.comparison_index[item_at[w].comparison_id] if w in item_at else -1,
            )
        for ei, (u, v) in enumerate(self.edges):
            same_cmp = (
                u in item_at and v in item_at
                and item_at[u].comparison_id == item_at[v].comparison_id
            )
            both_empty = u not in occupied and v not in occupied
            self.model.add_hint(self.edge_same[ei], 1 if same_cmp or both_empty else 0)
            self.model.add_hint(self.edge_p[ei], 1 if same_cmp else 0)
        self.model.add_hint(
            self.adjacency,
            sum(
                1 for u, v in self.edges
                if u in item_at and v in item_at
                and item_at[u].comparison_id == item_at[v].comparison_id
            ),
        )

        def hint_indicators(rr: dict, cc: dict) -> None:
            for key, vars_r in rr.items():
                covered = {r for r in range(self.rows)
                           if any((r, c) in occupied and self._belongs(key, item_at[(r, c)])
                                  for c in range(self.cols))}
                for r, var in enumerate(vars_r):
                    self.model.add_hint(var, 1 if r in covered else 0)
            for key, vars_c in cc.items():
                covered = {c for c in range(self.cols)
                           if any((r, c) in occupied and self._belongs(key, item_at[(r, c)])
                                  for r in range(self.rows))}
                for c, var in enumerate(vars_c):
                    self.model.add_hint(var, 1 if c in covered else 0)

        hint_indicators(self.batch_rr, self.batch_cc)
        hint_indicators(self.group_rr, self.group_cc)
        cover_val = 0
        for bid in self.batch_ids:
            cover_val += len({p[0] for p, it in item_at.items() if it.batch_id == bid})
            cover_val += len({p[1] for p, it in item_at.items() if it.batch_id == bid})
        for (bid, g) in self.group_rr:
            cover_val += len({p[0] for p, it in item_at.items()
                              if it.batch_id == bid and it.group == g})
            cover_val += len({p[1] for p, it in item_at.items()
                              if it.batch_id == bid and it.group == g})
        self.model.add_hint(self.coverage, cover_val)

    @staticmethod
    def _belongs(key, it: _Item) -> bool:
        if isinstance(key, tuple):
            bid, g = key
            return it.batch_id == bid and it.group == g
        return it.batch_id == key

    def assignment_from_solver(self, solver) -> dict[int, tuple[int, int]]:
        out: dict[int, tuple[int, int]] = {}
        for i in range(self.n):
            for j, w in enumerate(self.wells):
                if solver.value(self.x[i][j]) == 1:
                    out[i] = w
                    break
        return out


def _greedy_assignment(items: list[_Item],
                       wells: list[tuple[int, int]]) -> dict[int, tuple[int, int]]:
    """Deterministic feasible warm start: comparison-major row-major dealing.

    Items are already ordered (comparison, batch); filling row-major makes
    members of one comparison land on consecutive available wells, giving the
    optimiser an adjacency-rich feasible hint.
    """
    return {i: w for i, w in zip(range(len(items)), wells)}


def _solve_stage(stage: str, model: _ComparisonModel, seed: int, *,
                 first_solution: bool = False):
    det_limit = float(os.getenv("CP_DETERMINISTIC_TIME_SECONDS", "4"))
    wall_limit = float(os.getenv("CP_WALL_TIME_SECONDS", "12"))
    fast_det = float(os.getenv("CP_FAST_DETERMINISTIC_TIME_SECONDS", "1"))
    fast_wall = float(os.getenv("CP_FAST_WALL_TIME_SECONDS", "8"))
    if first_solution:
        solver = _make_solver(seed, fast_det, fast_wall,
                              stop_at_first=True, fast_presolve=True)
    else:
        solver = _make_solver(seed, det_limit, wall_limit)
    t0 = time.perf_counter()
    status = solver.solve(model.model)
    elapsed = time.perf_counter() - t0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return solver, (
            stage,
            int(solver.objective_value) if not first_solution else -1,
            solver.status_name(status),
            status == cp_model.OPTIMAL,
            elapsed,
        )
    if status == cp_model.INFEASIBLE:
        raise CrossBatchPlanInvalid(
            [VIOLATION_CROSS_BATCH_NO_SOLUTION],
            [f"CP-SAT proved the comparison layout infeasible at stage {stage}"],
        )
    raise SolverTimeout(stage, elapsed)


def build_comparison_plan(
    req: CrossBatchPlanRequest,
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
    items: list[_Item],
    batch_ids: list[str],
) -> ComparisonPlanLayout:
    """Run the three-stage lexicographic optimisation."""
    seed31 = int(req.seed) % (2**31)
    stages: list[tuple[str, int, str, bool, float]] = []
    wells = [
        (r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked
    ]

    # Capacity was pre-checked; guard anyway.
    if len(items) > len(wells):
        raise CrossBatchPlanInvalid(
            [VIOLATION_CROSS_BATCH_CAPACITY_EXCEEDED],
            [f"{len(items)} samples but only {len(wells)} available wells"],
        )

    hint0 = _greedy_assignment(items, wells)

    # Stage 1: maximise same-comparison orthogonal adjacency.
    m1 = _ComparisonModel(rows, cols, blocked, items, batch_ids)
    m1.maximize_adjacency()
    m1.add_hint(hint0)
    solver1, info1 = _solve_stage("1_comparison_adjacency", m1, seed31)
    best_adj = int(solver1.objective_value)
    stages.append(info1)

    # Stage 2: maximise per-batch / per-group row+column coverage.
    m2 = _ComparisonModel(rows, cols, blocked, items, batch_ids)
    m2.pin_adjacency(best_adj)
    m2.maximize_coverage()
    m2.add_hint(m1.assignment_from_solver(solver1))
    solver2, info2 = _solve_stage("2_batch_coverage", m2, seed31)
    best_cov = int(solver2.objective_value)
    stages.append(info2)

    # Stage 3: deterministic seed tie-break.
    m3 = _ComparisonModel(rows, cols, blocked, items, batch_ids)
    m3.pin_adjacency(best_adj)
    m3.pin_coverage(best_cov)
    m3.add_hint(m2.assignment_from_solver(solver2))
    solver3, info3 = _solve_stage("3_tiebreak", m3, seed31, first_solution=True)
    stages.append(info3)

    assignment = m3.assignment_from_solver(solver3)

    placement = {
        items[i].qualified_id: well_name(*pos) for i, pos in assignment.items()
    }

    # Recount the same-comparison edges independently (authoritative).
    item_at = {pos: items[i] for i, pos in assignment.items()}
    well_set = set(assignment.values())
    edge_records: list[tuple[str, str, str]] = []
    for r, c in wells:
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if (nr, nc) not in well_set or (r, c) not in well_set:
                continue
            a, b = item_at[(r, c)], item_at[(nr, nc)]
            if a.comparison_id == b.comparison_id:
                w1, w2 = sorted((well_name(r, c), well_name(nr, nc)))
                edge_records.append((a.comparison_id, w1, w2))
    edge_records.sort(key=lambda e: (
        next(it.comparison_order for it in items if it.comparison_id == e[0]),
        e[1], e[2],
    ))

    # Recount coverage independently.
    def lines(key) -> tuple[list[int], list[int]]:
        if isinstance(key, tuple):
            bid, g = key
            sel = [p for p, it in item_at.items()
                   if it.batch_id == bid and it.group == g]
        else:
            sel = [p for p, it in item_at.items() if it.batch_id == key]
        return sorted({r for r, _ in sel}), sorted({c for _, c in sel})

    coverage: dict[tuple[str, Optional[str]], tuple[list[int], list[int]]] = {}
    for bid in batch_ids:
        coverage[(bid, None)] = lines(bid)
        for g in sorted({it.group for it in items if it.batch_id == bid}):
            coverage[(bid, g)] = lines((bid, g))

    available = [
        well_name(r, c) for r in range(rows) for c in range(cols)
        if (r, c) not in blocked
    ]
    blocked_names = sorted(
        (well_name(r, c) for r, c in blocked),
        key=lambda w: parse_well(w, rows, cols),
    )
    return ComparisonPlanLayout(
        rows=rows,
        columns=cols,
        available_wells=available,
        blocked=blocked_names,
        items=items,
        placement=placement,
        adjacent_edges=edge_records,
        coverage=coverage,
        stages=stages,
    )


# ---------------------------------------------------------------- endpoint


def _plan_id(request_key: str) -> str:
    return "cmp-" + _sha256(
        f"cross-batch-comparison-plan\0{request_key}".encode("utf-8")
    )[:24]


def _plan_body(*, req: CrossBatchPlanRequest, plan_id: str,
               contexts: dict[str, dict], batch_ids: list[str],
               layout: ComparisonPlanLayout) -> dict:
    sources = []
    for bid in batch_ids:
        ctx = contexts[bid]
        sources.append({
            "batch_id": bid,
            "certificate_id": ctx["cert"]["certificate_id"],
            "status": STATUS_ACTIVE,
            "revision": ctx["revision"],
            "digest": ctx["digest"],
        })

    # samples[] follows target-well row-major order.
    sample_entries = sorted(
        (
            {
                "sample_id": it.sample_id,
                "well": layout.placement[it.qualified_id],
                "batch_id": it.batch_id,
                "comparison_id": it.comparison_id,
                "group": it.group,
            }
            for it in layout.items
        ),
        key=lambda e: parse_well(e["well"], layout.rows, layout.columns),
    )

    comparison_order = {
        entry.comparison_id: i for i, entry in enumerate(req.comparisons)
    }
    batch_order = {bid: i for i, bid in enumerate(batch_ids)}
    item_by_key = {(it.batch_id, it.comparison_id): it for it in layout.items}
    comparisons = []
    for entry in sorted(req.comparisons, key=lambda e: comparison_order[e.comparison_id]):
        members = []
        for bid in sorted(batch_ids, key=lambda b: batch_order[b]):
            it = item_by_key[(bid, entry.comparison_id)]
            members.append({
                "batch_id": bid,
                "sample_id": it.sample_id,
                "qualified_sample_id": it.qualified_id,
                "group": it.group,
                "well": layout.placement[it.qualified_id],
            })
        wells = sorted(
            (m["well"] for m in members),
            key=lambda w: parse_well(w, layout.rows, layout.columns),
        )
        comparisons.append({
            "comparison_id": entry.comparison_id,
            "members": members,
            "wells": wells,
            "same_comparison_adjacent_edges_count": sum(
                1 for cid, _, _ in layout.adjacent_edges
                if cid == entry.comparison_id
            ),
        })

    coverage_report = []
    for bid in batch_ids:
        rws, cls = layout.coverage[(bid, None)]
        per_group = []
        for g in sorted({it.group for it in layout.items if it.batch_id == bid}):
            grows, gcols = layout.coverage[(bid, g)]
            per_group.append({
                "group": g,
                "covered_rows": [well_name(r, 0)[0] for r in grows],
                "covered_columns": [c + 1 for c in gcols],
                "rows_covered": len(grows),
                "columns_covered": len(gcols),
            })
        coverage_report.append({
            "batch_id": bid,
            "covered_rows": [well_name(r, 0)[0] for r in rws],
            "covered_columns": [c + 1 for c in cls],
            "rows_covered": len(rws),
            "columns_covered": len(cls),
            "groups": per_group,
        })

    total_edges = 0
    avail_set = {parse_well(w, layout.rows, layout.columns)
                 for w in layout.available_wells}
    for r in range(layout.rows):
        for c in range(layout.columns):
            if (r + 1, c) in avail_set and (r, c) in avail_set:
                total_edges += 1
            if (r, c + 1) in avail_set and (r, c) in avail_set:
                total_edges += 1

    return {
        "plan_id": plan_id,
        "request_key": req.request_key,
        "status": STATUS_ACTIVE,
        "plate": {
            "type": req.plate.type,
            "rows": layout.rows,
            "columns": layout.columns,
            "blocked_wells": layout.blocked,
            "available_wells": layout.available_wells,
        },
        "sources": sources,
        "comparisons": comparisons,
        "layout": {k: layout.placement[k] for k in sorted(layout.placement)},
        "samples": sample_entries,
        "wells": [e["well"] for e in sample_entries],
        "metrics": {
            "same_comparison_adjacent_edges_count": len(layout.adjacent_edges),
            "same_comparison_adjacent_edges": [
                {"comparison_id": cid, "wells": [w1, w2]}
                for cid, w1, w2 in layout.adjacent_edges
            ],
            "total_orthogonal_edges": total_edges,
            "coverage": {
                "rows_plus_columns_sum": sum(
                    len(rs) + len(cs)
                    for key, (rs, cs) in layout.coverage.items()
                ),
                "by_batch": coverage_report,
            },
        },
        "seed": req.seed,
        "solver": {
            "stages": [
                {
                    "stage": st[0],
                    "objective_value": st[1],
                    "status": st[2],
                    "proven_optimal": st[3],
                    "wall_time_seconds": round(st[4], 4),
                }
                for st in layout.stages
            ]
        },
    }


def create_comparison_plan(req: CrossBatchPlanRequest) -> tuple[int, dict]:
    """Verify every certificate/sample, optimise the layout and persist it."""
    store = get_store()
    content_hash = _content_hash(_ENDPOINT_CROSS_BATCH, req)
    with store.transaction() as conn:
        # State of every named certificate is checked BEFORE the idempotency
        # lookup on purpose: a replay naming a revision that has since advanced
        # must be refused with 409 rather than returning a plan pinned to stale
        # evidence. Duplicate batch refs are de-duplicated here; the builder
        # still rejects them as a content error below.
        contexts: dict[str, dict] = {}
        seen: set[str] = set()
        for ref in req.batches:
            if ref.batch_id in seen:
                continue
            seen.add(ref.batch_id)
            cert = _get_certificate(conn, ref.batch_id)
            if cert is None:
                return 404, {
                    "detail": "no certificate exists for the batch",
                    "error": CERTIFICATE_NOT_FOUND,
                    "batch_id": ref.batch_id,
                }
            if cert["status"] == STATUS_REVOKED:
                return 409, {
                    "detail": "the certificate is REVOKED; no comparison plan "
                    "can use it and a revoked certificate can never be restored",
                    "error": CERTIFICATE_REVOKED,
                    "certificate_id": cert["certificate_id"],
                    "batch_id": ref.batch_id,
                    "status": STATUS_REVOKED,
                    "revoked_reason": cert["revoked_reason"],
                }
            if ref.expected_revision != cert["revision"]:
                return 409, {
                    "detail": "expected_revision does not match the certificate's "
                    "current revision; comparison plans are drawn only from "
                    "current revisions",
                    "error": CERTIFICATE_REVISION_MISMATCH,
                    "certificate_id": cert["certificate_id"],
                    "batch_id": ref.batch_id,
                    "current_revision": cert["revision"],
                    "expected_revision": ref.expected_revision,
                }
            rev_row = conn.execute(
                "SELECT digest, snapshot FROM certificate_revisions "
                "WHERE batch_id = ? AND revision = ?",
                (ref.batch_id, cert["revision"]),
            ).fetchone()
            if rev_row is None:  # pragma: no cover - invariant guard
                raise RuntimeError(
                    f"revision {cert['revision']} snapshot missing for "
                    f"batch {ref.batch_id!r}"
                )
            contexts[ref.batch_id] = {
                "cert": cert,
                "revision": cert["revision"],
                "digest": rev_row["digest"],
                "snapshot": json.loads(rev_row["snapshot"]),
            }

        replay = _idem_replay(conn, req.request_key, content_hash)
        if replay is not None:
            return replay

        # Content validation (structure + snapshot eligibility), capacity and
        # the CP model all run inside the transaction: any failure is recorded
        # like a normal response, and the transaction commits nothing but the
        # idempotency record. SolverTimeout intentionally propagates (503) so
        # a timed-out request is never recorded.
        try:
            rows, cols = _resolve_target_plate(req.plate)
            blocked = _parse_blocked(req.blocked_wells, rows, cols)
            items = _validate_and_build_items(req, contexts)
            batch_ids = []
            for ref in req.batches:
                if ref.batch_id not in batch_ids:
                    batch_ids.append(ref.batch_id)

            if len(items) > rows * cols - len(blocked):
                raise CrossBatchPlanInvalid(
                    [VIOLATION_CROSS_BATCH_CAPACITY_EXCEEDED],
                    [f"{len(items)} samples but only {rows * cols - len(blocked)} "
                     f"available wells ({rows * cols} total, {len(blocked)} blocked)"],
                )

            layout = build_comparison_plan(req, rows, cols, blocked, items, batch_ids)
        except CrossBatchPlanInvalid as exc:
            body = cross_batch_invalid_body(exc)
            _idem_record(conn, req.request_key, _ENDPOINT_CROSS_BATCH,
                         content_hash, 422, body)
            return 422, body

        plan_id = _plan_id(req.request_key)
        body = _plan_body(
            req=req, plan_id=plan_id, contexts=contexts,
            batch_ids=batch_ids, layout=layout,
        )
        now = _utcnow()
        conn.execute(
            "INSERT INTO cross_batch_plans (plan_id, request_key, batch_count, "
            "sources, seed, plan_body, created_at) VALUES (?,?,?,?,?,?,?)",
            (
                plan_id,
                req.request_key,
                len(batch_ids),
                json.dumps(
                    [{"batch_id": s["batch_id"], "revision": s["revision"],
                      "digest": s["digest"], "certificate_id": s["certificate_id"]}
                     for s in body["sources"]],
                    ensure_ascii=False,
                ),
                req.seed,
                json.dumps(body, ensure_ascii=False),
                now,
            ),
        )
        _idem_record(conn, req.request_key, _ENDPOINT_CROSS_BATCH,
                     content_hash, 201, body)
        return 201, body
