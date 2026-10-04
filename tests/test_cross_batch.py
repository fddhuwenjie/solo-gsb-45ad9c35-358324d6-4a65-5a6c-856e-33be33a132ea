"""Cross-batch comparison plan tests.

Two layers:

* the pure optimizer (``app.cross_batch.build_comparison_plan``) is checked
  against a brute-force enumeration of every placement on small plates: it
  must first maximise same-comparison orthogonal edges, then maximise the
  rows+columns covered per batch and per (batch, group); identical inputs
  must yield the identical plan and every returned plan must be lex-optimal;
* the endpoint (``POST /api/v1/certificates/comparison-plans``) is checked
  for persistence/transaction semantics: idempotent replay, same-key
  different-content 409, REVOKED 409, stale revision 409 (including replays
  of a key recorded at an older revision), 404, sample
  duplication/missing/unknown/ineligible failures, blocked-well/capacity
  failures and the single-transaction plan + idempotency record write.
"""
import itertools
import json
import random
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import certificates, cross_batch
from app.main import app

client = TestClient(app)

# Reuse the 4x4 certified-batch fixture layout from the retest tests.
LAYOUT_4X4 = {
    "gA1": "A1", "gA2": "A3", "gB1": "A2", "gB2": "A4",
    "gB3": "B1", "gB4": "B3", "gA3": "B2", "gA4": "B4",
    "gA5": "C1", "gA6": "C3", "gB5": "C2", "gB6": "C4",
    "gB7": "D1", "gB8": "D3", "gA7": "D2", "gA8": "D4",
}
SAMPLES_4X4 = [
    {"id": s, "group": "GA" if s.startswith("gA") else "GB"}
    for s in LAYOUT_4X4
]
PLATE_4X4 = {"type": "custom", "rows": 4, "columns": 4}


@pytest.fixture(autouse=True)
def fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CERT_DB_PATH", str(tmp_path / "certificates.db"))
    certificates.reset_stores()
    yield tmp_path / "certificates.db"
    certificates.reset_stores()


def _hash(text: str) -> int:
    import hashlib
    return int(hashlib.md5(text.encode()).hexdigest()[:6], 16)


def _readings(offset=0):
    out = {}
    for sid, well in LAYOUT_4X4.items():
        base = 100.0 if sid.startswith("gA") else 50.0
        noise = (_hash(sid) % 7) - 3
        out[well] = base + noise + offset
    return out


def _release_payload(**over):
    payload = {
        "plate": PLATE_4X4,
        "target": dict(LAYOUT_4X4),
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "blocked_wells": [],
        "samples": [dict(s) for s in SAMPLES_4X4],
        "readings": _readings(),
        "seed": 7,
        "max_spatial_effect": 100.0,
        "min_p_value": 0.001,
        "permutations": 50,
        "tolerance": 1e-6,
        "max_iterations": 100,
    }
    payload.update(over)
    return payload


def _create_cert(batch_id, key=None, **over):
    key = key or f"create-{batch_id}"
    r = client.post("/api/v1/certificates", json={
        "batch_id": batch_id,
        "request_key": key,
        "release": _release_payload(**over),
    })
    assert r.status_code == 201, r.text
    return r.json()


def _two_certs():
    c1 = _create_cert("BATCH-1")
    c2 = _create_cert("BATCH-2")
    return c1, c2


def _comparison_payload(**over):
    payload = {
        "request_key": "cmp-1",
        "batches": [
            {"batch_id": "BATCH-1", "expected_revision": 1},
            {"batch_id": "BATCH-2", "expected_revision": 1},
        ],
        "comparisons": [
            {"comparison_id": "C1",
             "samples": {"BATCH-1": "gA1", "BATCH-2": "gA2"}},
            {"comparison_id": "C2",
             "samples": {"BATCH-1": "gB3", "BATCH-2": "gB4"}},
        ],
        "plate": {"type": "custom", "rows": 3, "columns": 3},
        "blocked_wells": [],
        "seed": 42,
    }
    payload.update(over)
    return payload


def _post(**over):
    return client.post(
        "/api/v1/certificates/comparison-plans", json=_comparison_payload(**over)
    )


# =========================================================== pure optimizer


def _grid_snapshot(batch_id, rows, cols, groups, readings=None, scan=None,
                   blocked=None):
    """Build a frozen-snapshot-like dict for a fully populated small plate."""
    from app.plates import well_name
    target = {}
    samples = []
    for r in range(rows):
        for c in range(cols):
            sid = f"{batch_id}-s{r}_{c}"
            target[sid] = well_name(r, c)
            samples.append({"id": sid, "group": groups[r][c]})
    wells = list(target.values())
    return {
        "batch_id": batch_id,
        "plate": {"type": "custom", "rows": rows, "columns": cols},
        "target": target,
        "scan": scan or {w: s for s, w in target.items()},
        "readings": readings if readings is not None else {w: 1.0 for w in wells},
        "samples": samples,
        "blocked_wells": blocked or [],
        "seed": 0,
        "thresholds": {},
        "view_digest": None,
    }


def _build(batch_ids, comparisons, rows, cols, snapshots, blocked=(), seed=0):
    """Run validation + optimizer for fabricated snapshot contexts."""
    req = cross_batch.CrossBatchPlanRequest(
        request_key="k",
        batches=[{"batch_id": b, "expected_revision": 1} for b in batch_ids],
        comparisons=[
            {"comparison_id": cid, "samples": dict(spec)}
            for cid, spec in comparisons
        ],
        plate={"type": "custom", "rows": rows, "columns": cols},
        blocked_wells=list(blocked),
        seed=seed,
    )
    contexts = {b: {"snapshot": snapshots[b]} for b in batch_ids}
    items = cross_batch._validate_and_build_items(req, contexts)
    block_set = {cross_batch.parse_well(w, rows, cols) for w in blocked}
    return req, cross_batch.build_comparison_plan(
        req, rows, cols, block_set, items, batch_ids)


def _brute_force(batch_ids, comparisons, rows, cols, snapshots, blocked=()):
    """Enumerate every injective placement; return (items, (adj*, cov*), optima)."""
    contexts = {b: {"snapshot": snapshots[b]} for b in batch_ids}
    items = cross_batch._validate_and_build_items(
        cross_batch.CrossBatchPlanRequest(
            request_key="k",
            batches=[{"batch_id": b, "expected_revision": 1} for b in batch_ids],
            comparisons=[
                {"comparison_id": cid, "samples": dict(spec)}
                for cid, spec in comparisons
            ],
            plate={"type": "custom", "rows": rows, "columns": cols},
            blocked_wells=list(blocked),
        ),
        contexts,
    )
    wells = [
        (r, c) for r in range(rows) for c in range(cols)
        if (r, c) not in {cross_batch.parse_well(w, rows, cols) for w in blocked}
    ]
    best = None
    optima = []

    def evaluate(assign):
        pos_of = {i: p for i, p in assign}
        well_set = set(pos_of.values())
        adj = 0
        for i, p in pos_of.items():
            r, c = p
            for nxt in ((r + 1, c), (r, c + 1)):
                if nxt not in well_set:
                    continue
                j = next(k for k, q in pos_of.items() if q == nxt)
                if items[i].comparison_id == items[j].comparison_id:
                    adj += 1
        cov = 0
        for bid in batch_ids:
            sel = [pos_of[i] for i in range(len(items))
                   if items[i].batch_id == bid]
            cov += len({r for r, _ in sel}) + len({c for _, c in sel})
        for bid in batch_ids:
            for g in {it.group for it in items if it.batch_id == bid}:
                sel = [pos_of[i] for i in range(len(items))
                       if items[i].batch_id == bid and items[i].group == g]
                cov += len({r for r, _ in sel}) + len({c for _, c in sel})
        return adj, cov

    for wells_pick in itertools.permutations(wells, len(items)):
        assign = tuple(enumerate(wells_pick))
        adj, cov = evaluate(assign)
        key = (-adj, -cov)
        if best is None or key < best:
            best = key
            optima = [frozenset((i, p) for i, p in assign)]
        elif key == best:
            optima.append(frozenset((i, p) for i, p in assign))
    return items, (-best[0], -best[1]), optima


def test_build_matches_bruteforce_on_random_instances():
    rng = random.Random(20261004)
    for trial in range(40):
        rows = 2
        cols = rng.randint(2, 3)
        n_batches = rng.randint(2, 3)
        batch_ids = [f"B{k}" for k in range(n_batches)]
        snapshots = {}
        for k, bid in enumerate(batch_ids):
            groups = [
                [rng.choice(["GA", "GB", "GC"]) for _ in range(cols)]
                for _ in range(rows)
            ]
            snapshots[bid] = _grid_snapshot(bid, rows, cols, groups)

        n_cmp = rng.randint(1, 2)
        used = {bid: set() for bid in batch_ids}
        comparisons = []
        for ci in range(n_cmp):
            spec = {}
            for bid in batch_ids:
                choices = [
                    sid for sid in snapshots[bid]["target"]
                    if sid not in used[bid]
                ]
                sid = rng.choice(choices)
                used[bid].add(sid)
                spec[bid] = sid
            comparisons.append((f"C{ci}", spec))

        blocked = []
        if rng.random() < 0.3:
            from app.plates import well_name
            blocked.append(well_name(rng.randrange(rows), rng.randrange(cols)))

        n_items = n_batches * n_cmp
        avail = rows * cols - len(blocked)
        # Keep the enumeration tractable: P(avail, items) <= 720 here.
        if n_items > avail:
            continue

        _items, (adj_star, cov_star), optima = _brute_force(
            batch_ids, comparisons, rows, cols, snapshots, blocked)
        seed = rng.randint(0, 10_000)
        req, layout = _build(
            batch_ids, comparisons, rows, cols, snapshots, blocked, seed)
        assert len(layout.adjacent_edges) == adj_star, (trial,)
        cov_got = sum(
            len(rs) + len(cs) for rs, cs in layout.coverage.values())
        assert cov_got == cov_star, (trial, cov_got, cov_star)
        got = frozenset(
            (it.index, cross_batch.parse_well(
                layout.placement[it.qualified_id], rows, cols))
            for it in layout.items
        )
        assert got in optima, trial


def test_build_is_deterministic():
    snaps = {
        "B1": _grid_snapshot("B1", 3, 3, [["GA", "GB", "GA"]] * 3),
        "B2": _grid_snapshot("B2", 3, 3, [["GB", "GA", "GB"]] * 3),
    }
    cmps = [
        ("C1", {"B1": "B1-s0_0", "B2": "B2-s0_1"}),
        ("C2", {"B1": "B1-s1_1", "B2": "B2-s2_0"}),
    ]
    _, a = _build(["B1", "B2"], cmps, 3, 3, snaps, seed=7)
    _, b = _build(["B1", "B2"], cmps, 3, 3, snaps, seed=7)
    assert a.placement == b.placement


def test_build_different_seeds_stay_optimal():
    snaps = {
        "B1": _grid_snapshot("B1", 3, 4, [["GA", "GB", "GA", "GB"]] * 3),
        "B2": _grid_snapshot("B2", 3, 4, [["GB", "GA", "GB", "GA"]] * 3),
    }
    cmps = [
        ("C1", {"B1": "B1-s0_0", "B2": "B2-s0_1"}),
        ("C2", {"B1": "B1-s2_2", "B2": "B2-s1_3"}),
    ]
    _items, (adj_star, cov_star), _ = _brute_force(
        ["B1", "B2"], cmps, 3, 4, snaps)
    for seed in range(6):
        _, layout = _build(["B1", "B2"], cmps, 3, 4, snaps, seed=seed)
        assert len(layout.adjacent_edges) == adj_star
        assert sum(len(rs) + len(cs) for rs, cs in layout.coverage.values()) == cov_star


def test_build_respects_blocked_wells():
    snaps = {
        "B1": _grid_snapshot("B1", 3, 3, [["GA"] * 3] * 3),
        "B2": _grid_snapshot("B2", 3, 3, [["GB"] * 3] * 3),
    }
    cmps = [("C1", {"B1": "B1-s0_0", "B2": "B2-s0_0"})]
    _, layout = _build(["B1", "B2"], cmps, 3, 3, snaps, blocked=("B2", "C3"))
    assert "B2" not in layout.placement.values()
    assert "C3" not in layout.placement.values()


def test_validate_rejects_duplicate_batch():
    snaps = {
        "B1": _grid_snapshot("B1", 2, 2, [["GA", "GB"]] * 2),
    }
    req = cross_batch.CrossBatchPlanRequest(
        request_key="k",
        batches=[{"batch_id": "B1", "expected_revision": 1},
                 {"batch_id": "B1", "expected_revision": 1}],
        comparisons=[{"comparison_id": "C1",
                      "samples": {"B1": "B1-s0_0"}}],
        plate={"type": "custom", "rows": 2, "columns": 2},
    )
    with pytest.raises(cross_batch.CrossBatchPlanInvalid) as exc:
        cross_batch._validate_and_build_items(req, {"B1": {"snapshot": snaps["B1"]}})
    assert exc.value.categories == ["CROSS_BATCH_BATCH_DUPLICATED"]


def test_validate_rejects_missing_batch_sample():
    snaps = {
        "B1": _grid_snapshot("B1", 2, 2, [["GA", "GB"]] * 2),
        "B2": _grid_snapshot("B2", 2, 2, [["GA", "GB"]] * 2),
    }
    req = cross_batch.CrossBatchPlanRequest(
        request_key="k",
        batches=[{"batch_id": "B1", "expected_revision": 1},
                 {"batch_id": "B2", "expected_revision": 1}],
        comparisons=[{"comparison_id": "C1",
                      "samples": {"B1": "B1-s0_0"}}],  # B2 missing
        plate={"type": "custom", "rows": 2, "columns": 2},
    )
    with pytest.raises(cross_batch.CrossBatchPlanInvalid) as exc:
        cross_batch._validate_and_build_items(
            req, {"B1": {"snapshot": snaps["B1"]}, "B2": {"snapshot": snaps["B2"]}})
    assert exc.value.categories == ["CROSS_BATCH_COMPARISON_INCOMPLETE"]


def test_validate_rejects_unknown_batch_in_comparison():
    snaps = {
        "B1": _grid_snapshot("B1", 2, 2, [["GA", "GB"]] * 2),
        "B2": _grid_snapshot("B2", 2, 2, [["GA", "GB"]] * 2),
    }
    req = cross_batch.CrossBatchPlanRequest(
        request_key="k",
        batches=[{"batch_id": "B1", "expected_revision": 1},
                 {"batch_id": "B2", "expected_revision": 1}],
        comparisons=[{"comparison_id": "C1",
                      "samples": {"B1": "B1-s0_0", "BX": "x"}}],  # B2 also missing
        plate={"type": "custom", "rows": 2, "columns": 2},
    )
    with pytest.raises(cross_batch.CrossBatchPlanInvalid) as exc:
        cross_batch._validate_and_build_items(
            req, {"B1": {"snapshot": snaps["B1"]}, "B2": {"snapshot": snaps["B2"]}})
    assert "CROSS_BATCH_COMPARISON_INCOMPLETE" in exc.value.categories
    messages = " ".join(exc.value.messages)
    assert "not among the request's batches" in messages
    assert "missing the sample for batch 'B2'" in messages


def test_validate_rejects_sample_used_twice_in_one_batch():
    snaps = {
        "B1": _grid_snapshot("B1", 2, 2, [["GA", "GB"]] * 2),
        "B2": _grid_snapshot("B2", 2, 2, [["GA", "GB"]] * 2),
    }
    req = cross_batch.CrossBatchPlanRequest(
        request_key="k",
        batches=[{"batch_id": "B1", "expected_revision": 1},
                 {"batch_id": "B2", "expected_revision": 1}],
        comparisons=[
            {"comparison_id": "C1",
             "samples": {"B1": "B1-s0_0", "B2": "B2-s0_1"}},
            {"comparison_id": "C2",
             "samples": {"B1": "B1-s0_0", "B2": "B2-s1_0"}},
        ],
        plate={"type": "custom", "rows": 3, "columns": 3},
    )
    with pytest.raises(cross_batch.CrossBatchPlanInvalid) as exc:
        cross_batch._validate_and_build_items(
            req, {"B1": {"snapshot": snaps["B1"]}, "B2": {"snapshot": snaps["B2"]}})
    assert exc.value.categories == ["CROSS_BATCH_SAMPLE_DUPLICATED"]


def test_validate_rejects_duplicate_comparison_id():
    snaps = {
        "B1": _grid_snapshot("B1", 2, 2, [["GA", "GB"]] * 2),
        "B2": _grid_snapshot("B2", 2, 2, [["GA", "GB"]] * 2),
    }
    req = cross_batch.CrossBatchPlanRequest(
        request_key="k",
        batches=[{"batch_id": "B1", "expected_revision": 1},
                 {"batch_id": "B2", "expected_revision": 1}],
        comparisons=[
            {"comparison_id": "C1",
             "samples": {"B1": "B1-s0_0", "B2": "B2-s0_0"}},
            {"comparison_id": "C1",
             "samples": {"B1": "B1-s0_1", "B2": "B2-s0_1"}},
        ],
        plate={"type": "custom", "rows": 3, "columns": 3},
    )
    with pytest.raises(cross_batch.CrossBatchPlanInvalid) as exc:
        cross_batch._validate_and_build_items(
            req, {"B1": {"snapshot": snaps["B1"]}, "B2": {"snapshot": snaps["B2"]}})
    assert "CROSS_BATCH_COMPARISON_DUPLICATED" in exc.value.categories


def test_validate_rejects_unknown_sample():
    snaps = {
        "B1": _grid_snapshot("B1", 2, 2, [["GA", "GB"]] * 2),
        "B2": _grid_snapshot("B2", 2, 2, [["GA", "GB"]] * 2),
    }
    req = cross_batch.CrossBatchPlanRequest(
        request_key="k",
        batches=[{"batch_id": "B1", "expected_revision": 1},
                 {"batch_id": "B2", "expected_revision": 1}],
        comparisons=[{"comparison_id": "C1",
                      "samples": {"B1": "nope", "B2": "B2-s0_0"}}],
        plate={"type": "custom", "rows": 2, "columns": 2},
    )
    with pytest.raises(cross_batch.CrossBatchPlanInvalid) as exc:
        cross_batch._validate_and_build_items(
            req, {"B1": {"snapshot": snaps["B1"]}, "B2": {"snapshot": snaps["B2"]}})
    assert exc.value.categories == ["CROSS_BATCH_SAMPLE_UNKNOWN"]


def test_validate_rejects_scan_mismatch_and_null_reading():
    snaps = {
        "B1": _grid_snapshot("B1", 2, 2, [["GA", "GB"]] * 2),
        "B2": _grid_snapshot("B2", 2, 2, [["GA", "GB"]] * 2),
    }
    # B1: scan displaced (A1 carries a different sample).
    snaps["B1"]["scan"]["A1"] = "B1-s0_1"
    # B2: null reading at A1.
    snaps["B2"]["readings"]["A1"] = None
    req = cross_batch.CrossBatchPlanRequest(
        request_key="k",
        batches=[{"batch_id": "B1", "expected_revision": 1},
                 {"batch_id": "B2", "expected_revision": 1}],
        comparisons=[{"comparison_id": "C1",
                      "samples": {"B1": "B1-s0_0", "B2": "B2-s0_0"}}],
        plate={"type": "custom", "rows": 2, "columns": 2},
    )
    with pytest.raises(cross_batch.CrossBatchPlanInvalid) as exc:
        cross_batch._validate_and_build_items(
            req, {"B1": {"snapshot": snaps["B1"]}, "B2": {"snapshot": snaps["B2"]}})
    assert exc.value.categories == ["CROSS_BATCH_SAMPLE_NOT_ELIGIBLE"]
    assert len(exc.value.messages) == 2


def test_parse_blocked_rejects_bad_well():
    with pytest.raises(cross_batch.CrossBatchPlanInvalid) as exc:
        cross_batch._parse_blocked(["Z99"], 3, 3)
    assert exc.value.categories == ["WELL_REFERENCE_INVALID"]


# ============================================================ HTTP endpoint


def test_plan_success_full_body():
    c1, c2 = _two_certs()
    r = _post()
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["plan_id"] == cross_batch._plan_id("cmp-1")
    assert body["request_key"] == "cmp-1"
    assert body["status"] == "ACTIVE"
    assert body["plate"] == {
        "type": "custom", "rows": 3, "columns": 3,
        "blocked_wells": [],
        "available_wells": [
            "A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"],
    }
    sources = {(s["batch_id"], s["revision"], s["digest"], s["status"])
               for s in body["sources"]}
    assert sources == {
        ("BATCH-1", 1, c1["digest"], "ACTIVE"),
        ("BATCH-2", 1, c2["digest"], "ACTIVE"),
    }
    # 4 samples placed one per well, none blocked.
    assert len(body["layout"]) == 4
    wells = list(body["layout"].values())
    assert len(set(wells)) == 4
    assert set(body["wells"]) == set(wells)
    # Layout keys are qualified (batch:sample).
    assert set(body["layout"]) == {
        "BATCH-1:gA1", "BATCH-1:gB3", "BATCH-2:gA2", "BATCH-2:gB4"}
    # Each comparison's members match the requested sample picks.
    by_cmp = {c["comparison_id"]: c for c in body["comparisons"]}
    assert {m["batch_id"]: m["sample_id"] for m in by_cmp["C1"]["members"]} == {
        "BATCH-1": "gA1", "BATCH-2": "gA2"}
    assert {m["batch_id"]: m["sample_id"] for m in by_cmp["C2"]["members"]} == {
        "BATCH-1": "gB3", "BATCH-2": "gB4"}
    # samples are row-major; groups carried from snapshots.
    positions = [cross_batch.parse_well(w, 3, 3) for w in body["wells"]]
    assert positions == sorted(positions)
    for e in body["samples"]:
        assert set(e) == {"sample_id", "well", "batch_id",
                          "comparison_id", "group"}
        expected_group = "GA" if e["sample_id"].startswith("gA") else "GB"
        assert e["group"] == expected_group
    # Optimiser objective sanity: 2 comparisons on 2 free dominoes -> 2 edges.
    assert body["metrics"]["same_comparison_adjacent_edges_count"] >= 1
    stages = [s["stage"] for s in body["solver"]["stages"]]
    assert stages == ["1_comparison_adjacency", "2_batch_coverage", "3_tiebreak"]


def test_plan_maximises_same_comparison_edges():
    _two_certs()
    r = _post(plate={"type": "custom", "rows": 2, "columns": 2})
    assert r.status_code == 201, r.text
    body = r.json()
    # All 4 wells used; two comparisons can each be placed as a domino,
    # and the maximiser must find both edges.
    assert body["metrics"]["same_comparison_adjacent_edges_count"] == 2
    by_cmp = {c["comparison_id"]: c for c in body["comparisons"]}
    for c in body["comparisons"]:
        assert c["same_comparison_adjacent_edges_count"] == 1
        assert len(by_cmp[c["comparison_id"]]["wells"]) == 2


def test_plan_persists_plan_and_idempotency_record_in_one_transaction(fresh_store):
    _two_certs()
    r = _post()
    assert r.status_code == 201
    conn = sqlite3.connect(str(fresh_store))
    rows = conn.execute(
        "SELECT plan_id, request_key, batch_count, sources, seed, plan_body "
        "FROM cross_batch_plans").fetchall()
    keys = conn.execute(
        "SELECT response_status, response_body FROM idempotency_keys "
        "WHERE request_key = 'cmp-1'").fetchall()
    conn.close()
    assert len(rows) == 1 and len(keys) == 1
    plan_id, request_key, batch_count, sources, seed, plan_body = rows[0]
    assert plan_id == r.json()["plan_id"]
    assert request_key == "cmp-1" and batch_count == 2 and seed == 42
    stored_sources = json.loads(sources)
    assert {s["batch_id"] for s in stored_sources} == {"BATCH-1", "BATCH-2"}
    assert all(s["revision"] == 1 for s in stored_sources)
    assert keys[0][0] == 201
    assert json.loads(keys[0][1]) == r.json()
    assert json.loads(plan_body) == r.json()


def test_plan_replay_same_key_same_content_returns_recorded_response():
    _two_certs()
    payload = _comparison_payload()
    b1 = client.post("/api/v1/certificates/comparison-plans", json=payload)
    b2 = client.post("/api/v1/certificates/comparison-plans", json=payload)
    assert b1.status_code == 201 and b2.status_code == 201
    assert b1.json() == b2.json()


def test_plan_same_key_different_content_is_409_and_saves_nothing(fresh_store):
    _two_certs()
    client.post("/api/v1/certificates/comparison-plans",
                json=_comparison_payload())
    changed = _comparison_payload(seed=43)
    r = client.post("/api/v1/certificates/comparison-plans", json=changed)
    assert r.status_code == 409
    assert r.json()["error"] == "REQUEST_KEY_CONFLICT"
    assert r.json()["request_key"] == "cmp-1"
    conn = sqlite3.connect(str(fresh_store))
    n_plans = conn.execute("SELECT COUNT(*) FROM cross_batch_plans").fetchone()[0]
    n_keys = conn.execute(
        "SELECT COUNT(*) FROM idempotency_keys WHERE request_key = 'cmp-1'"
    ).fetchone()[0]
    conn.close()
    assert n_plans == 1 and n_keys == 1


def test_plan_422_is_recorded_and_replays_identically():
    _two_certs()
    payload = _comparison_payload(plate={"type": "custom", "rows": 1, "columns": 2})
    b1 = client.post("/api/v1/certificates/comparison-plans", json=payload)
    b2 = client.post("/api/v1/certificates/comparison-plans", json=payload)
    assert b1.status_code == 422 and b2.status_code == 422
    assert b1.json() == b2.json()
    assert "CROSS_BATCH_CAPACITY_EXCEEDED" in b1.json()["infeasibility_categories"]
    changed = json.loads(json.dumps(payload)); changed["seed"] = 9
    assert client.post("/api/v1/certificates/comparison-plans",
                       json=changed).status_code == 409


def test_plan_unknown_batch_is_404():
    _create_cert("BATCH-1")
    r = _post()
    assert r.status_code == 404
    assert r.json()["error"] == "CERTIFICATE_NOT_FOUND"
    assert r.json()["batch_id"] == "BATCH-2"


def test_plan_revoked_certificate_is_409(fresh_store):
    cert = _create_cert("BATCH-1")
    _create_cert("BATCH-2")
    client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-1", "request_key": "rv-1", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items() if w != "A1"},
        "readings": _readings(),
    })
    r = _post()
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_REVOKED"
    assert body["certificate_id"] == cert["certificate_id"]
    assert body["batch_id"] == "BATCH-1"
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute("SELECT COUNT(*) FROM cross_batch_plans").fetchone()[0] == 0
    conn.close()


def test_plan_stale_expected_revision_is_409_and_saves_nothing(fresh_store):
    _two_certs()
    client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-1", "request_key": "rv-1", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "readings": _readings(7.0),
    })
    r = _post()
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "CERTIFICATE_REVISION_MISMATCH"
    assert body["current_revision"] == 2 and body["expected_revision"] == 1
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute("SELECT COUNT(*) FROM cross_batch_plans").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM idempotency_keys WHERE request_key = 'cmp-1'"
    ).fetchone()[0] == 0
    conn.close()
    # Same key with the corrected revision succeeds (state 409s not recorded).
    ok = _post(batches=[
        {"batch_id": "BATCH-1", "expected_revision": 2},
        {"batch_id": "BATCH-2", "expected_revision": 1}])
    assert ok.status_code == 201, ok.text
    revisions = {s["batch_id"]: s["revision"] for s in ok.json()["sources"]}
    assert revisions == {"BATCH-1": 2, "BATCH-2": 1}


def test_plan_replay_after_revision_advances_returns_409():
    _two_certs()
    payload = _comparison_payload()
    first = client.post("/api/v1/certificates/comparison-plans", json=payload)
    assert first.status_code == 201
    client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-2", "request_key": "rv-1", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "readings": _readings(3.0),
    })
    replay = client.post("/api/v1/certificates/comparison-plans", json=payload)
    assert replay.status_code == 409
    assert replay.json()["error"] == "CERTIFICATE_REVISION_MISMATCH"
    assert replay.json()["current_revision"] == 2


def test_plan_duplicate_batch_in_request_is_422(fresh_store):
    _two_certs()
    payload = _comparison_payload(
        batches=[
            {"batch_id": "BATCH-1", "expected_revision": 1},
            {"batch_id": "BATCH-1", "expected_revision": 1}],
        comparisons=[{"comparison_id": "C1",
                      "samples": {"BATCH-1": "gA1"}}])
    r = client.post("/api/v1/certificates/comparison-plans", json=payload)
    assert r.status_code == 422
    cats = r.json()["infeasibility_categories"]
    assert cats == ["CROSS_BATCH_BATCH_DUPLICATED"]
    conn = sqlite3.connect(str(fresh_store))
    assert conn.execute("SELECT COUNT(*) FROM cross_batch_plans").fetchone()[0] == 0
    conn.close()
    # Content-dependent 422 is recorded under the key and replays identically.
    replay = client.post("/api/v1/certificates/comparison-plans", json=payload)
    assert replay.status_code == 422 and replay.json() == r.json()


def test_plan_comparison_missing_batch_sample_is_422():
    _two_certs()
    bad = [{"comparison_id": "C1",
            "samples": {"BATCH-1": "gA1"}}]  # BATCH-2 omitted
    r = _post(comparisons=bad)
    assert r.status_code == 422
    assert "CROSS_BATCH_COMPARISON_INCOMPLETE" in r.json()["infeasibility_categories"]


def test_plan_same_sample_in_two_comparisons_is_422():
    _two_certs()
    bad = [
        {"comparison_id": "C1",
         "samples": {"BATCH-1": "gA1", "BATCH-2": "gA2"}},
        {"comparison_id": "C2",
         "samples": {"BATCH-1": "gA1", "BATCH-2": "gB4"}},
    ]
    r = _post(comparisons=bad)
    assert r.status_code == 422
    assert "CROSS_BATCH_SAMPLE_DUPLICATED" in r.json()["infeasibility_categories"]


def test_plan_unknown_sample_is_422():
    _two_certs()
    r = _post(comparisons=[
        {"comparison_id": "C1",
         "samples": {"BATCH-1": "ghost", "BATCH-2": "gA2"}}])
    assert r.status_code == 422
    assert "CROSS_BATCH_SAMPLE_UNKNOWN" in r.json()["infeasibility_categories"]


def _tamper_snapshot(db_path, batch_id, mutate):
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT snapshot FROM certificates WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    snap = json.loads(row[0])
    mutate(snap)
    snap_json = certificates._canonical_json(snap)
    digest = certificates._sha256(snap_json)
    conn.execute(
        "UPDATE certificates SET snapshot = ?, digest = ? WHERE batch_id = ?",
        (snap_json.decode(), digest, batch_id))
    conn.execute(
        "UPDATE certificate_revisions SET snapshot = ?, digest = ? "
        "WHERE batch_id = ? AND revision = 1",
        (snap_json.decode(), digest, batch_id))
    conn.commit()
    conn.close()
    certificates.reset_stores()


def test_plan_scan_mismatched_sample_is_422(fresh_store):
    _two_certs()
    _tamper_snapshot(fresh_store, "BATCH-1",
                     lambda s: s["scan"].__setitem__("A1", "gA2"))
    r = _post()
    assert r.status_code == 422
    assert "CROSS_BATCH_SAMPLE_NOT_ELIGIBLE" in r.json()["infeasibility_categories"]
    assert "gA1" in r.json()["messages"][0]


def test_plan_null_reading_sample_is_422(fresh_store):
    _two_certs()
    _tamper_snapshot(fresh_store, "BATCH-2",
                     lambda s: s["readings"].__setitem__("A3", None))
    r = _post()
    assert r.status_code == 422
    assert "CROSS_BATCH_SAMPLE_NOT_ELIGIBLE" in r.json()["infeasibility_categories"]


def test_plan_blocked_target_wells_are_unused():
    _two_certs()
    r = _post(blocked_wells=["A1", "B2"])
    assert r.status_code == 201, r.text
    wells = set(r.json()["wells"])
    assert wells.isdisjoint({"A1", "B2"})
    assert r.json()["plate"]["blocked_wells"] == ["A1", "B2"]


def test_plan_bad_blocked_well_is_422():
    _two_certs()
    r = _post(blocked_wells=["Z9"])
    assert r.status_code == 422
    assert "WELL_REFERENCE_INVALID" in r.json()["infeasibility_categories"]


def test_plan_capacity_exceeded_is_422():
    _two_certs()
    r = _post(plate={"type": "custom", "rows": 2, "columns": 2},
              blocked_wells=["A1"])  # 3 available for 4 samples
    assert r.status_code == 422
    assert "CROSS_BATCH_CAPACITY_EXCEEDED" in r.json()["infeasibility_categories"]


def test_plan_seed_determinism_across_keys_and_restart():
    _two_certs()
    p1 = _comparison_payload(request_key="seed-a")
    p2 = _comparison_payload(request_key="seed-b")
    b1 = client.post("/api/v1/certificates/comparison-plans", json=p1)
    b2 = client.post("/api/v1/certificates/comparison-plans", json=p2)
    assert b1.json()["layout"] == b2.json()["layout"]

    certificates.reset_stores()
    b3 = client.post("/api/v1/certificates/comparison-plans",
                     json=_comparison_payload(request_key="seed-c"))
    assert b3.json()["layout"] == b1.json()["layout"]


def test_plan_supports_three_and_four_batches():
    for n in (3, 4):
        for k in range(1, n + 1):
            _create_cert(f"BATCH-{k}", key=f"create-{k}")
        sids = ["gA1", "gB3", "gA5", "gB7"]
        comparisons = [
            {"comparison_id": "C1",
             "samples": {f"BATCH-{k}": sids[k - 1] for k in range(1, n + 1)}},
        ]
        r = client.post("/api/v1/certificates/comparison-plans", json={
            "request_key": f"cmp-{n}",
            "batches": [{"batch_id": f"BATCH-{k}", "expected_revision": 1}
                        for k in range(1, n + 1)],
            "comparisons": comparisons,
            "plate": {"type": "custom", "rows": 2, "columns": 2},
            "seed": 1,
        })
        assert r.status_code == 201, r.text
        assert len(r.json()["sources"]) == n
        assert len(r.json()["layout"]) == n


def test_plan_same_key_reused_across_other_endpoint_conflicts():
    _two_certs()
    # Burn the key on the retest endpoint first.
    rr = client.post("/api/v1/certificates/retest-plans", json={
        "batch_id": "BATCH-1", "request_key": "shared",
        "expected_revision": 1, "group_counts": {"GA": 1}, "seed": 0})
    assert rr.status_code == 201
    r = _post(request_key="shared")
    assert r.status_code == 409
    assert r.json()["error"] == "REQUEST_KEY_CONFLICT"


def test_plan_revision2_sources_revision2_digest(fresh_store):
    c1, c2 = _two_certs()
    client.post("/api/v1/certificates/reverify", json={
        "batch_id": "BATCH-1", "request_key": "rv-1", "expected_revision": 1,
        "scan": {w: s for s, w in LAYOUT_4X4.items()},
        "readings": _readings(2.0),
    })
    r = _post(batches=[
        {"batch_id": "BATCH-1", "expected_revision": 2},
        {"batch_id": "BATCH-2", "expected_revision": 1}])
    assert r.status_code == 201
    conn = sqlite3.connect(str(fresh_store))
    rev2 = conn.execute(
        "SELECT digest FROM certificate_revisions "
        "WHERE batch_id = 'BATCH-1' AND revision = 2").fetchone()[0]
    conn.close()
    src = {s["batch_id"]: s for s in r.json()["sources"]}
    assert src["BATCH-1"]["revision"] == 2 and src["BATCH-1"]["digest"] == rev2
    assert src["BATCH-2"]["revision"] == 1 and src["BATCH-2"]["digest"] == c2["digest"]


# ------------------------------------------------------------ request shape


def test_request_requires_two_to_four_batches():
    _two_certs()
    one = _comparison_payload(
        batches=[{"batch_id": "BATCH-1", "expected_revision": 1}],
        comparisons=[{"comparison_id": "C1",
                      "samples": {"BATCH-1": "gA1"}}])
    r = client.post("/api/v1/certificates/comparison-plans", json=one)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_request_rejects_unknown_field():
    payload = _comparison_payload()
    payload["extra"] = 1
    r = client.post("/api/v1/certificates/comparison-plans", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_request_missing_request_key_is_validation_error():
    payload = _comparison_payload()
    del payload["request_key"]
    r = client.post("/api/v1/certificates/comparison-plans", json=payload)
    assert r.status_code == 422
    assert r.json()["infeasibility_categories"] == ["REQUEST_VALIDATION_ERROR"]


def test_request_empty_comparisons_is_validation_error():
    payload = _comparison_payload(comparisons=[])
    r = client.post("/api/v1/certificates/comparison-plans", json=payload)
    assert r.status_code == 422


def test_existing_endpoints_still_available_after_comparison_plans():
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/api/v1/arrangements/batch-release",
                    json=_release_payload())
    assert r.status_code == 200 and r.json()["verdict"] == "RELEASED"
