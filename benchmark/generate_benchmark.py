#!/usr/bin/env python3
"""
PowerBench Synthetic Benchmark Data Generator
==============================================
Generates synthetic manufacturing test facility data for the PowerSense
framework. Power profiles are calibrated to SPECpower_ssj2008 published
results. Facility topology follows NEC/NFPA 70 standards.

Usage:
    python generate_benchmark.py --output data/ --days 30
    python generate_benchmark.py --output data/ --days 30 --slots 2000 --seed 42
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ── Product Definitions (calibrated to SPECpower_ssj2008) ──────────────


@dataclass
class ProductType:
    """A product type with its power profile parameters."""
    id: int
    name: str
    nameplate_kw: float
    avg_power_kw: float
    peak_to_avg: float
    phases: int
    test_duration_hours: float
    cv: float  # coefficient of variation (sigma / mu)


DEFAULT_PRODUCTS = [
    ProductType(1,  "Type-A", 1.05, 0.62, 1.69, 4, 4.5, 0.08),
    ProductType(2,  "Type-B", 0.85, 0.51, 1.67, 3, 3.0, 0.06),
    ProductType(3,  "Type-C", 1.40, 0.88, 1.59, 5, 6.0, 0.10),
    ProductType(4,  "Type-D", 0.72, 0.50, 1.44, 3, 2.5, 0.05),
    ProductType(5,  "Type-E", 1.80, 1.12, 1.61, 6, 7.5, 0.12),
    ProductType(6,  "Type-F", 0.95, 0.58, 1.64, 4, 4.0, 0.07),
    ProductType(7,  "Type-G", 1.20, 0.75, 1.60, 5, 5.5, 0.09),
    ProductType(8,  "Type-H", 0.65, 0.46, 1.41, 3, 2.0, 0.05),
    ProductType(9,  "Type-I", 1.55, 0.98, 1.58, 5, 6.5, 0.11),
    ProductType(10, "Type-J", 2.10, 1.20, 1.75, 6, 8.0, 0.15),
    ProductType(11, "Type-K", 0.80, 0.53, 1.51, 4, 3.5, 0.06),
    ProductType(12, "Type-L", 1.10, 0.70, 1.57, 4, 5.0, 0.08),
]

PRODUCTION_MIX = [0.12, 0.10, 0.08, 0.11, 0.05, 0.09,
                  0.07, 0.12, 0.06, 0.04, 0.08, 0.08]


# ── Facility Topology ──────────────────────────────────────────────────


@dataclass
class FacilityNode:
    """A node in the power distribution hierarchy."""
    id: int
    level: int          # 0=source, 1=xfmr, 2=panel, 3=circuit, 4=slot
    level_name: str
    capacity_kw: float
    parent_id: Optional[int]
    children: list = field(default_factory=list)


def build_topology(n_slots: int = 5000,
                   facility_capacity_kw: float = 4500.0) -> list[FacilityNode]:
    """
    Build a 5-level power distribution hierarchy following NEC/NFPA 70
    design practices.

    Hierarchy: Source -> Transformers -> Panels -> Circuits -> Slots
    """
    nodes = []
    node_id = 0

    # Level 0: Source
    source = FacilityNode(node_id, 0, "Source", facility_capacity_kw, None)
    nodes.append(source)
    node_id += 1

    # Level 1: Transformers
    n_xfmr = 8
    xfmr_cap = facility_capacity_kw / n_xfmr
    xfmr_ids = []
    for _ in range(n_xfmr):
        xfmr = FacilityNode(node_id, 1, "Transformer", xfmr_cap, source.id)
        source.children.append(node_id)
        nodes.append(xfmr)
        xfmr_ids.append(node_id)
        node_id += 1

    # Level 2: Distribution Panels (6 per transformer)
    panels_per_xfmr = 6
    panel_ids = []
    for xid in xfmr_ids:
        panel_cap = nodes[xid].capacity_kw / panels_per_xfmr
        for _ in range(panels_per_xfmr):
            panel = FacilityNode(node_id, 2, "Panel", panel_cap, xid)
            nodes[xid].children.append(node_id)
            nodes.append(panel)
            panel_ids.append(node_id)
            node_id += 1

    # Level 3: Branch Circuits (8 per panel)
    circuits_per_panel = 8
    circuit_ids = []
    for pid in panel_ids:
        circ_cap = nodes[pid].capacity_kw / circuits_per_panel
        for _ in range(circuits_per_panel):
            circ = FacilityNode(node_id, 3, "Circuit", circ_cap, pid)
            nodes[pid].children.append(node_id)
            nodes.append(circ)
            circuit_ids.append(node_id)
            node_id += 1

    # Level 4: Test Slots (distributed across circuits)
    slots_assigned = 0
    for i, cid in enumerate(circuit_ids):
        if i < len(circuit_ids) - 1:
            slots_this = n_slots // len(circuit_ids)
        else:
            slots_this = n_slots - slots_assigned
        for _ in range(slots_this):
            slot = FacilityNode(node_id, 4, "Slot", 0.0, cid)
            nodes[cid].children.append(node_id)
            nodes.append(slot)
            node_id += 1
        slots_assigned += slots_this

    return nodes


# ── Power Profile Generation ──────────────────────────────────────────


def phase_shape(t_norm: np.ndarray, phase_idx: int, n_phases: int) -> np.ndarray:
    """
    Generate a normalized shape function g_phi(t/T_phi) for a test phase.
    Uses a combination of ramp-up, plateau, and ramp-down patterns
    that vary by phase position in the test sequence.
    """
    if phase_idx == 0:
        # Initialization: ramp up
        return np.clip(2.0 * t_norm, 0, 1)
    elif phase_idx == n_phases - 1:
        # Final phase: ramp down
        return np.clip(1.0 - t_norm, 0, 1)
    elif phase_idx == n_phases // 2:
        # Stress test (mid-sequence): sustained high
        return 0.85 + 0.15 * np.sin(np.pi * t_norm)
    else:
        # Functional test: moderate with variation
        return 0.5 + 0.3 * np.sin(2 * np.pi * t_norm)


def generate_power_trace(product: ProductType,
                         rng: np.random.Generator,
                         resolution_minutes: float = 1.0) -> np.ndarray:
    """
    Generate a stochastic power consumption trace for a single UUT test.

    Implements Eqs. (2)-(3) from the paper:
        p(t) = mu(t) + epsilon(t)
        mu(t) = p_base + (p_peak - p_base) * g_phi(t / T_phi)
    """
    total_minutes = product.test_duration_hours * 60
    n_samples = int(total_minutes / resolution_minutes)
    trace = np.zeros(n_samples)

    p_base = product.avg_power_kw * 0.6
    p_peak = product.avg_power_kw * product.peak_to_avg
    sigma = product.avg_power_kw * product.cv

    # Divide total duration into phases
    phase_lengths = np.diff(
        np.round(np.linspace(0, n_samples, product.phases + 1)).astype(int)
    )

    idx = 0
    for phi, length in enumerate(phase_lengths):
        t_norm = np.linspace(0, 1, length)
        g = phase_shape(t_norm, phi, product.phases)
        mu = p_base + (p_peak - p_base) * g
        noise = rng.normal(0, sigma, length)
        trace[idx:idx + length] = np.maximum(0.05, mu + noise)
        idx += length

    return trace


# ── Production Queue Generation ───────────────────────────────────────


def generate_production_queue(products: list[ProductType],
                              mix_weights: list[float],
                              daily_rate_range: tuple[int, int],
                              n_days: int,
                              rng: np.random.Generator) -> list[dict]:
    """Generate a production queue with arrival times and product types."""
    queue = []
    mix_probs = np.array(mix_weights) / sum(mix_weights)

    for day in range(n_days):
        daily_rate = rng.integers(daily_rate_range[0], daily_rate_range[1] + 1)
        # Arrivals spread across 3 shifts (24h) with higher density in shifts 1-2
        shift_weights = [0.4, 0.4, 0.2]  # day, evening, night

        for shift_idx, sw in enumerate(shift_weights):
            n_this_shift = int(daily_rate * sw)
            shift_start = day * 24 * 60 + shift_idx * 8 * 60  # minutes
            shift_end = shift_start + 8 * 60

            arrivals = rng.uniform(shift_start, shift_end, n_this_shift)
            arrivals.sort()

            for arr in arrivals:
                prod_idx = rng.choice(len(products), p=mix_probs)
                queue.append({
                    "arrival_minute": float(arr),
                    "product_id": products[prod_idx].id,
                    "product_name": products[prod_idx].name,
                    "test_duration_hours": products[prod_idx].test_duration_hours,
                    "nameplate_kw": products[prod_idx].nameplate_kw,
                    "avg_power_kw": products[prod_idx].avg_power_kw,
                })

    queue.sort(key=lambda x: x["arrival_minute"])
    return queue


# ── Snapshot Generation (for GAT training) ────────────────────────────


def generate_power_snapshots(topology: list[FacilityNode],
                             products: list[ProductType],
                             mix_weights: list[float],
                             n_snapshots: int,
                             rng: np.random.Generator) -> list[dict]:
    """
    Generate random facility power snapshots for GAT training.
    Each snapshot assigns random UUTs to slots at random test phases
    and computes aggregate power at each hierarchy level.
    """
    slot_nodes = [n for n in topology if n.level == 4]
    mix_probs = np.array(mix_weights) / sum(mix_weights)
    snapshots = []

    for snap_idx in range(n_snapshots):
        # Random occupancy: 40-90% of slots active
        occupancy_rate = rng.uniform(0.4, 0.9)
        n_active = int(len(slot_nodes) * occupancy_rate)
        active_slots = rng.choice(len(slot_nodes), n_active, replace=False)

        slot_power = {}
        for si in active_slots:
            slot = slot_nodes[si]
            prod = products[rng.choice(len(products), p=mix_probs)]
            # Random point in test → random power draw
            phase_frac = rng.uniform(0, 1)
            base_p = prod.avg_power_kw * 0.6
            peak_p = prod.avg_power_kw * prod.peak_to_avg
            shape_val = rng.uniform(0.3, 1.0)  # simplified
            mu = base_p + (peak_p - base_p) * shape_val
            noise = rng.normal(0, prod.avg_power_kw * prod.cv)
            power = max(0.05, mu + noise)
            slot_power[slot.id] = power

        # Aggregate up the hierarchy
        node_power = {n.id: 0.0 for n in topology}
        for sid, pw in slot_power.items():
            node_power[sid] = pw

        # Bottom-up aggregation
        for level in [3, 2, 1, 0]:
            for n in topology:
                if n.level == level:
                    node_power[n.id] = sum(
                        node_power[c] for c in n.children
                    )

        snapshots.append({
            "snapshot_id": snap_idx,
            "occupancy_rate": occupancy_rate,
            "n_active_slots": n_active,
            "total_power_kw": node_power[0],
            "node_powers": {
                str(nid): round(pw, 4) for nid, pw in node_power.items()
                if pw > 0
            },
        })

        if (snap_idx + 1) % 10000 == 0:
            print(f"  Generated {snap_idx + 1}/{n_snapshots} snapshots")

    return snapshots


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="PowerBench Synthetic Benchmark Generator"
    )
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory")
    parser.add_argument("--days", type=int, default=30,
                        help="Simulation horizon in days")
    parser.add_argument("--slots", type=int, default=5000,
                        help="Number of test slots")
    parser.add_argument("--capacity-kw", type=float, default=4500.0,
                        help="Total facility capacity in kW")
    parser.add_argument("--snapshots", type=int, default=100000,
                        help="Number of GAT training snapshots")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed")
    parser.add_argument("--traces", type=int, default=100,
                        help="Number of sample power traces to generate")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"PowerBench Generator — seed={args.seed}, slots={args.slots}, "
          f"days={args.days}")
    print("=" * 60)

    # 1. Build topology
    print("\n[1/4] Building facility topology...")
    topology = build_topology(args.slots, args.capacity_kw)
    level_counts = {}
    for n in topology:
        level_counts[n.level_name] = level_counts.get(n.level_name, 0) + 1
    for name, count in level_counts.items():
        print(f"  {name}: {count}")

    topo_export = [
        {"id": n.id, "level": n.level, "type": n.level_name,
         "capacity_kw": round(n.capacity_kw, 4), "parent": n.parent_id,
         "children": n.children}
        for n in topology
    ]
    with open(out / "topology.json", "w") as f:
        json.dump(topo_export, f, indent=2)
    print(f"  → Saved topology.json ({len(topology)} nodes)")

    # 2. Generate production queue
    print(f"\n[2/4] Generating {args.days}-day production queue...")
    queue = generate_production_queue(
        DEFAULT_PRODUCTS, PRODUCTION_MIX, (800, 1200), args.days, rng
    )
    with open(out / "production_queue.json", "w") as f:
        json.dump(queue, f, indent=2)
    print(f"  → Saved production_queue.json ({len(queue)} UUTs)")

    # 3. Generate sample power traces
    print(f"\n[3/4] Generating {args.traces} sample power traces...")
    traces = {}
    for i in range(args.traces):
        prod = DEFAULT_PRODUCTS[rng.choice(len(DEFAULT_PRODUCTS),
                                           p=np.array(PRODUCTION_MIX) /
                                           sum(PRODUCTION_MIX))]
        trace = generate_power_trace(prod, rng)
        traces[f"trace_{i:04d}"] = {
            "product_id": prod.id,
            "product_name": prod.name,
            "duration_hours": prod.test_duration_hours,
            "nameplate_kw": prod.nameplate_kw,
            "power_kw": [round(float(v), 4) for v in trace],
        }
    with open(out / "sample_traces.json", "w") as f:
        json.dump(traces, f, indent=2)
    print(f"  → Saved sample_traces.json")

    # 4. Generate GAT training snapshots
    print(f"\n[4/4] Generating {args.snapshots} power snapshots for GAT "
          f"training...")
    snapshots = generate_power_snapshots(
        topology, DEFAULT_PRODUCTS, PRODUCTION_MIX, args.snapshots, rng
    )
    with open(out / "power_snapshots.json", "w") as f:
        json.dump(snapshots, f)
    print(f"  → Saved power_snapshots.json")

    # Summary statistics
    powers = [s["total_power_kw"] for s in snapshots]
    print(f"\n{'=' * 60}")
    print(f"Benchmark generation complete.")
    print(f"  Facility: {args.slots} slots, {args.capacity_kw} kW capacity")
    print(f"  Production: {len(queue)} UUTs over {args.days} days")
    print(f"  Snapshot power stats: "
          f"mean={np.mean(powers):.1f} kW, "
          f"std={np.std(powers):.1f} kW, "
          f"max={np.max(powers):.1f} kW "
          f"({np.max(powers)/args.capacity_kw*100:.1f}% of capacity)")
    print(f"  Output: {out.resolve()}")


if __name__ == "__main__":
    main()
