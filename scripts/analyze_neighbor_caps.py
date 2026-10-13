"""Measure how often neighbor-selection policies drop approaching vehicles (design doc §8).

uv run python scripts/analyze_neighbor_caps.py
"""

from collections import defaultdict

import numpy as np

from prs.data.raw.waymo_reader import iter_scenarios
from prs.data.splits import shard_paths
from prs.data.targets import select_targets
from prs.data.types import ObjectType
from prs.data.windows import make_window
from prs.utils.config import load_config

cfg = load_config("configs/data_dev.yaml").data
window = make_window(cfg)
t0 = window.t0
HORIZON_S, MIN_CLOSING_MPS = 6.0, 1.0
# (cap, reserved vehicle slots)
POLICIES = [(16, 0), (16, 6), (32, 0), (32, 8), (48, 0), (64, 0)]

counts = []  # neighbors within radius, per target
stats = defaultdict(lambda: {"threats": 0, "dropped": 0, "targets_with_drop": 0})
num_targets = 0
num_targets_with_threat = 0

for split in ("train", "val", "test"):
    for path in shard_paths(cfg, split):
        for scenario in iter_scenarios(path):
            accepted, _ = select_targets(scenario, cfg, window)
            pos = scenario.xy[:, t0]
            vel = scenario.velocity[:, t0]
            valid = scenario.valid[:, t0]
            is_vehicle = scenario.track_types == ObjectType.VEHICLE
            for ti in accepted:
                num_targets += 1
                d = np.linalg.norm(pos - pos[ti], axis=1)
                cand = valid & (d <= cfg.neighbor_radius_m)
                cand[ti] = False
                idx = np.flatnonzero(cand)
                idx = idx[np.argsort(d[idx])]
                counts.append(len(idx))

                # "Threat": a non-SDC vehicle closing on the target fast enough to reach it
                # within the horizon (distance / closing speed <= 6 s).
                to_target = pos[ti] - pos[idx]
                unit = to_target / np.maximum(
                    np.linalg.norm(to_target, axis=1, keepdims=True), 1e-6
                )
                closing = (vel[idx] * unit).sum(axis=1)
                threat = (
                    is_vehicle[idx]
                    & (idx != scenario.sdc_index)
                    & (closing >= MIN_CLOSING_MPS)
                    & (d[idx] / np.maximum(closing, 1e-6) <= HORIZON_S)
                )
                threats = set(idx[threat].tolist())
                num_targets_with_threat += bool(threats)

                for cap, vehicle_quota in POLICIES:
                    chosen = []
                    if scenario.sdc_index in idx:
                        chosen.append(scenario.sdc_index)
                    vehicles = [i for i in idx if is_vehicle[i] and i not in chosen]
                    chosen += vehicles[:vehicle_quota]
                    for i in idx:
                        if len(chosen) >= cap:
                            break
                        if i not in chosen:
                            chosen.append(i)
                    dropped = threats - set(chosen)
                    s = stats[(cap, vehicle_quota)]
                    s["threats"] += len(threats)
                    s["dropped"] += len(dropped)
                    s["targets_with_drop"] += bool(dropped)

counts = np.array(counts)
print(
    f"targets: {num_targets}; targets with >=1 approaching vehicle (not SDC): "
    f"{num_targets_with_threat} ({num_targets_with_threat / num_targets:.1%})"
)
print(f"approaching vehicles in total: {stats[(16, 0)]['threats']}\n")
print(
    f"{'policy':<26}{'targets over cap':>18}{'approaching vehicles dropped':>32}"
    f"{'targets losing one':>22}"
)
for cap, quota in POLICIES:
    s = stats[(cap, quota)]
    name = f"N={cap}" + (f", {quota} vehicle slots" if quota else ", nearest only")
    print(
        f"{name:<26}{np.mean(counts > cap):>18.1%}"
        f"{s['dropped'] / max(s['threats'], 1):>32.1%}"
        f"{s['targets_with_drop'] / max(num_targets_with_threat, 1):>22.1%}"
    )
