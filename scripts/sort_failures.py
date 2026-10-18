"""Sort a folder of collected one_leg attempts by the stage each one broke at.

    python scripts/sort_failures.py ~/tmp/phys/base_fail            # every *.pkl under it, fails/ included
    python scripts/sort_failures.py ~/replay_traces/resip            # replay_abs --save traces (.npz)
    python scripts/sort_failures.py DIR --corner 0.01 --top_move 0.01 --top_yaw 8 --hole_xz 0.013 --tilt 16

No sim (NumPy only). Collect with ``--save_failure`` so the failures are kept (and ``--record``
for their videos, ``fails/failN.mp4`` beside ``fails/failN.pkl``). One line per attempt, then
the counts per stage, then the medians of each stage's metrics over the successes, to check
the thresholds against attempts that worked. The stages and their tests are in
``furniture_lab/failures.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from furniture_lab.failures import STAGES, Thresholds, classify, load_attempt, load_trace  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("dir", help="a collector --out_dir (searched recursively for *.pkl)")
d = Thresholds()
parser.add_argument("--corner", type=float, default=d.corner, help="m: top within this of the corner goal at its release")
parser.add_argument("--lift", type=float, default=d.lift, help="m: the leg rose this far after its grasp")
parser.add_argument("--drop", type=float, default=d.drop, help="m: leg-EE distance growth while carried that counts as a drop")
parser.add_argument("--hole_xz", type=float, default=d.hole_xz, help="m: leg origin within this of the hole axis")
parser.add_argument("--tilt", type=float, default=d.tilt_deg, help="deg: leg tilt from the seat axis")
parser.add_argument("--top_move", type=float, default=d.top_move, help="m: top translation during screwing")
parser.add_argument("--top_yaw", type=float, default=d.top_yaw_deg, help="deg: top rotation during screwing")
args = parser.parse_args()

METRICS = ("corner_mm", "lift_mm", "insert_xz_mm", "insert_tilt_deg", "insert_height_mm", "top_move_mm", "top_yaw_deg", "final_height_mm")
#: with the collector's ee_wrench recorded: the push-down while lowering and screwing
FORCES = (
    "lower_fz_peak_N", "lower_fz_end_N", "screw_fz_peak_N", "screw_fz_median_N", "screw_fxy_peak_N", "screw_tz_peak_Nm",
    "screw_fz_p95_held_N", "screw_fxy_p95_held_N", "screw_tz_p95_held_Nm",
    "screw_fz_p95_open_N", "screw_fxy_p95_open_N", "screw_tz_p95_open_Nm",
    "screw_fz_last10_N", "screw_fxy_last10_N",
)


def main() -> int:
    root = Path(args.dir).expanduser()
    # collector pkls, or replay_abs --save traces (.npz): the sim's replay of each demo
    paths = sorted(root.rglob("*.pkl")) + sorted(root.rglob("*.npz"))
    if not paths:
        raise SystemExit(f"no .pkl or .npz under {args.dir}")
    th = Thresholds(
        corner=args.corner, lift=args.lift, drop=args.drop, hole_xz=args.hole_xz,
        tilt_deg=args.tilt, top_move=args.top_move, top_yaw_deg=args.top_yaw,
    )
    rows = []
    for path in paths:
        try:
            r = classify((load_trace if path.suffix == ".npz" else load_attempt)(path), th)
        except Exception as exc:  # noqa: BLE001 -- report and keep going: one bad file is not the batch
            print(f"{path}: could not read ({type(exc).__name__}: {exc})")
            continue
        rows.append((path, r))
        shown = ", ".join(f"{k} {r[k]:.1f}" for k in METRICS + FORCES[:6] if k in r)
        if "fxy_peak_while_held" in r:
            shown += f", fxy peak {'held' if r['fxy_peak_while_held'] else 'fingers OPEN'}"
        at = f" at step {r['step']}" if r["step"] is not None else ""
        print(f"{path.relative_to(Path(args.dir).expanduser())}: {r['stage']}{at} ({r['steps']} steps; {shown})")

    counts = Counter(r["stage"] for _, r in rows)
    fails = sum(n for s, n in counts.items() if s != "success")
    print(f"\n===== {len(rows)} attempts: {counts.get('success', 0)} success, {fails} failed")
    for stage in STAGES:
        if counts.get(stage):
            share = f" ({100 * counts[stage] / fails:.0f}% of failures)" if stage != "success" and fails else ""
            print(f"  {counts[stage]:3d}  {stage}{share}")
    ok = [r for _, r in rows if r["stage"] == "success"]
    tight = Counter(r["passed_to"] for r in ok if r["passed_to"] != "screwing: stalled in the hole")
    if tight:
        print("\nWARNING: successes that a test would have failed (loosen that threshold):")
        for stage, n in tight.items():
            print(f"  {n:3d}  {stage}")
    if ok:
        print("\nsuccesses, median (check the thresholds against these):")
        for k in METRICS:
            vals = [r[k] for r in ok if k in r]
            if vals:
                print(f"  {k:18s} {np.median(vals):7.1f}  (max {np.max(vals):.1f})")
    have = [k for k in FORCES if any(k in r for _, r in rows)]
    if have:
        stages = [st for st in STAGES if counts.get(st)]
        print("\nforces, median per stage (+z: the parts pushing back up on the hand):")
        print("  " + " " * 18 + "".join(f"{st.split(': ')[-1][:16]:>18s}" for st in stages))
        held = [r["fxy_peak_while_held"] for _, r in rows if "fxy_peak_while_held" in r]
        if held:
            print(f"  (the screwing fxy peak came with the fingers open in {held.count(False)}/{len(held)} attempts: the "
                  "arm's own effort, not contact -- read the _held rows)")
        for k in have:
            cells = []
            for st in stages:
                vals = [r[k] for _, r in rows if r["stage"] == st and k in r]
                cells.append(f"{np.median(vals):18.1f}" if vals else f"{'-':>18s}")
            print(f"  {k:18s}" + "".join(cells))
    else:
        print("\n(no ee_wrench in these files: collect / replay again with the current scripts for the forces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
