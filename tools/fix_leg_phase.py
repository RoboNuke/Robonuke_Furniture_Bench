"""Turn the bolt of UWLab's square leg about the leg axis, so the thread seats with the leg's
faces square to the table top.

    python tools/fix_leg_phase.py [--phase_deg 26] [--src assets/SquareLeg/square_leg.usd]
                                  [--out assets/SquareLeg/square_leg_phase0.usd]

Why: in ``square_leg.usd`` the ``bolt`` prims (collision and visual) are authored ~26 deg out
of phase with the leg body: screwed down until the bolt tip bottoms on the table top's
plate, the thread's crest sits at 26 deg of twist, so a correctly assembled leg ends up 26
deg off square (``scripts/smoke.py --teleport`` measures it: every start from 0..25 deg is
turned to 26 +- 1.5). FurnitureBench's real part, and its assembled-orientation rule, seat
at 0 deg. Rotating the bolt prims by +phase about z moves the seat to 0 deg; the leg body,
masses and everything else are untouched. Writes a new file, never edits the source.

Needs ``pxr`` (``pip install usd-core``). Re-run ``smoke.py --teleport --yaws 0,10,20`` after:
the settled twist should read ~0.
"""

import argparse
import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

BOLT_PRIMS = ("/square_leg/collisions/bolt", "/square_leg/visuals/bolt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="assets/SquareLeg/square_leg.usd")
    parser.add_argument("--out", default="assets/SquareLeg/square_leg_phase0.usd")
    parser.add_argument("--phase_deg", type=float, default=26.0, help="yaw to apply to the bolt prims, degrees about +z")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.src)
    half = math.radians(args.phase_deg) / 2
    quat = Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half)))  # (w, xyz): yaw about z
    for path in BOLT_PRIMS:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise SystemExit(f"{path} not in {args.src}")
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        if "xformOp:orient" in ops:
            op = ops["xformOp:orient"]
            current = op.Get()
            if current != Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)):
                print(f"[fix] note: {path} already had orient {current}; composing on top of it")
            op.Set(quat * Gf.Quatf(current))
        else:
            xf.AddOrientOp().Set(quat)
        print(f"[fix] {path}: orient <- yaw {args.phase_deg:+.1f} deg")
    out = Path(args.out)
    stage.GetRootLayer().Export(str(out))
    print(f"[fix] wrote {out} (source {args.src} untouched)")


if __name__ == "__main__":
    main()
