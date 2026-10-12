"""Inspect part USDs: collision approximation, SDF settings, mesh size, and the frame.

    python tools/check_usd.py ~/furniture_assets/square_table_top.usd ~/furniture_assets/square_table_leg1.usd

Needs ``pxr`` (Isaac Lab's python has it; so does ``pip install usd-core``). Two things
matter for this port:

1. **Collision** must be an SDF mesh (``approx=sdf`` with a resolution), or the thread is
   gone and the leg cannot screw in.
2. **Frame**: the assembly check uses upstream's mesh frames. The bounds printed here should
   match the OBJs (metres, the mesh's own axes):

       square_table_top   x +-0.08125   y +-0.015625   z +-0.08125
       square_table_leg   x +-0.015     y -0.05625 .. +0.03125   z +-0.015

   A USD that is re-centred or re-oriented (UWLab's, possibly) needs the offset folded into
   ``square_table.ASSEMBLED_REL_POSES`` and the reset poses -- do not use it blind.
"""

import sys

from pxr import Usd, UsdGeom, UsdPhysics

try:  # PhysxSchema ships with Isaac Sim's pxr, not with plain usd-core
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None


def inspect(path: str) -> None:
    stage = Usd.Stage.Open(path)
    print(f"== {path}")
    print(f"   default prim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}")
    found = False
    for prim in stage.Traverse():
        has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
        is_mesh = prim.IsA(UsdGeom.Mesh)
        if not (has_col or is_mesh):
            continue
        found = True
        approx = "-"
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        sdf = ""
        if PhysxSchema is not None and prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
            api = PhysxSchema.PhysxSDFMeshCollisionAPI(prim)
            sdf = f" sdf_res={api.GetSdfResolutionAttr().Get()}"
        elif "PhysxSDFMeshCollisionAPI" in prim.GetAppliedSchemas():
            res = prim.GetAttribute("physxSDFMeshCollision:sdfResolution").Get()
            sdf = f" sdf_res={res}"
        npts = 0
        bounds = ""
        if is_mesh:
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            npts = len(pts)
            if npts:
                xs, ys, zs = (sorted(p[i] for p in pts) for i in range(3))
                bounds = (
                    f" bounds x[{xs[0]:+.4f},{xs[-1]:+.4f}] y[{ys[0]:+.4f},{ys[-1]:+.4f}]"
                    f" z[{zs[0]:+.4f},{zs[-1]:+.4f}]"
                )
        xform = ""
        if prim.IsA(UsdGeom.Xformable):
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
            if ops:
                xform = " xform=" + ",".join(f"{op.GetOpName()}={op.Get()}" for op in ops)
        print(
            f"   {prim.GetPath()}  collision={'yes' if has_col else 'no'} approx={approx}{sdf}"
            f" points={npts}{bounds}{xform}"
        )
    if not found:
        print("   (no mesh or collision prims found)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        inspect(p)
