# check_usd.py — usage: ./isaaclab.sh -p check_usd.py square_leg.usd square_table_top.usd
import sys
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
for path in sys.argv[1:]:
    stage = Usd.Stage.Open(path)
    print(f"== {path}")
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        approx = "none"
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        sdf = ""
        if prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
            sdf = f" sdf_res={PhysxSchema.PhysxSDFMeshCollisionAPI(prim).GetSdfResolutionAttr().Get()}"
        npts = len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or []) if prim.IsA(UsdGeom.Mesh) else 0
        print(f"  {prim.GetPath()}  approx={approx}{sdf}  points={npts}")
