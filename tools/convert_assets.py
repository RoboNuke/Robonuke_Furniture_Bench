"""Convert FurnitureBench's square-table OBJs to USDs with SDF collision.

    FURNITURE_BENCH=~/IsaacGym_Preview_4_Package/furniture-bench \\
    python tools/convert_assets.py --out ~/furniture_assets [--sdf_resolution 512] --headless

The thread on the legs and in the table top's holes only works as collision geometry under
PhysX's SDF mesh collision; a convex hull or decomposition flattens it into a peg and a
hole. FurnitureBench's URDFs ask Isaac Gym for ``<sdf resolution="512"/>`` on every part,
and that is what this writes into the USD (``PhysxSDFMeshCollisionAPI``). Run it with Isaac
Lab's python: it needs Isaac Sim's asset converter.

ResiP's fork (robust-rearrangement's ``furniture-bench``) keeps the SDF only on the top and leg 4
(legs 1-3 have it commented out, so Isaac Gym gives them a convex hull); ``--convex`` names the
parts to write with a convex hull instead. The port's FurnitureBench asset set
(``furniture.part_assets: furniturebench``, ``assets/FurnitureBench``) is ResiP's::

    python tools/convert_assets.py --furniture_bench ~/robust-rearrangement/furniture-bench \
        --out assets/FurnitureBench --convex square_table_leg1,square_table_leg2,square_table_leg3 \\
        --parts <one part> --headless      # once per part (see --parts)

Masses are not written here; the env cfg sets them (upstream's measured values).
"""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True, help="folder to write <part>.usd into")
parser.add_argument(
    "--furniture_bench",
    default=os.environ.get("FURNITURE_BENCH"),
    help="the furniture-bench repo root (or set FURNITURE_BENCH)",
)
parser.add_argument("--sdf_resolution", type=int, default=512)
parser.add_argument("--convex", default="", help="comma-separated parts to give a convex hull instead of an SDF")
parser.add_argument(
    "--parts", default="",
    help="comma-separated parts to convert (default all). Isaac Lab's MeshConverter fails on the second mesh in one "
    "process (a null prim from its stage cache), so convert one part per run",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg  # noqa: E402

PARTS = {
    "square_table_top": "square_table/square_table_top.obj",
    "square_table_leg1": "square_table/square_table_leg1.obj",
    "square_table_leg2": "square_table/square_table_leg2.obj",
    "square_table_leg3": "square_table/square_table_leg3.obj",
    "square_table_leg4": "square_table/square_table_leg4.obj",
}


def main() -> None:
    if not args.furniture_bench:
        raise SystemExit("pass --furniture_bench or set FURNITURE_BENCH")
    mesh_dir = Path(args.furniture_bench).expanduser() / "furniture_bench/assets/furniture/mesh"
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    convex = {c for c in args.convex.split(",") if c}
    unknown = convex - set(PARTS)
    if unknown:
        raise SystemExit(f"--convex names unknown parts {sorted(unknown)}; parts are {sorted(PARTS)}")
    only = {c for c in args.parts.split(",") if c} or set(PARTS)
    if only - set(PARTS):
        raise SystemExit(f"--parts names unknown parts {sorted(only - set(PARTS))}; parts are {sorted(PARTS)}")
    for name, rel in PARTS.items():
        if name not in only:
            continue
        src = mesh_dir / rel
        if not src.exists():
            raise SystemExit(f"missing mesh {src}")
        cfg = MeshConverterCfg(
            asset_path=str(src),
            usd_dir=str(out),
            usd_file_name=f"{name}.usd",
            force_usd_conversion=True,
            make_instanceable=False,
            # a free rigid body; the env cfg overrides iteration counts and mass
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mesh_collision_props=(
                sim_utils.ConvexHullPropertiesCfg()
                if name in convex
                else sim_utils.SDFMeshPropertiesCfg(sdf_resolution=args.sdf_resolution)
            ),
        )
        converter = MeshConverter(cfg)
        if name not in convex:
            _finish_sdf(converter.usd_path, args.sdf_resolution)
        print(f"{name}: {src} -> {converter.usd_path} ({'convex hull' if name in convex else f'sdf {args.sdf_resolution}'})", flush=True)


def _finish_sdf(path: str, resolution: int) -> None:
    """Isaac Lab 2.3's ``SDFMeshPropertiesCfg`` with a resolution writes only the
    ``physxSDFMeshCollision:sdfResolution`` attribute: no ``UsdPhysics.MeshCollisionAPI`` with
    ``approximation = sdf`` and no ``PhysxSDFMeshCollisionAPI``, so PhysX would not use an SDF. Apply
    both on every collision mesh, set the resolution, save, and check it took."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(path)
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)]
    if not meshes:
        raise SystemExit(f"{path}: no collision mesh to give an SDF")
    for prim in meshes:
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("sdf")  # a PhysX token, not in UsdPhysics.Tokens
        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr().Set(int(resolution))
    stage.GetRootLayer().Save()
    check = Usd.Stage.Open(path)
    for prim in meshes:
        p = check.GetPrimAtPath(prim.GetPath())
        approx = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()
        res = p.GetAttribute("physxSDFMeshCollision:sdfResolution").Get()
        if approx != "sdf" or res != resolution or not p.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
            raise SystemExit(f"{path}: {p.GetPath()} approximation {approx!r}, resolution {res!r} after the fix")


if __name__ == "__main__":
    main()
    app.close()
