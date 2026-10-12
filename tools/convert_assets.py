"""Convert FurnitureBench's square-table OBJs to USDs with SDF collision.

    FURNITURE_BENCH=~/IsaacGym_Preview_4_Package/furniture-bench \\
    python tools/convert_assets.py --out ~/furniture_assets [--sdf_resolution 512] --headless

The thread on the legs and in the table top's holes only works as collision geometry under
PhysX's SDF mesh collision; a convex hull or decomposition flattens it into a peg and a
hole. FurnitureBench's URDFs ask Isaac Gym for ``<sdf resolution="512"/>`` on every part,
and that is what this writes into the USD (``PhysxSDFMeshCollisionAPI``). Run it with Isaac
Lab's python: it needs Isaac Sim's asset converter.

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
    for name, rel in PARTS.items():
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
            mesh_collision_props=sim_utils.SDFMeshPropertiesCfg(sdf_resolution=args.sdf_resolution),
        )
        converter = MeshConverter(cfg)
        print(f"{name}: {src} -> {converter.usd_path}", flush=True)


if __name__ == "__main__":
    main()
    app.close()
