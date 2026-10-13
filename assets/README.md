# assets

UWLab's FurnitureBench USDs (`UW-Lab/uwlab-assets`, `Props/FurnitureBench/`), one folder per
part with their `metadata.yaml`, kept in their layout. Which folder serves which FurnitureBench
part -- and the mesh frame the file uses -- is `furniture_lab/tasks/furniture/catalog.py`;
adding a furniture means adding its rows there.

Their `metadata.yaml` fields, as this port reads them: `assembled_offset` is the point in the
part's own frame that coincides with the mating part's when seated (both quaternions are the
identity for the square table); `bottom_offset` is the part's lowest point in its frame.
`tests/test_catalog.py` checks them against FurnitureBench's numbers.

`Obstacle/obstacle.usd` is UWLab's one-piece obstacle (front wall with 16 cm side walls
attached). The env spawns FurnitureBench's own three-box layout from primitives instead, so
this file is unused.

`SquareLeg/square_leg_phase0.usd` is derived from `square_leg.usd` by `tools/fix_leg_phase.py`:
the bolt prims are turned +26° about the leg axis. In the original, a leg screwed down until
its bolt tip bottoms on the plate sits 26° off square (the thread was authored out of phase
with the leg body); the corrected file seats at 0°, matching the real part and FurnitureBench's
assembled-orientation rule. The catalog uses the corrected file.
