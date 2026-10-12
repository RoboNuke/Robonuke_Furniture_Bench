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
