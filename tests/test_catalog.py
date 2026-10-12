"""The repo's assets/ folder, as the catalog and the FurnitureBench numbers see it."""

import numpy as np
import pytest

from furniture_lab.tasks.furniture import catalog, square_table as st


def test_every_part_has_a_file_in_the_checkout():
    for part in st.PARTS:
        catalog.usd_path(part.name)  # raises if missing
    with pytest.raises(KeyError):
        catalog.entry("lamp_bulb")


def test_catalog_frames_agree_across_the_furniture():
    assert st.mesh_frame() == "uwlab"


def test_uwlab_metadata_corroborates_the_assembled_pose():
    """UWLab's metadata.yaml gives each part an ``assembled_offset``: the point that
    coincides when the leg is seated, in each part's own frame, with a quaternion. With both
    quaternions the identity, the leg's origin in the top's frame is
    ``top.assembled - leg.assembled``. It must land on one of FurnitureBench's four holes."""
    leg = catalog.metadata("square_table_leg4")
    top = catalog.metadata("square_table_top")
    assert leg is not None and top is not None
    for meta in (leg, top):
        assert meta["assembled_offset"]["quat"] == [1.0, 0.0, 0.0, 0.0]
    rel = np.array(top["assembled_offset"]["pos"]) - np.array(leg["assembled_offset"]["pos"])
    targets = st.rel_pose_targets(st.ONE_LEG_PAIRS, "uwlab")[0]
    dist = np.linalg.norm(targets - rel, axis=1)
    assert dist.min() < 0.001, f"closest hole is {dist.min() * 1000:.2f} mm away"
    # and well inside upstream's assembled tolerance along the screw axis (5 mm)
    assert abs(rel[2] - 0.046875) < st.ASSEMBLED_POS_THRESHOLD[2]


def test_uwlab_bottom_offsets_match_the_reset_heights():
    """``bottom_offset`` is the part's lowest point in its own frame; the top rests on its
    plate at reset, so its centre sits -bottom_offset.z above the surface -- upstream's
    0.015625 to within a quarter millimetre."""
    top = catalog.metadata("square_table_top")
    assert abs(-top["bottom_offset"]["pos"][2] - 0.015625) < 0.00025
