"""The tag->world transform must reproduce FurnitureBench's own matrices and scene layout."""

import numpy as np
import pytest

from furniture_lab.tasks.furniture import frames, square_table as st


def _get_mat(pos, angles):
    T = np.eye(4)
    T[:3, :3] = frames.rot_mat(angles)
    T[:3, 3] = pos
    return T


def test_tag_frame_matches_upstream_matrix():
    # config["robot"]["tag_base_from_robot_base"] = get_mat((0.3015, 0, -0.015), (pi, 0, pi/2))
    ref = _get_mat((0.23 + 0.0715, 0, -0.015), (np.pi, 0, np.pi / 2))
    assert np.allclose(frames.TAG_ROT_W, ref[:3, :3])
    assert np.allclose(frames.TAG_POS_W, ref[:3, 3])
    # tag z points down into the table, tag x is robot y, tag y is robot x
    assert np.allclose(frames.TAG_ROT_W @ [0, 0, 1], [0, 0, -1])
    assert np.allclose(frames.TAG_ROT_W @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(frames.TAG_ROT_W @ [0, 1, 0], [1, 0, 0])


def test_parts_rest_on_the_table_surface():
    """Each part's reset pose puts its centre half its thickness above the surface."""
    top = st.PARTS[0]
    pos, quat = st.reset_pose_world(top)
    # top is 31.25 mm thick (mesh y) and lies flat: centre 15.6 mm above the surface
    assert abs(pos[2] - (frames.TABLE_SURFACE_Z + 0.015625)) < 1e-6
    assert abs(pos[0] - (frames.TAG_POS_W[0] + 0.24)) < 1e-6 and abs(pos[1]) < 1e-6
    # its thickness axis (mesh y) ends up vertical
    R = frames.TAG_ROT_W @ top.reset_rot_tag
    assert np.allclose(np.abs(R @ [0, 1, 0]), [0, 0, 1], atol=1e-9)
    # upside down on the table: the legs screw in from above, so mesh +y points up
    assert np.allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-9)

    for leg in st.PARTS[1:]:
        pos, _ = st.reset_pose_world(leg)
        assert abs(pos[2] - (frames.TABLE_SURFACE_Z + 0.015)) < 1e-6
        # legs lie flat along the robot's x axis (mesh y -> world x), on a line in front of
        # the robot, spread across y
        R = frames.TAG_ROT_W @ leg.reset_rot_tag
        assert np.allclose(np.abs(R @ [0, 1, 0]), [1, 0, 0], atol=1e-9)
        assert abs(pos[0] - (frames.TAG_POS_W[0] + 0.07)) < 1e-6
    ys = sorted(st.reset_pose_world(leg)[0][1] for leg in st.PARTS[1:])
    assert np.allclose(ys, [-0.20, -0.12, 0.12, 0.20])


def test_parts_sit_behind_the_front_obstacle():
    """The side obstacles only bracket the far 17 cm before the front wall (x 0.52..0.69);
    the legs rest at x = 0.37, outside that span, so only the front wall bounds them."""
    front_x = frames.OBSTACLE_FRONT_POS[0] - frames.OBSTACLE_FRONT_SIZE[0] / 2
    side_x0 = frames.OBSTACLE_SIDE_POS["obstacle_left"][0] - frames.OBSTACLE_SIDE_SIZE[0] / 2
    for part in st.PARTS:
        pos, _ = st.reset_pose_world(part)
        assert pos[0] + part.footprint_radius < front_x
        if pos[0] + part.footprint_radius > side_x0:  # reaches into the bracketed span
            assert abs(pos[1]) + part.footprint_radius < 0.175 - frames.OBSTACLE_SIDE_SIZE[1] / 2


def test_quaternions_are_unit_wxyz():
    for part in st.PARTS:
        _, q = st.reset_pose_world(part)
        assert abs(np.linalg.norm(q) - 1) < 1e-9 and q[0] >= 0


def test_uwlab_frame_is_a_rotation_that_puts_the_obj_y_axis_on_z():
    R = st.MESH_FRAMES["uwlab"]
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    # OBJ y (leg axis / top thickness) -> USD +z: the bolt (OBJ -y) sits at USD -z, as the
    # file's bounds show (bolt z in [-0.0567, -0.0291])
    assert np.allclose(R @ [0, 1, 0], [0, 0, 1])


def test_uwlab_assembled_targets_are_the_four_holes_in_xy():
    targets = st.rel_pose_targets(st.ONE_LEG_PAIRS, "uwlab")[0]
    assert np.allclose(np.abs(targets[:, :2]), 0.05625)
    assert np.allclose(targets[:, 2], 0.046875)
    assert len({tuple(np.sign(t[:2])) for t in targets}) == 4
    # seated: the bolt (USD z in [-0.0567, -0.0291] of the leg) spans the hole thread
    # (USD z in [-0.0116, +0.0156] of the top)
    bolt = 0.046875 + np.array([-0.0567, -0.0291])
    assert bolt[0] < -0.0091 and bolt[1] > 0.0156


def test_uwlab_reset_poses_keep_parts_flat_and_bolts_toward_the_robot():
    # in the OBJ frame the legs lie along world x with the bolt (OBJ -y) toward the robot
    for part in st.PARTS[1:]:
        pos_obj, q_obj = st.reset_pose_world(part, "obj")
        pos_uw, q_uw = st.reset_pose_world(part, "uwlab")
        assert pos_obj == pytest.approx(pos_uw)
        R_obj = _quat_to_mat(q_obj)
        R_uw = _quat_to_mat(q_uw)
        assert np.allclose(R_obj @ [0, -1, 0], [-1, 0, 0], atol=1e-9)
        assert np.allclose(R_uw @ [0, 0, -1], [-1, 0, 0], atol=1e-9)
    # the top's plate (USD -z) faces down in both
    top = st.PARTS[0]
    assert np.allclose(_quat_to_mat(st.reset_pose_world(top, "uwlab")[1]) @ [0, 0, 1], [0, 0, 1], atol=1e-9)


def test_mesh_frame_env_var(monkeypatch):
    monkeypatch.delenv(st.MESH_FRAME_VAR, raising=False)
    assert st.mesh_frame() == "obj"
    monkeypatch.setenv(st.MESH_FRAME_VAR, "uwlab")
    assert st.mesh_frame() == "uwlab"
    monkeypatch.setenv(st.MESH_FRAME_VAR, "bogus")
    with pytest.raises(ValueError):
        st.mesh_frame()


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
