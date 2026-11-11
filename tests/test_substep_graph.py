"""``furniture_lab.substep_graph``'s PhysX boundary, on fake views (no Isaac Sim, no CUDA): a record pass
learns the call sequence and routes every read through a static buffer, writes are replayed with the
recorded arguments, a changed sequence / a read after a write / an unknown call fail loud, and the lazy
buffer invalidation leaves the finite-difference state alone."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from furniture_lab.substep_graph import PhysxBoundary, _classify


class FakeView:
    """A PhysX-view stand-in: reads allocate a new tensor per call (as the tensor API does), writes are logged."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.pos = torch.arange(n, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
        self.written = []
        self.kinematic_updates = 0

    def get_positions(self):
        return self.pos.clone()

    def get_velocities(self):
        return torch.ones(self.n, 3) * 0.5

    def set_targets(self, data, indices):
        self.written.append(("set_targets", data.clone(), indices))

    def apply_forces(self, force_data, indices, is_global):
        self.written.append(("apply_forces", force_data.clone(), is_global))

    def update_articulations_kinematic(self):
        self.kinematic_updates += 1

    def not_a_tensor_api_call(self):
        return 1


def region(view: FakeView, target: torch.Tensor, force: torch.Tensor, indices: torch.Tensor) -> None:
    """The kind of substep the boundary splits: reads, torch work into fixed buffers, writes."""
    view.update_articulations_kinematic()
    pos = view.get_positions()
    vel = view.get_velocities()
    target.copy_(pos + 0.1 * vel)
    force.copy_(-pos)
    view.set_targets(target, indices)
    view.apply_forces(force, indices, is_global=True)


def test_a_record_pass_learns_the_sequence_and_reads_through_static_buffers():
    view = FakeView(4)
    target, force, indices = torch.zeros(4, 3), torch.zeros(4, 3), torch.arange(4)
    b = PhysxBoundary({"robot": view})
    with b("record"):
        region(view, target, force, indices)
    assert [c.kind for c in b.calls] == ["void", "read", "read", "write", "write"]
    assert [c.name for c in b.calls] == ["update_articulations_kinematic", "get_positions", "get_velocities", "set_targets", "apply_forces"]
    # the record pass ran for real: the writes reached the view with the computed values
    assert len(view.written) == 2 and torch.allclose(view.written[0][1], view.pos + 0.05)
    # the reads' static buffers hold what the view returned
    assert torch.equal(b.calls[1].static, view.pos) and torch.all(b.calls[2].static == 0.5)
    assert "2 reads, 1 kinematic updates, 2 writes" in b.summary()
    # the view is itself again afterwards
    assert "get_positions" not in vars(view)


def test_a_second_record_pass_refills_the_same_buffers_and_the_replay_halves_work():
    view = FakeView(3)
    target, force, indices = torch.zeros(3, 3), torch.zeros(3, 3), torch.arange(3)
    b = PhysxBoundary({"robot": view})
    with b("record"):
        region(view, target, force, indices)
    static = b.calls[1].static
    view.pos += 1.0
    with b("record"):
        region(view, target, force, indices)
    assert b.calls[1].static is static and torch.equal(static, view.pos)
    # the replay's eager halves: fetch refreshes the buffers (and runs the kinematic update), write sends the
    # recorded arguments (the fixed buffers, whatever they hold now)
    view.pos += 1.0
    view.written.clear()
    b.fetch()
    assert torch.equal(static, view.pos) and view.kinematic_updates == 3
    target.fill_(7.0)
    b.write()
    assert [w[0] for w in view.written] == ["set_targets", "apply_forces"] and torch.all(view.written[0][1] == 7.0)


def test_a_capture_pass_touches_nothing_in_the_view():
    view = FakeView(2)
    target, force, indices = torch.zeros(2, 3), torch.zeros(2, 3), torch.arange(2)
    b = PhysxBoundary({"robot": view})
    with b("record"):
        region(view, target, force, indices)
    view.written.clear()
    updates = view.kinematic_updates
    b.calls[1].static.fill_(3.0)
    with b("capture"):
        region(view, target, force, indices)
    assert view.written == [] and view.kinematic_updates == updates
    assert torch.all(target == 3.0 + 0.05)  # the region read the static buffer, not the view


def test_the_sequence_must_not_change_between_passes():
    view = FakeView(2)
    target, force, indices = torch.zeros(2, 3), torch.zeros(2, 3), torch.arange(2)
    b = PhysxBoundary({"robot": view})
    with b("record"):
        region(view, target, force, indices)
    with pytest.raises(RuntimeError, match="previous pass made"):
        with b("record"):
            view.update_articulations_kinematic()
            view.get_velocities()  # positions first before
    with pytest.raises(RuntimeError, match="structure changed"):
        with b("record"):
            view.update_articulations_kinematic()
            view.get_positions()
            view.get_velocities()
            view.set_targets(target, indices)  # one write short


def test_a_read_after_a_write_and_an_unknown_call_are_refused():
    view = FakeView(2)
    b = PhysxBoundary({"robot": view})
    with pytest.raises(RuntimeError, match="reads after the first write"):
        with b("record"):
            view.set_targets(torch.zeros(2, 3), torch.arange(2))
            view.get_positions()
    b = PhysxBoundary({"robot": view})
    with pytest.raises(RuntimeError, match="neither a read"):
        with b("record"):
            view.not_a_tensor_api_call()
    assert _classify("x", "get_anything") == "read" and _classify("x", "apply_forces_and_torques_at_position") == "write"
    with pytest.raises(RuntimeError, match="a capture pass needs a record pass"):
        PhysxBoundary({"robot": view})("capture")
    with pytest.raises(RuntimeError, match="no PhysX call"):
        with PhysxBoundary({"robot": view})("record"):
            pass


def test_invalidation_marks_every_lazy_buffer_but_the_acceleration():
    from furniture_lab.substep_graph import invalidate_lazy_buffers

    class Buf:  # TimestampedBuffer's two fields
        def __init__(self, t):
            self.data, self.timestamp = None, t

    import sys

    stub = SimpleNamespace(TimestampedBuffer=Buf)
    saved = {k: sys.modules.get(k) for k in ("isaaclab", "isaaclab.utils", "isaaclab.utils.buffers")}
    sys.modules["isaaclab"] = SimpleNamespace(utils=SimpleNamespace(buffers=stub))
    sys.modules["isaaclab.utils"] = sys.modules["isaaclab"].utils
    sys.modules["isaaclab.utils.buffers"] = stub
    try:
        data = SimpleNamespace(_joint_pos=Buf(3.0), _body_link_pose_w=Buf(3.0), _joint_acc=Buf(2.5), other=1)
        invalidate_lazy_buffers([SimpleNamespace(data=data)])
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert data._joint_pos.timestamp == -math.inf and data._body_link_pose_w.timestamp == -math.inf
    assert data._joint_acc.timestamp == 2.5  # the finite difference's previous time, kept
