"""One physics substep's torch work as a CUDA graph (``furniture.cuda_graph``).

Before each of a control step's physics substeps (12 at ResiP's 120 Hz / 10 Hz) Isaac Lab runs
``action_manager.apply_action()`` and ``scene.write_data_to_sim()``: the action terms' IK and gripper
command, the parts' reset kick and the wrench estimator (``AssemblyEnv`` hangs them on ``apply_action``),
then the actuator model and the PhysX writes. That is a few hundred small kernels per substep, each
launched from Python, with the GPU idle in between -- the profile of the training step: 56 of 183 ms,
the GPU busy 27 %. Since the sync-free rewrite that work makes no host synchronization, so one CUDA
graph can hold it and replay it as a single launch.

What a graph cannot hold are the PhysX tensor API calls inside the region: a read such as
``get_dof_positions`` or ``get_jacobians`` allocates a new tensor per call and fills it in PhysX, a
write such as ``set_dof_position_targets`` runs in PhysX. So the substep is split at that boundary:

1. **reads**: every PhysX read the region makes is done eagerly, in order, into a static buffer (one
   copy each; ``update_articulations_kinematic`` runs where it did);
2. **the graph**: the torch work, reading those buffers and writing Isaac Lab's target buffers in place;
3. **writes**: the region's PhysX writes, in order, with the arguments the capture recorded (Isaac
   Lab's own static buffers, or tensors the graph owns and rewrites on every replay).

The boundary is learned, not hand-written: :class:`PhysxBoundary` intercepts the views' methods during
the warm-up substeps (``record``: eager, the call sequence learned, each read's result copied into its
static buffer) and the capture (nothing reaches PhysX: reads return their static buffers, writes are
recorded). Isaac Lab fetches its asset data lazily into timestamped buffers, so before each of those
passes they are invalidated (:func:`invalidate_lazy_buffers`): every read then goes through the views
and so through the boundary; the eager code outside the region refetches at the next timestamp as it
always did. The sequence must be the same in every pass, every read must come before the first write
(a read after a write could depend on it through PhysX), and a PhysX call of a kind the boundary does
not know fails loud. The region's torch code must keep its state in place -- a tensor reassigned per
substep would leave the graph reading a stale address: ``wrench.py``, ``mdp/events.py``'s kick,
``mdp/gripper.py`` and ``mdp/tcp_ik.py`` do (see their docstrings).

:class:`SubstepGraph` runs the region: eager for ``warmup`` substeps, then captured once, then replayed
(reads, graph, writes). A change in the region's structure after the capture -- a branch the Python
code took differently -- cannot be seen from a replay, so the caller gives a ``signature`` of the
Python-level choices the region makes, checked on every replay; a change raises.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = ["PhysxBoundary", "PhysxCall", "SubstepGraph", "invalidate_lazy_buffers"]

#: view methods that neither return nor take a tensor the graph sees, but must run in sequence before the reads
#: that follow them
VOID_CALLS = ("update_articulations_kinematic",)
READ_PREFIX = "get_"
WRITE_PREFIXES = ("set_", "apply_")


@dataclass
class PhysxCall:
    """One PhysX tensor API call of the region, as the boundary learned it."""

    kind: str  #: read | void | write
    owner: str  #: the view's name in the boundary (for messages)
    name: str  #: the method
    fn: Callable[..., Any]  #: the real bound method
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    #: reads: the buffer the graph reads, refilled from the real call before every replay
    static: Optional[torch.Tensor] = None

    def key(self) -> tuple:
        shape = None if self.static is None else (tuple(self.static.shape), self.static.dtype)
        return (self.kind, self.owner, self.name, shape)


def _classify(owner: str, name: str) -> str:
    if name.startswith(READ_PREFIX):
        return "read"
    if name in VOID_CALLS:
        return "void"
    if name.startswith(WRITE_PREFIXES):
        return "write"
    raise RuntimeError(
        f"PhysxBoundary: {owner}.{name}() inside the graphed substep is neither a read (get_*), a write (set_* / apply_*) "
        f"nor one of {VOID_CALLS}; the substep graph does not know how to replay it"
    )


class PhysxBoundary:
    """Intercepts the PhysX tensor API calls a region makes. ``views`` maps a name to a view object (an asset's
    ``root_physx_view``, the physics ``SimulationView``); every public method of theirs is intercepted while a pass
    is open (``with boundary("record"): region()``, ``with boundary("capture"): region()``). :meth:`fetch` and
    :meth:`write` are the replay's two eager halves."""

    def __init__(self, views: Dict[str, object]) -> None:
        if not views:
            raise ValueError("PhysxBoundary: no views to intercept")
        self.views = dict(views)
        #: the learned sequence: after the first pass, the latest pass's calls (same keys, that pass's arguments)
        self.calls: Optional[List[PhysxCall]] = None
        self._mode: Optional[str] = None
        self._pass: Optional[List[PhysxCall]] = None
        self._patched: List[Tuple[object, str]] = []

    # ------------------------------------------------------------------ the passes
    def __call__(self, mode: str) -> "PhysxBoundary":
        if mode not in ("record", "capture"):
            raise ValueError(f"PhysxBoundary: mode must be record or capture, got {mode!r}")
        if mode == "capture" and self.calls is None:
            raise RuntimeError("PhysxBoundary: a capture pass needs a record pass first")
        self._mode = mode
        return self

    def __enter__(self) -> "PhysxBoundary":
        if self._mode is None:
            raise RuntimeError("PhysxBoundary: use `with boundary('record' | 'capture'):`")
        self._pass = []
        for owner, view in self.views.items():
            for name in dir(view):
                if name.startswith("_") or not inspect.isfunction(getattr(type(view), name, None)):
                    continue
                real = getattr(view, name)
                setattr(view, name, self._wrapper(owner, name, real))
                self._patched.append((view, name))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for view, name in self._patched:
            delattr(view, name)  # the class method shows again
        self._patched.clear()
        done, self._pass, self._mode = self._pass, None, None
        if exc_type is not None:
            return
        if self.calls is not None and len(done) != len(self.calls):
            raise RuntimeError(
                f"PhysxBoundary: the pass made {len(done)} PhysX calls, the previous one {len(self.calls)}: the region's "
                f"structure changed ({[c.name for c in done]} vs {[c.name for c in self.calls]})"
            )
        first_write = next((i for i, c in enumerate(done) if c.kind == "write"), len(done))
        late = [c.name for c in done[first_write:] if c.kind != "write"]
        if late:
            raise RuntimeError(f"PhysxBoundary: PhysX reads after the first write ({late}): they may depend on it, the split cannot hold")
        if not done:
            raise RuntimeError("PhysxBoundary: the region made no PhysX call; nothing to split")
        self.calls = done

    def _wrapper(self, owner: str, name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args, **kwargs):
            return self._on_call(owner, name, real, args, kwargs)

        return wrapped

    def _on_call(self, owner: str, name: str, real: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        kind = _classify(owner, name)
        index = len(self._pass)
        expected = None
        if self.calls is not None:
            if index >= len(self.calls):
                raise RuntimeError(f"PhysxBoundary: call {index} {owner}.{name}() beyond the {len(self.calls)} calls of the previous pass")
            expected = self.calls[index]
        call = PhysxCall(kind, owner, name, real, args, dict(kwargs))
        result = None
        if kind == "read":
            if self._mode == "record":
                out = real(*args, **kwargs)
                if not torch.is_tensor(out):
                    raise RuntimeError(f"PhysxBoundary: {owner}.{name}() returned {type(out).__name__}, not a tensor")
                if expected is None:
                    call.static = out.clone()
                else:
                    call.static = expected.static
                    if expected.static.shape != out.shape or expected.static.dtype != out.dtype:
                        raise RuntimeError(f"PhysxBoundary: {owner}.{name}() returned {tuple(out.shape)} {out.dtype}, before {expected.key()[3]}")
                    call.static.copy_(out)
            else:
                call.static = expected.static  # filled by fetch() just before the capture
            result = call.static
        elif self._mode == "record":
            result = real(*args, **kwargs)  # void and write: eager in a record pass, skipped in the capture
        if expected is not None and call.key() != expected.key():
            raise RuntimeError(f"PhysxBoundary: call {index} is {call.key()}, the previous pass made {expected.key()}")
        self._pass.append(call)
        return result

    # ------------------------------------------------------------------ the replay's eager halves
    def fetch(self) -> None:
        """This substep's PhysX reads into the static buffers, the kinematic updates where they were."""
        for call in self.calls:
            if call.kind == "void":
                call.fn(*call.args, **call.kwargs)
            elif call.kind == "read":
                call.static.copy_(call.fn(*call.args, **call.kwargs))

    def write(self) -> None:
        """The region's PhysX writes, in order, with the recorded arguments."""
        for call in self.calls:
            if call.kind == "write":
                call.fn(*call.args, **call.kwargs)

    def summary(self) -> str:
        if self.calls is None:
            return "no pass yet"
        n = {k: sum(1 for c in self.calls if c.kind == k) for k in ("read", "void", "write")}
        return f"{n['read']} reads, {n['void']} kinematic updates, {n['write']} writes: " + ", ".join(f"{c.owner}.{c.name}" for c in self.calls)


def invalidate_lazy_buffers(assets: Sequence[object]) -> None:
    """Mark every timestamped buffer of the assets' data stale, so the next read of each fetches from PhysX
    (through the boundary, in a pass). ``_joint_acc`` is left alone: its timestamp is the finite difference's
    previous time, not a cache mark."""
    from isaaclab.utils.buffers import TimestampedBuffer

    for asset in assets:
        for name, buf in vars(asset.data).items():
            if isinstance(buf, TimestampedBuffer) and name != "_joint_acc":
                buf.timestamp = -math.inf


class SubstepGraph:
    """Runs ``region`` (the substep: Isaac Lab's ``apply_action`` then ``write_data_to_sim``) eagerly for ``warmup``
    substeps, captures it once, then replays it. ``assets`` are the scene's assets (their PhysX views and lazy data),
    ``sim_view`` the physics ``SimulationView``; ``signature`` returns the Python-level choices the region makes, so a
    change after the capture is caught."""

    def __init__(
        self,
        region: Callable[[], None],
        assets: Dict[str, object],
        sim_view: object,
        signature: Callable[[], tuple],
        *,
        warmup: int = 3,
    ) -> None:
        if warmup < 1:
            raise ValueError(f"SubstepGraph: warmup must be >= 1 (the record pass), got {warmup}")
        views: Dict[str, object] = {name: asset.root_physx_view for name, asset in assets.items()}
        views["physics_sim_view"] = sim_view
        for name, view in views.items():
            if view is None:
                raise RuntimeError(f"SubstepGraph: {name} has no PhysX view yet (the scene is not initialized)")
        self.boundary = PhysxBoundary(views)
        self.region = region
        self.assets = list(assets.values())
        self.signature = signature
        self.warmup = int(warmup)
        self.runs = 0
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured_signature: Optional[tuple] = None

    @property
    def captured(self) -> bool:
        return self.graph is not None

    def run(self) -> None:
        if self.graph is not None:
            now = self.signature()
            if now != self._captured_signature:
                raise RuntimeError(
                    f"SubstepGraph: the substep's structure changed after the capture ({self._captured_signature} -> {now}); "
                    "the graph would replay stale choices"
                )
            self.boundary.fetch()
            self.graph.replay()
            self.boundary.write()
            return
        self.runs += 1
        invalidate_lazy_buffers(self.assets)
        if self.runs <= self.warmup:
            with self.boundary("record"):
                self.region()
            return
        # the capture: this substep's reads first (eager), then the torch work recorded, then the replay does the
        # substep for real
        self.boundary.fetch()
        invalidate_lazy_buffers(self.assets)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            with self.boundary("capture"):
                self.region()
        self.graph = graph
        self._captured_signature = self.signature()
        graph.replay()
        self.boundary.write()
