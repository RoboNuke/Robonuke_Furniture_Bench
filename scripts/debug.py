"""Project entry point: delegates to :func:`robonuke_rl_core.debug.main`.

The ``setup`` hook runs after AppLauncher (so Isaac imports work) and before the config
loads -- every project registration goes in there, never at module top."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make furniture_lab importable


def setup() -> None:
    import furniture_lab.tasks  # noqa: F401  -- gym.register (Isaac Lab is imported lazily)


if __name__ == "__main__":
    from robonuke_rl_core.debug import main

    raise SystemExit(main(setup=setup))
