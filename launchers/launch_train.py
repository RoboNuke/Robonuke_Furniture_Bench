"""Submit one SLURM job per experiment config. Thin caller; the logic lives in the package.

Run from the project root on a cluster login node (a light python is enough: omegaconf +
pyyaml — no Isaac Lab, no torch):

    python launchers/launch_train.py configs/experiments --project P --group_prefix G

No ``setup`` hook here: the submitter never loads the full config, so nothing project-side
needs registering.
"""

if __name__ == "__main__":
    from robonuke_rl_core.hpc.launch_train import main

    raise SystemExit(main())
