"""Submit one SLURM job per (config x swept value). Thin caller; the logic lives in the
package.

    python launchers/launch_sweep.py configs/experiments/hybrid_fragile.yaml \\
        --project P --group_prefix G \\
        --sweep_param sac.actor_lr --label lr --value 1.0e-4 --value 3.0e-4
"""

if __name__ == "__main__":
    from robonuke_rl_core.hpc.launch_sweep import main

    raise SystemExit(main())
