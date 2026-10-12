"""Submit one SLURM eval job per (training run x eval config). Thin caller; the logic
lives in the package.

    python launchers/launch_eval.py --eval_config configs/eval/quick.yaml \\
        --project template_demo --group <group> ...

Runs are selected by explicit --run paths or a wandb query (--project with --group /
--wandb_tag), so this one needs wandb importable on the login node.
"""

if __name__ == "__main__":
    from robonuke_rl_core.hpc.launch_eval import main

    raise SystemExit(main())
