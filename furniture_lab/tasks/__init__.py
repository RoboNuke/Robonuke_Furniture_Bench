"""The project's environments: import this module (inside an entry script's ``setup``) to
register them. The entry points are strings, so Isaac Lab is imported only when a task is
actually loaded -- which is after ``AppLauncher`` has started the app.

The task names carry no ``Forge`` because none of the package's Forge-family wrappers
(controller, fragile, efficient reset, contact) apply to a manager-based env; the package
checks the name and skips or refuses them.
"""

import gymnasium as gym

gym.register(
    id="Isaac-Furniture-OneLeg-v0",
    entry_point=f"{__name__}.assembly.assembly_env:AssemblyEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.assembly.assembly_env_cfg:OneLegEnvCfg"},
)

gym.register(
    id="Isaac-Furniture-OneLeg-FB-v0",
    entry_point=f"{__name__}.assembly.assembly_env:AssemblyEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.assembly.assembly_env_cfg:OneLegFBEnvCfg"},
)

gym.register(
    id="Isaac-Furniture-SquareTable-v0",
    entry_point=f"{__name__}.assembly.assembly_env:AssemblyEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.assembly.assembly_env_cfg:SquareTableEnvCfg"},
)
