from gymnasium.envs.registration import register

register(
    id="simple_grid_world/GridWorld-v0",
    entry_point="simple_grid_world.envs:GridWorldEnv",
    max_episode_steps=300,
)