from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from mmg_env import MMGEnv

env = MMGEnv(u0=0)
env = Monitor(env)   # IMPORTANT: enables episode reward logging

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./mmg_tb/"
)

model.learn(total_timesteps=100_000)

model.save("mmg_ppo")


# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from mmg_env import MMGEnv

# env = MMGEnv(u0=2.0, rand_seed=0)
# env = Monitor(env)   # IMPORTANT: enables episode reward logging

# model = PPO(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     tensorboard_log="./mmg_tb/"
# )

# model.learn(total_timesteps=200_000)

# model.save("mmg_ppo")