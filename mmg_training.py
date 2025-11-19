"""
Parallelized PPO training using vectorized environments.
Adjust N_ENVS to control parallel rollout workers.
"""
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from mmg_env import MMGEnv
from general import N_RAYS

# Number of parallel envs (increase if CPU allows)
N_ENVS = 8

# Build vectorized env; monitor_dir enables episode stats logging
vec_env = make_vec_env(
    MMGEnv,
    n_envs=N_ENVS,
    env_kwargs={"u0": 0, "n_rays": N_RAYS, "max_steps": 20000, "n_obstacles": 10},
    monitor_dir="./mmg_logs",
)
vec_env = VecMonitor(vec_env)

# separate eval env (single instance) for best-model checkpointing
eval_env = make_vec_env(
    MMGEnv,
    n_envs=1,
    env_kwargs={"u0": 0, "n_rays": N_RAYS, "max_steps": 20000},
    monitor_dir="./mmg_eval_logs",
)
eval_env = VecMonitor(eval_env)

# n_steps is per-env rollout length; total steps per update = n_steps * N_ENVS
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./mmg_tb/",
    n_steps=10000,          # per-env steps; with 8 envs -> 2048 samples/update
    batch_size=256,
    learning_rate=3e-4,
    ent_coef=0.0,
    clip_range=0.2,
)

# save best model (highest mean reward) during training
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./mmg_best",
    log_path="./mmg_eval_logs",
    eval_freq=100_000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

model.learn(total_timesteps=3e6, callback=eval_callback)
model.save("mmg_ppo")
