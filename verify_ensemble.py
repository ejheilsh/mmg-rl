import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from mmg_env import MMGEnv
from general import N_RAYS

def run_ensemble():
    # Create environment
    # We will recreate the env inside the loop or reset it to get new obstacles if seed is not fixed
    # But MMGEnv takes n_obstacles in init.
    # To get different obstacles, we just need to call reset().
    # However, to plot obstacles, we need access to them.
    
    env = MMGEnv(u0=0, n_rays=N_RAYS, max_steps=2000, n_obstacles=10)
    
    # Load model
    try:
        model = PPO.load("mmg_ppo")
    except FileNotFoundError:
        print("Model not found. Please train first.")
        return

    n_episodes = 10
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        history = []
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Record state
            state = env.sim.x
            # state: x, y, psi, u, vm, r
            history.append({
                'x': state[0],
                'y': state[1],
                'u': state[3]
            })
            
        df = pd.DataFrame(history)
        
        # Plot on subplot
        ax = axs[i]
        ax.plot(df['x'], df['y'], label='Trajectory')
        ax.set_title(f'Run {i+1}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axhline(0, color='gray', linestyle='--')
        
        # Plot obstacles
        for obs_data in env.sim.obstacles:
            circle = plt.Circle((obs_data["x"], obs_data["y"]), obs_data["r"], color='red', alpha=0.3)
            ax.add_patch(circle)
            
        ax.axis('equal')
        
        # Print summary
        print(f"Run {i+1}: Final x={df['x'].iloc[-1]:.2f}, y={df['y'].iloc[-1]:.2f}, u={df['u'].iloc[-1]:.2f}")

    plt.tight_layout()
    plt.savefig('ensemble_plot.png')
    print("Ensemble plot saved to ensemble_plot.png")

if __name__ == "__main__":
    run_ensemble()
