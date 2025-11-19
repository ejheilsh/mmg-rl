import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from mmg_env import MMGEnv
from general import N_RAYS

def verify():
    # Load the trained model
    model_path = "mmg_ppo"
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path}.zip not found.")
        return

    # Create environment
    env = MMGEnv(u0=0, n_rays=N_RAYS, max_steps=2000, n_obstacles=10)
    
    obs, _ = env.reset()
    done = False
    truncated = False
    
    print("Running verification episode...")
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
    # Get history from the simulator
    df = env.sim.df_history
    
    print(f"Episode finished. Steps: {len(df)}")
    print(f"Final Position: x={df['x'].iloc[-1]:.2f}, y={df['y'].iloc[-1]:.2f}")
    print(f"Final Velocity: u={df['u'].iloc[-1]:.2f}")
    
    print("\nDataframe Head:")
    print(df.head())
    
    print("\nControl Statistics:")
    print(df[['nP', 'delta']].describe())
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trajectory
    axs[0, 0].plot(df['x'], df['y'], label='Trajectory')
    axs[0, 0].set_title('Ship Trajectory')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    axs[0, 0].axhline(0, color='gray', linestyle='--')
    
    # Plot obstacles
    for obs in env.sim.obstacles:
        circle = plt.Circle((obs["x"], obs["y"]), obs["r"], color='red', alpha=0.3)
        axs[0, 0].add_patch(circle)
        
    axs[0, 0].legend()
    axs[0, 0].axis('equal')
    
    # Velocity
    axs[0, 1].plot(df['t'], df['u'], label='Surge Velocity (u)')
    axs[0, 1].set_title('Surge Velocity vs Time')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('u (m/s)')
    axs[0, 1].legend()
    
    # Controls - RPS and Rudder
    axs[1, 0].plot(df['t'], df['nP'], label='Propeller RPS (nP)', color='orange')
    axs[1, 0].set_title('Controls vs Time')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('nP')
    
    ax2 = axs[1, 0].twinx()
    ax2.plot(df['t'], np.degrees(df['delta']), label='Rudder (deg)', color='purple', linestyle='--')
    ax2.set_ylabel('Delta (deg)')
    
    # Combine legends
    lines, labels = axs[1, 0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1, 0].legend(lines + lines2, labels + labels2, loc='upper left')
    
    # Heading
    axs[1, 1].plot(df['t'], np.degrees(df['psi']), label='Heading (deg)', color='green')
    axs[1, 1].set_title('Heading vs Time')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Psi (deg)')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('verification_plot.png')
    print("Plot saved to verification_plot.png")

if __name__ == "__main__":
    verify()
