#!/usr/bin/env python3
import os
import sys
import time

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from agdeeplearn import OpenAGEnv

def main():
    print("==================================================")
    print("    AGDeepLearn: Phase 3.1 - Movement Agent (PPO)")
    print("==================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("[+] Connecting to OpenAG Environment...")
    env = OpenAGEnv(step_delay=0.02, max_steps=500)

    print("[+] Initializing PPO Agent Policy on CPU (MlpPolicy)...")
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=logs_dir
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=models_dir,
        name_prefix="bhop_ppo_model"
    )

    total_timesteps = 20000
    print(f"[+] Starting RL Training for {total_timesteps} timesteps...")
    print("    Press Ctrl+C at any time to interrupt training and save current progress.\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        print("\n[+] Training completed successfully!")
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    final_model_path = os.path.join(models_dir, "bhop_agent_final")
    model.save(final_model_path)
    print(f"[+] Saved model checkpoint to: {final_model_path}.zip")

    env.close()

if __name__ == "__main__":
    main()
