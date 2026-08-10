#!/usr/bin/env python3
"""
AGDeepLearn Inference Runner: Run Trained Model in Real-Time Evaluation Mode.

Usage:
  python3 scripts/run_agent.py [--checkpoint PATH] [--fps 60]
"""

import os
import sys
import time
import argparse
from stable_baselines3 import PPO

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.train_phase2_directional import Phase2DirectionalEnv

def main():
    parser = argparse.ArgumentParser(description="AGDeepLearn Real-Time Inference Runner")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model .zip checkpoint")
    args = parser.parse_args()

    print("=================================================================")
    print("  🚀 AGDeepLearn Real-Time Evaluation Runner")
    print("=================================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Check Fine-Tuned Champion, Phase 4 Spatial Vision, Master Backup, Phase 2, Phase 1
        finetune_race = os.path.join(models_dir, "finetune_race_final.zip")
        phase4_spatial = os.path.join(models_dir, "phase4_spatial_vision_final.zip")
        backup_master = os.path.join(os.path.dirname(__file__), "..", "backups", "bhop_human_master_phase2_directional.zip")
        phase2_path = os.path.join(models_dir, "phase2_directional_final.zip")
        phase1_path = os.path.join(models_dir, "phase1_air_strafing_timed_final.zip")
        
        if os.path.exists(finetune_race):
            checkpoint_path = finetune_race
            env = Phase2DirectionalEnv(step_delay=0.0005, max_steps=100000)
        elif os.path.exists(phase4_spatial):
            checkpoint_path = phase4_spatial
            from scripts.train_phase4_spatial_vision import Phase4SpatialVisionEnv
            env = Phase4SpatialVisionEnv(step_delay=0.0005, max_steps=100000)
        elif os.path.exists(backup_master):
            checkpoint_path = backup_master
            env = Phase2DirectionalEnv(step_delay=0.0005, max_steps=100000)
        elif os.path.exists(phase2_path):
            checkpoint_path = phase2_path
            env = Phase2DirectionalEnv(step_delay=0.0005, max_steps=100000)
        elif os.path.exists(phase1_path):
            checkpoint_path = phase1_path
            env = Phase2DirectionalEnv(step_delay=0.0005, max_steps=100000)
        else:
            print(f"[!] Error: No trained model checkpoint found in {models_dir}!")
            sys.exit(1)

    print(f"[+] Loading trained policy weights from: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env, device="cpu")

    print("[+] Model loaded successfully!")
    print("[+] Running autonomous evaluation loop... (Press Ctrl+C to stop)")

    obs, info = env.reset()
    ep_reward = 0.0
    ep_steps = 0

    try:
        while True:
            # Deterministic inference (no random exploration noise)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_steps += 1

            current_speed = info.get("speed", 0.0)
            if ep_steps % 50 == 0:
                print(f"[INFERENCE] Speed: {current_speed:.1f} u/s | Step Reward: {reward:+.2f} | Ep Total: {ep_reward:+.1f}", end="\r", flush=True)

            if terminated or truncated:
                print(f"\n[+] Episode complete! Total Reward: {ep_reward:.1f} over {ep_steps} steps. Restarting...")
                obs, info = env.reset()
                ep_reward = 0.0
                ep_steps = 0

    except KeyboardInterrupt:
        print("\n[!] Stopping inference runner.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
