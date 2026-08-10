#!/usr/bin/env python3
import os
import sys
import time

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from agdeeplearn import OpenAGEnv

def main():
    print("==================================================")
    print("    AGDeepLearn: Phase 3.1 - Evaluation Agent")
    print("==================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    model_path = os.path.join(models_dir, "bhop_agent_final.zip")

    if not os.path.exists(model_path):
        # Look for latest checkpoint if final model is not found
        checkpoints = [f for f in os.listdir(models_dir) if f.endswith(".zip")] if os.path.exists(models_dir) else []
        if checkpoints:
            checkpoints.sort()
            model_path = os.path.join(models_dir, checkpoints[-1])
        else:
            print(f"[!] Error: No model checkpoints found in '{models_dir}'. Run train_bhop.py first!")
            return

    print(f"[+] Loading trained model from: {model_path}")
    model = PPO.load(model_path, device="cpu")

    env = OpenAGEnv(step_delay=0.02, max_steps=1000)
    obs, info = env.reset()

    print("[+] Model loaded! Controlling character in OpenAG (Ctrl+C to stop)...\n")
    try:
        step = 0
        total_reward = 0.0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if step % 20 == 0:
                print(f"  [Eval Step #{step:04d}] Speed: {info['speed']:.1f} u/s | HP: {info['health']} | Reward: {reward:+.3f} | Cumulative: {total_reward:+.3f}")

            if terminated or truncated:
                print("  [!] Episode ended. Resetting...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\nStopping evaluation...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
