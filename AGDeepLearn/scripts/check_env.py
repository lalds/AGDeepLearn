#!/usr/bin/env python3
import sys
import os
import time

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agdeeplearn import OpenAGEnv

def main():
    print("=============================================")
    print("   AGDeepLearn Gymnasium Env Validation")
    print("=============================================")

    try:
        print("[+] Initializing OpenAGEnv...")
        env = OpenAGEnv(step_delay=0.03, max_steps=300)
        obs, info = env.reset()

        print(f"[+] Observation Space: {env.observation_space}")
        print(f"[+] Action Space:      {env.action_space}")
        print(f"[+] Initial Obs shape:  {obs.shape} (Player HP: {obs[10]}, Speed: {obs[7]:.1f})")

        print("\n[+] Running 100 Random Agent steps...")
        total_reward = 0.0

        for step in range(1, 101):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 20 == 0 or terminated:
                print(f"  Step #{step:03d} | Reward: {reward:+.3f} | Cumulative: {total_reward:+.3f} | Speed: {info['speed']:.1f} u/s | HP: {info['health']}")

            if terminated:
                print("  [!] Player died! Resetting environment...")
                obs, info = env.reset()

        print(f"\n[+] Test Episode Completed! Total Reward Accumulated: {total_reward:+.3f}")
        env.close()

    except Exception as e:
        print(f"\n[!] Error checking environment: {e}")

if __name__ == "__main__":
    main()
