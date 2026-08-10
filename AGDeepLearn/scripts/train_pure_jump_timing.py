#!/usr/bin/env python3
"""
AGDeepLearn Isolated Skill: Pure 1D Jump Timing Trainer.

Goal:
- Character constantly runs forward (+forward = 400).
- 1D Action Space: Only Jump Trigger (Press/Release +jump).
- Neural Network focuses 100% of its learning capacity on ground-touch jump timing (Manual Bhop)!
"""

import os
import sys
import argparse
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from agdeeplearn import OpenAGEnv

class PureJumpTimingEnv(gym.Env):
    """
    Isolated Gymnasium Environment: Pure 1D Jump Timing (Forward Sprint Locked).
    """
    def __init__(self, step_delay=0.0005, max_steps=10000):
        super().__init__()
        self.base_env = OpenAGEnv(step_delay=step_delay, max_steps=max_steps)
        
        # 3D Observation: [speed/500, vz/500, on_ground]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # 1D Action Space: action[0] > 0.0 => Press JUMP (+jump)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.last_speed = 0.0
        self.jump_hold_ticks = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_speed = 0.0
        self.jump_hold_ticks = 0

        obs_vec, info = self.base_env.reset(seed=seed, options=options)
        if self.base_env.client.shm:
            self.base_env.client.request_reset(origin=(0.0, 0.0, 36.0), velocity=(320.0, 0.0, 0.0))

        obs = self.base_env.client.get_observation()
        return self._extract_obs(obs), info

    def _extract_obs(self, obs):
        if not obs:
            return np.zeros(3, dtype=np.float32)

        vx, vy, vz = obs.player_velocity[0], obs.player_velocity[1], obs.player_velocity[2]
        current_speed = math.sqrt(vx * vx + vy * vy)
        on_ground = float(obs.on_ground)

        return np.array([
            float(current_speed / 500.0),
            float(vz / 500.0),
            float(on_ground)
        ], dtype=np.float32)

    def step(self, action):
        jump_sel = float(action[0])
        jump_pressed = bool(jump_sel > 0.0)

        # Locked forward sprint (+forward = 400)
        forwardmove = 400.0
        buttons = 2 if jump_pressed else 0

        frame_skip = 4
        total_reward = 0.0
        terminated = False
        obs = None

        for _ in range(frame_skip):
            obs = self.base_env.client.get_observation()
            if not obs or obs.player_alive == 0:
                terminated = True
                break

            vx, vy = obs.player_velocity[0], obs.player_velocity[1]
            current_speed = math.sqrt(vx * vx + vy * vy)
            on_ground = bool(obs.on_ground)

            r = 0.0

            # 1. Anti-Macro / Anti-Spam Penalty (Holding jump continuously)
            if jump_pressed:
                self.jump_hold_ticks += 1
                if self.jump_hold_ticks > 3:
                    r -= 0.15
            else:
                self.jump_hold_ticks = 0

            # 2. Perfect Ground-Touch Jump Timing Bonus
            if on_ground and jump_pressed:
                r += 2.00  # High bonus for pressing jump on ground touch!

            # 3. Ground Friction Penalty (Staying stuck on ground)
            if on_ground and not jump_pressed:
                speed_delta = current_speed - self.last_speed
                if speed_delta < -10.0:
                    r -= 0.80  # Heavy penalty for friction eating velocity!

            # 4. Kinetic Speed Gain Bonus
            if current_speed > 320.0:
                r += (current_speed - 320.0) * 0.02

            total_reward += r
            self.last_speed = current_speed

            self.base_env.client.send_action(
                forwardmove=forwardmove,
                sidemove=0.0,
                upmove=0.0,
                pitch_delta=0.0,
                yaw_delta=0.0,
                buttons=buttons,
                weapon_select=0,
                last_reward=float(r),
                reward_reason="Pure 1D Jump Timing"
            )

        pure_obs = self._extract_obs(obs)
        truncated = bool(self.base_env.current_step >= self.base_env.max_steps)

        info = {
            "speed": self.last_speed,
            "jump_pressed": jump_pressed
        }

        return pure_obs, total_reward, terminated, truncated, info

    def close(self):
        self.base_env.close()

def main():
    parser = argparse.ArgumentParser(description="AGDeepLearn Pure 1D Jump Timing Trainer")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train")
    args = parser.parse_args()

    print("=================================================================")
    print("  🚀 AGDeepLearn Isolated Skill: Pure 1D Jump Timing Trainer")
    print("  Mode: Forward Sprint Locked (+forward = 400)")
    print("  Action: 1D Jump Trigger ONLY (+jump / -jump)")
    print("=================================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = PureJumpTimingEnv(step_delay=0.0005, max_steps=10000)
    checkpoint_name = os.path.join(models_dir, "pure_jump_timing_final.zip")

    if os.path.exists(checkpoint_name):
        print(f"[+] Continuing Pure Jump Timing training from existing weights ({checkpoint_name})...")
        model = PPO.load(checkpoint_name, env=env, device="cpu", tensorboard_log=logs_dir)
    else:
        print("[+] Initializing NEW 1D PPO Agent for Pure Jump Timing...")
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log=logs_dir
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="pure_jump_timing_ppo"
    )

    print(f"[+] Starting Pure 1D Jump Timing Training for {args.timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        print("\n[+] Pure 1D Jump Timing Training completed successfully!")
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    final_path = os.path.join(models_dir, "pure_jump_timing_final")
    model.save(final_path)
    print(f"[+] Saved Pure Jump Timing model weights to: {final_path}.zip")

    env.close()

if __name__ == "__main__":
    main()
