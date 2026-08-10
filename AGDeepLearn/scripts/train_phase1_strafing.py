#!/usr/bin/env python3
"""
Phase 1: Air-Strafing Physics Mastery Training Script

Goal:
- Teach the agent that smooth camera Yaw rotation synchronized with A/D keys yields infinite speed acceleration.
- Infinite flat surface, auto-bhop enabled (server jumps on ground contact).
- W/S movement locked to 0.0.
- Action Smoothness Penalty to eliminate mouse jitter.
"""

import os
import sys
import argparse
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from agdeeplearn import OpenAGEnv, Phase1AirStrafeRewardCalculator

class Phase1AirStrafeEnv(gym.Env):
    """
    Phase 1 Specialized Gymnasium Environment.
    Mastering Engine Physics (Air-Strafing Acceleration).
    Extended episode length: max_steps = 5000 (3.33 minutes of continuous bhop).
    """
    def __init__(self, step_delay=0.0005, max_steps=5000):
        super().__init__()
        self.base_env = OpenAGEnv(step_delay=step_delay, max_steps=max_steps)
        self.reward_calc = Phase1AirStrafeRewardCalculator(
            smoothness_penalty_weight=0.04,
            speed_scale=0.03
        )
        
        # 16D Continuous Observation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        # 2D Continuous Action Space: action[0] = yaw_delta, action[1] = sidemove selector (-1 => A key, +1 => D key)
        self.action_space = spaces.Box(
            low=np.array([-15.0, -1.0], dtype=np.float32),
            high=np.array([15.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.last_yaw_delta = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_yaw_delta = 0.0
        self.reward_calc.reset()

        obs_vec, info = self.base_env.reset(seed=seed, options=options)
        
        if self.base_env.client.shm:
            self.base_env.client.request_reset(origin=(0.0, 0.0, 36.0), velocity=(300.0, 0.0, 0.0))

        obs = self.base_env.client.get_observation()
        phase1_obs = self._extract_phase1_obs(obs)
        return phase1_obs, info

    def _extract_phase1_obs(self, obs):
        if not obs:
            return np.zeros(16, dtype=np.float32)

        vx, vy, vz = obs.player_velocity[0], obs.player_velocity[1], obs.player_velocity[2]
        current_speed_units = math.sqrt(vx * vx + vy * vy)
        speed = current_speed_units / 500.0

        lidar = [obs.wall_distances[i] / 1000.0 for i in range(8)]
        on_ground = float(obs.on_ground)

        obs_list = [
            float(speed), float(vz / 500.0), float(on_ground),
            *lidar,
            float(vx / 500.0), float(vy / 500.0), float(vz / 500.0),
            float(obs.viewangles[0] / 90.0),
            float(self.last_yaw_delta / 15.0)
        ]
        return np.nan_to_num(np.array(obs_list, dtype=np.float32), nan=0.0)

    def step(self, action):
        yaw_delta = float(action[0])
        side_sel = float(action[1])

        sidemove = -400.0 if side_sel < 0.0 else 400.0
        forwardmove = 0.0 # W/S LOCKED TO ZERO FOR PHASE 1 AIR-STRAFING!
        buttons = 2      # Auto-Bhop

        # Action Persistence (frame_skip=4): Repeat action over 4 physics frames (40ms window)
        frame_skip = 4
        total_reward = 0.0
        terminated = False
        obs = None

        smoothed_yaw_delta = 0.6 * self.last_yaw_delta + 0.4 * yaw_delta

        for _ in range(frame_skip):
            self.base_env.client.send_action(
                forwardmove=forwardmove,
                sidemove=sidemove,
                upmove=0.0,
                pitch_delta=0.0,
                yaw_delta=smoothed_yaw_delta,
                buttons=buttons,
                weapon_select=0
            )

            obs = self.base_env.client.get_observation()
            r = self.reward_calc.compute_reward(
                obs,
                yaw_delta=smoothed_yaw_delta,
                sidemove=sidemove
            )
            total_reward += r

            if obs and obs.player_alive == 0:
                terminated = True
                break

        phase1_obs = self._extract_phase1_obs(obs)
        truncated = bool(self.base_env.current_step >= self.base_env.max_steps)

        vx, vy = (obs.player_velocity[0] if obs else 0.0), (obs.player_velocity[1] if obs else 0.0)
        current_speed = math.sqrt(vx * vx + vy * vy)

        info = {
            "speed": current_speed,
            "yaw_delta": smoothed_yaw_delta,
            "sidemove": sidemove
        }

        self.last_yaw_delta = smoothed_yaw_delta
        return phase1_obs, total_reward, terminated, truncated, info

    def close(self):
        self.base_env.close()

def main():
    parser = argparse.ArgumentParser(description="AGDeepLearn Phase 1 Air-Strafing Mastery Trainer")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps to train")
    parser.add_argument("--side", type=str, choices=["left", "right", "both"], default="both", help="Target strafe side: left, right, or both")
    args = parser.parse_args()

    print("=================================================================")
    print(f"  🚀 AGDeepLearn Phase 1: Air-Strafing Physics Mastery Trainer (Side={args.side.upper()})")
    print("  Goal: Learn smooth continuous arc strafing & infinite velocity gain")
    print("=================================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = Phase1AirStrafeEnv(step_delay=0.0005, max_steps=5000)

    checkpoint_name = os.path.join(models_dir, "phase1_air_strafing_timed_final.zip")
    if os.path.exists(checkpoint_name):
        print(f"[+] Continuing Phase 1 Timed Arc Switch training from existing weights ({checkpoint_name})...")
        model = PPO.load(checkpoint_name, env=env, device="cpu", tensorboard_log=logs_dir)
    else:
        print("[+] Initializing NEW PPO Agent for Timed Sequential Arc Switching Policy (0.6s - 3.5s)...")
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
        name_prefix="phase1_bhop_timed_ppo"
    )

    print(f"[+] Starting Timed Arc Switch Training for {args.timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        print("\n[+] Timed Arc Switch Training completed successfully!")
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    final_path = os.path.join(models_dir, "phase1_air_strafing_timed_final")
    model.save(final_path)
    print(f"[+] Saved Timed Arc Switch model weights to: {final_path}.zip")

    env.close()

if __name__ == "__main__":
    main()
