#!/usr/bin/env python3
"""
AGDeepLearn Circuit Race Mode (AFK Free Exploration & Speedrun Trainer).

Goal:
- 3D Volume Box Triggers (Configurable size in ImGui HUD).
- Loop around circuit triggers continuously without resetting player position.
- Relaxed micro-penalties to allow the agent to experiment with wild high-speed shortcuts, wall-bounces & wide arcs while you rest (AFK)!
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
from agdeeplearn import OpenAGEnv, Phase2DirectionalRewardCalculator
from scripts.train_phase2_directional import Phase2DirectionalEnv

class CircuitRaceEnv(Phase2DirectionalEnv):
    """
    Gymnasium Environment: AFK Circuit Race & Exploration Mode.
    Uses proven Phase2DirectionalEnv movement physics with 50,000 max_steps per episode.
    """
    def __init__(self, step_delay=0.0005, max_steps=50000):
        super().__init__(step_delay=step_delay, max_steps=max_steps)
        # Relaxed smoothness penalty for wild high-speed exploration!
        self.reward_calc = Phase2DirectionalRewardCalculator(smoothness_penalty_weight=0.01)

    def _extract_obs(self, obs):
        if not obs:
            return np.zeros(18, dtype=np.float32)

        if obs.goal_active == 1:
            self.target_pos = (obs.goal_origin[0], obs.goal_origin[1], obs.goal_origin[2])

        px, py = obs.player_origin[0], obs.player_origin[1]
        tx, ty = self.target_pos[0], self.target_pos[1]
        dx, dy = tx - px, ty - py
        dist_to_target = math.sqrt(dx * dx + dy * dy)

        vx, vy, vz = obs.player_velocity[0], obs.player_velocity[1], obs.player_velocity[2]
        current_speed = math.sqrt(vx * vx + vy * vy)

        angle_to_target = math.atan2(dy, dx)
        facing_yaw = math.radians(obs.viewangles[1])
        rel_angle = math.atan2(math.sin(angle_to_target - facing_yaw), math.cos(angle_to_target - facing_yaw))

        if dist_to_target > 0.001:
            ux, uy = dx / dist_to_target, dy / dist_to_target
            v_proj = vx * ux + vy * uy
        else:
            v_proj = 0.0

        lidar = [obs.wall_distances[i] / 1000.0 for i in range(8)]
        on_ground = float(obs.on_ground)

        obs_list = [
            float(current_speed / 500.0), float(vz / 500.0), float(on_ground),
            *lidar,
            float(vx / 500.0), float(vy / 500.0), float(vz / 500.0),
            float(obs.viewangles[0] / 90.0),
            float(rel_angle / math.pi),
            float(dist_to_target / 1000.0),
            float(v_proj / 500.0)
        ]
        return np.nan_to_num(np.array(obs_list, dtype=np.float32), nan=0.0)

    def step(self, action):
        yaw_delta = float(action[0])
        side_sel = float(action[1])

        sidemove = -400.0 if side_sel < 0.0 else 400.0
        buttons = 2  # Auto-Bhop ONLY (Duck-Spam Completely REMOVED!)

        frame_skip = 4
        total_reward = 0.0
        terminated = False
        obs = None

        smoothed_yaw_delta = 0.6 * self.last_yaw_delta + 0.4 * yaw_delta

        forwardmove = 0.0  # Pure Air-Strafes ONLY (No W key!)

        for _ in range(frame_skip):
            obs = self.base_env.client.get_observation()
            r, reached = self.reward_calc.compute_reward(
                obs,
                target_pos=self.target_pos,
                yaw_delta=smoothed_yaw_delta,
                sidemove=sidemove
            )

            # Circuit Race Speedrun Bonus: Big jackpot for hitting 3D trigger!
            if reached:
                r += 50.0

            total_reward += r

            self.base_env.client.send_action(
                forwardmove=forwardmove,
                sidemove=sidemove,
                upmove=0.0,
                pitch_delta=0.0,
                yaw_delta=smoothed_yaw_delta,
                buttons=buttons,
                weapon_select=0,
                last_reward=float(r),
                reward_reason="AFK Circuit Race"
            )

            if obs and obs.player_alive == 0:
                terminated = True
                break

        race_obs = self._extract_obs(obs)
        truncated = bool(self.base_env.current_step >= self.base_env.max_steps)

        vx, vy = (obs.player_velocity[0] if obs else 0.0), (obs.player_velocity[1] if obs else 0.0)
        current_speed = math.sqrt(vx * vx + vy * vy)

        info = {
            "speed": current_speed,
            "yaw_delta": smoothed_yaw_delta,
            "sidemove": sidemove,
            "target_pos": self.target_pos
        }

        self.last_yaw_delta = smoothed_yaw_delta
        return race_obs, total_reward, terminated, truncated, info

    def close(self):
        self.base_env.close()

def main():
    parser = argparse.ArgumentParser(description="AGDeepLearn AFK Circuit Race Mode")
    parser.add_argument("--timesteps", type=int, default=10000000, help="Total timesteps to train")
    args = parser.parse_args()

    print("=================================================================")
    print("  🏁 AGDeepLearn AFK Circuit Race Mode & Exploration Trainer")
    print("  Triggers: 3D Volume Box Triggers (Adjustable in ImGui HUD)")
    print("  Target: 10,000,000 Timesteps (~2.7 Hours AFK Training)")
    print("  Optimization: n_steps=16384, batch_size=512 (Zero-Stutter Rollout)")
    print("=================================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    backups_dir = os.path.join(os.path.dirname(__file__), "..", "backups")
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = CircuitRaceEnv(step_delay=0.0005, max_steps=50000)
    checkpoint_name = os.path.join(models_dir, "circuit_race_final.zip")
    master_backup = os.path.join(backups_dir, "bhop_human_master_phase2_directional.zip")

    if os.path.exists(master_backup):
        print(f"[+] Loading Golden Master Phase 2 weights (642+ u/s) as base ({master_backup})...")
        model = PPO.load(master_backup, env=env, device="cpu", tensorboard_log=logs_dir)
    else:
        print("[+] Initializing NEW PPO Agent for Circuit Race Mode...")
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            verbose=1,
            learning_rate=0.0003,
            n_steps=65536,      # 65+ SECONDS of continuous 790 u/s flying per update!
            batch_size=1024,    # Fast CPU matrix math
            n_epochs=4,         # Lightning-fast 0.1s update on CPU (no pauses!)
            gamma=0.99,
            tensorboard_log=logs_dir
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="circuit_race_ppo"
    )

    print(f"[+] Starting AFK Circuit Race Training for {args.timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        print("\n[+] AFK Circuit Race Training completed successfully!")
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    final_path = os.path.join(models_dir, "circuit_race_final")
    model.save(final_path)
    print(f"[+] Saved Circuit Race model weights to: {final_path}.zip")

    env.close()

if __name__ == "__main__":
    main()
