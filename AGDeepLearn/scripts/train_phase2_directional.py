#!/usr/bin/env python3
"""
AGDeepLearn Phase 2: Directional Air-Strafing Trainer (Cure for Circular Drift)

Goal:
- Teach the agent to alternate strafes (A <-> D keys + camera mouse sync) to fly toward dynamic 3D Waypoint targets!
- Velocity Projection Reward: Maximize v_agent . u_target.
- Penalize circular drift turning away from target.
- Target Reached Bonus (+10.0) when reaching within 60 units of checkpoint.
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

class Phase2DirectionalEnv(gym.Env):
    """
    Phase 2 Specialized Gymnasium Environment: Waypoint-Targeted Directional Air-Strafing.
    Extended episode length: max_steps = 25000 (16.6 minutes of continuous bhop).
    """
    def __init__(self, step_delay=0.0005, max_steps=25000):
        super().__init__()
        self.base_env = OpenAGEnv(step_delay=step_delay, max_steps=max_steps)
        self.reward_calc = Phase2DirectionalRewardCalculator(smoothness_penalty_weight=0.04)
        
        # 18D Continuous Observation:
        # [0]: Horizontal Speed (V_xy / 500)
        # [1]: Vertical Velocity (V_z / 500)
        # [2]: On Ground State (1.0 / 0.0)
        # [3..10]: 8 Wall Lidar Rays
        # [11..13]: Velocity Vector (Vx, Vy, Vz normalized)
        # [14]: Pitch angle
        # [15]: Relative Angle to Target (-1.0 to +1.0)
        # [16]: Distance to Target (D_target / 1000.0)
        # [17]: Velocity Projection onto Target Vector (V_proj / 500.0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        # 2D Continuous Action Space: action[0] = yaw_delta, action[1] = sidemove selector
        self.action_space = spaces.Box(
            low=np.array([-15.0, -1.0], dtype=np.float32),
            high=np.array([15.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.last_yaw_delta = 0.0
        self.target_pos = (500.0, 0.0, 36.0)

    def _respawn_target(self, player_origin=(0.0, 0.0, 36.0)):
        # Spawn target at random distance (400 to 3000 units) and random angle (0 to 360 deg) from player
        dist = random.uniform(400.0, 3000.0)
        angle = random.uniform(0.0, 2 * math.pi)
        tx = player_origin[0] + dist * math.cos(angle)
        ty = player_origin[1] + dist * math.sin(angle)
        self.target_pos = (tx, ty, 36.0)

        # Sync target position to C++ shared memory for 3D beacon rendering & HUD display!
        if self.base_env.client:
            self.base_env.client.set_goal(self.target_pos)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_yaw_delta = 0.0
        self.reward_calc.reset()

        obs_vec, info = self.base_env.reset(seed=seed, options=options)
        
        if self.base_env.client.shm:
            self.base_env.client.request_reset(origin=(0.0, 0.0, 36.0), velocity=(300.0, 0.0, 0.0))

        obs = self.base_env.client.get_observation()
        player_origin = obs.player_origin if obs else (0.0, 0.0, 36.0)
        self._respawn_target(player_origin)

        phase2_obs = self._extract_phase2_obs(obs)
        return phase2_obs, info

    def _extract_phase2_obs(self, obs):
        if not obs:
            return np.zeros(18, dtype=np.float32)

        # Priority: If user placed active waypoints via ImGui HUD, use user's goal origin!
        if obs.goal_active == 1:
            self.target_pos = (obs.goal_origin[0], obs.goal_origin[1], obs.goal_origin[2])

        px, py = obs.player_origin[0], obs.player_origin[1]
        tx, ty = self.target_pos[0], self.target_pos[1]
        dx, dy = tx - px, ty - py
        dist_to_target = math.sqrt(dx * dx + dy * dy)

        vx, vy, vz = obs.player_velocity[0], obs.player_velocity[1], obs.player_velocity[2]
        current_speed = math.sqrt(vx * vx + vy * vy)

        # Calculate relative angle to target relative to facing viewangle
        angle_to_target = math.atan2(dy, dx)
        facing_yaw = math.radians(obs.viewangles[1])
        rel_angle = math.atan2(math.sin(angle_to_target - facing_yaw), math.cos(angle_to_target - facing_yaw))

        # Velocity Projection onto Target Vector
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
            float(rel_angle / math.pi),           # [15]: Relative angle to target (-1.0 to +1.0)
            float(dist_to_target / 1000.0),       # [16]: Distance to target
            float(v_proj / 500.0)                 # [17]: Velocity projection onto target
        ]
        return np.nan_to_num(np.array(obs_list, dtype=np.float32), nan=0.0)

    def step(self, action):
        yaw_delta = float(action[0])
        side_sel = float(action[1])

        sidemove = -400.0 if side_sel < 0.0 else 400.0
        forwardmove = 0.0 # W/S locked for air-strafing
        buttons = 2      # Auto-Bhop

        frame_skip = 4
        total_reward = 0.0
        terminated = False
        obs = None

        smoothed_yaw_delta = 0.6 * self.last_yaw_delta + 0.4 * yaw_delta

        for _ in range(frame_skip):
            obs = self.base_env.client.get_observation()
            r, reached = self.reward_calc.compute_reward(
                obs,
                target_pos=self.target_pos,
                yaw_delta=smoothed_yaw_delta,
                sidemove=sidemove
            )
            total_reward += r

            # Send action AND live reward value to C++ shared memory for ImGui HUD display!
            self.base_env.client.send_action(
                forwardmove=forwardmove,
                sidemove=sidemove,
                upmove=0.0,
                pitch_delta=0.0,
                yaw_delta=smoothed_yaw_delta,
                buttons=buttons,
                weapon_select=0,
                last_reward=float(r),
                reward_reason="Waypoint Nav"
            )

            if reached and obs and obs.goal_active == 0:
                self._respawn_target(obs.player_origin)

            if obs and obs.player_alive == 0:
                terminated = True
                break

        phase2_obs = self._extract_phase2_obs(obs)
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
        return phase2_obs, total_reward, terminated, truncated, info

    def close(self):
        self.base_env.close()

def main():
    parser = argparse.ArgumentParser(description="AGDeepLearn Phase 2: Directional Air-Strafing Trainer")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train")
    parser.add_argument("--from-phase1", action="store_true", help="Load Phase 1 weights as initial policy")
    args = parser.parse_args()

    print("=================================================================")
    print("  🚀 AGDeepLearn Phase 2: Directional Air-Strafing Trainer")
    print("  Goal: Master S-curve strafe navigation to dynamic 3D Waypoints!")
    print("=================================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = Phase2DirectionalEnv(step_delay=0.0005, max_steps=25000)

    checkpoint_name = os.path.join(models_dir, "phase2_directional_final.zip")
    phase1_checkpoint = os.path.join(models_dir, "phase1_air_strafing_timed_final.zip")

    if os.path.exists(checkpoint_name):
        print(f"[+] Continuing Phase 2 training from existing weights ({checkpoint_name})...")
        model = PPO.load(checkpoint_name, env=env, device="cpu", tensorboard_log=logs_dir)
    elif os.path.exists(phase1_checkpoint):
        print(f"[+] Automatically transferring Phase 1 air-strafing weights into Phase 2 ({phase1_checkpoint})...")
        model = PPO.load(phase1_checkpoint, env=env, device="cpu", tensorboard_log=logs_dir)
    else:
        print("[+] Initializing NEW PPO Agent for Phase 2 Directional Navigation Policy...")
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
        name_prefix="phase2_directional_ppo"
    )

    print(f"[+] Starting Phase 2 Directional Training for {args.timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        print("\n[+] Phase 2 Directional Training completed successfully!")
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    final_path = os.path.join(models_dir, "phase2_directional_final")
    model.save(final_path)
    print(f"[+] Saved Phase 2 Directional model weights to: {final_path}.zip")

    env.close()

if __name__ == "__main__":
    main()
