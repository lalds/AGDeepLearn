#!/usr/bin/env python3
"""
AGDeepLearn Phase 3: Manual Jump Timing & Friction-Preservation Bhop Trainer.

Goal:
- Disable Auto-Bhop!
- Force neural network to learn perfect ground-touch jump timing (Manual Bhop).
- Punish continuous jump holding (Anti-Macro penalty) and ground friction velocity loss.
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
from agdeeplearn import OpenAGEnv, Phase3ManualJumpRewardCalculator

class Phase3ManualJumpEnv(gym.Env):
    """
    Phase 3 Gymnasium Environment: Manual Jump Timing & Ground-Touch Friction Bhop.
    """
    def __init__(self, step_delay=0.0005, max_steps=25000):
        super().__init__()
        self.base_env = OpenAGEnv(step_delay=step_delay, max_steps=max_steps)
        self.reward_calc = Phase3ManualJumpRewardCalculator(smoothness_penalty_weight=0.04)
        
        # 18D Continuous Observation Space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        # 3D Continuous Action Space:
        # action[0]: yaw_delta (-15.0 to +15.0)
        # action[1]: sidemove (-1.0 to +1.0 => pure A/D keys -400 or +400)
        # action[2]: jump_sel (-1.0 to +1.0 => JUMP button trigger) -- MANUAL BHOP!
        self.action_space = spaces.Box(
            low=np.array([-15.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([15.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.last_yaw_delta = 0.0
        self.target_pos = (500.0, 0.0, 36.0)

    def _respawn_target(self, player_origin=(0.0, 0.0, 36.0)):
        dist = random.uniform(400.0, 3000.0)
        angle = random.uniform(0.0, 2 * math.pi)
        tx = player_origin[0] + dist * math.cos(angle)
        ty = player_origin[1] + dist * math.sin(angle)
        self.target_pos = (tx, ty, 36.0)

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

        phase3_obs = self._extract_obs(obs)
        return phase3_obs, info

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
        jump_sel = float(action[2])

        sidemove = -400.0 if side_sel < 0.0 else 400.0
        
        # NO AUTO-BHOP! Jump button bit 2 is controlled strictly by AI action[2]
        jump_pressed = bool(jump_sel > 0.0)
        buttons = 2 if jump_pressed else 0

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
                sidemove=sidemove,
                jump_pressed=jump_pressed
            )
            total_reward += r

            self.base_env.client.send_action(
                forwardmove=0.0,
                sidemove=sidemove,
                upmove=0.0,
                pitch_delta=0.0,
                yaw_delta=smoothed_yaw_delta,
                buttons=buttons,
                weapon_select=0,
                last_reward=float(r),
                reward_reason="Phase 3 Manual Jump Bhop"
            )

            if reached and obs and obs.goal_active == 0:
                self._respawn_target(obs.player_origin)

            if obs and obs.player_alive == 0:
                terminated = True
                break

        phase3_obs = self._extract_obs(obs)
        truncated = bool(self.base_env.current_step >= self.base_env.max_steps)

        vx, vy = (obs.player_velocity[0] if obs else 0.0), (obs.player_velocity[1] if obs else 0.0)
        current_speed = math.sqrt(vx * vx + vy * vy)

        info = {
            "speed": current_speed,
            "yaw_delta": smoothed_yaw_delta,
            "sidemove": sidemove,
            "jump_pressed": jump_pressed,
            "target_pos": self.target_pos
        }

        self.last_yaw_delta = smoothed_yaw_delta
        return phase3_obs, total_reward, terminated, truncated, info

    def close(self):
        self.base_env.close()

def main():
    parser = argparse.ArgumentParser(description="AGDeepLearn Phase 3: Manual Jump Timing Trainer")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train")
    args = parser.parse_args()

    print("=================================================================")
    print("  🚀 AGDeepLearn Phase 3: Manual Jump Timing & Friction Bhop Trainer")
    print("  Auto-Bhop: DISABLED! Network learns ground-touch jump timing.")
    print("=================================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = Phase3ManualJumpEnv(step_delay=0.0005, max_steps=25000)

    checkpoint_name = os.path.join(models_dir, "phase3_manual_jump_final.zip")
    phase2_checkpoint = os.path.join(models_dir, "phase2_directional_final.zip")

    if os.path.exists(checkpoint_name):
        print(f"[+] Continuing Phase 3 training from existing weights ({checkpoint_name})...")
        model = PPO.load(checkpoint_name, env=env, device="cpu", tensorboard_log=logs_dir)
    elif os.path.exists(phase2_checkpoint):
        print(f"[+] Transferring Phase 2 directional weights into Phase 3 ({phase2_checkpoint})...")
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
    else:
        print("[+] Initializing NEW PPO Agent for Phase 3 Policy...")
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
        name_prefix="phase3_manual_jump_ppo"
    )

    print(f"[+] Starting Phase 3 Manual Jump Training for {args.timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        print("\n[+] Phase 3 Manual Jump Training completed successfully!")
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    final_path = os.path.join(models_dir, "phase3_manual_jump_final")
    model.save(final_path)
    print(f"[+] Saved Phase 3 Manual Jump model weights to: {final_path}.zip")

    env.close()

if __name__ == "__main__":
    main()
