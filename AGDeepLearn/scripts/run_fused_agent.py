#!/usr/bin/env python3
"""
AGDeepLearn Master Fused Agent (Unified Autonomous Manual Bhop Runner).

Fuses two specialized expert neural networks into one autonomous super-agent:
1. Model A (Strafe & Navigation Expert): backups/bhop_human_master_phase2_directional.zip
   - Outputs: Frame-perfect mouse yaw arcs & pure A/D air-strafes (791 u/s speed).
2. Model B (Manual Jump Timing Expert): checkpoints/pure_jump_timing_final.zip
   - Outputs: Millisecond ground-touch jump trigger (+jump / -jump).

Result: 100% Fully Autonomous Manual Bhop Agent (Auto-Bhop DISABLED)!
"""

import os
import sys
import time
import math
import numpy as np

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from agdeeplearn import OpenAGEnv, OpenAGSHMClient

def main():
    print("=================================================================")
    print("  🚀 AGDeepLearn Fused Autonomous Master Agent Runner")
    print("  Fused Policy: 2D Air-Strafe Expert + 1D Manual Jump Timing Expert")
    print("  Auto-Bhop: DISABLED! 100% Autonomous Ground-Touch Jump Timing!")
    print("=================================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    backups_dir = os.path.join(os.path.dirname(__file__), "..", "backups")

    model_strafe_path = os.path.join(backups_dir, "bhop_human_master_phase2_directional.zip")
    model_jump_path = os.path.join(models_dir, "pure_jump_timing_final.zip")

    if not os.path.exists(model_strafe_path):
        model_strafe_path = os.path.join(models_dir, "phase2_directional_final.zip")

    if not os.path.exists(model_strafe_path) or not os.path.exists(model_jump_path):
        print("[!] Error: Required expert model checkpoints not found!")
        sys.exit(1)

    print(f"[+] Loading Strafe & Navigation Expert Model: {model_strafe_path}")
    model_strafe = PPO.load(model_strafe_path, device="cpu")

    print(f"[+] Loading Manual Jump Timing Expert Model: {model_jump_path}")
    model_jump = PPO.load(model_jump_path, device="cpu")

    print("[+] Both Expert Models loaded successfully!")

    client = OpenAGSHMClient()
    try:
        client.connect()
        print("[+] Connected to OpenAG Shared Memory!")
    except FileNotFoundError:
        print("[!] Error: OpenAG shared memory not found. Launch Half-Life with cl_agent_api 1.")
        sys.exit(1)

    print("\n[+] Autonomous Fused Master Agent ACTIVE! Running in Half-Life...")
    print("  Press Ctrl+C to stop.")

    target_pos = (500.0, 0.0, 36.0)
    last_seq = 0
    last_yaw_delta = 0.0

    try:
        while True:
            obs = client.get_observation()
            if not obs or obs.sequence == last_seq or obs.player_alive == 0:
                time.sleep(0.001)
                continue

            last_seq = obs.sequence

            if obs.goal_active == 1:
                target_pos = (obs.goal_origin[0], obs.goal_origin[1], obs.goal_origin[2])

            px, py = obs.player_origin[0], obs.player_origin[1]
            tx, ty = target_pos[0], target_pos[1]
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

            # 18D Observation for Strafe Model A
            obs_18d = np.nan_to_num(np.array([
                float(current_speed / 500.0), float(vz / 500.0), float(on_ground),
                *lidar,
                float(vx / 500.0), float(vy / 500.0), float(vz / 500.0),
                float(obs.viewangles[0] / 90.0),
                float(rel_angle / math.pi),
                float(dist_to_target / 1000.0),
                float(v_proj / 500.0)
            ], dtype=np.float32), nan=0.0)

            # 3D Observation for Jump Model B
            obs_3d = np.array([
                float(current_speed / 500.0),
                float(vz / 500.0),
                float(on_ground)
            ], dtype=np.float32)

            # 1. Model A Predicts Strafe & Yaw Mouse Turn
            act_strafe, _ = model_strafe.predict(obs_18d, deterministic=True)
            yaw_delta = float(act_strafe[0])
            side_sel = float(act_strafe[1])
            sidemove = -400.0 if side_sel < 0.0 else 400.0

            # Smooth mouse turn
            smoothed_yaw_delta = 0.6 * last_yaw_delta + 0.4 * yaw_delta
            last_yaw_delta = smoothed_yaw_delta

            # 2. Model B Predicts Ground-Touch Jump Timing
            act_jump, _ = model_jump.predict(obs_3d, deterministic=True)
            jump_sel = float(act_jump[0])
            jump_pressed = bool(jump_sel > 0.0)

            # Buttons: 2 = IN_JUMP (No Auto-Bhop, strictly controlled by Model B)
            buttons = 2 if jump_pressed else 0

            # Send Fused Action to C++:
            # eval_mode = 0 (Full Autonomous Agent Control)
            client.send_action(
                forwardmove=0.0,
                sidemove=sidemove,
                upmove=0.0,
                pitch_delta=0.0,
                yaw_delta=smoothed_yaw_delta,
                buttons=buttons,
                weapon_select=0,
                last_reward=0.0,
                reward_reason="Fused Master Agent"
            )
            client.shm.action.eval_mode = 0

            print(f"[FUSED MASTER] Speed: {current_speed:5.1f} u/s | Ground: {int(on_ground)} | AI Jump: {'[+JUMP]' if jump_pressed else '       '}", end="\r", flush=True)
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[!] Stopping Fused Master runner.")

if __name__ == "__main__":
    main()
