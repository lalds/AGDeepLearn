#!/usr/bin/env python3
"""
AGDeepLearn Human Demo Recorder.

Usage:
  1. Launch OpenAG Half-Life with cl_agent_api 1.
  2. Run: python3 scripts/record_human_demo.py
  3. Play in game and perform your ideal bhop / air-strafing!
  4. Press Ctrl+C in terminal when finished.
"""

import os
import sys
import time
import math
import argparse
import numpy as np

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agdeeplearn import OpenAGSHMClient

def main():
    parser = argparse.ArgumentParser(description="AGDeepLearn Human Demo Recorder")
    parser.add_argument("--output", type=str, default="datasets/human_bhop_demo.npz", help="Output demo dataset path")
    args = parser.parse_args()

    print("=================================================================")
    print("  🎥 AGDeepLearn Human Demo Recorder")
    print("  Instruction: Play in game! Perform your ideal bhop / strafes.")
    print("  Press Ctrl+C when finished to save dataset.")
    print("=================================================================")

    client = OpenAGSHMClient()
    try:
        client.connect()
        print("[+] Connected to OpenAG Shared Memory!")
    except FileNotFoundError:
        print("[!] Error: OpenAG shared memory not found. Make sure Half-Life is running with cl_agent_api 1.")
        sys.exit(1)

    dataset_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(dataset_dir, exist_ok=True)

    recorded_obs = []
    recorded_actions = []

    last_seq = 0
    last_yaw = 0.0

    print("[+] Recording active! Play now in game... (Press Ctrl+C to stop)")

    try:
        while True:
            obs = client.get_observation()
            if not obs or obs.sequence == last_seq or obs.player_alive == 0:
                time.sleep(0.002)
                continue

            last_seq = obs.sequence

            # Calculate human yaw delta (mouse turn)
            current_yaw = obs.viewangles[1]
            yaw_delta = current_yaw - last_yaw
            # Normalize yaw delta to [-180, 180]
            yaw_delta = (yaw_delta + 180.0) % 360.0 - 180.0
            last_yaw = current_yaw

            # 18D Perception Vector matching Phase2DirectionalEnv:
            px, py = obs.player_origin[0], obs.player_origin[1]
            tx, ty = obs.goal_origin[0], obs.goal_origin[1]
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

            obs_18d = [
                float(current_speed / 500.0), float(vz / 500.0), float(on_ground),
                *lidar,
                float(vx / 500.0), float(vy / 500.0), float(vz / 500.0),
                float(obs.viewangles[0] / 90.0),
                float(rel_angle / math.pi),
                float(dist_to_target / 1000.0),
                float(v_proj / 500.0)
            ]

            # Infer Human Action:
            # action[0] = yaw_delta (clipped to [-15, 15])
            # action[1] = sidemove (-1.0 if turning left/A, +1.0 if turning right/D, 0.0 if neutral)
            action_yaw = float(np.clip(yaw_delta, -15.0, 15.0))
            if abs(yaw_delta) < 0.05:
                action_side = 0.0 # Neutral when camera isn't turning!
            elif yaw_delta > 0.0:
                action_side = -1.0 # Turning LEFT -> Press A
            else:
                action_side = 1.0 # Turning RIGHT -> Press D

            human_action = [action_yaw, action_side]

            # Tell C++ C++ client to remain in eval mode (don't override player controls)
            client.send_action(
                forwardmove=0.0,
                sidemove=0.0,
                upmove=0.0,
                pitch_delta=0.0,
                yaw_delta=0.0,
                buttons=0,
                last_reward=0.0,
                reward_reason="Recording Demo"
            )
            client.shm.action.eval_mode = 1  # 1 = Telemetry only, let human control character!

            recorded_obs.append(obs_18d)
            recorded_actions.append(human_action)

            if len(recorded_obs) % 200 == 0:
                print(f"[RECORDER] Recorded {len(recorded_obs)} frames ({len(recorded_obs)/100:.1f}s) | Speed: {current_speed:.1f} u/s", end="\r", flush=True)

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[+] Recording stopped by user.")
    finally:
        if client.shm:
            client.shm.action.eval_mode = 0  # Restore normal mode

    obs_arr = np.array(recorded_obs, dtype=np.float32)
    act_arr = np.array(recorded_actions, dtype=np.float32)

    if len(obs_arr) > 0:
        np.savez_compressed(args.output, observations=obs_arr, actions=act_arr)
        print(f"\n[+] Saved {len(obs_arr)} human demo frames to: {args.output}")
        print(f"[+] Total Recording Duration: {len(obs_arr)/100:.1f} seconds")
    else:
        print("\n[!] No frames recorded.")

if __name__ == "__main__":
    main()
