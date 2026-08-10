#!/usr/bin/env python3
"""
AGDeepLearn Interactive Co-Op Evaluation Runner:
- AI controls JUMP timing (Manual Bhop from pure_jump_timing_final.zip).
- YOU control mouse steering and A/D strafes in game!
"""

import os
import sys
import time
import math
import numpy as np

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from agdeeplearn import OpenAGSHMClient

def main():
    print("=================================================================")
    print("  🎮 AGDeepLearn Co-Op Interactive Test Runner")
    print("  AI Role: Manual Jump Timing (+jump / -jump)")
    print("  YOUR Role: Mouse Steering & A/D Strafes in Half-Life!")
    print("=================================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    model_path = os.path.join(models_dir, "pure_jump_timing_final.zip")

    if not os.path.exists(model_path):
        print(f"[!] Error: Trained 1D jump model '{model_path}' not found!")
        sys.exit(1)

    print(f"[+] Loading 1D Jump Timing Model: {model_path}")
    model = PPO.load(model_path, device="cpu")
    print("[+] Model loaded successfully!")

    client = OpenAGSHMClient()
    try:
        client.connect()
        print("[+] Connected to OpenAG Shared Memory!")
    except FileNotFoundError:
        print("[!] Error: OpenAG shared memory not found. Launch Half-Life with cl_agent_api 1.")
        sys.exit(1)

    print("\n[+] Co-Op mode ACTIVE! Play in Half-Life now!")
    print("  AI will press JUMP at exact ground touch frames.")
    print("  YOU steer with mouse and A/D keys! (Press Ctrl+C to stop)")

    last_seq = 0
    try:
        while True:
            obs = client.get_observation()
            if not obs or obs.sequence == last_seq or obs.player_alive == 0:
                time.sleep(0.001)
                continue

            last_seq = obs.sequence

            vx, vy, vz = obs.player_velocity[0], obs.player_velocity[1], obs.player_velocity[2]
            current_speed = math.sqrt(vx * vx + vy * vy)
            on_ground = float(obs.on_ground)

            # 3D Observation: [speed/500, vz/500, on_ground]
            obs_3d = np.array([
                float(current_speed / 500.0),
                float(vz / 500.0),
                float(on_ground)
            ], dtype=np.float32)

            # Predict AI Jump action
            action, _ = model.predict(obs_3d, deterministic=True)
            jump_sel = float(action[0])
            jump_pressed = bool(jump_sel > 0.0)

            # Buttons: 2 = IN_JUMP
            buttons = 2 if jump_pressed else 0

            # Send action to C++ with eval_mode = 2 (Co-Op Mode):
            # C++ gives YOU 100% full control of W/A/S/D and mouse, while AI ONLY presses JUMP!
            client.send_action(
                forwardmove=0.0,
                sidemove=0.0,
                upmove=0.0,
                pitch_delta=0.0,
                yaw_delta=0.0,
                buttons=buttons,
                weapon_select=0,
                last_reward=0.0,
                reward_reason="Co-Op Jump AI"
            )
            client.shm.action.eval_mode = 2  # 2 = Co-Op Mode: Human controls movement, AI controls Jump!

            print(f"[CO-OP AI] Speed: {current_speed:5.1f} u/s | Ground: {int(on_ground)} | AI Jump: {'[+JUMP]' if jump_pressed else '       '}", end="\r", flush=True)
            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\n[!] Stopping Co-Op runner.")

if __name__ == "__main__":
    main()
