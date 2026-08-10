#!/usr/bin/env python3
"""
AGDeepLearn Human Demo Mix-in Trainer (Behavioral Cloning + PPO Fine-Tuning).

Goal:
- Mix human demonstration dataset into pre-trained PPO policy.
- Eliminates stumbling, hesitations, and awkward angles while preserving 500+ u/s bhop speed!
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from scripts.train_phase2_directional import Phase2DirectionalEnv

def main():
    parser = argparse.ArgumentParser(description="AGDeepLearn Human Demo Mix-in Trainer")
    parser.add_argument("--demo", type=str, default="datasets/human_bhop_demo.npz", help="Path to human demo dataset")
    parser.add_argument("--bc-epochs", type=int, default=35, help="Behavioral cloning pre-training epochs")
    parser.add_argument("--rl-steps", type=int, default=50000, help="PPO fine-tuning timesteps")
    args = parser.parse_args()

    print("=================================================================")
    print("  🚀 AGDeepLearn Human Demo Mix-in & Fine-Tuning Trainer")
    print("  Goal: Inject human fluid strafing into high-speed PPO model!")
    print("=================================================================")

    if not os.path.exists(args.demo):
        print(f"[!] Error: Human demo dataset '{args.demo}' not found!")
        print("  Please record a demo first using: python3 scripts/record_human_demo.py")
        sys.exit(1)

    print(f"[+] Loading human demonstration dataset: {args.demo}")
    data = np.load(args.demo)
    demo_obs = torch.tensor(data["observations"], dtype=torch.float32)
    demo_act = torch.tensor(data["actions"], dtype=torch.float32)

    print(f"[+] Dataset loaded: {len(demo_obs)} frames ({len(demo_obs)/100:.1f} seconds of expert human play).")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    backups_dir = os.path.join(os.path.dirname(__file__), "..", "backups")
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    master_backup = os.path.join(backups_dir, "bhop_human_master_phase2_directional.zip")
    phase2_checkpoint = os.path.join(models_dir, "phase2_directional_final.zip")

    env = Phase2DirectionalEnv(step_delay=0.0005, max_steps=25000)

    if os.path.exists(master_backup):
        print(f"[+] Loading master model weights from: {master_backup}")
        model = PPO.load(master_backup, env=env, device="cpu", tensorboard_log=logs_dir)
    elif os.path.exists(phase2_checkpoint):
        print(f"[+] Loading Phase 2 model weights from: {phase2_checkpoint}")
        model = PPO.load(phase2_checkpoint, env=env, device="cpu", tensorboard_log=logs_dir)
    else:
        print("[!] Error: No pre-trained Phase 2 model found to fine-tune!")
        sys.exit(1)

    # Step 1: Behavioral Cloning Fine-Tuning on Human Trajectories
    print(f"\n[+] Step 1: Behavioral Cloning Fine-Tuning ({args.bc_epochs} epochs)...")
    dataset = TensorDataset(demo_obs, demo_act)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    policy = model.policy
    # Optimize policy_net features and action output layer
    actor_params = list(policy.action_net.parameters()) + list(policy.mlp_extractor.policy_net.parameters())
    optimizer = optim.Adam(actor_params, lr=0.0001)
    mse_loss = nn.MSELoss()

    policy.train()
    for epoch in range(1, args.bc_epochs + 1):
        total_loss = 0.0
        for batch_obs, batch_act in loader:
            optimizer.zero_grad()
            distribution = policy.get_distribution(batch_obs)
            pred_act = distribution.mode()
            loss = mse_loss(pred_act, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == args.bc_epochs:
            print(f"  [BC Epoch {epoch:2d}/{args.bc_epochs}] Imitation Loss: {avg_loss:.6f}")

    print("[+] Behavioral Cloning pre-training complete! Human strafe patterns injected into neural net.")

    # Step 2: PPO RL Fine-Tuning with Human-Injected Weights
    print(f"\n[+] Step 2: PPO RL Fine-Tuning for {args.rl_steps} timesteps...")
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="phase2_human_refined"
    )

    try:
        model.learn(
            total_timesteps=args.rl_steps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        print("\n[+] Human Demo Fine-Tuning completed successfully!")
    except KeyboardInterrupt:
        print("\n[!] Fine-tuning interrupted by user.")

    final_path = os.path.join(models_dir, "phase2_directional_human_refined")
    model.save(final_path)
    print(f"[+] Saved Human-Refined model weights to: {final_path}.zip")

    env.close()

if __name__ == "__main__":
    main()
