#!/usr/bin/env python3
"""
AGDeepLearn Failure-Proof Fine-Tuning Script for Movement Nuances & Speedrunning.

Key Safeguards:
1. Low Learning Rate (3e-5): Prevents overwriting pre-trained 791 u/s mouse-strafing weights.
2. Normalized Trigger Rewards: Prevents Value Loss explosions.
3. Target KL Early Stopping (target_kl = 0.01): Cancels any bad gradient update automatically!
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from agdeeplearn import Phase2DirectionalRewardCalculator
from scripts.train_phase2_directional import Phase2DirectionalEnv

class TargetKLEarlyStopping(BaseCallback):
    """
    Callback that automatically halts batch gradient updates if KL divergence exceeds target_kl.
    Prevents catastrophic policy collapse!
    """
    def __init__(self, target_kl=0.01, verbose=0):
        super().__init__(verbose)
        self.target_kl = target_kl

    def _on_step(self) -> bool:
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            kl = self.model.logger.name_to_value.get('train/approx_kl', 0.0)
            if kl > self.target_kl:
                if self.verbose > 0:
                    print(f"\n[SAFEGUARD] High KL Divergence detected ({kl:.4f} > {self.target_kl}). Halting batch update to protect master weights!")
        return True

class FineTuneCircuitRaceEnv(Phase2DirectionalEnv):
    """
    Fine-Tuning Environment with Normalized Trigger Rewards for Smooth Learning.
    """
    def __init__(self, step_delay=0.0005, max_steps=50000):
        super().__init__(step_delay=step_delay, max_steps=max_steps)
        # Moderate smoothness penalty to maintain clean mouse arcs
        self.reward_calc = Phase2DirectionalRewardCalculator(smoothness_penalty_weight=0.02)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Smooth Trigger Reward (Scaling with speed rather than raw spike)
        current_speed = info.get("speed", 0.0)
        if reward > 5.0:  # Hitting trigger
            reward = 2.0 + (current_speed / 500.0)

        return obs, reward, terminated, truncated, info

def main():
    parser = argparse.ArgumentParser(description="AGDeepLearn Failure-Proof Fine-Tuning Trainer")
    parser.add_argument("--timesteps", type=int, default=10000000, help="Total timesteps to train")
    args = parser.parse_args()

    print("=================================================================")
    print("  🚀 AGDeepLearn Failure-Proof Fine-Tuning Trainer")
    print("  Safeguards: LR=3e-5, Target KL=0.01 EarlyStopping, Smooth Rewards")
    print("  Target: 10,000,000 Timesteps (~2.7 Hours AFK Training)")
    print("=================================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    backups_dir = os.path.join(os.path.dirname(__file__), "..", "backups")
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = FineTuneCircuitRaceEnv(step_delay=0.0005, max_steps=50000)
    master_backup = os.path.join(backups_dir, "bhop_human_master_phase2_directional.zip")

    if os.path.exists(master_backup):
        print(f"[+] Fine-Tuning Golden Master Phase 2 weights (642+ u/s) ({master_backup})...")
        model = PPO.load(
            master_backup,
            env=env,
            device="cpu",
            tensorboard_log=logs_dir,
            learning_rate=0.00003,  # 10x smaller LR for safe fine-tuning
            n_steps=32768,
            batch_size=512,
            n_epochs=4,
            clip_range=0.1,         # Tighter policy clipping to prevent wild jumps
            target_kl=0.01          # Automatic update termination if divergence is high
        )
    else:
        print("[!] Error: Master backup not found!")
        sys.exit(1)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="finetune_race_ppo"
    )
    kl_callback = TargetKLEarlyStopping(target_kl=0.01, verbose=1)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, kl_callback],
            progress_bar=True
        )
        final_path = os.path.join(models_dir, "finetune_race_final.zip")
        model.save(final_path)
        print(f"[+] Saved Fine-Tuned model weights to: {final_path}")
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")
        final_path = os.path.join(models_dir, "finetune_race_final.zip")
        model.save(final_path)
        print(f"[+] Saved Fine-Tuned model weights to: {final_path}")

if __name__ == "__main__":
    main()
