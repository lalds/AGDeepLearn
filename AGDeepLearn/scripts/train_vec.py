#!/usr/bin/env python3
import os
import sys
import argparse

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from agdeeplearn import OpenAGEnv

def make_env(instance_id, step_delay=0.0005, max_steps=500):
    def _init():
        return OpenAGEnv(instance_id=instance_id, step_delay=step_delay, max_steps=max_steps)
    return _init

def main():
    parser = argparse.ArgumentParser(description="AGDeepLearn Vectorized Multi-Environment PPO Trainer")
    parser.add_argument("--num-envs", type=int, default=2, help="Number of parallel environment instances")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps to train")
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv (multiprocessing)")
    args = parser.parse_args()

    print("==========================================================")
    print(f"  AGDeepLearn: Vectorized Multi-Env Trainer (Envs={args.num_envs})")
    print("==========================================================")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env_fns = [make_env(i, step_delay=0.0005) for i in range(args.num_envs)]
    if args.subproc and args.num_envs > 1:
        print(f"[+] Creating SubprocVecEnv with {args.num_envs} subprocesses...")
        vec_env = SubprocVecEnv(env_fns)
    else:
        print(f"[+] Creating DummyVecEnv with {args.num_envs} instances...")
        vec_env = DummyVecEnv(env_fns)

    final_checkpoint = os.path.join(models_dir, "bhop_agent_final.zip")
    if os.path.exists(final_checkpoint):
        print(f"[+] Found existing model checkpoint ({final_checkpoint}). CONTINUING training from previous weights!")
        model = PPO.load(final_checkpoint, env=vec_env, device="cpu", tensorboard_log=logs_dir)
    else:
        print("[+] Initializing NEW PPO Agent Policy on CPU (MlpPolicy)...")
        model = PPO(
            "MlpPolicy",
            vec_env,
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
        name_prefix="vec_bhop_ppo"
    )

    print(f"[+] Starting Parallel RL Training for {args.timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        print("\n[+] Training completed successfully!")
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    final_path = os.path.join(models_dir, "bhop_agent_final")
    model.save(final_path)
    print(f"[+] Saved model checkpoint to: {final_path}.zip")

    vec_env.close()

if __name__ == "__main__":
    main()
