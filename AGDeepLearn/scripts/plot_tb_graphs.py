#!/usr/bin/env python3
import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def main():
    log_dirs = glob.glob("/home/angel/AGDeepLearn/logs/PPO_*")
    if not log_dirs:
        print("No log dirs found!")
        return

    latest_dir = sorted(log_dirs, key=os.path.getmtime)[-1]
    event_files = glob.glob(os.path.join(latest_dir, "events.out.tfevents*"))
    if not event_files:
        print("No tfevents file found in", latest_dir)
        return

    event_file = event_files[0]
    print(f"[+] Loading TensorBoard metrics from: {event_file}")

    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    print(f"[+] Found scalar tags: {tags}")

    metrics = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        metrics[tag] = (steps, values)

    # Style dark theme plot
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle('🚀 AGDeepLearn Phase 1 PPO Training Metrics (TensorBoard)', fontsize=16, fontweight='bold', color='#00ffcc')

    # Panel 1: Explained Variance
    ax1 = axes[0, 0]
    if 'train/explained_variance' in metrics:
        s, v = metrics['train/explained_variance']
        ax1.plot(s, v, color='#00ffcc', linewidth=2, label='Explained Variance')
        ax1.set_title('Value Net Explained Variance (Goal: ~1.0)', color='#00ffcc')
        ax1.set_xlabel('Timesteps')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend()

    # Panel 2: Total Loss & Value Loss
    ax2 = axes[0, 1]
    if 'train/loss' in metrics:
        s, v = metrics['train/loss']
        ax2.plot(s, v, color='#ff0055', linewidth=2, label='Total Loss')
    if 'train/value_loss' in metrics:
        s, v = metrics['train/value_loss']
        ax2.plot(s, v, color='#ffaa00', linewidth=1.5, linestyle='--', label='Value Loss')
    ax2.set_title('PPO Loss Convergence', color='#ff0055')
    ax2.set_xlabel('Timesteps')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()

    # Panel 3: Policy Gradient Loss
    ax3 = axes[1, 0]
    if 'train/policy_gradient_loss' in metrics:
        s, v = metrics['train/policy_gradient_loss']
        ax3.plot(s, v, color='#00aaff', linewidth=2, label='Policy Gradient Loss')
        ax3.set_title('Policy Gradient Loss', color='#00aaff')
        ax3.set_xlabel('Timesteps')
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.legend()

    # Panel 4: Entropy Loss / Policy Std
    ax4 = axes[1, 1]
    if 'train/entropy_loss' in metrics:
        s, v = metrics['train/entropy_loss']
        ax4.plot(s, v, color='#cc00ff', linewidth=2, label='Entropy Loss')
        ax4.set_title('Entropy (Exploration Rate)', color='#cc00ff')
        ax4.set_xlabel('Timesteps')
        ax4.grid(True, linestyle='--', alpha=0.3)
        ax4.legend()

    plt.tight_layout()
    output_path = "/home/angel/AGDeepLearn/logs/phase1_tensorboard_plot.png"
    plt.savefig(output_path)
    print(f"[+] Successfully saved TensorBoard plot image to: {output_path}")

if __name__ == "__main__":
    main()
