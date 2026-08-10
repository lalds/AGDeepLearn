#!/usr/bin/env python3
"""
AGDeepLearn Training Analytics Plotter & Reporter (Russian Language).

Parses TensorBoard event logs across training runs and generates high-res dark-mode plots
with detailed analytical descriptions in Russian.
"""

import os
import sys
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def extract_tb_logs(run_dir):
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        return {}

    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()

    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = (steps, values)
    return data

def main():
    print("=================================================================")
    print("  📊 AGDeepLearn Training Analytics & Visualization Generator")
    print("=================================================================")

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Select representative runs:
    # PPO_18 (Phase 1 Pure Sync Mastery)
    # PPO_26 (Phase 2 Master Directional 500+ u/s Human Bhop)
    # PPO_27 (Phase 3 Full Controls Experiment)
    runs_to_plot = {
        "Phase 1: Чистая Физика (PPO_18)": os.path.join(logs_dir, "PPO_18"),
        "Phase 2: Направленный Бхоп 500+ (PPO_26)": os.path.join(logs_dir, "PPO_26"),
        "Phase 3: Эксперимент Кнопок (PPO_27)": os.path.join(logs_dir, "PPO_27"),
    }

    parsed_runs = {}
    for label, run_path in runs_to_plot.items():
        if os.path.exists(run_path):
            parsed_runs[label] = extract_tb_logs(run_path)

    # Dark-mode styling for high-res plots
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    fig.suptitle("AGDeepLearn: Аналитика Графиков Обучения Нейросети (PPO)", fontsize=18, color="#00e5ff", fontweight="bold")

    colors = ["#00e5ff", "#00ff75", "#ff0055"]

    # 1. Explained Variance (Сходимость сети Critic)
    ax1 = axes[0, 0]
    ax1.set_title("1. Explained Variance (Сходимость сети Критика)", fontsize=13, color="#00e5ff")
    for idx, (label, data) in enumerate(parsed_runs.items()):
        if "train/explained_variance" in data:
            steps, vals = data["train/explained_variance"]
            ax1.plot(steps, vals, label=label, color=colors[idx % len(colors)], linewidth=2)
    ax1.axhline(0.95, color="#ffea00", linestyle="--", alpha=0.7, label="Цель (0.95+)")
    ax1.set_xlabel("Тики обучения (Timesteps)")
    ax1.set_ylabel("Explained Variance (0.0 - 1.0)")
    ax1.set_ylim(-0.2, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(loc="lower right")

    # 2. Value Loss (Потери сети оценок ценности)
    ax2 = axes[0, 1]
    ax2.set_title("2. Value Loss (Ошибка прогнозирования наград)", fontsize=13, color="#00e5ff")
    for idx, (label, data) in enumerate(parsed_runs.items()):
        if "train/value_loss" in data:
            steps, vals = data["train/value_loss"]
            ax2.plot(steps, vals, label=label, color=colors[idx % len(colors)], linewidth=2)
    ax2.set_xlabel("Тики обучения (Timesteps)")
    ax2.set_ylabel("Value Loss")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="upper right")

    # 3. Entropy Loss (Стабильность и исследование)
    ax3 = axes[1, 0]
    ax3.set_title("3. Entropy Loss (Разброс исследований нейросети)", fontsize=13, color="#00e5ff")
    for idx, (label, data) in enumerate(parsed_runs.items()):
        if "train/entropy_loss" in data:
            steps, vals = data["train/entropy_loss"]
            ax3.plot(steps, vals, label=label, color=colors[idx % len(colors)], linewidth=2)
    ax3.set_xlabel("Тики обучения (Timesteps)")
    ax3.set_ylabel("Entropy Loss")
    ax3.grid(True, linestyle=":", alpha=0.4)
    ax3.legend(loc="upper right")

    # 4. Policy Gradient Loss (Градиентные шаги Актера)
    ax4 = axes[1, 1]
    ax4.set_title("4. Policy Gradient Loss (Шаги обновления Актера)", fontsize=13, color="#00e5ff")
    for idx, (label, data) in enumerate(parsed_runs.items()):
        if "train/policy_gradient_loss" in data:
            steps, vals = data["train/policy_gradient_loss"]
            ax4.plot(steps, vals, label=label, color=colors[idx % len(colors)], linewidth=2)
    ax4.set_xlabel("Тики обучения (Timesteps)")
    ax4.set_ylabel("Policy Loss")
    ax4.grid(True, linestyle=":", alpha=0.4)
    ax4.legend(loc="lower right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(logs_dir, "training_metrics_convergence.png")
    plt.savefig(output_path)
    plt.close()

    print(f"[+] High-res analytics plot saved to: {output_path}")

if __name__ == "__main__":
    main()
