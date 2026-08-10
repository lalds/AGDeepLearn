import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from stable_baselines3 import PPO
from agdeeplearn.openag_env import OpenAGEnv

def main():
    csv_path = "/home/angel/AGDeepLearn/data/bhop_demo.csv"
    if not os.path.exists(csv_path):
        print(f"[!] ERROR: Human demo dataset '{csv_path}' not found!")
        print("[!] In game console, run 'ag_record_demo', play and Bhop for 1 minute, then run 'ag_stop_demo'!")
        return

    print("==========================================================")
    print("  AGDeepLearn: Human Behavioral Cloning Pre-trainer (BC)")
    print("==========================================================")
    print(f"[+] Loading human demonstration dataset from: {csv_path}")

    # Load CSV data
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    print(f"[+] Loaded {len(data)} human demonstration frames!")
    if len(data) < 50:
        print("[!] Warning: Dataset is very small. Play for at least 30-60 seconds for best results!")

    actions = data[:, 0:3]       # fwd_sel (-1..1), side_sel (-1..1), yaw_delta (-20..20)
    observations = data[:, 3:]   # 132D observation vector

    # Convert to PyTorch Tensors
    obs_tensor = torch.tensor(observations, dtype=torch.float32)
    act_tensor = torch.tensor(actions, dtype=torch.float32)

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize Dummy OpenAGEnv for space alignment
    print("[+] Creating PPO Policy Network for Pre-training...")
    dummy_env = OpenAGEnv(instance_id=99)
    model = PPO(
        "MlpPolicy",
        dummy_env,
        device="cpu",
        verbose=0,
        learning_rate=0.001
    )

    policy = model.policy
    optimizer = optim.Adam(policy.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    print("[+] Pre-training Policy Network on Human Demonstrations (100 Epochs)...")
    policy.train()
    for epoch in range(1, 101):
        total_loss = 0.0
        for obs_batch, act_batch in loader:
            optimizer.zero_grad()

            # Predict mean actions from policy MLP network
            features = policy.extract_features(obs_batch)
            latent_pi, _ = policy.mlp_extractor(features)
            mean_actions = policy.action_net(latent_pi)

            loss = criterion(mean_actions, act_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(obs_batch)

        avg_loss = total_loss / len(dataset)
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/100 | Behavioral Cloning Loss: {avg_loss:.4f}")

    # Save pre-trained policy to checkpoints/bhop_agent_final.zip
    models_dir = "/home/angel/AGDeepLearn/checkpoints"
    os.makedirs(models_dir, exist_ok=True)
    final_path = os.path.join(models_dir, "bhop_agent_final")
    model.save(final_path)

    print("----------------------------------------------------------")
    print(f"[+] SUCCESS! Pre-trained model saved to: {final_path}.zip")
    print("[+] You can now run 'python scripts/train_vec.py' to continue RL training!")
    print("==========================================================")

if __name__ == "__main__":
    main()
