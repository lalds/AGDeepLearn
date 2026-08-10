# 🚀 OpenAG Autonomous Deep RL Agent — Project State & Technical Spec

> **Document Version:** 1.0  
> **Last Updated:** August 6, 2026  
> **Target System:** Linux (Ubuntu/Debian) | Half-Life GoldSrc Engine (OpenAG Mod)  
> **Repositories:**
> - **C++ Client DLL Engine:** [`/home/angel/OpenAG`](file:///home/angel/OpenAG)
> - **Python RL Framework:** [`/home/angel/AGDeepLearn`](file:///home/angel/AGDeepLearn)

---

## 1. Executive Summary & Project Goals

The objective of this project is to build an autonomous, high-speed AI Agent for **OpenAG** (Open-source Adrenaline Gamer / Half-Life Deathmatch engine) capable of:
1. **Advanced Bunnyhopping (Bhop)** and **Air-Strafing** at high speeds ($500 - 800+\text{ u/s}$).
2. **3D Interactive Waypoint Goal Navigation**: Navigating to arbitrary user-defined 3D targets across complex maps (including obstacle avoidance and corridor traversal).
3. **Map Item Looting**: Identifying and picking up weapons, ammo, health, armor, and longjump modules.
4. **Behavioral Cloning (BC) & PPO RL Integration**: Pre-training policies on human gameplay demonstrations and fine-tuning via Proximal Policy Optimization (PPO).

---

## 2. System Architecture & Data Flow

Communication between the C++ Game Engine Client DLL and the Python RL environment occurs via zero-latency POSIX Shared Memory IPC (`/dev/shm/openag_agent_shm`).

```
 +----------------------------------+            Shared Memory IPC            +------------------------------------+
 |         OpenAG Client DLL        | <=====================================> |        Python AGDeepLearn          |
 |    (/home/angel/OpenAG/cl_dll)   |       /dev/shm/openag_agent_shm        |     (/home/angel/AGDeepLearn)      |
 +----------------------------------+                                         +------------------------------------+
 | - Auto-Strafe Synchronizer Engine |    Observation (132D Struct) ------>    | - OpenAGSHMClient (ctypes wrapper) |
 | - Air-Release Auto-Bhop          |    <------ Action (1D Yaw Delta)        | - OpenAGEnv (Gymnasium Box Space)  |
 | - 8-Raycast Wall Proximity       |                                         | - RewardCalculator (Kinematics)    |
 | - ImGui 3D Overlay & Telemetry   |                                         | - PPO Vectorized Trainer (SB3)     |
 | - Human Demo Recorder (CSV)      |                                         | - Behavioral Cloning (PyTorch BC)  |
 +----------------------------------+                                         +------------------------------------+
```

---

## 3. C++ Game Engine Client DLL Architecture (`OpenAG/cl_dll/`)

### Key Source Files
- [`cl_dll/agent_api.h`](file:///home/angel/OpenAG/cl_dll/agent_api.h): C++ Shared Memory Structures (`AgentObservation`, `AgentAction`, `AgentItemInfo`, `AgentEnemyInfo`).
- [`cl_dll/agent_api.cpp`](file:///home/angel/OpenAG/cl_dll/agent_api.cpp): Perception scanning, Raycasts, Auto-Strafe Synchronizer, ImGui Overlay, Demo Recorder.
- [`cl_dll/input.cpp`](file:///home/angel/OpenAG/cl_dll/input.cpp): `CL_CreateMove` hook calling `AgentAPI_UpdateObservation()` and `AgentAPI_ApplyAction()`.

### Key C++ Features & Engine Modules

#### 1. Auto-Strafe Synchronizer Engine
Eliminates single-side key collapse by automatically mapping physical $A/D$ strafe keys to the agent's camera turn direction (`yaw_delta`):
- **Turning Left (`yaw_delta > +0.05°`)** $\rightarrow$ Automatically sets `sidemove = -400` ($A$ key) & `forwardmove = 0`.
- **Turning Right (`yaw_delta < -0.05°`)** $\rightarrow$ Automatically sets `sidemove = +400` ($D$ key) & `forwardmove = 0`.
- **Straight Motion** $\rightarrow$ Sets `forwardmove = +400` ($W$ key) & `sidemove = 0`.

#### 2. Frame-Perfect Air-Release Auto-Bhop
Prevents GoldSrc jump lockout:
- In mid-air ($\text{abs}(V_z) > 5\text{u/s}$): Automatically releases `IN_JUMP` bit so GoldSrc engine resets button state.
- On ground touch ($\text{abs}(V_z) \le 5\text{u/s}$): Immediately applies `IN_JUMP` bit on frame 1 for instant frame-perfect hop.

#### 3. 3D ImGui Visual Overlay & Telemetry HUD
- **8-Raycast Wall Sensors**: Draws 3D wall rays around player (color-coded green to red based on proximity).
- **Interactive Goal Target Beacon**: Renders glowing 3D Magenta/Purple beacon & line (`[INTERACTIVE GOAL: XXXu]`).
- **Item Beacons**: Renders 3D beacons over map items using absolute `world_pos[3]` (Cyan for target loot, Orange for others).
- **Live RL Reward Telemetry HUD**: Displays real-time step reward (`+0.45` / `-0.40`) and active reason tag (`[Goal Progression]`, `[Smooth S-Curve Switch]`, `[Room Stagnation]`, `[Wall Bump]`).

#### 4. Human Demonstration Recorder
Commands `ag_record_demo` and `ag_stop_demo` capture human observations & inputs to `/home/angel/AGDeepLearn/data/bhop_demo.csv`.

#### 5. In-Game Console Commands
| Command | Description |
| :--- | :--- |
| `cl_agent_api <0\|1>` | Toggle AI Shared Memory API override. |
| `ag_set_goal` | Set 3D Target Goal Marker at player's crosshair position (via `EV_PlayerTrace`). |
| `ag_clear_goal` | Clear current 3D target goal. |
| `ag_record_demo` | Start recording human gameplay demonstration to CSV. |
| `ag_stop_demo` | Stop demonstration recording and save CSV dataset. |
| `cl_agent_debug_items <0\|1>` | Print detected map entity model names in console. |

---

## 4. Shared Memory Specification (`agent_api.h` / `shm_client.py`)

### Shared Memory Buffer Name: `/openag_agent_shm`

### 132D Observation Vector Layout (`AgentObservation`)
- `0..13`: Player Stats (Alive, Origin $X,Y,Z$, Velocity $V_x,V_y,V_z$, 2D Speed, Pitch, Yaw, Health, Armor, Ammo).
- `14..21`: 8-Directional 360° Raycast Wall Proximity Distances ($0 .. 1000\text{u}$).
- `22..26`: Interactive Target Goal Vector (`goal_active`, $\Delta X, \Delta Y, \Delta Z, \text{distance}$).
- `27..51`: 5 Nearest Map Items ($5 \times 5$ features: `active`, $\Delta X, \Delta Y, \Delta Z, \text{distance}$).
- `52..131`: 16 Enemy Player Infos ($16 \times 5$ features).

### 1D Action Vector Layout (`AgentAction`)
- `action[0]`: `yaw_delta` ($-20.0^\circ .. +20.0^\circ$ per frame camera turn rate).
- Auto-strafe engine in C++ handles $A/D/W$ movement keys & autojump automatically.

---

## 5. Python Deep RL Framework (`AGDeepLearn/`)

### Key Python Files
- [`agdeeplearn/shm_client.py`](file:///home/angel/AGDeepLearn/agdeeplearn/shm_client.py): `ctypes` Shared Memory interface wrapper.
- [`agdeeplearn/openag_env.py`](file:///home/angel/AGDeepLearn/agdeeplearn/openag_env.py): Gymnasium Environment (`Box(132,)` obs space, `Box(1,)` action space).
- [`agdeeplearn/rewards.py`](file:///home/angel/AGDeepLearn/agdeeplearn/rewards.py): Kinematic Reward Calculator.
- [`scripts/pretrain_bc.py`](file:///home/angel/AGDeepLearn/scripts/pretrain_bc.py): Behavioral Cloning (PyTorch Supervised Pre-trainer).
- [`scripts/train_vec.py`](file:///home/angel/AGDeepLearn/scripts/train_vec.py): Vectorized PPO Trainer (Stable-Baselines3).

### Reward Function Mechanics (`rewards.py`)
- **Goal Progression Reward**: $+ (\Delta \text{dist} \times 0.02) \times (1.0 + \text{Speed} / 250.0)$ when moving towards target goal.
- **Goal Completion Bonus**: $+5.0$ when reaching target goal ($<60\text{u}$).
- **Air-Strafe Turning Bonus**: $+0.30$ for active turning ($\text{abs}(\text{yaw\_delta}) > 0.5^\circ$).
- **Smooth S-Curve Switch Bonus**: $+0.35$ for alternating turn direction (left $\leftrightarrow$ right curves).
- **Room Stagnation Penalty**: $-0.40$ if 40-step position displacement $<140\text{u}$ (prevents room looping).
- **Off-Course Goal Divergence Penalty**: $-0.35$ if 2D velocity points away from goal ($\vec{V} \cdot \vec{G} < -0.2$).
- **Wall Proximity Penalty**: $-0.50$ when stuck against wall ($<80\text{u}$).

---

## 6. How to Run & Complete Workflow

### 1. Build C++ Client DLL
```bash
cd /home/angel/OpenAG
cmake --build build --target client -- -j$(nproc)
cp build/client.so /mnt/Trash/@home/angeloo123/Games/steamapps/common/Half-Life/ag/cl_dlls/
```

### 2. Record Human Gameplay Demonstration
1. Launch Half-Life / OpenAG mod.
2. In game console, enable API & set target goal:
   ```
   cl_agent_api 1
   ag_set_goal
   ```
3. Start recording demonstration:
   ```
   ag_record_demo
   ```
4. Perform 30–60 seconds of clean bunnyhopping & air-strafing towards the goal.
5. Stop recording:
   ```
   ag_stop_demo
   ```

### 3. Pre-train Policy via Behavioral Cloning (BC)
```bash
cd /home/angel/AGDeepLearn
PYTHONPATH=. ./venv/bin/python scripts/pretrain_bc.py
```
*(Pre-trains policy network for 100 epochs on human dataset and saves weights to `checkpoints/bhop_agent_final.zip`)*

### 4. Run Vectorized PPO RL Training
```bash
cd /home/angel/AGDeepLearn
./venv/bin/python scripts/train_vec.py --num-envs 2 --timesteps 10000000
```

---

## 7. Status & Next Steps for AI Assistant

- [x] POSIX Shared Memory C++/Python IPC bridge.
- [x] 8-Directional Raycast Wall Proximity Sensors.
- [x] 3D Interactive Target Goal System & ImGui Visual Overlay.
- [x] Auto-Strafe Synchronizer Engine in C++ DLL.
- [x] Frame-Perfect Air-Release Auto-Bhop.
- [x] Real-Time Reward & Telemetry HUD.
- [x] Human Demonstration CSV Recorder.
- [x] Behavioral Cloning (BC) Pre-trainer (`scripts/pretrain_bc.py`).
- [ ] Record high-quality 60s human demo dataset (`data/bhop_demo.csv`).
- [ ] Pre-train model using `pretrain_bc.py`.
- [ ] Run 10M-step PPO training via `train_vec.py`.
