import time
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .shm_client import OpenAGSHMClient
from .rewards import RewardCalculator

class OpenAGEnv(gym.Env):
    """
    Gymnasium Environment for OpenAG Movement / Bhop Training.
    Communicates with OpenAG client DLL via POSIX Shared Memory.
    """
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, step_delay=0.02, max_steps=500):
        super().__init__()
        self.client = OpenAGSHMClient()
        self.step_delay = step_delay
        self.max_steps = max_steps
        self.current_step = 0

        self.reward_calculator = RewardCalculator()

        # Observation space: 14 player metrics + 16 * 5 enemy metrics = 94 floats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(94,), dtype=np.float32
        )

        # Movement Action Space (5D continuous):
        # [0]: forwardmove (-400..400)
        # [1]: sidemove (-400..400)
        # [2]: yaw_delta (-15..15 deg)
        # [3]: pitch_delta (-3..3 deg)
        # [4]: jump_flag (>0 = jump)
        self.action_space = spaces.Box(
            low=np.array([-400.0, -400.0, -15.0, -3.0, -1.0], dtype=np.float32),
            high=np.array([400.0, 400.0, 15.0, 3.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def _extract_observation_vector(self, obs):
        if not obs:
            return np.zeros(94, dtype=np.float32)

        vx, vy, vz = obs.player_velocity[0], obs.player_velocity[1], obs.player_velocity[2]
        speed = math.sqrt(vx * vx + vy * vy)

        player_vec = [
            float(obs.player_alive),
            float(obs.player_origin[0]),
            float(obs.player_origin[1]),
            float(obs.player_origin[2]),
            float(vx),
            float(vy),
            float(vz),
            float(speed),
            float(obs.viewangles[0]),
            float(obs.viewangles[1]),
            float(obs.health),
            float(obs.armor),
            float(obs.clip_ammo),
            float(obs.reserve_ammo)
        ]

        enemies_vec = []
        for i in range(16):
            if i < obs.num_enemies and obs.enemies[i].active:
                e = obs.enemies[i]
                enemies_vec.extend([
                    1.0,
                    float(e.relative_pos[0]),
                    float(e.relative_pos[1]),
                    float(e.relative_pos[2]),
                    float(e.distance)
                ])
            else:
                enemies_vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(player_vec + enemies_vec, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if not self.client.shm:
            self.client.connect()

        # Send instant reset to player (teleport + velocity injection)
        self.client.request_reset(origin=(0.0, 0.0, 36.0), velocity=(300.0, 0.0, 0.0))
        time.sleep(0.01)

        obs = self.client.get_observation()
        self.reward_calculator.reset(obs)

        obs_vector = self._extract_observation_vector(obs)
        info = {}

        return obs_vector, info

    def step(self, action):
        self.current_step += 1

        forwardmove = float(action[0])
        sidemove = float(action[1])
        yaw_delta = float(action[2])
        pitch_delta = float(action[3])
        jump_val = float(action[4])

        # Convert jump_val to bitmask (2 = IN_JUMP)
        buttons = 2 if jump_val > 0.0 else 0

        # Send action to Shared Memory
        self.client.send_action(
            forwardmove=forwardmove,
            sidemove=sidemove,
            upmove=0.0,
            pitch_delta=pitch_delta,
            yaw_delta=yaw_delta,
            buttons=buttons,
            weapon_select=0
        )

        time.sleep(self.step_delay)

        obs = self.client.get_observation()
        obs_vector = self._extract_observation_vector(obs)

        reward = self.reward_calculator.compute_reward(obs)

        terminated = bool(obs.player_alive == 0) if obs else False
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "step": self.current_step,
            "speed": float(obs_vector[7]) if obs else 0.0,
            "health": int(obs_vector[10]) if obs else 0
        }

        return obs_vector, reward, terminated, truncated, info

    def close(self):
        self.client.close()
