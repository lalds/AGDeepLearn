import os
import mmap
import ctypes

SHM_PATH = "/dev/shm/openag_agent_shm"

class AgentEnemyInfo(ctypes.Structure):
    _layout_ = 'ms'
    _pack_ = 1
    _fields_ = [
        ("active", ctypes.c_int32),
        ("index", ctypes.c_int32),
        ("origin", ctypes.c_float * 3),
        ("relative_pos", ctypes.c_float * 3),
        ("distance", ctypes.c_float),
    ]

class AgentItemInfo(ctypes.Structure):
    _layout_ = 'ms'
    _pack_ = 1
    _fields_ = [
        ("active", ctypes.c_int32),
        ("relative_pos", ctypes.c_float * 3),
        ("distance", ctypes.c_float),
        ("world_pos", ctypes.c_float * 3),
    ]

class AgentObservation(ctypes.Structure):
    _layout_ = 'ms'
    _pack_ = 1
    _fields_ = [
        ("sequence", ctypes.c_uint32),
        ("player_alive", ctypes.c_int32),
        ("player_origin", ctypes.c_float * 3),
        ("player_velocity", ctypes.c_float * 3),
        ("viewangles", ctypes.c_float * 3),
        ("health", ctypes.c_int32),
        ("armor", ctypes.c_int32),
        ("active_weapon_id", ctypes.c_int32),
        ("clip_ammo", ctypes.c_int32),
        ("reserve_ammo", ctypes.c_int32),
        ("wall_distances", ctypes.c_float * 16),
        ("goal_active", ctypes.c_int32),
        ("goal_origin", ctypes.c_float * 3),
        ("num_items", ctypes.c_int32),
        ("items", AgentItemInfo * 16),
        ("num_enemies", ctypes.c_int32),
        ("enemies", AgentEnemyInfo * 16),
        ("strafe_efficiency", ctypes.c_float),
        ("strafe_state", ctypes.c_int32),
        ("on_ground", ctypes.c_int32),
    ]

class AgentAction(ctypes.Structure):
    _layout_ = 'ms'
    _pack_ = 1
    _fields_ = [
        ("sequence_ack", ctypes.c_uint32),
        ("forwardmove", ctypes.c_float),
        ("sidemove", ctypes.c_float),
        ("upmove", ctypes.c_float),
        ("viewangles_delta", ctypes.c_float * 3),
        ("buttons", ctypes.c_uint32),
        ("weapon_select", ctypes.c_int32),
        ("last_reward", ctypes.c_float),
        ("reward_reason", ctypes.c_char * 64),
        ("reward_speed", ctypes.c_float),
        ("reward_scurve", ctypes.c_float),
        ("reward_bhop", ctypes.c_float),
        ("reward_streak", ctypes.c_float),
        ("reward_asym", ctypes.c_float),
        ("reward_wall", ctypes.c_float),
        ("reward_goal", ctypes.c_float),
        ("eval_mode", ctypes.c_int32),
        ("loss_policy", ctypes.c_float),
        ("loss_value", ctypes.c_float),
        ("entropy", ctypes.c_float),
        ("learning_rate", ctypes.c_float),
        ("system_fps", ctypes.c_float),
        ("reset_requested", ctypes.c_int32),
        ("reset_origin", ctypes.c_float * 3),
        ("reset_velocity", ctypes.c_float * 3),
        ("reset_angles", ctypes.c_float * 3),
    ]

class AgentSharedData(ctypes.Structure):
    _layout_ = 'ms'
    _pack_ = 1
    _fields_ = [
        ("observation", AgentObservation),
        ("action", AgentAction),
        ("api_enabled", ctypes.c_int32),
        ("client_connected", ctypes.c_int32),
    ]

class OpenAGSHMClient:
    def __init__(self, instance_id=0, shm_path=None):
        if shm_path is not None:
            self.shm_path = shm_path
        elif instance_id > 0:
            self.shm_path = f"/dev/shm/openag_agent_shm_{instance_id}"
        else:
            self.shm_path = SHM_PATH

        self.fd = None
        self.buf = None
        self.shm = None

    def connect(self):
        target_path = self.shm_path
        if not os.path.exists(target_path) and os.path.exists(SHM_PATH):
            target_path = SHM_PATH

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Shared Memory '{target_path}' not found. Make sure OpenAG is running with cl_agent_api 1.")

        self.fd = os.open(target_path, os.O_RDWR)
        self.buf = mmap.mmap(self.fd, ctypes.sizeof(AgentSharedData), mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.shm = AgentSharedData.from_buffer(self.buf)
        return True

    def get_observation(self):
        if not self.shm:
            return None
        return self.shm.observation

    def send_action(self, forwardmove=0.0, sidemove=0.0, upmove=0.0, pitch_delta=0.0, yaw_delta=0.0, buttons=0, weapon_select=0, last_reward=0.0, reward_reason="", eval_mode=0):
        if not self.shm:
            return
        act = self.shm.action
        act.sequence_ack = self.shm.observation.sequence
        act.eval_mode = int(eval_mode)  # 0 = Full AI Control Override, 1 = Telemetry Only, 2 = Co-Op Jump
        act.forwardmove = float(forwardmove)
        act.sidemove = float(sidemove)
        act.upmove = float(upmove)
        act.viewangles_delta[0] = float(pitch_delta)
        act.viewangles_delta[1] = float(yaw_delta)
        act.buttons = int(buttons)
        act.weapon_select = int(weapon_select)
        act.last_reward = float(last_reward)

        reason_bytes = reward_reason.encode('utf-8')[:63]
        act.reward_reason = reason_bytes

    def set_goal(self, origin=(0.0, 0.0, 0.0), active=True):
        if not self.shm:
            return
        self.shm.observation.goal_active = 1 if active else 0
        self.shm.observation.goal_origin[0] = float(origin[0])
        self.shm.observation.goal_origin[1] = float(origin[1])
        self.shm.observation.goal_origin[2] = float(origin[2])

    def request_reset(self, origin=(0.0, 0.0, 36.0), velocity=(300.0, 0.0, 0.0), angles=(0.0, 0.0, 0.0)):
        if not self.shm:
            return
        act = self.shm.action
        act.reset_requested = 1
        act.reset_origin[0], act.reset_origin[1], act.reset_origin[2] = float(origin[0]), float(origin[1]), float(origin[2])
        act.reset_velocity[0], act.reset_velocity[1], act.reset_velocity[2] = float(velocity[0]), float(velocity[1]), float(velocity[2])
        act.reset_angles[0], act.reset_angles[1], act.reset_angles[2] = float(angles[0]), float(angles[1]), float(angles[2])

    def close(self):
        self.shm = None
        if self.buf:
            try:
                self.buf.close()
            except BufferError:
                pass
            self.buf = None
        if self.fd:
            os.close(self.fd)
            self.fd = None
