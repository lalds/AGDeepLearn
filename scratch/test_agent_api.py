#!/usr/bin/env python3
import time
import ctypes
import os
import math

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
        ("num_enemies", ctypes.c_int32),
        ("enemies", AgentEnemyInfo * 16),
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

def main():
    print("==========================================")
    print("      OpenAG AI Agent Test Client")
    print("==========================================")
    print(f"Connecting to Shared Memory: {SHM_PATH}...")

    if not os.path.exists(SHM_PATH):
        print(f"[!] Error: Shared memory file '{SHM_PATH}' not found.")
        print("Please start OpenAG and enable 'cl_agent_api 1' in console or ImGui settings.")
        return

    fd = os.open(SHM_PATH, os.O_RDWR)
    try:
        import mmap
        buf = mmap.mmap(fd, ctypes.sizeof(AgentSharedData), mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        shm = AgentSharedData.from_buffer(buf)
        print("[+] Successfully attached to Shared Memory!")

        step = 0
        last_seq = 0

        print("\nStarting Control & Observation Test Loop (Ctrl+C to stop)...")
        while True:
            obs = shm.observation

            if obs.sequence != last_seq:
                last_seq = obs.sequence
                step += 1

                pos = list(obs.player_origin)
                vel = list(obs.player_velocity)
                va = list(obs.viewangles)

                speed = math.sqrt(vel[0]**2 + vel[1]**2)

                print(f"--- [Step #{step:04d} | Seq:{obs.sequence}] ---")
                print(f"  Player Status: Alive={obs.player_alive} | HP={obs.health} | AP={obs.armor}")
                print(f"  Position:      X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f}")
                print(f"  Speed:         {speed:.1f} u/s (Vx={vel[0]:.1f}, Vy={vel[1]:.1f})")
                print(f"  ViewAngles:    Pitch={va[0]:.1f}, Yaw={va[1]:.1f}")
                print(f"  Weapon Info:   ID={obs.active_weapon_id} | Clip={obs.clip_ammo} | Reserve={obs.reserve_ammo}")
                print(f"  Enemies Count: {obs.num_enemies}")

                for i in range(obs.num_enemies):
                    e = obs.enemies[i]
                    if e.active:
                        print(f"    -> Enemy #{e.index}: RelPos=({e.relative_pos[0]:.1f}, {e.relative_pos[1]:.1f}, {e.relative_pos[2]:.1f}) Dist={e.distance:.1f}u")

                # Test automated control sequence
                act = shm.action
                act.sequence_ack = obs.sequence

                # Demo routine: Walk around, spin camera, jump, and fire
                phase = (step // 20) % 4
                if phase == 0:
                    print("  [ACTION] Moving Forward + Jumping")
                    act.forwardmove = 400.0
                    act.sidemove = 0.0
                    act.buttons = 2 # IN_JUMP
                elif phase == 1:
                    print("  [ACTION] Strafe Right + Turn Yaw")
                    act.forwardmove = 0.0
                    act.sidemove = 400.0
                    act.viewangles_delta[1] = 2.0 # Yaw right 2 degrees per frame
                    act.buttons = 0
                elif phase == 2:
                    print("  [ACTION] Moving Backwards + Shooting")
                    act.forwardmove = -400.0
                    act.sidemove = 0.0
                    act.buttons = 1 # IN_ATTACK
                elif phase == 3:
                    # Auto-aim at first enemy if visible
                    if obs.num_enemies > 0 and obs.enemies[0].active:
                        e = obs.enemies[0]
                        target_yaw = math.atan2(e.relative_pos[1], e.relative_pos[0]) * 180.0 / math.pi
                        print(f"  [ACTION] Auto-aiming at Enemy #{e.index} (Target Yaw={target_yaw:.1f}) + Shooting!")
                        act.forwardmove = 200.0
                        act.buttons = 1 | 2 # IN_ATTACK | IN_JUMP
                    else:
                        print("  [ACTION] Ducking & Scanning")
                        act.forwardmove = 0.0
                        act.sidemove = 0.0
                        act.buttons = 4 # IN_DUCK

            time.sleep(0.05) # 20 Hz query rate

    except KeyboardInterrupt:
        print("\nStopping Agent test client...")
    finally:
        os.close(fd)

if __name__ == "__main__":
    main()
