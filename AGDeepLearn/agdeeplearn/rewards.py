import math

class RewardCalculator:
    """
    Focused Movement / Bhop Reward Calculator for OpenAG.
    Encourages forward momentum, bunnyhopping, and maintaining horizontal speed.
    Penalizes standing still and extreme vertical pitch angles.
    """
    def __init__(self, speed_reward_scale=0.02, min_speed_threshold=200.0):
        self.speed_reward_scale = speed_reward_scale
        self.min_speed_threshold = min_speed_threshold
        self.last_origin = None
        self.last_side = 0.0
        self.last_goal_dist = None
        self.pos_buffer = []
        self.same_side_streak = 0
        self.side_history = []

    def reset(self, initial_obs=None):
        self.last_side = 0.0
        self.last_goal_dist = None
        self.pos_buffer = []
        self.same_side_streak = 0
        self.side_history = []
        if initial_obs:
            self.last_origin = list(initial_obs.player_origin)
        else:
            self.last_origin = None

    def compute_reward(self, current_obs, current_action=None):
        if not current_obs or current_obs.player_alive == 0:
            return -5.0

        vx = current_obs.player_velocity[0]
        vy = current_obs.player_velocity[1]
        vz = current_obs.player_velocity[2]
        speed = math.sqrt(vx * vx + vy * vy)
        yaw_rad = math.radians(current_obs.viewangles[1])

        # Forward velocity component in looking direction
        v_forward = vx * math.cos(yaw_rad) + vy * math.sin(yaw_rad)
        pitch = current_obs.viewangles[0]

        reward = 0.0

        # Position History Buffer for Stagnation & Room Looping Detection
        self.pos_buffer.append(list(current_obs.player_origin))
        if len(self.pos_buffer) > 40:
            self.pos_buffer.pop(0)

        if len(self.pos_buffer) == 40:
            p_old = self.pos_buffer[0]
            p_now = self.pos_buffer[-1]
            disp = math.sqrt((p_now[0] - p_old[0])**2 + (p_now[1] - p_old[1])**2 + (p_now[2] - p_old[2])**2)
            if disp < 140.0:
                reward -= 0.4  # Stagnation & Room Looping Penalty!

        # Interactive Goal Progression & Completion Reward
        if current_obs.goal_active:
            gdx = current_obs.goal_origin[0] - current_obs.player_origin[0]
            gdy = current_obs.goal_origin[1] - current_obs.player_origin[1]
            gdz = current_obs.goal_origin[2] - current_obs.player_origin[2]
            gdist = math.sqrt(gdx * gdx + gdy * gdy + gdz * gdz)

            if self.last_goal_dist is not None:
                dist_delta = self.last_goal_dist - gdist
                if dist_delta > 0:
                    # Reward closing distance to goal, scaled by Bhop speed!
                    reward += dist_delta * 0.02 * (1.0 + speed / 250.0)

            self.last_goal_dist = gdist

            if gdist < 60.0:
                reward += 5.0  # Massive Completion Bonus for reaching the target goal!
                self.last_goal_dist = None
        else:
            self.last_goal_dist = None

        # 1. Forward Momentum Reward (strongly encourage moving forward relative to viewangle)
        if v_forward > 0:
            reward += (v_forward / 250.0) * 0.4
        else:
            reward -= 0.1  # Penalty for moving backwards or staying stationary

        # 2. Exponential Bonus for Bhop Speed over standard run (250 u/s)
        if speed > 250.0:
            reward += ((speed - 250.0) / 50.0) ** 1.5 * 0.5

        # 3. Penalty for holding W/S at high speed (>220 u/s) during air strafing
        if current_action is not None and speed > 220.0:
            fwd = float(current_action[0])
            side = float(current_action[1])

            if abs(fwd) > 50.0:
                reward -= 0.15

            # Strafe Alternation Bonus: Reward switching between A (-400) and D (+400)
            if abs(side) > 50.0:
                if self.last_side != 0.0 and (side * self.last_side < 0.0):
                    reward += 0.2  # Strong bonus for alternating A -> D or D -> A strafe!
                self.last_side = side

        # 4. Wall Collision & Open Corridor Steering Reward
        front_wall_dist = current_obs.wall_distances[0]
        max_open_dist = max(current_obs.wall_distances)
        
        # Heavy penalty if stuck against a wall (front wall < 80 units and speed < 50 u/s)
        if front_wall_dist < 80.0 and front_wall_dist > 0.0:
            reward -= (1.0 - (front_wall_dist / 80.0)) * 0.5
            if speed < 50.0:
                reward -= 0.3  # Extra penalty for getting stuck against a wall

        # 5. Item Collection & Target Looting Reward (Weapons, Ammo, Armor, Longjump)
        if current_obs.num_items > 0:
            closest_item_dist = current_obs.items[0].distance
            if closest_item_dist < 80.0 and closest_item_dist > 0.0:
                reward += 1.5  # Large Item Looting Bonus for picking up items!
            elif closest_item_dist < 600.0:
                reward += (1.0 - (closest_item_dist / 600.0)) * 0.2  # Guidance incentive towards closest item

        # 6. Penalty for looking straight up or straight down (pitch > 45 deg)
        if abs(pitch) > 45.0:
            reward -= 0.1

        # Save last origin
        self.last_origin = list(current_obs.player_origin)

        return float(reward)


class Phase1AirStrafeRewardCalculator:
    """
    Phase 1 Reward Calculator: Pure Synchronized Velocity Reward.
    
    Mechanics:
    - Synchronized Strafe: yaw_delta * sidemove < 0 (Turn Left + A key OR Turn Right + D key).
    - Speed Gain: High reward when velocity increases during synchronized strafes.
    - Smoothness: Action smoothness penalty for jerky mouse movements.
    """
    def __init__(self, smoothness_penalty_weight=0.04, speed_scale=0.03):
        self.smoothness_penalty_weight = smoothness_penalty_weight
        self.speed_scale = speed_scale
        self.last_speed = 0.0
        self.last_yaw_delta = 0.0

    def reset(self, initial_obs=None):
        self.last_speed = 0.0
        self.last_yaw_delta = 0.0

    def compute_reward(self, current_obs, yaw_delta=0.0, sidemove=0.0):
        if not current_obs or current_obs.player_alive == 0:
            return -5.0

        vx, vy = current_obs.player_velocity[0], current_obs.player_velocity[1]
        current_speed = math.sqrt(vx * vx + vy * vy)

        reward = 0.0

        # Synchronized Strafe: Mouse turn matches A/D key direction
        # Turn Left (yaw_delta > 0) + A key (sidemove < 0) => yaw_delta * sidemove < 0
        # Turn Right (yaw_delta < 0) + D key (sidemove > 0) => yaw_delta * sidemove < 0
        is_synchronized = bool(yaw_delta * sidemove < 0.0 and abs(yaw_delta) > 0.001)

        speed_gain = current_speed - self.last_speed

        if is_synchronized:
            if speed_gain > 0.0:
                reward += speed_gain * 0.15  # High reward for speed gain during synchronized strafe!
            reward += (current_speed / 100.0) * self.speed_scale
        else:
            # Desynchronized (turning camera wrong way relative to A/D key)
            if abs(yaw_delta) > 0.01:
                reward -= 0.25

        # Smoothness penalty for erratic mouse jitter
        yaw_jerk = abs(yaw_delta - self.last_yaw_delta)
        smoothness_penalty = (yaw_jerk ** 2) * self.smoothness_penalty_weight
        reward -= smoothness_penalty

        self.last_speed = current_speed
        self.last_yaw_delta = yaw_delta

        return float(reward)


class Phase2DirectionalRewardCalculator:
    """
    Phase 2 Reward Calculator: Waypoint-Targeted Directional Air-Strafing.
    
    Objective:
    - Active Angle Steering: Reward turning camera toward target waypoint angle.
    - Velocity Projection: Heavily reward velocity component pointing toward target (v . u_target).
    - Penalize flying or turning away from target.
    - Large +15.0 bonus for reaching checkpoint!
    """
    def __init__(self, smoothness_penalty_weight=0.04):
        self.smoothness_penalty_weight = smoothness_penalty_weight
        self.last_dist_to_target = None
        self.last_yaw_delta = 0.0
        self.same_spin_direction = 0 # -1 for left spin, +1 for right spin
        self.same_spin_ticks = 0
        self.last_speed = 0.0

    def reset(self, initial_obs=None):
        self.last_dist_to_target = None
        self.last_yaw_delta = 0.0
        self.same_spin_direction = 0
        self.same_spin_ticks = 0
        self.last_speed = 0.0

    def compute_reward(self, current_obs, target_pos, yaw_delta=0.0, sidemove=0.0):
        if not current_obs or current_obs.player_alive == 0:
            return -5.0, False

        px, py = current_obs.player_origin[0], current_obs.player_origin[1]
        tx, ty = target_pos[0], target_pos[1]

        dx = tx - px
        dy = ty - py
        dist_to_target = math.sqrt(dx * dx + dy * dy)

        reward = 0.0
        reached_target = False

        # 1. Target Reached Check (+15.0 bonus!)
        if dist_to_target < 75.0:
            reward += 15.0
            reached_target = True

        vx, vy = current_obs.player_velocity[0], current_obs.player_velocity[1]
        current_speed = math.sqrt(vx * vx + vy * vy)

        # 2. Kinetic Speed Delta Reward (Reward ONLY strafes that increase speed)
        speed_delta = current_speed - self.last_speed
        if speed_delta > 0.0:
            reward += min(1.0, speed_delta * 0.05)  # Bonus for strafes that ADD speed!
        elif speed_delta < -5.0:
            reward -= min(1.0, abs(speed_delta) * 0.05)  # Penalty for bad strafes that shed speed!

        # 3. Angle Steering Guidance & Velocity Projection
        if dist_to_target > 0.001:
            angle_to_target = math.atan2(dy, dx)
            facing_yaw = math.radians(current_obs.viewangles[1])
            rel_angle = math.atan2(math.sin(angle_to_target - facing_yaw), math.cos(angle_to_target - facing_yaw))

            # Steering towards checkpoint
            if (rel_angle > 0.05 and yaw_delta > 0.0) or (rel_angle < -0.05 and yaw_delta < 0.0):
                reward += 0.35  # Active steering toward checkpoint!
            elif abs(rel_angle) > 0.2 and ((rel_angle > 0 and yaw_delta < 0) or (rel_angle < 0 and yaw_delta > 0)):
                reward -= 0.45  # Steering away from checkpoint penalty!

            # Straight-Line High-Speed Bhop (When target is straight ahead, fly straight!)
            if abs(rel_angle) < 0.12 and current_speed > 350.0:
                if abs(yaw_delta) < 3.0:
                    reward += 0.35  # Bonus for flying straight to target without unnecessary wide strafes!

            # Velocity Projection onto Target Vector
            ux = dx / dist_to_target
            uy = dy / dist_to_target
            target_proj_speed = vx * ux + vy * uy

            if target_proj_speed > 0.0:
                reward += (target_proj_speed / 100.0) * 0.30  # High reward for flying TOWARD target!
            else:
                reward -= (abs(target_proj_speed) / 100.0) * 0.40 # Heavy penalty for flying AWAY from target!

            # 3. Orthogonal Lateral Drift Penalty (Cure for Far Distance Circular Bhop!)
            # V_ortho = speed wasting energy spinning in circles perpendicular to target line
            v_ortho_sq = max(0.0, (current_speed ** 2) - (target_proj_speed ** 2))
            v_ortho = math.sqrt(v_ortho_sq)
            if v_ortho > 100.0:
                # Heavy penalty for circular orbit drift when target is far away!
                reward -= (v_ortho / 100.0) * 0.45

            # 4. Straight-Line Velocity Efficiency Ratio
            if current_speed > 100.0:
                eta = target_proj_speed / current_speed  # Ratio of total velocity pointed at target (-1.0 to +1.0)
                if eta < 0.5:
                    reward -= (0.5 - eta) * 0.70
                elif eta >= 0.75:
                    reward += eta * 0.50

        # 4. Anti-Continuous Spin Saturation Penalty (Forces S-curves instead of spinning in circles)
        current_spin = -1 if yaw_delta > 0.02 else (1 if yaw_delta < -0.02 else 0)
        if current_spin != 0:
            if current_spin == self.same_spin_direction:
                self.same_spin_ticks += 1
            else:
                self.same_spin_direction = current_spin
                self.same_spin_ticks = 1
        
        if self.same_spin_ticks > 15:
            # Continuous single-direction spinning penalty
            spin_penalty = (self.same_spin_ticks - 15) * 0.05
            reward -= min(1.0, spin_penalty)

        # 5. High-Speed Velocity Preservation Bonus (500+ u/s)
        if current_speed > 450.0:
            high_speed_bonus = ((current_speed - 450.0) / 100.0) * 0.45
            reward += high_speed_bonus

            # High-speed mouse airbrake penalty: Wide mouse swings at 500+ u/s cause friction!
            if abs(yaw_delta) > 8.0:
                reward -= 0.50

        # 6. Synchronized Strafe Bonus
        is_synchronized = bool(yaw_delta * sidemove < 0.0 and abs(yaw_delta) > 0.001)
        if is_synchronized:
            reward += 0.15

        # 7. Smoothness Penalty
        yaw_jerk = abs(yaw_delta - self.last_yaw_delta)
        smoothness_penalty = (yaw_jerk ** 2) * self.smoothness_penalty_weight
        reward -= smoothness_penalty

        # 8. Smart Vectorized Wall Perception & Parallel Clearance
        if current_obs and hasattr(current_obs, "wall_distances"):
            facing_rad = math.radians(current_obs.viewangles[1])
            vx, vy = current_obs.player_velocity[0], current_obs.player_velocity[1]
            ray_angles = [0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0]

            for i in range(8):
                dist = current_obs.wall_distances[i]
                if dist < 120.0:
                    wall_angle = facing_rad + math.radians(ray_angles[i])
                    wall_nx = math.cos(wall_angle)
                    wall_ny = math.sin(wall_angle)

                    # Velocity component pointing DIRECTLY INTO this wall
                    v_into_wall = vx * wall_nx + vy * wall_ny

                    if v_into_wall > 50.0:
                        # Player is actively FLYING INTO wall -> Penalty proportional to collision velocity!
                        penalty = (v_into_wall / 100.0) * (120.0 - dist) * 0.005
                        reward -= min(1.0, penalty)
                    elif v_into_wall <= 0.0 and current_speed > 200.0:
                        # Player is flying PARALLEL or AWAY from wall -> Reward wall clearance!
                        reward += 0.15

        self.last_dist_to_target = dist_to_target
        self.last_yaw_delta = yaw_delta
        self.last_speed = current_speed

        return float(reward), reached_target


class Phase3FullControlsRewardCalculator(Phase2DirectionalRewardCalculator):
    """
    Phase 3 Reward Calculator: Unlocking W/S, Pitch, and Crouch-Bhop.
    
    Mechanics:
    - Inherits Phase 2 directional velocity projection & anti-spin rules.
    - W/S Guidance: Reward W on ground for launch, penalize W air-brake during side-strafes.
    - Crouch-Bhop: Reward air-ducking near ground / checkpoints.
    - Pitch Control: Maintain natural horizon pitch level.
    """
    def __init__(self, smoothness_penalty_weight=0.04):
        super().__init__(smoothness_penalty_weight=smoothness_penalty_weight)

    def compute_reward(self, current_obs, target_pos, yaw_delta=0.0, pitch_delta=0.0, sidemove=0.0, forwardmove=0.0, duck_active=False):
        base_reward, reached = super().compute_reward(
            current_obs,
            target_pos=target_pos,
            yaw_delta=yaw_delta,
            sidemove=sidemove
        )

        if not current_obs or current_obs.player_alive == 0:
            return base_reward, reached

        reward = base_reward
        on_ground = bool(current_obs.on_ground)

        # 1. Forwardmove (W/S) Physics Rules
        if on_ground:
            if forwardmove > 100.0:
                reward += 0.20  # Reward holding W on ground for forward launch acceleration!
        else:
            # In air: W key during side-strafing acts as an airbrake in GoldSrc
            if forwardmove > 100.0 and abs(sidemove) > 100.0:
                reward -= 0.20  # Penalize W airbrake during air-strafes!

        # 2. Crouch-Bhop / Air-Ducking Rules
        if duck_active:
            if not on_ground:
                reward += 0.15  # Reward air-ducking for crouch-bhop clearance!
            else:
                reward -= 0.10  # Slight penalty for crouching while stuck on ground

        # 3. Pitch Angle Comfort (Keep camera level near horizon)
        pitch = current_obs.viewangles[0]
        if abs(pitch) > 45.0:
            reward -= 0.20  # Penalize looking straight up or straight down

        return float(reward), reached


class Phase3ManualJumpRewardCalculator(Phase2DirectionalRewardCalculator):
    """
    Phase 3 Reward Calculator: Manual Jump Timing & Friction-Preservation Bhop.
    
    Mechanics:
    - Inherits Phase 2 directional velocity projection, speed delta & wall clearance.
    - Perfect Jump Timing Bonus: Reward +1.50 for pressing JUMP exactly on ground touch frame!
    - Jump Spam Penalty: Penalize holding jump continuously like a macro.
    - Ground Friction Loss Penalty: Penalize velocity loss caused by staying on ground without jumping.
    """
    def __init__(self, smoothness_penalty_weight=0.04):
        super().__init__(smoothness_penalty_weight=smoothness_penalty_weight)
        self.was_on_ground = True
        self.jump_hold_ticks = 0

    def reset(self, initial_obs=None):
        super().reset(initial_obs=initial_obs)
        self.was_on_ground = True
        self.jump_hold_ticks = 0

    def compute_reward(self, current_obs, target_pos, yaw_delta=0.0, sidemove=0.0, jump_pressed=False):
        base_reward, reached = super().compute_reward(
            current_obs,
            target_pos=target_pos,
            yaw_delta=yaw_delta,
            sidemove=sidemove
        )

        if not current_obs or current_obs.player_alive == 0:
            return base_reward, reached

        reward = base_reward
        on_ground = bool(current_obs.on_ground)
        vx, vy = current_obs.player_velocity[0], current_obs.player_velocity[1]
        current_speed = math.sqrt(vx * vx + vy * vy)

        # 1. Track Jump Hold Ticks (Anti-Spam / Anti-Macro)
        if jump_pressed:
            self.jump_hold_ticks += 1
            if self.jump_hold_ticks > 4:
                # Holding jump button continuously like a macro penalty!
                reward -= 0.15
        else:
            self.jump_hold_ticks = 0

        # 2. Perfect Ground-Touch Jump Timing Bonus
        if on_ground and jump_pressed:
            reward += 1.50  # Perfect jump timing!

        # 3. Ground Friction Velocity Loss Penalty
        if on_ground and not jump_pressed:
            speed_delta = current_speed - self.last_speed
            if speed_delta < -15.0:
                reward -= 0.60  # Friction speed-loss penalty for missing jump timing!

        self.was_on_ground = on_ground
        return float(reward), reached


class Phase4SpatialVisionRewardCalculator(Phase2DirectionalRewardCalculator):
    """
    Phase 4 Reward Calculator: Spatial Vision & Corner-Licking Obstacle Navigation.
    
    Mechanics:
    - 16 LiDAR 3D Spatial Perception.
    - Giant -100.0 Crash Penalty & Instant Episode Termination on Wall Collision.
    - Corner-Licking Clearance Bonus (+0.50): Reward passing near corners at high speed without hitting.
    - Curved Arc Wall-Bounce Air-Strafe Bonus (+0.40): Reward changing strafe arc away from approaching obstacles.
    """
    def __init__(self, smoothness_penalty_weight=0.04):
        super().__init__(smoothness_penalty_weight=smoothness_penalty_weight)

    def compute_reward(self, current_obs, target_pos, yaw_delta=0.0, sidemove=0.0):
        base_reward, reached = super().compute_reward(
            current_obs,
            target_pos=target_pos,
            yaw_delta=yaw_delta,
            sidemove=sidemove
        )

        if not current_obs or current_obs.player_alive == 0:
            return -100.0, False, True  # Crash penalty & terminated!

        vx, vy = current_obs.player_velocity[0], current_obs.player_velocity[1]
        current_speed = math.sqrt(vx * vx + vy * vy)
        reward = base_reward
        collided = False

        # Inspect 16 LiDAR raycasts
        if hasattr(current_obs, "wall_distances"):
            min_dist = min(current_obs.wall_distances)
            front_dists = [current_obs.wall_distances[i] for i in [0, 1, 15]]
            min_front = min(front_dists)

            # 1. Crash Collision Check (Giant -100.0 penalty and episode termination!)
            if min_dist < 28.0:
                reward -= 100.0
                collided = True
            elif self.last_speed > 250.0 and current_speed < 40.0 and min_front < 60.0:
                # Sudden velocity crash drop into wall!
                reward -= 100.0
                collided = True

            # 2. Corner-Licking Clearance Bonus (+0.50)
            # Reward skimming near corners (32 to 80 units) at 400+ u/s without colliding!
            if 32.0 < min_dist < 80.0 and current_speed > 400.0 and not collided:
                reward += 0.50  # Corner-licking high-speed clearance bonus!

            # 3. Obstacle Avoidance Arc Turn Bonus (+0.40)
            if min_front < 150.0 and current_speed > 200.0:
                # Approaching wall ahead: Reward turning camera & strafing away!
                if abs(yaw_delta) > 2.0 and not collided:
                    reward += 0.40  # Obstacle avoidance arc turn bonus!

        return float(reward), reached, collided

