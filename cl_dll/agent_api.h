#ifndef AGENT_API_H
#define AGENT_API_H

#include <cstdint>

#define AGENT_MAX_ENEMIES 16
#define AGENT_MAX_ITEMS 16
#define AGENT_SHM_NAME "/openag_agent_shm"

#pragma pack(push, 1)

struct AgentEnemyInfo {
    int32_t active;
    int32_t index;
    float origin[3];
    float relative_pos[3];
    float distance;
};

struct AgentItemInfo {
    int32_t active;
    float relative_pos[3];
    float distance;
    float world_pos[3];   // Absolute world position for 3D overlay rendering
};

struct AgentObservation {
    uint32_t sequence;        // Incremented every frame
    int32_t player_alive;
    float player_origin[3];
    float player_velocity[3];
    float viewangles[3];
    int32_t health;
    int32_t armor;
    int32_t active_weapon_id;
    int32_t clip_ammo;
    int32_t reserve_ammo;

    float wall_distances[16];  // 16-directional 3D Multi-Height LIDAR raycasts (0 to 1000 units)

    int32_t goal_active;
    float goal_origin[3];

    int32_t num_items;
    AgentItemInfo items[AGENT_MAX_ITEMS];

    int32_t num_enemies;
    AgentEnemyInfo enemies[AGENT_MAX_ENEMIES];

    float strafe_efficiency;  // 0.0 to 1.0 from OpenAG Strafe Analyzer
    int32_t strafe_state;     // 0=none, 1=perfect, 2=under, 3=over, 4=loss, 5=wrong_key
    int32_t on_ground;        // 1 if touching ground, 0 if airborne
};

struct AgentAction {
    uint32_t sequence_ack;    // Ack sequence from observation
    float forwardmove;       // -400 to 400
    float sidemove;          // -400 to 400
    float upmove;            // -320 to 320
    float viewangles_delta[3]; // Pitch, Yaw, Roll deltas
    uint32_t buttons;        // Bitmask: 1=Attack, 2=Jump, 4=Duck, 8=Attack2, 16=Reload
    int32_t weapon_select;   // 0 = no change, or weapon slot/ID
    float last_reward;       // Live reward calculated by Python
    char reward_reason[64];  // Reason label (e.g. "Goal Progression", "Stagnation")
    float reward_speed;      // Cumulative speed reward
    float reward_scurve;     // Cumulative S-curve switch reward
    float reward_bhop;       // Cumulative Bhop hop reward
    float reward_streak;     // Cumulative streak penalty
    float reward_asym;       // Cumulative side asymmetry penalty
    float reward_wall;       // Cumulative wall proximity penalty
    float reward_goal;       // Cumulative goal progress reward
    int32_t eval_mode;       // 1 = Evaluation / telemetry-only (do not override player controls)
    float loss_policy;       // PPO Policy Loss
    float loss_value;        // PPO Value Loss
    float entropy;           // PPO Entropy Loss
    float learning_rate;     // PPO Learning Rate
    float system_fps;        // Real-time training steps per second

    int32_t reset_requested;   // 1 = execute instant state reset
    float reset_origin[3];     // Target XYZ origin for reset
    float reset_velocity[3];   // Target XYZ velocity for reset
    float reset_angles[3];     // Target viewangles for reset
};

struct AgentSharedData {
    AgentObservation observation;
    AgentAction action;
    int32_t api_enabled;
    int32_t client_connected;
};

#pragma pack(pop)

struct cvar_s;
typedef struct cvar_s cvar_t;
extern cvar_t* cl_agent_cam;

void AgentAPI_RegisterCvar();
void AgentAPI_Init();
void AgentAPI_Shutdown();
void AgentAPI_UpdateObservation();
void AgentAPI_UpdateSimvel(const float* simvel);
void AgentAPI_ApplyAction(void* pcmd, float* pviewangles);
void AgentAPI_RenderHUD();
void AgentAPI_RenderCamera(float* origin, float* angles);
void AgentAPI_WaitStep();
void AgentAPI_SignalDone();
void AgentAPI_CheckAndExecuteReset();
void AgentAPI_RenderDebug3DLines();
void Cmd_EditorMouseClick();
bool AgentAPI_IsEnabled();

#endif // AGENT_API_H
