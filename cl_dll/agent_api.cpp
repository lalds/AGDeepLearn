#include <fstream>
#include <algorithm>
#include "hud.h"
#include "cl_util.h"
#include "ammohistory.h"
#include "agent_api.h"
#include "usercmd.h"
#include "cvardef.h"
#include "in_defs.h"
#include "pm_defs.h"
#include "event_api.h"
#include "pmtrace.h"
#include "com_model.h"
#include "triangleapi.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <cstring>
#include <cmath>

static cvar_t* cl_agent_api = nullptr;
static cvar_t* cl_agent_api_id = nullptr;
static cvar_t* cl_agent_debug_items = nullptr;
cvar_t* cl_agent_cam = nullptr; // 0 = 1st Person, 1 = 3rd Person Chase Cam, 2 = Freecam
static float g_freecamPos[3] = { 0.0f, 0.0f, 0.0f };
static bool g_freecamInit = false;
static float g_editorCamAngles[3] = { 0.0f, 0.0f, 0.0f };
static float s_AgentPlayerYaw = 0.0f;
static int g_shmFd = -1;
static AgentSharedData* g_pShm = nullptr;
static sem_t* g_semStep = SEM_FAILED;
static sem_t* g_semDone = SEM_FAILED;
static bool g_ApiInitialized = false;
static float g_AgentSimvel[3] = { 0.0f, 0.0f, 0.0f };

struct Waypoint {
    float origin[3];
};

static bool g_GoalActive = false;
static float g_GoalOrigin[3] = { 0.0f, 0.0f, 0.0f };
static std::vector<Waypoint> g_Waypoints;
static int g_CurrentWaypointIdx = 0;

// 3D Volume Box Trigger & Lap Timer Controls
static float g_TriggerRadiusX = 120.0f;
static float g_TriggerRadiusY = 120.0f;
static float g_TriggerRadiusZ = 120.0f;
static float g_LapStartTime = 0.0f;
static float g_LastLapTime = 0.0f;
static float g_BestLapTime = 999.9f;
static int g_TotalLapsCompleted = 0;

static bool g_DemoRecording = false;
static std::ofstream g_DemoFile;
static int g_DemoSampleCount = 0;
static float g_LastPlayerYaw = 0.0f;

static void UpdateCurrentGoalTarget()
{
    if (g_Waypoints.empty())
    {
        g_GoalActive = false;
        return;
    }
    if (g_CurrentWaypointIdx < 0 || g_CurrentWaypointIdx >= (int)g_Waypoints.size())
    {
        g_CurrentWaypointIdx = 0;
    }
    g_GoalOrigin[0] = g_Waypoints[g_CurrentWaypointIdx].origin[0];
    g_GoalOrigin[1] = g_Waypoints[g_CurrentWaypointIdx].origin[1];
    g_GoalOrigin[2] = g_Waypoints[g_CurrentWaypointIdx].origin[2];
    g_GoalActive = true;
}

void Cmd_EditorMouseClick()
{
    cl_entity_t* localPlayer = gEngfuncs.GetLocalPlayer();
    if (!localPlayer || !gEngfuncs.pEventAPI)
        return;

    float va[3];
    gEngfuncs.GetViewAngles(va);

    float start[3] = { localPlayer->origin[0], localPlayer->origin[1], localPlayer->origin[2] + 16.0f };
    float forward[3], right[3], up[3];
    gEngfuncs.pfnAngleVectors(va, forward, right, up);

    float end[3] = {
        start[0] + forward[0] * 8192.0f,
        start[1] + forward[1] * 8192.0f,
        start[2] + forward[2] * 8192.0f
    };

    pmtrace_t tr;
    gEngfuncs.pEventAPI->EV_PlayerTrace(start, end, PM_WORLD_ONLY, -1, &tr);

    Waypoint wp;
    wp.origin[0] = tr.endpos[0];
    wp.origin[1] = tr.endpos[1];
    wp.origin[2] = tr.endpos[2];
    g_Waypoints.push_back(wp);

    UpdateCurrentGoalTarget();

    gEngfuncs.Con_Printf("[AgentAPI] Waypoint #%d Added at (%.0f, %.0f, %.0f) [Circuit Total: %d]\n",
        (int)g_Waypoints.size(), wp.origin[0], wp.origin[1], wp.origin[2], (int)g_Waypoints.size());
}

void Cmd_SetGoal()
{
    Cmd_EditorMouseClick();
}

void Cmd_ClearGoal()
{
    g_Waypoints.clear();
    g_CurrentWaypointIdx = 0;
    g_GoalActive = false;
    gEngfuncs.Con_Printf("[AgentAPI] Cleared All Circuit Waypoints\n");
}

void Cmd_PopGoal()
{
    if (!g_Waypoints.empty())
    {
        g_Waypoints.pop_back();
        if (g_CurrentWaypointIdx >= (int)g_Waypoints.size())
        {
            g_CurrentWaypointIdx = 0;
        }
        UpdateCurrentGoalTarget();
        gEngfuncs.Con_Printf("[AgentAPI] Removed Last Waypoint. [Remaining: %d]\n", (int)g_Waypoints.size());
    }
}

void Cmd_RecordDemo()
{
    if (g_DemoRecording)
    {
        gEngfuncs.Con_Printf("[AgentAPI] Demo recording is ALREADY running! (%d samples recorded)\n", g_DemoSampleCount);
        return;
    }

    system("mkdir -p /home/angel/AGDeepLearn/data");
    g_DemoFile.open("/home/angel/AGDeepLearn/data/bhop_demo.csv", std::ios::out | std::ios::trunc);
    if (!g_DemoFile.is_open())
    {
        gEngfuncs.Con_Printf("[AgentAPI] ERROR: Could not open /home/angel/AGDeepLearn/data/bhop_demo.csv for recording!\n");
        return;
    }

    g_DemoRecording = true;
    g_DemoSampleCount = 0;

    float va[3];
    gEngfuncs.GetViewAngles(va);
    g_LastPlayerYaw = va[1];

    gEngfuncs.Con_Printf("[AgentAPI] STARTED RECORDING HUMAN DEMONSTRATION! Play and Bhop to the goal!\n");
}

void Cmd_StopDemo()
{
    if (!g_DemoRecording)
    {
        gEngfuncs.Con_Printf("[AgentAPI] Demo recording is not active.\n");
        return;
    }

    g_DemoRecording = false;
    if (g_DemoFile.is_open())
    {
        g_DemoFile.close();
    }

    gEngfuncs.Con_Printf("[AgentAPI] STOPPED RECORDING DEMO! Saved %d human samples to /home/angel/AGDeepLearn/data/bhop_demo.csv\n", g_DemoSampleCount);
}

void AgentAPI_UpdateSimvel(const float* simvel)
{
    if (simvel)
    {
        g_AgentSimvel[0] = simvel[0];
        g_AgentSimvel[1] = simvel[1];
        g_AgentSimvel[2] = simvel[2];
    }
}

void AgentAPI_RegisterCvar()
{
    if (!cl_agent_api)
    {
        cl_agent_api = gEngfuncs.pfnRegisterVariable("cl_agent_api", "1", FCVAR_ARCHIVE);
    }
    if (!cl_agent_api_id)
    {
        cl_agent_api_id = gEngfuncs.pfnRegisterVariable("cl_agent_api_id", "0", FCVAR_ARCHIVE);
    }
    if (!cl_agent_debug_items)
    {
        cl_agent_debug_items = gEngfuncs.pfnRegisterVariable("cl_agent_debug_items", "0", FCVAR_CLIENTDLL);
    }
    if (!cl_agent_cam)
    {
        cl_agent_cam = gEngfuncs.pfnRegisterVariable("cl_agent_cam", "0", FCVAR_ARCHIVE);
    }

    gEngfuncs.pfnAddCommand("ag_set_goal", Cmd_SetGoal);
    gEngfuncs.pfnAddCommand("ag_add_goal", Cmd_SetGoal);
    gEngfuncs.pfnAddCommand("ag_clear_goal", Cmd_ClearGoal);
    gEngfuncs.pfnAddCommand("ag_pop_goal", Cmd_PopGoal);
    gEngfuncs.pfnAddCommand("ag_record_demo", Cmd_RecordDemo);
    gEngfuncs.pfnAddCommand("ag_stop_demo", Cmd_StopDemo);
    gEngfuncs.pfnAddCommand("ag_editor_click", Cmd_EditorMouseClick);
}

void AgentAPI_Init()
{
    if (g_ApiInitialized)
        return;

    AgentAPI_RegisterCvar();

    cvar_t* pAutojump = gEngfuncs.pfnGetCvarPointer("cl_autojump");
    if (pAutojump)
    {
        pAutojump->value = 1.0f;
    }

    cvar_t* pHostFramerate = gEngfuncs.pfnGetCvarPointer("host_framerate");
    if (pHostFramerate)
    {
        pHostFramerate->value = 0.01f;
    }

    char shm_name[64];
    int inst_id = cl_agent_api_id ? (int)cl_agent_api_id->value : 0;
    if (inst_id > 0)
        snprintf(shm_name, sizeof(shm_name), "/openag_agent_shm_%d", inst_id);
    else
        snprintf(shm_name, sizeof(shm_name), "%s", AGENT_SHM_NAME);

    // Create shared memory object
    g_shmFd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (g_shmFd < 0)
    {
        gEngfuncs.Con_DPrintf("AgentAPI: Failed to shm_open %s\n", shm_name);
        return;
    }

    if (ftruncate(g_shmFd, sizeof(AgentSharedData)) < 0)
    {
        gEngfuncs.Con_DPrintf("AgentAPI: Failed to ftruncate shared memory\n");
        close(g_shmFd);
        g_shmFd = -1;
        return;
    }

    g_pShm = (AgentSharedData*)mmap(nullptr, sizeof(AgentSharedData), PROT_READ | PROT_WRITE, MAP_SHARED, g_shmFd, 0);
    if (g_pShm == MAP_FAILED || !g_pShm)
    {
        gEngfuncs.Con_DPrintf("AgentAPI: Failed to mmap shared memory\n");
        close(g_shmFd);
        g_shmFd = -1;
        g_pShm = nullptr;
        return;
    }

    g_pShm->api_enabled = 1;

    sem_unlink("/openag_sem_step");
    sem_unlink("/openag_sem_done");
    g_semStep = sem_open("/openag_sem_step", O_CREAT, 0666, 0);
    g_semDone = sem_open("/openag_sem_done", O_CREAT, 0666, 0);

    g_ApiInitialized = true;
    gEngfuncs.Con_Printf("AgentAPI: Initialized Shared Memory (%s, %zu bytes) & POSIX Semaphores\n", shm_name, sizeof(AgentSharedData));
}

void AgentAPI_Shutdown()
{
    if (!g_ApiInitialized)
        return;

    if (g_semStep != SEM_FAILED) { sem_close(g_semStep); sem_unlink("/openag_sem_step"); g_semStep = SEM_FAILED; }
    if (g_semDone != SEM_FAILED) { sem_close(g_semDone); sem_unlink("/openag_sem_done"); g_semDone = SEM_FAILED; }

    if (g_pShm)
    {
        g_pShm->api_enabled = 0;
        munmap(g_pShm, sizeof(AgentSharedData));
        g_pShm = nullptr;
    }

    if (g_shmFd >= 0)
    {
        close(g_shmFd);
        g_shmFd = -1;
    }

    shm_unlink(AGENT_SHM_NAME);
    g_ApiInitialized = false;
    gEngfuncs.Con_Printf("AgentAPI: Shutdown completed\n");
}

void AgentAPI_WaitStep()
{
    if (g_pShm && g_pShm->api_enabled && g_pShm->action.eval_mode == 0 && g_semStep != SEM_FAILED)
    {
        sem_wait(g_semStep);
    }
}

void AgentAPI_SignalDone()
{
    if (g_pShm && g_pShm->api_enabled && g_pShm->action.eval_mode == 0 && g_semDone != SEM_FAILED)
    {
        sem_post(g_semDone);
    }
}

void AgentAPI_CheckAndExecuteReset()
{
    if (!g_pShm || !g_pShm->action.reset_requested)
        return;

    cl_entity_t* localPlayer = gEngfuncs.GetLocalPlayer();
    if (localPlayer)
    {
        localPlayer->origin[0] = g_pShm->action.reset_origin[0];
        localPlayer->origin[1] = g_pShm->action.reset_origin[1];
        localPlayer->origin[2] = g_pShm->action.reset_origin[2];

        g_AgentSimvel[0] = g_pShm->action.reset_velocity[0];
        g_AgentSimvel[1] = g_pShm->action.reset_velocity[1];
        g_AgentSimvel[2] = g_pShm->action.reset_velocity[2];

        gEngfuncs.SetViewAngles(g_pShm->action.reset_angles);
    }

    g_pShm->action.reset_requested = 0;
}

void AgentAPI_RenderDebug3DLines()
{
    if (!g_pShm || g_pShm->action.eval_mode == 0 || !gEngfuncs.pTriAPI)
        return;

    cl_entity_t* localPlayer = gEngfuncs.GetLocalPlayer();
    if (!localPlayer) return;

    triangleapi_t* tri = gEngfuncs.pTriAPI;

    float origin[3] = { localPlayer->origin[0], localPlayer->origin[1], localPlayer->origin[2] + 8.0f };
    float vel_end[3] = {
        origin[0] + g_AgentSimvel[0] * 0.4f,
        origin[1] + g_AgentSimvel[1] * 0.4f,
        origin[2] + g_AgentSimvel[2] * 0.4f
    };

    tri->RenderMode(kRenderTransAdd);
    tri->Begin(TRI_LINES);
        tri->Color4f(0.0f, 1.0f, 0.0f, 1.0f); // Green velocity vector
        tri->Vertex3fv(origin);
        tri->Vertex3fv(vel_end);
    tri->End();
}

bool AgentAPI_IsEnabled()
{
    if (!cl_agent_api)
        cl_agent_api = gEngfuncs.pfnGetCvarPointer("cl_agent_api");

    if (!cl_agent_api || cl_agent_api->value == 0.0f)
    {
        if (g_ApiInitialized)
            AgentAPI_Shutdown();
        return false;
    }

    if (!g_ApiInitialized)
        AgentAPI_Init();

    return g_ApiInitialized && g_pShm != nullptr;
}

void AgentAPI_UpdateObservation()
{
    if (!AgentAPI_IsEnabled() || !g_pShm)
        return;

    cl_entity_t* localPlayer = gEngfuncs.GetLocalPlayer();
    if (!localPlayer)
    {
        g_pShm->client_connected = 0;
        return;
    }

    g_pShm->client_connected = 1;
    g_pShm->observation.sequence++;
    g_pShm->observation.player_alive = (gHUD.m_Health.m_iHealth > 0 && !gHUD.m_iIntermission) ? 1 : 0;

    g_pShm->observation.player_origin[0] = localPlayer->origin[0];
    g_pShm->observation.player_origin[1] = localPlayer->origin[1];
    g_pShm->observation.player_origin[2] = localPlayer->origin[2];

    g_pShm->observation.player_velocity[0] = g_AgentSimvel[0];
    g_pShm->observation.player_velocity[1] = g_AgentSimvel[1];
    g_pShm->observation.player_velocity[2] = g_AgentSimvel[2];

    float va[3] = { 0, 0, 0 };
    gEngfuncs.GetViewAngles(va);
    g_pShm->observation.viewangles[0] = va[0];
    g_pShm->observation.viewangles[1] = va[1];
    g_pShm->observation.viewangles[2] = va[2];

    g_pShm->observation.health = gHUD.m_Health.m_iHealth;
    g_pShm->observation.armor = gHUD.m_Battery.GetBattery();

    WEAPON* activeWep = gHUD.m_Ammo.GetActiveWeapon();
    if (activeWep)
    {
        g_pShm->observation.active_weapon_id = activeWep->iId;
        g_pShm->observation.clip_ammo = activeWep->iClip;
        g_pShm->observation.reserve_ammo = gWR.CountAmmo(activeWep->iAmmoType);
    }
    else
    {
        g_pShm->observation.active_weapon_id = 0;
        g_pShm->observation.clip_ammo = 0;
        g_pShm->observation.reserve_ammo = 0;
    }

    // Interactive 3D Volume Box Trigger & Lap Timer Engine
    if (g_GoalActive)
    {
        float dx = std::abs(localPlayer->origin[0] - g_GoalOrigin[0]);
        float dy = std::abs(localPlayer->origin[1] - g_GoalOrigin[1]);
        float dz = std::abs(localPlayer->origin[2] - g_GoalOrigin[2]);

        bool inside_trigger = (dx <= g_TriggerRadiusX && dy <= g_TriggerRadiusY && dz <= g_TriggerRadiusZ);

        if (inside_trigger)
        {
            if (!g_Waypoints.empty())
            {
                int oldIdx = g_CurrentWaypointIdx;
                g_CurrentWaypointIdx = (g_CurrentWaypointIdx + 1) % (int)g_Waypoints.size();
                UpdateCurrentGoalTarget();

                float now = gEngfuncs.GetClientTime();
                if (oldIdx == (int)g_Waypoints.size() - 1)
                {
                    // Completed a full lap of the circuit!
                    if (g_LapStartTime > 0.0f)
                    {
                        g_LastLapTime = now - g_LapStartTime;
                        g_TotalLapsCompleted++;
                        if (g_LastLapTime < g_BestLapTime)
                        {
                            g_BestLapTime = g_LastLapTime;
                            gEngfuncs.Con_Printf("[CIRCUIT RACE] 🎉 NEW LAP RECORD! %.2f seconds!\n", g_BestLapTime);
                        }
                        else
                        {
                            gEngfuncs.Con_Printf("[CIRCUIT RACE] 🏁 LAP FINISHED: %.2f s (Best: %.2f s)\n", g_LastLapTime, g_BestLapTime);
                        }

                        // Save persistent lap history to CSV file on disk!
                        FILE* fLap = fopen("/home/angel/AGDeepLearn/logs/race_lap_records.csv", "a");
                        if (fLap)
                        {
                            fprintf(fLap, "Lap %d, Time: %.2fs, Best: %.2fs\n", g_TotalLapsCompleted, g_LastLapTime, g_BestLapTime);
                            fclose(fLap);
                        }

                        // User Requested Feature: Every 2 Laps, reverse circuit direction!
                        // Forces agent to master BOTH clockwise and counter-clockwise high-speed turns!
                        if (g_TotalLapsCompleted % 2 == 0 && g_Waypoints.size() > 2)
                        {
                            std::reverse(g_Waypoints.begin(), g_Waypoints.end());
                            gEngfuncs.Con_Printf("[CIRCUIT RACE] 🔀 LAP %d FINISHED! REVERSED CIRCUIT DIRECTION FOR DYNAMIC TRAINING!\n", g_TotalLapsCompleted);
                        }
                    }
                    g_LapStartTime = now;
                }

                gEngfuncs.Con_Printf("[AgentAPI] 3D TRIGGER #%d REACHED! Advancing to Trigger #%d / %d\n",
                    oldIdx + 1, g_CurrentWaypointIdx + 1, (int)g_Waypoints.size());
            }
            else
            {
                gEngfuncs.Con_Printf("[AgentAPI] 3D TRIGGER GOAL REACHED!\n");
                g_GoalActive = false;
            }
        }
    }

    // Sync goal origin from Python if set by RL training environment
    if (g_pShm->observation.goal_active && !g_GoalActive)
    {
        g_GoalActive = true;
        g_GoalOrigin[0] = g_pShm->observation.goal_origin[0];
        g_GoalOrigin[1] = g_pShm->observation.goal_origin[1];
        g_GoalOrigin[2] = g_pShm->observation.goal_origin[2];
    }

    g_pShm->observation.goal_active = g_GoalActive ? 1 : 0;
    g_pShm->observation.goal_origin[0] = g_GoalOrigin[0];
    g_pShm->observation.goal_origin[1] = g_GoalOrigin[1];
    g_pShm->observation.goal_origin[2] = g_GoalOrigin[2];

    // 16-directional 3D Multi-Height LIDAR raycast sensors (0 to 1000 units)
    if (gEngfuncs.pEventAPI)
    {
        float yaw_deg = va[1];
        float heights[3] = { 4.0f, 20.0f, 44.0f };

        for (int i = 0; i < 16; i++)
        {
            float rel_deg = yaw_deg + (float)i * 22.5f;
            float rad = rel_deg * 3.14159265f / 180.0f;
            float dir[3] = { std::cos(rad), std::sin(rad), 0.0f };

            float min_dist = 1000.0f;

            // Multi-Height Scanning for complete 3D spatial awareness (platforms, overhangs, low walls)
            for (int h = 0; h < 3; h++)
            {
                float start[3] = { localPlayer->origin[0], localPlayer->origin[1], localPlayer->origin[2] + heights[h] };
                float end[3] = { start[0] + dir[0] * 1000.0f, start[1] + dir[1] * 1000.0f, start[2] };

                pmtrace_t tr;
                gEngfuncs.pEventAPI->EV_PlayerTrace(start, end, PM_WORLD_ONLY, -1, &tr);
                float dist = tr.fraction * 1000.0f;
                if (dist < min_dist)
                {
                    min_dist = dist;
                }
            }

            g_pShm->observation.wall_distances[i] = min_dist;
        }
    }

    // Scan map items (weapons, ammo, health, armor, longjump, weaponboxes)
    int itemCount = 0;
    bool debugItems = cl_agent_debug_items && cl_agent_debug_items->value > 0.0f;
    for (int i = 1; i <= 512; i++)
    {
        if (i == localPlayer->index)
            continue;
        if (itemCount >= AGENT_MAX_ITEMS)
            break;

        cl_entity_t* ent = gEngfuncs.GetEntityByIndex(i);
        if (!ent || !ent->model || !ent->model->name[0])
            continue;

        // Skip invisible / nodraw entities
        if (ent->curstate.effects & 128) // EF_NODRAW = 128
            continue;

        const char* mName = ent->model->name;

        // Skip brush models (*), player, view/held weapon models
        if (mName[0] == '*') continue;
        if (std::strstr(mName, "models/v_")) continue;
        if (std::strstr(mName, "models/p_")) continue;
        if (std::strstr(mName, "models/player")) continue;

        // Match weapon/item world models - any path containing w_ or item_
        bool isItem = std::strstr(mName, "w_") != nullptr ||
                      std::strstr(mName, "item_") != nullptr ||
                      std::strstr(mName, "ammo") != nullptr ||
                      std::strstr(mName, "health") != nullptr ||
                      std::strstr(mName, "battery") != nullptr ||
                      std::strstr(mName, "longjump") != nullptr;

        float dx = ent->origin[0] - localPlayer->origin[0];
        float dy = ent->origin[1] - localPlayer->origin[1];
        float dz = ent->origin[2] - localPlayer->origin[2];
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        // Debug: print ALL non-brush, non-player entity model names in range
        if (debugItems && dist < 2000.0f)
            gEngfuncs.Con_Printf("[ItemScan] ent=%d model=%s dist=%.0f item=%d\n", i, mName, dist, (int)isItem);

        if (isItem && dist > 10.0f && dist < 3500.0f)
        {
            AgentItemInfo& item = g_pShm->observation.items[itemCount];
            item.active = 1;
            item.relative_pos[0] = dx;
            item.relative_pos[1] = dy;
            item.relative_pos[2] = dz;
            item.distance = dist;
            item.world_pos[0] = ent->origin[0];
            item.world_pos[1] = ent->origin[1];
            item.world_pos[2] = ent->origin[2];
            itemCount++;
        }
    }
    g_pShm->observation.num_items = itemCount;

    // Scan enemy player entities (1 to 32)
    int enemyCount = 0;
    for (int i = 1; i <= 32; i++)
    {
        if (i == localPlayer->index)
            continue;

        cl_entity_t* ent = gEngfuncs.GetEntityByIndex(i);
        if (!ent || !ent->player || ent->curstate.messagenum != localPlayer->curstate.messagenum)
            continue;

        if (enemyCount >= AGENT_MAX_ENEMIES)
            break;

        AgentEnemyInfo& info = g_pShm->observation.enemies[enemyCount];
        info.active = 1;
        info.index = i;
        info.origin[0] = ent->origin[0];
        info.origin[1] = ent->origin[1];
        info.origin[2] = ent->origin[2];

        float dx = ent->origin[0] - localPlayer->origin[0];
        float dy = ent->origin[1] - localPlayer->origin[1];
        float dz = ent->origin[2] - localPlayer->origin[2];

        info.relative_pos[0] = dx;
        info.relative_pos[1] = dy;
        info.relative_pos[2] = dz;
        info.distance = std::sqrt(dx * dx + dy * dy + dz * dz);

        enemyCount++;
    }

    g_pShm->observation.num_enemies = enemyCount;

    // OpenAG Native Strafe Analyzer Telemetry
    g_pShm->observation.strafe_efficiency = g_StrafeData.efficiency;
    g_pShm->observation.strafe_state = g_StrafeData.strafe_state;
    g_pShm->observation.on_ground = g_StrafeData.on_ground ? 1 : 0;
}

void AgentAPI_ApplyAction(void* pcmd_void, float* pviewangles)
{
    if (!AgentAPI_IsEnabled() || !g_pShm)
        return;

    usercmd_t* cmd = (usercmd_t*)pcmd_void;
    if (!cmd)
        return;

    // Freecam Flying Controller (cl_agent_cam == 2):
    // WASD + Space (Up) + Ctrl (Down) + Shift (Sprint) control Freecam movement in 3D world!
    if (cl_agent_cam && cl_agent_cam->value == 2.0f)
    {
        float mouse_va[3];
        gEngfuncs.GetViewAngles(mouse_va);

        float forward[3], right[3], up[3];
        gEngfuncs.pfnAngleVectors(mouse_va, forward, right, up);

        float fly_speed = 15.0f;
        if (cmd->buttons & IN_RUN) fly_speed = 35.0f;

        if (cmd->forwardmove > 50.0f)
        {
            g_freecamPos[0] += forward[0] * fly_speed;
            g_freecamPos[1] += forward[1] * fly_speed;
            g_freecamPos[2] += forward[2] * fly_speed;
        }
        else if (cmd->forwardmove < -50.0f)
        {
            g_freecamPos[0] -= forward[0] * fly_speed;
            g_freecamPos[1] -= forward[1] * fly_speed;
            g_freecamPos[2] -= forward[2] * fly_speed;
        }

        if (cmd->sidemove > 50.0f)
        {
            g_freecamPos[0] += right[0] * fly_speed;
            g_freecamPos[1] += right[1] * fly_speed;
            g_freecamPos[2] += right[2] * fly_speed;
        }
        else if (cmd->sidemove < -50.0f)
        {
            g_freecamPos[0] -= right[0] * fly_speed;
            g_freecamPos[1] -= right[1] * fly_speed;
            g_freecamPos[2] -= right[2] * fly_speed;
        }

        if (cmd->buttons & IN_JUMP)
        {
            g_freecamPos[2] += fly_speed;
        }
        if (cmd->buttons & IN_DUCK)
        {
            g_freecamPos[2] -= fly_speed;
        }

        cmd->forwardmove = 0.0f;
        cmd->sidemove = 0.0f;
        cmd->upmove = 0.0f;
        cmd->buttons &= ~(IN_FORWARD | IN_BACK | IN_MOVELEFT | IN_MOVERIGHT | IN_JUMP | IN_DUCK);
    }

    // Human Demonstration Recording logic
    if (g_DemoRecording && g_DemoFile.is_open() && g_pShm)
    {
        // Force autojump for human player on ground touch when Spacebar is held!
        static bool s_human_jump_held = false;
        if (cmd->buttons & IN_JUMP)
            s_human_jump_held = true;

        float va[3];
        gEngfuncs.GetViewAngles(va);
        float yaw_delta = va[1] - g_LastPlayerYaw;
        while (yaw_delta > 180.0f) yaw_delta -= 360.0f;
        while (yaw_delta < -180.0f) yaw_delta += 360.0f;
        g_LastPlayerYaw = va[1];

        float fwd_sel = (cmd->forwardmove > 50.0f) ? 1.0f : ((cmd->forwardmove < -50.0f) ? -1.0f : 0.0f);
        float side_sel = (cmd->sidemove > 50.0f) ? 1.0f : ((cmd->sidemove < -50.0f) ? -1.0f : 0.0f);

        g_DemoFile << fwd_sel << "," << side_sel << "," << yaw_delta;

        const auto& obs = g_pShm->observation;
        g_DemoFile << "," << obs.player_alive << "," << obs.player_origin[0] << "," << obs.player_origin[1] << "," << obs.player_origin[2];
        g_DemoFile << "," << g_AgentSimvel[0] << "," << g_AgentSimvel[1] << "," << g_AgentSimvel[2];
        float spd = std::sqrt(g_AgentSimvel[0]*g_AgentSimvel[0] + g_AgentSimvel[1]*g_AgentSimvel[1]);
        g_DemoFile << "," << spd << "," << obs.viewangles[0] << "," << obs.viewangles[1] << "," << obs.health << "," << obs.armor << "," << obs.clip_ammo << "," << obs.reserve_ammo;

        for (int i = 0; i < 8; i++) g_DemoFile << "," << obs.wall_distances[i];

        if (obs.goal_active) {
            float gdx = obs.goal_origin[0] - obs.player_origin[0];
            float gdy = obs.goal_origin[1] - obs.player_origin[1];
            float gdz = obs.goal_origin[2] - obs.player_origin[2];
            float gdist = std::sqrt(gdx*gdx + gdy*gdy + gdz*gdz);
            g_DemoFile << ",1.0," << gdx << "," << gdy << "," << gdz << "," << gdist;
        } else {
            g_DemoFile << ",0.0,0.0,0.0,0.0,0.0";
        }

        for (int i = 0; i < 5; i++) {
            if (i < obs.num_items && obs.items[i].active) {
                g_DemoFile << ",1.0," << obs.items[i].relative_pos[0] << "," << obs.items[i].relative_pos[1] << "," << obs.items[i].relative_pos[2] << "," << obs.items[i].distance;
            } else {
                g_DemoFile << ",0.0,0.0,0.0,0.0,0.0";
            }
        }

        for (int i = 0; i < 16; i++) {
            if (i < obs.num_enemies && obs.enemies[i].active) {
                g_DemoFile << ",1.0," << obs.enemies[i].relative_pos[0] << "," << obs.enemies[i].relative_pos[1] << "," << obs.enemies[i].relative_pos[2] << "," << obs.enemies[i].distance;
            } else {
                g_DemoFile << ",0.0,0.0,0.0,0.0,0.0";
            }
        }

        g_DemoFile << "\n";
        g_DemoSampleCount++;
        return;
    }

    // If recording human demonstration, DO NOT override player controls with Python agent!
    if (g_DemoRecording)
        return;

    AgentAction& act = g_pShm->action;

    // If in evaluation / telemetry-only mode, DO NOT override human player controls!
    if (act.eval_mode == 1)
        return;

    // Co-Op Mode (eval_mode == 2): AI ONLY controls buttons (+jump), human controls all movement (W/A/S/D and mouse)!
    if (act.eval_mode == 2)
    {
        if (act.buttons & 2)
        {
            bool is_in_air = (std::abs(g_AgentSimvel[2]) > 5.0f);
            if (is_in_air)
                cmd->buttons &= ~IN_JUMP;
            else
                cmd->buttons |= IN_JUMP;
        }
        else
        {
            cmd->buttons &= ~IN_JUMP;
        }
        return;
    }

    // Timeout check: If Python agent is not actively sending actions (idle > 300 frames = ~0.3 sec),
    // automatically release API override so human player has full control!
    uint32_t current_seq = g_pShm->observation.sequence;
    if (act.sequence_ack == 0 || (current_seq > act.sequence_ack && current_seq - act.sequence_ack > 300))
    {
        return;
    }

    // Auto-Strafe Synchronizer Engine:
    // Automatically synchronizes A/D strafe keys with agent yaw mouse turns!
    static uint32_t s_last_handled_seq = 0;
    static float s_current_yaw_delta = 0.0f;

    if (act.sequence_ack != s_last_handled_seq)
    {
        s_last_handled_seq = act.sequence_ack;
        s_current_yaw_delta = act.viewangles_delta[1];
    }

    float yaw_delta = s_current_yaw_delta;
    float spd = std::sqrt(g_AgentSimvel[0] * g_AgentSimvel[0] + g_AgentSimvel[1] * g_AgentSimvel[1]);

    cmd->forwardmove = 0.0f;
    cmd->buttons &= ~IN_FORWARD;

    if (act.sidemove != 0.0f)
    {
        cmd->sidemove = act.sidemove;
        if (act.sidemove < 0.0f) cmd->buttons |= IN_MOVELEFT;
        else if (act.sidemove > 0.0f) cmd->buttons |= IN_MOVERIGHT;
    }
    else if (yaw_delta > 0.05f)
    {
        cmd->sidemove = -400.0f;
        cmd->buttons |= IN_MOVELEFT;
    }
    else if (yaw_delta < -0.05f)
    {
        cmd->sidemove = 400.0f;
        cmd->buttons |= IN_MOVERIGHT;
    }
    else
    {
        cmd->sidemove = 0.0f;
    }

    if (act.upmove != 0.0f)
        cmd->upmove = act.upmove;

    // Apply buttons
    if (act.buttons & 1) // Attack
        cmd->buttons |= IN_ATTACK;

    // Perfect Air-Release Auto-Bhop for Agent:
    // In air: release IN_JUMP so GoldSrc engine resets button state.
    // On ground: press IN_JUMP to trigger immediate frame-perfect hop!
    if (act.buttons & 2)
    {
        bool is_in_air = (std::abs(g_AgentSimvel[2]) > 5.0f);
        if (is_in_air)
        {
            cmd->buttons &= ~IN_JUMP;
        }
        else
        {
            cmd->buttons |= IN_JUMP;
        }
    }

    if (act.buttons & 4) // Duck
        cmd->buttons |= IN_DUCK;
    if (act.buttons & 8) // Attack2
        cmd->buttons |= IN_ATTACK2;

    // Apply view angle deltas
    if (cl_agent_cam && cl_agent_cam->value > 0.0f)
    {
        if (s_AgentPlayerYaw == 0.0f && pviewangles)
        {
            s_AgentPlayerYaw = pviewangles[1];
        }

        // 3rd Person / Freecam Editor mode:
        // Update agent's internal heading s_AgentPlayerYaw from Python RL agent action
        if (act.viewangles_delta[1] != 0.0f)
        {
            s_AgentPlayerYaw += act.viewangles_delta[1];
            while (s_AgentPlayerYaw > 180.0f) s_AgentPlayerYaw -= 360.0f;
            while (s_AgentPlayerYaw < -180.0f) s_AgentPlayerYaw += 360.0f;
            act.viewangles_delta[1] = 0.0f;
        }

        // Set cmd->viewangles[1] (sent to server physics) to s_AgentPlayerYaw
        cmd->viewangles[0] = 0.0f;
        cmd->viewangles[1] = s_AgentPlayerYaw;
        cmd->viewangles[2] = 0.0f;

        // DO NOT overwrite pviewangles! Keep engine mouse look active so mouse camera turns smoothly.
    }
    else if (pviewangles)
    {
        // 1st Person mode: Direct view angle override
        pviewangles[0] = 0.0f;

        if (act.viewangles_delta[1] != 0.0f)
        {
            pviewangles[1] += act.viewangles_delta[1];

            // Wrap yaw (-180 to 180)
            while (pviewangles[1] > 180.0f) pviewangles[1] -= 360.0f;
            while (pviewangles[1] < -180.0f) pviewangles[1] += 360.0f;

            act.viewangles_delta[1] = 0.0f;
        }
        act.viewangles_delta[0] = 0.0f;

        s_AgentPlayerYaw = pviewangles[1];
        cmd->viewangles[0] = 0.0f;
        cmd->viewangles[1] = pviewangles[1];
        cmd->viewangles[2] = 0.0f;
    }
}

#include "../external/imgui/imgui.h"

void AgentAPI_RenderCamera(float* origin, float* angles)
{
    if (!cl_agent_cam || cl_agent_cam->value == 0.0f)
    {
        g_freecamInit = false;
        return;
    }

    cl_entity_t* localPlayer = gEngfuncs.GetLocalPlayer();
    if (!localPlayer || !gEngfuncs.pEventAPI)
        return;

    int mode = (int)cl_agent_cam->value;
    float mouse_va[3];
    gEngfuncs.GetViewAngles(mouse_va);

    if (mode == 1) // 3rd Person Chase Cam (Orbit with mouse)
    {
        float forward[3], right[3], up[3];
        gEngfuncs.pfnAngleVectors(mouse_va, forward, right, up);

        float p_start[3] = { localPlayer->origin[0], localPlayer->origin[1], localPlayer->origin[2] + 35.0f };
        float target_cam[3] = {
            p_start[0] - forward[0] * 140.0f,
            p_start[1] - forward[1] * 140.0f,
            p_start[2] - forward[2] * 140.0f + 20.0f
        };

        pmtrace_t tr;
        gEngfuncs.pEventAPI->EV_PlayerTrace(p_start, target_cam, PM_WORLD_ONLY, -1, &tr);

        origin[0] = tr.endpos[0];
        origin[1] = tr.endpos[1];
        origin[2] = tr.endpos[2];

        angles[0] = mouse_va[0];
        angles[1] = mouse_va[1];
        angles[2] = mouse_va[2];
    }
    else if (mode == 2) // Freecam / FlyCam Mode
    {
        if (!g_freecamInit)
        {
            g_freecamPos[0] = localPlayer->origin[0] - 100.0f;
            g_freecamPos[1] = localPlayer->origin[1] - 100.0f;
            g_freecamPos[2] = localPlayer->origin[2] + 120.0f;
            g_freecamInit = true;
        }

        origin[0] = g_freecamPos[0];
        origin[1] = g_freecamPos[1];
        origin[2] = g_freecamPos[2];

        angles[0] = mouse_va[0];
        angles[1] = mouse_va[1];
        angles[2] = mouse_va[2];
    }
}

extern bool g_ShowAGSettings;
extern bool g_ShowImGuiMenu;

void AgentAPI_RenderHUD()
{
    if (!AgentAPI_IsEnabled() || !g_pShm)
        return;

    cl_entity_t* localPlayer = gEngfuncs.GetLocalPlayer();
    const auto& obs = g_pShm->observation;
    const auto& act = g_pShm->action;

    // 1. Draw 3D Perception Overlay (Wall Rays, Velocity Vector & Target Item Line)
    if (localPlayer && gEngfuncs.pTriAPI)
    {
        ImDrawList* bg_draw = ImGui::GetBackgroundDrawList();
        float p_start[3] = { localPlayer->origin[0], localPlayer->origin[1], localPlayer->origin[2] + 16.0f };

        // Draw 3D Velocity Vector (Green Line on Ground/Screen)
        float vx = obs.player_velocity[0];
        float vy = obs.player_velocity[1];
        float speed = std::sqrt(vx * vx + vy * vy);
        if (speed > 50.0f)
        {
            float p_vel_end[3] = {
                p_start[0] + (vx / speed) * 80.0f,
                p_start[1] + (vy / speed) * 80.0f,
                p_start[2]
            };
            float s_pstart[3], s_pend[3];
            if (gEngfuncs.pTriAPI->WorldToScreen(p_start, s_pstart) == 0 &&
                gEngfuncs.pTriAPI->WorldToScreen(p_vel_end, s_pend) == 0)
            {
                ImVec2 pt_start(XPROJECT(s_pstart[0]), YPROJECT(s_pstart[1]));
                ImVec2 pt_end(XPROJECT(s_pend[0]), YPROJECT(s_pend[1]));
                bg_draw->AddLine(pt_start, pt_end, ImColor(0, 255, 120, 255), 4.0f);
                bg_draw->AddCircleFilled(pt_end, 5.0f, ImColor(0, 255, 120, 255));
            }
        }

        // Render 8-Directional Wall Rays
        for (int i = 0; i < 8; i++)
        {
            float dist = obs.wall_distances[i];
            float rel_deg = obs.viewangles[1] + (float)i * 45.0f;
            float rad = rel_deg * 3.14159265f / 180.0f;

            float dir_x = std::cos(rad);
            float dir_y = std::sin(rad);

            float p_ray_start[3] = {
                p_start[0] + dir_x * 30.0f,
                p_start[1] + dir_y * 30.0f,
                p_start[2]
            };

            float p_ray_end[3] = {
                p_start[0] + dir_x * dist,
                p_start[1] + dir_y * dist,
                p_start[2]
            };

            float s_start[3], s_end[3];
            if (gEngfuncs.pTriAPI->WorldToScreen(p_ray_start, s_start) == 0 &&
                gEngfuncs.pTriAPI->WorldToScreen(p_ray_end, s_end) == 0)
            {
                ImVec2 pt1(XPROJECT(s_start[0]), YPROJECT(s_start[1]));
                ImVec2 pt2(XPROJECT(s_end[0]), YPROJECT(s_end[1]));

                ImU32 rayColor = ImColor(50, 220, 80, 220);
                if (dist < 150.0f) rayColor = ImColor(230, 40, 40, 240);
                else if (dist < 400.0f) rayColor = ImColor(240, 210, 40, 220);

                bg_draw->AddLine(pt1, pt2, rayColor, (i == 0) ? 3.5f : 2.0f);
            }
        }

        // Render ALL detected item beacons (nearest = cyan target, others = orange)
        for (int j = 0; j < obs.num_items; j++)
        {
            if (!obs.items[j].active)
                continue;

            const auto& item = obs.items[j];

            // Use stored absolute world position — not relative to avoid frame drift
            float item_wpos[3] = { item.world_pos[0], item.world_pos[1], item.world_pos[2] };

            float s_item[3];
            if (gEngfuncs.pTriAPI->WorldToScreen(item_wpos, s_item) != 0)
                continue;

            ImVec2 pt_it(XPROJECT(s_item[0]), YPROJECT(s_item[1]));

            bool isNearest = (j == 0);
            ImU32 dotColor   = isNearest ? ImColor(0, 230, 255, 255) : ImColor(255, 160, 30, 255);
            ImU32 ringColor  = isNearest ? ImColor(255, 255, 255, 220) : ImColor(255, 160, 30, 180);
            ImU32 textColor  = isNearest ? ImColor(0, 230, 255, 255)   : ImColor(255, 200, 80, 255);

            if (isNearest)
            {
                float s_player[3];
                if (gEngfuncs.pTriAPI->WorldToScreen(p_start, s_player) == 0)
                {
                    ImVec2 pt_p(XPROJECT(s_player[0]), YPROJECT(s_player[1]));
                    bg_draw->AddLine(pt_p, pt_it, ImColor(0, 230, 255, 180), 2.5f);
                }
            }

            bg_draw->AddCircleFilled(pt_it, isNearest ? 7.0f : 4.0f, dotColor);
            bg_draw->AddCircle(pt_it, isNearest ? 14.0f : 9.0f, ringColor, 12, 1.5f);

            char tag[64];
            snprintf(tag, sizeof(tag), "%s%.0fu", isNearest ? "[TARGET] " : "", item.distance);
            bg_draw->AddText(ImVec2(pt_it.x - 30, pt_it.y - 22), textColor, tag);
        }

        // Render Interactive Waypoint Circuit Targets & Path Lines
        if (!g_Waypoints.empty())
        {
            for (size_t i = 0; i < g_Waypoints.size(); i++)
            {
                bool is_current = ((int)i == g_CurrentWaypointIdx);
                float s_wp[3];
                bool wp_on_screen = (gEngfuncs.pTriAPI->WorldToScreen(g_Waypoints[i].origin, s_wp) == 0);

                if (wp_on_screen)
                {
                    ImVec2 pt_wp(XPROJECT(s_wp[0]), YPROJECT(s_wp[1]));
                    ImU32 wpColor = is_current ? ImColor(255, 0, 200, 255) : ImColor(0, 220, 255, 200);

                    // Render 3D Wireframe Volume Box Trigger around origin
                    float ox = g_Waypoints[i].origin[0];
                    float oy = g_Waypoints[i].origin[1];
                    float oz = g_Waypoints[i].origin[2];
                    float rx = g_TriggerRadiusX;
                    float ry = g_TriggerRadiusY;
                    float rz = g_TriggerRadiusZ;

                    float corners[8][3] = {
                        { ox - rx, oy - ry, oz - rz },
                        { ox + rx, oy - ry, oz - rz },
                        { ox + rx, oy + ry, oz - rz },
                        { ox - rx, oy + ry, oz - rz },
                        { ox - rx, oy - ry, oz + rz },
                        { ox + rx, oy - ry, oz + rz },
                        { ox + rx, oy + ry, oz + rz },
                        { ox - rx, oy + ry, oz + rz }
                    };

                    ImVec2 sc[8];
                    bool sc_valid[8];
                    for (int c = 0; c < 8; c++) {
                        float scr[3];
                        sc_valid[c] = (gEngfuncs.pTriAPI->WorldToScreen(corners[c], scr) == 0);
                        sc[c] = ImVec2(XPROJECT(scr[0]), YPROJECT(scr[1]));
                    }

                    int edges[12][2] = {
                        {0,1}, {1,2}, {2,3}, {3,0}, // Bottom face
                        {4,5}, {5,6}, {6,7}, {7,4}, // Top face
                        {0,4}, {1,5}, {2,6}, {3,7}  // Vertical edges
                    };

                    ImU32 boxColor = is_current ? ImColor(255, 0, 200, 220) : ImColor(0, 220, 255, 180);
                    for (int e = 0; e < 12; e++) {
                        int idxA = edges[e][0];
                        int idxB = edges[e][1];
                        if (sc_valid[idxA] && sc_valid[idxB]) {
                            bg_draw->AddLine(sc[idxA], sc[idxB], boxColor, is_current ? 2.5f : 1.5f);
                        }
                    }

                    if (is_current)
                    {
                        float s_player[3];
                        if (gEngfuncs.pTriAPI->WorldToScreen(p_start, s_player) == 0)
                        {
                            ImVec2 pt_p(XPROJECT(s_player[0]), YPROJECT(s_player[1]));
                            bg_draw->AddLine(pt_p, pt_wp, ImColor(255, 0, 200, 220), 3.0f);
                        }
                    }

                    bg_draw->AddCircleFilled(pt_wp, is_current ? 6.0f : 4.0f, wpColor);
                    bg_draw->AddCircle(pt_wp, is_current ? 12.0f : 8.0f, ImColor(255, 255, 255, 240), 16, 1.5f);

                    float dx = g_Waypoints[i].origin[0] - localPlayer->origin[0];
                    float dy = g_Waypoints[i].origin[1] - localPlayer->origin[1];
                    float dz = g_Waypoints[i].origin[2] - localPlayer->origin[2];
                    float wp_dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                    char tag[64];
                    if (is_current)
                        snprintf(tag, sizeof(tag), "[3D TRIGGER #%d: %.0fu]", (int)i + 1, wp_dist);
                    else
                        snprintf(tag, sizeof(tag), "[TRIGGER #%d: %.0fu]", (int)i + 1, wp_dist);

                    bg_draw->AddText(ImVec2(pt_wp.x - 40, pt_wp.y - 28), wpColor, tag);
                }

                // Render connecting line from this waypoint to the next waypoint in circuit
                if (g_Waypoints.size() > 1)
                {
                    size_t next_idx = (i + 1) % g_Waypoints.size();
                    float s_next[3];
                    bool next_on_screen = (gEngfuncs.pTriAPI->WorldToScreen(g_Waypoints[next_idx].origin, s_next) == 0);
                    if (wp_on_screen && next_on_screen)
                    {
                        ImVec2 pt_a(XPROJECT(s_wp[0]), YPROJECT(s_wp[1]));
                        ImVec2 pt_b(XPROJECT(s_next[0]), YPROJECT(s_next[1]));
                        bg_draw->AddLine(pt_a, pt_b, ImColor(255, 255, 0, 160), 2.0f);
                    }
                }
            }
        }
    }

    // 2. Render Telemetry & Editor HUD Window
    ImGuiWindowFlags hud_flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings;
    bool interactive = g_ShowAGSettings || g_ShowImGuiMenu;
    if (!interactive)
    {
        hud_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs;
    }

    ImGui::SetNextWindowPos(ImVec2(16.0f, 60.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360.0f, 320.0f), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("AI Agent 3D Inspector & Editor", nullptr, hud_flags))
    {
        if (g_DemoRecording)
            ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), "[DEMO RECORDING ACTIVE: %d samples]", g_DemoSampleCount);
        else
            ImGui::TextColored(ImVec4(0.2f, 0.9f, 0.3f, 1.0f), "[AI Agent API: ACTIVE]");

        if (interactive)
        {
            ImGui::SeparatorText("Camera View Mode");
            int cur_cam = cl_agent_cam ? (int)cl_agent_cam->value : 0;
            if (ImGui::RadioButton("1st Person", cur_cam == 0)) { if (cl_agent_cam) cl_agent_cam->value = 0.0f; } ImGui::SameLine();
            if (ImGui::RadioButton("3rd Person Chase", cur_cam == 1)) { if (cl_agent_cam) cl_agent_cam->value = 1.0f; } ImGui::SameLine();
            if (ImGui::RadioButton("Freecam", cur_cam == 2)) { if (cl_agent_cam) cl_agent_cam->value = 2.0f; }

            ImGui::SeparatorText("3D Interactive Goal Setter");
            if (ImGui::Button("🎯 Set Goal at Crosshair")) Cmd_EditorMouseClick();
            ImGui::SameLine();
            if (ImGui::Button("❌ Clear Goal")) Cmd_ClearGoal();
        }

        ImGui::Separator();

        float vx = obs.player_velocity[0];
        float vy = obs.player_velocity[1];
        float speed = std::sqrt(vx * vx + vy * vy);

        ImGui::Text("Sequence: #%u", obs.sequence);
        ImGui::Text("Player Speed: %.1f u/s", speed);
        ImGui::Text("Health: %d | Armor: %d", obs.health, obs.armor);
        ImGui::Text("ViewAngles: P %.1f° | Y %.1f°", obs.viewangles[0], obs.viewangles[1]);

        ImGui::SeparatorText("Perception & Sensors");
        ImGui::Text("Front Wall Distance: %.0f u", obs.wall_distances[0]);
        if (obs.num_items > 0 && obs.items[0].active)
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 1.0f, 1.0f), "Target Loot: %.0f u", obs.items[0].distance);

        ImGui::SeparatorText("📍 3D Volume Box Trigger & Circuit Manager");
        ImGui::Text("Triggers in Loop: %d | Current Target: #%d", (int)g_Waypoints.size(), g_Waypoints.empty() ? 0 : g_CurrentWaypointIdx + 1);

        if (interactive)
        {
            ImGui::SliderFloat("Trigger Size X", &g_TriggerRadiusX, 40.0f, 400.0f, "%.0f u");
            ImGui::SliderFloat("Trigger Size Y", &g_TriggerRadiusY, 40.0f, 400.0f, "%.0f u");
            ImGui::SliderFloat("Trigger Size Z", &g_TriggerRadiusZ, 40.0f, 400.0f, "%.0f u");
        }

        if (g_TotalLapsCompleted > 0)
        {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), "🏁 Last Lap: %.2f s | Best Lap: %.2f s", g_LastLapTime, g_BestLapTime);
            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.0f, 1.0f), "🏆 Total Laps Completed: %d", g_TotalLapsCompleted);
        }
        else
        {
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "⏱ Circuit Race Timer Active");
        }

        if (ImGui::Button("➕ Add Trigger (Crosshair)"))
        {
            Cmd_SetGoal();
        }
        ImGui::SameLine();
        if (ImGui::Button("❌ Clear All"))
        {
            Cmd_ClearGoal();
        }
        ImGui::SameLine();
        if (ImGui::Button("↩ Pop Last"))
        {
            Cmd_PopGoal();
        }

        ImGui::SeparatorText("Live RL Rewards & Telemetry");
        if (act.last_reward >= 0.0f)
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.3f, 1.0f), "Reward: +%.2f", act.last_reward);
        else
            ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), "Reward: %.2f", act.last_reward);

        if (act.reward_reason[0] != '\0')
            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f), "Reason: %s", act.reward_reason);
        else
            ImGui::Text("Reason: [Exploring]");

        ImGui::SeparatorText("Neural Network Metrics & Loss");
        ImGui::Text("Policy Loss:   %.6f", act.loss_policy);
        ImGui::Text("Value Loss:    %.6f", act.loss_value);
        ImGui::Text("Entropy Loss:  %.6f", act.entropy);
        ImGui::Text("Learning Rate: %.6f", act.learning_rate);
        ImGui::TextColored(ImVec4(0.0f, 0.9f, 1.0f, 1.0f), "Train Speed:   %.0f steps/s", act.system_fps);

        ImGui::SeparatorText("Agent Inputs");
        ImGui::Text("Move: Fwd %.0f | Side %.0f", act.forwardmove, act.sidemove);

        char btnStr[128] = "";
        if (act.buttons & 1) strcat(btnStr, "[ATTACK] ");
        if (act.buttons & 2) strcat(btnStr, "[JUMP] ");
        if (act.buttons & 4) strcat(btnStr, "[DUCK] ");
        if (act.buttons & 8) strcat(btnStr, "[ATTACK2] ");
        if (strlen(btnStr) == 0) strcpy(btnStr, "[NONE]");

        ImGui::Text("Buttons: %s", btnStr);
    }
    ImGui::End();

    // 3. Render Accumulated Rewards Breakdown Panel (Bottom Telemetry Window)
    ImGui::SetNextWindowPos(ImVec2(16.0f, 400.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380.0f, 220.0f), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("RL Episode Reward Telemetry", nullptr, hud_flags))
    {
        ImGui::TextColored(ImVec4(0.0f, 0.9f, 1.0f, 1.0f), "Cumulative Component Totals (Current Episode):");
        ImGui::Separator();

        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.3f, 1.0f), "🟢 Directional Speed (V_goal): %+6.2f", act.reward_speed);
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "🟢 Air-Strafe Sync (R_sync):   %+6.2f", act.reward_scurve);
        ImGui::TextColored(ImVec4(0.9f, 0.2f, 0.9f, 1.0f), "🟣 Goal Progress & Reach:      %+6.2f", act.reward_goal);

        if (act.reward_wall < 0.0f)
            ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), "🔴 Obstacle & Wall Penalty:    %6.2f", act.reward_wall);
        else
            ImGui::Text("⚪ Obstacle & Wall Penalty:      0.00");

        ImGui::Separator();
        float total_cum = act.reward_speed + act.reward_scurve + act.reward_wall + act.reward_goal;
        if (total_cum >= 0.0f)
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.3f, 1.0f), "TOTAL EPISODE BALANCE:         %+6.2f", total_cum);
        else
            ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), "TOTAL EPISODE BALANCE:         %6.2f", total_cum);
    }
    ImGui::End();

    // 4. Render ML Training Metrics Panel (Loss & Policy Telemetry Window)
    ImGui::SetNextWindowPos(ImVec2(16.0f, 650.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360.0f, 160.0f), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("ML Neural Net Telemetry (Loss & Stats)", nullptr, hud_flags))
    {
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f), "PPO Loss & Neural Network Metrics:");
        ImGui::Separator();
        ImGui::Text("Policy Loss:   %.6f", act.loss_policy);
        ImGui::Text("Value Loss:    %.6f", act.loss_value);
        ImGui::Text("Entropy Loss:  %.6f", act.entropy);
        ImGui::Text("Learning Rate: %.6f", act.learning_rate);
        ImGui::TextColored(ImVec4(0.0f, 0.9f, 1.0f, 1.0f), "Train Speed:   %.0f steps/s", act.system_fps);
    }
    ImGui::End();
}
