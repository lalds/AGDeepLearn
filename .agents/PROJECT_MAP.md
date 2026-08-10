# OpenAG: Developer & Agent Codebase Guide

This document serves as a comprehensive architectural map and technical guide for OpenAG, designed to help incoming AI agents and developers quickly understand the project layout, key subsystems, custom additions, and code integration patterns.

---

## 1. Project Overview
**OpenAG** is a modern client/server mod for Half-Life (GoldSrc engine) based on the Adrenaline Gamer (AG) ruleset. It implements modern HUD systems via Dear ImGui, modern networking, Discord integration, and advanced rendering enhancements (e.g., motion blur, custom crosshairs, spawns display).

---

## 2. Directory Structure & Key Subsystems

- **/cl_dll/**: The Client-side library (compiles to `client.so` on Linux, `client.dll` on Windows). Includes `agent_api.cpp` / `agent_api.h` for POSIX Shared Memory interface with AGDeepLearn.
- **/dlls/**: The Server-side library (compiles to `hl.so`/`hl.dll`). Manages game rules, player logic, weapons physics, and server-side commands.
- **/game_shared/** & **/pm_shared/**: Shared logic between client and server (movement prediction physics, weapon characteristics, player states).
- **/external/**: External libraries compiled statically or linked (Dear ImGui, SDL2, Discord RPC, curl).
- **/build/**: Active Linux compilation directory.
- **/home/angel/AGDeepLearn/**: Companion Deep Learning / Reinforcement Learning repository (Python Gymnasium, Stable-Baselines3, PPO). Communicates with OpenAG's `agent_api` via POSIX Shared Memory (`/dev/shm/openag_agent_shm`).

---

## 3. Core Subsystem Architectures & Entry Points

### A. UI and Dear ImGui Integration
- **Main Helper**: [imgui_helper.cpp](file:///home/angel/OpenAG/cl_dll/imgui_helper.cpp)
  - Manages the lifecycle of the ImGui context, event handling (mouse/keyboard input capturing), style definitions (custom dark glassmorphic theme), and rendering of overlays.
  - **Key Variables**: `g_ShowImGuiMenu`, `g_ShowCrosshairEditor`, `g_ShowAGSettings`, `g_ShowPauseMenu`, `g_ShowServerBrowser`.
- **Initialization & Input Injection**: [cdll_int.cpp](file:///home/angel/OpenAG/cl_dll/cdll_int.cpp)
  - Hooks into engine initialization, frame redraw loops, and translates SDL2 events to ImGui input events.
  - Manages engine `GameUI` state changes to toggle the cursor and input focus.

### B. Custom Rendering Hooks
- **View Setup & Viewmodel**: [view.cpp](file:///home/angel/OpenAG/cl_dll/view.cpp)
  - Calculates viewangles, vieworg (position), viewmodel (weapon models) offset coordinates, and triggers weapon sway and motion blur calculations.
- **Model Rendering**: [StudioModelRenderer.cpp](file:///home/angel/OpenAG/cl_dll/StudioModelRenderer.cpp)
  - Overrides standard GoldSrc studio model drawing to adjust viewmodel FOV (`cl_viewmodel_fov`).
- **HUD Frame Redraw**: [hud_redraw.cpp](file:///home/angel/OpenAG/cl_dll/hud_redraw.cpp)
  - Main client-side drawing loop. Calls individual HUD elements and invokes ImGui new-frame routines.

---

## 4. Technical Details of Custom Additions

### 1. Unified Post-Processing Pipeline
- **Files**: [post_process.cpp](file:///home/angel/OpenAG/cl_dll/post_process.cpp) & [post_process.h](file:///home/angel/OpenAG/cl_dll/post_process.h)
- **Cvars**: 
  - Bloom: `cl_bloom`, `cl_bloom_intensity`, `cl_bloom_radius`, `cl_bloom_threshold`, `cl_bloom_passes`.
  - Motion Blur: `cl_motion_blur`, `cl_motion_blur_shutter`, `cl_motion_blur_alpha`, `cl_motion_blur_max`, `cl_motion_blur_chromatic`.
  - Vignette: `cl_vignette`, `cl_vignette_strength`, `cl_vignette_radius`, `cl_vignette_softness`.
  - Film Grain: `cl_film_grain`, `cl_film_grain_amount`.
  - SSAO (Screen Space Ambient Occlusion): `cl_ssao`, `cl_ssao_radius`, `cl_ssao_intensity` *(Stage: Depth Capture Diagnostics)*.
- **Architecture**:
  - Implements a unified render-pass stack (SSAO → Bloom → Motion Blur → Vignette → Film Grain) using fixed-function OpenGL 2.x.
  - Captures the clean 3D scene framebuffer exactly *once* per frame into `g_TexScreen` to prevent feedback loops/exponential ghosting buildup.
  - **SSAO (Screen Space Ambient Occlusion) [Current Stage: Depth Diagnostics]**: 
    - **Status**: SSAO shader exists but is blocked on reliable depth buffer capture. Currently in diagnostic mode to determine which GL depth-read method works on the target driver.
    - **Depth Capture (`capture_depth()` in `post_process.cpp`)**: Called from `HUD_DrawTransparentTriangles` (in `tri.cpp`). **NOT** `HUD_DrawNormalTriangles` — by that point the engine has already cleared the depth buffer (all values read as 1.0). The transparent pass runs earlier in the 3D render while scene depth is still populated. `glReadPixels(GL_FLOAT)` works on this driver (confirmed via debug panel, no GL error), but returns 1.0 everywhere from the Normal hook — moved to Transparent hook to get real geometry depth values.
    - **ImGui Debug Panel**: Located in AG Settings → Graphics / Post-Processing → "SSAO (Depth Debug)". Shows capture method, GL errors, center depth value, and 3×3 depth grid around screen center.
    - **Known Issue**: GoldSrc's default framebuffer state causes `glReadPixels(GL_DEPTH_COMPONENT)` to fail with `GL_INVALID_OPERATION` (0x502) on some drivers. The diagnostic panel identifies which method works to unblock SSAO shader development.
    - **Critical GL State Rule**: `capture_depth()` must NOT call `InitTextures()` or modify engine GL state (texture bindings, shaders, FBOs) — it runs mid-3D-render-pass. All texture bindings are saved/restored via `glGetIntegerv(GL_TEXTURE_BINDING_2D)`.
  - **Bloom**: Downsamples scene to 512x512, applies multiplicative thresholding to isolate highlights, performs iterative Gaussian-like ping-pong blur, and blends additively. Overwritten bottom-left screen regions are cleanly restored from `g_TexScreen`.
  - **Motion Blur**: Blends 6 shifted transparent copies of `g_TexScreen` along the camera's angular velocity vector ($\Delta\text{Pitch}$, $\Delta\text{Yaw}$) calculated from frame time delta. Features optional chromatic aberration (RGB split).
  - **Vignette**: Renders a circular aspect-corrected darkening overlay using a smooth $100 \times 100$ quad vertex grid with per-vertex alpha calculated via `smoothstep`.
  - **Film Grain**: Renders animated screen noise at 1/4 pixel resolution using a fast LCG randomizer.
  - **OpenGL Safety**: Employs attributes pushing/popping (`glPushAttrib(GL_ALL_ATTRIB_BITS)`) to avoid corrupting engine state.

### 2. Speedometer, Strafe Analyzer & Key Overlay
- **Files**: [hud_strafeguide.h](file:///home/angel/OpenAG/cl_dll/hud_strafeguide.h), [hud_strafeguide.cpp](file:///home/angel/OpenAG/cl_dll/hud_strafeguide.cpp), & [imgui_helper.cpp](file:///home/angel/OpenAG/cl_dll/imgui_helper.cpp)
- **Cvars**: `cl_speedometer`, `cl_speedometer_show_strafes`.
- **Architecture**:
  - `CHudStrafeGuide::Update` acts as the data collector. It reads prediction velocities from `ref_params_s` and computes current horizontal speed and frame acceleration.
  - Calculates mathematically optimal strafe angle $\theta_{opt} = \arccos(30.0 / \text{speed})$.
  - Measures the actual angle between velocity and input direction to categorize player performance:
    - `1 (PERFECT)`: Within $\sim 3^{\circ}$ of $\theta_{opt}$.
    - `2 (TURN FASTER)`: Turning angle is narrower than optimal.
    - `3 (TURN SLOWER)`: Turning angle is wider than optimal.
    - `4 (SPEED LOSS)`: Extremely wide turning causing deceleration.
    - `5 (RELEASE W/S)`: Warning state when pressing forward/back keys in mid-air.
  - Data is stored in a globally visible structure `strafe_data_t g_StrafeData`.
  - ImGui renders this data in a glassmorphic bottom-center HUD panel.

### 3. Viewmodel Position Offsets & FOV Customizations
- **Files**: [view.cpp](file:///home/angel/OpenAG/cl_dll/view.cpp), [StudioModelRenderer.cpp](file:///home/angel/OpenAG/cl_dll/StudioModelRenderer.cpp), & [imgui_helper.cpp](file:///home/angel/OpenAG/cl_dll/imgui_helper.cpp)
- **Cvars**: `cl_viewmodel_ofs_right` (X), `cl_viewmodel_ofs_forward` (Y), `cl_viewmodel_ofs_up` (Z), `cl_viewmodel_fov` (Viewmodel FOV).
- **Architecture**:
  - `V_CalcRefdef` (in `view.cpp`) reads these CVar values and applies them to the viewmodel entity's origin along the view's local coordinate vectors (`pparams->right`, `pparams->forward`, `pparams->up`).
  - `CStudioModelRenderer::SetViewmodelFovProjection` (in `StudioModelRenderer.cpp`) dynamically scales projection matrices for weapon models during rendering when `cl_viewmodel_fov` is non-zero.
  - Added full configuration sliders to the "FOV & Viewmodel Settings" ImGui panel.

### 4. Gauss Predicted Damage Numbers
- **Files**: [ev_hldm.cpp](file:///home/angel/OpenAG/cl_dll/ev_hldm.cpp) & [damage_numbers.cpp](file:///home/angel/OpenAG/cl_dll/damage_numbers.cpp)
- **Architecture**:
  - Implements instant client-side predicted damage numbers display for the Gauss gun (Tau Cannon) inside `EV_FireGauss` when hit traces strike a player entity.
  - Correctly calls `EV_GetPhysent` *before* `EV_PopPMStates()` so that player physics entities are preserved when checking hit targets during prediction on remote servers.
  - Differentiates between primary fire (`20` damage) and charged secondary fire (`flDamage` up to `200`). Utilizes a `bShowedDamage` check per frame trace to avoid double-displaying damage numbers when the piercing beam loops through a player model.
  - Clears prediction state correctly on map/server change using `damage_numbers::reset()`.

### 5. Persistent Projectile Owner Damage Attribution (Server)
- **Files**: [weapons.h](file:///home/angel/OpenAG/dlls/weapons.h), [rpg.cpp](file:///home/angel/OpenAG/dlls/rpg.cpp), [ggrenade.cpp](file:///home/angel/OpenAG/dlls/ggrenade.cpp), [crossbow.cpp](file:///home/angel/OpenAG/dlls/crossbow.cpp), [satchel.cpp](file:///home/angel/OpenAG/dlls/satchel.cpp)
- **Architecture**:
  - Fixes the classic GoldSrc engine bug where physical projectile weapons (RPG Rockets, MP5 Grenades, Satchels, Hand Grenades, Crossbow Bolts) occasionally fail to attribute damage or kill credits to the attacker because the engine clears `pev->owner` upon direct collision impacts.
  - Introduces `EHANDLE m_hThrower` inside the base `CGrenade` class to store a persistent reference to the launching player. All damage-inflicting explosion routines retrieve the attacker from `m_hThrower` instead of `pev->owner`.
  - All EHANDLE logic uses preprocessor checks `#ifndef CLIENT_DLL` to ensure smooth compilation in both the server library (`hl.so`) and client prediction routines (`client.so`).

### 6. Spawn Points Display System
- **Files**: [spawns.cpp](file:///home/angel/OpenAG/cl_dll/spawns.cpp), [spawns.h](file:///home/angel/OpenAG/cl_dll/spawns.h)
- **Cvars**: `cl_show_spawns`, `cl_show_spawns_only_active`, `cl_show_spawns_dist`.
- **Architecture**:
  - Parses the `.bsp` map file directly from disk upon level load to locate entity lump entries for `info_player_deathmatch`.
  - Dynamically updates active player coordinates to highlight active and cooldown spawn points.
  - Renders 3D boxes/positions at coordinates in the game world utilizing OpenGL drawing pipelines (`gEngfuncs.pTriangleAPI`).

### 7. HL2-style Weapon Sway (Bobbing)
- **Files**: [weapon_sway.cpp](file:///home/angel/OpenAG/cl_dll/weapon_sway.cpp), [weapon_sway.h](file:///home/angel/OpenAG/cl_dll/weapon_sway.h)
- **Cvars**: `cl_weapon_sway`, `cl_weapon_sway_pitch`, `cl_weapon_sway_yaw`, `cl_weapon_sway_roll`, `cl_weapon_sway_clamp`, `cl_weapon_sway_roll_factor`, `cl_weapon_sway_pos_yaw`, `cl_weapon_sway_pos_pitch`.
- **Architecture**:
  - Interpolates and smooths weapon model alignment angles relative to the player's view rotation rates.
  - Simulates dynamic weapon drag, roll rotation, and sway displacement.

### 8. AG Online Player Count Overlay
- **Files**: [imgui_helper.cpp](file:///home/angel/OpenAG/cl_dll/imgui_helper.cpp)
- **Architecture**:
  - Designed to run when the player is in the main menu (`g_pGameUI->IsGameUIVisible()`).
  - Triggers a background server query via `g_pServers->RequestList()` every 30 seconds.
  - Loops over the returned server list, parsing the `"current"` key from each server's info string, and displays the sum in a sleek top-right overlay window.

---

## 5. Development & Verification Guide

### Building
The project compiles via CMake. On Linux, run from the root:
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Deployment (Copy Path)
The compiled binary `client.so` must be copied to the mod's `cl_dlls` directory:
```bash
cp build/client.so /mnt/Trash/@home/angeloo123/Games/steamapps/common/Half-Life/ag/cl_dlls/
```

### Coding Rules for AI Agents
1. **Preserve Engine State**: GoldSrc uses a single-threaded OpenGL pipeline. Whenever writing raw OpenGL commands (like inside `motion_blur.cpp`), always push current state matrices and client attributes, and pop them immediately after finishing.
2. **CVar Lookups**: All CVars registered by other files must be accessed using `gEngfuncs.pfnGetCvarPointer("cvar_name")`. Ensure pointers are null-checked before accessing `->value` or `->string`.
3. **ImGui HUD Rendering**: All HUD overlays (speedometer, online player count, health/armor) must use `hud_flags` containing `ImGuiWindowFlags_NoInputs` to prevent blocking mouse look and movements in-game.

---

## 6. SSAO Depth Capture Troubleshooting Context (For External AI Analysis)

If you are feeding this project to an external AI to solve the SSAO depth capture issue, here is the exact context of the problem:

### The Goal
We want to extract the 3D scene depth buffer (Z-buffer) in a client-side OpenGL mod for Half-Life (GoldSrc engine) to feed into a custom Screen Space Ambient Occlusion (SSAO) shader.

### The Problem
Currently, any attempt to read depth via `glReadPixels(..., GL_DEPTH_COMPONENT, ...)` or GPU copy via `glCopyTexSubImage2D` returns `1.0` (far plane / cleared depth) across the entire screen, even when looking directly at walls, players, or nearby geometry. 

### What We Have Tried & Discovered
1. **Calling `capture_depth()` in `HUD_DrawNormalTriangles`**:
   - Depth is always `1.0`.
   - Reason: The engine has already finished rendering the 3D scene and cleared the depth buffer before drawing 2D HUD elements and overlays.
2. **Calling `capture_depth()` in `HUD_DrawTransparentTriangles` (in `tri.cpp`)**:
   - This hook runs earlier during the 3D scene rendering pass.
   - However, depth still reads as `1.0` everywhere. 
   - No OpenGL errors (like `GL_INVALID_OPERATION` 0x502) are triggered. The driver successfully executes the read, but the buffer itself only contains `1.0`.
3. **OpenGL Driver Diagnostics**:
   - We implemented a diagnostic panel in ImGui ("SSAO (Depth Debug)").
   - It verifies that `glReadPixels(..., GL_DEPTH_COMPONENT, GL_FLOAT, ...)` executes successfully (returns `GL_NO_ERROR`), but the depth values in the 3x3 grid around the center are always exactly `1.0000`.

### Key Questions to Solve
- Why is the depth buffer already cleared/empty (filled with `1.0`) during `HUD_DrawTransparentTriangles`?
- Is there a more appropriate rendering hook in the GoldSrc client library (e.g., in `StudioModelRenderer.cpp`, `view.cpp` / `V_CalcRefdef`, or custom tri API hooks) where the depth buffer still holds the opaque world/player geometry?
- Is the engine rendering to a custom Framebuffer Object (FBO) or a separate context/renderbuffer, and if so, how can we bind/query it to read its depth attachment?
- Does the engine disable depth writes or change the depth function (`glDepthMask`, `glDepthFunc`) during certain passes, causing the read to fail or return default cleared values?

### PROBLEM SOLVED!

## 6. Resolved Challenges: SSAO & MSAA Depth Capture

*Context for AI agents reviewing Git history regarding early SSAO iterations.*

**The Problem**: Early attempts to read the depth buffer returned `1.0` or triggered `GL_INVALID_OPERATION` (0x502).
**The Cause**: The 25th Anniversary Half-Life update utilizes an MSAA (Multisample Anti-Aliasing) Framebuffer. OpenGL prohibits direct pixel reading (`glReadPixels`) or direct GPU texture sampling from multisampled FBOs.
**The Solution**: 
We now generate a custom single-sampled FBO (`g_DepthBlitFBO`). During `HUD_DrawTransparentTriangles` (when scene geometry is fully rendered but the depth buffer is not yet cleared by the viewmodel pass), we hardware-blit the engine's MSAA depth buffer into our custom FBO via `glBlitFramebufferEXT`. This allows standard shaders to sample `g_TexDepth` without memory-bandwidth bottlenecks or GL errors.