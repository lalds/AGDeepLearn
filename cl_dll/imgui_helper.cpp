#include "imgui_helper.h"
#include "imgui.h"
#include "backends/imgui_impl_sdl2.h"
#include "backends/imgui_impl_opengl2.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include "hud.h"
#include "cl_util.h"
#include "keydefs.h"
#include "in_buttons.h"
#include "post_process.h"
#include "ammo.h"
#include "ammohistory.h"
#include "gameui.h"
#include "hud_servers_priv.h"
#include "agent_api.h"

extern CHudServers *g_pServers;

extern "C" void HUD_UpdateCursorState();

unsigned int g_dwKeyPressButtons = 0;
static cvar_t* cl_keypress_overlay = nullptr;
cvar_t* cl_custom_hud = nullptr;
cvar_t* cl_custom_menu = nullptr;

bool g_ShowImGuiMenu = false;
bool g_ShowCrosshairEditor = false;
bool g_ShowAGSettings = false;
bool g_ShowPauseMenu = false;
bool g_ShowMainMenu = false;
bool g_ShowServerBrowser = false;
bool g_IsBackgroundMap = true;
bool g_FilterAGOnly = true;
bool g_ShowMatchmaking = false;

// Matchmaking state
enum MatchmakingState { MM_IDLE, MM_SEARCHING, MM_FOUND, MM_CONNECTING };
static MatchmakingState g_MMState = MM_IDLE;
static char g_MMBestAddress[128] = "";
static char g_MMBestName[128]    = "";
static char g_MMBestMap[64]      = "";
static int  g_MMBestPing         = 0;
static int  g_MMBestPlayers      = 0;
static int  g_MMBestMaxPlayers   = 0;
static double g_MMConnectAt      = 0.0; // time to auto-connect

ImFont* g_FontDefault = nullptr;
ImFont* g_FontMedium = nullptr;
ImFont* g_FontLarge = nullptr;

static const char* g_BackgroundMapName = "maps/echo.bsp";

static bool g_ImGuiInitialized = false;
static bool g_EventWatchRegistered = false;

static int SDLCALL ImGui_SDLEventWatch(void* userdata, SDL_Event* event)
{
	if (g_ImGuiInitialized && (g_ShowImGuiMenu || g_ShowCrosshairEditor || g_ShowAGSettings || g_ShowPauseMenu || g_ShowMainMenu || g_ShowServerBrowser))
	{
		ImGui_ImplSDL2_ProcessEvent(event);
	}
	return 0;
}

void ImGuiHelper_UpdateInputState()
{
	bool anyVisible = g_ShowImGuiMenu || g_ShowCrosshairEditor || g_ShowAGSettings || g_ShowPauseMenu || g_ShowMainMenu || g_ShowServerBrowser;
	gEngfuncs.pfnSetMouseEnable(anyVisible);
	SDL_ShowCursor(anyVisible ? SDL_ENABLE : SDL_DISABLE);
	HUD_UpdateCursorState();
}

void ToggleImGuiMenu_Callback()
{
	g_ShowImGuiMenu = !g_ShowImGuiMenu;
	ImGuiHelper_UpdateInputState();
}

void ImGuiHelper_Init()
{
	gEngfuncs.pfnAddCommand("toggle_imgui", ToggleImGuiMenu_Callback);
	cl_keypress_overlay = gEngfuncs.pfnRegisterVariable("cl_keypress_overlay", "0", 1); // FCVAR_ARCHIVE is 1
	cl_custom_hud = gEngfuncs.pfnRegisterVariable("cl_custom_hud", "1", 1);
	cl_custom_menu = gEngfuncs.pfnRegisterVariable("cl_custom_menu", "1", 1);
}

void ImGuiHelper_VidInit()
{
	if (g_ImGuiInitialized)
	{
		ImGui_ImplOpenGL2_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();
		g_ImGuiInitialized = false;
	}

	SDL_Window* window = nullptr;
	for (Uint32 i = 1; i < 32; ++i)
	{
		window = SDL_GetWindowFromID(i);
		if (window != nullptr)
			break;
	}

	if (!window)
	{
		gEngfuncs.Con_Printf("ImGuiHelper: Failed to get SDL Window\n");
		return;
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	// Load Roboto-Medium font at different sizes
	char fontPath[256];
	snprintf(fontPath, sizeof(fontPath), "%s/Roboto-Medium.ttf", gEngfuncs.pfnGetGameDirectory());

	g_FontDefault = io.Fonts->AddFontFromFileTTF(fontPath, 14.0f);
	if (!g_FontDefault)
	{
		gEngfuncs.Con_Printf("ImGuiHelper: Failed to load default font from %s, using fallback AddFontDefault()\n", fontPath);
		g_FontDefault = io.Fonts->AddFontDefault();
	}

	g_FontMedium = io.Fonts->AddFontFromFileTTF(fontPath, 18.0f);
	if (!g_FontMedium)
	{
		g_FontMedium = io.Fonts->AddFontDefault();
	}

	g_FontLarge = io.Fonts->AddFontFromFileTTF(fontPath, 45.0f);
	if (!g_FontLarge)
	{
		g_FontLarge = io.Fonts->AddFontDefault();
	}

	// Use premium dark style
	ImGui::StyleColorsDark();
	
	        // Custom styling for premium aesthetic (deep space dark mode)
        ImGuiStyle& style = ImGui::GetStyle();
        
        // Rounding & Spacing
        style.WindowRounding = 10.0f;
        style.FrameRounding = 5.0f;
        style.PopupRounding = 6.0f;
        style.GrabRounding = 4.0f;
        style.ScrollbarRounding = 8.0f;
        
        style.WindowPadding = ImVec2(16.0f, 16.0f);
        style.FramePadding = ImVec2(10.0f, 6.0f);
        style.ItemSpacing = ImVec2(12.0f, 8.0f);
        style.ItemInnerSpacing = ImVec2(8.0f, 6.0f);
        style.ScrollbarSize = 8.0f;
        style.GrabMinSize = 10.0f;
        
        style.WindowBorderSize = 1.0f;
        style.FrameBorderSize = 1.0f;
        style.PopupBorderSize = 1.0f;
        
        // Colors
        style.Colors[ImGuiCol_Text] = ImVec4(0.95f, 0.95f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.55f, 1.00f);
        
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.07f, 0.94f);
        style.Colors[ImGuiCol_ChildBg] = ImVec4(0.06f, 0.06f, 0.07f, 0.00f);
        style.Colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.10f, 0.98f);
        
        style.Colors[ImGuiCol_Border] = ImVec4(0.16f, 0.16f, 0.20f, 1.00f);
        style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        
        style.Colors[ImGuiCol_FrameBg] = ImVec4(0.10f, 0.10f, 0.13f, 1.00f);
        style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.15f, 0.15f, 0.18f, 1.00f);
        style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.18f, 0.18f, 0.22f, 1.00f);
        
        style.Colors[ImGuiCol_TitleBg] = ImVec4(0.06f, 0.06f, 0.07f, 1.00f);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
        style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.06f, 0.06f, 0.07f, 0.50f);
        
        style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.06f, 0.06f, 0.07f, 0.50f);
        style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.30f, 0.30f, 0.35f, 1.00f);
        style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.40f, 0.40f, 0.45f, 1.00f);
        
        // Active purple-blue accent
        style.Colors[ImGuiCol_CheckMark] = ImVec4(0.38f, 0.45f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.38f, 0.45f, 0.98f, 1.00f);
        style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.45f, 0.52f, 1.00f, 1.00f);
        
        // Button
        style.Colors[ImGuiCol_Button] = ImVec4(0.12f, 0.12f, 0.16f, 1.00f);
        style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.38f, 0.45f, 0.98f, 0.85f);
        style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.38f, 0.45f, 0.98f, 1.00f);
        
        // Header
        style.Colors[ImGuiCol_Header] = ImVec4(0.12f, 0.12f, 0.16f, 1.00f);
        style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.38f, 0.45f, 0.98f, 0.80f);
        style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.38f, 0.45f, 0.98f, 1.00f);
        
        style.Colors[ImGuiCol_Separator] = ImVec4(0.16f, 0.16f, 0.20f, 1.00f);
        style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
        style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.38f, 0.45f, 0.98f, 1.00f);
        
        style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.20f, 0.20f, 0.25f, 0.20f);
        style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.38f, 0.45f, 0.98f, 0.67f);
        style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.38f, 0.45f, 0.98f, 0.95f);
        
        style.Colors[ImGuiCol_Tab] = ImVec4(0.10f, 0.10f, 0.13f, 1.00f);
        style.Colors[ImGuiCol_TabHovered] = ImVec4(0.38f, 0.45f, 0.98f, 0.80f);
        style.Colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
        style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.08f, 0.08f, 0.10f, 0.97f);
        style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.14f, 0.14f, 0.18f, 1.00f);

	ImGui_ImplSDL2_InitForOpenGL(window, nullptr);
	ImGui_ImplOpenGL2_Init();

	if (!g_EventWatchRegistered)
	{
		SDL_AddEventWatch(ImGui_SDLEventWatch, nullptr);
		g_EventWatchRegistered = true;
	}

	g_ImGuiInitialized = true;
	gEngfuncs.Con_Printf("ImGuiHelper: Successfully initialized Dear ImGui (OpenGL2 + SDL2)\n");
}

void ImGuiHelper_Shutdown()
{
	if (g_ImGuiInitialized)
	{
		if (g_EventWatchRegistered)
		{
			SDL_DelEventWatch(ImGui_SDLEventWatch, nullptr);
			g_EventWatchRegistered = false;
		}
		ImGui_ImplOpenGL2_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();
		g_ImGuiInitialized = false;
	}
}

static void GetInfoValue(const char* info, const char* key, char* out, size_t outSize)
{
	out[0] = '\0';
	if (!info || !key) return;
	char searchKey[128];
	snprintf(searchKey, sizeof(searchKey), "\\%s\\", key);
	const char* p = strstr(info, searchKey);
	if (p)
	{
		p += strlen(searchKey);
		const char* end = strchr(p, '\\');
		size_t len = end ? (end - p) : strlen(p);
		if (len >= outSize) len = outSize - 1;
		strncpy(out, p, len);
	}
}

static int GetTotalAGOnline()
{
	int total = 0;
	if (g_pServers)
	{
		CHudServers::server_t* list = g_pServers->GetServersList();
		while (list)
		{
			char current[16] = "0";
			GetInfoValue(list->info, "current", current, sizeof(current));
			total += atoi(current);
			list = list->next;
		}
	}
	return total;
}

static bool CaseInsensitiveContains(const char* haystack, const char* needle)
{
	if (!haystack || !needle) return false;
	if (needle[0] == '\0') return true;

	size_t hLen = haystack ? strlen(haystack) : 0;
	size_t nLen = needle ? strlen(needle) : 0;
	if (hLen < nLen) return false;

	for (size_t i = 0; i <= hLen - nLen; ++i)
	{
		bool match = true;
		for (size_t j = 0; j < nLen; ++j)
		{
			char hc = haystack[i + j];
			char nc = needle[j];
			if (hc >= 'A' && hc <= 'Z') hc += 32;
			if (nc >= 'A' && nc <= 'Z') nc += 32;
			if (hc != nc)
			{
				match = false;
				break;
			}
		}
		if (match) return true;
	}
	return false;
}

void ImGuiHelper_Draw()
{
	if (!g_ImGuiInitialized)
		return;

	const char* mapName = gEngfuncs.pfnGetLevelName();
	bool hasMap = (mapName && mapName[0] != '\0');
	bool customMenuEnabled = cl_custom_menu ? (cl_custom_menu->value != 0.0f) : true;

	// Check if we're on the background map used for main menu rendering
	bool onBackgroundMap = false;
	if (hasMap && g_BackgroundMapName)
	{
		onBackgroundMap = (strcmp(mapName, g_BackgroundMapName) == 0);
	}
	g_IsBackgroundMap = onBackgroundMap;

	// Determine effective "in game" state
	bool inGame = hasMap && !onBackgroundMap;

	g_ShowMainMenu = false;
	/*
	if ((onBackgroundMap || !hasMap) && customMenuEnabled)
	{
		g_ShowMainMenu = true;
		g_ShowPauseMenu = false;
	}
	else
	{
		g_ShowMainMenu = false;
	}
	*/

	static bool lastAnyVisible = false;
	bool anyVisible = g_ShowImGuiMenu || g_ShowCrosshairEditor || g_ShowAGSettings || g_ShowPauseMenu || g_ShowMainMenu || g_ShowServerBrowser;
	if (anyVisible != lastAnyVisible)
	{
		ImGuiHelper_UpdateInputState();
		lastAnyVisible = anyVisible;
	}

	ImGui_ImplOpenGL2_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();

	/*
	// 1. Fullscreen Main Menu (when disconnected)
	// Don't render the main menu when the server browser is open (it would overlap and eat all clicks)
	if (g_ShowMainMenu && !g_ShowServerBrowser)
	{
		ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
		ImGui::SetNextWindowSize(ImVec2((float)ScreenWidth, (float)ScreenHeight));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.00f, 0.00f, 0.00f, 0.00f));

		ImGui::Begin("Main Menu", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);

		// Left-aligned Title "OPENAG"
		ImGui::SetCursorPos(ImVec2(80.0f, (float)ScreenHeight * 0.30f));
		ImGui::PushFont(g_FontLarge);
		ImGui::TextColored(ImVec4(0.38f, 0.45f, 0.98f, 1.00f), "OPENAG");
		ImGui::PopFont();

		// Subtitle
		ImGui::SetCursorPos(ImVec2(80.0f, (float)ScreenHeight * 0.30f + 65.0f));
		ImGui::PushFont(g_FontMedium);
		const char* sub = "Modern Dear ImGui Interface";
		ImGui::TextDisabled("%s", sub);
		ImGui::PopFont();

		// Navigation buttons (Left-aligned)
		float btn_w = 240.0f;
		float start_x = 80.0f;

		ImGui::SetCursorPos(ImVec2(start_x, (float)ScreenHeight * 0.30f + 120.0f));
		if (ImGui::Button("FIND MATCH", ImVec2(btn_w, 42)))
		{
			g_ShowMatchmaking = true;
			g_MMState = MM_SEARCHING;
			g_MMBestAddress[0] = '\0';
			g_MMBestName[0] = '\0';
			if (g_pServers) g_pServers->RequestList();
		}

		ImGui::Spacing();
		ImGui::SetCursorPosX(start_x);
		if (ImGui::Button("BROWSE SERVERS", ImVec2(btn_w, 34)))
		{
			g_ShowServerBrowser = true;
			if (g_pServers) g_pServers->RequestList();
		}

		ImGui::Spacing();
		ImGui::SetCursorPosX(start_x);
		if (ImGui::Button("AG SETTINGS", ImVec2(btn_w, 42)))
		{
			g_ShowAGSettings = true;
		}

		ImGui::Spacing();
		ImGui::SetCursorPosX(start_x);
		if (ImGui::Button("CROSSHAIR EDITOR", ImVec2(btn_w, 42)))
		{
			g_ShowCrosshairEditor = true;
		}

		ImGui::Spacing();
		ImGui::SetCursorPosX(start_x);
		if (ImGui::Button("QUIT GAME", ImVec2(btn_w, 42)))
		{
			gEngfuncs.pfnClientCmd("quit\n");
		}

	}
	*/

	// 1.5. Online player count overlay on top of original main menu
	bool gameUIVisible = (g_pGameUI && g_pGameUI->IsGameUIVisible() != 0);
	if ((gameUIVisible || onBackgroundMap || !hasMap) && !inGame)
	{
		static double last_query_time = -999.0;
		double now = gEngfuncs.GetClientTime();
		if (now - last_query_time > 30.0)
		{
			if (g_pServers && !g_pServers->IsRequesting())
			{
				g_pServers->RequestList();
				last_query_time = now;
			}
		}

		ImGuiWindowFlags overlay_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground;
		
		float ov_w = 175.0f;
		float ov_h = 42.0f;
		ImGui::SetNextWindowPos(ImVec2((float)ScreenWidth - ov_w - 20.0f, 20.0f), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(ov_w, ov_h));
		
		if (ImGui::Begin("HUD Online Overlay", nullptr, overlay_flags))
		{
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			ImVec2 p_min = ImGui::GetWindowPos();
			ImVec2 p_max = ImVec2(p_min.x + ov_w, p_min.y + ov_h);
			
			draw_list->AddRectFilled(p_min, p_max, ImColor(12, 12, 14, 180), 8.0f);
			draw_list->AddRect(p_min, p_max, ImColor(40, 40, 48, 120), 8.0f, 0, 1.0f);
			
			float pulse = 0.6f + 0.4f * sinf((float)now * 3.0f);
			bool loading = g_pServers && g_pServers->IsRequesting();
			ImColor dotColor = loading ? ImColor(50, 150, 255, (int)(255 * pulse)) : ImColor(50, 220, 100, (int)(255 * pulse));
			
			draw_list->AddCircleFilled(ImVec2(p_min.x + 18.0f, p_min.y + 21.0f), 4.5f, dotColor);
			
			ImGui::SetCursorPos(ImVec2(32.0f, 11.0f));
			int online = GetTotalAGOnline();
			if (loading)
			{
				ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.75f, 1.0f), "AG Online: %d*", online);
			}
			else
			{
				ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "AG Online: %d", online);
			}
		}
		ImGui::End();
	}

	// 2. Pause Menu Overlay (when in-game and g_ShowPauseMenu is true)
	if (g_ShowPauseMenu && inGame)
	{
		ImGui::SetNextWindowPos(ImVec2((float)ScreenWidth * 0.5f - 140.0f, (float)ScreenHeight * 0.5f - 170.0f), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(280, 330));
		if (ImGui::Begin("Game Paused", &g_ShowPauseMenu, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings))
		{
			float btn_w = 248.0f;
			ImGui::Spacing();
			
			if (ImGui::Button("RESUME GAME", ImVec2(btn_w, 42)))
			{
				g_ShowPauseMenu = false;
				ImGuiHelper_UpdateInputState();
			}

			ImGui::Spacing();
			if (ImGui::Button("DISCONNECT", ImVec2(btn_w, 42)))
			{
				gEngfuncs.pfnClientCmd("disconnect; map echo\n");
				g_ShowPauseMenu = false;
			}

			ImGui::Spacing();
			if (ImGui::Button("AG SETTINGS", ImVec2(btn_w, 42)))
			{
				g_ShowAGSettings = true;
				g_ShowPauseMenu = false;
			}

			ImGui::Spacing();
			if (ImGui::Button("CROSSHAIR EDITOR", ImVec2(btn_w, 42)))
			{
				g_ShowCrosshairEditor = true;
				g_ShowPauseMenu = false;
			}

			ImGui::Spacing();
			if (ImGui::Button("QUIT TO DESKTOP", ImVec2(btn_w, 42)))
			{
				gEngfuncs.pfnClientCmd("quit\n");
			}
		}
		ImGui::End();
	}

	// 3. Custom HUD Overlay (when in-game and cl_custom_hud is enabled)
	bool customHudEnabled = cl_custom_hud ? (cl_custom_hud->value != 0.0f) : true;
	if (inGame && customHudEnabled && gHUD.m_Health.m_iHealth > 0 && !gHUD.m_iIntermission)
	{
		ImGuiWindowFlags hud_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground;
		
		// Bottom-Left Health & Armor panel
		ImGui::SetNextWindowPos(ImVec2(24.0f, (float)ScreenHeight - 110.0f), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(240, 90));
		if (ImGui::Begin("HUD HealthArmor", nullptr, hud_flags))
		{
			// Render background plate manually for clean glassmorphism look
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			ImVec2 p_min = ImGui::GetWindowPos();
			ImVec2 p_max = ImVec2(p_min.x + 240.0f, p_min.y + 82.0f);
			draw_list->AddRectFilled(p_min, p_max, ImColor(12, 12, 14, 180), 8.0f);
			draw_list->AddRect(p_min, p_max, ImColor(40, 40, 48, 120), 8.0f, 0, 1.0f);

			ImGui::SetCursorPos(ImVec2(15.0f, 12.0f));

			int health = max(0, gHUD.m_Health.m_iHealth);
			int armor = max(0, gHUD.m_Battery.GetBattery());

			// Health bar
			char health_label[64];
			std::snprintf(health_label, sizeof(health_label), "HEALTH: %d", health);
			ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.20f, 0.85f, 0.40f, 0.90f)); // Mint green
			ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.60f));
			ImGui::ProgressBar(min(1.0f, health / 100.0f), ImVec2(210, 22), health_label);
			ImGui::PopStyleColor(2);

			ImGui::SetCursorPosX(15.0f);
			ImGui::Spacing();

			// Armor bar
			ImGui::SetCursorPosX(15.0f);
			char armor_label[64];
			std::snprintf(armor_label, sizeof(armor_label), "ARMOR: %d", armor);
			ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.20f, 0.60f, 1.00f, 0.90f)); // Electric blue
			ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.18f, 0.60f));
			ImGui::ProgressBar(min(1.0f, armor / 100.0f), ImVec2(210, 22), armor_label);
			ImGui::PopStyleColor(2);
		}
		ImGui::End();

		// Bottom-Right Ammo & Active Weapon panel
		ImGui::SetNextWindowPos(ImVec2((float)ScreenWidth - 264.0f, (float)ScreenHeight - 110.0f), ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(240, 90));
		if (ImGui::Begin("HUD AmmoWeapon", nullptr, hud_flags))
		{
			WEAPON* pw = gHUD.m_Ammo.GetActiveWeapon();
			if (pw)
			{
				ImDrawList* draw_list = ImGui::GetWindowDrawList();
				ImVec2 p_min = ImGui::GetWindowPos();
				ImVec2 p_max = ImVec2(p_min.x + 240.0f, p_min.y + 82.0f);
				draw_list->AddRectFilled(p_min, p_max, ImColor(12, 12, 14, 180), 8.0f);
				draw_list->AddRect(p_min, p_max, ImColor(40, 40, 48, 120), 8.0f, 0, 1.0f);

				auto CleanWeaponName = [](const char* rawName) -> const char* {
					if (!rawName) return "";
					if (strncmp(rawName, "weapon_", 7) == 0)
						return rawName + 7;
					return rawName;
				};

				// Drawing Weapon Name
				ImGui::SetCursorPos(ImVec2(15.0f, 12.0f));
				ImGui::SetWindowFontScale(1.1f);
				ImGui::TextColored(ImVec4(0.38f, 0.45f, 0.98f, 1.00f), "%s", CleanWeaponName(pw->szName));
				ImGui::SetWindowFontScale(1.0f);

				// Drawing Ammo counts
				int clip = pw->iClip;
				int reserve = gWR.CountAmmo(pw->iAmmoType);
				char ammo_label[64];
				if (pw->iAmmoType < 0)
				{
					std::snprintf(ammo_label, sizeof(ammo_label), "--");
				}
				else if (clip < 0)
				{
					std::snprintf(ammo_label, sizeof(ammo_label), "%d", reserve);
				}
				else
				{
					std::snprintf(ammo_label, sizeof(ammo_label), "%d / %d", clip, reserve);
				}

				ImGui::SetCursorPos(ImVec2(15.0f, 38.0f));
				ImGui::SetWindowFontScale(1.6f);
				ImGui::Text("%s", ammo_label);
				ImGui::SetWindowFontScale(1.0f);
			}
		}
		ImGui::End();

		cvar_t* pSpeedometer = gEngfuncs.pfnGetCvarPointer("cl_speedometer");
		bool speedometerEnabled = pSpeedometer ? (pSpeedometer->value != 0.0f) : true;
		if (speedometerEnabled)
		{
			float sw_w = 260.0f;
			float sw_h = 92.0f;
			float sw_x = ((float)ScreenWidth - sw_w) / 2.0f;
			float sw_y = (float)ScreenHeight - sw_h - 24.0f;

			ImGui::SetNextWindowPos(ImVec2(sw_x, sw_y), ImGuiCond_Always);
			ImGui::SetNextWindowSize(ImVec2(sw_w, sw_h));
			if (ImGui::Begin("HUD Speedometer", nullptr, hud_flags))
			{
				ImDrawList* draw_list = ImGui::GetWindowDrawList();
				ImVec2 p_min = ImGui::GetWindowPos();
				ImVec2 p_max = ImVec2(p_min.x + sw_w, p_min.y + sw_h);
				draw_list->AddRectFilled(p_min, p_max, ImColor(12, 12, 14, 180), 8.0f);
				draw_list->AddRect(p_min, p_max, ImColor(40, 40, 48, 120), 8.0f, 0, 1.0f);

				int speedVal = (int)std::round(g_StrafeData.speed);
				float accelVal = g_StrafeData.accel;

				ImVec4 speedColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
				if (accelVal > 15.0f)
				{
					speedColor = ImVec4(0.20f, 0.85f, 0.40f, 1.0f);
				}
				else if (accelVal < -15.0f)
				{
					speedColor = ImVec4(1.00f, 0.30f, 0.30f, 1.0f);
				}

				ImGui::SetCursorPos(ImVec2(15.0f, 12.0f));
				ImGui::SetWindowFontScale(2.2f);
				ImGui::TextColored(speedColor, "%d", speedVal);
				ImGui::SetWindowFontScale(1.0f);

				ImGui::SetCursorPos(ImVec2(90.0f, 26.0f));
				ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.60f, 1.0f), "ups");

				cvar_t* pShowStrafes = gEngfuncs.pfnGetCvarPointer("cl_speedometer_show_strafes");
				bool showStrafes = pShowStrafes ? (pShowStrafes->value != 0.0f) : true;
				if (showStrafes)
				{
					int keys = g_StrafeData.keys;
					bool holdA = (keys & IN_MOVELEFT) != 0;
					bool holdD = (keys & IN_MOVERIGHT) != 0;
					bool holdW = (keys & IN_FORWARD) != 0;
					bool holdS = (keys & IN_BACK) != 0;

					ImGui::SetCursorPos(ImVec2(135.0f, 12.0f));
					if (!g_StrafeData.on_ground && (holdW || holdS))
					{
						float pulse = 0.5f + 0.5f * sinf((float)gEngfuncs.GetClientTime() * 10.0f);
						ImGui::TextColored(ImVec4(1.0f, 0.1f * pulse, 0.1f * pulse, 1.0f), "%s%s", holdW ? "W" : "", holdS ? "S" : "");
					}
					else
					{
						ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "%s%s", (holdW || holdS) ? "" : "-", (holdW || holdS) ? "" : "-");
					}

					ImGui::SetCursorPos(ImVec2(135.0f, 28.0f));
					ImGui::TextColored(holdA ? ImVec4(0.20f, 0.85f, 0.40f, 1.0f) : ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "A");
					ImGui::SameLine();
					ImGui::TextColored(holdD ? ImVec4(0.20f, 0.85f, 0.40f, 1.0f) : ImVec4(0.4f, 0.4f, 0.45f, 1.0f), "D");

					const char* stateText = "IDLE";
					ImVec4 stateColor = ImVec4(0.55f, 0.55f, 0.60f, 1.0f);

					switch (g_StrafeData.strafe_state)
					{
						case 1:
							stateText = "PERFECT";
							stateColor = ImVec4(0.20f, 0.85f, 0.40f, 1.0f);
							break;
						case 2:
							stateText = "TURN FASTER";
							stateColor = ImVec4(0.90f, 0.80f, 0.20f, 1.0f);
							break;
						case 3:
							stateText = "TURN SLOWER";
							stateColor = ImVec4(0.95f, 0.50f, 0.10f, 1.0f);
							break;
						case 4:
							stateText = "SPEED LOSS";
							stateColor = ImVec4(1.00f, 0.20f, 0.20f, 1.0f);
							break;
						case 5:
							stateText = "RELEASE W/S";
							stateColor = ImVec4(1.00f, 0.10f, 0.10f, 1.0f);
							break;
					}

					ImGui::SetCursorPos(ImVec2(15.0f, 58.0f));
					ImGui::TextColored(stateColor, "%s", stateText);

					int eff_pct = (int)std::round(g_StrafeData.efficiency * 100.0f);
					if (g_StrafeData.strafe_state > 0 && g_StrafeData.strafe_state < 5)
					{
						ImGui::SetCursorPos(ImVec2(135.0f, 58.0f));
						ImGui::TextColored(stateColor, "%d%% EFF", eff_pct);
					}
				}
			}
			ImGui::End();
		}
	}

	AgentAPI_RenderHUD();

	if (g_ShowImGuiMenu)
	{
		ImGui::SetNextWindowSize(ImVec2(500, 350), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("OpenAG - Dear ImGui Menu", &g_ShowImGuiMenu, ImGuiWindowFlags_NoCollapse))
		{
			ImGui::Text("Welcome to the modern OpenAG Settings UI!");
			ImGui::Separator();

			static bool autoBhop = false;
			ImGui::Checkbox("Enable Auto-Bhop (sv_autobhop)", &autoBhop);

			static float airAccelerate = 10.0f;
			ImGui::SliderFloat("Air Accelerate (sv_airaccelerate)", &airAccelerate, 1.0f, 100.0f);

			static int maxFps = 100;
			ImGui::SliderInt("Max FPS (fps_max)", &maxFps, 60, 500);

			ImGui::Separator();
			if (ImGui::Button("Close Menu"))
			{
				ToggleImGuiMenu_Callback();
			}
		}
		ImGui::End();
	}

	if (g_ShowCrosshairEditor)
	{
		ImGui::SetNextWindowSize(ImVec2(350, 480), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Crosshair Editor", &g_ShowCrosshairEditor, ImGuiWindowFlags_NoCollapse))
		{
			cvar_t* pCvarCross = gEngfuncs.pfnGetCvarPointer("cl_cross");
			bool enableCross = pCvarCross ? (pCvarCross->value != 0.0f) : false;
			if (ImGui::Checkbox("Enable Crosshair", &enableCross))
			{
				gEngfuncs.Cvar_SetValue("cl_cross", enableCross ? 1.0f : 0.0f);
			}

			ImGui::Separator();

			cvar_t* pCvarSize = gEngfuncs.pfnGetCvarPointer("cl_cross_size");
			int sizeVal = pCvarSize ? (int)pCvarSize->value : 10;
			if (ImGui::SliderInt("Size", &sizeVal, 0, 100))
			{
				gEngfuncs.Cvar_SetValue("cl_cross_size", (float)sizeVal);
			}

			cvar_t* pCvarThickness = gEngfuncs.pfnGetCvarPointer("cl_cross_thickness");
			int thicknessVal = pCvarThickness ? (int)pCvarThickness->value : 2;
			if (ImGui::SliderInt("Thickness", &thicknessVal, 1, 20))
			{
				gEngfuncs.Cvar_SetValue("cl_cross_thickness", (float)thicknessVal);
			}

			cvar_t* pCvarGap = gEngfuncs.pfnGetCvarPointer("cl_cross_gap");
			int gapVal = pCvarGap ? (int)pCvarGap->value : 3;
			if (ImGui::SliderInt("Gap", &gapVal, -20, 50))
			{
				gEngfuncs.Cvar_SetValue("cl_cross_gap", (float)gapVal);
			}

			cvar_t* pCvarOutline = gEngfuncs.pfnGetCvarPointer("cl_cross_outline");
			int outlineVal = pCvarOutline ? (int)pCvarOutline->value : 0;
			if (ImGui::SliderInt("Outline", &outlineVal, 0, 10))
			{
				gEngfuncs.Cvar_SetValue("cl_cross_outline", (float)outlineVal);
			}

			cvar_t* pCvarDotSize = gEngfuncs.pfnGetCvarPointer("cl_cross_dot_size");
			int dotSizeVal = pCvarDotSize ? (int)pCvarDotSize->value : 0;
			if (ImGui::SliderInt("Dot Size", &dotSizeVal, 0, 50))
			{
				gEngfuncs.Cvar_SetValue("cl_cross_dot_size", (float)dotSizeVal);
			}

			cvar_t* pCvarCircleRadius = gEngfuncs.pfnGetCvarPointer("cl_cross_circle_radius");
			int circleRadiusVal = pCvarCircleRadius ? (int)pCvarCircleRadius->value : 0;
			if (ImGui::SliderInt("Circle Radius", &circleRadiusVal, 0, 100))
			{
				gEngfuncs.Cvar_SetValue("cl_cross_circle_radius", (float)circleRadiusVal);
			}

			cvar_t* pCvarAlpha = gEngfuncs.pfnGetCvarPointer("cl_cross_alpha");
			int alphaVal = pCvarAlpha ? (int)pCvarAlpha->value : 200;
			if (ImGui::SliderInt("Alpha", &alphaVal, 0, 255))
			{
				gEngfuncs.Cvar_SetValue("cl_cross_alpha", (float)alphaVal);
			}

			ImGui::Separator();
			ImGui::Text("Crosshair Color");

			cvar_t* pCvarColor = gEngfuncs.pfnGetCvarPointer("cl_cross_color");
			int r = 0, g = 255, b = 0;
			if (pCvarColor && pCvarColor->string)
			{
				std::sscanf(pCvarColor->string, "%d %d %d", &r, &g, &b);
			}
			
			float color[3] = { r / 255.0f, g / 255.0f, b / 255.0f };
			if (ImGui::ColorEdit3("Color", color))
			{
				r = (int)(color[0] * 255.0f + 0.5f);
				g = (int)(color[1] * 255.0f + 0.5f);
				b = (int)(color[2] * 255.0f + 0.5f);
				char cmd[128];
				std::snprintf(cmd, sizeof(cmd), "cl_cross_color \"%d %d %d\"\n", r, g, b);
				gEngfuncs.pfnClientCmd(cmd);
			}

			ImGui::Separator();
			ImGui::Text("Draw Lines");

			cvar_t* pTop = gEngfuncs.pfnGetCvarPointer("cl_cross_top_line");
			cvar_t* pBottom = gEngfuncs.pfnGetCvarPointer("cl_cross_bottom_line");
			cvar_t* pLeft = gEngfuncs.pfnGetCvarPointer("cl_cross_left_line");
			cvar_t* pRight = gEngfuncs.pfnGetCvarPointer("cl_cross_right_line");

			bool topVal = pTop ? (pTop->value != 0.0f) : true;
			bool bottomVal = pBottom ? (pBottom->value != 0.0f) : true;
			bool leftVal = pLeft ? (pLeft->value != 0.0f) : true;
			bool rightVal = pRight ? (pRight->value != 0.0f) : true;

			if (ImGui::Checkbox("Top", &topVal))
			{
				gEngfuncs.Cvar_SetValue("cl_cross_top_line", topVal ? 1.0f : 0.0f);
			}
			ImGui::SameLine();
			if (ImGui::Checkbox("Bottom", &bottomVal))
			{
				gEngfuncs.Cvar_SetValue("cl_cross_bottom_line", bottomVal ? 1.0f : 0.0f);
			}
			ImGui::SameLine();
			if (ImGui::Checkbox("Left", &leftVal))
			{
				gEngfuncs.Cvar_SetValue("cl_cross_left_line", leftVal ? 1.0f : 0.0f);
			}
			ImGui::SameLine();
			if (ImGui::Checkbox("Right", &rightVal))
			{
				gEngfuncs.Cvar_SetValue("cl_cross_right_line", rightVal ? 1.0f : 0.0f);
			}

			ImGui::Separator();
			if (ImGui::Button("Close", ImVec2(-FLT_MIN, 0)))
			{
				g_ShowCrosshairEditor = false;
				ImGuiHelper_UpdateInputState();
			}
		}
		ImGui::End();

		if (!g_ShowCrosshairEditor)
		{
			ImGuiHelper_UpdateInputState();
		}
	}

	if (g_ShowAGSettings)
	{
		ImGui::SetNextWindowSize(ImVec2(380, 450), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("AG Settings", &g_ShowAGSettings, ImGuiWindowFlags_NoCollapse))
		{
			cvar_t* pTeammate = gEngfuncs.pfnGetCvarPointer("cl_forceteammate_enable");
			bool teammateVal = pTeammate ? (pTeammate->value != 0.0f) : true;
			if (ImGui::Checkbox("Force Teammate Models", &teammateVal))
			{
				gEngfuncs.Cvar_SetValue("cl_forceteammate_enable", teammateVal ? 1.0f : 0.0f);
				gEngfuncs.pfnClientCmd("cl_forceteammodel_list\n");
			}

			cvar_t* pEnemy = gEngfuncs.pfnGetCvarPointer("cl_forceenemy_enable");
			bool enemyVal = pEnemy ? (pEnemy->value != 0.0f) : true;
			if (ImGui::Checkbox("Force Enemy Models", &enemyVal))
			{
				gEngfuncs.Cvar_SetValue("cl_forceenemy_enable", enemyVal ? 1.0f : 0.0f);
				gEngfuncs.pfnClientCmd("cl_forcemodel_list\n");
			}

			cvar_t* pExplosions = gEngfuncs.pfnGetCvarPointer("cl_explosions_enable");
			bool explosionsVal = pExplosions ? (pExplosions->value != 0.0f) : true;
			if (ImGui::Checkbox("Enable Explosion Effects", &explosionsVal))
			{
				gEngfuncs.Cvar_SetValue("cl_explosions_enable", explosionsVal ? 1.0f : 0.0f);
			}

			cvar_t* pDiscord = gEngfuncs.pfnGetCvarPointer("cl_discord_rpc");
			bool discordVal = pDiscord ? (pDiscord->value != 0.0f) : true;
			if (ImGui::Checkbox("Discord Rich Presence", &discordVal))
			{
				gEngfuncs.Cvar_SetValue("cl_discord_rpc", discordVal ? 1.0f : 0.0f);
			}

			cvar_t* pTimers = gEngfuncs.pfnGetCvarPointer("cl_item_timers");
			bool timersVal = pTimers ? (pTimers->value != 0.0f) : true;
			if (ImGui::Checkbox("Item Spawn Timers", &timersVal))
			{
				gEngfuncs.Cvar_SetValue("cl_item_timers", timersVal ? 1.0f : 0.0f);
			}

			cvar_t* pDamage = gEngfuncs.pfnGetCvarPointer("cl_damage_numbers");
			bool damageVal = pDamage ? (pDamage->value != 0.0f) : true;
			if (ImGui::Checkbox("Show Damage Numbers", &damageVal))
			{
				gEngfuncs.Cvar_SetValue("cl_damage_numbers", damageVal ? 1.0f : 0.0f);
			}

			bool keypressVal = cl_keypress_overlay ? (cl_keypress_overlay->value != 0.0f) : false;
			if (ImGui::Checkbox("Keypress Overlay", &keypressVal))
			{
				gEngfuncs.Cvar_SetValue("cl_keypress_overlay", keypressVal ? 1.0f : 0.0f);
			}

			bool customHudVal = cl_custom_hud ? (cl_custom_hud->value != 0.0f) : true;
			if (ImGui::Checkbox("Custom ImGui HUD", &customHudVal))
			{
				gEngfuncs.Cvar_SetValue("cl_custom_hud", customHudVal ? 1.0f : 0.0f);
			}

			bool customMenuVal = cl_custom_menu ? (cl_custom_menu->value != 0.0f) : true;
			if (ImGui::Checkbox("Custom ImGui Menus", &customMenuVal))
			{
				gEngfuncs.Cvar_SetValue("cl_custom_menu", customMenuVal ? 1.0f : 0.0f);
			}

			cvar_t* pAgentApi = gEngfuncs.pfnGetCvarPointer("cl_agent_api");
			bool agentApiVal = pAgentApi ? (pAgentApi->value != 0.0f) : false;
			if (ImGui::Checkbox("AI Agent API (Shared Memory)", &agentApiVal))
			{
				gEngfuncs.Cvar_SetValue("cl_agent_api", agentApiVal ? 1.0f : 0.0f);
			}

			ImGui::Separator();

			if (ImGui::CollapsingHeader("Weapon Sway Settings"))
			{
				cvar_t* pSway = gEngfuncs.pfnGetCvarPointer("cl_weapon_sway");
				bool swayVal = pSway ? (pSway->value != 0.0f) : true;
				if (ImGui::Checkbox("Enable Weapon Sway", &swayVal))
				{
					gEngfuncs.Cvar_SetValue("cl_weapon_sway", swayVal ? 1.0f : 0.0f);
				}

				if (swayVal)
				{
					cvar_t* pSwayPitch = gEngfuncs.pfnGetCvarPointer("cl_weapon_sway_pitch");
					float swayPitchVal = pSwayPitch ? pSwayPitch->value : 6.0f;
					if (ImGui::SliderFloat("Pitch Speed", &swayPitchVal, 1.0f, 20.0f, "%.1f"))
					{
						gEngfuncs.Cvar_SetValue("cl_weapon_sway_pitch", swayPitchVal);
					}

					cvar_t* pSwayYaw = gEngfuncs.pfnGetCvarPointer("cl_weapon_sway_yaw");
					float swayYawVal = pSwayYaw ? pSwayYaw->value : 6.0f;
					if (ImGui::SliderFloat("Yaw Speed", &swayYawVal, 1.0f, 20.0f, "%.1f"))
					{
						gEngfuncs.Cvar_SetValue("cl_weapon_sway_yaw", swayYawVal);
					}

					cvar_t* pSwayRollSpeed = gEngfuncs.pfnGetCvarPointer("cl_weapon_sway_roll");
					float swayRollSpeedVal = pSwayRollSpeed ? pSwayRollSpeed->value : 4.0f;
					if (ImGui::SliderFloat("Roll Speed", &swayRollSpeedVal, 1.0f, 20.0f, "%.1f"))
					{
						gEngfuncs.Cvar_SetValue("cl_weapon_sway_roll", swayRollSpeedVal);
					}

					cvar_t* pSwayClamp = gEngfuncs.pfnGetCvarPointer("cl_weapon_sway_clamp");
					float swayClampVal = pSwayClamp ? pSwayClamp->value : 10.0f;
					if (ImGui::SliderFloat("Sway Limit", &swayClampVal, 1.0f, 30.0f, "%.1f deg"))
					{
						gEngfuncs.Cvar_SetValue("cl_weapon_sway_clamp", swayClampVal);
					}

					cvar_t* pSwayRollFactor = gEngfuncs.pfnGetCvarPointer("cl_weapon_sway_roll_factor");
					float swayRollFactorVal = pSwayRollFactor ? pSwayRollFactor->value : 0.15f;
					if (ImGui::SliderFloat("Roll Intensity", &swayRollFactorVal, 0.0f, 1.0f, "%.2f"))
					{
						gEngfuncs.Cvar_SetValue("cl_weapon_sway_roll_factor", swayRollFactorVal);
					}

					cvar_t* pSwayPosY = gEngfuncs.pfnGetCvarPointer("cl_weapon_sway_pos_yaw");
					float swayPosYVal = pSwayPosY ? pSwayPosY->value : 0.02f;
					if (ImGui::SliderFloat("Horizontal Shift", &swayPosYVal, 0.0f, 0.1f, "%.3f"))
					{
						gEngfuncs.Cvar_SetValue("cl_weapon_sway_pos_yaw", swayPosYVal);
					}

					cvar_t* pSwayPosP = gEngfuncs.pfnGetCvarPointer("cl_weapon_sway_pos_pitch");
					float swayPosPVal = pSwayPosP ? pSwayPosP->value : 0.02f;
					if (ImGui::SliderFloat("Vertical Shift", &swayPosPVal, 0.0f, 0.1f, "%.3f"))
					{
						gEngfuncs.Cvar_SetValue("cl_weapon_sway_pos_pitch", swayPosPVal);
					}
				}
			}

			if (ImGui::CollapsingHeader("FOV & Viewmodel Settings"))
			{
				cvar_t* pDefaultFov = gEngfuncs.pfnGetCvarPointer("default_fov");
				float defaultFovVal = pDefaultFov ? pDefaultFov->value : 90.0f;
				if (ImGui::SliderFloat("Default FOV", &defaultFovVal, 70.0f, 130.0f, "%.0f"))
				{
					gEngfuncs.Cvar_SetValue("default_fov", defaultFovVal);
				}

				cvar_t* pDrawViewmodel = gEngfuncs.pfnGetCvarPointer("r_drawviewmodel");
				bool drawViewmodelVal = pDrawViewmodel ? (pDrawViewmodel->value != 0.0f) : true;
				if (ImGui::Checkbox("Draw Viewmodel (Weapon)", &drawViewmodelVal))
				{
					gEngfuncs.Cvar_SetValue("r_drawviewmodel", drawViewmodelVal ? 1.0f : 0.0f);
				}

				cvar_t* pViewmodelFov = gEngfuncs.pfnGetCvarPointer("cl_viewmodel_fov");
				float viewmodelFovVal = pViewmodelFov ? pViewmodelFov->value : 0.0f;
				if (ImGui::SliderFloat("Viewmodel FOV", &viewmodelFovVal, 0.0f, 130.0f, "%.0f (0 to use default)"))
				{
					gEngfuncs.Cvar_SetValue("cl_viewmodel_fov", viewmodelFovVal);
				}

				if (drawViewmodelVal)
				{
					ImGui::Separator();

					cvar_t* pViewmodelOfsX = gEngfuncs.pfnGetCvarPointer("cl_viewmodel_ofs_right");
					float viewmodelOfsXVal = pViewmodelOfsX ? pViewmodelOfsX->value : 0.0f;
					if (ImGui::SliderFloat("Viewmodel X Offset (Right/Left)", &viewmodelOfsXVal, -8.0f, 8.0f, "%.1f"))
					{
						gEngfuncs.Cvar_SetValue("cl_viewmodel_ofs_right", viewmodelOfsXVal);
					}

					cvar_t* pViewmodelOfsY = gEngfuncs.pfnGetCvarPointer("cl_viewmodel_ofs_forward");
					float viewmodelOfsYVal = pViewmodelOfsY ? pViewmodelOfsY->value : 0.0f;
					if (ImGui::SliderFloat("Viewmodel Y Offset (Forward/Back)", &viewmodelOfsYVal, -8.0f, 8.0f, "%.1f"))
					{
						gEngfuncs.Cvar_SetValue("cl_viewmodel_ofs_forward", viewmodelOfsYVal);
					}

					cvar_t* pViewmodelOfsZ = gEngfuncs.pfnGetCvarPointer("cl_viewmodel_ofs_up");
					float viewmodelOfsZVal = pViewmodelOfsZ ? pViewmodelOfsZ->value : 0.0f;
					if (ImGui::SliderFloat("Viewmodel Z Offset (Up/Down)", &viewmodelOfsZVal, -8.0f, 8.0f, "%.1f"))
					{
						gEngfuncs.Cvar_SetValue("cl_viewmodel_ofs_up", viewmodelOfsZVal);
					}
				}
			}

			if (ImGui::CollapsingHeader("Speedometer & Strafe Analyzer"))
			{
				cvar_t* pSpeedometer = gEngfuncs.pfnGetCvarPointer("cl_speedometer");
				bool speedometerVal = pSpeedometer ? (pSpeedometer->value != 0.0f) : true;
				if (ImGui::Checkbox("Enable Speedometer HUD", &speedometerVal))
				{
					gEngfuncs.Cvar_SetValue("cl_speedometer", speedometerVal ? 1.0f : 0.0f);
				}

				if (speedometerVal)
				{
					cvar_t* pShowStrafes = gEngfuncs.pfnGetCvarPointer("cl_speedometer_show_strafes");
					bool showStrafesVal = pShowStrafes ? (pShowStrafes->value != 0.0f) : true;
					if (ImGui::Checkbox("Enable Strafe Analyzer & Key Overlay", &showStrafesVal))
					{
						gEngfuncs.Cvar_SetValue("cl_speedometer_show_strafes", showStrafesVal ? 1.0f : 0.0f);
					}
				}
			}

			if (ImGui::CollapsingHeader("Spawns Display Settings"))
			{
				cvar_t* pShowSpawns = gEngfuncs.pfnGetCvarPointer("cl_show_spawns");
				bool showSpawnsVal = pShowSpawns ? (pShowSpawns->value != 0.0f) : true;
				if (ImGui::Checkbox("Show Spawn Points on Screen", &showSpawnsVal))
				{
					gEngfuncs.Cvar_SetValue("cl_show_spawns", showSpawnsVal ? 1.0f : 0.0f);
				}

				if (showSpawnsVal)
				{
					cvar_t* pOnlyActive = gEngfuncs.pfnGetCvarPointer("cl_show_spawns_only_active");
					bool onlyActiveVal = pOnlyActive ? (pOnlyActive->value != 0.0f) : false;
					if (ImGui::Checkbox("Show Only Active Spawns", &onlyActiveVal))
					{
						gEngfuncs.Cvar_SetValue("cl_show_spawns_only_active", onlyActiveVal ? 1.0f : 0.0f);
					}

					cvar_t* pSpawnsDist = gEngfuncs.pfnGetCvarPointer("cl_show_spawns_dist");
					float spawnsDistVal = pSpawnsDist ? pSpawnsDist->value : 2000.0f;
					if (ImGui::SliderFloat("Max Render Distance", &spawnsDistVal, 200.0f, 5000.0f, "%.0f units"))
					{
						gEngfuncs.Cvar_SetValue("cl_show_spawns_dist", spawnsDistVal);
					}
				}
			}

			if (ImGui::CollapsingHeader("Graphics / Post-Processing"))
			{
				ImGui::TextDisabled("Settings apply in real-time and are saved automatically.");
				ImGui::Spacing();

				// ── Bloom ────────────────────────────────────────────────
				ImGui::SeparatorText("Bloom");
				cvar_t* pBloom = gEngfuncs.pfnGetCvarPointer("cl_bloom");
				bool bloomVal = pBloom ? (pBloom->value != 0.0f) : false;
				if (ImGui::Checkbox("Enable Bloom##pp", &bloomVal))
					gEngfuncs.Cvar_SetValue("cl_bloom", bloomVal ? 1.0f : 0.0f);
				if (bloomVal)
				{
					ImGui::Indent();
					cvar_t* pBI = gEngfuncs.pfnGetCvarPointer("cl_bloom_intensity");
					float bi = pBI ? pBI->value : 0.6f;
					if (ImGui::SliderFloat("Intensity##bloom", &bi, 0.1f, 2.0f, "%.2f"))
						gEngfuncs.Cvar_SetValue("cl_bloom_intensity", bi);
					cvar_t* pBR = gEngfuncs.pfnGetCvarPointer("cl_bloom_radius");
					float br = pBR ? pBR->value : 3.0f;
					if (ImGui::SliderFloat("Spread Radius##bloom", &br, 0.5f, 8.0f, "%.1f px"))
						gEngfuncs.Cvar_SetValue("cl_bloom_radius", br);
					cvar_t* pBT = gEngfuncs.pfnGetCvarPointer("cl_bloom_threshold");
					int bt = pBT ? (int)pBT->value : 2;
					if (ImGui::SliderInt("Brightness Threshold##bloom", &bt, 0, 5))
						gEngfuncs.Cvar_SetValue("cl_bloom_threshold", (float)bt);
					cvar_t* pBP = gEngfuncs.pfnGetCvarPointer("cl_bloom_passes");
					int bp = pBP ? (int)pBP->value : 3;
					if (ImGui::SliderInt("Blur Passes##bloom", &bp, 1, 6))
						gEngfuncs.Cvar_SetValue("cl_bloom_passes", (float)bp);
					ImGui::Unindent();
				}

				ImGui::Spacing();

				// ── Motion Blur ──────────────────────────────────────────
				ImGui::SeparatorText("Motion Blur");
				cvar_t* pMB = gEngfuncs.pfnGetCvarPointer("cl_motion_blur");
				bool mbVal = pMB ? (pMB->value != 0.0f) : false;
				if (ImGui::Checkbox("Enable Motion Blur##pp", &mbVal))
					gEngfuncs.Cvar_SetValue("cl_motion_blur", mbVal ? 1.0f : 0.0f);
				if (mbVal)
				{
					ImGui::Indent();
					cvar_t* pSh = gEngfuncs.pfnGetCvarPointer("cl_motion_blur_shutter");
					float sv = pSh ? pSh->value : 0.015f;
					if (ImGui::SliderFloat("Shutter Speed##mb", &sv, 0.003f, 0.05f, "%.3f s"))
						gEngfuncs.Cvar_SetValue("cl_motion_blur_shutter", sv);
					cvar_t* pAl = gEngfuncs.pfnGetCvarPointer("cl_motion_blur_alpha");
					float av = pAl ? pAl->value : 0.10f;
					if (ImGui::SliderFloat("Sample Opacity##mb", &av, 0.01f, 0.30f, "%.2f"))
						gEngfuncs.Cvar_SetValue("cl_motion_blur_alpha", av);
					cvar_t* pMx = gEngfuncs.pfnGetCvarPointer("cl_motion_blur_max");
					float mv = pMx ? pMx->value : 25.f;
					if (ImGui::SliderFloat("Max Offset##mb", &mv, 5.f, 80.f, "%.0f px"))
						gEngfuncs.Cvar_SetValue("cl_motion_blur_max", mv);
					cvar_t* pCh = gEngfuncs.pfnGetCvarPointer("cl_motion_blur_chromatic");
					float cv = pCh ? pCh->value : 0.0f;
					if (ImGui::SliderFloat("Chromatic Aberration##mb", &cv, 0.0f, 3.0f, "%.1f"))
						gEngfuncs.Cvar_SetValue("cl_motion_blur_chromatic", cv);
					ImGui::Unindent();
				}

				ImGui::Spacing();

				// ── Vignette ─────────────────────────────────────────────
				ImGui::SeparatorText("Vignette");
				cvar_t* pVig = gEngfuncs.pfnGetCvarPointer("cl_vignette");
				bool vigVal = pVig ? (pVig->value != 0.0f) : false;
				if (ImGui::Checkbox("Enable Vignette##pp", &vigVal))
					gEngfuncs.Cvar_SetValue("cl_vignette", vigVal ? 1.0f : 0.0f);
				if (vigVal)
				{
					ImGui::Indent();
					cvar_t* pVS = gEngfuncs.pfnGetCvarPointer("cl_vignette_strength");
					float vs = pVS ? pVS->value : 0.75f;
					if (ImGui::SliderFloat("Strength##vig", &vs, 0.05f, 1.0f, "%.2f"))
						gEngfuncs.Cvar_SetValue("cl_vignette_strength", vs);
					cvar_t* pVR = gEngfuncs.pfnGetCvarPointer("cl_vignette_radius");
					float vr = pVR ? pVR->value : 0.65f;
					if (ImGui::SliderFloat("Inner Radius##vig", &vr, 0.2f, 1.2f, "%.2f"))
						gEngfuncs.Cvar_SetValue("cl_vignette_radius", vr);
					cvar_t* pVSo = gEngfuncs.pfnGetCvarPointer("cl_vignette_softness");
					float vso = pVSo ? pVSo->value : 0.45f;
					if (ImGui::SliderFloat("Softness##vig", &vso, 0.05f, 1.0f, "%.2f"))
						gEngfuncs.Cvar_SetValue("cl_vignette_softness", vso);
					ImGui::Unindent();
				}

				ImGui::Spacing();

				// ── Film Grain ───────────────────────────────────────────
				ImGui::SeparatorText("Film Grain");
				cvar_t* pGrain = gEngfuncs.pfnGetCvarPointer("cl_film_grain");
				bool grainVal = pGrain ? (pGrain->value != 0.0f) : false;
				if (ImGui::Checkbox("Enable Film Grain##pp", &grainVal))
					gEngfuncs.Cvar_SetValue("cl_film_grain", grainVal ? 1.0f : 0.0f);
				if (grainVal)
				{
					ImGui::Indent();
					cvar_t* pGA = gEngfuncs.pfnGetCvarPointer("cl_film_grain_amount");
					float ga = pGA ? pGA->value : 0.04f;
					if (ImGui::SliderFloat("Amount##grain", &ga, 0.005f, 0.20f, "%.3f"))
						gEngfuncs.Cvar_SetValue("cl_film_grain_amount", ga);
					ImGui::Unindent();
				}

				ImGui::Spacing();

// ── SSAO ─────────────────────────────────────────────────
				ImGui::SeparatorText("Screen Space Ambient Occlusion");
				cvar_t* pSSAO = gEngfuncs.pfnGetCvarPointer("cl_ssao");
				bool ssaoVal = pSSAO ? (pSSAO->value != 0.0f) : false;
				if (ImGui::Checkbox("Enable SSAO##pp", &ssaoVal))
					gEngfuncs.Cvar_SetValue("cl_ssao", ssaoVal ? 1.0f : 0.0f);

				if (ssaoVal)
				{
					ImGui::Indent();
					
					// Ползунок радиуса
					cvar_t* pSSAORad = gEngfuncs.pfnGetCvarPointer("cl_ssao_radius");
					float ssaoRad = pSSAORad ? pSSAORad->value : 8.0f;
					if (ImGui::SliderFloat("Shadow Radius##ssao", &ssaoRad, 1.0f, 20.0f, "%.1f"))
						gEngfuncs.Cvar_SetValue("cl_ssao_radius", ssaoRad);
						
					// Ползунок интенсивности
					cvar_t* pSSAOInt = gEngfuncs.pfnGetCvarPointer("cl_ssao_intensity");
					float ssaoInt = pSSAOInt ? pSSAOInt->value : 1.2f;
					if (ImGui::SliderFloat("Intensity##ssao", &ssaoInt, 0.1f, 5.0f, "%.2f"))
						gEngfuncs.Cvar_SetValue("cl_ssao_intensity", ssaoInt);
						
					ImGui::Spacing();
					
					// Прячем техническую информацию в спойлер, чтобы не засорять меню
					if (ImGui::TreeNode("Diagnostics (Depth Debug)"))
					{
						const auto& dbg = post_process::get_depth_debug();

						// Status
						if (!dbg.capture_attempted) {
							ImGui::TextColored(ImVec4(1,1,0,1), "Not capturing (waiting for first frame)");
						} else if (dbg.capture_succeeded) {
							// Добавлен "MSAA Blit" под индексом 4
							const char* methods[] = { "None", "glReadPixels(FLOAT)", "glReadPixels(UINT)", "glCopyTexSubImage2D", "MSAA Blit" };
							int methodIdx = (dbg.method_used >= 0 && dbg.method_used <= 4) ? dbg.method_used : 0;
							ImGui::TextColored(ImVec4(0,1,0,1), "OK: %s", methods[methodIdx]);
						} else {
							ImGui::TextColored(ImVec4(1,0,0,1), "FAILED! GL error: 0x%X", dbg.gl_error);
						}

						ImGui::Text("Viewport: %d x %d", dbg.viewport_w, dbg.viewport_h);
						ImGui::Text("Frames captured: %d", dbg.frame_count);

						if (dbg.capture_succeeded && dbg.center_depth >= 0.0f) {
							ImGui::Text("Center depth: %.6f", dbg.center_depth);
							ImGui::Spacing();
							ImGui::Text("3x3 depth grid (center +/- 40px):");
							for (int row = 2; row >= 0; row--) {
								ImGui::Text("  %.4f  %.4f  %.4f",
									dbg.sample_depths[row*3+0],
									dbg.sample_depths[row*3+1],
									dbg.sample_depths[row*3+2]);
							}
						}
						ImGui::TreePop();
					}

					ImGui::Unindent();
				}
			}

			ImGui::Separator();

			cvar_t* pFootstep = gEngfuncs.pfnGetCvarPointer("cl_footstep_volume");
			int footstepVal = pFootstep ? (int)(pFootstep->value * 100.0f) : 100;
			if (ImGui::SliderInt("Footstep Sound Volume", &footstepVal, 0, 200, "%d%%"))
			{
				gEngfuncs.Cvar_SetValue("cl_footstep_volume", (float)footstepVal / 100.0f);
			}

			ImGui::Separator();
			if (ImGui::Button("Close", ImVec2(-FLT_MIN, 0)))
			{
				g_ShowAGSettings = false;
				ImGuiHelper_UpdateInputState();
			}
		}
		ImGui::End();

		if (!g_ShowAGSettings)
		{
			ImGuiHelper_UpdateInputState();
		}
	}

	if (cl_keypress_overlay && cl_keypress_overlay->value != 0.0f && !g_ShowServerBrowser)
	{
		ImGuiWindowFlags flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings;
		bool interactive = g_ShowImGuiMenu || g_ShowCrosshairEditor || g_ShowAGSettings;
		if (!interactive)
		{
			flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs;
		}

		ImGui::SetNextWindowSize(ImVec2(190, 150), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Keypress Overlay", nullptr, flags))
		{
			auto DrawKey = [](const char* label, bool pressed, ImVec2 size) {
				ImVec4 col_bg = pressed ? ImVec4(0.38f, 0.45f, 0.98f, 1.00f) : ImVec4(0.12f, 0.12f, 0.16f, 0.60f);
				ImVec4 col_text = pressed ? ImVec4(1.00f, 1.00f, 1.00f, 1.00f) : ImVec4(0.60f, 0.60f, 0.65f, 0.80f);
				ImVec4 col_border = pressed ? ImVec4(0.45f, 0.52f, 1.00f, 1.00f) : ImVec4(0.20f, 0.20f, 0.25f, 0.40f);

				ImGui::PushStyleColor(ImGuiCol_Button, col_bg);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, col_bg);
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, col_bg);
				ImGui::PushStyleColor(ImGuiCol_Text, col_text);
				ImGui::PushStyleColor(ImGuiCol_Border, col_border);
				
				ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
				ImGui::Button(label, size);
				ImGui::PopStyleVar();
				
				ImGui::PopStyleColor(5);
			};

			bool w = (g_dwKeyPressButtons & IN_FORWARD) != 0;
			bool a = (g_dwKeyPressButtons & IN_MOVELEFT) != 0;
			bool s = (g_dwKeyPressButtons & IN_BACK) != 0;
			bool d = (g_dwKeyPressButtons & IN_MOVERIGHT) != 0;
			bool lmb = (g_dwKeyPressButtons & IN_ATTACK) != 0;
			bool rmb = (g_dwKeyPressButtons & IN_ATTACK2) != 0;
			bool space = (g_dwKeyPressButtons & IN_JUMP) != 0;
			bool ctrl = (g_dwKeyPressButtons & IN_DUCK) != 0;

			// Row 1: LMB [W] RMB
			DrawKey("LMB", lmb, ImVec2(50, 30));
			ImGui::SameLine();
			DrawKey("W", w, ImVec2(50, 30));
			ImGui::SameLine();
			DrawKey("RMB", rmb, ImVec2(50, 30));

			// Row 2: [A] [S] [D]
			DrawKey("A", a, ImVec2(50, 30));
			ImGui::SameLine();
			DrawKey("S", s, ImVec2(50, 30));
			ImGui::SameLine();
			DrawKey("D", d, ImVec2(50, 30));

			// Row 3: [Ctrl] [Space]
			DrawKey("Ctrl", ctrl, ImVec2(77, 30));
			ImGui::SameLine();
			DrawKey("Space", space, ImVec2(77, 30));
		}
		ImGui::End();
	}

	// ── Matchmaking Modal ────────────────────────────────────────────────────
	if (g_ShowMatchmaking)
	{
		// When searching: try to pick best server once list finishes loading
		if (g_MMState == MM_SEARCHING && g_pServers && !g_pServers->IsRequesting())
		{
			// Score every server and find the best one
			CHudServers::server_t* list = g_pServers->GetServersList();
			CHudServers::server_t* best = nullptr;
			float bestScore = -1e9f;

			while (list)
			{
				char current[16]    = "0";
				char maxPlayers[16] = "32";
				char address[128]   = "";
				char name[128]      = "";
				char map[64]        = "";

				GetInfoValue(list->info, "current", current, sizeof(current));
				GetInfoValue(list->info, "max", maxPlayers, sizeof(maxPlayers));
				GetInfoValue(list->info, "address", address, sizeof(address));
				GetInfoValue(list->info, "hostname", name, sizeof(name));
				GetInfoValue(list->info, "map", map, sizeof(map));

				if (address[0] == '\0')
					strncpy(address, gEngfuncs.pNetAPI->AdrToString(&list->remote_address), sizeof(address) - 1);

				int cur  = atoi(current);
				int maxP = atoi(maxPlayers);
				int ping = list->ping;

				// Skip servers with absurdly high ping
				if (ping > 350) { list = list->next; continue; }

				// Score: reward active servers (+players) and low ping, penalise completely full servers
				float fillRatio = maxP > 0 ? (float)cur / (float)maxP : 0.0f;
				float pingScore = 1.0f - (float)ping / 350.0f;         // 1.0 at 0ms, 0.0 at 350ms
				float playerScore = fillRatio > 0.05f ? fillRatio : 0.0f; // prefer non-empty
				float fullPenalty = fillRatio >= 0.98f ? -2.0f : 0.0f;
				float score = pingScore * 3.0f + playerScore * 5.0f + fullPenalty;

				if (score > bestScore)
				{
					bestScore = score;
					best = list;
					strncpy(g_MMBestAddress, address,  sizeof(g_MMBestAddress) - 1);
					strncpy(g_MMBestName,    name,     sizeof(g_MMBestName)    - 1);
					strncpy(g_MMBestMap,     map,      sizeof(g_MMBestMap)     - 1);
					g_MMBestPing       = ping;
					g_MMBestPlayers    = cur;
					g_MMBestMaxPlayers = maxP;
				}
				list = list->next;
			}

			if (best && g_MMBestAddress[0] != '\0')
			{
				g_MMState = MM_FOUND;
				g_MMConnectAt = ImGui::GetTime() + 5.0; // auto-connect after 5 seconds
			}
			// else: no servers yet — keep waiting (user can cancel)
		}

		// Auto-connect when timer expires
		if (g_MMState == MM_FOUND && ImGui::GetTime() >= g_MMConnectAt)
		{
			g_MMState = MM_CONNECTING;
		}
		if (g_MMState == MM_CONNECTING && g_MMBestAddress[0] != '\0')
		{
			char cmd[256];
			snprintf(cmd, sizeof(cmd), "connect %s\n", g_MMBestAddress);
			gEngfuncs.pfnClientCmd(cmd);
			g_ShowMatchmaking = false;
			g_MMState = MM_IDLE;
		}

		// ── Draw the modal ────────────────────────────────────────────────────
		float mw = 480.0f, mh = 260.0f;
		ImGui::SetNextWindowPos(
			ImVec2((float)ScreenWidth * 0.5f - mw * 0.5f, (float)ScreenHeight * 0.5f - mh * 0.5f),
			ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(mw, mh), ImGuiCond_Always);

		ImGui::PushStyleColor(ImGuiCol_WindowBg,    ImVec4(0.05f, 0.07f, 0.12f, 0.97f));
		ImGui::PushStyleColor(ImGuiCol_Border,      ImVec4(0.20f, 0.35f, 0.70f, 0.80f));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,  8.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.5f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,    ImVec2(28.0f, 22.0f));

		if (ImGui::Begin("##Matchmaking", nullptr,
			ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings))
		{
			float t = (float)ImGui::GetTime();
			ImDrawList* dl = ImGui::GetWindowDrawList();
			ImVec2 wpos   = ImGui::GetWindowPos();

			// Thin blue accent line at top
			dl->AddRectFilled(
				ImVec2(wpos.x, wpos.y),
				ImVec2(wpos.x + mw, wpos.y + 3.0f),
				IM_COL32(60, 130, 255, 255));

			// ── Title
			ImGui::SetCursorPos(ImVec2(28.0f, 18.0f));
			ImGui::PushFont(g_FontMedium);
			if (g_MMState == MM_SEARCHING)
				ImGui::TextColored(ImVec4(0.85f, 0.90f, 1.00f, 1.0f), "FINDING BEST SERVER");
			else if (g_MMState == MM_FOUND)
				ImGui::TextColored(ImVec4(0.30f, 0.90f, 0.50f, 1.0f), "MATCH FOUND!");
			ImGui::PopFont();

			// ── Spinner / check mark ─────────────────────────────────────────
			ImVec2 spinCenter = ImVec2(wpos.x + mw - 50.0f, wpos.y + 38.0f);
			if (g_MMState == MM_SEARCHING)
			{
				// Rotating arc
				for (int i = 0; i < 12; i++)
				{
					float angle = (2.0f * 3.14159f / 12.0f) * i + t * 3.0f;
					float alpha = (float)i / 12.0f;
					ImVec2 p = ImVec2(spinCenter.x + cosf(angle) * 14.0f, spinCenter.y + sinf(angle) * 14.0f);
					dl->AddCircleFilled(p, 2.5f, IM_COL32(80, 140, 255, (int)(alpha * 220)));
				}
			}
			else if (g_MMState == MM_FOUND)
			{
				// Pulsing green circle
				float pulse = (sinf(t * 4.0f) + 1.0f) * 0.5f;
				dl->AddCircle(spinCenter, 14.0f, IM_COL32(50, 200, 80, (int)(100 + 155 * pulse)), 32, 2.0f);
				dl->AddCircleFilled(spinCenter, 9.0f, IM_COL32(50, 220, 80, 200));
				// Checkmark
				dl->AddLine(ImVec2(spinCenter.x - 5.0f, spinCenter.y),
				            ImVec2(spinCenter.x - 1.0f, spinCenter.y + 5.0f), IM_COL32(255,255,255,255), 2.0f);
				dl->AddLine(ImVec2(spinCenter.x - 1.0f, spinCenter.y + 5.0f),
				            ImVec2(spinCenter.x + 6.0f, spinCenter.y - 4.0f), IM_COL32(255,255,255,255), 2.0f);
			}

			// ── Status text ──────────────────────────────────────────────────
			ImGui::SetCursorPos(ImVec2(28.0f, 52.0f));
			if (g_MMState == MM_SEARCHING)
			{
				int dots = (int)(t * 1.5f) % 4;
				char dotStr[5] = { 0 };
				for (int d = 0; d < dots; d++) dotStr[d] = '.';

				int count = 0;
				if (g_pServers) { auto* l = g_pServers->GetServersList(); while (l) { count++; l = l->next; } }
				bool loading = g_pServers && g_pServers->IsRequesting();

				if (loading)
					ImGui::TextColored(ImVec4(0.45f, 0.65f, 1.0f, 0.9f), "Scanning servers%s", dotStr);
				else if (count == 0)
					ImGui::TextColored(ImVec4(0.80f, 0.40f, 0.30f, 1.0f), "No servers found. Try refreshing.");
				else
					ImGui::TextColored(ImVec4(0.45f, 0.65f, 1.0f, 0.9f), "Evaluating %d servers%s", count, dotStr);
			}
			else if (g_MMState == MM_FOUND)
			{
				// Server details
				ImGui::TextColored(ImVec4(0.75f, 0.85f, 1.00f, 1.0f), "%s", g_MMBestName[0] ? g_MMBestName : "Unknown Server");
				ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 4.0f);

				// Ping dot
				ImVec4 pingColor = g_MMBestPing < 60  ? ImVec4(0.20f, 0.85f, 0.35f, 1.0f) :
				                   g_MMBestPing < 120 ? ImVec4(0.85f, 0.80f, 0.10f, 1.0f) :
				                                        ImVec4(0.90f, 0.35f, 0.10f, 1.0f);
				ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 1.0f), "Map: ");
				ImGui::SameLine(0.0f, 0.0f);
				ImGui::TextColored(ImVec4(0.65f, 0.75f, 0.95f, 1.0f), "%s", g_MMBestMap[0] ? g_MMBestMap : "unknown");
				ImGui::SameLine(0.0f, 16.0f);
				ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 1.0f), "Players: ");
				ImGui::SameLine(0.0f, 0.0f);
				ImGui::TextColored(ImVec4(0.70f, 0.80f, 1.0f, 1.0f), "%d/%d", g_MMBestPlayers, g_MMBestMaxPlayers);
				ImGui::SameLine(0.0f, 16.0f);
				ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 1.0f), "Ping: ");
				ImGui::SameLine(0.0f, 0.0f);
				ImGui::TextColored(pingColor, "%d ms", g_MMBestPing);

				// Countdown bar
				double timeLeft = g_MMConnectAt - ImGui::GetTime();
				if (timeLeft < 0.0) timeLeft = 0.0;
				float frac = (float)(timeLeft / 5.0);
				ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10.0f);
				ImVec2 barPos = ImGui::GetCursorScreenPos();
				float barW = mw - 56.0f;
				dl->AddRectFilled(barPos, ImVec2(barPos.x + barW, barPos.y + 5.0f), IM_COL32(18, 24, 40, 220), 3.0f);
				dl->AddRectFilled(barPos, ImVec2(barPos.x + barW * frac, barPos.y + 5.0f), IM_COL32(55, 200, 100, 220), 3.0f);
				ImGui::Dummy(ImVec2(0.0f, 10.0f));
				ImGui::TextColored(ImVec4(0.35f, 0.50f, 0.70f, 1.0f), "Connecting in %.0f seconds...", timeLeft > 0 ? timeLeft : 0.0);
			}

			// ── Buttons ──────────────────────────────────────────────────────
			ImGui::SetCursorPos(ImVec2(28.0f, mh - 58.0f));

			if (g_MMState == MM_FOUND)
			{
				ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.15f, 0.38f, 0.82f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.22f, 0.52f, 1.00f, 1.0f));
				if (ImGui::Button("CONNECT NOW", ImVec2(150.0f, 34.0f)))
				{
					char cmd[256];
					snprintf(cmd, sizeof(cmd), "connect %s\n", g_MMBestAddress);
					gEngfuncs.pfnClientCmd(cmd);
					g_ShowMatchmaking = false;
					g_MMState = MM_IDLE;
				}
				ImGui::PopStyleColor(2);
				ImGui::SameLine(0.0f, 10.0f);
			}
			else if (g_MMState == MM_SEARCHING)
			{
				// No servers found yet – offer a retry
				int count = 0;
				if (g_pServers) { auto* l = g_pServers->GetServersList(); while (l) { count++; l = l->next; } }
				bool loading = g_pServers && g_pServers->IsRequesting();

				if (!loading && count == 0)
				{
					if (ImGui::Button("RETRY", ImVec2(90.0f, 34.0f)))
					{
						if (g_pServers) g_pServers->RequestList();
					}
					ImGui::SameLine(0.0f, 10.0f);
				}
			}

			ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.12f, 0.16f, 0.26f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.22f, 0.28f, 0.45f, 1.0f));
			if (ImGui::Button("CANCEL", ImVec2(90.0f, 34.0f)))
			{
				g_ShowMatchmaking = false;
				g_MMState = MM_IDLE;
			}
			ImGui::PopStyleColor(2);
		}
		ImGui::End();
		ImGui::PopStyleVar(3);
		ImGui::PopStyleColor(2);
	}

	if (g_ShowServerBrowser)
	{
		// Windowed CS:GO-style server browser
		float sbW = (float)ScreenWidth  * 0.88f;
		float sbH = (float)ScreenHeight * 0.88f;
		ImGui::SetNextWindowPos(
			ImVec2((float)ScreenWidth  * 0.5f - sbW * 0.5f,
			       (float)ScreenHeight * 0.5f - sbH * 0.5f),
			ImGuiCond_Always);
		ImGui::SetNextWindowSize(ImVec2(sbW, sbH), ImGuiCond_Always);


		// Deep dark navy background like CS:GO matchmaking
		ImGui::PushStyleColor(ImGuiCol_WindowBg,        ImVec4(0.04f, 0.05f, 0.07f, 0.98f));
		ImGui::PushStyleColor(ImGuiCol_TableHeaderBg,   ImVec4(0.07f, 0.09f, 0.13f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_TableRowBg,       ImVec4(0.05f, 0.07f, 0.10f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt,    ImVec4(0.07f, 0.09f, 0.13f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_TableBorderLight, ImVec4(0.12f, 0.16f, 0.22f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_TableBorderStrong,ImVec4(0.16f, 0.20f, 0.28f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_Header,           ImVec4(0.22f, 0.30f, 0.60f, 0.60f));
		ImGui::PushStyleColor(ImGuiCol_HeaderHovered,    ImVec4(0.26f, 0.40f, 0.80f, 0.55f));
		ImGui::PushStyleColor(ImGuiCol_HeaderActive,     ImVec4(0.30f, 0.50f, 1.00f, 0.65f));
		ImGui::PushStyleColor(ImGuiCol_Button,           ImVec4(0.10f, 0.14f, 0.22f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered,    ImVec4(0.20f, 0.30f, 0.60f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive,     ImVec4(0.26f, 0.45f, 0.90f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_FrameBg,          ImVec4(0.07f, 0.09f, 0.14f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered,   ImVec4(0.12f, 0.16f, 0.26f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_ScrollbarBg,      ImVec4(0.03f, 0.04f, 0.06f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab,    ImVec4(0.20f, 0.30f, 0.50f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_Text,             ImVec4(0.85f, 0.90f, 1.00f, 1.00f));
		ImGui::PushStyleColor(ImGuiCol_Separator,        ImVec4(0.12f, 0.18f, 0.30f, 1.00f));

		ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.18f, 0.28f, 0.55f, 1.00f));

		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding,   0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize,  1.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,    3.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,       ImVec2(10.0f, 6.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_CellPadding,       ImVec2(10.0f, 7.0f));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,     ImVec2(0.0f, 0.0f));

		if (ImGui::Begin("##ServerBrowserWin", &g_ShowServerBrowser,
			ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBringToFrontOnFocus |
			ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar))
		{
			static int activeTab = 0;
			static char searchFilter[128] = "";
			static int selectedIndex = -1;
			static char selectedAddress[128] = "";

			float winW = ImGui::GetWindowWidth();
			float winH = ImGui::GetWindowHeight();

			// ─── HEADER ─────────────────────────────────────────────────────────
			{
				// Animated scan line using time
				float t = (float)(ImGui::GetTime());
				float scanPhase = fmodf(t * 0.8f, 1.0f); // 0..1 loop

				ImDrawList* dl = ImGui::GetWindowDrawList();
				ImVec2 winPos = ImGui::GetWindowPos();

				// Thin accent line top
				dl->AddRectFilled(
					ImVec2(winPos.x, winPos.y),
					ImVec2(winPos.x + winW, winPos.y + 3.0f),
					IM_COL32(50, 110, 230, 255)
				);

				// Header background
				dl->AddRectFilled(
					ImVec2(winPos.x, winPos.y + 3.0f),
					ImVec2(winPos.x + winW, winPos.y + 72.0f),
					IM_COL32(8, 11, 20, 255)
				);

				// Animated scan line
				float scanX = winPos.x + scanPhase * winW;
				dl->AddRectFilled(
					ImVec2(scanX - 2.0f, winPos.y + 3.0f),
					ImVec2(scanX + 80.0f, winPos.y + 72.0f),
					IM_COL32(50, 110, 230, 18)
				);

				// Bottom separator line of header
				dl->AddRectFilled(
					ImVec2(winPos.x, winPos.y + 72.0f),
					ImVec2(winPos.x + winW, winPos.y + 74.0f),
					IM_COL32(30, 50, 100, 200)
				);

				// Title
				ImGui::SetCursorPos(ImVec2(28.0f, 18.0f));
				ImGui::PushFont(g_FontMedium);
				ImGui::TextColored(ImVec4(0.85f, 0.90f, 1.00f, 1.0f), "FIND A SERVER");
				ImGui::PopFont();

				// Pulsing dot next to title (indicates activity)
				float pulse = (sinf(t * 3.0f) + 1.0f) * 0.5f; // 0..1
				bool isLoading = g_pServers && g_pServers->IsRequesting();
				ImVec2 dotPos = ImVec2(winPos.x + 204.0f, winPos.y + 30.0f);
				if (isLoading)
				{
					dl->AddCircleFilled(dotPos, 5.0f, IM_COL32(80 + (int)(150 * pulse), 170 + (int)(60 * pulse), 255, 220));
				}
				else
				{
					dl->AddCircleFilled(dotPos, 5.0f, IM_COL32(60, 160, 80, 200));
				}

				// Status text
				ImGui::SetCursorPos(ImVec2(28.0f, 42.0f));
				if (isLoading)
				{
					int dotCount = ((int)(t * 2.0f) % 4);
					char dots[5] = { 0 };
					for (int d = 0; d < dotCount; d++) dots[d] = '.';
					char statusStr[64];
					snprintf(statusStr, sizeof(statusStr), "Scanning for servers%s", dots);
					ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.00f, 0.85f), "%s", statusStr);
				}
				else
				{
					int serverCount = 0;
					if (g_pServers)
					{
						CHudServers::server_t* cnt = g_pServers->GetServersList();
						while (cnt) { serverCount++; cnt = cnt->next; }
					}
					ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.70f, 1.0f), "%d servers found", serverCount);
				}

				// Tabs — right of title
				ImGui::SetCursorPos(ImVec2(340.0f, 22.0f));
				{
					bool internet = (activeTab == 0);
					if (internet)
						ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.34f, 0.70f, 1.0f));
					else
						ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.10f, 0.14f, 0.22f, 1.0f));

					if (ImGui::Button("INTERNET", ImVec2(110.0f, 28.0f)) && activeTab != 0)
					{
						activeTab = 0;
						selectedIndex = -1;
						if (g_pServers) g_pServers->RequestList();
					}
					ImGui::PopStyleColor();
					ImGui::SameLine(0.0f, 4.0f);

					bool lan = (activeTab == 1);
					if (lan)
						ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.34f, 0.70f, 1.0f));
					else
						ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.10f, 0.14f, 0.22f, 1.0f));

					if (ImGui::Button("LAN", ImVec2(70.0f, 28.0f)) && activeTab != 1)
					{
						activeTab = 1;
						selectedIndex = -1;
						if (g_pServers) g_pServers->RequestBroadcastList(1);
					}
					ImGui::PopStyleColor();
				}

				// REFRESH button
				ImGui::SameLine(0.0f, 16.0f);
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.22f, 0.40f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.38f, 0.70f, 1.0f));
				if (ImGui::Button("REFRESH", ImVec2(80.0f, 28.0f)))
				{
					selectedIndex = -1;
					if (g_pServers)
					{
						if (activeTab == 0) g_pServers->RequestList();
						else g_pServers->RequestBroadcastList(1);
					}
				}
				ImGui::PopStyleColor(2);

				// Only AG Servers toggle
				ImGui::SameLine(0.0f, 20.0f);
				ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5.0f);
				if (ImGui::Checkbox("Only AG", &g_FilterAGOnly))
				{
					selectedIndex = -1;
					if (g_pServers)
					{
						if (activeTab == 0) g_pServers->RequestList();
						else g_pServers->RequestBroadcastList(1);
					}
				}

				// Search bar (top right, leave room for X button)
				ImGui::SetCursorPos(ImVec2(winW - 330.0f, 22.0f));
				ImGui::SetNextItemWidth(290.0f);
				ImGui::InputTextWithHint("##SBSearch", "Search servers or maps...", searchFilter, sizeof(searchFilter));

				// Close button — drawn via screen-space DrawList so it's always top-right
				{
					ImVec2 btnMin = ImVec2(winPos.x + winW - 34.0f, winPos.y + 8.0f);
					ImVec2 btnMax = ImVec2(winPos.x + winW - 8.0f,  winPos.y + 34.0f);
					bool hovered = ImGui::IsMouseHoveringRect(btnMin, btnMax);
					if (hovered)
						dl->AddRectFilled(btnMin, btnMax, IM_COL32(160, 30, 30, 200), 3.0f);
					ImU32 xCol = hovered ? IM_COL32(255,255,255,255) : IM_COL32(160,170,200,200);
					float cx = (btnMin.x + btnMax.x) * 0.5f;
					float cy = (btnMin.y + btnMax.y) * 0.5f;
					dl->AddLine(ImVec2(cx-5,cy-5), ImVec2(cx+5,cy+5), xCol, 2.0f);
					dl->AddLine(ImVec2(cx+5,cy-5), ImVec2(cx-5,cy+5), xCol, 2.0f);
					if (hovered && ImGui::IsMouseClicked(0))
						g_ShowServerBrowser = false;
				}
			}

			// ─── SERVER TABLE ────────────────────────────────────────────────────
			float tableTop = 80.0f;
			float footerH  = 66.0f;
			ImGui::SetCursorPos(ImVec2(8.0f, tableTop));

			ImGuiTableFlags tableFlags =
				ImGuiTableFlags_ScrollY |
				ImGuiTableFlags_RowBg |
				ImGuiTableFlags_BordersInnerV |
				ImGuiTableFlags_SortMulti |
				ImGuiTableFlags_Resizable;

			if (ImGui::BeginTable("SBTable", 6, tableFlags, ImVec2(winW, winH - tableTop - footerH)))
			{
				ImGui::TableSetupScrollFreeze(0, 1);
				ImGui::TableSetupColumn("Server Name",  ImGuiTableColumnFlags_WidthStretch);
				ImGui::TableSetupColumn("Map",          ImGuiTableColumnFlags_WidthFixed, 160.0f);
				ImGui::TableSetupColumn("Players",      ImGuiTableColumnFlags_WidthFixed, 110.0f);
				ImGui::TableSetupColumn("Ping",         ImGuiTableColumnFlags_WidthFixed, 80.0f);
				ImGui::TableSetupColumn("Bots",         ImGuiTableColumnFlags_WidthFixed, 50.0f);
				ImGui::TableSetupColumn("Address",      ImGuiTableColumnFlags_WidthFixed, 180.0f);

				// Custom header row
				ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
				for (int col = 0; col < 6; col++)
				{
					ImGui::TableSetColumnIndex(col);
					ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.55f, 0.80f, 1.0f));
					ImGui::TextUnformatted(ImGui::TableGetColumnName(col));
					ImGui::PopStyleColor();
				}

				if (g_pServers)
				{
					CHudServers::server_t* list = g_pServers->GetServersList();
					int index = 0;
					while (list)
					{
						char name[128]      = "";
						char map[64]        = "";
						char current[16]    = "0";
						char maxPlayers[16] = "0";
						char address[128]   = "";

						GetInfoValue(list->info, "hostname", name, sizeof(name));
						GetInfoValue(list->info, "map", map, sizeof(map));
						GetInfoValue(list->info, "current", current, sizeof(current));
						GetInfoValue(list->info, "max", maxPlayers, sizeof(maxPlayers));
						GetInfoValue(list->info, "address", address, sizeof(address));

						if (address[0] == '\0')
							strncpy(address, gEngfuncs.pNetAPI->AdrToString(&list->remote_address), sizeof(address) - 1);

						bool match = true;
						if (searchFilter[0] != '\0')
							match = (CaseInsensitiveContains(name, searchFilter) || CaseInsensitiveContains(map, searchFilter));

						if (match)
						{
							ImGui::TableNextRow();
							bool isSelected = (index == selectedIndex);

							// Row highlight for selected
							if (isSelected)
								ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(28, 54, 110, 180));

							// ── Server Name ──
							ImGui::TableNextColumn();
							{
								char label[256];
								snprintf(label, sizeof(label), "%s##sb_%d", name[0] ? name : "Unnamed Server", index);

								ImGui::PushStyleColor(ImGuiCol_Header,        ImVec4(0.18f, 0.34f, 0.72f, 0.70f));
								ImGui::PushStyleColor(ImGuiCol_HeaderHovered,  ImVec4(0.22f, 0.40f, 0.85f, 0.55f));

								if (ImGui::Selectable(label, isSelected,
									ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick,
									ImVec2(0, 0)))
								{
									if (ImGui::IsMouseDoubleClicked(0))
									{
										char cmd[256];
										snprintf(cmd, sizeof(cmd), "connect %s\n", address);
										gEngfuncs.pfnClientCmd(cmd);
										g_ShowServerBrowser = false;
									}
									else
									{
										selectedIndex = index;
										strncpy(selectedAddress, address, sizeof(selectedAddress) - 1);
									}
								}
								ImGui::PopStyleColor(2);
							}

							// ── Map ──
							ImGui::TableNextColumn();
							ImGui::TextColored(ImVec4(0.65f, 0.75f, 0.95f, 1.0f), "%s", map[0] ? map : "—");

							// ── Players (bar) ──
							ImGui::TableNextColumn();
							{
								int cur = atoi(current);
								int maxP = atoi(maxPlayers);
								if (maxP <= 0) maxP = 32;
								float frac = (float)cur / (float)maxP;
								if (frac > 1.0f) frac = 1.0f;

								// Color: green->yellow->red depending on fill
								ImVec4 barColor;
								if (frac < 0.5f)
									barColor = ImVec4(0.20f + frac * 0.8f, 0.75f, 0.25f, 1.0f);
								else if (frac < 0.85f)
									barColor = ImVec4(0.90f, 0.75f - frac * 0.5f, 0.10f, 1.0f);
								else
									barColor = ImVec4(0.90f, 0.20f, 0.20f, 1.0f);

								ImVec2 barPos = ImGui::GetCursorScreenPos();
								float barW    = ImGui::GetContentRegionAvail().x - 2.0f;
								float barH    = 4.0f;
								ImDrawList* dl2 = ImGui::GetWindowDrawList();
								// Track bg
								dl2->AddRectFilled(barPos, ImVec2(barPos.x + barW, barPos.y + barH), IM_COL32(20, 25, 40, 200), 2.0f);
								// Fill
								dl2->AddRectFilled(barPos, ImVec2(barPos.x + barW * frac, barPos.y + barH),
									IM_COL32((int)(barColor.x * 255), (int)(barColor.y * 255), (int)(barColor.z * 255), 200), 2.0f);

								ImGui::Dummy(ImVec2(0.0f, barH + 2.0f));
								char playerStr[32];
								snprintf(playerStr, sizeof(playerStr), "%d / %d", cur, maxP);
								ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 2.0f);
								ImGui::TextColored(ImVec4(0.70f, 0.80f, 1.0f, 0.9f), "%s", playerStr);
							}

							// ── Ping (colored) ──
							ImGui::TableNextColumn();
							{
								int ping = list->ping;
								ImVec4 pingColor;
								if (ping < 60)        pingColor = ImVec4(0.20f, 0.85f, 0.35f, 1.0f); // green
								else if (ping < 120)  pingColor = ImVec4(0.85f, 0.80f, 0.10f, 1.0f); // yellow
								else if (ping < 200)  pingColor = ImVec4(0.95f, 0.50f, 0.10f, 1.0f); // orange
								else                   pingColor = ImVec4(0.90f, 0.20f, 0.20f, 1.0f); // red

								ImGui::TextColored(pingColor, "%d ms", ping);
							}

							// ── Bots ──
							ImGui::TableNextColumn();
							{
								char bots[16] = "0";
								GetInfoValue(list->info, "bots", bots, sizeof(bots));
								int botCount = atoi(bots);
								if (botCount > 0)
									ImGui::TextColored(ImVec4(0.60f, 0.45f, 0.20f, 1.0f), "%d", botCount);
								else
									ImGui::TextDisabled("—");
							}

							// ── Address ──
							ImGui::TableNextColumn();
							ImGui::TextColored(ImVec4(0.35f, 0.45f, 0.60f, 1.0f), "%s", address);
						}

						list = list->next;
						index++;
					}
				}
				ImGui::EndTable();
			}

			// ─── FOOTER ──────────────────────────────────────────────────────────
			{
				ImDrawList* dl3 = ImGui::GetWindowDrawList();
				ImVec2 winPos2  = ImGui::GetWindowPos();
				// Footer separator
				dl3->AddRectFilled(
					ImVec2(winPos2.x, winPos2.y + winH - footerH),
					ImVec2(winPos2.x + winW, winPos2.y + winH - footerH + 2.0f),
					IM_COL32(30, 50, 100, 180)
				);
				dl3->AddRectFilled(
					ImVec2(winPos2.x, winPos2.y + winH - footerH + 2.0f),
					ImVec2(winPos2.x + winW, winPos2.y + winH),
					IM_COL32(5, 7, 12, 250)
				);

				ImGui::SetCursorPos(ImVec2(20.0f, winH - footerH + 16.0f));

				// Connect button (only active if selection made)
				bool hasSelection = (selectedIndex >= 0 && selectedAddress[0] != '\0');
				if (!hasSelection)
				{
					ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.08f, 0.10f, 0.18f, 1.0f));
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.08f, 0.10f, 0.18f, 1.0f));
					ImGui::PushStyleColor(ImGuiCol_Text,           ImVec4(0.30f, 0.35f, 0.45f, 1.0f));
				}
				else
				{
					ImGui::PushStyleColor(ImGuiCol_Button,         ImVec4(0.15f, 0.38f, 0.80f, 1.0f));
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered,  ImVec4(0.22f, 0.50f, 1.00f, 1.0f));
					ImGui::PushStyleColor(ImGuiCol_Text,           ImVec4(1.00f, 1.00f, 1.00f, 1.0f));
				}

				if (ImGui::Button("CONNECT  >", ImVec2(150.0f, 34.0f)) && hasSelection)
				{
					char cmd[256];
					snprintf(cmd, sizeof(cmd), "connect %s\n", selectedAddress);
					gEngfuncs.pfnClientCmd(cmd);
					g_ShowServerBrowser = false;
				}
				ImGui::PopStyleColor(3);

				ImGui::SameLine(0.0f, 14.0f);
				ImGui::SetCursorPosY(winH - footerH + 16.0f);
				if (hasSelection)
					ImGui::TextColored(ImVec4(0.45f, 0.55f, 0.75f, 1.0f), "Double-click a server or press CONNECT");
				else
					ImGui::TextColored(ImVec4(0.25f, 0.30f, 0.42f, 1.0f), "Select a server to connect");

				// ESC hint right side
				ImGui::SetCursorPos(ImVec2(winW - 180.0f, winH - footerH + 22.0f));
				ImGui::TextColored(ImVec4(0.25f, 0.30f, 0.42f, 1.0f), "[ ESC ] Close");
			}
		}
		ImGui::End();

		ImGui::PopStyleVar(6);
		ImGui::PopStyleColor(19);
	}

	ImGui::Render();
	ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
}

int ImGuiHelper_HandleKey(int down, int keynum, const char *pszCurrentBinding)
{
	// Always allow toggleconsole to pass to the engine
	if (pszCurrentBinding && strcmp(pszCurrentBinding, "toggleconsole") == 0)
	{
		return 1;
	}

	// Toggle key: F8
	if (keynum == K_F8 && down)
	{
		ToggleImGuiMenu_Callback();
		return 0; // Handled
	}

	// Intercept Escape key
	if (keynum == K_ESCAPE && down)
	{
		// 1. If any editor or browser is visible, close it
		if (g_ShowImGuiMenu || g_ShowCrosshairEditor || g_ShowAGSettings || g_ShowServerBrowser)
		{
			g_ShowImGuiMenu = false;
			g_ShowCrosshairEditor = false;
			g_ShowAGSettings = false;
			g_ShowServerBrowser = false;
			ImGuiHelper_UpdateInputState();
			return 0; // Handled, block from engine
		}

		// 2. If pause menu is visible, close it
		if (g_ShowPauseMenu)
		{
			g_ShowPauseMenu = false;
			ImGuiHelper_UpdateInputState();
			return 0; // Handled, block from engine
		}

		// 3. Otherwise, if in-game and custom menu enabled, open pause menu
		const char* mapName = gEngfuncs.pfnGetLevelName();
		bool inGame = (mapName && mapName[0] != '\0');
		bool customMenuEnabled = cl_custom_menu ? (cl_custom_menu->value != 0.0f) : true;
		if (inGame && customMenuEnabled)
		{
			g_ShowPauseMenu = true;
			ImGuiHelper_UpdateInputState();
			return 0; // Handled, block from engine
		}
	}

	if (g_ShowImGuiMenu || g_ShowCrosshairEditor || g_ShowAGSettings || g_ShowPauseMenu || g_ShowMainMenu || g_ShowServerBrowser)
	{
		ImGuiIO& io = ImGui::GetIO();
		if (io.WantCaptureKeyboard || g_ShowPauseMenu || g_ShowMainMenu || g_ShowServerBrowser)
		{
			return 0; // Handled, block from engine
		}
	}

	return 1; // Pass to engine
}
