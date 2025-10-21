-- recording_frame_ts_with_overlay.lua
-- One script to (1) log per-frame Unix epoch ms and (2) update a Text source overlay.
-- Writes a JSONL file next to the final recording on stop.
-- Robust across OBS versions (no directory APIs during recording).

obs = obslua

-- ===== User-configurable via script UI =====
local overlay_enable = true
local overlay_only_while_recording = true
local overlay_source dropdown_name = ""
local overlay_source_custom = ""

-- ===== Internal state =====
local is_recording = false
local frame_idx = 0
local start_epoch_ms = 0
local mono0_ns = 0
local epoch0_ms = 0
local lines = {}         -- in-memory JSONL lines
local last_overlay_text = nil

-- ---------- Helpers ----------
local function json_escape(s)
    return tostring(s):gsub('\\','\\\\'):gsub('"','\\"')
end

local function add_jsonl(tbl)
    local parts = {}
    for k, v in pairs(tbl) do
        local key = '"' .. tostring(k) .. '"'
        local val
        if type(v) == "number" then
            val = tostring(v)
        elseif type(v) == "boolean" then
            val = v and "true" or "false"
        else
            val = '"' .. json_escape(v) .. '"'
        end
        table.insert(parts, key .. ":" .. val)
    end
    table.insert(lines, "{" .. table.concat(parts, ",") .. "}\n")
end

-- Stable epoch during recording = epoch0_ms + (monotonic_now - mono0_ns)
local function epoch_ms_now_recording()
    local ns = obs.os_gettime_ns()
    return epoch0_ms + math.floor((ns - mono0_ns) / 1000000)
end

-- Live wall-clock epoch (ms) for non-recording overlay (simple, fine)
local function epoch_ms_now_wall()
    return os.time() * 1000 + math.floor((obs.os_gettime_ns() % 1000000000) / 1000000)
end

local function dirname(path)
    if not path or path == "" then return "." end
    local sep = package.config:sub(1,1)
    local i = path:match(".*"..sep.."()")
    if i then return path:sub(1, i-2+1) end
    local other = (sep == "\\") and "/" or "\\"
    i = path:match(".*"..other.."()")
    if i then return path:sub(1, i-2+1) end
    return "."
end

local function join_path(dir, file)
    if not dir or dir == "" then dir = "." end
    local sep = package.config:sub(1,1)
    local last = dir:sub(-1)
    if last == "\\" or last == "/" then
        return dir .. file
    else
        return dir .. sep .. file
    end
end

-- ---------- Overlay ----------
local function current_overlay_source_name()
    if overlay_source_custom and overlay_source_custom ~= "" then
        return overlay_source_custom
    end
    return overlay_source_dropdown_name or ""
end

local function set_overlay_text(text)
    if not overlay_enable then return end
    local name = current_overlay_source_name()
    if name == nil or name == "" then return end
    if last_overlay_text == text then return end  -- avoid redundant updates

    local src = obs.obs_get_source_by_name(name)
    if src ~= nil then
        local s = obs.obs_data_create()
        obs.obs_data_set_string(s, "text", text)
        obs.obs_source_update(src, s)
        obs.obs_data_release(s)
        obs.obs_source_release(src)
        last_overlay_text = text
    end
end

local function clear_overlay_if_needed()
    if not overlay_enable then return end
    if overlay_only_while_recording then
        set_overlay_text("")  -- blank when not recording
        last_overlay_text = ""  -- keep cache consistent
    end
end

-- ---------- Session control ----------
local function start_session()
    is_recording = true
    frame_idx = 0
    lines = {}

    mono0_ns = obs.os_gettime_ns()
    epoch0_ms = os.time() * 1000
    start_epoch_ms = epoch0_ms

    add_jsonl({event="recording_started", epoch_ms=start_epoch_ms})
end

local function stop_session()
    if not is_recording then
        clear_overlay_if_needed()
        return
    end
    local stop_ms = epoch_ms_now_recording()
    local last_file = obs.obs_frontend_get_last_recording() or ""
    add_jsonl({
        event="recording_stopped",
        epoch_ms=stop_ms,
        started_epoch_ms=start_epoch_ms,
        file=last_file
    })

    -- write JSONL file next to the recording
    local dir = dirname(last_file)
    local log_path = join_path(dir, string.format("recording_%d.jsonl", start_epoch_ms))
    local f = io.open(log_path, "w")
    if f then
        for _, line in ipairs(lines) do f:write(line) end
        f:close()
    else
        -- fallback: write to CWD if we can't open target
        local fb = string.format("recording_%d.jsonl", start_epoch_ms)
        local g = io.open(fb, "w")
        if g then
            for _, line in ipairs(lines) do g:write(line) end
            g:close()
        end
    end

    is_recording = false
    lines = {}
    clear_overlay_if_needed()
end

-- ---------- OBS callbacks ----------
local function on_frontend_event(event)
    if event == obs.OBS_FRONTEND_EVENT_RECORDING_STARTED then
        start_session()
    elseif event == obs.OBS_FRONTEND_EVENT_RECORDING_STOPPED then
        stop_session()
    elseif event == obs.OBS_FRONTEND_EVENT_RECORDING_PAUSED then
        if is_recording then
            add_jsonl({event="recording_paused", epoch_ms=epoch_ms_now_recording()})
        end
    elseif event == obs.OBS_FRONTEND_EVENT_RECORDING_UNPAUSED then
        if is_recording then
            add_jsonl({event="recording_unpaused", epoch_ms=epoch_ms_now_recording()})
        end
    end
end

function script_tick(seconds)
    local show_text = ""

    if is_recording then
        frame_idx = frame_idx + 1
        local now_ms = epoch_ms_now_recording()
        add_jsonl({event="frame", idx=frame_idx, epoch_ms=now_ms, rel_ms=now_ms - start_epoch_ms})
        if overlay_enable then
            show_text = tostring(now_ms)
        end
    else
        if overlay_enable and not overlay_only_while_recording then
            show_text = tostring(epoch_ms_now_wall())
        end
    end

    if overlay_enable then
        set_overlay_text(show_text)
    end
end

-- ---------- Script UI ----------
function script_description()
    return "Per-frame Unix epoch (ms) logging + optional Text overlay.\n" ..
           "- Writes recording_<start_epoch_ms>.jsonl next to the MKV on stop.\n" ..
           "- Updates a Text source with epoch ms each frame (optional)."
end

function script_properties()
    local props = obs.obs_properties_create()

    obs.obs_properties_add_bool(props, "overlay_enable", "Enable overlay text")
    obs.obs_properties_add_bool(props, "overlay_only_while_recording", "Show overlay only while recording")

    local p = obs.obs_properties_add_list(
        props, "overlay_source_dropdown_name", "Text Source (dropdown)",
        obs.OBS_COMBO_TYPE_EDITABLE, obs.OBS_COMBO_FORMAT_STRING
    )
    -- populate with existing text sources
    local sources = obs.obs_enum_sources()
    if sources ~= nil then
        for _, src in ipairs(sources) do
            local id = obs.obs_source_get_id(src)
            if id == "text_gdiplus" or id == "text_ft2_source" then
                local nm = obs.obs_source_get_name(src)
                obs.obs_property_list_add_string(p, nm, nm)
            end
        end
        obs.source_list_release(sources)
    end

    obs.obs_properties_add_text(props, "overlay_source_custom", "Or type source name", obs.OBS_TEXT_DEFAULT)

    return props
end

function script_update(settings)
    overlay_enable = obs.obs_data_get_bool(settings, "overlay_enable")
    overlay_only_while_recording = obs.obs_data_get_bool(settings, "overlay_only_while_recording")
    overlay_source_dropdown_name = obs.obs_data_get_string(settings, "overlay_source_dropdown_name")
    overlay_source_custom = obs.obs_data_get_string(settings, "overlay_source_custom")

    if not is_recording then
        clear_overlay_if_needed()
    end
end

function script_load(settings)
    obs.obs_frontend_add_event_callback(on_frontend_event)
end

function script_unload()
    -- nothing to flush here; we write on stop
end
