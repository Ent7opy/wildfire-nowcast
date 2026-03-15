import { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  CircularProgress,
  IconButton,
  TextField,
  Typography
} from "@mui/material";
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline";
import SendIcon from "@mui/icons-material/Send";
import ReactMarkdown from "react-markdown";

import { EARTH_TOOLS_ASSISTANT_SYSTEM_PROMPT, SAFETY_ASSISTANT_SYSTEM_PROMPT } from "../config/assistant";
import { useAppStore } from "../state/store";
import type { FireEvent } from "../types/api";
import { computeTimeRange } from "../utils/time";

type ChatMessage = {
  role: "assistant" | "user";
  text: string;
};

const INITIAL_MESSAGE: ChatMessage = {
  role: "assistant",
  text: "Ecological assistant is ready. Select a fire event to ask contextual questions."
};

function compactEventContext(selectedEvent: FireEvent | null): Record<string, unknown> | null {
  if (!selectedEvent) {
    return null;
  }
  return {
    event_id: selectedEvent.event_id,
    lat: selectedEvent.lat,
    lon: selectedEvent.lon,
    source: selectedEvent.source,
    sensor: selectedEvent.sensor,
    detection_count: selectedEvent.detection_count,
    front_count: selectedEvent.front_count,
    event_score: selectedEvent.event_score,
    denoiser_decision: selectedEvent.denoiser_decision,
    review_required: selectedEvent.review_required,
    start_time: selectedEvent.start_time,
    end_time: selectedEvent.end_time,
    location_name: selectedEvent.location_name,
    region_name: selectedEvent.region_name,
    admin1_name: selectedEvent.admin1_name,
    admin0_name: selectedEvent.admin0_name,
    country: selectedEvent.country,
    frp_max: selectedEvent.frp_max ?? null,
    frp_mean: selectedEvent.frp_mean ?? null,
    brightness_max: selectedEvent.brightness_max ?? null,
    brightness_mean: selectedEvent.brightness_mean ?? null,
    geom_source: selectedEvent.geom_source ?? null,
    geom_method: selectedEvent.geom_method ?? null
  };
}

function extractReply(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") return null;

  const data = payload as {
    candidates?: Array<{
      content?: {
        parts?: Array<{ text?: unknown }>;
      };
    }>;
  } & Record<string, unknown>;

  const candidateParts = data.candidates?.[0]?.content?.parts;
  if (Array.isArray(candidateParts)) {
    const text = candidateParts
      .map((part) => (typeof part.text === "string" ? part.text.trim() : ""))
      .filter((part) => part.length > 0)
      .join("\n")
      .trim();
    if (text.length > 0) {
      return text;
    }
  }

  const fallback = data.reply || data.response || data.answer || data.message;
  return typeof fallback === "string" && fallback.trim().length > 0 ? fallback.trim() : null;
}

export default function AIChatAssistant(): JSX.Element {
  const [messages, setMessages] = useState<ChatMessage[]>([INITIAL_MESSAGE]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [autoBriefing, setAutoBriefing] = useState<string | null>(null);
  const [isBriefing, setIsBriefing] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const lastBriefedEventId = useRef<string | null>(null);
  const lastBriefedPrompt = useRef<string | null>(null);

  const selectedEvent = useAppStore((s) => s.selectedEvent);
  const filters = useAppStore((s) => s.filters);
  const layers = useAppStore((s) => s.layers);
  const mapView = useAppStore((s) => s.mapView);
  const activePreset = useAppStore((s) => s.activePreset);
  const forecast = useAppStore((s) => s.forecast);
  const assistantViewContext = useAppStore((s) => s.assistantViewContext);
  const safety = useAppStore((s) => s.safety);
  const clearAssistantBriefingPrompt = useAppStore((s) => s.clearAssistantBriefingPrompt);
  const isSafetyMode = safety.enabled;

  const geminiApiBaseUrl = String(import.meta.env.VITE_GEMINI_API_BASE_URL || "https://generativelanguage.googleapis.com/v1beta").trim();
  const geminiModel = String(import.meta.env.VITE_GEMINI_MODEL || "gemini-2.5-flash").trim();
  const geminiApiKey = String(import.meta.env.VITE_GEMINI_API_KEY || "").trim();
  const assistantConfigured = geminiModel.length > 0 && geminiApiKey.length > 0;

  const eventContext = useMemo(() => compactEventContext(selectedEvent), [selectedEvent]);
  const timeRange = useMemo(() => computeTimeRange(filters), [filters]);

  const viewingContext = useMemo(
    () => ({
      selected_event: eventContext,
      filters: {
        ...filters
      },
      map_view: {
        latitude: mapView.latitude,
        longitude: mapView.longitude,
        zoom: mapView.zoom,
        pitch: mapView.pitch,
        bearing: mapView.bearing
      },
      layers: {
        ...layers
      },
      active_preset: activePreset || "Custom",
      time_window: {
        start_time: timeRange.startTime.toISOString(),
        end_time: timeRange.endTime.toISOString()
      },
      visible_feed: assistantViewContext,
      forecast_context: {
        active_job_id: forecast.jobId,
        last_run_id: forecast.lastForecast?.run.id || null,
        last_event_key: forecast.lastForecast?.eventKey || null,
        run_quality: forecast.lastForecast?.runMeta
          ? {
              weather_available: forecast.lastForecast.runMeta.weatherRunId !== null,
              confidence_level: forecast.lastForecast.runMeta.confidenceLevel
            }
          : null,
        active_request: forecast.activeRequest
          ? {
              event_id: forecast.activeRequest.eventId || null,
              event_key: forecast.activeRequest.eventKey || null,
              front_id: forecast.activeRequest.frontId || null,
              lat: forecast.activeRequest.lat,
              lon: forecast.activeRequest.lon,
              location_label: forecast.activeRequest.locationLabel
            }
          : null
      }
    }),
    [activePreset, assistantViewContext, eventContext, filters, forecast, layers, mapView, timeRange]
  );

  const triggerBriefing = async (prompt: string): Promise<void> => {
    if (!assistantConfigured || isBriefing) return;
    setIsBriefing(true);
    setAutoBriefing(null);
    try {
      const systemPrompt = isSafetyMode ? SAFETY_ASSISTANT_SYSTEM_PROMPT : EARTH_TOOLS_ASSISTANT_SYSTEM_PROMPT;
      const response = await fetch(
        `${geminiApiBaseUrl}/models/${encodeURIComponent(geminiModel)}:generateContent?key=${encodeURIComponent(geminiApiKey)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            systemInstruction: { parts: [{ text: systemPrompt }] },
            contents: [{
              role: "user",
              parts: [{
                text: [
                  "Use the following wildfire dashboard context as the source of truth.",
                  `DASHBOARD_CONTEXT_JSON: ${JSON.stringify(viewingContext)}`,
                  `BRIEFING_REQUEST: ${prompt}`
                ].join("\n")
              }]
            }],
            generationConfig: { temperature: 0.4 }
          })
        }
      );
      if (!response.ok) throw new Error(`briefing call failed with ${response.status}`);
      const payload = (await response.json()) as unknown;
      const reply = extractReply(payload);
      if (reply) setAutoBriefing(reply);
    } catch {
      // Silently fail for auto-briefings — don't disrupt UX
    } finally {
      setIsBriefing(false);
    }
  };

  // Auto-briefing when a new event is selected
  useEffect(() => {
    if (!selectedEvent || !assistantConfigured) return;
    const eventId = String(selectedEvent.event_id ?? "");
    if (!eventId || eventId === lastBriefedEventId.current) return;
    lastBriefedEventId.current = eventId;
    const prompt = isSafetyMode
      ? "Give a 2-sentence safety briefing for this fire. Plain language only: risk level and what the person should do right now."
      : "Give a 2-sentence analyst briefing: fire behavior context, intensity interpretation, and spread risk.";
    void triggerBriefing(prompt);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedEvent?.event_id]);

  // Watch for imperatively requested briefings (from FireDetailsPanel "Get Safety Info")
  useEffect(() => {
    const prompt = safety.pendingBriefingPrompt;
    if (!prompt || prompt === lastBriefedPrompt.current) return;
    lastBriefedPrompt.current = prompt;
    clearAssistantBriefingPrompt();
    void triggerBriefing(prompt);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safety.pendingBriefingPrompt]);

  useEffect(() => {
    if (!scrollRef.current) {
      return;
    }
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const handleSend = async (): Promise<void> => {
    const prompt = input.trim();
    if (!prompt || !assistantConfigured) {
      return;
    }

    setMessages((prev) => [...prev, { role: "user", text: prompt }]);
    setInput("");
    setIsSending(true);

    try {
      const response = await fetch(
        `${geminiApiBaseUrl}/models/${encodeURIComponent(geminiModel)}:generateContent?key=${encodeURIComponent(geminiApiKey)}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            systemInstruction: {
              parts: [{ text: EARTH_TOOLS_ASSISTANT_SYSTEM_PROMPT }]
            },
            contents: [
              {
                role: "user",
                parts: [
                  {
                    text: [
                      "Use the following wildfire dashboard context as the source of truth for this answer.",
                      "If a field is missing or null, state uncertainty explicitly.",
                      "",
                      `DASHBOARD_CONTEXT_JSON: ${JSON.stringify(viewingContext)}`,
                      "",
                      `USER_QUESTION: ${prompt}`
                    ].join("\n")
                  }
                ]
              }
            ],
            generationConfig: {
              temperature: 0.5
            }
          })
        }
      );

      if (!response.ok) {
        throw new Error(`assistant call failed with ${response.status}`);
      }

      const payload = (await response.json()) as unknown;
      const reply = extractReply(payload);
      if (!reply) {
        throw new Error("assistant response is missing a reply field");
      }

      setMessages((prev) => [...prev, { role: "assistant", text: reply }]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Gemini request failed. Check API key/model configuration and retry."
        }
      ]);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        minHeight: 300,
        bgcolor: "#0d1117",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 3,
        overflow: "hidden",
        boxShadow: "0 24px 80px rgba(0,0,0,0.35)"
      }}
    >
      <Box sx={{ px: 2, py: 1.5, borderBottom: "1px solid rgba(255,255,255,0.08)", bgcolor: "#161b22", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.9 }}>
          <ChatBubbleOutlineIcon sx={{ fontSize: 16, color: isSafetyMode ? "#ef4444" : "#f97316" }} />
          <Typography sx={{ fontSize: 10, fontWeight: 800, letterSpacing: "0.14em", textTransform: "uppercase", color: "#fff" }}>
            {isSafetyMode ? "Safety Assistant" : "Fire Analyst"}
          </Typography>
        </Box>

        <Box sx={{ display: "flex", alignItems: "center", gap: 0.65 }}>
          <Box
            sx={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              bgcolor: assistantConfigured ? "#22c55e" : "#f59e0b",
              animation: assistantConfigured ? "pulse 1.4s ease-in-out infinite" : "none"
            }}
          />
          <Typography sx={{ fontSize: 10, color: "#6b7280", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase" }}>
            {assistantConfigured ? "Live" : "Offline"}
          </Typography>
        </Box>
      </Box>

      {!assistantConfigured && (
        <Box sx={{ px: 2, py: 1.1, borderBottom: "1px solid rgba(245,158,11,0.25)", bgcolor: "rgba(245,158,11,0.1)" }}>
          <Typography sx={{ fontSize: 11, color: "#fbbf24" }}>
            Gemini is not configured. Set `VITE_GEMINI_API_KEY` (and optionally `VITE_GEMINI_MODEL`).
          </Typography>
        </Box>
      )}

      <Box ref={scrollRef} sx={{ flex: 1, overflowY: "auto", p: 2, display: "flex", flexDirection: "column", gap: 1.2 }}>
        {/* AI Briefing card — shown above chat thread when a briefing is available */}
        {(isBriefing || autoBriefing) && (
          <Box
            sx={{
              p: 1.5,
              borderRadius: 2,
              bgcolor: isSafetyMode ? "rgba(239,68,68,0.08)" : "rgba(59,130,246,0.08)",
              border: `1px solid ${isSafetyMode ? "rgba(239,68,68,0.25)" : "rgba(59,130,246,0.25)"}`,
              mb: 0.5
            }}
          >
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 0.75 }}>
              <Box sx={{ width: 5, height: 5, borderRadius: "50%", bgcolor: isSafetyMode ? "#ef4444" : "#60a5fa" }} />
              <Typography sx={{ fontSize: 9, fontWeight: 900, letterSpacing: "0.14em", textTransform: "uppercase", color: isSafetyMode ? "#f87171" : "#93c5fd" }}>
                AI Briefing
              </Typography>
            </Box>
            {isBriefing
              ? <CircularProgress size={14} sx={{ color: isSafetyMode ? "#ef4444" : "#60a5fa" }} />
              : <Typography sx={{ fontSize: 12, color: "#d1d5db", lineHeight: 1.6, fontStyle: "italic" }}>{autoBriefing}</Typography>
            }
          </Box>
        )}

        {messages.map((message, index) => (
          <Box key={`${message.role}-${index}`} sx={{ display: "flex", justifyContent: message.role === "user" ? "flex-end" : "flex-start" }}>
            <Box
              sx={{
                maxWidth: "85%",
                p: 1.25,
                borderRadius: 2,
                borderTopLeftRadius: message.role === "assistant" ? 2 : 10,
                borderTopRightRadius: message.role === "user" ? 2 : 10,
                bgcolor: message.role === "user" ? "#f97316" : "#1c2128",
                color: message.role === "user" ? "#fff" : "#d1d5db",
                border: message.role === "assistant" ? "1px solid rgba(255,255,255,0.07)" : "none",
                fontSize: 13,
                lineHeight: 1.6,
                "& p": { m: 0, mb: 0.75, "&:last-child": { mb: 0 } },
                "& strong": { color: "#f1f5f9", fontWeight: 700 },
                "& ul, & ol": { pl: 2.5, m: 0, mb: 0.75 },
                "& li": { mb: 0.25 },
                "& code": { fontFamily: "monospace", fontSize: 11, bgcolor: "rgba(255,255,255,0.07)", px: 0.5, borderRadius: 0.5 }
              }}
            >
              {message.role === "assistant" ? (
                <ReactMarkdown>{message.text}</ReactMarkdown>
              ) : (
                message.text
              )}
            </Box>
          </Box>
        ))}

        {isSending && (
          <Box sx={{ display: "flex", justifyContent: "flex-start" }}>
            <Box sx={{ p: 1.25, borderRadius: 2, bgcolor: "#1c2128", border: "1px solid rgba(255,255,255,0.07)" }}>
              <CircularProgress size={16} sx={{ color: "#f97316" }} />
            </Box>
          </Box>
        )}
      </Box>

      <Box sx={{ p: 1.5, borderTop: "1px solid rgba(255,255,255,0.08)", bgcolor: "#161b22" }}>
        <Box sx={{ position: "relative" }}>
          <TextField
            value={input}
            fullWidth
            size="small"
            placeholder={isSafetyMode ? "Ask about safety, evacuation, risk..." : "Ask AI about this region..."}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                void handleSend();
              }
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                bgcolor: "#0d1117",
                color: "#fff",
                pr: 5,
                borderRadius: 2,
                fontSize: 12
              }
            }}
          />
          <IconButton
            onClick={() => {
              void handleSend();
            }}
            disabled={isSending || !assistantConfigured || !input.trim()}
            sx={{
              position: "absolute",
              right: 6,
              top: 5,
              color: "#9ca3af",
              '&:hover': { color: "#f97316" }
            }}
          >
            <SendIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Box>
      </Box>
    </Box>
  );
}
