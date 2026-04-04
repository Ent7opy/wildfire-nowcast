import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, fireEvent, waitFor, cleanup } from "@testing-library/react";

import AIChatAssistant from "../components/AIChatAssistant";
import { EARTH_TOOLS_ASSISTANT_SYSTEM_PROMPT, SAFETY_ASSISTANT_SYSTEM_PROMPT } from "../config/assistant";
import { useAppStore } from "../state/store";

// Capture the body sent to /assistant/chat in each test.
let capturedChatBodies: unknown[] = [];

function makeFetchMock(chatReply = "ok") {
  return vi.fn((url: string, init?: RequestInit) => {
    const urlStr = String(url);

    if (urlStr.includes("/assistant/health")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ configured: true, circuit_state: "closed" })
      });
    }

    if (urlStr.includes("/assistant/chat")) {
      if (init?.body) {
        capturedChatBodies.push(JSON.parse(String(init.body)));
      }
      return Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            candidates: [{ content: { parts: [{ text: chatReply }] } }]
          })
      });
    }

    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
  });
}

const baseState = useAppStore.getState();

describe("AIChatAssistant handleSend system prompt selection", () => {
  beforeEach(() => {
    capturedChatBodies = [];
    vi.stubGlobal("fetch", makeFetchMock());

    // Reset store to a clean, non-safety state with no selected event
    // so the auto-briefing useEffect does not fire.
    useAppStore.setState({
      ...baseState,
      selectedEvent: null,
      safety: {
        enabled: false,
        userLocation: null,
        locationPermission: "unknown",
        nearestFireDistanceKm: null,
        safetyTier: "SAFE",
        proximityRadiusKm: 50,
        pendingBriefingPrompt: null
      }
    });
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("sends EARTH_TOOLS prompt when safety mode is off", async () => {
    const { getByPlaceholderText, getByRole } = render(<AIChatAssistant />);

    // Wait for health check to resolve so the input is enabled.
    await waitFor(() => {
      expect(getByPlaceholderText("Ask AI about this region...")).toBeTruthy();
    });

    const input = getByPlaceholderText("Ask AI about this region...");
    fireEvent.change(input, { target: { value: "What fires are active?" } });

    const sendButton = getByRole("button");
    fireEvent.click(sendButton);

    await waitFor(() => {
      const chatBody = capturedChatBodies.find(
        (b) =>
          typeof b === "object" &&
          b !== null &&
          "systemInstruction" in b
      ) as { systemInstruction: { parts: Array<{ text: string }> } } | undefined;

      expect(chatBody).toBeDefined();
      expect(chatBody!.systemInstruction.parts[0].text).toBe(EARTH_TOOLS_ASSISTANT_SYSTEM_PROMPT);
    });
  });

  it("sends SAFETY prompt when safety mode is on", async () => {
    // Enable safety mode before rendering.
    useAppStore.setState({
      ...useAppStore.getState(),
      safety: {
        enabled: true,
        userLocation: null,
        locationPermission: "granted",
        nearestFireDistanceKm: 15,
        safetyTier: "WARNING",
        proximityRadiusKm: 50,
        pendingBriefingPrompt: null
      }
    });

    const { getByPlaceholderText, getByRole } = render(<AIChatAssistant />);

    await waitFor(() => {
      expect(getByPlaceholderText("Ask about safety, evacuation, risk...")).toBeTruthy();
    });

    const input = getByPlaceholderText("Ask about safety, evacuation, risk...");
    fireEvent.change(input, { target: { value: "Should I evacuate?" } });

    const sendButton = getByRole("button");
    fireEvent.click(sendButton);

    await waitFor(() => {
      const chatBody = capturedChatBodies.find(
        (b) =>
          typeof b === "object" &&
          b !== null &&
          "systemInstruction" in b
      ) as { systemInstruction: { parts: Array<{ text: string }> } } | undefined;

      expect(chatBody).toBeDefined();
      expect(chatBody!.systemInstruction.parts[0].text).toBe(SAFETY_ASSISTANT_SYSTEM_PROMPT);
    });
  });

  it("the two system prompts are distinct strings", () => {
    expect(EARTH_TOOLS_ASSISTANT_SYSTEM_PROMPT).not.toBe(SAFETY_ASSISTANT_SYSTEM_PROMPT);
    expect(SAFETY_ASSISTANT_SYSTEM_PROMPT).toContain("safety assistant");
  });
});
