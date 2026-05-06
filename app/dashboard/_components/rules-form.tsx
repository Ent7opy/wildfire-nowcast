/**
 * Client-side rules editor. Submits via PUT /api/aoi/[id]/rules.
 *
 * Channel list is a simple add/remove form — exactly the shape the Stage 4
 * dispatcher reads (`{type, target}` pairs).
 */
"use client";

import { useState } from "react";

const TZ_CHOICES = [
  "UTC",
  "America/Los_Angeles",
  "America/Denver",
  "America/Chicago",
  "America/New_York",
  "Europe/London",
  "Europe/Berlin",
  "Europe/Athens",
  "Australia/Sydney",
];

type Channel = { type: "email" | "webhook"; target: string };

type Rules = {
  distanceBufferKm: number;
  minConfidence: "low" | "nominal" | "high";
  minFrpMw: number;
  quietHours: { tz: string; startHour: number; endHour: number } | null;
  pausedUntil: string | null;
  notifyChannels: Channel[];
};

const DEFAULT: Rules = {
  distanceBufferKm: 25,
  minConfidence: "nominal",
  minFrpMw: 5,
  quietHours: null,
  pausedUntil: null,
  notifyChannels: [],
};

export function RulesForm({
  aoiId,
  initial,
}: {
  aoiId: string;
  initial: Rules | null;
}) {
  const [rules, setRules] = useState<Rules>(initial ?? DEFAULT);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  async function save() {
    setBusy(true);
    setMsg(null);
    try {
      const res = await fetch(`/api/aoi/${aoiId}/rules`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(rules),
      });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as {
          error?: { message?: string };
        };
        throw new Error(body.error?.message ?? `HTTP ${res.status}`);
      }
      setMsg("Saved.");
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Save failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-2 flex flex-col gap-4 text-sm">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <label className="flex flex-col gap-1">
          <span>Distance buffer (km)</span>
          <input
            type="number"
            min={0.1}
            step={0.1}
            value={rules.distanceBufferKm}
            onChange={(e) =>
              setRules({ ...rules, distanceBufferKm: Number(e.target.value) })
            }
            className="rounded border px-2 py-1"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span>Min confidence</span>
          <select
            value={rules.minConfidence}
            onChange={(e) =>
              setRules({ ...rules, minConfidence: e.target.value as Rules["minConfidence"] })
            }
            className="rounded border px-2 py-1"
          >
            <option value="low">low</option>
            <option value="nominal">nominal</option>
            <option value="high">high</option>
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span>Min FRP (MW)</span>
          <input
            type="number"
            min={0}
            step={0.5}
            value={rules.minFrpMw}
            onChange={(e) =>
              setRules({ ...rules, minFrpMw: Number(e.target.value) })
            }
            className="rounded border px-2 py-1"
          />
        </label>
      </div>

      <fieldset className="rounded border p-3">
        <legend className="px-1 text-xs uppercase tracking-wide text-[color:var(--muted)]">
          Quiet hours
        </legend>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={rules.quietHours !== null}
            onChange={(e) =>
              setRules({
                ...rules,
                quietHours: e.target.checked
                  ? { tz: "UTC", startHour: 22, endHour: 7 }
                  : null,
              })
            }
          />
          Enable quiet hours
        </label>
        {rules.quietHours ? (
          <div className="mt-2 grid grid-cols-3 gap-2">
            <label className="flex flex-col gap-1">
              <span>Timezone</span>
              <select
                value={rules.quietHours.tz}
                onChange={(e) =>
                  setRules({
                    ...rules,
                    quietHours: { ...rules.quietHours!, tz: e.target.value },
                  })
                }
                className="rounded border px-2 py-1"
              >
                {TZ_CHOICES.map((tz) => (
                  <option key={tz} value={tz}>
                    {tz}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span>Start hour</span>
              <input
                type="number"
                min={0}
                max={23}
                value={rules.quietHours.startHour}
                onChange={(e) =>
                  setRules({
                    ...rules,
                    quietHours: {
                      ...rules.quietHours!,
                      startHour: Number(e.target.value),
                    },
                  })
                }
                className="rounded border px-2 py-1"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span>End hour</span>
              <input
                type="number"
                min={0}
                max={23}
                value={rules.quietHours.endHour}
                onChange={(e) =>
                  setRules({
                    ...rules,
                    quietHours: {
                      ...rules.quietHours!,
                      endHour: Number(e.target.value),
                    },
                  })
                }
                className="rounded border px-2 py-1"
              />
            </label>
          </div>
        ) : null}
      </fieldset>

      <label className="flex flex-col gap-1">
        <span>Paused until (ISO 8601, blank = active)</span>
        <input
          type="text"
          value={rules.pausedUntil ?? ""}
          onChange={(e) =>
            setRules({ ...rules, pausedUntil: e.target.value || null })
          }
          placeholder="2026-12-31T00:00:00Z"
          className="rounded border px-2 py-1 font-mono text-xs"
        />
      </label>

      <fieldset className="rounded border p-3">
        <legend className="px-1 text-xs uppercase tracking-wide text-[color:var(--muted)]">
          Notification channels
        </legend>
        <div className="flex flex-col gap-2">
          {rules.notifyChannels.map((c, i) => (
            <div key={i} className="flex items-center gap-2">
              <select
                value={c.type}
                onChange={(e) => {
                  const next = [...rules.notifyChannels];
                  next[i] = { ...c, type: e.target.value as Channel["type"] };
                  setRules({ ...rules, notifyChannels: next });
                }}
                className="rounded border px-2 py-1"
              >
                <option value="email">email</option>
                <option value="webhook">webhook</option>
              </select>
              <input
                type="text"
                value={c.target}
                onChange={(e) => {
                  const next = [...rules.notifyChannels];
                  next[i] = { ...c, target: e.target.value };
                  setRules({ ...rules, notifyChannels: next });
                }}
                className="flex-1 rounded border px-2 py-1"
                placeholder={c.type === "email" ? "you@example.com" : "https://…"}
              />
              <button
                type="button"
                onClick={() => {
                  const next = rules.notifyChannels.filter((_, j) => j !== i);
                  setRules({ ...rules, notifyChannels: next });
                }}
                className="rounded border px-2 py-1 text-xs"
              >
                Remove
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={() =>
              setRules({
                ...rules,
                notifyChannels: [
                  ...rules.notifyChannels,
                  { type: "email", target: "" },
                ],
              })
            }
            className="self-start rounded border px-3 py-1 text-xs"
          >
            Add channel
          </button>
        </div>
      </fieldset>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={save}
          disabled={busy}
          className="rounded bg-[color:var(--accent)] px-4 py-2 text-white disabled:opacity-50"
        >
          {busy ? "Saving…" : "Save rules"}
        </button>
        {msg ? <span className="text-sm">{msg}</span> : null}
      </div>
    </div>
  );
}
