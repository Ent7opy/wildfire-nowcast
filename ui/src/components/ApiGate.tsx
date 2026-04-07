import { useCallback, useEffect, useState } from "react";
import { apiBaseUrlCandidates } from "../config/runtime";
import ServiceDownPage from "./ServiceDownPage";

const HEALTH_TIMEOUT_MS = 5_000;

type Status = "checking" | "up" | "down";

async function checkApi(): Promise<boolean> {
  for (const base of apiBaseUrlCandidates()) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
    try {
      const res = await fetch(`${base}/health`, { signal: controller.signal });
      clearTimeout(timer);
      if (res.ok) return true;
    } catch {
      clearTimeout(timer);
    }
  }
  return false;
}

/**
 * Wraps the entire app: renders children only when the API is reachable,
 * otherwise shows the ServiceDownPage.
 */
export default function ApiGate({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<Status>("checking");

  const probe = useCallback(async () => {
    setStatus("checking");
    const ok = await checkApi();
    setStatus(ok ? "up" : "down");
  }, []);

  useEffect(() => {
    probe();
  }, [probe]);

  if (status === "checking") {
    return null; // brief flash while probing — could add a spinner if desired
  }

  if (status === "down") {
    return <ServiceDownPage onRetry={probe} />;
  }

  return <>{children}</>;
}
