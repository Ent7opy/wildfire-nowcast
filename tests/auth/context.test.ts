/**
 * Stage 5 — `requireUserId` branch coverage.
 *
 * Uses `_setTestAuth` to bypass Clerk's runtime; covers the env-gate path and
 * the success/failure shapes the route handlers depend on.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { _setTestAuth, requireUserId } from "@/lib/auth/context";

const SAVED_ENV = { ...process.env };

describe("requireUserId", () => {
  beforeEach(() => {
    _setTestAuth(null);
  });
  afterEach(() => {
    _setTestAuth(null);
    process.env = { ...SAVED_ENV };
  });

  it("returns config_missing when CLERK_SECRET_KEY is unset and no test auth", async () => {
    delete process.env.CLERK_SECRET_KEY;
    const r = await requireUserId();
    expect(r).toEqual({ ok: false, code: "config_missing" });
  });

  it("returns the test-injected userId regardless of env", async () => {
    delete process.env.CLERK_SECRET_KEY;
    _setTestAuth(() => ({ ok: true, userId: "user_2abcStubbed" }));
    const r = await requireUserId();
    expect(r).toEqual({ ok: true, userId: "user_2abcStubbed" });
  });

  it("returns unauthenticated when test auth says so", async () => {
    _setTestAuth(() => ({ ok: false, code: "unauthenticated" }));
    const r = await requireUserId();
    expect(r).toEqual({ ok: false, code: "unauthenticated" });
  });
});
