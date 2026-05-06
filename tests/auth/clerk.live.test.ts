/**
 * Live Clerk SDK smoke test — spot-check that the real `clerkClient` loads
 * and a known test user can be fetched. Skipped unless `CLERK_LIVE=1` AND
 * `CLERK_SECRET_KEY` are both set; CI never sets these.
 *
 * Run locally:
 *   CLERK_LIVE=1 CLERK_SECRET_KEY=sk_test_... CLERK_TEST_USER_ID=user_... pnpm test tests/auth/clerk.live
 */
import { describe, expect, it } from "vitest";

const live =
  process.env.CLERK_LIVE === "1" &&
  Boolean(process.env.CLERK_SECRET_KEY) &&
  Boolean(process.env.CLERK_TEST_USER_ID);

const describeLive = live ? describe : describe.skip;

describeLive("Clerk SDK live", () => {
  it("clerkClient.users.getUser resolves a non-empty email", async () => {
    const { clerkClient } = await import("@clerk/nextjs/server");
    const client = await clerkClient();
    const userId = process.env.CLERK_TEST_USER_ID as string;
    const user = await client.users.getUser(userId);
    expect(user).toBeTruthy();
    expect(user.id).toBe(userId);
    const primary = user.emailAddresses.find(
      (e) => e.id === user.primaryEmailAddressId,
    );
    expect(primary?.emailAddress?.length ?? 0).toBeGreaterThan(0);
  });
});
