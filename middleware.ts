/**
 * Next.js middleware — Clerk-protected route gates.
 *
 * Per Stage 5 brief §Middleware:
 *   - Protect: /api/aoi/* (CRUD + rules + briefs).
 *   - Pass-through: /api/aoi/poll (cron, uses CRON_SECRET),
 *                   /api/webhooks/clerk (Svix-signed),
 *                   /, /sign-in/*, /sign-up/*.
 *   - Build-without-blocking: when CLERK_SECRET_KEY is unset, no-op pass-through
 *     so the app boots without Clerk env. The route handlers' `requireUserId()`
 *     short-circuits to 503 `service_unavailable` instead.
 */
import { NextResponse } from "next/server";
import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

const isProtectedRoute = createRouteMatcher([
  "/api/aoi/:path*",
  "/api/brief/:path*",
  "/api/export/:path*",
  "/api/me",
  "/dashboard/:path*",
]);

const clerkConfigured = Boolean(process.env.CLERK_SECRET_KEY);

const handler = clerkConfigured
  ? clerkMiddleware(async (auth, req) => {
      if (isProtectedRoute(req)) {
        await auth.protect();
      }
    })
  : () => NextResponse.next();

export default handler;

export const config = {
  matcher: [
    // Skip Next internals and static assets; run on API + page routes.
    "/((?!_next|.*\\..*).*)",
    "/(api|trpc)(.*)",
  ],
};
