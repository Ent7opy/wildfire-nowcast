import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import "./globals.css";

export const metadata: Metadata = {
  title: "Wildfire Nowcast",
  description:
    "Free, open, AI-native fire intelligence for stewardship — depth over speed.",
};

function ClerkConfigBanner() {
  return (
    <div
      role="status"
      className="bg-yellow-100 px-4 py-2 text-center text-sm text-yellow-900"
    >
      Auth not configured — running in read-only public mode.
    </div>
  );
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  const clerkConfigured = Boolean(
    process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY,
  );

  const body = clerkConfigured ? (
    <ClerkProvider>{children}</ClerkProvider>
  ) : (
    <>
      <ClerkConfigBanner />
      {children}
    </>
  );

  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{body}</body>
    </html>
  );
}
