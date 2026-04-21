import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Wildfire Nowcast",
  description:
    "Free, open, AI-native fire intelligence for stewardship — depth over speed.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
