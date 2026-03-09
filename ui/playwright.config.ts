import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  retries: 0,
  use: {
    baseURL: "http://127.0.0.1:8501",
    trace: "on-first-retry"
  },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 8501",
    url: "http://127.0.0.1:8501",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    }
  ]
});
