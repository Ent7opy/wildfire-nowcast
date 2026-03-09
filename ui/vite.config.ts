import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 8501,
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/tests/setup.ts",
    include: ["src/tests/**/*.test.ts", "src/tests/**/*.test.tsx"],
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
      exclude: ["src/tests/**"]
    }
  }
});
