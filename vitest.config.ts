/**
 * Vitest configuration for Stage 1 (Drizzle + PGlite tests).
 *
 * - Test files live in `tests/` (excluded from the Next.js compile graph).
 * - Path alias `@/` matches the Next.js tsconfig so test imports look the same
 *   as runtime imports.
 * - Node environment; pool=forks so PGlite (which uses a node binary) does not
 *   trip on Vitest's worker-thread isolation.
 */
import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const root = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      "@": resolve(root, "."),
    },
  },
  test: {
    environment: "node",
    pool: "forks",
    include: ["tests/**/*.test.ts", "tests/**/*.test.tsx"],
    testTimeout: 20_000,
    hookTimeout: 20_000,
  },
});
