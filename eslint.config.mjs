import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Existing codebases outside the Next.js scaffold — Stage 0 is purely additive.
    ".claude/**",
    ".venv*/**",
    ".git/**",
    "api/**",
    "ml/**",
    "ingest/**",
    "ui/**",
    "models/**",
    "configs/**",
    "tools/**",
    "data/**",
    "infra/**",
    "examples/**",
    "reports/**",
    "test-results/**",
    ".venv_ml_sweep/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".playwright-mcp/**",
  ]),
]);

export default eslintConfig;
