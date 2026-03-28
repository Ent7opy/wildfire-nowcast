import { defineConfig } from "vite";

/**
 * Vite config for the standalone embeddable forecast widget.
 *
 * Builds a self-contained IIFE bundle at ui/dist-widget/widget.js that can
 * be embedded in any web page via a single <script> tag.
 *
 * Usage: cd ui && npm run build:widget
 */
export default defineConfig({
  build: {
    lib: {
      entry: "widget/main.ts",
      name: "WildfireWidget",
      formats: ["iife"],
      fileName: () => "widget.js",
    },
    outDir: "dist-widget",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // Inline all dynamic imports so the output is a single file
        inlineDynamicImports: true,
      },
    },
  },
});
