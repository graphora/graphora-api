import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { visualizer } from "rollup-plugin-visualizer";

// The spike's whole point is measuring bundle output, so production
// settings matter more than usual:
// - no sourcemaps (would inflate the .js we're measuring)
// - esbuild minify (default; terser would be slightly smaller but slower)
// - rollup visualiser emits stats.html for manual inspection
export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    mode === "analyze" &&
      visualizer({
        filename: "dist/stats.html",
        template: "treemap",
        gzipSize: true,
        brotliSize: true,
      }),
  ],
  build: {
    target: "es2020",
    sourcemap: false,
    minify: "esbuild",
    rollupOptions: {
      output: {
        manualChunks: undefined, // single chunk to simplify measurement
      },
    },
  },
}));
