# Evidence Explorer bundle-size + render-perf spike

**Status:** throwaway — lives in `prototyping/` so lint, tests, and the
wheel never see it. Delete after the spike succeeds or fails.

## Purpose

Prove the stack picked by `work/Graphora/evidence-explorer-design.md`
can hit both exit criteria before we commit to the full A1 Explorer
build:

1. **Bundle size:** < 500 KB gzipped preferred, 800 KB gzipped hard cap.
2. **Render perf:** 10,000-node / 30,000-edge fixture interactive at
   30 fps on a MacBook Air M-series.

## Stack under test

- **Vite 5 + React 18** (build + shell)
- **Cytoscape.js** with the WebGL renderer (graph layout + interaction)
- **Tailwind** + **shadcn/ui primitives** (layout chrome)
- **Zustand** (state)

If the stack blows the size budget, options in order of preference:

1. Drop shadcn; use unstyled Radix primitives + Tailwind directly
2. Swap React 18 → Preact-compat
3. Swap Vite → esbuild alone (no HMR, tiny runtime)

If the stack blows the perf budget, fall back to Cytoscape's canvas
renderer (simpler but CPU-bound above ~3k nodes) or switch to
`sigma.js` for larger graphs.

## Running it

```bash
cd prototyping/explorer-spike
pnpm install
pnpm build           # production bundle, measure .js + .css gzipped
pnpm dev             # local dev server for perf testing
pnpm analyze         # rollup-plugin-visualizer output
```

Record results in [RESULTS.md](./RESULTS.md).
