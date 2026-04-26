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
- **`@neo4j-nvl/react`** v0.3.8 (graph rendering — matches the version
  already used by `graphora-fe/app`, so a successful spike ports 1:1
  to the real Explorer)
- **Zustand** (state)

NVL was chosen over Cytoscape.js explicitly so the Explorer uses the
same rendering library as the existing graphora-fe components (see
`graphora-fe/app/src/components/graph-viz.tsx` for the real-product
usage — `nvlOptions` in the spike mirror it).

If the stack blows the size budget, options in order of preference:

1. Lazy-load NVL via `React.lazy` / dynamic import so the Evidence
   tab loads fast and the graph canvas streams in after
2. Swap React 18 → Preact-compat
3. Swap Vite → esbuild alone (no HMR, tiny runtime)

If the stack blows the perf budget, options:

1. Set `useWebGL: true` (already on in the spike) and audit that
   WebGL is actually active in the devtools
2. Drop `hover: { enabled: true }` — hover testing every frame is
   expensive at large fixtures
3. Swap `layout: 'forceDirected'` for `'grid'` at > 5k nodes — NVL's
   layout algorithms dominate first-paint at scale

## Running it

```bash
cd prototyping/explorer-spike
pnpm install
pnpm build           # production bundle, measure .js + .css gzipped
pnpm dev             # local dev server for perf testing
pnpm analyze         # rollup-plugin-visualizer output
```

Record results in [RESULTS.md](./RESULTS.md).
