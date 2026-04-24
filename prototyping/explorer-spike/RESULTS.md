# Explorer Spike — Results

Record each measurement run here. Keep the bar honest.

## Exit criteria (from `work/Graphora/evidence-explorer-design.md`)

| Metric                | Preferred | Hard cap |
| --------------------- | --------- | -------- |
| Bundle gzipped        | < 500 KB  | < 800 KB |
| 10k-node interactive  | 30 fps    | 20 fps   |

## Measurements

### Run 1 — baseline stack: Vite 5 + React 18 + Cytoscape 3.33 + Zustand 5

Date: 2026-04-24
Machine: MacBook (local dev), `pnpm build --mode production`
Node: v22.15 / pnpm 9
Browser: (pending manual perf test)

| Metric                            | Measured    | Pass / fail |
| --------------------------------- | ----------- | ----------- |
| `dist/assets/*.js` raw            | 589,062 B   | (info)      |
| `dist/assets/*.js` gzipped        | **188,773 B (~185 KB)** | ✅ preferred (<500 KB) |
| `dist/assets/*.js` brotli         | 160,556 B (~157 KB) | ✅ |
| First paint (dev, 1k fixture)     | pending     |             |
| Drag FPS, 1k fixture              | pending     |             |
| Drag FPS, 10k fixture             | pending     |             |
| Zoom/pan FPS, 10k fixture         | pending     |             |

Notes:
- Single-chunk production build. Vite warns about chunks > 500 KB
  raw, which doesn't apply to the gzipped budget. Fine to ignore.
- Bundle composition: Cytoscape dominates (~320 KB raw of the 589 KB).
  React + React-DOM is ~140 KB. Zustand + fixture + app shell is
  tiny.
- No perf measurements yet — run `pnpm dev`, open the devtools perf
  tab, interact with the 10k node grid, record frame times. Fill in
  the "pending" cells.

### Run 2 — if needed, fallback stack

Options in priority order:

1. Drop shadcn → unstyled Radix + Tailwind
2. Preact-compat shim for React
3. Cytoscape canvas renderer instead of WebGL
4. Swap build tool Vite → esbuild direct

Record which fallback was tried and the deltas.

## Decision

Date: (pending perf measurement)

- [ ] Pass — proceed with the baseline stack into A1-shell
- [ ] Pass with fallback #N — proceed with that stack
- [ ] Fail — escalate; options exhausted, re-scope Explorer

Notes:
- Bundle size gate already cleared at 185 KB gzipped (well under
  the 500 KB preferred threshold, 23% of the 800 KB hard cap).
- Perf gate still pending a human running `pnpm dev` and dragging
  the 10k-node fixture around in Chrome devtools.
