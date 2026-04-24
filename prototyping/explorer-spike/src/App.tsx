import { useEffect } from "react";
import { GraphView } from "./GraphView";
import { createFpsMeter } from "./fps";
import { useSpikeStore } from "./store";

const FIXTURE_SIZES = [100, 500, 1000, 5000, 10000];

export function App() {
  const { nodeCount, fps, graph, setNodeCount, setFps } = useSpikeStore();

  useEffect(() => {
    const meter = createFpsMeter(60, setFps);
    meter.start();
    return () => meter.stop();
  }, [setFps]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <header
        style={{
          padding: "8px 16px",
          borderBottom: "1px solid #e2e8f0",
          display: "flex",
          gap: 16,
          alignItems: "center",
          fontSize: 14,
          background: "#f8fafc",
        }}
      >
        <strong>Explorer Spike</strong>
        <span>nodes: {graph.nodes.length.toLocaleString()}</span>
        <span>edges: {graph.edges.length.toLocaleString()}</span>
        <span style={{ color: fps >= 30 ? "#16a34a" : "#dc2626" }}>
          fps: {fps}
        </span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
          {FIXTURE_SIZES.map((size) => (
            <button
              key={size}
              onClick={() => setNodeCount(size)}
              style={{
                padding: "4px 10px",
                border: "1px solid #cbd5e1",
                background: size === nodeCount ? "#3b82f6" : "white",
                color: size === nodeCount ? "white" : "#0f172a",
                borderRadius: 4,
                cursor: "pointer",
                fontSize: 13,
              }}
            >
              {size.toLocaleString()}
            </button>
          ))}
        </div>
      </header>
      <main style={{ flex: 1, minHeight: 0 }}>
        <GraphView graph={graph} />
      </main>
    </div>
  );
}
