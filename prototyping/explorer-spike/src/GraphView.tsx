import { useMemo, useRef } from "react";
import { InteractiveNvlWrapper } from "@neo4j-nvl/react";
import { SpikeGraph } from "./fixture";

interface Props {
  graph: SpikeGraph;
  width: number;
  height: number;
}

const COLOUR_BY_TYPE: Record<string, string> = {
  Company: "#3b82f6",
  Person: "#10b981",
  Transaction: "#f59e0b",
};

// Renders the spike fixture via NVL. Converts our abstract SpikeGraph
// shape into the NVL (nodes, rels) tuple — NVL rels use from/to, not
// source/target like Cytoscape. Options mirror graphora-fe's graph-viz
// component so any perf conclusion here carries over to the real
// product install.
export function GraphView({ graph, width, height }: Props) {
  const nvlRef = useRef<unknown>(null);

  const { nodes, rels } = useMemo(() => {
    const nvlNodes = graph.nodes.map((n) => ({
      id: n.id,
      caption: n.label,
      color: COLOUR_BY_TYPE[n.type] ?? "#64748b",
    }));
    const nvlRels = graph.edges.map((e) => ({
      id: e.id,
      from: e.source,
      to: e.target,
      caption: e.label,
    }));
    return { nodes: nvlNodes, rels: nvlRels };
  }, [graph]);

  // Dev-only: expose the NVL wrapper ref so the perf harness can
  // trigger pan/zoom programmatically from Playwright without
  // synthesising mouse events.
  const captureRef = (ref: unknown) => {
    nvlRef.current = ref;
    (window as unknown as { __nvl?: unknown }).__nvl = ref;
  };

  return (
    <InteractiveNvlWrapper
      ref={captureRef as never}
      nodes={nodes}
      rels={rels}
      nvlOptions={{
        initialZoom: 0.8,
        // Grid layout is O(n) placement — isolates the pure
        // render/interaction cost of NVL from the layout algorithm.
        // forceDirected iterates for many frames on random-edge
        // graphs; grid answers "what's the steady-state fps?"
        layout: graph.nodes.length > 500 ? "grid" : "forceDirected",
        layoutSettings: {
          nodeDistance: 100,
          nodeRepulsion: 5000,
        },
        renderer: "canvas",
        useWebGL: true,
        nodeLabelsVisible: false,
        relationshipLabelsVisible: false,
        nodeSize: 8,
        backgroundColor: "#ffffff",
      }}
      interactionOptions={{
        zoom: { enabled: true, minZoom: 0.05, maxZoom: 10 },
        drag: { enabled: true },
        pan: { enabled: true },
        hover: { enabled: true },
      }}
      style={{ width, height }}
    />
  );
}
